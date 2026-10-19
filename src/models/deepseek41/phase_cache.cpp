#include "phase_cache.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <numeric>

namespace strata::models::deepseek41 {
namespace {
constexpr int keys=40*384;
int index(ExpertKey key){if(key.first<0||key.first>=40||key.second<0||key.second>=384)throw std::out_of_range("phase expert ID");return key.first*384+key.second;}
double milliseconds(std::chrono::steady_clock::time_point start){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
}
PhaseCache::PhaseCache(PhasePlan p,std::function<const uint8_t*(ExpertKey)> cold,cudaStream_t stream,uint64_t migration,bool pageable,bool trace)
    :plan_(std::move(p)),cold_(std::move(cold)),upload_(stream),migration_(migration),trace_(trace),original_(keys,-1),resident_(keys,-1),temporary_(keys,-1),core_(keys,0),heat_(keys,0),last_move_(keys,0){
    if(plan_.bytes==0||plan_.decode.empty())throw std::invalid_argument("empty phase region");
    arena_=std::make_unique<core::DeviceArena>(plan_.bytes);
    backup_.reset(new uint8_t[plan_.bytes]);backup_valid_.resize(plan_.decode.size());
    for(auto key:plan_.core)core_[index(key)]=1;
    for(auto& counts:counts_)counts.assign(keys,0);
    for(size_t i=0;i<plan_.decode.size();++i){
        const auto& slice=plan_.decode[i];original_[index(slice.key)]=i;
        slots_.push_back(std::make_unique<Slot>());stage_stride_=std::max(stage_stride_,slice.bytes);
    }
    for(size_t i=0;i<plan_.encoder.size();++i)stage_stride_=std::max(stage_stride_,plan_.encoder[i].bytes);
    staging_=std::make_unique<Pinned>(8*stage_stride_,pageable);
}
PhaseCache::~PhaseCache(){cudaStreamSynchronize(upload_);}
uint8_t* PhaseCache::address(size_t slot)const{return static_cast<uint8_t*>(arena_->base())+plan_.decode.at(slot).offset;}
uint8_t* PhaseCache::stage(ExecutionStats& stats,size_t& i){
    i=stage_index_++%8;
    if(stage_used_[i]){const auto start=std::chrono::steady_clock::now();gpu_check(cudaEventSynchronize(stage_ready_[i].value));stats.phase_host_wait_ms+=milliseconds(start);}
    return staging_->data+i*stage_stride_;
}
void PhaseCache::protect_readers(cudaStream_t stream){gpu_check(cudaEventRecord(boundary_.value,stream));gpu_check(cudaStreamWaitEvent(upload_,boundary_.value,0));}
void PhaseCache::initialize(size_t i,const uint8_t* source){
    auto& slot=*slots_.at(i);const auto& slice=plan_.decode[i];
    gpu_check(cudaMemcpyAsync(address(i),source,slice.bytes,cudaMemcpyHostToDevice,upload_));
    gpu_check(cudaEventRecord(slot.ready.value,upload_));slot.pending=true;slot.initialized=true;slot.key=slice.key;resident_[index(slice.key)]=i;
}
const uint8_t* PhaseCache::cpu(ExpertKey key)const{
    const int i=original_.at(index(key));return i>=0&&backup_valid_[i]?backup_.get()+plan_.decode[i].offset:nullptr;
}
const uint8_t* PhaseCache::host(ExpertKey key)const{
    if(const auto* saved=cpu(key))return saved;
    const auto* data=cold_(key);if(!data)throw std::runtime_error("phase expert has neither a RAM complement nor a valid backup");return data;
}
void PhaseCache::backup(size_t i,ExecutionStats& stats){
    prefill::LedgerScope trace(trace_,"ds41.phase.backup");
    auto& slot=*slots_.at(i);const int target=original_.at(index(slot.key));
    if(target<0||backup_valid_[target])return;
    size_t bank=0;auto* buffer=stage(stats,bank);
    if(slot.pending)gpu_check(cudaStreamWaitEvent(upload_,slot.ready.value,0));
    gpu_check(cudaMemcpyAsync(buffer,address(i),plan_.decode[i].bytes,cudaMemcpyDeviceToHost,upload_));
    gpu_check(cudaEventRecord(stage_ready_[bank].value,upload_));stage_used_[bank]=true;
    const auto before=std::chrono::steady_clock::now();gpu_check(cudaEventSynchronize(stage_ready_[bank].value));stats.phase_host_wait_ms+=milliseconds(before);
    std::memcpy(backup_.get()+plan_.decode[target].offset,buffer,plan_.decode[i].bytes);backup_valid_[target]=1;
    stats.phase_backup_bytes+=plan_.decode[i].bytes;
}
void PhaseCache::upload(size_t i,ExpertKey key,ExecutionStats& stats,bool restore){
    auto& slot=*slots_.at(i);const auto bytes=plan_.decode[i].bytes;
    if(key.first!=plan_.decode[i].key.first)throw std::logic_error("dynamic replacement must preserve the slot's layer/format");
    size_t bank=0;auto* buffer=stage(stats,bank);std::memcpy(buffer,host(key),bytes);
    gpu_check(cudaMemcpyAsync(address(i),buffer,bytes,cudaMemcpyHostToDevice,upload_));
    gpu_check(cudaEventRecord(stage_ready_[bank].value,upload_));stage_used_[bank]=true;
    gpu_check(cudaEventRecord(slot.ready.value,upload_));
    if(slot.key.first>=0)resident_[index(slot.key)]=-1;
    slot.key=key;slot.pending=true;resident_[index(key)]=i;
    stats.h2d_expert_bytes+=bytes;
    if(restore)stats.phase_restore_h2d_bytes+=bytes;else stats.phase_dynamic_h2d_bytes+=bytes;
}
void PhaseCache::restore_slot(size_t i,ExecutionStats& stats){
    auto& slot=*slots_[i];const auto& slice=plan_.decode[i];
    if(slot.key==slice.key)return;
    if(slice.release_after>=0&&borrowed_){
        if(!released_[slice.release_after])return;
        gpu_check(cudaStreamWaitEvent(upload_,consumed_[slice.release_after].value,0));
    }
    upload(i,slice.key,stats,true);slot.restored=true;slot.replay_used=false;
}
const uint8_t* PhaseCache::gpu(ExpertKey key,cudaStream_t stream,ExecutionStats& stats,int phase){
    const int k=index(key),temp=temporary_[k];
    if(temp>=0)return static_cast<const uint8_t*>(arena_->base())+plan_.encoder[temp].offset;
    const int i=resident_[k];if(i<0)return nullptr;auto& slot=*slots_[i];
    if(slot.pending){
        const auto ready=cudaEventQuery(slot.ready.value);
        if(ready==cudaErrorNotReady){gpu_check(cudaStreamWaitEvent(stream,slot.ready.value,0));++stats.phase_ready_waits;}
        else {gpu_check(ready);slot.pending=false;}
    }
    if(phase==1&&slot.restored&&!slot.replay_used){slot.replay_used=true;stats.phase_replay_reuse_bytes+=plan_.decode[i].bytes;}
    return address(i);
}
bool PhaseCache::begin_prefill(int remaining,int chunk,bool ced,cudaStream_t stream,ExecutionStats& stats){
    prefill::LedgerScope trace(trace_,"ds41.phase.borrow.prepare");
    const auto start=std::chrono::steady_clock::now();
    protect_readers(stream);
    if(borrowed_)finish_prefill(stream,stats);
    const bool borrow=ced&&remaining>=std::max(4096,chunk*2)&&!plan_.encoder.empty();
    for(auto& slot:slots_){slot->moved_at=0;slot->replay_used=false;slot->restored=false;}
    for(size_t i=0;i<slots_.size();++i)if(!borrow||plan_.decode[i].release_after<0)restore_slot(i,stats);
    for(auto& count:counts_)std::fill(count.begin(),count.end(),0);
    std::fill(heat_.begin(),heat_.end(),0);decode_rows_=adjustment_rows_=0;
    std::fill(last_move_.begin(),last_move_.end(),0);
    if(!borrow){stats.phase_prepare_ms+=milliseconds(start);return false;}
    // Backups are retained across requests. Temporary front experts keep their existing cold RAM copy.
    for(size_t i=0;i<slots_.size();++i)if(plan_.decode[i].release_after>=0&&!backup_valid_[i]){
        const int resident=resident_[index(plan_.decode[i].key)];
        if(resident<0)throw std::logic_error("GPU-only decode target lost before backup");
        backup(resident,stats);
    }
    for(size_t i=0;i<slots_.size();++i)if(plan_.decode[i].release_after>=0){
        resident_[index(slots_[i]->key)]=-1;slots_[i]->key={-1,-1};slots_[i]->restored=false;
    }
    std::fill(released_.begin(),released_.end(),false);borrowed_=true;
    for(size_t i=0;i<plan_.encoder.size();++i){
        const auto& slice=plan_.encoder[i];size_t bank=0;auto* buffer=stage(stats,bank);std::memcpy(buffer,host(slice.key),slice.bytes);
        gpu_check(cudaMemcpyAsync(static_cast<uint8_t*>(arena_->base())+slice.offset,buffer,slice.bytes,cudaMemcpyHostToDevice,upload_));
        gpu_check(cudaEventRecord(stage_ready_[bank].value,upload_));stage_used_[bank]=true;
        temporary_[index(slice.key)]=i;stats.phase_encoder_h2d_bytes+=slice.bytes;stats.h2d_expert_bytes+=slice.bytes;
    }
    gpu_check(cudaEventRecord(boundary_.value,upload_));gpu_check(cudaStreamWaitEvent(stream,boundary_.value,0));
    stats.phase_borrow_bytes=plan_.bytes;stats.phase_prepare_ms+=milliseconds(start);return true;
}
void PhaseCache::encoder_layer_done(int layer,cudaStream_t stream,ExecutionStats& stats){
    if(!borrowed_)return;
    if(layer<0||layer>=20||released_[layer])throw std::logic_error("phase last-chunk release order");
    const std::string label=trace_?"ds41.phase.restore.after_encoder_layer."+std::to_string(layer):"";
    prefill::LedgerScope trace(trace_,label.c_str());
    const auto start=std::chrono::steady_clock::now();gpu_check(cudaEventRecord(consumed_[layer].value,stream));released_[layer]=true;
    for(const auto& slice:plan_.encoder)if(slice.key.first==layer)temporary_[index(slice.key)]=-1;
    // Sorted decode slots prioritize the first replay layers. Wait only for intersecting front readers.
    for(size_t i=0;i<slots_.size();++i)if(plan_.decode[i].release_after<=layer)restore_slot(i,stats);
    stats.phase_restore_submit_ms+=milliseconds(start);
}
void PhaseCache::finish_prefill(cudaStream_t stream,ExecutionStats& stats){
    if(!borrowed_)return;
    prefill::LedgerScope trace(trace_,"ds41.phase.restore.finish");
    const auto start=std::chrono::steady_clock::now();protect_readers(stream);
    std::fill(temporary_.begin(),temporary_.end(),-1);
    // Cancellation/early exit uses an actual all-reader fence. No temporary front D2H is needed.
    for(int l=0;l<20;++l)if(!released_[l]){gpu_check(cudaEventRecord(consumed_[l].value,stream));released_[l]=true;}
    for(size_t i=0;i<slots_.size();++i)restore_slot(i,stats);
    borrowed_=false;stats.phase_restore_submit_ms+=milliseconds(start);
}
void PhaseCache::observe(int layer,const int* ids,int tokens,int phase){
    if(layer<0||layer>=40||tokens<1||phase<0||phase>2)throw std::invalid_argument("phase route observation");
    for(int i=0;i<tokens*6;++i){const int k=index({layer,ids[i]});counts_[phase][k]+=1;if(phase==2)heat_[k]+=1;}
}
int PhaseCache::until_adjustment()const{return migration_?16-static_cast<int>(adjustment_rows_%16):INT_MAX;}
void PhaseCache::adjust_decode(int tokens,cudaStream_t stream,ExecutionStats& stats){
    if(tokens<1)return;
    decode_rows_+=tokens;adjustment_rows_+=tokens;
    if(!migration_||borrowed_||adjustment_rows_<16)return;
    prefill::LedgerScope trace(trace_,"ds41.phase.decode.adjust");
    adjustment_rows_%=16;
    const auto started=std::chrono::steady_clock::now();protect_readers(stream);
    const uint64_t before_backup=stats.phase_backup_bytes;uint64_t used=0;
    struct Choice{size_t slot;ExpertKey key;double margin;};std::vector<Choice> choices;
    const double seed_weight=32.0*std::exp(-double(decode_rows_)/64.0);
    std::array<double,40> replay_totals{};
    for(int l=20;l<40;++l)replay_totals[l]=std::accumulate(counts_[1].begin()+l*384,counts_[1].begin()+(l+1)*384,0.0);
    auto score=[&](ExpertKey key){const int k=index(key);return heat_[k]+(replay_totals[key.first]?counts_[1][k]/replay_totals[key.first]*seed_weight:0);};
    for(size_t i=0;i<slots_.size();++i){
        const auto& slot=*slots_[i];if(slot.key.first<20||(slot.moved_at&&decode_rows_-slot.moved_at<64))continue;
        const int l=slot.key.first;ExpertKey best{-1,-1};double best_score=0;
        for(int e=0;e<384;++e){const int k=l*384+e;if(core_[k]||resident_[k]>=0||(last_move_[k]&&decode_rows_-last_move_[k]<64))continue;
            const double value=score({l,e});if(value>best_score){best={l,e};best_score=value;}}
        if(best.first>=0&&best_score>=score(slot.key)*1.25+2)choices.push_back({i,best,best_score-score(slot.key)});
    }
    std::stable_sort(choices.begin(),choices.end(),[](const auto& a,const auto& b){return a.margin>b.margin;});
    for(const auto& choice:choices){
        auto& slot=*slots_[choice.slot];const auto bytes=plan_.decode[choice.slot].bytes;const int k=index(choice.key);
        if(resident_[k]>=0)continue;
        const int old_original=original_[index(slot.key)];const uint64_t backup_cost=old_original>=0&&!backup_valid_[old_original]?bytes:0;
        if(bytes+backup_cost>migration_-used)continue;
        const auto old=slot.key;backup(choice.slot,stats);upload(choice.slot,choice.key,stats,false);
        used+=bytes+backup_cost;slot.moved_at=decode_rows_;last_move_[index(old)]=last_move_[k]=decode_rows_;++stats.phase_migrations;
    }
    for(auto& value:heat_)value*=.875;
    stats.phase_dynamic_d2h_bytes+=stats.phase_backup_bytes-before_backup;stats.phase_dynamic_ms+=milliseconds(started);
}
std::vector<double> PhaseCache::snapshot()const{
    std::vector<double> data;data.reserve(2*keys+2);data.insert(data.end(),counts_[2].begin(),counts_[2].end());data.insert(data.end(),heat_.begin(),heat_.end());
    data.push_back(decode_rows_);data.push_back(adjustment_rows_);return data;
}
void PhaseCache::restore_snapshot(const std::vector<double>& data){
    if(data.size()!=2*keys+2)throw std::invalid_argument("phase routing snapshot shape");
    std::copy(data.begin(),data.begin()+keys,counts_[2].begin());std::copy(data.begin()+keys,data.begin()+2*keys,heat_.begin());
    decode_rows_=data[2*keys];adjustment_rows_=data[2*keys+1];
}
void PhaseCache::write_heat(std::ostream& out)const{
    out<<"{\"loan_gpu_bytes\":"<<plan_.bytes<<",\"backup_ram_capacity\":"<<plan_.bytes<<",\"encoder_temporary_bytes\":"<<plan_.encoder_bytes
       <<",\"phase_host_staging_bytes\":"<<staging_->bytes<<",\"phase_host_pinned_bytes\":"<<pinned_staging_bytes()<<",\"committed_decode_rows\":"<<decode_rows_;
    const char* names[]={"prefill_counts","replay_counts","decode_counts"};
    for(int p=0;p<3;++p){out<<",\""<<names[p]<<"\":[";for(int i=0;i<keys;++i){if(i)out<<',';out<<counts_[p][i];}out<<']';}out<<'}';
}
} // namespace strata::models::deepseek41
