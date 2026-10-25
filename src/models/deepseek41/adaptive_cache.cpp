#include "adaptive_cache.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <iostream>

namespace strata::models::deepseek41 {
namespace {
constexpr int keys=40*384;
constexpr uint64_t no_ram=core::detail::kNoCacheComplement;
constexpr double candidate_heat=2.0,migration_gain=1.5;
double elapsed(std::chrono::steady_clock::time_point start){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
uint64_t largest(const std::array<uint64_t,40>& sizes){
    for(auto n:sizes)if(!n||n%256)throw std::invalid_argument("adaptive expert geometry");
    return *std::max_element(sizes.begin(),sizes.end());
}
}
uint64_t AdaptiveCache::staging_bytes(const std::array<uint64_t,40>& sizes,uint64_t budget){
    const auto stride=largest(sizes),capacity=std::min<uint64_t>(96,budget/2/stride);
    if(!capacity)throw std::invalid_argument("adaptive migration budget cannot hold one largest expert exchange");
    return 2*capacity*stride;
}
AdaptiveCache::AdaptiveCache(core::ExpertCache& cache,std::vector<ExpertKey>& hot,std::vector<uint64_t>& offsets,
                             uint8_t* ram,const std::array<uint64_t,40>& sizes,uint64_t budget,bool trace,bool grouped,bool pinned_ram,bool huge_pinned,
                             const std::array<int,40>* layout_groups,bool maya)
    :cache_(cache),hot_(hot),offsets_(offsets),ram_(ram),sizes_(sizes),budget_(budget),stride_(largest(sizes)),
     capacity_(std::min<uint64_t>(96,budget/2/stride_)),trace_(trace),grouped_(grouped),pinned_ram_(pinned_ram),maya_(maya),recent_(keys),
     staging_(std::make_unique<Pinned>(staging_bytes(sizes,budget),false,huge_pinned)),heat_(keys),previous_(keys),learned_(keys),
     generations_(hot.size()),overrides_(keys),sources_(keys),unavailable_(keys){
    if(!ram_||offsets_.size()!=keys||hot_.size()!=static_cast<size_t>(cache_.slots()))throw std::invalid_argument("adaptive cache ownership");
    if(!staging_->pinned)throw std::runtime_error("adaptive overlap requires pinned exchange buffers");
    for(size_t i=0;i<hot_.size();++i){const auto [l,e]=hot_[i];if(cache_.slot_of(l,e)!=static_cast<int>(i))throw std::invalid_argument("adaptive cache slot order");}
    for(int l=0;l<40;++l)groups_[l]=layout_groups?layout_groups->at(l):adaptive_group(l);
    rotation_eligible_=pinned_ram_;
    for(int l=0;l<40;++l){
        if(groups_[l]<0||groups_[l]>=40)throw std::invalid_argument("adaptive format group ID");
        // Two bounded spare pools retain the existing Q2/Q3 zero-copy rotation.
        // More layouts use direct pinned H2D and a host commit within the same budget.
        if(groups_[l]>=2)rotation_eligible_=false;
        for(int j=0;j<l;++j)if(groups_[j]==groups_[l]&&sizes[j]!=sizes[l]){
            rotation_eligible_=false;
            if(grouped_)throw std::invalid_argument("adaptive format-group sizes differ");
        }
    }
    for(int k=0;k<keys;++k)if(offsets_[k]!=no_ram)sources_[k]=ram_+offsets_[k];
    for(int g=0;g<2;++g)for(size_t i=0;i<capacity_;++i)
        spares_[g].push_back(staging_->data+(g*capacity_+i)*stride_);
    // A spare never crosses its format group after rotation, even though the
    // initial allocation uses max-size strides. No CUDA aliases are needed:
    // this engine computes experts on the GPU, never through host-mapped weights.
    std::cerr<<"DeepSeek host_memory adaptive ";staging_->info.write(std::cerr);std::cerr<<'\n';
}
AdaptiveCache::~AdaptiveCache(){if(host_job_.valid())host_job_.wait();cudaStreamSynchronize(copies_.value);}
bool AdaptiveCache::resident(int layer,int expert)const{return cache_.slot_of(layer,expert)>=0&&!unavailable_.at(layer*384+expert);}
const uint8_t* AdaptiveCache::cpu(int layer,int expert)const{
    const int key=layer*384+expert;return overrides_.at(key)?overrides_[key]:sources_.at(key);
}
void AdaptiveCache::validate(const Swap& s)const{
    if(s.slot<0||static_cast<size_t>(s.slot)>=hot_.size()||hot_[s.slot]!=s.out||
       cache_.slot_of(s.out.first,s.out.second)!=s.slot||cache_.slot_of(s.in.first,s.in.second)>=0||
       generations_[s.slot]!=s.generation||sizes_[s.in.first]!=s.bytes||sizes_[s.out.first]!=s.bytes||
       offsets_[adaptive_index(s.in)]!=s.offset||offsets_[adaptive_index(s.out)]!=no_ram||
       sources_[adaptive_index(s.in)]!=s.source||sources_[adaptive_index(s.out)]||
       (s.rotate&&spares_[groups_[s.in.first]].at(s.spare)!=s.saved)||
       (grouped_?groups_[s.in.first]!=groups_[s.out.first]:s.in.first!=s.out.first))
        throw std::logic_error("adaptive stale exchange ownership/generation");
}
void AdaptiveCache::choose(cudaStream_t readers,ExecutionStats& stats){
    std::vector<int32_t> slots(keys);
    for(int k=0;k<keys;++k)slots[k]=cache_.slot_of(k/384,k%384);
    swaps_.clear();std::array<size_t,40> spare{};
    for(const auto& choice:choose_adaptive_exchanges(sizes_,heat_,slots,offsets_,grouped_,capacity_,budget_,&groups_,maya_?&recent_:nullptr)){
        Swap s;static_cast<AdaptiveExchange&>(s)=choice;
        s.generation=generations_.at(s.slot);s.source=sources_.at(adaptive_index(s.in));
        s.direct=(aligned_||maya_)&&pinned_ram_;s.rotate=(aligned_||maya_)&&rotation_eligible_;
        s.spare=spare[groups_[s.in.first]]++;
        s.saved=s.rotate?spares_[groups_[s.in.first]].at(s.spare):saved(swaps_.size());
        validate(s);swaps_.push_back(s);
    }
    since_pick_=0;
    if(!maya_)for(int k=0;k<keys;++k){if(aligned_)learned_[k]+=heat_[k];heat_[k]*=.7;}
    if(swaps_.empty())return;
    prefill::LedgerScope trace(trace_,"ds41.adapt.d2h.submit");
    // Keep the old GPU expert readable during the copy back. H2D is a later state.
    gpu_check(cudaEventRecord(readers_done_.value,readers));
    gpu_check(cudaStreamWaitEvent(copies_.value,readers_done_.value,0));
    for(size_t i=0;i<swaps_.size();++i){
        const auto& s=swaps_[i];gpu_check(cudaMemcpyAsync(s.saved,cache_.device_slot(s.slot),s.bytes,cudaMemcpyDeviceToHost,copies_.value));
        stats.phase_dynamic_d2h_bytes+=s.bytes;
    }
    gpu_check(cudaEventRecord(copies_done_.value,copies_.value));
    if(!swaps_.front().direct)host_job_=std::async(std::launch::async,[this]{
        prefill::LedgerScope trace(trace_,"ds41.adapt.host.prepare");
        const auto start=std::chrono::steady_clock::now();
        for(size_t i=0;i<swaps_.size();++i)std::memcpy(incoming(i),swaps_[i].source,swaps_[i].bytes);
        return elapsed(start);
    });
    state_=State::saving;++batches_;
}
void AdaptiveCache::advance(cudaStream_t readers,ExecutionStats& stats,bool wait){
    if(state_==State::idle)return;
    const auto before=std::chrono::steady_clock::now();
    if(state_==State::saving||state_==State::loading){
        if(wait)gpu_check(cudaEventSynchronize(copies_done_.value));
        else {const auto ready=cudaEventQuery(copies_done_.value);if(ready==cudaErrorNotReady)return;gpu_check(ready);}
    }
    if((state_==State::saving||state_==State::committing)&&host_job_.valid()){
        if(wait)host_job_.wait();
        else if(host_job_.wait_for(std::chrono::seconds(0))!=std::future_status::ready)return;
        host_copy_ms_+=host_job_.get();
    }
    if(wait)stats.phase_host_wait_ms+=elapsed(before);
    if(state_==State::saving){
        prefill::LedgerScope trace(trace_,"ds41.adapt.h2d.submit");
        // All previously planned readers precede this fence. CPU readers from the
        // just-finished forward have drained before tick(), and the saved bytes are ready.
        gpu_check(cudaEventRecord(readers_done_.value,readers));
        gpu_check(cudaStreamWaitEvent(copies_.value,readers_done_.value,0));
        for(size_t i=0;i<swaps_.size();++i){
            const auto& s=swaps_[i];validate(s);const int old=adaptive_index(s.out);
            overrides_[old]=s.saved;unavailable_[old]=1;
            gpu_check(cudaMemcpyAsync(cache_.device_slot(s.slot),s.direct?s.source:incoming(i),s.bytes,cudaMemcpyHostToDevice,copies_.value));
            stats.phase_dynamic_h2d_bytes+=s.bytes;stats.h2d_expert_bytes+=s.bytes;
            if(s.direct){direct_bytes_+=s.bytes;avoided_prepare_+=s.bytes;}
        }
        gpu_check(cudaEventRecord(copies_done_.value,copies_.value));state_=State::loading;
    }else if(state_==State::loading){
        prefill::LedgerScope trace(trace_,"ds41.adapt.publish");
        for(const auto& s:swaps_)validate(s);
        for(const auto& s:swaps_){
            if(!cache_.replace_checked(s.out.first,s.out.second,s.in.first,s.in.second,s.slot,s.bytes))
                throw std::logic_error("adaptive slot publication rejected");
            hot_.at(s.slot)=s.in;++generations_[s.slot];
            unavailable_[adaptive_index(s.out)]=0;++published_;++stats.phase_migrations;
        }
        if(swaps_.front().rotate){
            // The completed H2D event releases incoming source leases. Layer
            // progress is called only after the demand issuer/CPU readers drain.
            // Old bytes stay where D2H wrote them; the former incoming block is
            // now a spare of the same format. Publish all owners on this thread.
            commit_owners();return;
        }
        // Incoming experts are now GPU-visible. Old experts remain CPU-visible in
        // exchange buffers until the next safe boundary publishes the RAM ownership flip.
        host_job_=std::async(std::launch::async,[this]{
            prefill::LedgerScope trace(trace_,"ds41.adapt.host.commit");
            const auto start=std::chrono::steady_clock::now();
            for(size_t i=0;i<swaps_.size();++i)std::memcpy(swaps_[i].source,swaps_[i].saved,swaps_[i].bytes);
            return elapsed(start);
        });
        state_=State::committing;
    }else{
        prefill::LedgerScope trace(trace_,"ds41.adapt.commit");
        commit_owners();
    }
}
void AdaptiveCache::commit_owners(){
    for(const auto& s:swaps_){
        const int in=adaptive_index(s.in),out=adaptive_index(s.out);
        if(sources_[in]!=s.source||sources_[out]||overrides_[out]!=s.saved)
            throw std::logic_error("adaptive RAM ownership publication rejected");
        sources_[out]=s.rotate?s.saved:s.source;sources_[in]=nullptr;
        if(s.rotate){
            auto& spare=spares_[groups_[s.in.first]].at(s.spare);
            if(spare!=s.saved)throw std::logic_error("adaptive spare generation changed");
            spare=s.source;++rotated_;avoided_commit_+=s.bytes;
        }
        offsets_[in]=no_ram;offsets_[out]=s.offset;overrides_[out]=nullptr;
        if(record_exchanges_)committed_exchanges_.push_back(s);
    }
    swaps_.clear();state_=State::idle;
}
void AdaptiveCache::observe(int tokens,const std::vector<double>& counts,bool window){
    if(tokens<0||counts.size()!=keys)throw std::invalid_argument("adaptive route counters");
    if(!tokens)return;
    for(int k=0;k<keys;++k){
        if(!std::isfinite(counts[k])||counts[k]<previous_[k])throw std::logic_error("adaptive routes went backwards/nonfinite");
        const double delta=counts[k]-previous_[k];heat_[k]+=delta;
        if(!window||maya_)learned_[k]+=delta;
        if(maya_)recent_[k]=delta>0;
        previous_[k]=counts[k];
    }
    rows_+=tokens;since_pick_+=window?1:tokens;
    if(window)++windows_;
    if(maya_){
        decay_rows_+=tokens;
        // Match Maya's whole-window boundary: reset the counter after decay.
        if(decay_rows_>=256){for(auto& h:heat_)h*=.9;decay_rows_=0;}
    }
}
void AdaptiveCache::observe_window(int tokens,const std::vector<double>& counts){
    if(aligned_||maya_)observe(tokens,counts,true);
}
void AdaptiveCache::tick(int tokens,const std::vector<double>& counts,cudaStream_t readers,ExecutionStats& stats){
    prefill::LedgerScope trace(trace_,"ds41.adapt.tick");const auto start=std::chrono::steady_clock::now();
    advance(readers,stats,false);
    if(!aligned_&&!maya_)observe(tokens,counts,false);
    if(state_==State::idle&&since_pick_>=(maya_?1:4))choose(readers,stats);
    stats.phase_dynamic_ms+=elapsed(start);
}
void AdaptiveCache::progress(cudaStream_t readers,ExecutionStats& stats){
    prefill::LedgerScope trace(trace_,"ds41.adapt.progress");
    const auto start=std::chrono::steady_clock::now();advance(readers,stats,false);
    stats.phase_dynamic_ms+=elapsed(start);
}
void AdaptiveCache::drain(cudaStream_t readers,ExecutionStats& stats){while(state_!=State::idle)advance(readers,stats,true);}
void AdaptiveCache::begin_request(cudaStream_t readers,ExecutionStats& stats){
    drain(readers,stats);std::fill(previous_.begin(),previous_.end(),0);
    if(maya_){since_pick_=0;std::fill(recent_.begin(),recent_.end(),0);}
    else if(!aligned_)rows_=since_pick_=0;
}
void AdaptiveCache::write_status(std::ostream& out)const{
    const char* name=state_==State::idle?"idle":state_==State::saving?"saving":state_==State::loading?"loading":"committing";
    out<<"{\"scope\":\""<<(grouped_?"format-group":"layer")<<"\",\"state\":\""<<name<<"\",\"pending_swaps\":"<<swaps_.size()<<",\"batches\":"<<batches_<<",\"published_swaps\":"<<published_
       <<",\"observed_rows\":"<<rows_<<",\"migration_budget_bytes\":"<<budget_<<",\"host_staging_bytes\":"<<staging_->bytes
       <<",\"host_copy_worker_ms\":"<<host_copy_ms_<<",\"policy\":\""<<(maya_?"maya":"legacy")<<"\",\"period\":"<<(maya_?1:4)<<",\"period_unit\":\""<<((aligned_||maya_)?"verify_window":"accepted_row")<<"\",\"decay\":"<<(maya_?.9:.7)<<",\"decay_rows\":"<<decay_rows_
       <<",\"windows\":"<<windows_<<",\"exchange_limit\":"<<capacity_<<",\"pinned_direct_bytes\":"<<direct_bytes_
       <<",\"rotated_exchanges\":"<<rotated_<<",\"avoided_prepare_bytes\":"<<avoided_prepare_<<",\"avoided_commit_bytes\":"<<avoided_commit_
       <<",\"minimum_candidate_heat\":"<<(maya_?0:candidate_heat)<<",\"minimum_gain\":"<<(maya_?1:migration_gain)<<",\"relative_margin\":"<<(maya_?1.25:1.0)
       <<",\"heat_sum\":"<<std::accumulate(heat_.begin(),heat_.end(),0.0)<<",\"previous_sum\":"<<std::accumulate(previous_.begin(),previous_.end(),0.0)
       <<",\"cpu_overrides\":"<<std::count_if(overrides_.begin(),overrides_.end(),[](auto p){return p!=nullptr;})
       <<",\"unavailable_slots\":"<<std::count(unavailable_.begin(),unavailable_.end(),1)<<'}';
}
} // namespace strata::models::deepseek41
