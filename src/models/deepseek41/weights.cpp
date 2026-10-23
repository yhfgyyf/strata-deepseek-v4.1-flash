#include "runtime_internal.hpp"
#include "adaptive_cache.hpp"
#include "strata/kernels/cpu/deepseek41_down.hpp"
#include "phase_cache.hpp"
#include <algorithm>
#include <climits>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::models::deepseek41 {
namespace {
bool packed_projection(const std::string& name) {
    const auto suffix=name.substr(name.find('.',4)+1);
    return suffix=="attn_q_a.weight" || suffix=="attn_q_b.weight" || suffix=="attn_kv.weight" ||
           suffix=="attn_output_a.weight" || suffix=="attn_output_b.weight" ||
           suffix=="ffn_gate_shexp.weight" || suffix=="ffn_up_shexp.weight" || suffix=="ffn_down_shexp.weight";
}
int storage_type(const TensorView& v,const std::string& name,Arithmetic arithmetic,const Model& model,bool draft,bool dense_q8) {
    if(draft&&(v.info->type==2||(v.info->type==8&&model.expert(0,0,true).type!=39)))return v.info->type;
    if(dense_q8 && arithmetic==Arithmetic::production && (v.info->type==8||v.info->type==14) && packed_projection(name))return v.info->type;
    // Read the large vocabulary heads in native Q8_0/Q6_K with FP32 activations. This preserves their weights and
    // avoids a 2.65 GB FP32 copy; the fixed five-row DSpark head and Markov tables use the same contract.
    if((v.info->type==8||v.info->type==14) && (name=="output.weight" || (draft &&
        (name=="token_embd.weight"||name=="markov_w1.weight"||name=="markov_w2.weight"))))return v.info->type;
    const bool compressor_f32=name.find("attn_compressor_")!=std::string::npos &&
        model.ownership()[std::stoi(name.substr(4))].compression>1;
    if (arithmetic == Arithmetic::f32 || v.info->type == 0 || v.info->shape.size()==1 || name == "output.weight" || compressor_f32) return 0;
    return 30;
}
bool include_weight(TensorRole role) { return role != TensorRole::engram && role != TensorRole::token_map && role != TensorRole::routed_expert; }
bool vocab_weight(const std::string& name) {
    return name=="token_embd.weight"||name=="output.weight"||name=="markov_w1.weight"||name=="markov_w2.weight";
}
}
WeightBank::WeightBank(const Model& model,Arithmetic arithmetic,cudaStream_t stream,bool draft,bool vision,bool dense_q8,
                       const std::vector<int32_t>* draft_vocab,const std::vector<int32_t>* draft_input_vocab) {
    if(draft_vocab&&(!draft||vision||draft_vocab->empty()))throw std::invalid_argument("draft vocabulary weight bank");
    uint64_t bytes = 0;
    for (const auto& [name,v] : model.tensors(draft,vision)) {
        const auto& t = *v.info;
        if (!include_weight(v.role)) continue;
        Matrix m; m.type = storage_type(v,t.name,arithmetic,model,draft,dense_q8); m.cols = t.shape[0]; m.elements = 1;
        m.packed_bf16=(dense_q8&&(m.type==8||m.type==14)&&packed_projection(t.name))||(draft&&(m.type==2||m.type==8)&&!vocab_weight(t.name));
        for (auto dim : t.shape) m.elements *= dim;
        m.rows = m.elements / m.cols;
        const auto* selected=(t.name=="token_embd.weight"||t.name=="markov_w1.weight")?draft_input_vocab:draft_vocab;
        if(selected&&vocab_weight(t.name)){
            if((m.type!=8&&m.type!=2&&m.type!=14&&!(t.type==30&&(m.type==30||m.type==0)))||
               t.shape.size()!=2||m.rows!=129280)throw std::runtime_error("draft vocabulary tensor layout");
            m.rows=selected->size();m.elements=m.rows*m.cols;
        }
        bytes += aligned(m.type==2?m.elements/32*18:m.type==8?m.elements/32*34:m.type==14?m.elements/256*210:m.elements*(m.type==0?4:2));matrices_.emplace(t.name,m);
    }
    arena_ = std::make_unique<core::DeviceArena>(bytes);
    core::DeviceArena temporary(128ull << 20);
    auto* packed = allocate<uint8_t>(temporary,32ull << 20);
    auto* decoded = allocate<float>(temporary,16ull << 20);
    int last_layer = -1;
    for (auto& [name,m] : matrices_) {
        const auto& v = model.tensor(name,draft,vision);
        m.data=arena_->alloc(m.type==2?m.elements/32*18:m.type==8?m.elements/32*34:m.type==14?m.elements/256*210:m.elements*(m.type==0?4:2));
        const auto* selected_bf16=(name=="token_embd.weight"||name=="markov_w1.weight")?draft_input_vocab:draft_vocab;
        if(selected_bf16&&vocab_weight(name)&&v.info->type==30){
            const uint64_t row_bytes=uint64_t(m.cols)*2,rows_per_chunk=(32ull<<20)/row_bytes;
            std::vector<uint8_t> rows(rows_per_chunk*row_bytes);
            for(uint64_t begin=0;begin<selected_bf16->size();begin+=rows_per_chunk){
                const uint64_t count=std::min<uint64_t>(rows_per_chunk,selected_bf16->size()-begin);
                for(uint64_t i=0;i<count;++i){
                    const int id=selected_bf16->at(begin+i);
                    if(id<0||id>=129280)throw std::invalid_argument("draft vocabulary row ID");
                    std::memcpy(rows.data()+i*row_bytes,v.data+uint64_t(id)*row_bytes,row_bytes);
                }
                if(m.type==30)gpu_check(cudaMemcpyAsync(static_cast<uint8_t*>(m.data)+begin*row_bytes,
                    rows.data(),count*row_bytes,cudaMemcpyHostToDevice,stream));
                else{
                    gpu_check(cudaMemcpyAsync(packed,rows.data(),count*row_bytes,cudaMemcpyHostToDevice,stream));
                    gpu_check(dequant_gpu(packed,30,static_cast<float*>(m.data)+begin*m.cols,count*m.cols,stream));
                }
                gpu_check(cudaStreamSynchronize(stream));
            }
            continue;
        }
        if(m.type==8||m.type==2||m.type==14){
            const auto* selected=(name=="token_embd.weight"||name=="markov_w1.weight")?draft_input_vocab:draft_vocab;
            if(selected&&vocab_weight(name)){
                const uint64_t row_bytes=m.type==14?m.cols/256*210:m.cols/32*(m.type==2?18:34),rows_per_chunk=(32ull<<20)/row_bytes;
                std::vector<uint8_t> rows(rows_per_chunk*row_bytes);
                for(uint64_t begin=0;begin<selected->size();begin+=rows_per_chunk){
                    const auto count=std::min<uint64_t>(rows_per_chunk,selected->size()-begin);
                    for(uint64_t i=0;i<count;++i){
                        const int id=selected->at(begin+i);
                        if(id<0||id>=129280)throw std::invalid_argument("draft vocabulary row ID");
                        std::memcpy(rows.data()+i*row_bytes,v.data+uint64_t(id)*row_bytes,row_bytes);
                    }
                    gpu_check(cudaMemcpyAsync(static_cast<uint8_t*>(m.data)+begin*row_bytes,rows.data(),count*row_bytes,cudaMemcpyHostToDevice,stream));
                    gpu_check(cudaStreamSynchronize(stream));
                }
                continue;
            }
            for(uint64_t offset=0;offset<v.bytes;offset+=32ull<<20){
                const uint64_t count=std::min<uint64_t>(32ull<<20,v.bytes-offset);
                gpu_check(cudaMemcpyAsync(static_cast<uint8_t*>(m.data)+offset,v.data+offset,count,cudaMemcpyHostToDevice,stream));
                gpu_check(cudaStreamSynchronize(stream));
            }
            continue;
        }
        int block = 0, block_bytes = 0;
        if (!block_geometry(v.info->type,block,block_bytes)) throw std::runtime_error("DeepSeek weight type");
        const uint64_t max_elements = std::min<uint64_t>(16ull << 20, (32ull << 20) / block_bytes * block);
        for (uint64_t offset = 0; offset < m.elements;) {
            const uint64_t count = std::min<uint64_t>(max_elements / block * block,m.elements - offset);
            const uint64_t packed_bytes = count / block * block_bytes;
            gpu_check(cudaMemcpyAsync(packed,v.data + offset / block * block_bytes,packed_bytes,cudaMemcpyHostToDevice,stream));
            gpu_check(dequant_gpu(packed,v.info->type,decoded,count,stream));
            if (m.type == 0) gpu_check(cudaMemcpyAsync(static_cast<float*>(m.data)+offset,decoded,count*4,cudaMemcpyDeviceToDevice,stream));
            else gpu_check(float_to_bf16_gpu(decoded,static_cast<uint16_t*>(m.data)+offset,count,stream));
            gpu_check(cudaStreamSynchronize(stream)); offset += count;
        }
        if (name.rfind("blk.",0) == 0) {
            const int l = std::stoi(name.substr(4));
            if (l != last_layer) { std::cerr << "DeepSeek dense layer " << l << '\n'; last_layer = l; }
        }
    }
    std::cerr << "DeepSeek dense bytes=" << arena_->used() << '\n';
}
GgufExperts::GgufExperts(const Model& model,const RuntimeOptions& options,cudaStream_t upload) : model_(model) {
    std::string error;
    const auto sizes = model.expert_bytes();
    for (int l = 0; l < 40; ++l) {
        const auto e = model.expert(l,0);
        if (!kernels::cpu::native_fmt(e.type,e.down_type,5120,2304,formats_[l],error)) throw std::runtime_error(error);
        formats_[l].swiglu_limit = 10.f;
        formats_[l].bf16_io = options.arithmetic == Arithmetic::production;
        if(formats_[l].bf16_io){
            if(!(options.cpu_gu_q8_0 && (e.type==10||e.type==11)?
                kernels::cpu::native_enable_q8_0_gu(formats_[l],error):kernels::cpu::native_enable_bf16_gu(formats_[l],error)))
                throw std::runtime_error(error);
            const bool q3=kernels::cpu::deepseek41_down_fusion_enabled()&&e.down_type==11;
            if(!(q3?kernels::cpu::native_enable_q3_q8_1_down(formats_[l],error):kernels::cpu::native_enable_bf16_down(formats_[l],error)))
                throw std::runtime_error(error);
        }
        if (formats_[l].bytes != sizes[l]) throw std::runtime_error("DeepSeek expert layout mismatch");
    }
    std::array<int,40> layout_groups{};
    int groups=0;
    for(int l=0;l<40;++l){
        const auto& f=formats_[l];int representative=-1;
        for(int j=0;j<l;++j){const auto& ref=formats_[j];
            if(f.gu_type==ref.gu_type&&f.d_type==ref.d_type&&f.gu_row==ref.gu_row&&f.d_row==ref.d_row&&
               f.up_off==ref.up_off&&f.down_off==ref.down_off&&f.bytes==ref.bytes&&f.tail_off==ref.tail_off){
                representative=j;break;
            }
        }
        layout_groups[l]=representative<0?groups++:layout_groups[representative];
        if(f.bytes%256)throw std::invalid_argument("grouped residency requires aligned expert blobs");
    }
    offsets_.assign(40 * 384,core::detail::kNoCacheComplement);
    for(auto& counts:route_counts_)counts.assign(40*384,0);
    physical_counts_.assign(40*384,0);
    if (options.arithmetic != Arithmetic::f32) {
        if (!options.profile.empty()) {
            std::vector<std::pair<int32_t,int32_t>> ranked; int64_t profile_slots = 0;
            if (!core::read_expert_profile(options.profile,40,384,ranked,profile_slots,error)) throw std::runtime_error(error);
            if (options.hot_front > UINT64_MAX - options.hot_back) throw std::overflow_error("DeepSeek expert budget overflow");
            hot_ = allocate_ranked_hot_experts(sizes,ranked,options.hot_front+options.hot_back).experts;
            if (hot_.empty()) throw std::invalid_argument("DeepSeek first ranked expert does not fit GPU capacity");
        } else {
            const auto quotas = allocate_hot_experts(sizes,options.hot_front,options.hot_back);
            for (int e = 0; e < 384; ++e) for (int l = 0; l < 40; ++l)
                if (e < quotas.count[l]) hot_.emplace_back(l,e);
        }
        std::vector<uint64_t> layer_sizes(sizes.begin(),sizes.end());
        if (!core::detail::make_cache_complement_plan(40,384,layer_sizes,hot_,{},offsets_,ram_bytes_,error)) throw std::runtime_error(error);
        std::vector<uint64_t> host_segments;
        if(options.pinned_experts && options.huge_pinned_memory && ram_bytes_){
            std::size_t segments=0;
            if(const char* value=std::getenv("STRATA_DS41_BLOCK_SEGMENTS")){
                if(std::string(value)!="40")throw std::invalid_argument("DeepSeek explicit host segment count must be 40");
                segments=40;
            }
            host_segments=make_expert_host_segments(layer_sizes,384,offsets_,segments);
            ram_bytes_=host_segments.back(); // Include padding in RAM and phase-cache budgets.
        }
        std::vector<ExpertKey> core=hot_;
        if(options.phase_residency&&options.experts!=ExpertMode::cpu){
            if(options.slots!=1)throw std::invalid_argument("phase residency currently requires one request slot");
            std::vector<ExpertKey> prefill_rank;int64_t prefill_slots=0;
            if(!options.prefill_profile.empty()&&!core::read_expert_profile(options.prefill_profile,40,384,prefill_rank,prefill_slots,error))throw std::runtime_error(error);
            const uint64_t allowance=ram_bytes_>=options.ram_experts?0:options.ram_experts-ram_bytes_;
            auto plan=make_phase_plan(sizes,hot_,prefill_rank,std::min(options.borrow_bytes,allowance));core=plan.core;
            if(plan.bytes)phase_=std::make_unique<PhaseCache>(std::move(plan),[this](ExpertKey key){
                const uint64_t offset=offsets_.at(key.first*384+key.second);
                return ram_&&offset!=core::detail::kNoCacheComplement?ram_.get()+offset:nullptr;
            },upload,options.migration_bytes,false,options.trace);
        }
        std::vector<int64_t> slot_sizes;
        for (const auto& [l,e] : core) { (void)e; slot_sizes.push_back(sizes[l]); }
        if (!slot_sizes.empty() && !cache_.open_sized(slot_sizes,40,384,error)) throw std::runtime_error(error);
        core::detail::HostMemory memory;
        if (!core::detail::host_available_memory(memory)) throw std::runtime_error("DeepSeek host memory unavailable");
        const uint64_t total_ram=ram_bytes_+backup_bytes();
        const uint64_t adaptive_staging=options.adaptive_experts?AdaptiveCache::staging_bytes(sizes,options.migration_bytes):0;
        if (total_ram > options.ram_experts || memory.available < total_ram + options.host_headroom + options.engram_cache + options.ring + phase_staging_bytes()+adaptive_staging)
            throw std::runtime_error("DeepSeek RAM complement/headroom does not fit; increase GPU expert budget or reduce other explicit budgets");
        ram_.allocate(ram_bytes_,options.pinned_experts && options.huge_pinned_memory,host_segments);
    }
#ifndef _WIN32
    for(const auto& [name,view]:model.tensors())if(view.role==TensorRole::routed_expert){
        const auto& path=view.file->path();
        if(!fds_.count(path))fds_[path]=::open(path.c_str(),O_RDONLY);
    }
#endif
    Pinned staging(*std::max_element(sizes.begin(),sizes.end()));
    // Cold blobs are copied once; hot blobs have only a GPU copy after startup.
    size_t next_hot = 0;
    std::vector<int> hot_slot(40*384,-1);
    std::vector<int> phase_slot(40*384,-1);
    if(phase_)for(size_t i=0;i<phase_->plan().decode.size();++i){const auto key=phase_->plan().decode[i].key;phase_slot[key.first*384+key.second]=i;}
    for (const auto& [l,e] : hot_) {
        if(phase_slot[l*384+e]>=0)continue;
        const int slot = cache_.admit(l,e);
        if (slot != static_cast<int>(next_hot++)) throw std::runtime_error("DeepSeek sized-cache admission order");
        hot_slot[l*384+e] = slot;
    }
    if (options.arithmetic != Arithmetic::f32) for (int l = 0; l < 40; ++l) {
        for (int e = 0; e < 384; ++e) {
            const int slot = hot_slot[l*384+e];
            if (slot >= 0) {
                copy_blob(l,e,staging.data);
                if (!cache_.fill_slot(slot,staging.data,upload,error,sizes[l])) throw std::runtime_error(error);
                gpu_check(cudaStreamSynchronize(upload));
            } else if(phase_slot[l*384+e]>=0){
                copy_blob(l,e,staging.data);phase_->initialize(phase_slot[l*384+e],staging.data);gpu_check(cudaStreamSynchronize(upload));
            } else copy_blob(l,e,ram_.get() + offsets_[l*384+e]);
            release_mapped(l,e);
        }
        std::cerr << "DeepSeek resident experts layer " << l << '\n';
    }
    assembled_.resize(*std::max_element(sizes.begin(),sizes.end()));
    if(options.pinned_experts){
        if(!ram_ || !ram_bytes_)throw std::invalid_argument("pinned expert sources require a RAM complement");
        ram_.register_if_needed();
        std::cerr<<"DeepSeek host_memory expert ";ram_.memory_info().write(std::cerr);std::cerr<<'\n';
    }
    if(options.adaptive_experts)adaptive_=std::make_unique<AdaptiveCache>(cache_,hot_,offsets_,ram_.get(),sizes,options.migration_bytes,options.trace,options.adaptive_groups,ram_.pinned(),options.huge_pinned_memory,&layout_groups);
}
GgufExperts::~GgufExperts() {
#ifndef _WIN32
    for(const auto& [path,fd]:fds_)if(fd>=0)::close(fd);
#endif
}
bool GgufExperts::copy_blob(int64_t layer,int64_t expert,uint8_t* out) {
    if(layer<0||layer>=40||expert<0||expert>=384)throw std::out_of_range("DeepSeek expert assembly ID");
    const auto e = model_.expert(layer,expert);
    const auto& f = formats_[layer];
    std::memcpy(out,e.gate,f.up_off); std::memcpy(out+f.up_off,e.up,f.up_off);
    std::memcpy(out+f.down_off,e.down,f.bytes-f.down_off);
    return true;
}
const uint8_t* GgufExperts::blob(int64_t layer,int64_t expert) {
    if (layer < 0 || layer >= 40 || expert < 0 || expert >= 384) throw std::out_of_range("DeepSeek expert ID");
    const uint64_t offset = offsets_[layer*384+expert];
    if(adaptive_)if(const auto* saved=adaptive_->cpu(layer,expert))return saved;
    if(phase_)if(const auto* saved=phase_->cpu({layer,expert}))return saved;
    if (ram_ && offset != core::detail::kNoCacheComplement) return ram_.get()+offset;
    copy_blob(layer,expert,assembled_.data()); return assembled_.data();
}
const uint8_t* GgufExperts::blob_stable(int64_t layer,int64_t expert) {
    if (transient(layer,expert)) throw std::runtime_error("DeepSeek GPU-held expert has no stable RAM duplicate");
    return blob(layer,expert);
}
const uint8_t* GgufExperts::pinned_blob(int layer,int expert) {
    // Overrides live in AdaptiveCache's pinned exchange allocation. Its RAM
    // commit is published only at a layer boundary after the issuer DMA drain.
    if(ram_.pinned() && adaptive_)if(const auto* saved=adaptive_->cpu(layer,expert))return saved;
    if(!ram_.pinned() || transient(layer,expert))return nullptr;
    const auto* pointer=blob_stable(layer,expert);
    const auto base=reinterpret_cast<uintptr_t>(ram_.get()),at=reinterpret_cast<uintptr_t>(pointer);
    return at>=base && at-base<=ram_bytes_-formats_[layer].bytes?pointer:nullptr;
}
bool GgufExperts::transient(int64_t layer,int64_t expert) const {
    if(adaptive_&&adaptive_->cpu(layer,expert))return false;
    if(phase_&&phase_->cpu({layer,expert}))return false;
    return !ram_ || offsets_.at(layer*384+expert) == core::detail::kNoCacheComplement;
}
const uint8_t* GgufExperts::gpu(int layer,int expert,cudaStream_t stream,ExecutionStats& stats,int phase) {
    if(adaptive_&&!adaptive_->resident(layer,expert))return nullptr;
    const int slot = cache_.slot_of(layer,expert);
    if(slot>=0)return cache_.device_slot(slot);
    return phase_?phase_->gpu({layer,expert},stream,stats,phase):nullptr;
}
uint64_t GgufExperts::ram_bytes()const{return ram_bytes_+backup_bytes();}
uint64_t GgufExperts::gpu_bytes()const{return cache_.bytes()+(phase_?phase_->gpu_bytes():0);}
uint64_t GgufExperts::backup_bytes()const{return phase_?phase_->backup_bytes():0;}
uint64_t GgufExperts::phase_staging_bytes()const{return phase_?phase_->staging_bytes():0;}
uint64_t GgufExperts::adaptive_staging_bytes()const{return adaptive_?adaptive_->staging_bytes():0;}
std::vector<ExpertKey> GgufExperts::hot_pairs()const{
    if(!adaptive_)return hot_;
    std::vector<ExpertKey> ready;for(const auto& key:hot_)if(adaptive_->resident(key.first,key.second))ready.push_back(key);
    return ready;
}
bool GgufExperts::begin_prefill(int remaining,int chunk,bool ced,cudaStream_t stream,ExecutionStats& stats){
    if(adaptive_)adaptive_->begin_request(stream,stats);
    for(auto& counts:route_counts_)std::fill(counts.begin(),counts.end(),0);
    std::fill(physical_counts_.begin(),physical_counts_.end(),0);
    resident_hits_.fill(0);
    return phase_&&phase_->begin_prefill(remaining,chunk,ced,stream,stats);
}
void GgufExperts::encoder_layer_done(int layer,cudaStream_t stream,ExecutionStats& stats){if(phase_)phase_->encoder_layer_done(layer,stream,stats);}
void GgufExperts::finish_prefill(cudaStream_t stream,ExecutionStats& stats){if(phase_)phase_->finish_prefill(stream,stats);}
void GgufExperts::observe_routes(int layer,const int* ids,int tokens,int phase){
    if(phase_)phase_->observe(layer,ids,tokens,phase);
    else for(int i=0;i<tokens*6;++i){
        route_counts_.at(phase).at(layer*384+ids[i])+=1;
        if(adaptive_&&adaptive_->resident(layer,ids[i]))++resident_hits_.at(phase);
    }
}
void GgufExperts::adjust_decode(int tokens,cudaStream_t stream,ExecutionStats& stats){
    if(phase_)phase_->adjust_decode(tokens,stream,stats);
    if(adaptive_)adaptive_->tick(tokens,route_counts_[2],stream,stats);
}
void GgufExperts::observe_physical_routes(int layer,const int* ids,int tokens){
    if(adaptive_)for(int i=0;i<tokens*6;++i)physical_counts_.at(layer*384+ids[i])+=1;
}
void GgufExperts::observe_decode_window(int tokens){
    if(adaptive_)adaptive_->observe_window(tokens,physical_counts_);
}
void GgufExperts::progress_decode(cudaStream_t stream,ExecutionStats& stats){if(adaptive_)adaptive_->progress(stream,stats);}
int GgufExperts::until_adjustment()const{return phase_?phase_->until_adjustment():INT_MAX;}
std::vector<double> GgufExperts::heat_snapshot()const{
    if(phase_)return phase_->snapshot();
    auto result=route_counts_[2];if(adaptive_)result.push_back(resident_hits_[2]);return result;
}
void GgufExperts::restore_heat(const std::vector<double>& snapshot){
    if(phase_)phase_->restore_snapshot(snapshot);
    else if(adaptive_){
        if(snapshot.size()!=40*384+1)throw std::invalid_argument("adaptive routing snapshot");
        std::copy_n(snapshot.begin(),40*384,route_counts_[2].begin());resident_hits_[2]=snapshot.back();
    }else route_counts_[2]=snapshot;
}
void GgufExperts::write_heat(std::ostream& out)const{
    if(phase_){phase_->write_heat(out);return;}
    out<<"{\"phase_residency\":false";const char* names[]={"prefill_counts","replay_counts","decode_counts"};
    for(int p=0;p<3;++p){out<<",\""<<names[p]<<"\":[";for(size_t i=0;i<route_counts_[p].size();++i){if(i)out<<',';out<<route_counts_[p][i];}out<<']';}
    if(adaptive_){
        out<<",\"physical_decode_counts\":[";
        for(size_t i=0;i<physical_counts_.size();++i){if(i)out<<',';out<<physical_counts_[i];}out<<']';
        out<<",\"resident_route_hits\":["<<resident_hits_[0]<<','<<resident_hits_[1]<<','<<resident_hits_[2]<<"],\"adaptive\":";
        adaptive_->write_status(out);
    }
    out<<'}';
}
void GgufExperts::write_layout(std::ostream& out)const{
    out<<'[';
    for(size_t slot=0;slot<hot_.size();++slot) {
        const auto [layer,expert]=hot_[slot];if(slot)out<<',';
        const auto offset=static_cast<const uint8_t*>(cache_.device_slot(slot))-static_cast<const uint8_t*>(cache_.device_slot(0));
        out<<'['<<slot<<','<<layer<<','<<expert<<','<<offset<<','<<formats_[layer].bytes<<']';
    }
    out<<']';
}
void GgufExperts::save_learned_profile(const std::string& path,cudaStream_t stream){
    if(!adaptive_)throw std::logic_error("profile learning requires adaptive experts");
    // Publish complete exchanges before snapshotting; no half-replaced slot is saved.
    ExecutionStats unused;adaptive_->drain(stream,unused);
    std::vector<uint8_t> resident(40*384);
    for(const auto key:hot_)resident[adaptive_index(key)]=1;
    const auto ranked=core::rank_learned_profile(40,384,resident,adaptive_->learned_heat(),hot_);
    std::string error;
    if(!core::write_expert_profile(path,40,384,ranked,error))throw std::runtime_error(error);
}
void GgufExperts::release_mapped(int layer,int expert) const {
#ifndef _WIN32
    const auto e = model_.expert(layer,expert);
    const auto& f = formats_[layer];
    for (const auto& span : {std::pair{e.gate,f.up_off},std::pair{e.up,f.up_off},std::pair{e.down,f.bytes-f.down_off}}) {
        const uintptr_t begin = (reinterpret_cast<uintptr_t>(span.first) + 4095) & ~uintptr_t(4095);
        const uintptr_t end = (reinterpret_cast<uintptr_t>(span.first) + span.second) & ~uintptr_t(4095);
        if (end > begin) ::madvise(reinterpret_cast<void*>(begin),end-begin,MADV_DONTNEED);
    }
    if (!fds_.empty()) {
        for (const char* suffix : {"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}) {
            const auto& view = model_.tensor("blk." + std::to_string(layer) + "." + suffix);
            const uint64_t count = suffix[4] == 'd' ? f.bytes-f.down_off : f.up_off;
            const uint64_t offset = view.file->data_start() + view.info->offset + count*expert;
            const auto fd=fds_.at(view.file->path());
            if(fd>=0)::posix_fadvise(fd,offset,count,POSIX_FADV_DONTNEED);
        }
    }
#else
    (void)layer; (void)expert;
#endif
}
} // namespace strata::models::deepseek41
