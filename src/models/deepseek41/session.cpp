#include "runtime_internal.hpp"
#include "strata/models/deepseek41/adaptive_verification.hpp"
#include "expert_pipeline.hpp"
#include "strata/core/spec_prob.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

namespace strata::models::deepseek41 {
namespace {
constexpr char prefix_magic[16]="STRATA-DS41-P03";
constexpr char main_sha[]="11f46543370256ef616b6e458b6950e148625b5f8b7545173d124445d3393c53";
void write_bytes(std::ostream& out,const void* data,uint64_t size) {
    out.write(static_cast<const char*>(data),size);if(!out)throw std::runtime_error("DeepSeek prefix write");
}
void read_bytes(std::istream& in,void* data,uint64_t size) {
    in.read(static_cast<char*>(data),size);if(!in)throw std::runtime_error("DeepSeek truncated prefix");
}
template<class T> void write_scalar(std::ostream& out,const T& value){write_bytes(out,&value,sizeof(value));}
template<class T> T read_scalar(std::istream& in){T value;read_bytes(in,&value,sizeof(value));return value;}
uint64_t prefix_checksum(const std::string& path,uint64_t bytes) {
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("DeepSeek prefix checksum open");
    std::array<uint8_t,65536> buffer;uint64_t hash=14695981039346656037ull;
    while(bytes){const uint64_t n=std::min<uint64_t>(bytes,buffer.size());read_bytes(in,buffer.data(),n);
        for(uint64_t i=0;i<n;++i){hash^=buffer[i];hash*=1099511628211ull;}bytes-=n;}
    return hash;
}
void device_file(std::fstream& file,void* device,uint64_t bytes,bool save,cudaStream_t stream) {
    std::vector<uint8_t> chunk(std::min<uint64_t>(bytes,1<<20));
    for(uint64_t offset=0;offset<bytes;offset+=chunk.size()) {
        const auto size=std::min<uint64_t>(chunk.size(),bytes-offset);
        if(save){gpu_check(cudaMemcpyAsync(chunk.data(),static_cast<uint8_t*>(device)+offset,size,cudaMemcpyDeviceToHost,stream));gpu_check(cudaStreamSynchronize(stream));write_bytes(file,chunk.data(),size);}
        else{read_bytes(file,chunk.data(),size);gpu_check(cudaMemcpyAsync(static_cast<uint8_t*>(device)+offset,chunk.data(),size,cudaMemcpyHostToDevice,stream));gpu_check(cudaStreamSynchronize(stream));}
    }
}
int replay_start(int count,const std::vector<ImageEmbedding>& images) {
    int start=std::max(0,count-128);bool changed;
    do {changed=false;for(const auto& image:images)
        if(image.begin<start && image.begin+image.tokens()>start){start=image.begin;changed=true;}}
    while(changed);
    if(count-start>1152)throw std::runtime_error("CED image replay span exceeds 1152 tokens; use full prefill");
    return start;
}
}
struct Session::Impl::Snapshot {
    std::array<int64_t,40> local_position{},global_position{},global_rows{};
    std::array<int,40> local_count{};
    std::array<bool,40> partial{};
    EngramHistory history;
    int64_t position=0,encoder=0,decoder=0;
    int64_t draft_position=0;
    size_t token_count=0;
    std::vector<float> last_logits;
    std::vector<double> routing_heat;
};
Session::Impl::Impl(Runtime::Impl& r) : runtime(r),work(*r.work),capture(r.options.capture_directory) {
    capture.stages=r.options.capture_stages;
    state=std::make_unique<core::DeviceArena>(r.options.state/r.options.slots);
    const bool f32=r.options.arithmetic==Arithmetic::f32;
    partial_owner.fill(-1);int owner=0;
    for(int layer=0;layer<40;++layer) {
        if(f32)local[layer].values=allocate<float>(*state,128*512);
        else local[layer].packed=allocate<uint8_t>(*state,128*528);
        const auto& own=r.model.ownership()[layer];
        if(!own.owns_global())continue;
        partial_owner[layer]=owner++;
        auto& cache=global[layer];const int64_t rows=std::max(1,r.options.context/own.compression);
        if(f32){cache.values=allocate<float>(*state,rows*512);cache.index_values=allocate<float>(*state,rows*128);}
        else{cache.packed=allocate<uint8_t>(*state,((rows+63)/64)*64*288);cache.index_packed=allocate<uint8_t>(*state,rows*68);}
        cache.partial_kv=allocate<float>(*state,512);cache.partial_gate=allocate<float>(*state,512);
    }
    boundary_h=allocate<float>(*state,replay_capacity*20480);boundary_pre=allocate<float>(*state,replay_capacity*4);
    draft_hidden=allocate<float>(*state,128*15360);snapshot_hidden=allocate<float>(*state,128*15360);
    snapshot_local=allocate<uint8_t>(*state,40*128*(f32?512*4:528));snapshot_partial=allocate<float>(*state,4*2*512);
    verified_partial=allocate<float>(*state,4*8*2*512);
    if(r.options.dspark)draft_kv=std::make_unique<DraftKvCache>(*state,f32);
    gpu_check(cudaMemsetAsync(state->base(),0,state->used(),r.stream.value));
    gpu_check(cudaStreamSynchronize(r.stream.value));
    std::cerr<<"DeepSeek slot state used="<<state->used()<<" reserved="<<state->capacity()<<'\n';
}
Session::Impl::~Impl(){
    if(engram_ticket)engram_ticket->cancel();
    if(engram_lookahead)engram_lookahead->cancel();
    if(engram_pipeline)engram_pipeline->stop();
    try{runtime.experts->finish_prefill(runtime.stream.value,counters);}catch(...){}
    cudaStreamSynchronize(runtime.stream.value);
}
void Session::Impl::check_cancel(){if(cancelled.load()){if(engram_ticket)engram_ticket->cancel();if(engram_lookahead)engram_lookahead->cancel();throw std::runtime_error("DeepSeek request cancelled");}}
void Session::Impl::prefetch_engram(const std::vector<int>& prompt,int64_t start) {
    if(!runtime.options.engram_pipeline || start>=static_cast<int64_t>(prompt.size()))return;
    if(engram_lookahead || history.position!=start)throw std::logic_error("Engram lookahead history/credit");
    const int count=std::min<int64_t>(runtime.options.chunk,prompt.size()-start);
    const std::vector<int> ids(prompt.begin()+start,prompt.begin()+start+count);
    std::vector<uint8_t> mask(count);
    for(const auto& image:images)for(int64_t p=std::max(start,image.begin);p<std::min(start+count,image.begin+image.tokens());++p)mask[p-start]=1;
    auto query=engram_hash(engram_hash_spec(runtime.model),runtime.model.token_map(),ids,mask,history);
    lookahead_start=start;lookahead_after=query.after;
    // Immutable snapshot only: history/position are committed when init consumes it.
    engram_lookahead=runtime.engram->submit(std::move(query),generation,true);
}
void Session::Impl::reset_local(int first,int64_t at) {
    for(int l=first;l<40;++l){local[l].position=at;local[l].count=0;}
}
void Session::Impl::image_mask(int count,int64_t start) {
    std::vector<uint8_t> mask(count);
    for(const auto& image:images){
        const int64_t begin=std::max<int64_t>(start,image.begin),end=std::min<int64_t>(start+count,image.begin+image.tokens());
        for(int64_t p=begin;p<end;++p)mask[p-start]=1;
    }
    gpu_check(cudaMemcpyAsync(work.image,mask.data(),count,cudaMemcpyHostToDevice,runtime.stream.value));
}
void Session::Impl::init(const std::vector<int>& ids,int64_t start) {
    check_cancel();if(ids.empty()||ids.size()>static_cast<size_t>(work.max_tokens)||history.position!=start)throw std::runtime_error("DeepSeek input/history position");
    std::vector<uint8_t> mask(ids.size());std::vector<float> values;
    for(int id:ids)if(id<0||id>=129280)throw std::out_of_range("DeepSeek token ID");
    for(const auto& image:images){
        const int64_t begin=std::max<int64_t>(start,image.begin),end=std::min<int64_t>(start+ids.size(),image.begin+image.tokens());
        if(begin>=end)continue;
        if(values.empty())values.resize(ids.size()*5120);
        for(int64_t p=begin;p<end;++p){mask[p-start]=1;std::memcpy(values.data()+(p-start)*5120,image.values.data()+(p-image.begin)*5120,5120*4);}
    }
    // Submit immutable row hashes before embedding/layer0. History commits once here.
    if(engram_lookahead) {
        if(lookahead_start!=start || engram_lookahead->generation!=generation || lookahead_after.position!=start+static_cast<int64_t>(ids.size()))
            throw std::runtime_error("Engram stale lookahead");
        history=lookahead_after;engram_ticket=std::move(engram_lookahead);lookahead_start=-1;
    }else {
        auto query=engram_hash(engram_hash_spec(runtime.model),runtime.model.token_map(),ids,mask,history);
        history=query.after;engram_ticket=runtime.engram->submit(std::move(query),generation,runtime.options.engram_pipeline);
    }
    ++batch_epoch;index_epoch=0;candidate_epoch=0;index_source=-1;
    if(runtime.options.engram_pipeline) {
        if(!engram_pipeline)engram_pipeline=std::make_unique<EngramPipeline>(work,runtime.options.arithmetic,runtime.options.trace,runtime.options.huge_pinned_memory);
        engram_pipeline->start(engram_ticket,batch_epoch,ids.size(),runtime.tensor(1,"engram_wkv"),runtime.tensor(14,"engram_wkv"));
    }
    gpu_check(cudaMemcpyAsync(work.ids,ids.data(),ids.size()*sizeof(int),cudaMemcpyHostToDevice,runtime.stream.value));
    gpu_check(cudaMemcpyAsync(work.image,mask.data(),mask.size(),cudaMemcpyHostToDevice,runtime.stream.value));
    if(!values.empty())gpu_check(cudaMemcpyAsync(work.image_values,values.data(),values.size()*4,cudaMemcpyHostToDevice,runtime.stream.value));
    const auto& embedding=runtime.weights->at("token_embd.weight");
    gpu_check(init_residual_gpu(embedding.data,embedding.type,work.ids,work.image_values,work.image,work.h,work.pre,ids.size(),runtime.stream.value));runtime.round(work.h,ids.size()*20480);
    dump("hc_init",work.h,{static_cast<int64_t>(ids.size()),4,5120});
    dump("hc_pre_init",work.pre,{static_cast<int64_t>(ids.size()),4});
}
void Session::Impl::save_boundary(int count,int64_t start) {
    const auto stream=runtime.stream.value;
    const int skip=std::max(0,count-replay_capacity);start+=skip;count-=skip;
    const int slot=start%replay_capacity,first=std::min(count,replay_capacity-slot);
    gpu_check(cudaMemcpyAsync(boundary_h+int64_t(slot)*20480,work.h+int64_t(skip)*20480,int64_t(first)*20480*4,cudaMemcpyDeviceToDevice,stream));
    gpu_check(cudaMemcpyAsync(boundary_pre+slot*4,work.pre+skip*4,int64_t(first)*4*4,cudaMemcpyDeviceToDevice,stream));
    if(first<count){
        gpu_check(cudaMemcpyAsync(boundary_h,work.h+int64_t(skip+first)*20480,int64_t(count-first)*20480*4,cudaMemcpyDeviceToDevice,stream));
        gpu_check(cudaMemcpyAsync(boundary_pre,work.pre+(skip+first)*4,int64_t(count-first)*4*4,cudaMemcpyDeviceToDevice,stream));
    }
    boundary_end=start+count;boundary_begin=std::max<int64_t>(0,boundary_end-replay_capacity);
}
void Session::Impl::load_boundary(int count,int64_t start) {
    if(start<boundary_begin||start+count>boundary_end)throw std::runtime_error("DeepSeek missing CED boundary rows");
    const auto stream=runtime.stream.value;const int slot=start%replay_capacity,first=std::min(count,replay_capacity-slot);
    gpu_check(cudaMemcpyAsync(work.h,boundary_h+int64_t(slot)*20480,int64_t(first)*20480*4,cudaMemcpyDeviceToDevice,stream));
    gpu_check(cudaMemcpyAsync(work.pre,boundary_pre+slot*4,int64_t(first)*4*4,cudaMemcpyDeviceToDevice,stream));
    if(first<count){
        gpu_check(cudaMemcpyAsync(work.h+int64_t(first)*20480,boundary_h,int64_t(count-first)*20480*4,cudaMemcpyDeviceToDevice,stream));
        gpu_check(cudaMemcpyAsync(work.pre+first*4,boundary_pre,int64_t(count-first)*4*4,cudaMemcpyDeviceToDevice,stream));
    }
    image_mask(count,start);++batch_epoch;index_epoch=0;candidate_epoch=0;index_source=-1;
}
Session::Session(Runtime::Impl& runtime):impl_(std::make_unique<Impl>(runtime)){}
Session::~Session(){if(impl_){std::lock_guard<std::mutex> lock(impl_->runtime.execution_mutex);--impl_->runtime.active_sessions;impl_.reset();}}
std::vector<float> Session::prefill(const std::vector<int>& ids,bool ced,const std::vector<ImageEmbedding>& images) {
    auto& s=*impl_;auto& r=s.runtime;
    prefill::LedgerScope trace(r.options.trace,"ds41.prefill");
    if(ids.empty()||ids.size()>static_cast<size_t>(r.options.context)||s.transaction_active)throw std::invalid_argument("DeepSeek prompt/context/transaction");
    if(s.position!=s.original_prompt || s.tokens.size()>ids.size() || !std::equal(s.tokens.begin(),s.tokens.end(),ids.begin()))
        throw std::runtime_error("DeepSeek prefix cannot reclassify generated tokens as prompt");
    if(s.position && s.ced!=ced)throw std::runtime_error("DeepSeek prefix CED mode mismatch");
    int64_t last_end=0;
    for(const auto& image:images){
        if(image.begin<last_end||image.begin<0||image.values.empty()||image.values.size()%5120||image.tokens()>1024||image.begin+image.tokens()>static_cast<int64_t>(ids.size()))
            throw std::invalid_argument("DeepSeek image span/geometry");
        if(!std::all_of(image.values.begin(),image.values.end(),[](float value){return std::isfinite(value);}))throw std::invalid_argument("DeepSeek nonfinite image embedding");
        last_end=image.begin+image.tokens();
    }
    if(s.position){
        for(const auto& old:s.images){
            const auto it=std::find_if(images.begin(),images.end(),[&](const auto& candidate){return candidate.begin==old.begin;});
            if(it==images.end()||it->values!=old.values)throw std::runtime_error("DeepSeek prefix image content mismatch");
        }
    }
    const int tail_begin=ced?replay_start(ids.size(),images):0;
    s.ced=ced;s.images=images;s.original_prompt=ids.size();s.capture.tokens(ids);
    const auto before=std::chrono::steady_clock::now();
    std::vector<float> result;
    // Include the first read in request timing and start before phase preparation.
    if(r.options.engram_continuous && s.position<static_cast<int64_t>(ids.size()))s.prefetch_engram(ids,s.position);
    {std::lock_guard<std::mutex> lock(r.execution_mutex);
     if(r.decode_raw_cache)r.decode_raw_cache->clear(r.upload.value);
     r.experts->begin_prefill(ids.size()-s.position,r.options.chunk,ced,r.stream.value,s.counters);}
    for(int64_t start=s.position;start<static_cast<int64_t>(ids.size());start+=r.options.chunk){
        const int count=std::min<int64_t>(r.options.chunk,ids.size()-start);
        std::vector<int> chunk(ids.begin()+start,ids.begin()+start+count);
        const std::string label=r.options.trace?"ds41.encoder.chunk."+std::to_string(start):"";
        prefill::LedgerScope chunk_trace(r.options.trace,label.c_str());
        std::lock_guard<std::mutex> lock(r.execution_mutex);s.last_encoder_chunk=ced&&start+count==static_cast<int64_t>(ids.size());s.init(chunk,start);
        s.prefetch_engram(ids,start+count);
        s.layers(count,start,0,ced?20:40,true);
        if(ced){
            prefill::LedgerScope kv_trace(r.options.trace,"ds41.decoder.kv_only");
            s.save_boundary(count,start);
            gpu_check(hc_pre_gpu(s.work.h,s.work.pre,s.work.x,5120,count,r.stream.value));r.round(s.work.x,int64_t(count)*5120);
            gpu_check(rms_gpu(s.work.x,static_cast<const float*>(r.tensor(20,"attn_norm").data),s.work.x,5120,count,r.model.geometry().norm_eps,r.stream.value));r.round(s.work.x,int64_t(count)*5120);
            s.compression(20,count,start,true);
        }else if(start+count==static_cast<int64_t>(ids.size()))result=s.logits(count);
        gpu_check(cudaStreamSynchronize(r.stream.value));s.position=start+count;s.encoder_position=s.position;
        s.tokens.insert(s.tokens.end(),chunk.begin(),chunk.end());
    }
    s.last_encoder_chunk=false;
    {std::lock_guard<std::mutex> lock(r.execution_mutex);r.experts->finish_prefill(r.stream.value,s.counters);}
    if(ced){
        prefill::LedgerScope replay_trace(r.options.trace,"ds41.decoder.tail_replay");
        s.reset_local(20,tail_begin);
        for(int64_t start=tail_begin;start<static_cast<int64_t>(ids.size());start+=r.options.chunk){
            const int count=std::min<int64_t>(r.options.chunk,ids.size()-start);
            std::lock_guard<std::mutex> lock(r.execution_mutex);s.check_cancel();s.load_boundary(count,start);s.layers(count,start,20,40,true);
            if(start+count==static_cast<int64_t>(ids.size()))result=s.logits(count);
            gpu_check(cudaStreamSynchronize(r.stream.value));s.decoder_position=start+count;s.counters.replay_tokens+=count;
        }
    }else s.decoder_position=s.position;
    if(!result.empty())s.last_logits.assign(result.end()-129280,result.end());
    if(s.last_logits.empty()||s.encoder_position!=s.original_prompt||s.decoder_position!=s.original_prompt||s.history.position!=s.position)
        throw std::runtime_error("DeepSeek incomplete prefill state");
    s.counters.prefill_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
    return s.last_logits;
}
std::vector<float> Session::resume_history(const std::vector<int>& ids,int64_t original,bool ced,
                                          const std::vector<ImageEmbedding>& images,const std::string& checkpoint) {
    auto& s=*impl_;auto& r=s.runtime;
    if(original<1||original>static_cast<int64_t>(ids.size())||ids.size()>static_cast<size_t>(r.options.context)||
       s.position>original||std::any_of(ids.begin(),ids.end(),[](int id){return id<0||id>=129280;}))
        throw std::invalid_argument("DeepSeek restored history/original prompt boundary");
    prefill::LedgerScope trace(r.options.trace,"ds41.resume_history");
    const std::vector<int> prompt(ids.begin(),ids.begin()+original);
    auto result=prefill(prompt,ced,images);
    if(!checkpoint.empty())save_prefix(checkpoint);
    // Keep verification-sized reductions and migration boundaries identical to ordinary decode.
    while(s.position<static_cast<int64_t>(ids.size())){
        const int count=std::min<int64_t>({8,r.options.chunk,r.experts->until_adjustment(),static_cast<int64_t>(ids.size())-s.position});
        const std::vector<int> suffix(ids.begin()+s.position,ids.begin()+s.position+count);
        auto rows=forward(suffix);result.assign(rows.end()-129280,rows.end());
        if(s.original_prompt!=original)throw std::runtime_error("DeepSeek restored suffix changed the original prompt boundary");
    }
    return result;
}
std::vector<float> Session::forward(const std::vector<int>& ids) {
    auto& s=*impl_;auto& r=s.runtime;
    prefill::LedgerScope trace(r.options.trace,"ds41.decode_or_verify");
    if(ids.empty()||ids.size()>static_cast<size_t>(std::max(8,r.options.chunk))||s.position+ids.size()>static_cast<size_t>(r.options.context)||
        s.position<s.original_prompt||s.encoder_position!=s.position||s.decoder_position!=s.position)
        throw std::invalid_argument("DeepSeek forward/context/boundary state");
    const auto before=std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(r.execution_mutex);s.init(ids,s.position);s.layers(ids.size(),s.position,0,40);
    auto result=s.logits(ids.size(),true);s.position+=ids.size();s.encoder_position=s.decoder_position=s.position;
    s.tokens.insert(s.tokens.end(),ids.begin(),ids.end());s.last_logits.assign(result.end()-129280,result.end());s.counters.decoded_tokens+=ids.size();
    // Learning describes physical work. Prefix acceptance/rollback must not
    // erase rejected rows or count a six-row verification as six windows.
    r.experts->observe_decode_window(ids.size());
    if(!s.transaction_active)r.experts->adjust_decode(ids.size(),r.stream.value,s.counters);
    s.counters.decode_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();return result;
}
int64_t Session::position()const{return impl_->position;}
int64_t Session::original_prompt_length()const{return impl_->original_prompt;}
ExecutionStats Session::stats()const{return impl_->counters;}
void Session::cancel(){impl_->cancelled.store(true);}
void Session::begin_transaction() {
    auto& s=*impl_;auto& r=s.runtime;std::lock_guard<std::mutex> lock(r.execution_mutex);
    if(s.transaction_active)throw std::runtime_error("DeepSeek nested transaction");
    auto snap=std::make_unique<Impl::Snapshot>();snap->position=s.position;snap->encoder=s.encoder_position;snap->decoder=s.decoder_position;
    snap->routing_heat=r.experts->heat_snapshot();
    snap->history=s.history;snap->token_count=s.tokens.size();snap->last_logits=s.last_logits;
    snap->draft_position=s.draft_capture_position;
    gpu_check(cudaMemcpyAsync(s.snapshot_hidden,s.draft_hidden,128*15360*4,cudaMemcpyDeviceToDevice,r.stream.value));
    const int bytes=r.options.arithmetic==Arithmetic::f32?128*512*4:128*528;int owner=0;
    for(int l=0;l<40;++l){
        snap->local_position[l]=s.local[l].position;snap->local_count[l]=s.local[l].count;
        gpu_check(cudaMemcpyAsync(s.snapshot_local+int64_t(l)*bytes,s.local[l].packed?static_cast<void*>(s.local[l].packed):static_cast<void*>(s.local[l].values),bytes,cudaMemcpyDeviceToDevice,r.stream.value));
        if(r.model.ownership()[l].owns_global()){
            auto& g=s.global[l];snap->global_position[l]=g.position;snap->global_rows[l]=g.rows;snap->partial[l]=g.partial;
            gpu_check(cudaMemcpyAsync(s.snapshot_partial+owner*1024,g.partial_kv,512*4,cudaMemcpyDeviceToDevice,r.stream.value));
            gpu_check(cudaMemcpyAsync(s.snapshot_partial+owner*1024+512,g.partial_gate,512*4,cudaMemcpyDeviceToDevice,r.stream.value));++owner;
        }
    }
    gpu_check(cudaStreamSynchronize(r.stream.value));s.transaction=std::move(snap);s.transaction_active=true;
    s.transaction_routes_deferred=r.options.dspark_prefix_commit;
    for(auto& routes:s.transaction_routes)routes.clear();
}
void Session::Impl::capture_partial(int layer,int count,int64_t start) {
    if(!transaction_active||!transaction_routes_deferred)return;
    const int64_t offset=start-transaction->position;
    // Large public transactions retain full rollback; prefix commit is bounded to eight rows.
    if(offset<0||offset+count>8)return;
    const int owner=partial_owner.at(layer);
    if(owner<0||owner>=4)throw std::logic_error("DeepSeek partial checkpoint owner");
    float* out=verified_partial+(owner*8+offset)*1024;
    gpu_check(cudaMemcpy2DAsync(out,1024*4,work.comp_raw,512*4,512*4,count,cudaMemcpyDeviceToDevice,runtime.stream.value));
    gpu_check(cudaMemcpy2DAsync(out+512,1024*4,work.comp_gate,512*4,512*4,count,cudaMemcpyDeviceToDevice,runtime.stream.value));
}
void Session::Impl::commit_routes(int keep) {
    if(!transaction_routes_deferred)return;
    for(int layer=0;layer<40;++layer){
        if(transaction_routes[layer].size()<static_cast<size_t>(keep*6))throw std::logic_error("DeepSeek transaction route count");
    }
    for(int layer=0;layer<40;++layer)runtime.experts->observe_routes(layer,transaction_routes[layer].data(),keep,2);
}
void Session::commit_verified_prefix(int keep,const std::vector<float>& verified_logits) {
    auto& s=*impl_;auto& r=s.runtime;std::lock_guard<std::mutex> lock(r.execution_mutex);
    if(!s.transaction_active||!s.transaction_routes_deferred)throw std::runtime_error("DeepSeek prefix commit requires a recorded transaction");
    const auto& snap=*s.transaction;const int64_t total=s.position-snap.position,end=snap.position+keep;
    if(keep<1||keep>total||total>8||verified_logits.size()!=static_cast<size_t>(total*129280)||
       s.tokens.size()!=snap.token_count+total)throw std::invalid_argument("DeepSeek verified prefix bounds");
    std::vector<int> ids(s.tokens.begin()+snap.token_count,s.tokens.begin()+snap.token_count+keep);
    std::vector<uint8_t> mask(keep);
    for(const auto& image:s.images)for(int64_t p=std::max(snap.position,image.begin);p<std::min(end,image.begin+image.tokens());++p)mask[p-snap.position]=1;
    const auto history=engram_hash(engram_hash_spec(r.model),r.model.token_map(),ids,mask,snap.history).after;
    const int row_bytes=r.options.arithmetic==Arithmetic::f32?512*4:528;
    for(int layer=0;layer<40;++layer){
        auto* local=s.local[layer].packed?s.local[layer].packed:reinterpret_cast<uint8_t*>(s.local[layer].values);
        // At most eight writes: rejected ring slots cannot alias accepted writes.
        for(int64_t p=end;p<snap.position+total;++p){
            const int slot=p%128;
            gpu_check(cudaMemcpyAsync(local+slot*row_bytes,s.snapshot_local+(int64_t(layer)*128+slot)*row_bytes,
                row_bytes,cudaMemcpyDeviceToDevice,r.stream.value));
        }
        s.local[layer].position=end;s.local[layer].count=std::min(128,snap.local_count[layer]+keep);
        const auto& own=r.model.ownership()[layer];
        if(own.owns_global()){
            auto& g=s.global[layer];g.position=end;g.rows=end/own.compression;g.partial=own.compression==2&&end%2;
            if(g.partial){
                const auto* checkpoint=s.verified_partial+(s.partial_owner[layer]*8+keep-1)*1024;
                gpu_check(cudaMemcpyAsync(g.partial_kv,checkpoint,512*4,cudaMemcpyDeviceToDevice,r.stream.value));
                gpu_check(cudaMemcpyAsync(g.partial_gate,checkpoint+512,512*4,cudaMemcpyDeviceToDevice,r.stream.value));
            }
        }
    }
    for(int64_t p=end;p<snap.position+total;++p){const int slot=p%128;
        gpu_check(cudaMemcpyAsync(s.draft_hidden+int64_t(slot)*15360,s.snapshot_hidden+int64_t(slot)*15360,
            15360*4,cudaMemcpyDeviceToDevice,r.stream.value));
    }
    s.position=s.encoder_position=s.decoder_position=end;s.history=history;s.tokens.resize(snap.token_count+keep);
    if(s.draft_kv&&s.draft_kv->position>end)s.draft_kv->position=-1;
    s.last_logits.assign(verified_logits.begin()+int64_t(keep-1)*129280,verified_logits.begin()+int64_t(keep)*129280);
    s.draft_capture_position=r.options.dspark||s.capture.enabled()?end:snap.draft_position;
    ++s.generation;s.index_epoch=s.candidate_epoch=0;s.index_source=-1;
    if(s.engram_ticket){s.engram_ticket->cancel();s.engram_ticket.reset();}
    if(s.engram_lookahead){s.engram_lookahead->cancel();s.engram_lookahead.reset();}s.lookahead_start=-1;
    gpu_check(cudaStreamSynchronize(r.stream.value));s.commit_routes(keep);
    // Physical forward/H2D counters include rejected verification work.
    r.experts->adjust_decode(keep,r.stream.value,s.counters);
    s.transaction.reset();s.transaction_active=false;s.transaction_routes_deferred=false;
}
void Session::rollback_transaction() {
    auto& s=*impl_;auto& r=s.runtime;std::lock_guard<std::mutex> lock(r.execution_mutex);
    if(!s.transaction_active)throw std::runtime_error("DeepSeek no transaction");
    const auto& snap=*s.transaction;
    const int bytes=r.options.arithmetic==Arithmetic::f32?128*512*4:128*528;int owner=0;
    for(int l=0;l<40;++l){
        s.local[l].position=snap.local_position[l];s.local[l].count=snap.local_count[l];
        gpu_check(cudaMemcpyAsync(s.local[l].packed?static_cast<void*>(s.local[l].packed):static_cast<void*>(s.local[l].values),s.snapshot_local+int64_t(l)*bytes,bytes,cudaMemcpyDeviceToDevice,r.stream.value));
        if(r.model.ownership()[l].owns_global()){
            auto& g=s.global[l];g.position=snap.global_position[l];g.rows=snap.global_rows[l];g.partial=snap.partial[l];
            gpu_check(cudaMemcpyAsync(g.partial_kv,s.snapshot_partial+owner*1024,512*4,cudaMemcpyDeviceToDevice,r.stream.value));
            gpu_check(cudaMemcpyAsync(g.partial_gate,s.snapshot_partial+owner*1024+512,512*4,cudaMemcpyDeviceToDevice,r.stream.value));++owner;
        }
    }
    s.position=snap.position;s.encoder_position=snap.encoder;s.decoder_position=snap.decoder;s.history=snap.history;s.tokens.resize(snap.token_count);s.last_logits=snap.last_logits;
    if(s.draft_kv&&s.draft_kv->position>snap.position)s.draft_kv->position=-1;
    r.experts->restore_heat(snap.routing_heat);
    s.draft_capture_position=snap.draft_position;
    gpu_check(cudaMemcpyAsync(s.draft_hidden,s.snapshot_hidden,128*15360*4,cudaMemcpyDeviceToDevice,r.stream.value));
    ++s.generation;s.index_epoch=s.candidate_epoch=0;s.index_source=-1;if(s.engram_ticket)s.engram_ticket->cancel();
    gpu_check(cudaStreamSynchronize(r.stream.value));s.transaction.reset();s.transaction_active=false;s.transaction_routes_deferred=false;
}
void Session::commit_transaction(){
    auto& s=*impl_;std::lock_guard<std::mutex> lock(s.runtime.execution_mutex);
    if(!s.transaction_active)throw std::runtime_error("DeepSeek no transaction");
    const int count=s.position-s.transaction->position;s.commit_routes(count);
    s.runtime.experts->adjust_decode(count,s.runtime.stream.value,s.counters);
    s.transaction.reset();s.transaction_active=false;s.transaction_routes_deferred=false;
}
void Session::save_prefix(const std::string& path)const {
    auto& s=*impl_;auto& r=s.runtime;std::lock_guard<std::mutex> lock(r.execution_mutex);
    if(s.transaction_active||s.position!=s.original_prompt||s.tokens.size()!=static_cast<size_t>(s.position)||s.last_logits.size()!=129280)
        throw std::runtime_error("DeepSeek prefix requires a completed original prompt with no generated suffix");
    if(std::filesystem::exists(path)||std::filesystem::exists(path+".tmp"))throw std::runtime_error("DeepSeek prefix destination exists");
    std::fstream out(path+".tmp",std::ios::out|std::ios::binary);if(!out)throw std::runtime_error("DeepSeek prefix destination");
    try {
        write_bytes(out,prefix_magic,sizeof(prefix_magic));write_bytes(out,main_sha,sizeof(main_sha));
        write_scalar<int32_t>(out,static_cast<int32_t>(r.options.arithmetic));write_scalar<int32_t>(out,r.options.context);write_scalar<uint8_t>(out,s.ced);
        write_scalar<int64_t>(out,s.position);write_scalar<int64_t>(out,s.history.position);write_bytes(out,s.history.recent.data(),12);
        write_scalar<int64_t>(out,s.boundary_begin);write_scalar<int64_t>(out,s.boundary_end);
        write_scalar<int64_t>(out,s.draft_capture_position);device_file(out,s.draft_hidden,128*15360*4,true,r.stream.value);
        write_bytes(out,s.tokens.data(),s.tokens.size()*4);write_bytes(out,s.last_logits.data(),129280*4);
        write_scalar<uint64_t>(out,s.images.size());
        for(const auto& image:s.images){write_scalar<int64_t>(out,image.begin);write_scalar<uint64_t>(out,image.values.size());write_bytes(out,image.values.data(),image.values.size()*4);}
        const uint64_t local_bytes=r.options.arithmetic==Arithmetic::f32?128*512*4:128*528;
        for(int l=0;l<40;++l){
            write_scalar<int64_t>(out,s.local[l].position);write_scalar<int32_t>(out,s.local[l].count);
            device_file(out,s.local[l].packed?static_cast<void*>(s.local[l].packed):static_cast<void*>(s.local[l].values),local_bytes,true,r.stream.value);
            if(r.model.ownership()[l].owns_global()){
                auto& g=s.global[l];write_scalar<int64_t>(out,g.position);write_scalar<int64_t>(out,g.rows);write_scalar<uint8_t>(out,g.partial);
                device_file(out,g.partial_kv,512*4,true,r.stream.value);device_file(out,g.partial_gate,512*4,true,r.stream.value);
                const int ratio=r.model.ownership()[l].compression;
                (void)ratio;
                if(r.options.arithmetic==Arithmetic::f32){device_file(out,g.values,g.rows*512*4,true,r.stream.value);device_file(out,g.index_values,g.rows*128*4,true,r.stream.value);}
                else{device_file(out,g.packed,g.rows*288,true,r.stream.value);device_file(out,g.index_packed,g.rows*68,true,r.stream.value);}
            }
        }
        if(s.ced){device_file(out,s.boundary_h,Impl::replay_capacity*20480*4,true,r.stream.value);device_file(out,s.boundary_pre,Impl::replay_capacity*4*4,true,r.stream.value);}
        out.close();
        const uint64_t bytes=std::filesystem::file_size(path+".tmp"),checksum=prefix_checksum(path+".tmp",bytes);
        std::ofstream trailer(path+".tmp",std::ios::binary|std::ios::app);write_scalar<uint64_t>(trailer,checksum);trailer.close();
        std::filesystem::rename(path+".tmp",path);
    }catch(...){out.close();std::error_code error;std::filesystem::remove(path+".tmp",error);throw;}
}
void Session::load_prefix(const std::string& path,const std::vector<int>& expected) {
    auto& s=*impl_;auto& r=s.runtime;std::lock_guard<std::mutex> lock(r.execution_mutex);
    if(s.position||s.transaction_active)throw std::runtime_error("DeepSeek prefix restore requires an empty session");
    const uint64_t file_bytes=std::filesystem::file_size(path);
    if(file_bytes<8)throw std::runtime_error("DeepSeek truncated prefix");
    std::ifstream trailer(path,std::ios::binary);trailer.seekg(file_bytes-8);const uint64_t expected_checksum=read_scalar<uint64_t>(trailer);
    if(prefix_checksum(path,file_bytes-8)!=expected_checksum)throw std::runtime_error("DeepSeek prefix checksum mismatch");
    try {
    std::fstream in(path,std::ios::in|std::ios::binary);if(!in)throw std::runtime_error("DeepSeek prefix open");
    char magic[sizeof(prefix_magic)],sha[sizeof(main_sha)];read_bytes(in,magic,sizeof(magic));read_bytes(in,sha,sizeof(sha));
    if(std::memcmp(magic,prefix_magic,sizeof(magic))||std::memcmp(sha,main_sha,sizeof(sha))||read_scalar<int32_t>(in)!=static_cast<int32_t>(r.options.arithmetic)||
        read_scalar<int32_t>(in)!=r.options.context)throw std::runtime_error("DeepSeek prefix version/model/arithmetic/context mismatch");
    const uint8_t ced=read_scalar<uint8_t>(in);const auto position=read_scalar<int64_t>(in);
    if(ced>1||position<1||position>r.options.context||expected.size()!=static_cast<size_t>(position))throw std::runtime_error("DeepSeek prefix length/mode mismatch");
    EngramHistory history;history.position=read_scalar<int64_t>(in);read_bytes(in,history.recent.data(),12);
    if(history.position!=position)throw std::runtime_error("DeepSeek prefix Engram history mismatch");
    for(int32_t id:history.recent)if(id < -1 || id>=99092)throw std::runtime_error("DeepSeek prefix compressed ID");
    const auto begin=read_scalar<int64_t>(in),end=read_scalar<int64_t>(in);
    const auto draft_position=read_scalar<int64_t>(in);
    if(draft_position!=0&&draft_position!=position)throw std::runtime_error("DeepSeek prefix draft capture watermark");
    device_file(in,s.draft_hidden,128*15360*4,false,r.stream.value);
    if(ced&&(end!=position||begin!=std::max<int64_t>(0,end-Impl::replay_capacity)))throw std::runtime_error("DeepSeek prefix CED boundary");
    std::vector<int> tokens(position);read_bytes(in,tokens.data(),tokens.size()*4);if(tokens!=expected)throw std::runtime_error("DeepSeek prefix token identity");
    std::vector<float> last(129280);read_bytes(in,last.data(),last.size()*4);
    const auto image_count=read_scalar<uint64_t>(in);if(image_count>static_cast<uint64_t>(position))throw std::runtime_error("DeepSeek prefix image count");
    std::vector<ImageEmbedding> images;int64_t image_end=0;
    for(uint64_t i=0;i<image_count;++i){ImageEmbedding image;image.begin=read_scalar<int64_t>(in);const auto size=read_scalar<uint64_t>(in);
        if(!size||size%5120||size>1024ull*5120||image.begin<image_end||image.begin+int64_t(size/5120)>position)throw std::runtime_error("DeepSeek prefix image span");
        image.values.resize(size);read_bytes(in,image.values.data(),size*4);image_end=image.begin+image.tokens();images.push_back(std::move(image));}
    const uint64_t local_bytes=r.options.arithmetic==Arithmetic::f32?128*512*4:128*528;
    for(int l=0;l<40;++l){
        const auto watermark=read_scalar<int64_t>(in);const int count=read_scalar<int32_t>(in);
        const int64_t local_start=ced&&l>=20?replay_start(position,images):0;
        if(watermark!=position||count!=std::min<int64_t>(128,position-local_start))throw std::runtime_error("DeepSeek prefix missing layer SWA");
        device_file(in,s.local[l].packed?static_cast<void*>(s.local[l].packed):static_cast<void*>(s.local[l].values),local_bytes,false,r.stream.value);
        s.local[l].position=watermark;s.local[l].count=count;
        if(r.model.ownership()[l].owns_global()){
            auto& g=s.global[l];g.position=read_scalar<int64_t>(in);g.rows=read_scalar<int64_t>(in);const uint8_t partial=read_scalar<uint8_t>(in);
            const int ratio=r.model.ownership()[l].compression;
            if(g.position!=position||g.rows!=position/ratio||partial>1||partial!=bool(position%ratio))throw std::runtime_error("DeepSeek prefix global/partial watermark");
            g.partial=partial;
            device_file(in,g.partial_kv,512*4,false,r.stream.value);device_file(in,g.partial_gate,512*4,false,r.stream.value);
            if(r.options.arithmetic==Arithmetic::f32){device_file(in,g.values,g.rows*512*4,false,r.stream.value);device_file(in,g.index_values,g.rows*128*4,false,r.stream.value);}
            else{device_file(in,g.packed,g.rows*288,false,r.stream.value);device_file(in,g.index_packed,g.rows*68,false,r.stream.value);}
        }
    }
    if(ced){device_file(in,s.boundary_h,Impl::replay_capacity*20480*4,false,r.stream.value);device_file(in,s.boundary_pre,Impl::replay_capacity*4*4,false,r.stream.value);}
    if(read_scalar<uint64_t>(in)!=expected_checksum||in.peek()!=EOF)throw std::runtime_error("DeepSeek prefix trailing data");
    s.position=s.original_prompt=s.encoder_position=s.decoder_position=position;s.ced=ced;s.history=history;s.tokens=std::move(tokens);s.images=std::move(images);s.last_logits=std::move(last);
    s.boundary_begin=begin;s.boundary_end=end;++s.generation;s.index_epoch=s.candidate_epoch=0;
    s.draft_capture_position=draft_position;
    }catch(...){
        for(auto& cache:s.local){cache.position=-1;cache.count=0;}
        for(auto& cache:s.global){cache.position=cache.rows=0;cache.partial=false;}
        s.position=s.original_prompt=s.encoder_position=s.decoder_position=s.boundary_begin=s.boundary_end=0;
        s.history={};s.tokens.clear();s.images.clear();s.last_logits.clear();
        s.draft_capture_position=0;
        cudaMemsetAsync(s.state->base(),0,s.state->used(),r.stream.value);cudaStreamSynchronize(r.stream.value);
        throw;
    }
}
std::vector<int> Session::generate(const std::vector<float>& initial,int maximum,const std::vector<int>& eos,
                                  const kernels::SamplerParams& sampling,bool speculative,
                                  const std::function<void(int)>& callback) {
    auto& s=*impl_;
    if(initial.size()!=129280||maximum<0||sampling.counter>=(1ull<<60)||
        (speculative&&!s.runtime.options.dspark))throw std::invalid_argument("DeepSeek generation contract");
    if(speculative&&s.runtime.options.dspark_adaptive_verification&&!(sampling.greedy||sampling.temperature==0))
        throw std::invalid_argument("DSpark adaptive verification currently requires greedy sampling");
    std::vector<int> output;std::vector<float> current=initial;int pending=-1;
    auto stopped=[&](int id){return std::find(eos.begin(),eos.end(),id)!=eos.end();};
    auto emit=[&](int id){output.push_back(id);if(callback)callback(id);};
    while(static_cast<int>(output.size())<maximum&&s.position<s.runtime.options.context){
        s.check_cancel();
        const auto p0=distribution(current.data(),129280,sampling,s.tokens);
        const int first=pending>=0?pending:sample(p0,draw_uniform(sampling.seed,sampling.counter+s.position));pending=-1;
        const int remaining=maximum-output.size();
        if(stopped(first)){emit(first);break;}
        if(!speculative||remaining<2||s.runtime.options.context-s.position<2||s.runtime.experts->until_adjustment()<2){
            emit(first);if(static_cast<int>(output.size())==maximum)break;current=decode(first);continue;
        }
        auto proposal=propose(first,sampling);
        if(proposal.tokens.empty()){
            emit(first);if(static_cast<int>(output.size())==maximum)break;current=decode(first);continue;
        }
        int count=std::min<int>({int(proposal.tokens.size()),remaining-1,int(s.runtime.options.context-s.position-1),s.runtime.experts->until_adjustment()-1});
        for(int i=0;i<count;++i)if(stopped(proposal.tokens[i])){count=i+1;break;}
        if(s.runtime.options.dspark_adaptive_verification){
            const auto& o=s.runtime.options;
            const int selected=adaptive_verify_count(proposal.confidence,count,o.dspark_draft_cost_ms,o.dspark_verify_cost_ms);
            s.counters.adaptive_available_tokens+=count;s.counters.adaptive_trimmed_tokens+=count-selected;
            ++s.counters.adaptive_length_histogram[selected];count=selected;
        }
        std::vector<int> inputs{first};inputs.insert(inputs.end(),proposal.tokens.begin(),proposal.tokens.begin()+count);
        const int64_t base_position=s.position;const auto base_history=s.tokens;
        begin_transaction();bool active=true;
        try{
            const auto before=std::chrono::steady_clock::now();auto verified=forward(inputs);
            ++s.counters.verify_calls;s.counters.verify_rows+=inputs.size();
            s.counters.verify_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();s.counters.proposed_tokens+=count;
            std::vector<int> history=base_history;history.push_back(first);
            int accepted=0,replacement=-1;
            for(int i=0;i<count;++i){
                const auto target=distribution(verified.data()+int64_t(i)*129280,129280,sampling,history);
                if(sampling.greedy||sampling.temperature==0){
                    if(target.ids[0]!=proposal.tokens[i]){replacement=target.ids[0];break;}
                }else{
                    const auto& q=proposal.distributions[i];
                    const core::SpecP p{target.ids.data(),target.probabilities.data(),static_cast<int>(target.ids.size())};
                    const core::SpecQ draft{q.ids.data(),q.proposal_probabilities.data(),static_cast<int>(q.ids.size())};
                    const uint64_t counter=sampling.counter+base_position+i+1;
                    const auto decision=core::spec_verify_row(p,proposal.tokens[i],draft,
                        draw_uniform(sampling.seed,counter|core::kSpecCtrAccept),draw_uniform(sampling.seed,counter|core::kSpecCtrResid));
                    if(!decision.accepted){replacement=decision.token;break;}
                }
                ++accepted;history.push_back(proposal.tokens[i]);
            }
            if(s.runtime.options.dspark_prefix_commit){
                commit_verified_prefix(accepted+1,verified);active=false;
                current.assign(verified.begin()+int64_t(accepted)*129280,verified.begin()+int64_t(accepted+1)*129280);
                if(accepted<count)pending=replacement;
            }else if(accepted<count){
                rollback_transaction();active=false;
                inputs.resize(accepted+1);const auto before_replay=std::chrono::steady_clock::now();auto replayed=forward(inputs);
                ++s.counters.rejection_replay_calls;s.counters.rejection_replay_rows+=inputs.size();
                s.counters.verify_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before_replay).count();
                current.assign(replayed.end()-129280,replayed.end());pending=replacement;
            }else{commit_transaction();active=false;current.assign(verified.end()-129280,verified.end());}
            s.counters.accepted_tokens+=accepted;
            emit(first);for(int i=0;i<accepted;++i)emit(proposal.tokens[i]);
            if(!output.empty()&&stopped(output.back()))break;
        }catch(...){if(active)rollback_transaction();throw;}
    }
    return output;
}
} // namespace strata::models::deepseek41
