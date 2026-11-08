#include "runtime_internal.hpp"
#include "expert_pipeline.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include <algorithm>
#include <chrono>

namespace strata::models::deepseek41 {
void pipeline_peer_copy(void* to,int to_device,const void* from,int from_device,uint64_t bytes,cudaStream_t stream,Pinned& staging,ExecutionStats& stats) {
    if(!bytes)return;
    if(pipeline_p2p_enabled()){
        DeviceScope destination(to_device);
        gpu_check(cudaMemcpyPeerAsync(to,to_device,from,from_device,bytes,stream));
        stats.peer_copy_bytes+=bytes;
        return;
    }
    constexpr uint64_t capacity=16ull<<20;
    for(uint64_t offset=0;offset<bytes;offset+=capacity){
        const uint64_t count=std::min(capacity,bytes-offset);
        {DeviceScope source(from_device);gpu_check(cudaMemcpy(staging.data,static_cast<const uint8_t*>(from)+offset,count,cudaMemcpyDeviceToHost));}
        DeviceScope destination(to_device);
        gpu_check(cudaMemcpyAsync(static_cast<uint8_t*>(to)+offset,staging.data,count,cudaMemcpyHostToDevice,stream));
        gpu_check(cudaStreamSynchronize(stream)); // The next D2H may reuse this host range.
    }
    stats.peer_copy_bytes+=bytes;
}
void add_execution_stats(ExecutionStats& a,const ExecutionStats& b) {
#define ADD(field) a.field+=b.field
    ADD(expert_gpu);ADD(expert_cpu);ADD(h2d_expert_bytes);ADD(encoder_token_layers);ADD(decoder_token_layers);
    ADD(cpu_input_d2h_bytes);ADD(cpu_output_h2d_bytes);
    ADD(kv_only_tokens);ADD(replay_tokens);ADD(decoded_tokens);ADD(prefill_ms);ADD(decode_ms);ADD(engram_wait_ms);
    ADD(draft_ms);ADD(verify_ms);ADD(proposed_tokens);ADD(accepted_tokens);ADD(verify_calls);ADD(verify_rows);
    ADD(rejection_replay_calls);ADD(rejection_replay_rows);ADD(confidence_rounds);ADD(adaptive_available_tokens);ADD(adaptive_trimmed_tokens);
    ADD(draft_gpu_rounds);ADD(draft_context_rows);ADD(draft_context_projection_rows);ADD(draft_fused_moe_calls);ADD(draft_batch_projection_calls);
    ADD(phase_borrow_bytes);ADD(phase_backup_bytes);ADD(phase_encoder_h2d_bytes);ADD(phase_restore_h2d_bytes);
    ADD(phase_replay_reuse_bytes);ADD(phase_dynamic_h2d_bytes);ADD(phase_dynamic_d2h_bytes);ADD(phase_migrations);ADD(phase_ready_waits);
    ADD(phase_prepare_ms);ADD(phase_host_wait_ms);ADD(phase_restore_submit_ms);ADD(phase_dynamic_ms);
    ADD(demand_expert_tokens);ADD(demand_expert_bytes);ADD(cold_expert_groups);ADD(demand_expert_groups);
    ADD(raw_cache_hits);ADD(raw_cache_d2d_bytes);ADD(direct_expert_bytes);ADD(prefetch_expert_bytes);ADD(prefetch_unused_bytes);ADD(prefetch_pruned_bytes);
    ADD(shared_graph_captures);ADD(shared_graph_launches);ADD(peer_copy_bytes);ADD(pipeline_encoder_chunks);
#undef ADD
    for(size_t i=0;i<a.adaptive_length_histogram.size();++i)a.adaptive_length_histogram[i]+=b.adaptive_length_histogram[i];
    for(size_t i=0;i<a.engram_layer_wait_ms.size();++i)a.engram_layer_wait_ms[i]+=b.engram_layer_wait_ms[i];
}
void Session::Impl::copy_global(Impl& source,int layer,bool full) {
    DeviceScope selected(runtime.device);
    {DeviceScope from(source.runtime.device);gpu_check(cudaStreamSynchronize(source.runtime.stream.value));}
    auto& dst=global.at(layer);const auto& src=source.global.at(layer);
    const auto stream=runtime.stream.value;
    auto copy=[&](void* to,const void* from,uint64_t bytes){pipeline_peer_copy(to,runtime.device,from,source.runtime.device,bytes,stream,*runtime.peer_staging,counters);};
    // Only append new immutable rows. Rollback reduces the watermarks; its next
    // append overwrites rejected rows. A loaded prefix explicitly copies all rows.
    const int64_t begin=full?0:std::min(dst.rows,src.rows),rows=src.rows-begin;
    if(runtime.options.arithmetic==Arithmetic::f32){
        copy(dst.values+begin*512,src.values+begin*512,rows*512*4);
        copy(dst.index_values+begin*128,src.index_values+begin*128,rows*128*4);
    }else{
        copy(dst.packed+begin*288,src.packed+begin*288,rows*288);
        copy(dst.index_packed+begin*68,src.index_packed+begin*68,rows*68);
    }
    copy(dst.partial_kv,src.partial_kv,512*4);copy(dst.partial_gate,src.partial_gate,512*4);
    dst.position=src.position;dst.rows=src.rows;dst.partial=src.partial;
}
void Session::Impl::handoff(Impl& dst,int count,bool index,bool candidates) {
    DeviceScope selected(runtime.device);gpu_check(cudaStreamSynchronize(runtime.stream.value));
    DeviceScope destination(dst.runtime.device);
    prefill::LedgerScope trace(runtime.options.trace,"ds41.pipeline.handoff");
    const auto stream=dst.runtime.stream.value;
    auto copy=[&](void* to,const void* from,uint64_t bytes){pipeline_peer_copy(to,dst.runtime.device,from,runtime.device,bytes,stream,*dst.runtime.peer_staging,dst.counters);};
    copy(dst.work.h,work.h,uint64_t(count)*20480*4);copy(dst.work.pre,work.pre,uint64_t(count)*4*4);
    copy(dst.work.image,work.image,count);
    if(index)copy(dst.work.global_slots,work.global_slots,uint64_t(count)*512*8);
    if(candidates)copy(dst.work.candidate_blocks,work.candidate_blocks,uint64_t(count)*2048*4);
    // The source workspace may be reused by the next chunk only after this copy.
    gpu_check(cudaStreamSynchronize(stream));
    dst.generation=generation;dst.batch_epoch=batch_epoch;
    dst.index_epoch=index?index_epoch:0;dst.index_source=index?index_source:-1;
    dst.candidate_epoch=candidates?candidate_epoch:0;
    dst.ced=ced;dst.original_prompt=original_prompt;dst.last_encoder_chunk=last_encoder_chunk;
    dst.engram_ticket=engram_ticket;
}
void Session::Impl::layers(int count,int64_t start,int first,int last,bool prefill,const BlockBenchmarkHooks* hooks) {
    if(!peer){local_layers(count,start,first,last,prefill,hooks);return;}
    if(hooks)throw std::invalid_argument("single-device block benchmark cannot instrument the CED pipeline");
    if((first!=0&&first!=20)||(last!=20&&last!=40))throw std::invalid_argument("CED pipeline layer interval");
    DeviceScope selected(runtime.device);auto& p=*peer;
    if(runtime.options.pipeline_contiguous){
        if(first==0)local_layers(count,start,0,20,prefill,nullptr);
        if(last==40){
            p.copy_global(*this,8);p.copy_global(*this,14);handoff(p,count,false,false);
            DeviceScope other(p.runtime.device);p.local_layers(count,start,20,40,prefill,nullptr);
        }
        collect_peer();return;
    }
    if(first==0){
        local_layers(count,start,0,10,prefill,nullptr);
        p.copy_global(*this,8);handoff(p,count,true,false);
        {DeviceScope other(p.runtime.device);p.start_engram(count);p.local_layers(count,start,10,20,prefill,nullptr);}
        if(last==40)p.handoff(*this,count,false,false);
    }
    if(last==40){
        local_layers(count,start,20,32,prefill,nullptr);
        p.copy_global(*this,20);handoff(p,count,false,true);
        {DeviceScope other(p.runtime.device);p.local_layers(count,start,32,40,prefill,nullptr);}
    }
    collect_peer();
}
void Session::Impl::collect_peer(bool kv_only) {
    if(!peer)return;
    auto& p=*peer;DeviceScope selected(runtime.device);
    {DeviceScope other(p.runtime.device);gpu_check(cudaStreamSynchronize(p.runtime.stream.value));}
    const auto stream=runtime.stream.value;
    auto copy=[&](void* to,const void* from,uint64_t bytes){pipeline_peer_copy(to,runtime.device,from,p.runtime.device,bytes,stream,*runtime.peer_staging,counters);};
    const int bytes=runtime.options.arithmetic==Arithmetic::f32?128*512*4:128*528;
    for(int layer=0;layer<40;++layer)if(!runtime.options.owns_layer(layer)){
        auto& dst=local[layer];const auto& src=p.local[layer];
        copy(dst.packed?static_cast<void*>(dst.packed):dst.values,src.packed?static_cast<const void*>(src.packed):src.values,bytes);
        dst.position=src.position;dst.count=src.count;
        if(transaction_active)transaction_routes[layer]=p.transaction_routes[layer];
    }
    for(int layer:{8,14,20})if(!runtime.options.owns_layer(layer))copy_global(p,layer);
    if(kv_only){
        if(runtime.options.owns_layer(20))copy_global(p,20);
        copy(boundary_h,p.boundary_h,replay_capacity*20480ull*4);copy(boundary_pre,p.boundary_pre,replay_capacity*4*4);
        boundary_begin=p.boundary_begin;boundary_end=p.boundary_end;
    }
    copy(draft_hidden,p.draft_hidden,128*15360*4);draft_capture_position=p.draft_capture_position;
    if(transaction_active)for(int layer:{8,14,20})if(!runtime.options.owns_layer(layer)){
        const int offset=partial_owner[layer]*8*1024;
        copy(verified_partial+offset,p.verified_partial+offset,8*1024*4);
    }
    gpu_check(cudaStreamSynchronize(stream));add_execution_stats(counters,p.counters);p.counters={};
}
void Session::Impl::sync_peer_metadata() {
    if(!peer)return;auto& p=*peer;
    p.position=position;p.encoder_position=encoder_position;p.decoder_position=decoder_position;
    p.original_prompt=original_prompt;p.ced=ced;p.history=history;p.tokens=tokens;p.images=images;
    p.last_logits=last_logits;p.generation=generation;
}
void Session::Impl::restore_peer(bool full) {
    if(!peer)return;
    auto& p=*peer;DeviceScope selected(runtime.device);gpu_check(cudaStreamSynchronize(runtime.stream.value));
    DeviceScope other(p.runtime.device);const auto stream=p.runtime.stream.value;
    auto copy=[&](void* to,const void* from,uint64_t bytes){pipeline_peer_copy(to,p.runtime.device,from,runtime.device,bytes,stream,*p.runtime.peer_staging,p.counters);};
    const int bytes=runtime.options.arithmetic==Arithmetic::f32?128*512*4:128*528;
    for(int layer=0;layer<40;++layer)if(!runtime.options.owns_layer(layer)){
        auto& dst=p.local[layer];const auto& src=local[layer];
        copy(dst.packed?static_cast<void*>(dst.packed):dst.values,src.packed?static_cast<const void*>(src.packed):src.values,bytes);
        dst.position=src.position;dst.count=src.count;
    }
    for(int layer:{8,14,20})p.copy_global(*this,layer,full);
    copy(p.draft_hidden,draft_hidden,128*15360*4);p.draft_capture_position=draft_capture_position;
    if(full){copy(p.boundary_h,boundary_h,replay_capacity*20480ull*4);copy(p.boundary_pre,boundary_pre,replay_capacity*4*4);}
    p.boundary_begin=boundary_begin;p.boundary_end=boundary_end;
    if(p.draft_kv&&p.draft_kv->position>position)p.draft_kv->position=-1;
    p.index_epoch=p.candidate_epoch=0;p.index_source=-1;
    p.clear_transaction();
    gpu_check(cudaStreamSynchronize(stream));sync_peer_metadata();
}
std::vector<float> Session::Impl::pipeline_prefill(const std::vector<int>& ids,int tail_begin) {
    DeviceScope selected(runtime.device);auto& p=*peer;
    std::lock_guard<std::mutex> lock(runtime.execution_mutex);
    const auto before=std::chrono::steady_clock::now();std::future<void> pending;
    sync_peer_metadata();
    if(runtime.options.engram_continuous&&position<static_cast<int64_t>(ids.size()))prefetch_engram(ids,position);
    for(auto* s:{this,&p}){
        DeviceScope other(s->runtime.device);auto& r=s->runtime;
        if(r.decode_raw_cache)r.decode_raw_cache->clear(r.upload.value);
        r.experts->begin_prefill(ids.size()-position,r.options.chunk,true,r.stream.value,s->counters);
    }
    try{
        for(int64_t start=position;start<static_cast<int64_t>(ids.size());start+=runtime.options.chunk){
            check_cancel();const int count=std::min<int64_t>(runtime.options.chunk,ids.size()-start);
            const std::vector<int> chunk(ids.begin()+start,ids.begin()+start+count);
            last_encoder_chunk=start+count==static_cast<int64_t>(ids.size());
            {prefill::LedgerScope trace(runtime.options.trace,"ds41.pipeline.encoder0");
             init(chunk,start);prefetch_engram(ids,start+count);
             local_layers(count,start,0,runtime.options.pipeline_contiguous?20:10,true,nullptr);}
            // One destination workspace is enough: overlap the next first stage
            // with the current second stage, then drain before reusing GPU1.
            if(pending.valid())pending.get();
            p.copy_global(*this,8);handoff(p,count,true,false);
            if(runtime.options.pipeline_contiguous)p.copy_global(*this,14);
            pending=std::async(std::launch::async,[&p,count,start]{
                DeviceScope selected(p.runtime.device);auto& r=p.runtime;
                prefill::LedgerScope trace(r.options.trace,"ds41.pipeline.encoder1");
                if(!r.options.pipeline_contiguous){p.start_engram(count);p.local_layers(count,start,10,20,true,nullptr);}
                p.save_boundary(count,start);
                {prefill::LedgerScope kv_trace(r.options.trace,"ds41.decoder.kv_only");
                 gpu_check((r.options.arithmetic==Arithmetic::production?hc_pre_bf16_gpu:hc_pre_gpu)(p.work.h,p.work.pre,p.work.x,5120,count,r.stream.value));
                 gpu_check((r.options.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(p.work.x,static_cast<const float*>(r.tensor(20,"attn_norm").data),p.work.x,5120,count,r.model.geometry().norm_eps,r.stream.value));
                 p.compression(20,count,start,true);}
                gpu_check(cudaStreamSynchronize(r.stream.value));
            });
            if(!runtime.options.pipeline_overlap)pending.get();
            position=encoder_position=start+count;tokens.insert(tokens.end(),chunk.begin(),chunk.end());++counters.pipeline_encoder_chunks;
        }
        if(pending.valid())pending.get();
    }catch(...){
        cancelled.store(true);p.cancelled.store(true);
        if(pending.valid())try{pending.get();}catch(...){}
        throw;
    }
    last_encoder_chunk=p.last_encoder_chunk=false;collect_peer(true);
    for(auto* s:{this,&p}){DeviceScope other(s->runtime.device);s->runtime.experts->finish_prefill(s->runtime.stream.value,s->counters);}
    reset_local(20,tail_begin);
    for(int64_t start=tail_begin;start<static_cast<int64_t>(ids.size());start+=runtime.options.chunk){
        prefill::LedgerScope trace(runtime.options.trace,"ds41.decoder.tail_replay");
        const int count=std::min<int64_t>(runtime.options.chunk,ids.size()-start);
        check_cancel();load_boundary(count,start);layers(count,start,20,40,true);
        if(start+count==static_cast<int64_t>(ids.size())){
            const auto result=logits(count);last_logits.assign(result.end()-129280,result.end());
        }
        decoder_position=start+count;counters.replay_tokens+=count;
    }
    sync_peer_metadata();
    if(last_logits.size()!=129280||encoder_position!=original_prompt||decoder_position!=original_prompt||history.position!=position)
        throw std::runtime_error("DeepSeek incomplete pipeline prefill state");
    counters.prefill_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
    return last_logits;
}
} // namespace strata::models::deepseek41
