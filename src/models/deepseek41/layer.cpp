#include "runtime_internal.hpp"
#include "expert_pipeline.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>

namespace strata::models::deepseek41 {
namespace {
CacheLayout contiguous(uint8_t* data,int64_t rows,int row_bytes,int page_size) {
    return {data,((rows+page_size-1)/page_size)*page_size*row_bytes,int64_t(page_size)*row_bytes,row_bytes,page_size};
}
void copy_window(uint8_t* destination,const uint8_t* source,int begin,int rows,int row_bytes,cudaStream_t stream) {
    if (!rows) return;
    const int first=std::min(rows,128-begin);
    gpu_check(cudaMemcpyAsync(destination,source+int64_t(begin)*row_bytes,int64_t(first)*row_bytes,cudaMemcpyDeviceToDevice,stream));
    if(first<rows)gpu_check(cudaMemcpyAsync(destination+int64_t(first)*row_bytes,source,int64_t(rows-first)*row_bytes,cudaMemcpyDeviceToDevice,stream));
}
void store_window(uint8_t* destination,const uint8_t* source,int begin,int rows,int row_bytes,cudaStream_t stream) {
    const int first=std::min(rows,128-begin);
    gpu_check(cudaMemcpyAsync(destination+int64_t(begin)*row_bytes,source,int64_t(first)*row_bytes,cudaMemcpyDeviceToDevice,stream));
    if(first<rows)gpu_check(cudaMemcpyAsync(destination,source+int64_t(first)*row_bytes,int64_t(rows-first)*row_bytes,cudaMemcpyDeviceToDevice,stream));
}
}
void Session::Impl::dump(const std::string& name,const float* data,const std::vector<int64_t>& shape) {
    capture.put(name,data,shape,runtime.stream.value);
}
void Session::Impl::eng(int layer,int count) {
    prefill::LedgerScope trace(runtime.options.trace,"ds41.engram");
    const auto before=std::chrono::steady_clock::now();
    if(!engram_ticket || engram_ticket->generation!=generation)throw std::runtime_error("DeepSeek Engram generation mismatch");
    const int which=layer==1?0:1;
    if(engram_pipeline) {
        engram_pipeline->wait(which,generation,batch_epoch,runtime.stream.value);
        const double wait=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
        counters.engram_wait_ms+=wait;counters.engram_layer_wait_ms[which]+=wait;
    }
    else {
    const auto& data=engram_ticket->wait();
    const double wait=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
    counters.engram_wait_ms+=wait;counters.engram_layer_wait_ms[which]+=wait;
    const uint64_t row=runtime.model.engram_row_bytes();
    if(data.size()!=uint64_t(count)*48*row)throw std::runtime_error("Engram packed batch size");
    gpu_check(cudaMemcpy2DAsync(work.engram_packed,24*row,data.data()+which*24*row,48*row,24*row,count,cudaMemcpyHostToDevice,runtime.stream.value));
    if(runtime.options.arithmetic==Arithmetic::production)
        gpu_check(dequant_round_bf16_gpu(work.engram_packed,runtime.model.engram_type(),work.engram_emb,int64_t(count)*6144,runtime.stream.value));
    else gpu_check(dequant_gpu(work.engram_packed,runtime.model.engram_type(),work.engram_emb,int64_t(count)*6144,runtime.stream.value));
    dump("engram_embd-"+std::to_string(layer),work.engram_emb,{count,6144});
    runtime.matmul(runtime.tensor(layer,"engram_wkv"),work.engram_emb,work.engram_kv,count,work);
    }
    if(engram_pipeline)dump("engram_embd-"+std::to_string(layer),work.engram_emb,{count,6144});
    dump("engram_kv-"+std::to_string(layer),work.engram_kv,{count,25600});
    const bool folded=runtime.model.ple_engram();
    gpu_check((runtime.options.arithmetic==Arithmetic::production?engram_gate_bf16_gpu:engram_gate_gpu)(work.h,work.engram_kv,static_cast<const float*>(runtime.tensor(layer,folded?"engram_gate":"engram_q").data),
        folded?nullptr:static_cast<const float*>(runtime.tensor(layer,"engram_k").data),work.image,count,5120,runtime.stream.value));
    if(engram_pipeline)engram_pipeline->consume(which,batch_epoch,runtime.stream.value);
    dump("engram_out-"+std::to_string(layer),work.h,{count,4,5120});
}
void Session::Impl::compression(int layer,int count,int64_t start,bool only) {
    const auto& ownership=runtime.model.ownership()[layer];
    if(!ownership.owns_global())throw std::runtime_error("DeepSeek non-owner compressed write");
    auto& cache=global[layer];
    const int ratio=ownership.compression;
    if(!only && start+count<=cache.position)return; // CED layer20 was prepared over the original prompt.
    if(cache.position!=start || cache.rows!=start/ratio || (ratio==2 && cache.partial!=bool(start%2)))
        throw std::runtime_error("DeepSeek compressor watermark/partial mismatch");
    const auto stream=runtime.stream.value;
    runtime.matmul(runtime.tensor(layer,"attn_compressor_kv"),work.x,work.comp_raw,count,work,0,0,0,0,ratio==1);
    const int groups=(start+count)/ratio-start/ratio;
    if(ratio==2) {
        runtime.matmul(runtime.tensor(layer,"attn_compressor_gate"),work.x,work.comp_gate,count,work,0,0,0,0,false);
        dump("comp_state_score-"+std::to_string(layer),work.comp_gate,{count,512});
        capture_partial(layer,count,start);
        if(groups)gpu_check(pool_groups_gpu(work.comp_raw,work.comp_gate,cache.partial_kv,cache.partial_gate,work.latent,groups,start%2,stream));
        if((start+count)%2) {
            gpu_check(cudaMemcpyAsync(cache.partial_kv,work.comp_raw+int64_t(count-1)*512,512*4,cudaMemcpyDeviceToDevice,stream));
            gpu_check(cudaMemcpyAsync(cache.partial_gate,work.comp_gate+int64_t(count-1)*512,512*4,cudaMemcpyDeviceToDevice,stream));
        }
        cache.partial=(start+count)%2;
    } else gpu_check(cudaMemcpyAsync(work.latent,work.comp_raw,int64_t(count)*512*4,cudaMemcpyDeviceToDevice,stream));
    if(groups) {
        runtime.round(work.latent,int64_t(groups)*512);
        gpu_check((runtime.options.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(work.latent,static_cast<const float*>(runtime.tensor(layer,"attn_compressor_norm").data),work.latent,512,groups,runtime.model.geometry().norm_eps,stream));
        // Index K consumes the unrotated latent. Each row has the first absolute position of its group.
        runtime.matmul(runtime.tensor(layer,"indexer.attn_k"),work.latent,work.index_k,groups,work);
        gpu_check((runtime.options.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(work.index_k,static_cast<const float*>(runtime.tensor(layer,"indexer.k_norm").data),work.index_k,128,groups,runtime.model.geometry().norm_eps,stream));
        gpu_check(rotate_strided_gpu(work.index_k,1,128,groups,(start/ratio)*ratio,ratio,runtime.model.geometry(),true,false,stream));runtime.round(work.index_k,int64_t(groups)*128);
        if(runtime.options.arithmetic==Arithmetic::f32)
            gpu_check(cudaMemcpyAsync(cache.index_values+cache.rows*128,work.index_k,int64_t(groups)*128*4,cudaMemcpyDeviceToDevice,stream));
        else gpu_check(pack_index_gpu(work.index_k,cache.index_packed+cache.rows*68,groups,stream));
        gpu_check(rotate_strided_gpu(work.latent,1,512,groups,(start/ratio)*ratio,ratio,runtime.model.geometry(),true,false,stream));runtime.round(work.latent,int64_t(groups)*512);
        if(runtime.options.arithmetic==Arithmetic::f32)
            gpu_check(cudaMemcpyAsync(cache.values+cache.rows*512,work.latent,int64_t(groups)*512*4,cudaMemcpyDeviceToDevice,stream));
        else {
            std::vector<int64_t> slots(groups);std::iota(slots.begin(),slots.end(),cache.rows);
            gpu_check(cudaMemcpyAsync(work.write_slots,slots.data(),slots.size()*8,cudaMemcpyHostToDevice,stream));
            gpu_check(float_to_bf16_gpu(work.latent,work.kv_bf,int64_t(groups)*512,stream));
            gpu_check(pack_global_gpu(work.kv_bf,work.write_slots,groups,512,contiguous(cache.packed,runtime.options.context/ratio,288,64),stream));
        }
    }
    cache.position=start+count;cache.rows+=groups;
    if(cache.rows!=cache.position/ratio)throw std::runtime_error("DeepSeek compressed row count");
    if(only)counters.kv_only_tokens+=count;
}
void Runtime::Impl::decode_segment(int part,int layer,int tokens,Workspace& w,bool eligible,const std::function<void()>& body) {
    if(!(decode_segment_mask&(1u<<part)) || !eligible || tokens<1 || tokens>8 || (part==1 && tokens==1)){
        if(decode_segment_mask&(1u<<part))++w.segment_fallbacks[part];
        body();return;
    }
    auto& graph=w.decode_segments[part][layer][tokens-1];
    auto& disabled=w.decode_segment_disabled[part][layer][tokens-1];
    if(disabled){++w.segment_fallbacks[part];body();return;}
    std::string error;
    if(!graph.valid()){
        constexpr uint64_t cache_limit=128ull<<20;
        size_t before=0,total=0;gpu_check(cudaMemGetInfo(&before,&total));
        if(before<options.gpu_headroom+(16ull<<20) || w.segment_cache_bytes>=cache_limit){
            disabled=true;++w.segment_fallbacks[part];body();return;
        }
        const auto begin=std::chrono::steady_clock::now();
        if(!graph.begin(stream.value,error))throw std::runtime_error("DeepSeek decode segment "+error);
        try{body();}
        catch(...){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(stream.value,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);throw;}
        if(!graph.end(stream.value,error))throw std::runtime_error("DeepSeek decode segment "+error);
        size_t after=0;gpu_check(cudaMemGetInfo(&after,&total));
        const uint64_t charged=before>after?before-after:0;
        if(after<options.gpu_headroom || charged>cache_limit-w.segment_cache_bytes){
            // Capture has not executed. Discard it before the one eager execution.
            graph=core::CapturedGraph{};disabled=true;++w.segment_fallbacks[part];
            gpu_check(cudaMemGetInfo(&after,&total));
            if(after<options.gpu_headroom)throw std::runtime_error("DeepSeek graph headroom guard");
            body();return;
        }
        w.segment_cache_bytes+=charged;w.segment_nodes[part]+=graph.nodes();++w.segment_captures[part];
        w.segment_capture_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
    }
    // A failed launch fails the request: never replay its writes through eager fallback.
    if(!graph.launch(stream.value,error))throw std::runtime_error("DeepSeek decode segment "+error);
    ++w.segment_replays[part];
}
void Session::Impl::attention_input(int layer,int count,const std::function<void()>& before_q) {
    const auto& g=runtime.model.geometry();const auto stream=runtime.stream.value;
    auto mixes=[&](const char* stage,float* pre){
        gpu_check(hc_project_gpu(work.h,static_cast<const float*>(runtime.tensor(layer,std::string("hc_")+stage+"_fn").data),work.mixes,5120,count,g.norm_eps,stream));
        gpu_check(hc_split_gpu(work.mixes,static_cast<const float*>(runtime.tensor(layer,std::string("hc_")+stage+"_scale").data),
            static_cast<const float*>(runtime.tensor(layer,std::string("hc_")+stage+"_base").data),pre,work.post,work.comb,count,g.hc_eps,g.sinkhorn_iters,stream));
    };
    mixes("attn",work.attn_pre);
    gpu_check((runtime.options.arithmetic==Arithmetic::production?hc_pre_bf16_gpu:hc_pre_gpu)(work.h,work.pre,work.x,5120,count,stream));
    dump("hc_attn_pre-"+std::to_string(layer),work.x,{count,5120});
    gpu_check((runtime.options.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(work.x,static_cast<const float*>(runtime.tensor(layer,"attn_norm").data),work.x,5120,count,g.norm_eps,stream));
    dump("attn_norm-"+std::to_string(layer),work.x,{count,5120});
    if(before_q)before_q();
    runtime.matmul(runtime.tensor(layer,"attn_q_a"),work.x,work.qr,count,work);
    gpu_check((runtime.options.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(work.qr,static_cast<const float*>(runtime.tensor(layer,"attn_q_a_norm").data),work.qr,1280,count,g.norm_eps,stream));
    dump("qr-"+std::to_string(layer),work.qr,{count,1280});
    runtime.matmul(runtime.tensor(layer,"attn_q_b"),work.qr,work.q,count,work);
}
void Session::Impl::attention_output(int layer,int count,bool with_routes) {
    const auto& g=runtime.model.geometry();const auto stream=runtime.stream.value;
    const auto& wo=runtime.tensor(layer,"attn_output_a");
    for(int group=0;group<8;++group)runtime.matmul(wo,work.q+group*4096,work.o_low+group*1024,count,work,
        1024,32768,8192,int64_t(group)*1024*4096);
    runtime.matmul(runtime.tensor(layer,"attn_output_b"),work.o_low,work.branch,count,work);
    dump("attn_out-"+std::to_string(layer),work.branch,{count,5120});
    auto mixes=[&](const char* stage,float* pre){
        gpu_check(hc_project_gpu(work.h,static_cast<const float*>(runtime.tensor(layer,std::string("hc_")+stage+"_fn").data),work.mixes,5120,count,g.norm_eps,stream));
        gpu_check(hc_split_gpu(work.mixes,static_cast<const float*>(runtime.tensor(layer,std::string("hc_")+stage+"_scale").data),
            static_cast<const float*>(runtime.tensor(layer,std::string("hc_")+stage+"_base").data),pre,work.post,work.comb,count,g.hc_eps,g.sinkhorn_iters,stream));
    };
    gpu_check((runtime.options.arithmetic==Arithmetic::production?hc_post_bf16_gpu:hc_post_gpu)(work.h,work.branch,work.post,work.comb,work.h,5120,count,stream));
    dump("hc_attn_post-"+std::to_string(layer),work.h,{count,4,5120});
    mixes("ffn",work.pre);
    gpu_check((runtime.options.arithmetic==Arithmetic::production?hc_pre_bf16_gpu:hc_pre_gpu)(work.h,work.attn_pre,work.x,5120,count,stream));
    dump("hc_ffn_pre-"+std::to_string(layer),work.x,{count,5120});
    gpu_check((runtime.options.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(work.x,static_cast<const float*>(runtime.tensor(layer,"ffn_norm").data),work.x,5120,count,g.norm_eps,stream));
    dump("ffn_norm-"+std::to_string(layer),work.x,{count,5120});
    if(with_routes)runtime.route(layer,count,work);
}
void Session::Impl::attention(int layer,int count,int64_t start) {
    const auto& g=runtime.model.geometry();const auto& own=runtime.model.ownership()[layer];
    const auto stream=runtime.stream.value;
    auto& swa=local[layer];
    if(swa.position<0){swa.position=start;swa.count=0;}
    if(swa.position!=start)throw std::runtime_error("DeepSeek local SWA watermark layer="+std::to_string(layer));
    const int previous=swa.count,total=previous+count;
    gpu_check(rotate_gpu(work.q,64,512,count,start,g,own.compression!=0,false,stream));runtime.round(work.q,int64_t(count)*32768);
    dump("q-"+std::to_string(layer),work.q,{count,64,512});
    runtime.matmul(runtime.tensor(layer,"attn_kv"),work.x,work.kv,count,work);
    gpu_check((runtime.options.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(work.kv,static_cast<const float*>(runtime.tensor(layer,"attn_kv_a_norm").data),work.kv,512,count,g.norm_eps,stream));
    gpu_check(rotate_gpu(work.kv,1,512,count,start,g,own.compression!=0,false,stream));runtime.round(work.kv,int64_t(count)*512);
    dump("kv-"+std::to_string(layer),work.kv,{count,512});
    const int ring_begin=(start-previous)%128;
    if(runtime.options.arithmetic==Arithmetic::f32) {
        copy_window(reinterpret_cast<uint8_t*>(work.temp_swa_float),reinterpret_cast<const uint8_t*>(swa.values),ring_begin,previous,512*4,stream);
        gpu_check(cudaMemcpyAsync(work.temp_swa_float+int64_t(previous)*512,work.kv,int64_t(count)*512*4,cudaMemcpyDeviceToDevice,stream));
    }else {
        copy_window(work.temp_swa,swa.packed,ring_begin,previous,528,stream);
        std::vector<int64_t> slots(count);std::iota(slots.begin(),slots.end(),previous);
        gpu_check(cudaMemcpyAsync(work.write_slots,slots.data(),count*8,cudaMemcpyHostToDevice,stream));
        gpu_check(float_to_bf16_gpu(work.kv,work.kv_bf,int64_t(count)*512,stream));
        gpu_check(pack_swa_gpu(work.kv_bf,work.write_slots,count,512,contiguous(work.temp_swa,work.max_tokens+128,528,32),stream));
    }
    gpu_check(window_slots_gpu(work.swa_slots,count,previous,stream));
    GlobalCache* compressed=nullptr;int rows=0;
    if(own.compression) {
        if(own.owns_global())compression(layer,count,start,false);
        compressed=&global[own.global_owner];rows=(start+count)/own.compression;
        if(rows>compressed->rows)throw std::runtime_error("DeepSeek shared global watermark");
        if(own.computes_index()) {
            if(rows) {
                runtime.matmul(runtime.tensor(layer,"indexer.attn_q_b"),work.qr,work.index_q,count,work);
                gpu_check(rotate_gpu(work.index_q,32,128,count,start,g,true,false,stream));runtime.round(work.index_q,int64_t(count)*4096);
                if(runtime.options.arithmetic==Arithmetic::production)gpu_check(fake_quant_index_gpu(work.index_q,count*32,stream));
                runtime.matmul(runtime.tensor(layer,"indexer.proj"),work.x,work.index_w,count,work);
                if(layer>20 && candidate_epoch!=batch_epoch)throw std::runtime_error("DeepSeek candidate query epoch");
                const bool compact=layer>20;
                const int width=compact?16384:rows;
                const int batch=std::min<int>(work.index_max_queries,work.index_elements/width);
                if(batch<1)throw std::runtime_error("DeepSeek index workspace cannot hold one query");
                for(int begin=0;begin<count;begin+=batch){
                    const int n=std::min(batch,count-begin);const int64_t at=start+begin;
                    auto* candidates=work.candidate_blocks+int64_t(begin)*2048;
                    if(compressed->index_packed && scores_fast_enabled())
                        gpu_check(index_scores_mxfp4_gpu(work.index_q+int64_t(begin)*4096,work.index_w+int64_t(begin)*32,
                            compressed->index_packed,nullptr,compact?candidates:nullptr,work.scores,work.index_mx,
                            size_t(work.index_max_queries)*2304,n,rows,width,at,own.compression,stream));
                    else if(compact)gpu_check(compact_index_scores_gpu(work.index_q+int64_t(begin)*4096,work.index_w+int64_t(begin)*32,
                        compressed->index_values,compressed->index_packed,candidates,work.scores,n,rows,at,own.compression,stream));
                    else gpu_check(index_scores_gpu(work.index_q+int64_t(begin)*4096,work.index_w+int64_t(begin)*32,
                        compressed->index_values,compressed->index_packed,nullptr,work.scores,n,rows,at,own.compression,stream));
                    if(layer==20){
                        gpu_check(candidate_scores_gpu(work.scores,work.block_scores,n,rows,at,own.compression,stream));
                        if((rows+7)/8<=2048)gpu_check(candidate_all_blocks_gpu(work.block_scores,candidates,n,(rows+7)/8,stream));
                        else {
                            work.block_sort.temporary_bytes=work.block_sort_capacity;
                            gpu_check(sort_index_gpu(work.block_scores,n,(rows+7)/8,work.block_sort,stream));
                            gpu_check(candidate_blocks_gpu(work.block_sort,candidates,n,rows,stream));
                        }
                    }
                    work.sort.temporary_bytes=work.sort_capacity;
                    gpu_check(sort_index_gpu(work.scores,n,width,work.sort,stream));
                    auto* selected=work.global_slots+int64_t(begin)*512;
                    if(compact)gpu_check(compact_index_topk_gpu(work.sort,candidates,selected,n,rows,at,own.compression,stream));
                    else gpu_check(index_topk_gpu(work.sort,selected,n,rows,at,own.compression,stream));
                }
                if(layer==20)candidate_epoch=batch_epoch;
            }else gpu_check(cudaMemsetAsync(work.global_slots,0xff,int64_t(count)*512*8,stream));
            index_epoch=batch_epoch;index_source=layer;
        }
        if(index_epoch!=batch_epoch||index_source!=own.index_source)throw std::runtime_error("DeepSeek index source/query epoch");
    }
    if(runtime.options.arithmetic==Arithmetic::f32) {
        gpu_check(float_attention_gpu(work.q,work.temp_swa_float,compressed?compressed->values:nullptr,work.swa_slots,
            own.compression&&rows?work.global_slots:nullptr,static_cast<const float*>(runtime.tensor(layer,"attn_sinks").data),
            work.q,count,64,128,own.compression&&rows?512:0,total,rows,stream));
    }else {
        gpu_check(float_to_bf16_gpu(work.q,work.q_bf,int64_t(count)*32768,stream));
        PackedAttention p;p.q=work.q_bf;p.swa=contiguous(work.temp_swa,work.max_tokens+128,528,32);
        if(compressed)p.global=contiguous(compressed->packed,runtime.options.context/own.compression,288,64);
        p.swa_slots=work.swa_slots;p.global_slots=own.compression?work.global_slots:nullptr;
        p.sinks=static_cast<const float*>(runtime.tensor(layer,"attn_sinks").data);p.partial=work.partial_out;p.lse=work.lse;p.output=work.out_bf;
        p.tokens=count;p.global_topk=own.compression&&rows?512:0;p.splits=count<=8?8:1;
        if(count<=8 && !runtime.options.target_verify_batch){
            // The control path launches each query separately. Batched verify uses
            // the same per-query head/split reduction with disjoint scratch rows.
            p.tokens=1;
            for(int t=0;t<count;++t){p.q=work.q_bf+int64_t(t)*32768;p.output=work.out_bf+int64_t(t)*32768;
                p.swa_slots=work.swa_slots+int64_t(t)*128;
                p.global_slots=own.compression?work.global_slots+int64_t(t)*512:nullptr;
                gpu_check(packed_attention_cake_gpu(p,reinterpret_cast<int32_t*>(work.qr),size_t(count)*1280*4,stream));}
        }else gpu_check(packed_attention_cake_gpu(p,reinterpret_cast<int32_t*>(work.qr),size_t(count)*1280*4,stream));
        gpu_check(bf16_to_float_gpu(work.out_bf,work.q,int64_t(count)*32768,stream));
    }
    // SWA is private to every layer, including readers of shared global/index storage.
    const int keep=std::min(128,total),new_begin=(start+count-keep)%128;
    if(runtime.options.arithmetic==Arithmetic::f32)
        store_window(reinterpret_cast<uint8_t*>(swa.values),reinterpret_cast<const uint8_t*>(work.temp_swa_float+int64_t(total-keep)*512),new_begin,keep,512*4,stream);
    else store_window(swa.packed,work.temp_swa+int64_t(total-keep)*528,new_begin,keep,528,stream);
    swa.position=start+count;swa.count=keep;
    gpu_check(rotate_gpu(work.q,64,512,count,start,g,own.compression!=0,true,stream));runtime.round(work.q,int64_t(count)*32768);
    dump("attn_derope-"+std::to_string(layer),work.q,{count,64,512});

}
void Session::Impl::layers(int count,int64_t start,int first,int last,bool prefill,const BlockBenchmarkHooks* hooks) {
    const auto& g=runtime.model.geometry();const auto stream=runtime.stream.value;
    // Small replay windows keep demand-only uploads: streaming every expert
    // there would transfer many blobs that their routing never selects.
    const bool lookahead=runtime.options.prefill_expert_lookahead && prefill && count>=1024;
    if(lookahead && !hooks)runtime.begin_expert_prefetch(first,last,counters);
    struct PrefetchGuard {
        Runtime::Impl& runtime;bool enabled;
        ~PrefetchGuard(){if(enabled)runtime.cancel_expert_prefetch();}
    } prefetch_guard{runtime,lookahead};
    for(int layer=first;layer<last;++layer){
        const std::string label=runtime.options.trace?"ds41."+std::string(prefill?(ced&&first>=20?"tail_replay":"encoder"):"decode")+".layer."+std::to_string(layer):"";
        strata::prefill::LedgerScope trace(runtime.options.trace,label.c_str());
        check_cancel();
        if(layer==1||layer==14)eng(layer,count);
        capture_hidden(layer,count,start);
        const bool graph_eligible=!prefill && count>=1 && count<=8 && !hooks && !capture.enabled() &&
            runtime.options.arithmetic==Arithmetic::production && runtime.options.pcie_frac==1 &&
            runtime.options.slots==1;
        std::function<void()> before_q;
        if(hooks && layer==first)before_q=[&]{
            hooks->attention_begin();
            if(lookahead)runtime.begin_expert_prefetch(first,last,counters);
        };
        runtime.decode_segment(0,layer,count,work,graph_eligible,[&]{attention_input(layer,count,before_q);});
        attention(layer,count,start);
        const bool route_graph=graph_eligible && count>=2 && (runtime.decode_segment_mask&2);
        runtime.decode_segment(1,layer,count,work,route_graph,[&]{attention_output(layer,count,route_graph);});
        runtime.moe(layer,count,work,counters,prefill?(ced&&layer>=20?1:0):2,
            transaction_active&&transaction_routes_deferred&&!prefill?&transaction_routes[layer]:nullptr,route_graph);
        if(ced&&last_encoder_chunk&&layer<20)runtime.experts->encoder_layer_done(layer,stream,counters);
        dump("ffn_out-"+std::to_string(layer),work.branch,{count,5120});
        gpu_check((runtime.options.arithmetic==Arithmetic::production?hc_post_bf16_gpu:hc_post_gpu)(work.h,work.branch,work.post,work.comb,work.h,5120,count,stream));
        dump("l_last-"+std::to_string(layer),work.h,{count,4,5120});
        if(hooks && layer==last-1)hooks->last_block_end();
        if(layer<20)counters.encoder_token_layers+=count;else counters.decoder_token_layers+=count;
        if(!prefill && !transaction_active && runtime.options.adaptive_layer_progress)
            runtime.experts->progress_decode(stream,counters);
    }
    if(lookahead){runtime.end_expert_prefetch();prefetch_guard.enabled=false;}
}
std::vector<float> Session::Impl::logits(int count,bool all) {
    const auto stream=runtime.stream.value;const auto& g=runtime.model.geometry();
    gpu_check((runtime.options.arithmetic==Arithmetic::production?hc_pre_bf16_gpu:hc_pre_gpu)(work.h,work.pre,work.x,5120,count,stream));
    dump("hc_out",work.x,{count,5120});
    gpu_check((runtime.options.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(work.x,static_cast<const float*>(runtime.weights->at("output_norm.weight").data),work.x,5120,count,g.norm_eps,stream));
    dump("result_norm",work.x,{count,5120});
    const int rows=all||capture.enabled()?count:1;
    std::vector<float> result(int64_t(rows)*129280);
    const float* input=rows==count?work.x:work.x+int64_t(count-1)*5120;
    for(int first=0;first<rows;first+=work.max_logit_tokens) {
        const int nt=std::min(work.max_logit_tokens,rows-first);
        runtime.matmul(runtime.weights->at("output.weight"),input+int64_t(first)*5120,work.logits,nt,work,0,0,0,0,false);
        gpu_check(cudaMemcpyAsync(result.data()+int64_t(first)*129280,work.logits,int64_t(nt)*129280*4,cudaMemcpyDeviceToHost,stream));
    }
    gpu_check(cudaStreamSynchronize(stream));
    if(capture.enabled())capture.put_host("result_output",result,{rows,129280});
    return result;
}
} // namespace strata::models::deepseek41
