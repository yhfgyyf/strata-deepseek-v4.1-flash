#include "runtime_internal.hpp"
#include "strata/core/spec_prob.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <limits>

namespace strata::models::deepseek41 {
void DraftWork::init(WorkspaceLayout& layout,int vocab) {
    if(vocab<1||vocab>129280)throw std::invalid_argument("draft workspace vocabulary");
    auto f=[&](uint64_t n){return layout.alloc<float>(n);};
    main_hidden=f(128*15360);main_x=f(128*5120);h=f(7*20480);pre=f(7*4);attn_pre=f(7*4);post=f(7*4);comb=f(7*16);mixes=f(7*24);
    x=f(7*5120);qr=f(7*1280);q=f(7*32768);kv=f(7*512);o_low=f(7*8192);branch=f(7*5120);seed_kv=f(128*512);
    shared_gate=f(7*2304);shared_up=f(7*2304);shared_act=f(7*2304);shared_out=f(7*5120);route_logits=f(7*128);route_weights=f(7*3);parts=f(7*3*5120);
    mean=f(7*5120);markov=f(7*256);bias=f(vocab);confidence_input=f(5376);confidence=f(7);head=f(7*vocab);
    cache=layout.alloc<uint8_t>(160*528);q_bf=layout.alloc<uint16_t>(7*32768);kv_bf=layout.alloc<uint16_t>(128*512);
    out_bf=layout.alloc<uint16_t>(7*32768);partial=layout.alloc<uint16_t>(8*32768);lse=f(8*64);
    ids=layout.alloc<int>(7);routes=layout.alloc<int>(7*3);slots=layout.alloc<int64_t>(7*135);writes=layout.alloc<int64_t>(135);
    greedy_ids=layout.alloc<int>(9);penalty_counts=layout.alloc<int>(vocab);
    inject_hidden=f(128*15360);
}
DraftKvCache::DraftKvCache(core::DeviceArena& arena,bool floating) {
    for(int layer=0;layer<3;++layer){
        if(floating)values[layer]=allocate<float>(arena,128*512);
        else packed[layer]=allocate<uint8_t>(arena,128*528);
    }
}
DraftVocab::DraftVocab(const std::string& path):inverse(129280,-1) {
    if(path.empty()){ids.resize(129280);std::iota(ids.begin(),ids.end(),0);}
    else{
        std::ifstream in(path,std::ios::binary|std::ios::ate);const auto bytes=in.tellg();
        if(!in||bytes<=0||bytes>129280*4||bytes%4)throw std::invalid_argument("draft vocabulary file size");
        ids.resize(static_cast<size_t>(bytes)/4);in.seekg(0);in.read(reinterpret_cast<char*>(ids.data()),bytes);
        if(!in)throw std::runtime_error("draft vocabulary read");
    }
    int previous=-1;
    for(size_t i=0;i<ids.size();++i){const int id=ids[i];
        if(id<=previous||id>=129280)throw std::invalid_argument("draft vocabulary IDs must be sorted, unique and in range");
        inverse[id]=i;previous=id;
    }
    for(int id:{0,1,2,128799})if(inverse[id]<0)throw std::invalid_argument("draft vocabulary missing required special ID");
}
namespace {
const std::vector<int32_t>* checked_input_vocab(const DraftVocab& output,const DraftVocab& input,bool selected){
    for(int id:output.ids)if(input.inverse[id]<0)throw std::invalid_argument("draft output vocabulary must be a subset of input vocabulary");
    return selected?&input.ids:nullptr;
}
}
DraftWeights::DraftWeights(const Model& model,Arithmetic arithmetic,cudaStream_t stream,const std::string& vocab_path,const std::string& input_vocab_path)
    :vocab(vocab_path),input_vocab(input_vocab_path.empty()?vocab_path:input_vocab_path),
     dense(model,arithmetic,stream,true,false,false,vocab_path.empty()?nullptr:&vocab.ids,
           checked_input_vocab(vocab,input_vocab,!vocab_path.empty()||!input_vocab_path.empty())) {
    uint64_t total=0,largest=0;
    for(int layer=0;layer<3;++layer){
        const auto s=model.expert(layer,0,true);
        if(s.type!=39&&s.type!=10&&s.type!=11&&s.type!=42)throw std::runtime_error("DSpark expert storage type");
        layer_offsets_[layer]=total;projection_bytes_[layer]=s.projection_bytes;expert_types_[layer]=s.type;
        total+=128*s.blob_bytes;largest=std::max(largest,s.blob_bytes);
    }
    experts_=std::make_unique<core::DeviceArena>(total+aligned(vocab.size()*4)+2*aligned(129280*4));experts_->alloc(total);
    gpu_vocab_ids=allocate<int>(*experts_,vocab.size());gpu_vocab_inverse=allocate<int>(*experts_,129280);
    gpu_input_inverse=allocate<int>(*experts_,129280);
    gpu_check(cudaMemcpyAsync(gpu_vocab_ids,vocab.ids.data(),vocab.size()*4,cudaMemcpyHostToDevice,stream));
    gpu_check(cudaMemcpyAsync(gpu_vocab_inverse,vocab.inverse.data(),129280*4,cudaMemcpyHostToDevice,stream));
    gpu_check(cudaMemcpyAsync(gpu_input_inverse,input_vocab.inverse.data(),129280*4,cudaMemcpyHostToDevice,stream));
    Pinned staging(largest);
    for(int layer=0;layer<3;++layer)for(int e=0;e<128;++e){
        const auto s=model.expert(layer,e,true);
        std::memcpy(staging.data,s.gate,s.projection_bytes);std::memcpy(staging.data+s.projection_bytes,s.up,s.projection_bytes);
        std::memcpy(staging.data+2*s.projection_bytes,s.down,s.projection_bytes);
        auto* out=static_cast<uint8_t*>(experts_->base())+layer_offsets_[layer]+e*s.blob_bytes;
        gpu_check(cudaMemcpyAsync(out,staging.data,s.blob_bytes,cudaMemcpyHostToDevice,stream));gpu_check(cudaStreamSynchronize(stream));
    }
}
const uint8_t* DraftWeights::expert(int layer,int id)const {
    if(layer<0||layer>=3||id<0||id>=128)throw std::out_of_range("DSpark expert ID");
    return static_cast<const uint8_t*>(experts_->base())+layer_offsets_[layer]+id*3*projection_bytes_[layer];
}
void draft_matrix_product(const Matrix& m,const float* x,float* out,int tokens,cublasHandle_t blas,
                          cudaStream_t stream,Workspace& work,Arithmetic arithmetic,int rows,
                          int64_t ldx,int64_t ldy,uint64_t offset,bool round_output) {
    if(tokens<1 || tokens>work.max_tokens || (tokens>8 && arithmetic==Arithmetic::production && m.cols>work.max_bf_input_cols))
        throw std::invalid_argument("DSpark matrix exceeds workspace scratch");
    if(m.type!=2&&!m.packed_bf16){
        matrix_product(m,x,out,tokens,blas,stream,work.bf_input,arithmetic,rows,ldx,ldy,offset,round_output);return;
    }
    const int n=rows?rows:m.rows,k=m.cols;
    if(!ldx)ldx=k;if(!ldy)ldy=n;
    if((m.type!=2&&m.type!=8)||tokens<1||tokens>work.max_tokens||n<1||k<1||k%32||offset%32||
       offset+uint64_t(n)*k>m.elements||ldx<k||ldy<n)throw std::invalid_argument("DSpark packed matrix view");
    const int block_bytes=m.type==2?18:34;
    const auto* packed=static_cast<const uint8_t*>(m.data)+offset/32*block_bytes;
    if(tokens<=8){
        for(int t=0;t<tokens;++t){
            if(m.packed_bf16&&arithmetic==Arithmetic::production)
                gpu_check(packed_bf16_gemv_gpu(packed,m.type,x+int64_t(t)*ldx,out+int64_t(t)*ldy,n,k,1,stream));
            else gpu_check(packed_gemv_gpu(packed,m.type,x+int64_t(t)*ldx,out+int64_t(t)*ldy,n,k,1,stream));
        }
        if(round_output&&arithmetic==Arithmetic::production)gpu_check(round_bf16_strided_gpu(out,tokens,n,ldy,stream));
        return;
    }
    // Reuse existing expert scratch; no resident BF16 copy of the packed dense matrix.
    for(int begin=0;begin<n;begin+=512){
        const int count=std::min(512,n-begin);const uint64_t elements=uint64_t(count)*k;
        if(elements>5120*2304)throw std::invalid_argument("DSpark packed matrix tile exceeds scratch");
        const auto* tile=packed+uint64_t(begin)*k/32*block_bytes;
        if(arithmetic==Arithmetic::production){
            gpu_check(dequant_bf16_gpu(tile,m.type,work.expert_gate,elements,stream));
            Matrix expanded{work.expert_gate,30,count,k,elements};
            matrix_product(expanded,x,out+begin,tokens,blas,stream,work.bf_input,arithmetic,0,ldx,ldy,0,round_output);
        }else{
            gpu_check(dequant_gpu(tile,m.type,work.dequant,elements,stream));
            Matrix expanded{work.dequant,0,count,k,elements};
            matrix_product(expanded,x,out+begin,tokens,blas,stream,work.bf_input,arithmetic,0,ldx,ldy,0,round_output);
        }
    }
}
void Session::Impl::capture_hidden(int layer,int count,int64_t start) {
    if(layer<37||layer>39)return;
    if(!runtime.options.dspark&&!capture.enabled())return;
    gpu_check(mean_residual_gpu(work.h,work.engram_emb,count,runtime.stream.value));runtime.round(work.engram_emb,int64_t(count)*5120);
    dump("draft_capture_before_attn-"+std::to_string(layer),work.engram_emb,{count,5120});
    const int skip=std::max(0,count-128);
    for(int t=skip;t<count;++t)gpu_check(cudaMemcpyAsync(draft_hidden+((start+t)%128)*15360+(layer-37)*5120,
        work.engram_emb+int64_t(t)*5120,5120*4,cudaMemcpyDeviceToDevice,runtime.stream.value));
    if(layer==39)draft_capture_position=start+count;
}
const char* draft_cache_label(Arithmetic arithmetic) {
    return arithmetic==Arithmetic::f32 ? "F32 cache and attention" : "BF16 input, E4M3 group32 E8M0 scales, BF16 probabilities and split outputs";
}
DraftProposal draft_forward(DraftContext& c,const float* hidden,int window,int64_t position,
                            int first,const std::vector<int>& token_history,const kernels::SamplerParams& sampling) {
    const int draft_tokens=c.draft_tokens;
    if((draft_tokens!=5&&draft_tokens!=7)||window<1||window>128||position<window||first<0||first>=129280||!hidden||c.work.max_tokens<std::max(window,draft_tokens))
        throw std::invalid_argument("DSpark supplied hidden/position contract");
    auto& d=c.draft;auto& weights=c.weights;auto& w=c.work;const auto stream=c.stream;const auto& g=c.model.geometry();
    const int vocab=weights.vocab.size();
    if(weights.input_vocab.inverse[first]<0)return {}; // The main model handles excluded first tokens.
    const bool floating=c.arithmetic==Arithmetic::f32;
    if(c.capture&&c.capture->enabled()){
        std::ofstream out(c.capture->root/"draft-arithmetic.json");
        out<<"{\"arithmetic\":\""<<(floating?"f32":"production")<<"\",\"cache\":\""<<draft_cache_label(c.arithmetic)<<"\"}\n";
        if(!out)throw std::runtime_error("draft capture arithmetic metadata");
    }
    auto check_cancel=[&]{if(c.check_cancel)c.check_cancel();};
    auto round=[&](float* x,uint64_t n){if(c.arithmetic==Arithmetic::production)gpu_check(round_bf16_gpu(x,n,stream));};
    auto matmul=[&](const Matrix& m,const float* x,float* out,int tokens,Workspace& work,int rows=0,int64_t ldx=0,
                   int64_t ldy=0,uint64_t offset=0,bool round_output=true){
        if(c.batch_projection&&tokens==draft_tokens&&(m.type==0||m.type==2||m.type==8||m.type==14||m.type==30)){
            const int n=rows?rows:m.rows,k=m.cols;const bool packed=m.type==2||m.type==8||m.type==14;
            const int block=m.type==14?256:32;
            if(!ldx)ldx=k;if(!ldy)ldy=n;
            if(n<1||k<1||(packed&&(offset%block||k%block))||offset+uint64_t(n)*k>m.elements)
                throw std::invalid_argument("DSpark draft matrix view");
            const uint64_t bytes=packed?offset/block*(m.type==14?210:m.type==2?18:34):offset*(m.type==0?4:2);
            gpu_check((tokens==5?draft_projection5_gpu:draft_projection7_gpu)(static_cast<const uint8_t*>(m.data)+bytes,m.type,x,out,n,k,ldx,ldy,
                m.packed_bf16&&c.arithmetic==Arithmetic::production,round_output&&c.arithmetic==Arithmetic::production,stream));
            if(c.stats)++c.stats->draft_batch_projection_calls;
            return;
        }
        draft_matrix_product(m,x,out,tokens,c.blas,stream,work,c.arithmetic,rows,ldx,ldy,offset,round_output);
    };
    auto dump=[&](const std::string& name,const float* x,const std::vector<int64_t>& shape){
        if(c.capture)c.capture->put(name,x,shape,stream);
    };
    const int64_t begin=position-window;
    auto* context_cache=c.capture&&c.capture->enabled()?nullptr:c.kv_cache;
    const bool reuse=context_cache&&window==128&&context_cache->weights==&weights&&
        context_cache->position>=128&&context_cache->position<=position&&position-context_cache->position<=128;
    const int updates=reuse?int(position-context_cache->position):window;
    const int64_t update_begin=position-updates;
    // Retain the original GEMM shape for resident BF16 context weights. Their
    // wide projection changes rounding between 64 and 128 columns; packed
    // weights use fixed 512-row tiles and retain the 64-column update path.
    const int minimum_projection=weights.dense.at("fc.weight").type==30?128:64;
    const int projected=reuse&&updates>0?std::max(updates,minimum_projection):updates;
    if(context_cache){context_cache->position=-1;context_cache->weights=&weights;}
    if(hidden!=d.main_hidden)gpu_check(cudaMemcpyAsync(d.main_hidden,hidden,int64_t(window)*15360*4,cudaMemcpyDeviceToDevice,stream));
    dump("draft.main_hidden",d.main_hidden,{window,15360});
    if(updates){
        const float* input_hidden=d.main_hidden+int64_t(window-updates)*15360;
        if(projected!=updates){
            gpu_check(cudaMemcpyAsync(d.inject_hidden,input_hidden,int64_t(updates)*15360*4,cudaMemcpyDeviceToDevice,stream));
            gpu_check(cudaMemsetAsync(d.inject_hidden+int64_t(updates)*15360,0,int64_t(projected-updates)*15360*4,stream));
            input_hidden=d.inject_hidden;
        }
        matmul(weights.dense.at("fc.weight"),input_hidden,d.main_x,projected,w);
        dump("draft.fc",d.main_x,{updates,5120});
        gpu_check((c.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(d.main_x,static_cast<const float*>(weights.dense.at("enc.output_norm.weight").data),d.main_x,5120,projected,g.norm_eps,stream));
        dump("draft.main_x",d.main_x,{updates,5120});
    }
    if(c.stats){c.stats->draft_context_rows+=updates;c.stats->draft_context_projection_rows+=projected;}
    const int mask=weights.input_vocab.inverse[128799];
    std::vector<int> input(draft_tokens,mask);input[0]=weights.input_vocab.inverse[first];
    gpu_check(cudaMemcpyAsync(d.ids,input.data(),draft_tokens*4,cudaMemcpyHostToDevice,stream));
    const auto& embedding=weights.dense.at("token_embd.weight");
    gpu_check(init_residual_gpu(embedding.data,embedding.type,d.ids,nullptr,nullptr,d.h,d.pre,draft_tokens,stream));round(d.h,draft_tokens*20480);
    dump("draft.init",d.h,{draft_tokens,4,5120});
    auto norm=[&](const float* x,const std::string& name,float* out,int width,int count,int layer){
        gpu_check((c.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(x,static_cast<const float*>(weights.tensor(layer,name).data),out,width,count,g.norm_eps,stream));
    };
    std::vector<int64_t> writes(window+draft_tokens);std::iota(writes.begin(),writes.end(),0);
    std::vector<int64_t> slots(draft_tokens*(window+draft_tokens));for(int t=0;t<draft_tokens;++t)std::iota(slots.begin()+t*(window+draft_tokens),slots.begin()+(t+1)*(window+draft_tokens),0);
    gpu_check(cudaMemcpyAsync(d.writes,writes.data(),writes.size()*8,cudaMemcpyHostToDevice,stream));
    gpu_check(cudaMemcpyAsync(d.slots,slots.data(),slots.size()*8,cudaMemcpyHostToDevice,stream));
    std::vector<int64_t> context_writes(updates);
    if(context_cache&&updates){
        for(int t=0;t<updates;++t)context_writes[t]=(update_begin+t)%128;
        gpu_check(cudaMemcpyAsync(w.write_slots,context_writes.data(),updates*8,cudaMemcpyHostToDevice,stream));
    }
    for(int layer=0;layer<3;++layer){
        check_cancel();auto hc=[&](const char* stage,float* pre){
            const std::string p=std::string("hc_")+stage;
            gpu_check(hc_project_gpu(d.h,static_cast<const float*>(weights.tensor(layer,p+"_fn").data),d.mixes,5120,draft_tokens,g.norm_eps,stream));
            gpu_check(hc_split_gpu(d.mixes,static_cast<const float*>(weights.tensor(layer,p+"_scale").data),static_cast<const float*>(weights.tensor(layer,p+"_base").data),pre,d.post,d.comb,draft_tokens,g.hc_eps,20,stream));
        };
        hc("attn",d.attn_pre);gpu_check((c.arithmetic==Arithmetic::production?hc_pre_bf16_gpu:hc_pre_gpu)(d.h,d.pre,d.x,5120,draft_tokens,stream));norm(d.x,"attn_norm",d.x,5120,draft_tokens,layer);
        const std::string prefix="draft.blk."+std::to_string(layer)+".";
        dump(prefix+"attn_norm",d.x,{draft_tokens,5120});
        if(updates){
            matmul(weights.tensor(layer,"attn_kv"),d.main_x,d.seed_kv,projected,w);norm(d.seed_kv,"attn_kv_a_norm",d.seed_kv,512,updates,layer);
            gpu_check(rotate_gpu(d.seed_kv,1,512,updates,update_begin,g,false,false,stream));round(d.seed_kv,int64_t(updates)*512);
            dump(prefix+"seed_kv",d.seed_kv,{updates,512});
        }
        CacheLayout cache{d.cache,160*528,32*528,528,32};
        if(context_cache){
            const int stride=floating?512*4:528;
            auto* saved=floating?reinterpret_cast<uint8_t*>(context_cache->values[layer]):context_cache->packed[layer];
            if(updates){
                if(floating){
                    for(int t=0;t<updates;++t)gpu_check(cudaMemcpyAsync(saved+context_writes[t]*stride,
                        d.seed_kv+int64_t(t)*512,stride,cudaMemcpyDeviceToDevice,stream));
                }else{
                    gpu_check(float_to_bf16_gpu(d.seed_kv,d.kv_bf,int64_t(updates)*512,stream));
                    CacheLayout ring{saved,128*528,32*528,528,32};
                    gpu_check(pack_swa_gpu(d.kv_bf,w.write_slots,updates,512,ring,stream));
                }
            }
            auto* output=floating?reinterpret_cast<uint8_t*>(w.temp_swa_float):d.cache;
            const int at=begin%128,first_count=std::min(window,128-at);
            gpu_check(cudaMemcpyAsync(output,saved+int64_t(at)*stride,int64_t(first_count)*stride,cudaMemcpyDeviceToDevice,stream));
            if(first_count<window)gpu_check(cudaMemcpyAsync(output+int64_t(first_count)*stride,saved,int64_t(window-first_count)*stride,cudaMemcpyDeviceToDevice,stream));
        }else if(floating)gpu_check(cudaMemcpyAsync(w.temp_swa_float,d.seed_kv,int64_t(window)*512*4,cudaMemcpyDeviceToDevice,stream));
        else{
            gpu_check(float_to_bf16_gpu(d.seed_kv,d.kv_bf,int64_t(window)*512,stream));
            gpu_check(pack_swa_gpu(d.kv_bf,d.writes,window,512,cache,stream));
        }
        matmul(weights.tensor(layer,"attn_q_a"),d.x,d.qr,draft_tokens,w);norm(d.qr,"attn_q_a_norm",d.qr,1280,draft_tokens,layer);
        matmul(weights.tensor(layer,"attn_q_b"),d.qr,d.q,draft_tokens,w);gpu_check(rotate_gpu(d.q,64,512,draft_tokens,position,g,false,false,stream));round(d.q,draft_tokens*32768);
        dump(prefix+"qr",d.qr,{draft_tokens,1280});dump(prefix+"q",d.q,{draft_tokens,64,512});
        matmul(weights.tensor(layer,"attn_kv"),d.x,d.kv,draft_tokens,w);norm(d.kv,"attn_kv_a_norm",d.kv,512,draft_tokens,layer);
        gpu_check(rotate_gpu(d.kv,1,512,draft_tokens,position,g,false,false,stream));round(d.kv,draft_tokens*512);
        dump(prefix+"kv",d.kv,{draft_tokens,512});
        if(floating){
            gpu_check(cudaMemcpyAsync(w.temp_swa_float+int64_t(window)*512,d.kv,draft_tokens*512*4,cudaMemcpyDeviceToDevice,stream));
            gpu_check(float_attention_gpu(d.q,w.temp_swa_float,nullptr,d.slots,nullptr,
                static_cast<const float*>(weights.tensor(layer,"attn_sinks").data),d.q,draft_tokens,64,window+draft_tokens,0,window+draft_tokens,0,stream));
        }else{
            gpu_check(float_to_bf16_gpu(d.kv,d.kv_bf,draft_tokens*512,stream));gpu_check(pack_swa_gpu(d.kv_bf,d.writes+window,draft_tokens,512,cache,stream));
            gpu_check(float_to_bf16_gpu(d.q,d.q_bf,draft_tokens*32768,stream));
            PackedAttention a;a.q=d.q_bf;a.output=d.out_bf;a.tokens=draft_tokens;a.swa=cache;
            a.swa_slots=d.slots;a.swa_topk=window+draft_tokens;a.splits=1;a.partial=d.partial;a.lse=d.lse;
            a.sinks=static_cast<const float*>(weights.tensor(layer,"attn_sinks").data);
            // Q projection scratch is dead until the next layer; keep the draft arena unchanged.
            gpu_check(packed_attention_cake_draft_gpu(a,reinterpret_cast<int32_t*>(d.qr),size_t(draft_tokens)*1280*4,stream));
            gpu_check(bf16_to_float_gpu(d.out_bf,d.q,draft_tokens*32768,stream));
        }
        gpu_check(rotate_gpu(d.q,64,512,draft_tokens,position,g,false,true,stream));round(d.q,draft_tokens*32768);
        dump(prefix+"attention",d.q,{draft_tokens,64,512});
        const auto& wo=weights.tensor(layer,"attn_output_a");for(int group=0;group<8;++group)matmul(wo,d.q+group*4096,d.o_low+group*1024,draft_tokens,w,1024,32768,8192,int64_t(group)*1024*4096);
        matmul(weights.tensor(layer,"attn_output_b"),d.o_low,d.branch,draft_tokens,w);
        dump(prefix+"attn_out",d.branch,{draft_tokens,5120});
        gpu_check((c.arithmetic==Arithmetic::production?hc_post_bf16_gpu:hc_post_gpu)(d.h,d.branch,d.post,d.comb,d.h,5120,draft_tokens,stream));
        dump(prefix+"attn_post",d.h,{draft_tokens,4,5120});
        hc("ffn",d.pre);gpu_check((c.arithmetic==Arithmetic::production?hc_pre_bf16_gpu:hc_pre_gpu)(d.h,d.attn_pre,d.x,5120,draft_tokens,stream));norm(d.x,"ffn_norm",d.x,5120,draft_tokens,layer);
        dump(prefix+"ffn_norm",d.x,{draft_tokens,5120});
        matmul(weights.tensor(layer,"ffn_gate_inp"),d.x,d.route_logits,draft_tokens,w,0,0,0,0,false);
        const auto& shared_gate=weights.tensor(layer,"ffn_gate_shexp");
        const auto& shared_up=weights.tensor(layer,"ffn_up_shexp");
        const auto& shared_down=weights.tensor(layer,"ffn_down_shexp");
        const bool fused=c.fused_moe&&c.arithmetic==Arithmetic::production&&
            (weights.expert_type(layer)==10||weights.expert_type(layer)==39)&&
            (shared_gate.type==2||shared_gate.type==8||shared_gate.type==30)&&
            shared_up.type==shared_gate.type&&shared_down.type==shared_gate.type;
        const auto* route_bias=static_cast<const float*>(weights.dense.at("blk."+std::to_string(layer)+".exp_probs_b.bias").data);
        gpu_check((fused?route_parallel_gpu:route_gpu)(d.route_logits,route_bias,nullptr,nullptr,d.routes,d.route_weights,128,3,draft_tokens,1.5f,stream));
        if(!fused||(c.capture&&c.capture->enabled())){
            gpu_check(cudaMemcpyAsync(w.host_ids,d.routes,(draft_tokens*3)*4,cudaMemcpyDeviceToHost,stream));
            gpu_check(cudaMemcpyAsync(w.host_weights,d.route_weights,(draft_tokens*3)*4,cudaMemcpyDeviceToHost,stream));gpu_check(cudaStreamSynchronize(stream));
        }
        dump(prefix+"route_weights",d.route_weights,{draft_tokens,3});
        if(c.capture&&c.capture->enabled()){
            std::ofstream out(c.capture->root/("routes-"+std::to_string(layer)+".json"));out<<'[';
            for(int i=0;i<(draft_tokens*3);++i){if(i)out<<',';out<<w.host_ids[i];}out<<"]\n";
        }
        if(fused){
            gpu_check(draft_fused_moe_gpu(weights.expert(layer,0),weights.expert_type(layer),weights.projection_bytes(layer),shared_gate.data,
                shared_up.data,shared_down.data,shared_gate.type,d.x,d.routes,d.route_weights,w.group_act,d.branch,draft_tokens,stream));
            if(c.stats)++c.stats->draft_fused_moe_calls;
        }else{
        matmul(shared_gate,d.x,d.shared_gate,draft_tokens,w);matmul(shared_up,d.x,d.shared_up,draft_tokens,w);
        gpu_check((c.arithmetic==Arithmetic::production?swiglu_bf16_gpu:swiglu_gpu)(d.shared_gate,d.shared_up,nullptr,d.shared_act,2304,draft_tokens,10,stream));
        matmul(shared_down,d.shared_act,d.shared_out,draft_tokens,w);
        int offset=0;
        struct Group{int id,offset,count;};std::vector<Group> groups;
        for(int e=0;e<128;++e){const int first=offset;for(int t=0;t<draft_tokens;++t)for(int k=0;k<3;++k)if(w.host_ids[t*3+k]==e){
            w.host_group_rows[offset]=t;w.host_group_slots[offset]=t*3+k;w.host_group_weights[offset]=w.host_weights[t*3+k];++offset;}
            if(offset>first)groups.push_back({e,first,offset-first});}
        gpu_check(cudaMemcpyAsync(w.group_rows,w.host_group_rows,(draft_tokens*3)*4,cudaMemcpyHostToDevice,stream));gpu_check(cudaMemcpyAsync(w.group_slots,w.host_group_slots,(draft_tokens*3)*4,cudaMemcpyHostToDevice,stream));
        gpu_check(cudaMemcpyAsync(w.group_weights,w.host_group_weights,(draft_tokens*3)*4,cudaMemcpyHostToDevice,stream));
        for(const auto& group:groups){
            const auto* blob=weights.expert(layer,group.id);const uint64_t projection=weights.projection_bytes(layer);const uint32_t type=weights.expert_type(layer);
            gpu_check(gather_rows_gpu(d.x,w.group_rows+group.offset,w.group_x,group.count,5120,stream));
            auto project=[&](const uint8_t* packed,const float* input,float* output,int rows,int cols,uint16_t* bf){
                if(c.arithmetic==Arithmetic::f32)gpu_check(packed_gemv_gpu(packed,type,input,output,rows,cols,group.count,stream));
                else{gpu_check(dequant_gpu(packed,type,w.dequant,int64_t(rows)*cols,stream));gpu_check(float_to_bf16_gpu(w.dequant,bf,int64_t(rows)*cols,stream));
                    Matrix m{bf,30,rows,cols,uint64_t(rows)*cols};matmul(m,input,output,group.count,w);}
            };
            project(blob,w.group_x,w.group_gate,2304,5120,w.expert_gate);project(blob+projection,w.group_x,w.group_up,2304,5120,w.expert_up);
            gpu_check((c.arithmetic==Arithmetic::production?swiglu_bf16_gpu:swiglu_gpu)(w.group_gate,w.group_up,w.group_weights+group.offset,w.group_act,2304,group.count,10,stream));
            project(blob+2*projection,w.group_act,w.group_out,5120,2304,w.expert_down);
            gpu_check(scatter_expert_gpu(w.group_out,w.group_slots+group.offset,d.parts,group.count,3,stream));
        }
        gpu_check(combine_experts_gpu(d.parts,d.routes,d.shared_out,d.branch,draft_tokens,3,stream));round(d.branch,draft_tokens*5120);
        }
        dump(prefix+"ffn_out",d.branch,{draft_tokens,5120});
        gpu_check((c.arithmetic==Arithmetic::production?hc_post_bf16_gpu:hc_post_gpu)(d.h,d.branch,d.post,d.comb,d.h,5120,draft_tokens,stream));
        dump(prefix+"output",d.h,{draft_tokens,4,5120});
    }
    if(context_cache)context_cache->position=position;
    gpu_check((c.arithmetic==Arithmetic::production?hc_pre_bf16_gpu:hc_pre_gpu)(d.h,d.pre,d.mean,5120,draft_tokens,stream));
    dump("draft.mean",d.mean,{draft_tokens,5120});
    gpu_check((c.arithmetic==Arithmetic::production?rms_bf16_gpu:rms_gpu)(d.mean,static_cast<const float*>(weights.dense.at("output_norm.weight").data),d.x,5120,draft_tokens,g.norm_eps,stream));
    matmul(weights.dense.at("output.weight"),d.x,d.head,draft_tokens,w,0,0,0,0,false);
    dump("draft.head",d.head,{draft_tokens,vocab});
    validate_sampling(sampling);
    const bool diagnostic=c.capture&&c.capture->enabled();
    const bool penalties=sampling.penalty_last_n>0&&(sampling.penalty_repeat!=1||sampling.penalty_freq!=0||sampling.penalty_present!=0);
    const int history_size=penalties?std::min<int64_t>(sampling.penalty_last_n,token_history.size()+1):0;
    if(c.gpu_greedy&&!diagnostic&&(sampling.greedy||sampling.temperature==0)&&history_size<=w.max_tokens){
        // The CPU and GPU both add head+bias in FP32, then apply penalties in FP64.
        std::array<int,9> chain{};chain[0]=first;const size_t chain_bytes=(draft_tokens+2)*sizeof(int);
        gpu_check(cudaMemcpyAsync(d.greedy_ids,chain.data(),chain_bytes,cudaMemcpyHostToDevice,stream));
        std::vector<int> history;
        if(penalties){
            if(history_size>1)history.assign(token_history.end()-(history_size-1),token_history.end());
            history.push_back(first);
            gpu_check(cudaMemcpyAsync(w.ids,history.data(),history.size()*4,cudaMemcpyHostToDevice,stream));
            gpu_check(cudaMemsetAsync(d.penalty_counts,0,vocab*4,stream));
            gpu_check(draft_penalty_counts_gpu(w.ids,history_size,weights.gpu_vocab_inverse,d.penalty_counts,stream));
        }
        for(int i=0;i<draft_tokens;++i){
            check_cancel();const auto& markov=weights.dense.at("markov_w1.weight");
            gpu_check(draft_markov_row_gpu(markov.data,markov.type,d.greedy_ids+i,weights.gpu_input_inverse,
                d.markov+i*256,c.arithmetic==Arithmetic::production,d.greedy_ids+draft_tokens+1,stream));
            matmul(weights.dense.at("markov_w2.weight"),d.markov+i*256,d.bias,1,w,0,0,0,0,false);
            gpu_check(draft_greedy_gpu(d.head+int64_t(i)*vocab,d.bias,weights.gpu_vocab_ids,weights.gpu_vocab_inverse,vocab,
                penalties?w.ids:nullptr,history_size,i,d.greedy_ids,penalties?d.penalty_counts:nullptr,sampling,d.greedy_ids+draft_tokens+1,stream));
        }
        DraftProposal proposal;
        if(c.adaptive_confidence){
            proposal.confidence.resize(draft_tokens);
            gpu_check(draft_confidence_gpu(d.mean,d.markov,static_cast<const float*>(weights.dense.at("conf_proj.weight").data),
                d.confidence,nullptr,draft_tokens,stream));
            gpu_check(cudaMemcpyAsync(proposal.confidence.data(),d.confidence,draft_tokens*4,cudaMemcpyDeviceToHost,stream));
            if(c.stats)++c.stats->confidence_rounds;
        }
        gpu_check(cudaMemcpyAsync(chain.data(),d.greedy_ids,chain_bytes,cudaMemcpyDeviceToHost,stream));
        gpu_check(cudaStreamSynchronize(stream));
        if(chain[draft_tokens+1])throw std::runtime_error("DeepSeek GPU draft sampling support/nonfinite logit");
        for(int i=1;i<=draft_tokens;++i){proposal.tokens.push_back(chain[i]);proposal.distributions.push_back({{chain[i]},{1.0},{1.0f}});}
        if(c.stats)++c.stats->draft_gpu_rounds;
        return proposal;
    }
    std::vector<float> logits(draft_tokens*vocab);gpu_check(cudaMemcpyAsync(logits.data(),d.head,logits.size()*4,cudaMemcpyDeviceToHost,stream));gpu_check(cudaStreamSynchronize(stream));
    DraftProposal proposal;std::vector<int> history=token_history;history.push_back(first);int previous=first;
    for(int i=0;i<draft_tokens;++i){
        const auto& markov=weights.dense.at("markov_w1.weight");
        const uint64_t row_bytes=markov.type==2?256/32*18:markov.type==8?256/32*34:256*(markov.type==0?4:2);
        const auto* row=static_cast<const uint8_t*>(markov.data)+uint64_t(weights.input_vocab.inverse[previous])*row_bytes;
        gpu_check(dequant_gpu(row,markov.type,d.markov,256,stream));round(d.markov,256);
        matmul(weights.dense.at("markov_w2.weight"),d.markov,d.bias,1,w,0,0,0,0,false);
        dump("draft.markov."+std::to_string(i),d.markov,{256});dump("draft.bias."+std::to_string(i),d.bias,{vocab});
        std::vector<float> bias(vocab);gpu_check(cudaMemcpyAsync(bias.data(),d.bias,bias.size()*4,cudaMemcpyDeviceToHost,stream));gpu_check(cudaStreamSynchronize(stream));
        std::vector<float> full(129280,-std::numeric_limits<float>::infinity());
        for(int v=0;v<vocab;++v)full[weights.vocab.ids[v]]=logits[i*vocab+v]+bias[v];
        auto p=distribution(full.data(),129280,sampling,history);
        const int token=sample(p,draw_uniform(sampling.seed,(sampling.counter+position+i+1)|core::kSpecCtrDraft));
        if(c.adaptive_confidence){
            gpu_check(draft_confidence_gpu(d.mean+i*5120,d.markov,static_cast<const float*>(weights.dense.at("conf_proj.weight").data),
                d.confidence+i,nullptr,1,stream));
        }else if(diagnostic||!c.gpu_greedy){
            gpu_check(cudaMemcpyAsync(d.confidence_input,d.mean+i*5120,5120*4,cudaMemcpyDeviceToDevice,stream));gpu_check(cudaMemcpyAsync(d.confidence_input+5120,d.markov,256*4,cudaMemcpyDeviceToDevice,stream));
            matmul(weights.dense.at("conf_proj.weight"),d.confidence_input,d.confidence,1,w,0,0,0,0,false);
            dump("draft.confidence."+std::to_string(i),d.confidence,{1});
            float confidence;gpu_check(cudaMemcpyAsync(&confidence,d.confidence,4,cudaMemcpyDeviceToHost,stream));gpu_check(cudaStreamSynchronize(stream));
            proposal.confidence.push_back(confidence);
        }
        proposal.tokens.push_back(token);proposal.distributions.push_back(std::move(p));history.push_back(token);previous=token;
    }
    if(c.adaptive_confidence){
        proposal.confidence.resize(draft_tokens);
        gpu_check(cudaMemcpyAsync(proposal.confidence.data(),d.confidence,draft_tokens*4,cudaMemcpyDeviceToHost,stream));
        gpu_check(cudaStreamSynchronize(stream));
        if(c.stats)++c.stats->confidence_rounds;
    }
    return proposal;
}
DraftProposal Session::propose(int first,const kernels::SamplerParams& sampling) {
    auto& s=*impl_;auto& r=s.runtime;std::lock_guard<std::mutex> lock(r.execution_mutex);
    if(!r.options.dspark||!r.draft_weights||!r.draft_work||s.position<1||s.draft_capture_position!=s.position||first<0||first>=129280)
        throw std::runtime_error("DSpark weights/capture/position contract");
    prefill::LedgerScope trace(r.options.trace,"ds41.draft");
    s.check_cancel();const auto before=std::chrono::steady_clock::now();auto& d=*r.draft_work;
    const int window=std::min<int64_t>(128,s.position);const int64_t begin=s.position-window;
    for(int t=0;t<window;++t)gpu_check(cudaMemcpyAsync(d.main_hidden+int64_t(t)*15360,
        s.draft_hidden+((begin+t)%128)*15360,15360*4,cudaMemcpyDeviceToDevice,r.stream.value));
    DraftContext context{r.model,r.options.arithmetic,*r.draft_weights,d,s.work,r.blas,r.stream.value,
                         &s.capture,[&]{s.check_cancel();}};
    context.gpu_greedy=r.options.dspark_gpu_greedy;context.stats=&s.counters;
    context.adaptive_confidence=r.options.dspark_adaptive_verification;context.draft_tokens=r.options.dspark_draft_tokens;
    context.kv_cache=r.options.dspark_incremental_kv?s.draft_kv.get():nullptr;
    context.fused_moe=r.options.dspark_fused_moe;context.batch_projection=r.options.dspark_batch_projection;
    auto proposal=draft_forward(context,d.main_hidden,window,s.position,first,s.tokens,sampling);
    s.counters.draft_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
    return proposal;
}
} // namespace strata::models::deepseek41
