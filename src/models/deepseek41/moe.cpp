#include "runtime_internal.hpp"
#include "expert_pipeline.hpp"
#include "bounded_moe.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include "strata/models/deepseek41/format_down.hpp"
#include "strata/models/deepseek41/mixed_expert.hpp"
#include "strata/kernels/cpu/deepseek41_down.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace strata::models::deepseek41 {
void Runtime::Impl::route(int layer,int tokens,Workspace& w) {
    matmul(tensor(layer,"ffn_gate_inp"),w.x,w.branch,tokens,w,0,0,0,0,false);
    const auto& bias=weights->at("blk."+std::to_string(layer)+".exp_probs_b.bias");
    const auto& vl=weights->at("blk."+std::to_string(layer)+".exp_probs_b_vl.bias");
    gpu_check((options.parallel_router?route_parallel_gpu:route_gpu)(w.branch,static_cast<const float*>(bias.data),static_cast<const float*>(vl.data),w.image,
        w.route_ids,w.route_weights,384,6,tokens,1.5f,stream.value));
    gpu_check(cudaMemcpyAsync(w.host_ids,w.route_ids,tokens*6*sizeof(int),cudaMemcpyDeviceToHost,stream.value));
    gpu_check(cudaMemcpyAsync(w.host_weights,w.route_weights,tokens*6*sizeof(float),cudaMemcpyDeviceToHost,stream.value));
}
void Runtime::Impl::moe(int layer,int tokens,Workspace& w,ExecutionStats& stats,int phase,std::vector<int>* deferred_routes,bool routes_ready) {
    prefill::LedgerScope moe_trace(options.trace,"ds41.moe");
    const bool prefill=phase!=2;
    const bool mixed_fast=options.arithmetic==Arithmetic::production&&expert_mixed_fast_enabled();
    if(decode_raw_cache && !prefill && tokens>8)decode_raw_cache->clear(upload.value);
    const auto& f=experts->format(layer);
    if(!routes_ready)route(layer,tokens,w);
    const bool cpu_path=options.arithmetic==Arithmetic::production &&
        (options.experts==ExpertMode::cpu || (options.experts==ExpertMode::hybrid && !prefill && tokens<=kernels::cpu::MAXT));
    if(cpu_path && options.pcie_frac<1)gpu_check(cudaMemcpyAsync(w.host_x,w.x,int64_t(tokens)*5120*4,cudaMemcpyDeviceToHost,stream.value));
    gpu_check(cudaStreamSynchronize(stream.value));
    for(int i=0;i<tokens*6;++i)if(w.host_ids[i]<0||w.host_ids[i]>=384||!std::isfinite(w.host_weights[i])){
        // Validate before updating counters or indexing expert tables. Preserve
        // enough context to distinguish a router failure from cache ownership.
        std::vector<float> logits(tokens*384),input(tokens*5120);
        gpu_check(cudaMemcpy(logits.data(),w.branch,logits.size()*4,cudaMemcpyDeviceToHost));
        gpu_check(cudaMemcpy(input.data(),w.x,input.size()*4,cudaMemcpyDeviceToHost));
        const auto bad=[](const auto& values){return std::count_if(values.begin(),values.end(),[](float x){return !std::isfinite(x);});};
        throw std::runtime_error("DeepSeek invalid route layer="+std::to_string(layer)+" rows="+std::to_string(tokens)+
            " route="+std::to_string(i)+" expert="+std::to_string(w.host_ids[i])+" nonfinite_input="+std::to_string(bad(input))+
            " nonfinite_logits="+std::to_string(bad(logits)));
    }
    if(phase==2)experts->observe_physical_routes(layer,w.host_ids,tokens);
    if(deferred_routes)deferred_routes->insert(deferred_routes->end(),w.host_ids,w.host_ids+tokens*6);
    else experts->observe_routes(layer,w.host_ids,tokens,phase);
    struct Group { int expert=0,offset=0; std::vector<int> rows,slots; std::vector<float> weights; };
    std::array<Group,384> groups;
    for(int e=0;e<384;++e)groups[e].expert=e;
    for(int t=0;t<tokens;++t)for(int k=0;k<6;++k){
        const int e=w.host_ids[t*6+k];if(e<0||e>=384)throw std::runtime_error("DeepSeek routed expert ID");
        groups[e].rows.push_back(t);groups[e].slots.push_back(t*6+k);groups[e].weights.push_back(w.host_weights[t*6+k]);
    }
    int metadata_offset=0;
    for(auto& g:groups){
        g.offset=metadata_offset;
        std::copy(g.rows.begin(),g.rows.end(),w.host_group_rows+metadata_offset);
        std::copy(g.slots.begin(),g.slots.end(),w.host_group_slots+metadata_offset);
        std::copy(g.weights.begin(),g.weights.end(),w.host_group_weights+metadata_offset);
        metadata_offset+=g.rows.size();
    }
    const char* fusion_env=std::getenv("STRATA_DS41_MAIN_MOE_FUSION");
    const int fusion=fusion_env&&fusion_env[0]=='2'?2:fusion_env&&fusion_env[0]=='1'?1:0;
    const bool grouped_down=fusion&&options.target_verify_batch&&phase==2&&tokens<=8&&
        options.arithmetic==Arithmetic::production&&options.expert_gu_mode&&
        (f.gu_type==10||f.gu_type==11)&&(f.d_type==10||f.d_type==11)&&
        kernels::cpu::deepseek41_down_fusion_enabled()&&w.max_tokens>=tokens*6;
    const bool single=options.expert_gu_mode && tokens==1&&!grouped_down;
    const bool short_possible=mixed_fast&&expert_mixed_short_enabled()&&phase==2&&tokens<=8&&
        options.target_verify_batch&&options.expert_direct_gemv&&expert_mixed_type(f.gu_type)&&
        expert_mixed_type(f.d_type)&&w.max_tokens>=tokens*6;
    if(!single){
        gpu_check(cudaMemcpyAsync(w.group_rows,w.host_group_rows,metadata_offset*sizeof(int),cudaMemcpyHostToDevice,stream.value));
        gpu_check(cudaMemcpyAsync(w.group_slots,w.host_group_slots,metadata_offset*sizeof(int),cudaMemcpyHostToDevice,stream.value));
        gpu_check(cudaMemcpyAsync(w.group_weights,w.host_group_weights,metadata_offset*sizeof(float),cudaMemcpyHostToDevice,stream.value));
    }
    std::vector<const Group*> cold,cpu;
    std::vector<std::pair<const Group*,const uint8_t*>> resident;
    const bool cache_active=decode_raw_cache && phase==2 && tokens<=8;
    for(const auto& g:groups)if(!g.rows.empty()) {
        if(options.experts==ExpertMode::cpu)cpu.push_back(&g);
        else if(const auto* blob=experts->gpu(layer,g.expert,stream.value,stats,phase))resident.push_back({&g,blob});
        else if(const auto* cached=cache_active?decode_raw_cache->find(layer*384+g.expert):nullptr){
            resident.push_back({&g,cached});
            stats.raw_cache_hits+=g.rows.size();stats.demand_expert_tokens+=g.rows.size();++stats.cold_expert_groups;
        }else if(cpu_path)cpu.push_back(&g);
        else cold.push_back(&g);
    }
    if(cpu_path && options.pcie_frac>0) {
        const size_t take=cpu.size()*static_cast<unsigned>(std::lround(options.pcie_frac*256))/256;
        for(size_t i=0;i<take;++i)stats.demand_expert_tokens+=cpu[i]->rows.size();
        stats.demand_expert_bytes+=take*f.bytes;stats.cold_expert_groups+=take;stats.demand_expert_groups+=take;
        cold.insert(cold.end(),cpu.begin(),cpu.begin()+take);cpu.erase(cpu.begin(),cpu.begin()+take);
    }
    // Route observation already counted true residency. Demand offload is kept
    // separate and never contributes to that hit counter.
    if(prefetch_active && !prefill)throw std::logic_error("prefill stream reached decode");
    if(prefetch_active){
        std::array<bool,384> selected{};
        for(const auto* group:cold)selected[group->expert]=true;
        prune_expert_prefetch(layer,selected,stats);
    }
    const bool issued=prefetch_active || (expert_pipeline && !cold.empty() &&
        (options.expert_pipeline || (!prefill && cpu_path && options.pcie_frac>0)));
    struct FailureGuard {
        ExpertPipeline* pipeline;cudaStream_t dequant,reader;int exceptions=std::uncaught_exceptions();
        ~FailureGuard(){if(pipeline && std::uncaught_exceptions()>exceptions){pipeline->stop();cudaStreamSynchronize(dequant);cudaStreamSynchronize(reader);}}
    } failure{issued?expert_pipeline.get():nullptr,dequant_stream.value,stream.value};
    if(issued && !prefetch_active) {
        const size_t issuer_slots=cache_active?8:0;
        std::vector<ExpertPipeline::Upload> direct;
        if(options.pinned_experts)for(const auto* group:cold){
            const auto* source=experts->pinned_blob(layer,group->expert);
            if(!source){direct.clear();break;}
            direct.push_back({source,f.bytes,options.trace?layer:-1,group->expert});
        }
        if(!direct.empty()){
            expert_pipeline->start(++expert_generation,std::move(direct),issuer_slots);
            stats.direct_expert_bytes+=cold.size()*f.bytes;
        }else expert_pipeline->start(++expert_generation,cold.size(),f.bytes,[&](size_t i,uint8_t* host) {
            prefill::LedgerScope fill_trace(options.trace,"ds41.expert.issuer.hostfill");
            if(experts->transient(layer,cold[i]->expert))experts->copy_blob(layer,cold[i]->expert,host);
            else std::memcpy(host,experts->blob_stable(layer,cold[i]->expert),f.bytes);
            if(options.arithmetic==Arithmetic::f32)experts->release_mapped(layer,cold[i]->expert);
        },issuer_slots);
        stats.h2d_expert_bytes+=cold.size()*f.bytes;
    }
    // Two existing expert banks bound every batch. Activation rows also fit
    // the original per-expert workspace; no residency or arena allocation changes.
    constexpr uint64_t expert_elements=5120ull*2304;
    // Only the measured full prefill chunk uses this path; smaller batches regress.
    const bool bounded=mixed_fast && options.expert_bf16 && options.expert_pipeline && tokens==8192 &&
        // These two pairs fail the whole-layer accuracy gate; keep567.
        !(f.gu_type==17 && f.d_type==39) && !(f.gu_type==11 && f.d_type==11) &&
        (expert_mixed_type(f.gu_type)||f.gu_type==11) && (expert_mixed_type(f.d_type)||f.d_type==11) &&
        w.expert_up==w.expert_gate+expert_elements && w.expert_ahead[0] &&
        w.expert_ahead[1]==w.expert_ahead[0]+expert_elements;
    const int bounded_capacity=2;
    auto batch_group=[&](const Group& g,const uint8_t* blob,int slot){
        uint16_t* gate=slot==0?w.expert_gate:w.expert_ahead[0];
        uint16_t* down=slot==0?w.expert_down:w.expert_ahead[2];
        return BoundedMoeGroup{blob,blob+f.up_off,blob+f.down_off,gate,down,g.offset,static_cast<int>(g.rows.size())};
    };
    auto run_batch=[&](const BoundedMoeGroup* batch,int count,bool prepared=false){
        const int rc=bounded_moe_gpu(batch,count,f.gu_type,f.d_type,w.x,w.group_rows,w.group_weights,w.group_slots,w.parts,
            w.bf_input,reinterpret_cast<uint16_t*>(w.group_gate),reinterpret_cast<uint16_t*>(w.group_act),
            reinterpret_cast<void**>(w.dequant),w.max_tokens,stream.value,prepared);
        if(rc)throw std::runtime_error("DeepSeek bounded grouped MoE error="+std::to_string(rc));
        for(int i=0;i<count;++i)stats.expert_gpu+=batch[i].tokens;
    };
    // Start demand upload before submitting shared and resident GPU work.
    auto shared=[&]{
        const auto& gate=tensor(layer,"ffn_gate_shexp");
        const auto& up=tensor(layer,"ffn_up_shexp");
        // Keep the validated shared path: the experimental grouped Q8 chain
        // has no material speed gain and two activation-reference failures.
        if(options.shared_q8_pair && options.arithmetic==Arithmetic::production && tokens<=8 &&
           options.dense_q8_1 && (tokens==1||options.dense_q8_batch) &&
           gate.packed_bf16 && up.packed_bf16 && gate.type==8 && up.type==8 &&
           gate.rows==2304 && up.rows==2304 && gate.cols==5120 && up.cols==5120){
            kernels::native_quantize_q8_1(w.x,w.dense_input_q8,5120,tokens,stream.value);
            if(!kernels::native_mmvq_pair(8,gate.data,up.data,w.dense_input_q8,w.shared_gate,w.shared_up,
                                         5120,2304,tokens,stream.value)){
                kernels::native_q8_0_mmvq(gate.data,w.dense_input_q8,w.shared_gate,5120,2304,tokens,stream.value);
                kernels::native_q8_0_mmvq(up.data,w.dense_input_q8,w.shared_up,5120,2304,tokens,stream.value);
            }
            gpu_check(round_bf16_strided_gpu(w.shared_gate,tokens,2304,2304,stream.value));
            gpu_check(round_bf16_strided_gpu(w.shared_up,tokens,2304,2304,stream.value));
        }else{
            matmul(gate,w.x,w.shared_gate,tokens,w);
            matmul(up,w.x,w.shared_up,tokens,w);
        }
        gpu_check((options.arithmetic==Arithmetic::production?swiglu_bf16_gpu:swiglu_gpu)(w.shared_gate,w.shared_up,nullptr,w.shared_act,2304,tokens,10.f,stream.value));
        matmul(tensor(layer,"ffn_down_shexp"),w.shared_act,w.shared_out,tokens,w);
    };
    if(options.decode_graph && phase==2 && tokens<=8){
        auto& graph=shared_graphs[layer][tokens-1];std::string error;
        if(!graph.valid()){
            if(!graph.begin(stream.value,error))throw std::runtime_error(error);
            try{shared();}catch(...){graph.end(stream.value,error);throw;}
            if(!graph.end(stream.value,error))throw std::runtime_error(error);
            ++stats.shared_graph_captures;
        }
        if(!graph.launch(stream.value,error))throw std::runtime_error(error);
        ++stats.shared_graph_launches;
    }else shared();
    auto compute=[&](const Group& g,const uint8_t* blob,const std::array<uint16_t*,3>* prepared=nullptr){
        const int nt=g.rows.size();
        if(!single)gpu_check(gather_rows_gpu(w.x,w.group_rows+g.offset,w.group_x,nt,5120,stream.value));
        const float* input=single?w.x:w.group_x;
        const float* routing=single?w.route_weights+g.slots[0]:w.group_weights+g.offset;
        const int* slots=single?nullptr:w.group_slots+g.offset;
        float* output=single?w.parts+int64_t(g.slots[0])*5120:w.parts;
        int projection_index=0;
        auto projection=[&](const uint8_t* packed,int rows,int cols,uint16_t* bf,const float* input,float* output){
            const int index=projection_index++;
            const int type=index==2?f.d_type:f.gu_type;
            const bool packed_supported=type==10||type==11||type==39;
            if(prepared) {
                Matrix matrix{(*prepared)[index],30,rows,cols,uint64_t(rows)*cols};
                matmul(matrix,input,output,nt,w);return;
            }
            if(mixed_fast && options.expert_direct_gemv && nt<=8 && expert_mixed_type(type)) {
                gpu_check(expert_mixed_projection_gpu(packed,type,input,output,rows,cols,nt,nullptr,stream.value));return;
            }
            if(options.arithmetic==Arithmetic::production && options.expert_direct_gemv && nt<=8 && packed_supported) {
                gpu_check(packed_bf16_gemv_gpu(packed,type,input,output,rows,cols,nt,stream.value));
                round(output,int64_t(nt)*rows);return;
            }
            if(options.arithmetic==Arithmetic::f32 && nt<=8 && packed_supported) {
                gpu_check(packed_gemv_gpu(packed,type,input,output,rows,cols,nt,stream.value));return;
            }
            Matrix matrix;matrix.rows=rows;matrix.cols=cols;matrix.elements=int64_t(rows)*cols;
            if(options.arithmetic==Arithmetic::production){
                if(options.expert_bf16)gpu_check((mixed_fast&&expert_mixed_type(type)?expert_mixed_dequant_gpu:dequant_bf16_gpu)(
                    packed,type,bf,matrix.elements,stream.value));
                else {
                    gpu_check(dequant_gpu(packed,type,w.dequant,matrix.elements,stream.value));
                    gpu_check(float_to_bf16_gpu(w.dequant,bf,matrix.elements,stream.value));
                }
                matrix.data=bf;matrix.type=30;
            }else {
                gpu_check(dequant_gpu(packed,type,w.dequant,matrix.elements,stream.value));matrix.data=w.dequant;
            }
            matmul(matrix,input,output,nt,w);
        };
        if(mixed_fast && options.expert_direct_gemv && !prepared &&
           expert_mixed_gu_activation_selected(f.gu_type,nt) &&
           (expert_mixed_type(f.d_type)||(f.d_type==11&&w.dense_input_q8))){
            gpu_check(expert_mixed_gu_activation_gpu(blob,blob+f.up_off,f.gu_type,input,routing,
                w.group_act,2304,5120,nt,stream.value));
            if(expert_mixed_type(f.d_type))
                gpu_check(expert_mixed_projection_gpu(blob+f.down_off,f.d_type,w.group_act,output,
                    5120,2304,nt,slots,stream.value));
            else{
                kernels::native_quantize_q8_1(w.group_act,w.dense_input_q8,2304,nt,stream.value);
                kernels::native_q3_k_mmvq(blob+f.down_off,w.dense_input_q8,w.group_out,2304,5120,nt,stream.value);
                round(w.group_out,int64_t(nt)*5120);
                if(slots)gpu_check(scatter_expert_gpu(w.group_out,slots,output,nt,6,stream.value));
                else gpu_check(cudaMemcpyAsync(output,w.group_out,int64_t(nt)*5120*4,cudaMemcpyDeviceToDevice,stream.value));
            }
            stats.expert_gpu+=nt;return;
        }
        if(mixed_fast && options.expert_direct_gemv && nt<=8 && !prepared && expert_mixed_type(f.gu_type)){
            gpu_check(expert_mixed_gu_gpu(blob,blob+f.up_off,f.gu_type,input,w.group_gate,w.group_up,2304,5120,nt,stream.value));
            projection_index=2;
        }else if(options.expert_gu_mode && nt<=8 && !prepared && (f.gu_type==10 || f.gu_type==11)){
            gpu_check((options.target_verify_batch&&phase==2?expert_gu_verify_gpu:expert_gu_gpu)(blob,f.gu_type,input,w.group_gate,w.group_up,2304,5120,nt,
                options.expert_gu_mode,w.group_act,stream.value));projection_index=2;
        }else{
            projection(blob,2304,5120,w.expert_gate,input,w.group_gate);
            projection(blob+f.up_off,2304,5120,w.expert_up,input,w.group_up);
        }
        if(options.arithmetic==Arithmetic::production&&nt<=8&&(f.d_type==10||f.d_type==11)&&kernels::cpu::deepseek41_down_fusion_enabled()){
            gpu_check((options.target_verify_batch&&phase==2?format_down_verify_gpu:format_down_gpu)(f.d_type,blob+f.down_off,w.group_gate,w.group_up,routing,
                w.group_out,output,slots,5120,2304,nt,w.group_act,size_t(nt)*2304*4,stream.value));
            stats.expert_gpu+=nt;return;
        }
        if(mixed_fast && options.expert_direct_gemv && nt<=8 && !prepared && expert_mixed_type(f.d_type)){
            gpu_check(expert_mixed_down_gpu(blob+f.down_off,f.d_type,w.group_gate,w.group_up,routing,w.group_act,
                output,5120,2304,nt,slots,stream.value));
            stats.expert_gpu+=nt;return;
        }
        if(mixed_fast && expert_mixed_type(f.d_type))
            gpu_check(expert_mixed_swiglu_gpu(w.group_gate,w.group_up,routing,w.group_act,2304,nt,stream.value));
        else{
            gpu_check((options.arithmetic==Arithmetic::production?swiglu_bf16_gpu:swiglu_gpu)(w.group_gate,w.group_up,routing,w.group_act,2304,nt,10.f,stream.value));
        }
        projection(blob+f.down_off,5120,2304,w.expert_down,w.group_act,single?output:w.group_out);
        if(!single)gpu_check(scatter_expert_gpu(w.group_out,slots,output,nt,6,stream.value));stats.expert_gpu+=nt;
    };
    const bool grouped_verify=grouped_down||(options.target_verify_batch&&phase==2&&tokens>1&&tokens<=8&&
        options.arithmetic==Arithmetic::production&&options.expert_gu_mode&&
        (f.gu_type==10||f.gu_type==11)&&
        ((f.d_type==10||f.d_type==11)&&kernels::cpu::deepseek41_down_fusion_enabled()||
         mixed_fast&&expert_mixed_type(f.d_type))&&w.max_tokens>=tokens*6);
    const bool short_uniform=std::all_of(resident.begin(),resident.end(),[&](const auto& item){return item.first->rows.size()==static_cast<size_t>(tokens);});
    const bool short_batch=short_possible&&expert_mixed_short_selected(f.gu_type,f.d_type,tokens,resident.size(),short_uniform);
    if(short_batch){
        if(single){
            gpu_check(cudaMemcpyAsync(w.group_rows,w.host_group_rows,metadata_offset*sizeof(int),cudaMemcpyHostToDevice,stream.value));
            gpu_check(cudaMemcpyAsync(w.group_slots,w.host_group_slots,metadata_offset*sizeof(int),cudaMemcpyHostToDevice,stream.value));
            gpu_check(cudaMemcpyAsync(w.group_weights,w.host_group_weights,metadata_offset*sizeof(float),cudaMemcpyHostToDevice,stream.value));
        }
        std::array<VerifyExpertGroup,48> batch{};int count=0;
        for(const auto& item:resident){const auto& g=*item.first;
            if(count==48||g.offset+g.rows.size()>static_cast<size_t>(w.max_tokens))throw std::logic_error("IQ short group workspace");
            batch[count++]={item.second,g.offset,static_cast<int>(g.rows.size())};
        }
        gpu_check(expert_mixed_short_gpu(batch.data(),count,f.gu_type,f.d_type,f.up_off,f.down_off,
            w.x,w.group_rows,w.group_weights,w.group_slots,w.group_gate,w.group_up,w.group_act,w.parts,tokens,stream.value));
        for(const auto& item:resident)stats.expert_gpu+=item.first->rows.size();
    }else if(grouped_verify&&!resident.empty()){
        // All GU inputs refer to the original token rows, so quantize once per layer.
        // Group outputs occupy disjoint row spans until every resident Down has consumed them.
        std::array<VerifyExpertGroup,48> batch{};int count=0;
        for(const auto& item:resident){const auto& g=*item.first;
            if(count==48||g.offset+g.rows.size()>static_cast<size_t>(w.max_tokens))throw std::logic_error("verify group workspace");
            batch[count++]={item.second,g.offset,static_cast<int>(g.rows.size())};
        }
        const bool fused_activation=grouped_down&&fusion==2&&f.gu_type==10&&f.d_type==10&&options.expert_gu_mode==2;
        if(fused_activation)gpu_check(expert_q2_activation_grouped_gpu(batch.data(),count,w.x,w.group_rows,w.group_weights,
            reinterpret_cast<uint16_t*>(w.group_act),tokens,w.group_x,stream.value));
        else gpu_check(expert_gu_grouped_verify_gpu(batch.data(),count,f.gu_type,w.x,w.group_rows,
            w.group_gate,w.group_up,tokens,options.expert_gu_mode,w.group_x,stream.value));
        if(grouped_down){
            for(int i=0;i<count;++i)batch[i].packed=static_cast<const uint8_t*>(batch[i].packed)+f.down_off;
            gpu_check(format_down_grouped_gpu(batch.data(),count,f.d_type,w.group_gate,w.group_up,w.group_weights,
                w.parts,w.group_slots,w.group_act,size_t(w.max_tokens)*2304*4,fused_activation,stream.value));
        }else for(const auto& item:resident){const auto& g=*item.first;const int nt=g.rows.size();
            if(mixed_fast&&expert_mixed_type(f.d_type))
                gpu_check(expert_mixed_down_gpu(item.second+f.down_off,f.d_type,
                    w.group_gate+int64_t(g.offset)*2304,w.group_up+int64_t(g.offset)*2304,w.group_weights+g.offset,
                    w.group_act,w.parts,5120,2304,nt,w.group_slots+g.offset,stream.value));
            else gpu_check(format_down_verify_gpu(f.d_type,item.second+f.down_off,
                    w.group_gate+int64_t(g.offset)*2304,w.group_up+int64_t(g.offset)*2304,w.group_weights+g.offset,
                    w.group_out,w.parts,w.group_slots+g.offset,5120,2304,nt,w.group_act,size_t(nt)*2304*4,stream.value));
        }
        for(const auto& item:resident)stats.expert_gpu+=item.first->rows.size();
    }else if(bounded){
        std::array<BoundedMoeGroup,BOUNDED_MOE_BATCH> batch{};int count=0,rows=0;
        auto flush=[&]{if(count)run_batch(batch.data(),count);count=rows=0;};
        for(const auto& item:resident){const auto& g=*item.first;
            if(g.rows.size()<=8){flush();compute(g,item.second);continue;}
            if(count==bounded_capacity || rows+g.rows.size()>static_cast<size_t>(w.max_tokens))flush();
            batch[count]=batch_group(g,item.second,count);++count;rows+=g.rows.size();
        }
        flush();
    }else for(const auto& item:resident)compute(*item.first,item.second);
    if(issued && bounded){
        std::array<BoundedMoeGroup,BOUNDED_MOE_BATCH> batch{};int count=0,rows=0;
        auto flush=[&]{if(count)run_batch(batch.data(),count,true);count=rows=0;};
        for(size_t i=0;i<cold.size();++i){const auto& g=*cold[i];
            if(g.rows.size()<=8 || count==bounded_capacity || rows+g.rows.size()>static_cast<size_t>(w.max_tokens))flush();
            size_t sequence=i;
            if(prefetch_active){sequence=prefetch_positions[layer][g.expert];retire_expert_prefetch(sequence,stats);}
            const uint8_t* blob;
            {prefill::LedgerScope wait_trace(options.trace,"ds41.expert.issuer.wait");blob=expert_pipeline->wait(sequence,stream.value);}
            if(g.rows.size()<=8)compute(g,blob);
            else{
                batch[count]=batch_group(g,blob,count);
                const int rc=bounded_moe_prepare_gpu(batch[count],f.gu_type,f.d_type,stream.value);
                if(rc)throw std::runtime_error("DeepSeek bounded dequant error="+std::to_string(rc));
                rows+=g.rows.size();++count;
            }
            // Return each raw lease after DQ, before requesting the next one.
            // This stays bounded even when a ring has fewer slots than a batch.
            expert_pipeline->release(sequence,stream.value);
            if(prefetch_active)prefetch_cursor=sequence+1;
        }
        flush();
        if(prefetch_active)retire_expert_prefetch(prefetch_offsets[layer+1],stats);
        else expert_pipeline->finish();
    }else if(issued) {
        // The first auxiliary dequant waits for resident readers of the shared
        // BF16 buffers. Subsequent two-buffer reuse waits for that buffer only.
        gpu_check(cudaEventRecord(expert_begin.value,stream.value));
        gpu_check(cudaStreamWaitEvent(dequant_stream.value,expert_begin.value,0));
        std::array<bool,2> used{};
        for(size_t i=0;i<cold.size();++i) {
            const int bank=i%2,nt=cold[i]->rows.size();
            size_t sequence=i;
            if(prefetch_active){
                sequence=prefetch_positions[layer][cold[i]->expert];
                retire_expert_prefetch(sequence,stats);
            }
            const bool expanded=options.expert_pipeline && options.arithmetic==Arithmetic::production &&
                !(nt<=8 && (options.expert_direct_gemv || kernels::cpu::deepseek41_down_fusion_enabled()));
            if(expanded) {
                auto bf=bank?w.expert_ahead:std::array<uint16_t*,3>{w.expert_gate,w.expert_up,w.expert_down};
                if(used[bank])gpu_check(cudaStreamWaitEvent(dequant_stream.value,expert_used[bank].value,0));
                const uint8_t* blob;
                {prefill::LedgerScope wait_trace(options.trace,"ds41.expert.issuer.wait");blob=expert_pipeline->wait(sequence,dequant_stream.value);}
                const uint64_t offsets[3]={0,f.up_off,f.down_off};
                for(int p=0;p<3;++p) {
                    const auto type=p==2?f.d_type:f.gu_type;
                    if(options.expert_bf16)gpu_check((mixed_fast&&expert_mixed_type(type)?expert_mixed_dequant_gpu:dequant_bf16_gpu)(
                        blob+offsets[p],type,bf[p],5120*2304,dequant_stream.value));
                    else {
                        gpu_check(dequant_gpu(blob+offsets[p],p==2?f.d_type:f.gu_type,w.dequant,5120*2304,dequant_stream.value));
                        gpu_check(float_to_bf16_gpu(w.dequant,bf[p],5120*2304,dequant_stream.value));
                    }
                }
                // Fully expanded path has no further raw readers.
                expert_pipeline->release(sequence,dequant_stream.value);
                gpu_check(cudaEventRecord(expert_ready[bank].value,dequant_stream.value));
                gpu_check(cudaStreamWaitEvent(stream.value,expert_ready[bank].value,0));
                compute(*cold[i],blob,&bf);
                gpu_check(cudaEventRecord(expert_used[bank].value,stream.value));used[bank]=true;
            } else {
                const uint8_t* blob;
                {prefill::LedgerScope wait_trace(options.trace,"ds41.expert.issuer.wait");blob=expert_pipeline->wait(sequence,stream.value);}
                if(cache_active){
                    prefill::LedgerScope retain_trace(options.trace,"ds41.expert.rawcache.retain");
                    blob=decode_raw_cache->retain(layer*384+cold[i]->expert,blob,f.bytes);stats.raw_cache_d2d_bytes+=f.bytes;
                }
                compute(*cold[i],blob);
                // Direct GEMV/fused-down still read packed weights through compute.
                expert_pipeline->release(sequence,stream.value);
            }
            if(prefetch_active)prefetch_cursor=sequence+1;
        }
        if(prefetch_active)retire_expert_prefetch(prefetch_offsets[layer+1],stats);
        else expert_pipeline->finish();
    }else if(!cold.empty()) {
        const int bank_slots=ring_slots/2;
        const size_t batches=(cold.size()+bank_slots-1)/bank_slots;
        auto prepare=[&](size_t batch){
            const int bank=batch%2;
            if(ring_used[bank]) {
                // Host memory can be rewritten after DMA; device slots wait for their last consumer.
                gpu_check(cudaEventSynchronize(ring_ready[bank].value));
                gpu_check(cudaStreamWaitEvent(upload.value,ring_consumed[bank].value,0));
            }
            const size_t first=batch*bank_slots,last=std::min(cold.size(),first+bank_slots);
            for(size_t i=first;i<last;++i){
                const uint64_t offset=(uint64_t(bank)*bank_slots+i-first)*ring_slot_bytes;
                uint8_t* host=host_ring->data+offset;
                if(experts->transient(layer,cold[i]->expert))experts->copy_blob(layer,cold[i]->expert,host);
                else std::memcpy(host,experts->blob_stable(layer,cold[i]->expert),f.bytes);
                gpu_check(cudaMemcpyAsync(static_cast<uint8_t*>(ring->base())+offset,host,f.bytes,cudaMemcpyHostToDevice,upload.value));
                stats.h2d_expert_bytes+=f.bytes;
                if(options.arithmetic==Arithmetic::f32)experts->release_mapped(layer,cold[i]->expert);
            }
            gpu_check(cudaEventRecord(ring_ready[bank].value,upload.value));ring_used[bank]=true;
        };
        prepare(0);
        for(size_t batch=0;batch<batches;++batch){
            const int bank=batch%2;
            gpu_check(cudaStreamWaitEvent(stream.value,ring_ready[bank].value,0));
            const size_t first=batch*bank_slots,last=std::min(cold.size(),first+bank_slots);
            for(size_t i=first;i<last;++i){
                const uint64_t offset=(uint64_t(bank)*bank_slots+i-first)*ring_slot_bytes;
                compute(*cold[i],static_cast<const uint8_t*>(ring->base())+offset);
            }
            gpu_check(cudaEventRecord(ring_consumed[bank].value,stream.value));
            if(batch+1<batches)prepare(batch+1);
        }
    }
    if(!cpu.empty()) {
        prefill::LedgerScope cpu_trace(options.trace,"ds41.cpu.experts");
        // One pool batch has at most eight tokens per expert and 96 independent experts.
        struct CpuJob { kernels::cpu::ExpertJobMulti job; std::vector<uint8_t> act,owned; std::vector<float> weights; };
        std::vector<CpuJob> jobs;
        auto drain=[&]{
            std::vector<kernels::cpu::ExpertJobMulti> native;native.reserve(jobs.size());
            for(auto& item:jobs)native.push_back(item.job);
            pool->run_split_multi_native(f,native.data(),native.size());
            for(auto& item:jobs)for(int t=0;t<item.job.nt;++t){
                float* output=item.job.out[t];
                for(int d=0;d<5120;++d)output[d]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(output[d]));
            }
            jobs.clear();
        };
        for(const Group* group:cpu)for(size_t begin=0;begin<group->rows.size();begin+=kernels::cpu::MAXT){
            if(jobs.size()==kernels::cpu::ExpertPool::kMaxSplitMulti)drain();
            CpuJob item;item.job.nt=std::min<size_t>(kernels::cpu::MAXT,group->rows.size()-begin);
            if(experts->transient(layer,group->expert)) { item.owned.resize(f.bytes);experts->copy_blob(layer,group->expert,item.owned.data());item.job.blob=item.owned.data(); }
            else item.job.blob=experts->blob_stable(layer,group->expert);
            item.act.resize(item.job.nt*f.act_bytes);item.weights.resize(item.job.nt);
            for(int t=0;t<item.job.nt;++t){
                const int row=group->rows[begin+t],slot=group->slots[begin+t];
                kernels::cpu::native_quant_act(f,w.host_x+int64_t(row)*5120,item.act.data()+t*f.act_bytes);
                item.job.nact[t]=item.act.data()+t*f.act_bytes;item.job.out[t]=w.host_parts+int64_t(slot)*5120;
                item.weights[t]=group->weights[begin+t];
            }
            item.job.ff_weights=item.weights.data();stats.expert_cpu+=item.job.nt;jobs.push_back(std::move(item));
        }
        if(!jobs.empty())drain();
        for(const Group* group:cpu)for(int slot:group->slots)
            gpu_check(cudaMemcpyAsync(w.parts+int64_t(slot)*5120,w.host_parts+int64_t(slot)*5120,5120*4,cudaMemcpyHostToDevice,stream.value));
    }
    gpu_check(combine_experts_gpu(w.parts,w.route_ids,w.shared_out,w.branch,tokens,6,stream.value));round(w.branch,int64_t(tokens)*5120);
}
} // namespace strata::models::deepseek41
