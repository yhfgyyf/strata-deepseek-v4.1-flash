#include "runtime_internal.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include "expert_pipeline.hpp"
#include "strata/models/deepseek41/integrity.hpp"
#include "strata/models/deepseek41/prefill_chunk.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <cstdlib>

namespace strata::models::deepseek41 {
namespace {
const std::string& verified_path(const std::string& path,const RuntimeOptions& o,
                                 const std::string& draft,const std::string& vision) {
    require_integrity(o.integrity_file,"main",path);
    if(!draft.empty())require_integrity(o.integrity_file,"draft",draft,o.draft_sha256);
    if(!vision.empty())require_integrity(o.integrity_file,"vision",vision,o.vision_sha256);
    return path;
}
}
void Workspace::layout(WorkspaceLayout& memory,int tokens,int context,bool compact_logits,bool compact_workspace,bool dspark) {
    max_tokens=tokens;max_rows=context;max_logit_tokens=compact_logits?std::min(tokens,128):tokens;
    max_bf_input_cols=compact_workspace?(dspark?15360:8192):32768;
    auto floats = [&](uint64_t n) { return memory.alloc<float>(n); };
    auto bytes = [&](uint64_t n) { return memory.alloc<uint8_t>(n); };
    const uint64_t t = tokens, r = context, b = (r + 7) / 8;
    h=floats(t*20480);pre=floats(t*4);attn_pre=floats(t*4);post=floats(t*4);comb=floats(t*16);mixes=floats(t*24);
    x=floats(t*5120);branch=floats(t*5120);qr=floats(t*1280);q=floats(t*32768);kv=floats(t*512);latent=floats(t*512);
    comp_raw=floats(t*512);comp_gate=floats(t*512);index_q=floats(t*4096);index_w=floats(t*32);index_k=floats(t*128);
    engram_emb=floats(t*6144);engram_kv=floats(t*25600);image_values=floats(t*5120);o_low=floats(t*8192);
    if(!compact_workspace){shared_gate=floats(t*2304);shared_up=floats(t*2304);shared_act=floats(t*2304);}
    shared_out=floats(t*5120);
    parts=compact_workspace?q:floats(t*6*5120);group_x=compact_workspace?o_low:floats(t*5120);
    if(!compact_workspace){group_gate=floats(t*2304);group_up=floats(t*2304);group_act=floats(t*2304);group_out=floats(t*5120);}
    logits=floats(uint64_t(max_logit_tokens)*129280);dequant=floats(5120*2304);
    expert_gate=memory.alloc<uint16_t>(5120*2304);expert_up=memory.alloc<uint16_t>(5120*2304);expert_down=memory.alloc<uint16_t>(5120*2304);
    bf_input=memory.alloc<uint16_t>(t*max_bf_input_cols);q_bf=memory.alloc<uint16_t>(t*32768);kv_bf=memory.alloc<uint16_t>(t*512);
    out_bf=memory.alloc<uint16_t>(t*32768);
    // Up to eight verification queries, each retaining decode's eight splits.
    partial_out=memory.alloc<uint16_t>(8*8*32768);lse=floats(std::max<uint64_t>(t,64)*64);
    if(compact_workspace && memory.arena) {
        // Attention and MoE execute in order on the main stream. Engram and
        // ahead-dequant buffers are private and deliberately excluded here.
        shared_gate=reinterpret_cast<float*>(out_bf);shared_up=shared_gate+t*2304;shared_act=shared_up+t*2304;
        group_gate=reinterpret_cast<float*>(q_bf);group_up=group_gate+t*2304;group_act=group_up+t*2304;group_out=group_act+t*2304;
        static_assert(6*5120<=32768 && 3*2304+5120<=32768/2,"phase aliases fit attention storage");
    }
    engram_packed=bytes(t*24*272);temp_swa=bytes(((t+128+31)/32)*32*528);temp_swa_float=floats((t+128)*512);
    image=bytes(t);
    ids=memory.alloc<int>(t);group_rows=memory.alloc<int>(t*6);group_slots=memory.alloc<int>(t*6);
    route_ids=memory.alloc<int>(t*6);route_weights=floats(t*6);group_weights=floats(t*6);
    route_handoff=memory.alloc<int>(8*6+1);
    swa_slots=memory.alloc<int64_t>(t*128);global_slots=memory.alloc<int64_t>(t*512);write_slots=memory.alloc<int64_t>(t);
    // Bound index queries independently of the model chunk. Candidate readers use only 2048*8 keys.
    const uint64_t index_begin=memory.bytes;
    index_elements=std::min<uint64_t>(t*std::max<uint64_t>(r,16384),(32ull<<20)/sizeof(float));
    index_max_queries=static_cast<int>(std::min<uint64_t>(t,index_elements/16384));
    index_mx=bytes(uint64_t(index_max_queries)*2304);
    const uint64_t block_elements=std::min<uint64_t>(t*b,(index_elements+7)/8+index_max_queries);
    candidate_blocks=memory.alloc<int32_t>(t*2048);
    scores=floats(index_elements);block_scores=floats(block_elements);
    sort.sorted_scores=floats(index_elements);sort.input_ids=memory.alloc<int>(index_elements);sort.sorted_ids=memory.alloc<int>(index_elements);
    sort.offsets=memory.alloc<int64_t>(index_max_queries+1);
    gpu_check(sort_index_gpu(nullptr,1,static_cast<int>(index_elements),sort,nullptr));
    sort_capacity=sort.temporary_bytes;sort.temporary=bytes(sort_capacity);
    block_sort.sorted_scores=floats(block_elements);block_sort.input_ids=memory.alloc<int>(block_elements);block_sort.sorted_ids=memory.alloc<int>(block_elements);
    block_sort.offsets=memory.alloc<int64_t>(index_max_queries+1);
    gpu_check(sort_index_gpu(nullptr,1,static_cast<int>(block_elements),block_sort,nullptr));
    block_sort_capacity=block_sort.temporary_bytes;block_sort.temporary=bytes(block_sort_capacity);
    index_gpu_bytes=memory.bytes-index_begin;
}
Workspace::Workspace(uint64_t budget,int tokens,int context,bool compact_logits,bool compact_workspace,bool dspark) {
    arena=std::make_unique<core::DeviceArena>(budget);
    WorkspaceLayout memory(arena.get());layout(memory,tokens,context,compact_logits,compact_workspace,dspark);
    // Pinned exchanges are request-independent and reused while the execution mutex owns this workspace.
    const uint64_t t=tokens,host_bytes=t*5120*4+t*6*5120*4+t*6*4*5+sizeof(int);
    host=std::make_unique<Pinned>(host_bytes);
    host_x=reinterpret_cast<float*>(host->data);
    host_parts=host_x+t*5120;host_weights=host_parts+t*6*5120;host_ids=reinterpret_cast<int*>(host_weights+t*6);
    host_group_rows=host_ids+t*6+1;host_group_slots=host_group_rows+t*6;
    host_group_weights=reinterpret_cast<float*>(host_group_slots+t*6);
    std::cerr << "DeepSeek workspace used=" << arena->used() << " reserved=" << arena->capacity() << '\n';
}
void Workspace::extras(WorkspaceLayout& memory,const RuntimeOptions& options) {
    if(options.dense_q8){
        dense_weight=memory.alloc<uint16_t>(8192ull*5120);
        dense_input_q8=memory.alloc<uint8_t>(8ull*8192/32*36);
    }
    if(options.engram_pipeline)engram_input=memory.alloc<uint16_t>(uint64_t(max_tokens)*6144);
    if(options.expert_pipeline)for(auto& bf:expert_ahead)bf=memory.alloc<uint16_t>(5120*2304);
}
uint64_t Workspace::bytes_needed(int tokens,const RuntimeOptions& options,bool vision) {
    Workspace work;WorkspaceLayout memory;
    work.layout(memory,std::max(128,tokens),options.context,options.compact_logits,options.compact_workspace,options.dspark);
    work.extras(memory,options);
    if(vision){VisionWork extra(memory);}
    if(options.dspark){DraftWork extra(memory,DraftVocab(options.draft_vocab).size());}
    return memory.bytes;
}
Runtime::Impl::Impl(const std::string& path,const RuntimeOptions& o,const std::string& draft,const std::string& vision)
    : model(verified_path(path,o,draft,vision),draft,vision), options(o) {
    if (o.context < 1 || o.context > model.geometry().context || o.chunk < 1 || o.chunk > PREFILL_CHUNK_MAX || o.slots < 1 || o.slots > 2)
        throw std::invalid_argument("DeepSeek context/chunk/slots bounds");
    if (o.arithmetic == Arithmetic::f32 && o.experts != ExpertMode::gpu)
        throw std::invalid_argument("F32 oracle arithmetic requires GPU streaming experts without activation quantization");
    if (o.hot_auto && (o.arithmetic == Arithmetic::f32 || o.profile.empty()))
        throw std::invalid_argument("DeepSeek hot-auto requires production arithmetic and a profile");
    if (!o.work || !o.state) throw std::invalid_argument("DeepSeek work/state budgets must be positive");
    if(o.dspark&&(o.dspark_draft_tokens!=5&&o.dspark_draft_tokens!=7))throw std::invalid_argument("DSpark supports 5 or 7 draft tokens");
    if(o.dspark&&!model.has_draft())throw std::invalid_argument("DSpark requires the validated draft GGUF");
    if(o.phase_residency&&o.slots!=1)throw std::invalid_argument("experimental phase residency requires one request slot");
    if(o.adaptive_experts&&(o.phase_residency||o.slots!=1||o.arithmetic!=Arithmetic::production||o.experts!=ExpertMode::hybrid))
        throw std::invalid_argument("adaptive experts require single-slot production hybrid, without phase residency");
    if((o.adaptive_groups||o.adaptive_maya||!o.expert_profile_save.empty())&&!o.adaptive_experts)
        throw std::invalid_argument("grouped residency/profile learning requires adaptive experts");
    if(!o.expert_profile_save.empty()&&!o.profile.empty()&&
       std::filesystem::weakly_canonical(o.expert_profile_save)==std::filesystem::weakly_canonical(o.profile))
        throw std::invalid_argument("learned profile output must differ from the startup profile");
    if(!std::isfinite(o.pcie_frac) || o.pcie_frac<0 || o.pcie_frac>1)throw std::invalid_argument("DeepSeek pcie-frac bounds");
    if((o.expert_pipeline || o.pcie_frac>0) && (o.phase_residency || o.slots!=1))
        throw std::invalid_argument("expert issuer requires one slot without phase residency");
    if(o.pcie_frac>0 && (o.arithmetic!=Arithmetic::production || o.experts!=ExpertMode::hybrid || !o.expert_direct_gemv))
        throw std::invalid_argument("PCIe offload requires production hybrid direct BF16 GEMV");
    const auto info = core::device_info();
    if (info.cc_major < 8) throw std::runtime_error("DeepSeek native attention requires CUDA SM80 or later");
    struct BlasGuard {cublasHandle_t& handle;bool success=false;~BlasGuard(){if(!success&&handle)cublasDestroy(handle);}} guard{blas};
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas,stream.value) != CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(blas,static_cast<cublasMath_t>(CUBLAS_PEDANTIC_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("DeepSeek cuBLAS initialization");
    if(o.dense_q8_1 && !o.dense_q8)throw std::invalid_argument("Q8_1 dense inputs require packed dense weights");
    if(o.expert_gu_mode<0 || o.expert_gu_mode>2)throw std::invalid_argument("expert GU mode must be 0, 1 or 2");
    if(o.decode_raw_cache && (o.pcie_frac!=1 || !o.expert_pipeline || o.slots!=1))
        throw std::invalid_argument("decode raw cache requires single-slot GPU-offload issuer");
    if((o.pinned_experts || o.prefill_expert_lookahead) &&
       (!o.expert_pipeline || o.pcie_frac!=1 || o.slots!=1 || o.phase_residency || o.arithmetic!=Arithmetic::production))
        throw std::invalid_argument("pinned/prefetch experts require the single-slot production GPU issuer");
    if(o.prefill_expert_lookahead && !o.pinned_experts)throw std::invalid_argument("expert lookahead requires pinned sources");
    if((o.dense_q8 || o.expert_gu_mode || o.cpu_gu_q8_0 || o.decode_graph) && (o.arithmetic!=Arithmetic::production))
        throw std::invalid_argument("quantized/graph experiments require production arithmetic");
    if(o.chunk_auto){
        uint64_t needed=0;
        options.chunk=select_prefill_chunk(o.chunk,o.context,o.work,[&](int tokens){
            needed=Workspace::bytes_needed(tokens,o,model.has_vision());return needed;
        });
        std::cerr<<"DeepSeek prefill auto: chunk="<<options.chunk<<" ceiling="<<o.chunk
                 <<" context="<<o.context<<" workspace="<<needed<<" budget="<<o.work<<'\n';
    }
    weights=std::make_unique<WeightBank>(model,o.arithmetic,stream.value,false,false,o.dense_q8);
    if(model.has_vision())vision_weights=std::make_unique<WeightBank>(model,Arithmetic::f32,stream.value,false,true);
    if(o.dspark){
        draft_weights=std::make_unique<DraftWeights>(model,o.arithmetic,stream.value,o.draft_vocab,o.draft_input_vocab);
        if(draft_weights->bytes()>(11ull<<30))throw std::runtime_error("DSpark weights exceed the 11 GiB draft region");
    }
    const auto expert_sizes=model.expert_bytes();
    ring_slot_bytes=*std::max_element(expert_sizes.begin(),expert_sizes.end());
    if (!ring_slot_bytes) throw std::invalid_argument("DeepSeek zero expert ring slot bytes");
    // Use the actual even-slot ring capacity in the fit check, rather than its rounded-up budget.
    ring_slots=static_cast<int>(std::min<uint64_t>(128,o.ring/ring_slot_bytes/2*2));
    if (ring_slots < 2) throw std::invalid_argument("DeepSeek ring cannot hold two expert blobs");
    const uint64_t ring_bytes=uint64_t(ring_slots)*ring_slot_bytes;
    size_t free=0,total=0;gpu_check(cudaMemGetInfo(&free,&total));
    const uint64_t capacity=available_hot_expert_bytes(free,ring_bytes,o.work,o.state,o.gpu_headroom);
    if (o.hot_auto) { options.hot_front=capacity; options.hot_back=0; }
    if (options.hot_front > UINT64_MAX-options.hot_back) throw std::overflow_error("DeepSeek expert budget overflow");
    const uint64_t wanted_hot=o.arithmetic==Arithmetic::f32?0:options.hot_front+options.hot_back;
    if (wanted_hot > capacity)
        throw std::runtime_error("DeepSeek GPU weight/ring/work/state/headroom budgets do not fit");
    experts=std::make_unique<GgufExperts>(model,options,upload.value);
    engram=std::make_unique<EngramReader>(model,o.engram_cache,o.engram_continuous,o.engram_aio);
    pool=std::make_unique<kernels::cpu::ExpertPool>(o.cpu_workers,true,true);
    // Fixed 64/128-slot byte ring, with two independent DMA/consumption lifetimes.
    ring=std::make_unique<core::DeviceArena>(ring_bytes);
    host_ring=std::make_unique<Pinned>(ring_bytes);
    if(o.expert_pipeline || o.pcie_frac>0)expert_pipeline=std::make_unique<ExpertPipeline>(ring->base(),*host_ring,ring_slot_bytes,ring_slots,upload.value);
    if(o.decode_raw_cache){
        if(ring_slots<=8)throw std::invalid_argument("decode raw cache requires more than eight ring slots");
        decode_raw_cache=std::make_unique<DecodeRawCache>(static_cast<uint8_t*>(ring->base())+8*ring_slot_bytes,ring_slot_bytes,ring_slots-8,stream.value);
    }
    work=std::make_unique<Workspace>(o.work,std::max(128,options.chunk),o.context,o.compact_logits,o.compact_workspace,o.dspark);
    WorkspaceLayout memory(work->arena.get());work->extras(memory,o);
    if(model.has_vision())vision_work=std::make_unique<VisionWork>(*work->arena);
    if(o.dspark)draft_work=std::make_unique<DraftWork>(*work->arena,draft_weights->vocab.size());
    if(const char* mode=std::getenv("STRATA_DS41_DECODE_SEGMENTS")){
        if(mode[0]<'0'||mode[0]>'3'||mode[1])throw std::invalid_argument("decode segment mask must be 0..3");
        decode_segment_mask=static_cast<unsigned>(mode[0]-'0');
    }
    guard.success=true;
}
Runtime::Impl::~Impl() { cudaStreamSynchronize(hc_stream.value);if (blas) cublasDestroy(blas); }
void Runtime::Impl::round(float* x,uint64_t n) {
    if (options.arithmetic == Arithmetic::production) gpu_check(round_bf16_gpu(x,n,stream.value));
}
void matrix_product(const Matrix& m,const float* x,float* out,int tokens,cublasHandle_t blas,cudaStream_t stream,
                    uint16_t* bf_input,Arithmetic arithmetic,int rows,int64_t ldx,int64_t ldy,
                    uint64_t weight_offset,bool round_output,bool verify_batch,int bf_scratch_cols) {
    const int n=rows ? rows : m.rows,k=m.cols;
    if (!ldx) ldx=k;
    if (!ldy) ldy=n;
    if (tokens < 1 || n < 1 || k < 1 || ldx < k || ldy < n ||
        weight_offset + uint64_t(n)*k > m.elements) throw std::invalid_argument("DeepSeek matrix view");
    if(m.type!=0&&m.type!=30&&m.type!=8&&m.type!=14)throw std::invalid_argument("DeepSeek matrix storage type");
    const int block=m.type==14?256:32;
    if((m.type==8||m.type==14) && (weight_offset%block||k%block||tokens>8))throw std::invalid_argument("DeepSeek packed matrix requires a short row window");
    const uint64_t byte_offset=m.type==14?weight_offset/256*210:m.type==8?weight_offset/32*34:weight_offset*(m.type==0?4:2);
    const void* weights=static_cast<const uint8_t*>(m.data)+byte_offset;
    // Match single-token and short verification reductions; strided grouped output uses the BLAS contract.
    if (verify_batch && tokens>1 && tokens<=8) {
        gpu_check(packed_verify_gpu(weights,m.type,x,out,n,k,tokens,ldx,ldy,stream));
    } else if (tokens <= 8) {
        for(int t=0;t<tokens;++t)
            gpu_check(packed_gemv_gpu(weights,m.type,x+int64_t(t)*ldx,out+int64_t(t)*ldy,n,k,1,stream));
    }
    else {
        const float alpha=1,beta=0;
        cublasStatus_t status;
        if (m.type==0) status=cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,tokens,k,&alpha,
            static_cast<const float*>(weights),k,x,ldx,&beta,out,ldy);
        else {
            gpu_check(to_bf16_strided_gpu(x,bf_input,tokens,k,ldx,stream));
            // MoE projection output has an immediate BF16 boundary. Reuse the
            // remaining input arena, with explicit capacity and contiguous output.
            if(round_output && arithmetic==Arithmetic::production && ldy==n && k+n<=bf_scratch_cols &&
               n==5120 && k==2304 && tokens>=256) {
                auto* bf_output=bf_input+int64_t(tokens)*k;
                status=cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,tokens,k,&alpha,weights,CUDA_R_16BF,k,
                    bf_input,CUDA_R_16BF,k,&beta,bf_output,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
                if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("DeepSeek BF16 output multiplication status="+std::to_string(status));
                gpu_check(bf16_to_float_gpu(bf_output,out,int64_t(tokens)*n,stream));
                return;
            }
            status=cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,tokens,k,&alpha,weights,CUDA_R_16BF,k,
                bf_input,CUDA_R_16BF,k,&beta,out,CUDA_R_32F,ldy,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
        }
        if (status!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("DeepSeek matrix multiplication status="+std::to_string(status));
    }
    if (round_output && arithmetic==Arithmetic::production) {
        if (ldy==n) gpu_check(round_bf16_gpu(out,uint64_t(tokens)*n,stream));
        else gpu_check(round_bf16_strided_gpu(out,tokens,n,ldy,stream));
    }
}
void Runtime::Impl::matmul(const Matrix& m,const float* x,float* out,int tokens,Workspace& w,
                          int rows,int64_t ldx,int64_t ldy,uint64_t weight_offset,bool round_output) {
    if(tokens>w.max_tokens)throw std::invalid_argument("DeepSeek matrix exceeds workspace token capacity");
    if(m.type==30 && tokens>8 && m.cols>w.max_bf_input_cols)throw std::invalid_argument("DeepSeek matrix exceeds BF16 input scratch");
    if(m.packed_bf16) {
        const int n=rows?rows:m.rows,k=m.cols;
        const int block=m.type==14?256:32,block_bytes=m.type==14?210:34;
        if(!ldx)ldx=k;if(!ldy)ldy=n;
        if((m.type!=8&&m.type!=14) || n<1 || k<1 || k>8192 || k%block || weight_offset%block || weight_offset+uint64_t(n)*k>m.elements || ldx<k || ldy<n)
            throw std::invalid_argument("DeepSeek packed projection view");
        const auto* packed=static_cast<const uint8_t*>(m.data)+weight_offset/block*block_bytes;
        if(tokens<=8) {
            if(options.dense_q8_1&&options.dense_q8_batch&&tokens>1){
                // Existing exact multi-column kernel shares each loaded weight across rows.
                if(ldx==k)kernels::native_quantize_q8_1(x,w.dense_input_q8,k,tokens,stream.value);
                else for(int t=0;t<tokens;++t)kernels::native_quantize_q8_1(x+int64_t(t)*ldx,
                    w.dense_input_q8+int64_t(t)*k/32*36,k,1,stream.value);
                float* result=ldy==n?out:reinterpret_cast<float*>(w.dense_weight);
                kernels::native_mmvq(m.type,packed,w.dense_input_q8,result,k,n,tokens,stream.value);
                if(ldy!=n)gpu_check(cudaMemcpy2DAsync(out,ldy*4,result,n*4,n*4,tokens,cudaMemcpyDeviceToDevice,stream.value));
            }else for(int t=0;t<tokens;++t) {
                if(options.dense_q8_1){
                    kernels::native_quantize_q8_1(x+int64_t(t)*ldx,w.dense_input_q8,k,1,stream.value);
                    kernels::native_mmvq(m.type,packed,w.dense_input_q8,out+int64_t(t)*ldy,k,n,1,stream.value);
                }else gpu_check(packed_bf16_gemv_gpu(packed,m.type,x+int64_t(t)*ldx,out+int64_t(t)*ldy,n,k,1,stream.value));
            }
            if(round_output)gpu_check(round_bf16_strided_gpu(out,tokens,n,ldy,stream.value));
        }else {
            if(uint64_t(n)*k>8192ull*5120)throw std::invalid_argument("packed projection scratch bound");
            prefill::LedgerScope expand_trace(options.trace,"ds41.dense.expand");
            gpu_check(dequant_bf16_gpu(packed,m.type,w.dense_weight,uint64_t(n)*k,stream.value));
            Matrix expanded{w.dense_weight,30,n,k,uint64_t(n)*k};
            matrix_product(expanded,x,out,tokens,blas,stream.value,w.bf_input,options.arithmetic,0,ldx,ldy,0,round_output,false,w.max_bf_input_cols);
        }
        return;
    }
    if((m.type==8||m.type==14)&&tokens>8){
        const int n=rows?rows:m.rows,k=m.cols;
        const int block=m.type==14?256:32,block_bytes=m.type==14?210:34;
        if(!ldx)ldx=k;
        if(!ldy)ldy=n;
        if(weight_offset%block||k%block)throw std::invalid_argument("DeepSeek packed matrix offset");
        const auto* base=static_cast<const uint8_t*>(m.data)+weight_offset/block*block_bytes;
        for(int begin=0;begin<n;begin+=2048){
            const int count=std::min(2048,n-begin);
            if(int64_t(count)*k>5120*2304)throw std::runtime_error("DeepSeek head dequant workspace bound");
            gpu_check(dequant_gpu(base+int64_t(begin)*k/block*block_bytes,m.type,w.dequant,int64_t(count)*k,stream.value));
            Matrix tile{w.dequant,0,count,k,uint64_t(count)*k};
            matrix_product(tile,x,out+begin,tokens,blas,stream.value,w.bf_input,options.arithmetic,0,ldx,ldy,0,round_output);
        }
        return;
    }
    matrix_product(m,x,out,tokens,blas,stream.value,w.bf_input,options.arithmetic,rows,ldx,ldy,weight_offset,round_output,options.target_verify_batch,w.max_bf_input_cols);
}
Runtime::Runtime(const std::string& main,const RuntimeOptions& options,const std::string& draft,const std::string& vision)
    : impl_(std::make_unique<Impl>(main,options,draft,vision)) {}
Runtime::~Runtime()=default;
const Model& Runtime::model() const { return impl_->model; }
std::unique_ptr<Session> Runtime::create_session() {
    std::lock_guard<std::mutex> lock(impl_->execution_mutex);
    if (impl_->active_sessions >= impl_->options.slots) throw std::runtime_error("DeepSeek session slot limit");
    auto session=std::unique_ptr<Session>(new Session(*impl_));++impl_->active_sessions;return session;
}
void Runtime::write_ledger(std::ostream& out) const {
    const auto& p=*impl_;
    out << "{\"dense_gpu_bytes\":" << p.weights->bytes() << ",\"expert_gpu_bytes\":" << p.experts->gpu_bytes()
        <<",\"prefill_chunk\":"<<p.options.chunk<<",\"prefill_chunk_auto\":"<<(p.options.chunk_auto?"true":"false")
        <<",\"dense_q8\":"<<(p.options.dense_q8?"true":"false")<<",\"dense_q8_1\":"<<(p.options.dense_q8_1?"true":"false")
        <<",\"dense_q8_batch\":"<<(p.options.dense_q8_batch?"true":"false")
        <<",\"target_verify_batch\":"<<(p.options.target_verify_batch?"true":"false")
        <<",\"dspark_prefix_commit\":"<<(p.options.dspark_prefix_commit?"true":"false")
        <<",\"dspark_gpu_greedy\":"<<(p.options.dspark_gpu_greedy?"true":"false")
        <<",\"dspark_incremental_kv\":"<<(p.options.dspark_incremental_kv?"true":"false")
        <<",\"dspark_fused_moe\":"<<(p.options.dspark_fused_moe?"true":"false")
        <<",\"dspark_batch_projection\":"<<(p.options.dspark_batch_projection?"true":"false")
        <<",\"expert_gu_mode\":"<<p.options.expert_gu_mode<<",\"cpu_gu_q8_0\":"<<(p.options.cpu_gu_q8_0?"true":"false")
        <<",\"decode_segment_mask\":"<<p.decode_segment_mask
        <<",\"decode_segment_cache_bytes\":"<<p.work->segment_cache_bytes
        <<",\"decode_segment_captures\":"<<(p.work->segment_captures[0]+p.work->segment_captures[1])
        <<",\"decode_segment_replays\":"<<(p.work->segment_replays[0]+p.work->segment_replays[1])
        <<",\"decode_shared_graph\":"<<(p.options.decode_graph?"true":"false")<<",\"engram_continuous\":"<<(p.options.engram_continuous?"true":"false")
        <<",\"parallel_router\":"<<(p.options.parallel_router?"true":"false")
        <<",\"engram_aio\":"<<(p.options.engram_aio?"true":"false")
        <<",\"decode_raw_cache_slots\":"<<(p.decode_raw_cache?p.decode_raw_cache->slots():0)
        <<",\"pinned_expert_bytes\":"<<p.experts->pinned_bytes()<<",\"huge_pinned_memory\":"<<(p.options.huge_pinned_memory?"true":"false");
    out<<",\"expert_host_memory\":";p.experts->host_memory_info().write(out);
    out<<",\"prefill_expert_lookahead\":"<<(p.options.prefill_expert_lookahead?"true":"false")
        << ",\"expert_ram_bytes\":" << p.experts->ram_bytes() << ",\"ring_gpu_bytes\":" << p.ring->capacity()
        << ",\"ring_host_pinned_bytes\":" << (p.host_ring->pinned?p.host_ring->bytes:0) << ",\"work_gpu_reserved\":" << p.work->arena->capacity()
        << ",\"work_gpu_used\":" << p.work->arena->used() << ",\"logit_workspace_rows\":"<<p.work->max_logit_tokens
        << ",\"bf_input_columns\":"<<p.work->max_bf_input_cols<<",\"compact_workspace\":"<<(p.options.compact_workspace?"true":"false")
        << ",\"exchange_host_pinned_bytes\":" << (p.work->host->pinned?p.work->host->bytes:0)
        << ",\"index_gpu_bytes\":" << p.work->index_gpu_bytes << ",\"index_score_elements\":" << p.work->index_elements
        << ",\"candidate_block_id_bytes\":" << uint64_t(p.work->max_tokens)*2048*sizeof(int32_t)
        << ",\"ring_host_pageable_bytes\":" << (p.host_ring->pinned?0:p.host_ring->bytes)
        << ",\"phase_backup_ram_bytes\":" << p.experts->backup_bytes()<<",\"phase_host_staging_bytes\":"<<p.experts->phase_staging_bytes()
        << ",\"adaptive_experts\":"<<(p.options.adaptive_experts?"true":"false")<<",\"adaptive_host_staging_bytes\":"<<p.experts->adaptive_staging_bytes()
        << ",\"adaptive_expert_scope\":\""<<(p.options.adaptive_groups?"format-group":"layer")<<"\""
        << ",\"adaptive_expert_policy\":\""<<(p.options.adaptive_maya?"maya":"legacy")<<"\""
        << ",\"learned_profile_save_enabled\":"<<(!p.options.expert_profile_save.empty()?"true":"false")
        << ",\"state_gpu_budget\":" << p.options.state << ",\"slots\":" << p.options.slots
        << ",\"vision_gpu_bytes\":" << (p.vision_weights?p.vision_weights->bytes():0)
        << ",\"draft_gpu_bytes\":" << (p.draft_weights?p.draft_weights->bytes():0)
        << ",\"draft_host_bytes\":0,\"draft_host_pinned\":false,\"draft_expert_h2d_bytes\":0"
        << ",\"draft_vocab_size\":" << (p.draft_weights?p.draft_weights->vocab.size():0)
        << ",\"engram_cache_budget\":" << p.options.engram_cache
        << ",\"hot_policy\":\"" << (p.options.arithmetic==Arithmetic::f32?"streaming_f32":p.options.profile.empty()?"legacy_layer_quota":"global_profile_prefix")
        << "\",\"hot_auto\":" << (p.options.hot_auto?"true":"false")
        << ",\"hot_capacity_bytes\":" << (p.options.arithmetic==Arithmetic::f32?0:p.options.hot_front+p.options.hot_back)
        << ",\"gpu_headroom_bytes\":" << p.options.gpu_headroom << ",\"hot_expert_layer_counts\":[";
    std::array<int,40> hot_counts{};for(const auto& pair:p.experts->hot_pairs())++hot_counts[pair.first];
    for(int l=0;l<40;++l){if(l)out<<',';out<<hot_counts[l];}out<<"],\"hot_experts\":[";
    bool comma=false;for(const auto& [l,e]:p.experts->hot_pairs()){if(comma)out<<',';comma=true;out<<'['<<l<<','<<e<<']';}
    out << "],\"hot_slots\":";p.experts->write_layout(out);
    const auto io=p.engram->stats();
    out<<",\"engram_io\":{\"batches\":"<<io.batches<<",\"logical_bytes\":"<<io.logical_bytes<<",\"disk_bytes\":"<<io.disk_bytes
       <<",\"reads\":"<<io.reads<<",\"p95_batch_us\":"<<io.p95_us<<",\"page_hits\":"<<io.page_hits<<",\"cache_owned_bytes\":"<<io.cache_owned_bytes
       <<",\"layer_disk_bytes\":["<<io.layer_disk_bytes[0]<<','<<io.layer_disk_bytes[1]<<"],\"layer_reads\":["<<io.layer_reads[0]<<','<<io.layer_reads[1]
       <<"],\"layer_wall_us\":["<<io.layer_wall_us[0]<<','<<io.layer_wall_us[1]<<"],\"layer_post_ready_cache_us\":["<<io.layer_cache_fill_us[0]<<','<<io.layer_cache_fill_us[1]<<"]}}";
}
void Runtime::write_expert_heat(std::ostream& out)const{impl_->experts->write_heat(out);}
void Runtime::save_learned_profile(){
    if(impl_->options.expert_profile_save.empty())return;
    std::lock_guard<std::mutex> lock(impl_->execution_mutex);
    impl_->experts->save_learned_profile(impl_->options.expert_profile_save,impl_->stream.value);
}
ImageEmbedding Runtime::encode_image(const std::vector<float>& patches,int h,int w,int64_t begin) {
    std::lock_guard<std::mutex> lock(impl_->execution_mutex);
    if(!impl_->vision_weights||!impl_->vision_work)throw std::runtime_error("DeepSeek mmproj was not configured");
    ImageEmbedding image;image.begin=begin;
    image.values=vision_forward(*impl_->vision_weights,*impl_->vision_work,patches,h,w,Arithmetic::f32,
        impl_->blas,impl_->stream.value,impl_->work->bf_input,impl_->work->max_tokens,nullptr);
    return image;
}
Capture::Capture(const std::string& path) {
    if (path.empty()) return;
    root=path;std::filesystem::create_directories(root);
    if (std::filesystem::exists(root/"tensors.jsonl")) throw std::runtime_error("DeepSeek capture destination already has tensors");
    manifest.open(root/"tensors.jsonl");if(!manifest)throw std::runtime_error("DeepSeek capture manifest");
}
void Capture::tokens(const std::vector<int>& ids) {
    if(!enabled())return;
    std::ofstream out(root/"tokens.json");out<<'[';
    for(size_t i=0;i<ids.size();++i){if(i)out<<',';out<<ids[i];}out<<"]\n";
}
void Capture::put(const std::string& name,const float* device,const std::vector<int64_t>& shape,cudaStream_t stream) {
    if(!enabled())return;
    if(!stages.empty() && std::find(stages.begin(),stages.end(),name)==stages.end())return;
    uint64_t count=1;for(auto n:shape){if(n<1||count>UINT64_MAX/uint64_t(n))throw std::overflow_error("DeepSeek capture shape");count*=n;}
    std::vector<float> values(count);gpu_check(cudaMemcpyAsync(values.data(),device,count*4,cudaMemcpyDeviceToHost,stream));
    gpu_check(cudaStreamSynchronize(stream));
    put_host(name,values,shape);
}
void Capture::put_host(const std::string& name,const std::vector<float>& values,const std::vector<int64_t>& shape) {
    if(!enabled())return;
    if(!stages.empty() && std::find(stages.begin(),stages.end(),name)==stages.end())return;
    uint64_t count=1;for(auto n:shape){if(n<1||count>UINT64_MAX/uint64_t(n))throw std::overflow_error("DeepSeek capture shape");count*=n;}
    if(count!=values.size())throw std::invalid_argument("DeepSeek capture values/shape");
    const std::string file="tensor-"+std::to_string(serial++)+".f32";
    std::ofstream binary(root/file,std::ios::binary);binary.write(reinterpret_cast<const char*>(values.data()),count*4);
    if(!binary)throw std::runtime_error("DeepSeek capture write");
    manifest<<"{\"name\":\""<<name<<"\",\"file\":\""<<file<<"\",\"type\":\"f32\",\"ne\":[";
    bool comma=false;for(auto i=shape.rbegin();i!=shape.rend();++i){if(comma)manifest<<',';comma=true;manifest<<*i;}
    manifest<<"],\"nb\":[";uint64_t stride=4;comma=false;
    for(auto i=shape.rbegin();i!=shape.rend();++i){if(comma)manifest<<',';comma=true;manifest<<stride;stride*=*i;}
    manifest<<"]}\n";manifest.flush();
    if(name=="result_output") { std::ofstream logits(root/"logits.f32",std::ios::binary);logits.write(reinterpret_cast<const char*>(values.data()),count*4); }
}
} // namespace strata::models::deepseek41
