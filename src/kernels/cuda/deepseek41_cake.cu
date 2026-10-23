// Experimental adapter. Upstream generated kernels remain unmodified and Apache-2.0.
#include "strata/models/deepseek41/attention.hpp"
#include "cake/cake_sparse_mla_dsv41_mixed_kernels.h"
#include <cmath>
#include <climits>

using strata::models::deepseek41::PackedAttention;
namespace cake = cake_sm120_dsv41_mixed;
namespace {
__global__ void convert_slots(const int64_t* swa, const int64_t* global, int* out,
                             int n, int top, int g, int64_t swa_rows, int64_t global_rows) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x, total=n*(top+g);
    if(i>=total)return;
    const bool main=i<n*top;
    const int64_t x=main?swa[i]:global[i-n*top];
    const int64_t limit=main?swa_rows:global_rows;
    out[i]=x>=0&&x<limit&&x<=INT_MAX?int(x):-1;
}
}
namespace strata::models::deepseek41 {
static cudaError_t cake_attention(const PackedAttention& params, int* scratch, size_t bytes,
                                      cudaStream_t stream, bool draft) {
    const auto* p=&params;
    if(!p||!scratch||p->tokens<1||p->heads!=64||
       (draft?((p->tokens!=5&&p->tokens!=7)||p->swa_topk<1||p->swa_topk>135||p->global_topk):p->swa_topk!=128)||
       (p->global_topk!=0&&p->global_topk!=512)||p->q_token_stride!=32768||
       p->q_head_stride!=512||p->grouped_output||!p->q||!p->swa.data||
       !p->swa_slots||!p->sinks||!p->output||!p->partial||!p->lse||
       p->swa.row_stride!=528||p->swa.page_stride!=int64_t(p->swa.page_size)*528||
       (p->global_topk&&(!p->global.data||!p->global_slots||p->global.row_stride!=288||
         p->global.page_stride!=int64_t(p->global.page_size)*288))||
       bytes<size_t(p->tokens)*(p->swa_topk+p->global_topk+64)*4)return cudaErrorInvalidValue;
    const int n=p->tokens,top=p->swa_topk,g=p->global_topk;
    // Only short rows split, within the original 8*8*64*512 BF16 partial arena.
    const int chunks=(top+63)/64+g/64,cpb=draft?chunks:(n<=8?(g?2:1):16),splits=(chunks+cpb-1)/cpb;
    convert_slots<<<(n*(top+g)+255)/256,256,0,stream>>>(p->swa_slots,p->global_slots,
        scratch,n,top,g,p->swa.bytes/528,p->global.bytes/288);
    auto e=cudaGetLastError();if(e!=cudaSuccess)return e;
    cake::DecodeArgs a{};
    a.q=reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(p->q));
    a.kv_cache=p->swa.data;a.extra_kv_cache=g?p->global.data:p->swa.data;
    a.indices=scratch;a.extra_indices=g?scratch+n*top:scratch;
    a.attn_sink=const_cast<float*>(p->sinks);
    a.output=reinterpret_cast<__nv_bfloat16*>(p->output);
    a.out_lse=reinterpret_cast<float*>(scratch+n*(top+g));
    a.mid_out=reinterpret_cast<__nv_bfloat16*>(p->partial);a.mid_lse=p->lse;
    a.topk=top;a.extra_topk=g;a.page_size=1;a.page_stride_bytes=528;
    a.extra_page_size=1;a.extra_page_stride_bytes=288;
    a.scratch_split_stride=splits;a.chunks_per_block=cpb;
    a.sm_scale=1.f/std::sqrt(512.f);a.lse_scale=1.f;a.write_direct=splits==1;a.hb_major=1;
    const bool tiny=n==1;
    const dim3 grid(tiny?4:2,n,splits);
    if(!g)e=tiny?cake::h64::decode_sa_pw_lq_t1::launch(a,grid,stream):
        cake::h64::decode_default::launch(a,grid,stream);
    else if(tiny)e=cake::h64::decode_dual_sa_pw_lq_t1_el_ew_ra::launch(a,grid,stream);
    else if(n<=8)e=cake::h64::decode_dual_sa_tl_ip_el_e2m1_nomul::launch(a,grid,stream);
    else e=cake::h64::decode_dual_sa_ip_pw_el_e2m1_nomul::launch(a,grid,stream);
    if(e!=cudaSuccess||splits==1)return e;
    cake::MergeArgs m{a.mid_out,a.mid_lse,a.attn_sink,a.output,a.out_lse,
                      splits,splits,1.f,1};
    return g&&n>1?cake::h64::merge_mo2::launch(m,dim3(64,n),false,stream):
        cake::h64::merge_default::launch(m,dim3(64,n),false,stream);
}

cudaError_t packed_attention_cake_gpu(const PackedAttention& p,int32_t* scratch,size_t bytes,cudaStream_t stream) {
    return cake_attention(p,scratch,bytes,stream,false);
}
cudaError_t packed_attention_cake_draft_gpu(const PackedAttention& p,int32_t* scratch,size_t bytes,cudaStream_t stream) {
    return cake_attention(p,scratch,bytes,stream,true);
}

} // namespace strata::models::deepseek41
