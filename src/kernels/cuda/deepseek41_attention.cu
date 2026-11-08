// Native attention headers retain FlashInfer's Apache-2.0 notice and mathematics.
// Strata uses raw pointer/stream bindings and independently validates single-GPU H64.
#include "strata/models/deepseek41/attention.hpp"
#include "strata/models/deepseek41/deepseek_v41_mixed_sparse.cuh"
#include <limits>

namespace strata::models::deepseek41 {
namespace {
bool layout_valid(const CacheLayout& cache, int payload) {
    return cache.data && cache.bytes >= payload && cache.page_size > 0 && cache.row_stride >= payload &&
           cache.row_stride <= std::numeric_limits<int64_t>::max() / cache.page_size &&
           cache.page_stride >= int64_t(cache.page_size) * cache.row_stride;
}
__device__ float block_sum(float x) {
    __shared__ float values[8];
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff,x,d);
    if ((threadIdx.x & 31) == 0) values[threadIdx.x >> 5] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? values[threadIdx.x] : 0;
    if (threadIdx.x < 32) for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff,x,d);
    return x;
}
__global__ void float_attention_kernel(const float* q, const float* swa, const float* global,
                                       const int64_t* swa_slots, const int64_t* global_slots, const float* sinks,
                                       float* out, int heads, int swa_topk, int global_topk,
                                       int64_t swa_rows, int64_t global_rows) {
    const int token = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const float* query = q + (int64_t(token) * heads + head) * 512;
    float even = 0, odd = 0;
    __shared__ float maximum, denominator, alpha, probability;
    if (!lane) { maximum = -INFINITY; denominator = 0; }
    __syncthreads();
    for (int c = 0; c < swa_topk + global_topk; ++c) {
        const bool use_global = c >= swa_topk;
        const int64_t slot = use_global ? global_slots[int64_t(token) * global_topk + c - swa_topk]
                                        : swa_slots[int64_t(token) * swa_topk + c];
        if (slot < 0 || slot >= (use_global ? global_rows : swa_rows)) continue;
        const float* value = (use_global ? global : swa) + slot * 512;
        float dot = query[lane] * value[lane] + query[lane + 256] * value[lane + 256];
        dot = block_sum(dot);
        if (!lane) {
            const float score = dot * 0.04419417382415922f;
            const float next = fmaxf(maximum, score);
            alpha = maximum == -INFINITY ? 0.f : expf(maximum - next);
            probability = expf(score - next);
            denominator = denominator * alpha + probability;
            maximum = next;
        }
        __syncthreads();
        even = even * alpha + probability * value[lane];
        odd = odd * alpha + probability * value[lane + 256];
        __syncthreads();
    }
    if (!lane) {
        const float sink = sinks[head];
        if (sink > -INFINITY) {
            const float next = fmaxf(maximum,sink);
            alpha = maximum == -INFINITY ? 0.f : expf(maximum-next);
            denominator = denominator * alpha + expf(sink-next);
        } else alpha = 1.f;
    }
    __syncthreads();
    out[(int64_t(token) * heads + head) * 512 + lane] = denominator > 0 ? even * alpha / denominator : 0.f;
    out[(int64_t(token) * heads + head) * 512 + lane + 256] = denominator > 0 ? odd * alpha / denominator : 0.f;
}
}
namespace {
bool attention_valid(const PackedAttention& p, bool expanded) {
    if (!p.q || !p.sinks || !p.output || !p.lse || p.tokens < 1 || p.heads != 64 ||
        p.swa_topk < 0 || p.global_topk < 0 || p.splits < 1 || p.splits > native::kV41MaxSplits ||
        p.q_head_stride != 512 || p.q_token_stride < p.heads * p.q_head_stride ||
        !layout_valid(p.swa,expanded?1024:528) || p.swa.page_size != 32 || (p.swa_topk && !p.swa_slots) ||
        (p.global_topk && (!layout_valid(p.global,expanded?1024:288) || !p.global_slots ||
                          (p.global.page_size != 64 && p.global.page_size != 128))) ||
        (p.splits > 1 && !p.partial)) return false;
    if (p.grouped_output) {
        if (!p.positions || !p.cos_sin || p.cos_positions < 1 || p.total_tokens < p.tokens ||
            p.token_offset < 0 || p.token_offset > p.total_tokens - p.tokens ||
            p.grouped_output == p.q || p.grouped_output == p.output || p.grouped_output == p.partial)
            return false;
    }
    return true;
}
}
template <bool ExpandedCache = false, bool FastMath = false>
static cudaError_t packed_attention_impl(const PackedAttention& p, cudaStream_t stream) {
    if (!attention_valid(p,ExpandedCache)) return cudaErrorInvalidValue;
    native::V41EpilogueParams epilogue{};
    if (p.grouped_output) {
        epilogue = {p.positions,p.cos_sin,reinterpret_cast<native::bf16*>(p.grouped_output),64,
                    p.cos_positions,p.total_tokens,p.token_offset,8};
    }
    const int64_t token_stride = int64_t(p.heads) * 512;
    native::V41AttentionParams params{
        reinterpret_cast<const native::bf16*>(p.q),p.swa.data,p.global.data,p.swa_slots,p.global_slots,p.sinks,
        reinterpret_cast<native::bf16*>(p.splits == 1 ? p.output : p.partial),p.lse,
        p.q_token_stride,p.q_head_stride,p.swa_topk,p.global_topk,
        token_stride * p.splits,token_stride,512,int64_t(p.splits)*p.heads,p.heads,1,
        p.swa.bytes,p.global.bytes,p.swa.page_stride,p.swa.row_stride,p.global.page_stride,p.global.row_stride,
        p.swa.page_size,p.global.page_size,p.swa_topk,p.global_topk,p.tokens,p.heads,p.splits,
        0.04419417382415922f,p.global_topk > 0,false,epilogue};
    return native::launch_deepseek_v41_attention<int64_t,int64_t,ExpandedCache,FastMath>(params,
        reinterpret_cast<native::bf16*>(p.output),token_stride,512,stream);
}
cudaError_t packed_attention_gpu(const PackedAttention& p, cudaStream_t stream) {
    return packed_attention_impl(p,stream);
}

namespace {
template<bool Global>
__global__ void expand_attention_cache(CacheLayout cache, uint16_t* output, int64_t rows) {
    const int64_t row=blockIdx.x;
    if(row>=rows)return;
    const int dim=threadIdx.x*2;
    const auto* src=cache.data+row*cache.row_stride;
    float a,b;
    if constexpr(Global) {
        const float scale=native::decode_fp8_e4m3(src[256+dim/16]);
        const uint8_t code=src[dim/2];
        a=native::decode_e2m1(code&15)*scale;
        b=native::decode_e2m1(code>>4)*scale;
    } else {
        const float scale=__uint_as_float(uint32_t(src[512+dim/32])<<23);
        a=native::decode_fp8_e4m3(src[dim])*scale;
        b=native::decode_fp8_e4m3(src[dim+1])*scale;
    }
    *reinterpret_cast<__nv_bfloat162*>(output+row*512+dim)=__floats2bfloat162_rn(a,b);
}
}

cudaError_t packed_attention_ada_gpu(const PackedAttention& p, void* scratch, size_t bytes, cudaStream_t stream) {
    if (!attention_valid(p,false)) return cudaErrorInvalidValue;
    if (p.tokens<=32) return packed_attention_impl<false,true>(p,stream);
    const bool supported=p.splits==1&&p.swa_topk==128&&
        p.swa.page_stride==p.swa.row_stride*p.swa.page_size&&
        (!p.global_topk||p.global.page_stride==p.global.row_stride*p.global.page_size);
    if(!supported||!scratch||reinterpret_cast<uintptr_t>(scratch)%16)return packed_attention_gpu(p,stream);
    const int64_t sw_rows=(p.swa.bytes-528)/p.swa.row_stride+1;
    const int64_t g_rows=p.global_topk?(p.global.bytes-288)/p.global.row_stride+1:0;
    if(sw_rows>0x7fffffffLL||g_rows>0x7fffffffLL||uint64_t(sw_rows+g_rows)>bytes/1024)
        return packed_attention_gpu(p,stream);
    auto* output=static_cast<uint16_t*>(scratch);
    expand_attention_cache<false><<<sw_rows,256,0,stream>>>(p.swa,output,sw_rows);
    if(g_rows)expand_attention_cache<true><<<g_rows,256,0,stream>>>(p.global,output+sw_rows*512,g_rows);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    auto expanded=p;
    expanded.swa={reinterpret_cast<uint8_t*>(output),sw_rows*1024,int64_t(p.swa.page_size)*1024,1024,p.swa.page_size};
    if(g_rows)expanded.global={reinterpret_cast<uint8_t*>(output+sw_rows*512),g_rows*1024,int64_t(p.global.page_size)*1024,1024,p.global.page_size};
    return packed_attention_impl<true,true>(expanded,stream);
}

cudaError_t float_attention_gpu(const float* q,const float* swa,const float* global,
                               const int64_t* sw_slots,const int64_t* global_slots,const float* sinks,float* out,
                               int tokens,int heads,int sw_topk,int global_topk,int64_t sw_rows,int64_t global_rows,cudaStream_t stream) {
    if (!q || !swa || !sinks || !out || tokens < 1 || heads != 64 || sw_topk < 0 || global_topk < 0 ||
        sw_rows < 1 || global_rows < 0 || (sw_topk && !sw_slots) ||
        (global_topk && (!global || !global_slots))) return cudaErrorInvalidValue;
    float_attention_kernel<<<dim3(tokens,heads),256,0,stream>>>(q,swa,global,sw_slots,global_slots,sinks,out,heads,sw_topk,global_topk,sw_rows,global_rows);
    return cudaGetLastError();
}
} // namespace strata::models::deepseek41
