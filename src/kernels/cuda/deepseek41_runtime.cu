#include "../../models/deepseek41/runtime_kernels.hpp"
#include "strata/models/deepseek41/ops.hpp"
#include "deepseek41_rotary_shared.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

namespace strata::models::deepseek41 {
namespace {
__device__ float embedding_value(const void* weights, int type, int64_t offset) {
    if(type==2){const auto* block=static_cast<const uint8_t*>(weights)+(offset/32)*18;
        return __half2float(*reinterpret_cast<const __half*>(block))*(int((block[2+offset%16]>>((offset%32)/16*4))&15)-8);}
    if(type==8){const auto* block=static_cast<const uint8_t*>(weights)+(offset/32)*34;
        return __half2float(*reinterpret_cast<const __half*>(block))*reinterpret_cast<const int8_t*>(block+2)[offset%32];}
    return type == 0 ? static_cast<const float*>(weights)[offset] :
        __bfloat162float(static_cast<const __nv_bfloat16*>(weights)[offset]);
}
__global__ void draft_markov_row_kernel(const void* weights,int type,const int* previous,const int* inverse,
                                       float* out,bool bf16,int* status) {
    const int id=*previous;
    const int row=id>=0&&id<129280?inverse[id]:-1;
    if(row<0){if(threadIdx.x==0)atomicExch(status,1);out[threadIdx.x]=0;return;}
    float x=embedding_value(weights,type,int64_t(row)*256+threadIdx.x);
    out[threadIdx.x]=bf16?__bfloat162float(__float2bfloat16_rn(x)):x;
}
__global__ void draft_penalty_counts_kernel(const int* history,int count,const int* inverse,int* counts) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){
        const int id=history[i];const int row=id>=0&&id<129280?inverse[id]:-1;
        if(row>=0)atomicAdd(counts+row,1);
    }
}
__global__ void draft_greedy_kernel(const float* head,const float* bias,const int* ids,const int* inverse,int vocab,
                                   const int* history,int history_size,int step,int* chain,int* counts,
                                   kernels::SamplerParams p,int* status) {
    __shared__ double values[256];__shared__ int tokens[256];
    double best=-INFINITY;int token=INT_MAX;
    for(int v=threadIdx.x;v<vocab;v+=blockDim.x){
        const float combined=__fadd_rn(head[v],bias[v]);
        if(isnan(combined)||combined==INFINITY)atomicExch(status,2);
        double value=combined;
        const int count=counts?counts[v]:0;
        if(count){
            value=value<=0?value*double(p.penalty_repeat):value/double(p.penalty_repeat);
            value-=double(__fadd_rn(__fmul_rn(float(count),p.penalty_freq),p.penalty_present));
        }
        if(value>best||(value==best&&ids[v]<token)){best=value;token=ids[v];}
    }
    values[threadIdx.x]=best;tokens[threadIdx.x]=token;__syncthreads();
    for(int width=128;width;width/=2){
        if(threadIdx.x<width){const int peer=threadIdx.x+width;
            if(values[peer]>values[threadIdx.x]||(values[peer]==values[threadIdx.x]&&tokens[peer]<tokens[threadIdx.x])){
                values[threadIdx.x]=values[peer];tokens[threadIdx.x]=tokens[peer];
            }
        }__syncthreads();
    }
    if(threadIdx.x==0){
        if(values[0]==-INFINITY||tokens[0]==INT_MAX){atomicExch(status,3);tokens[0]=ids[0];}
        const int selected=tokens[0];chain[step+1]=selected;
        if(counts){
            ++counts[inverse[selected]];
            const int remove=history_size+step-p.penalty_last_n;
            if(remove>=0){const int old=remove<history_size?history[remove]:chain[remove-history_size+1];
                const int row=old>=0&&old<129280?inverse[old]:-1;if(row>=0)--counts[row];}
        }
    }
}
__global__ void init_kernel(const void* w, int type, const int* ids, const float* pixels,
                            const uint8_t* image, float* h, float* pre, int tokens) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(tokens) * 20480; i += int64_t(gridDim.x) * blockDim.x) {
        const int t = i / 20480, d = i % 5120;
        h[i] = image && image[t] ? pixels[int64_t(t) * 5120 + d] : embedding_value(w,type,int64_t(ids[t]) * 5120 + d);
        if (i % 20480 < 4) pre[t * 4 + i % 20480] = i % 20480 == 0 ? 1.f : 0.f;
    }
}
__global__ void mean_kernel(const float* h, float* out, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t base = (i / 5120) * 20480 + i % 5120;
        out[i] = (h[base] + h[base + 5120] + h[base + 10240] + h[base + 15360]) * .25f;
    }
}
__global__ void gather_kernel(const float* x, const int* rows, float* out, int width, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        out[i] = x[int64_t(rows[i / width]) * width + i % width];
}
__global__ void scatter_kernel(const float* x, const int* slots, float* parts, int top, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        parts[int64_t(slots[i / 5120]) * 5120 + i % 5120] = x[i];
}
__global__ void combine_kernel(const float* parts, const int* ids, const float* shared, float* out, int top, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int t = i / 5120, d = i % 5120;
        int order[6]; for (int k = 0; k < top; ++k) order[k] = k;
        for (int k = 1; k < top; ++k) for (int j = k; j && ids[t * top + order[j]] < ids[t * top + order[j - 1]]; --j) {
            const int a = order[j]; order[j] = order[j - 1]; order[j - 1] = a;
        }
        float sum = 0;
        for (int k = 0; k < top; ++k) sum += parts[(int64_t(t) * top + order[k]) * 5120 + d];
        out[i] = sum + shared[i];
    }
}
__global__ void pool_kernel(const float* kv, const float* scores, const float* previous,
                            const float* previous_score, float* out, int64_t n, bool odd) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int group = i / 512, d = i % 512, first = group * 2 - int(odd);
        const float a = first < 0 ? previous[d] : kv[first * 512 + d];
        const float ga = first < 0 ? previous_score[d] : scores[first * 512 + d];
        const float b = kv[(first + 1) * 512 + d], gb = scores[(first + 1) * 512 + d];
        const float maximum = fmaxf(ga,gb), wa = expf(ga - maximum), wb = expf(gb - maximum);
        out[i] = (a * wa + b * wb) / (wa + wb);
    }
}
__global__ void rotate_kernel(float* x, int heads, int dim, int tokens, int64_t start, int step,
                              RotaryFrequencies f, bool inverse) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(tokens) * heads * 32; i += int64_t(gridDim.x) * blockDim.x) {
        const int token = i / (heads * 32), head = (i / 32) % heads, d = i % 32;
        const float angle = float(start + int64_t(token) * step) * f.value[d];
        float sine, cosine; sincosf(angle,&sine,&cosine); if (inverse) sine = -sine;
        float* p = x + (int64_t(token) * heads + head) * dim + dim - 64 + d * 2;
        const float a = p[0], b = p[1]; p[0] = a * cosine - b * sine; p[1] = a * sine + b * cosine;
    }
}
__global__ void slots_kernel(int64_t* slots, int tokens, int previous) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tokens * 128; i += gridDim.x * blockDim.x) {
        const int t = i / 128, k = i % 128, end = previous + t + 1, begin = max(0,end - 128);
        slots[i] = begin + k < end ? begin + k : -1;
    }
}
__global__ void bf_strided_kernel(const float* x,__nv_bfloat16* out,int cols,int64_t stride,int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        out[i] = __float2bfloat16_rn(x[(i / cols) * stride + i % cols]);
}
__global__ void round_strided_kernel(float* x,int cols,int64_t stride,int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t offset = (i / cols) * stride + i % cols;
        x[offset] = __bfloat162float(__float2bfloat16_rn(x[offset]));
    }
}
int grid(int64_t n) { return static_cast<int>(std::min<int64_t>((n + 255) / 256,65535)); }
}
cudaError_t init_residual_gpu(const void* w,int type,const int* ids,const float* pixels,const uint8_t* image,
                             float* h,float* pre,int tokens,cudaStream_t s) {
    if ((type != 0 && type != 30 && type != 8 && type != 2) || tokens < 1) return cudaErrorInvalidValue;
    init_kernel<<<grid(int64_t(tokens)*20480),256,0,s>>>(w,type,ids,pixels,image,h,pre,tokens); return cudaGetLastError();
}
cudaError_t draft_markov_row_gpu(const void* weights,int type,const int* previous,const int* inverse,
                                float* out,bool bf16,int* status,cudaStream_t stream) {
    if(!weights||!previous||!inverse||!out||!status||(type!=0&&type!=2&&type!=8&&type!=30))return cudaErrorInvalidValue;
    draft_markov_row_kernel<<<1,256,0,stream>>>(weights,type,previous,inverse,out,bf16,status);return cudaGetLastError();
}
cudaError_t draft_penalty_counts_gpu(const int* history,int count,const int* inverse,int* counts,cudaStream_t stream) {
    if(count<0||!inverse||!counts||(count&&!history))return cudaErrorInvalidValue;
    if(count)draft_penalty_counts_kernel<<<std::min(256,(count+255)/256),256,0,stream>>>(history,count,inverse,counts);
    return cudaGetLastError();
}
cudaError_t draft_greedy_gpu(const float* head,const float* bias,const int* ids,const int* inverse,int vocab,
                            const int* history,int history_size,int step,int* chain,int* counts,
                            const kernels::SamplerParams& sampling,int* status,cudaStream_t stream) {
    if(!head||!bias||!ids||!inverse||!chain||!status||vocab<1||vocab>129280||step<0||step>6||
       history_size<0||(history_size&&!history))return cudaErrorInvalidValue;
    draft_greedy_kernel<<<1,256,0,stream>>>(head,bias,ids,inverse,vocab,history,history_size,step,chain,counts,sampling,status);
    return cudaGetLastError();
}
cudaError_t mean_residual_gpu(const float* h,float* out,int tokens,cudaStream_t s) {
    mean_kernel<<<grid(int64_t(tokens)*5120),256,0,s>>>(h,out,int64_t(tokens)*5120); return cudaGetLastError();
}
cudaError_t gather_rows_gpu(const float* x,const int* rows,float* out,int tokens,int width,cudaStream_t s) {
    gather_kernel<<<grid(int64_t(tokens)*width),256,0,s>>>(x,rows,out,width,int64_t(tokens)*width); return cudaGetLastError();
}
cudaError_t scatter_expert_gpu(const float* x,const int* slots,float* parts,int tokens,int top,cudaStream_t s) {
    scatter_kernel<<<grid(int64_t(tokens)*5120),256,0,s>>>(x,slots,parts,top,int64_t(tokens)*5120); return cudaGetLastError();
}
cudaError_t combine_experts_gpu(const float* parts,const int* ids,const float* shared,float* out,int tokens,int top,cudaStream_t s) {
    combine_kernel<<<grid(int64_t(tokens)*5120),256,0,s>>>(parts,ids,shared,out,top,int64_t(tokens)*5120); return cudaGetLastError();
}
cudaError_t pool_groups_gpu(const float* kv,const float* gate,const float* previous,const float* previous_gate,
                           float* out,int groups,bool odd,cudaStream_t s) {
    if (groups < 1) return cudaErrorInvalidValue;
    pool_kernel<<<grid(int64_t(groups)*512),256,0,s>>>(kv,gate,previous,previous_gate,out,int64_t(groups)*512,odd); return cudaGetLastError();
}
cudaError_t rotate_strided_gpu(float* x,int heads,int dim,int tokens,int64_t start,int step,const Geometry& g,
                              bool compressed,bool inverse,cudaStream_t s) {
    if (tokens < 1 || heads < 1 || dim < 64 || start < 0 || step < 1) return cudaErrorInvalidValue;
    if(rotary_detail::shared_shape(heads,dim,tokens))
        return rotary_detail::launch(x,heads,dim,tokens,start,step,rotary_frequencies(g,compressed),inverse,s);
    rotate_kernel<<<grid(int64_t(tokens)*heads*32),256,0,s>>>(x,heads,dim,tokens,start,step,rotary_frequencies(g,compressed),inverse); return cudaGetLastError();
}
cudaError_t window_slots_gpu(int64_t* slots,int tokens,int previous,cudaStream_t s) {
    if (tokens < 1 || previous < 0 || previous > 128) return cudaErrorInvalidValue;
    slots_kernel<<<grid(int64_t(tokens)*128),256,0,s>>>(slots,tokens,previous); return cudaGetLastError();
}
cudaError_t to_bf16_strided_gpu(const float* x,uint16_t* out,int tokens,int cols,int64_t stride,cudaStream_t s) {
    if (tokens < 1 || cols < 1 || stride < cols) return cudaErrorInvalidValue;
    bf_strided_kernel<<<grid(int64_t(tokens)*cols),256,0,s>>>(x,reinterpret_cast<__nv_bfloat16*>(out),cols,stride,int64_t(tokens)*cols);
    return cudaGetLastError();
}
cudaError_t round_bf16_strided_gpu(float* x,int tokens,int cols,int64_t stride,cudaStream_t s) {
    if (tokens < 1 || cols < 1 || stride < cols) return cudaErrorInvalidValue;
    round_strided_kernel<<<grid(int64_t(tokens)*cols),256,0,s>>>(x,cols,stride,int64_t(tokens)*cols);
    return cudaGetLastError();
}
} // namespace strata::models::deepseek41
