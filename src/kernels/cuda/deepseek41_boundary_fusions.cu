// Validated 2026-10-21 operator candidates; preserve the existing BF16 boundaries.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include "strata/models/deepseek41/kernels.hpp"
namespace strata::models::deepseek41 {
namespace boundary_detail {
__device__ float sum_block(float v) {
    __shared__ float sums[8];
    for (int delta = 16; delta; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8 ? sums[threadIdx.x] : 0;
    if (threadIdx.x < 32) for (int delta = 16; delta; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
    return v;
}
__device__ float sigmoid(float x) { return x >= 0 ? 1.f / (1.f + expf(-x)) : expf(x) / (1.f + expf(x)); }
int grid(int64_t n){return int((n+255)/256<65535?(n+255)/256:65535);}
__global__ void rms_fused_kernel(const float* x, const float* weight, float* out, int width, float eps) {
    const int token = blockIdx.x;
    float square = 0;
    for (int col = threadIdx.x; col < width; col += blockDim.x) { const float v = x[int64_t(token) * width + col]; square += v * v; }
    square = sum_block(square);
    __shared__ float inverse;
    if (threadIdx.x == 0) inverse = rsqrtf(square / width + eps);
    __syncthreads();
    for (int col = threadIdx.x; col < width; col += blockDim.x)
        out[int64_t(token) * width + col] = __bfloat162float(__float2bfloat16_rn(x[int64_t(token) * width + col] * inverse * (weight ? weight[col] : 1.f)));
}

__global__ void hc_pre_fused_kernel(const float* x, const float* pre, float* out, int hidden, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int token = i / hidden, d = i % hidden;
        float sum = 0; for (int j = 0; j < 4; ++j) sum += pre[token * 4 + j] * x[(int64_t(token) * 4 + j) * hidden + d];
        out[i] = __bfloat162float(__float2bfloat16_rn(sum));
    }
}

__global__ void hc_post_fused_kernel(const float* x, const float* branch, const float* post, const float* comb,
                              float* out, int hidden, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int token = i / hidden, d = i % hidden;
        float copies[4]; for (int j = 0; j < 4; ++j) copies[j] = x[(int64_t(token) * 4 + j) * hidden + d];
        for (int destination = 0; destination < 4; ++destination) {
            float value = post[token * 4 + destination] * branch[i];
            for (int source = 0; source < 4; ++source) value += comb[token * 16 + source * 4 + destination] * copies[source];
            out[(int64_t(token) * 4 + destination) * hidden + d] = __bfloat162float(__float2bfloat16_rn(value));
        }
    }
}

__global__ void swiglu_fused_kernel(const float* gate, const float* up, const float* weights, float* out, int width, int64_t n, float limit) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        float g = gate[i], u = up[i];
        if (limit > 0) { g = fminf(g, limit); u = fminf(fmaxf(u, -limit), limit); }
        out[i] = __bfloat162float(__float2bfloat16_rn(g * sigmoid(g) * u * (weights ? weights[i / width] : 1.f)));
    }
}

template<int count>
__device__ void sum_columns(float (&values)[count]) {
    __shared__ float partial[count][8];
    for(int delta=16;delta;delta>>=1){
        #pragma unroll
        for(int j=0;j<count;++j)values[j]+=__shfl_down_sync(0xffffffff,values[j],delta);
    }
    if((threadIdx.x&31)==0){
        #pragma unroll
        for(int j=0;j<count;++j)partial[j][threadIdx.x>>5]=values[j];
    }
    __syncthreads();
    #pragma unroll
    for(int j=0;j<count;++j)values[j]=threadIdx.x<(blockDim.x+31)/32?partial[j][threadIdx.x]:0.f;
    if(threadIdx.x<32)for(int delta=16;delta;delta>>=1){
        #pragma unroll
        for(int j=0;j<count;++j)values[j]+=__shfl_down_sync(0xffffffff,values[j],delta);
    }
}
__global__ void engram_gate_reduce_kernel(float* h, const float* kv, const float* qw, const float* kw,
                                  const uint8_t* image, int width) {
    const int t = blockIdx.x / 4, c = blockIdx.x % 4;
    float* x = h + int64_t(t * 4 + c) * width;
    const float* key = kv + int64_t(t * 5 + c) * width;
    float hh = 0, kk = 0, dot = 0;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        hh += x[d] * x[d]; kk += key[d] * key[d];
        const float q=x[d]*qw[c*width+d];
        dot += (kw?q*kw[c*width+d]:q)*key[d];
    }
    __shared__ float sums[3], gate;
    float triple[3]={hh,kk,dot};sum_columns(triple);
    hh=triple[0];kk=triple[1];dot=triple[2];
    if(!threadIdx.x){sums[0]=hh;sums[1]=kk;}
    if (!threadIdx.x) {
        dot *= rsqrtf(sums[0] / width + 1e-20f) * rsqrtf(sums[1] / width + 1e-20f) * rsqrtf(float(width));
        float input = sqrtf(fmaxf(fabsf(dot), 1e-6f)); if (dot < 0) input = -input;
        gate = image && image[t] ? 0.f : sigmoid(input);
    }
    __syncthreads();
    const float* value = kv + int64_t(t * 5 + 4) * width;
    for (int d = threadIdx.x; d < width; d += blockDim.x) x[d] = __bfloat162float(__float2bfloat16_rn(x[d] + gate * value[d]));
}
__global__ void hc_split_warp_kernel(const float* mixes,const float* scale,const float* base,float* pre,float* post,float* comb,int tokens,float eps,int iterations){
 const int token=(blockIdx.x*blockDim.x+threadIdx.x)/16,lane=threadIdx.x%16;
 const unsigned mask=__ballot_sync(0xffffffff,token<tokens);if(token>=tokens)return;
 const float* x=mixes+token*24;
 if(lane<4){pre[token*4+lane]=sigmoid(x[lane]*scale[0]+base[lane])+eps;post[token*4+lane]=2.f*sigmoid(x[4+lane]*scale[1]+base[4+lane]);}
 float v=x[8+lane]*scale[2]+base[8+lane];const int row=lane/4,col=lane%4;
 float maximum=-INFINITY;
 #pragma unroll
 for(int i=0;i<4;++i)maximum=fmaxf(maximum,__shfl_sync(mask,v,row*4+i,16));
 v=expf(v-maximum);float sum=0;
 #pragma unroll
 for(int i=0;i<4;++i)sum+=__shfl_sync(mask,v,row*4+i,16);
 v=v/sum+eps;
 for(int it=0;it<iterations;++it){
  if(it){sum=eps;
   #pragma unroll
   for(int i=0;i<4;++i)sum+=__shfl_sync(mask,v,row*4+i,16);
   v/=sum;
  }
  sum=eps;
  #pragma unroll
  for(int i=0;i<4;++i)sum+=__shfl_sync(mask,v,i*4+col,16);
  v/=sum;
 }
 comb[token*16+lane]=v;
}

} // namespace boundary_detail
using namespace boundary_detail;
cudaError_t rms_bf16_gpu(const float* x, const float* w, float* out, int width, int tokens, float eps, cudaStream_t s) {
    if (width < 1 || tokens < 1 || eps <= 0) return cudaErrorInvalidValue;
    rms_fused_kernel<<<tokens,256,0,s>>>(x,w,out,width,eps); return cudaGetLastError();
}
cudaError_t hc_pre_bf16_gpu(const float* x, const float* pre, float* out, int hidden, int tokens, cudaStream_t s) {
    if (hidden < 1 || tokens < 1) return cudaErrorInvalidValue;
    hc_pre_fused_kernel<<<grid(int64_t(hidden)*tokens),256,0,s>>>(x,pre,out,hidden,int64_t(hidden)*tokens); return cudaGetLastError();
}
cudaError_t hc_post_bf16_gpu(const float* x, const float* branch, const float* post, const float* comb, float* out,
                        int hidden, int tokens, cudaStream_t s) {
    if (hidden < 1 || tokens < 1) return cudaErrorInvalidValue;
    hc_post_fused_kernel<<<grid(int64_t(hidden)*tokens),256,0,s>>>(x,branch,post,comb,out,hidden,int64_t(hidden)*tokens); return cudaGetLastError();
}
cudaError_t swiglu_bf16_gpu(const float* g, const float* u, const float* w, float* out, int width, int tokens, float limit, cudaStream_t s) {
    if (width < 1 || tokens < 1) return cudaErrorInvalidValue;
    swiglu_fused_kernel<<<grid(int64_t(width)*tokens),256,0,s>>>(g,u,w,out,width,int64_t(width)*tokens,limit); return cudaGetLastError();
}
cudaError_t engram_gate_bf16_gpu(float* h,const float* kv,const float* qw,const float* kw,const uint8_t* image,int n,int d,cudaStream_t stream){
 if(n<1||d<1)return cudaErrorInvalidValue;
 engram_gate_reduce_kernel<<<n*4,256,0,stream>>>(h,kv,qw,kw,image,d);return cudaGetLastError();
}
cudaError_t hc_split_warp_gpu(const float* x,const float* s,const float* b,float* p,float* q,float* c,int n,float e,int it,cudaStream_t stream){
 if(n<1||e<=0||it<1)return cudaErrorInvalidValue;
 hc_split_warp_kernel<<<(n+7)/8,128,0,stream>>>(x,s,b,p,q,c,n,e,it);return cudaGetLastError();
}
} // namespace strata::models::deepseek41
