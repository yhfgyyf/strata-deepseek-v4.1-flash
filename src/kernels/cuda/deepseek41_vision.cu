#include "strata/models/deepseek41/vision.hpp"
#include <cuda_bf16.h>
#include <cmath>

namespace strata::models::deepseek41 {
namespace {
__device__ float rounded(float x,bool bf16){return bf16?__bfloat162float(__float2bfloat16_rn(x)):x;}
__global__ void qkv_kernel(const float* input,float* q,float* k,float* v,int n,int width,bool bf16) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(n)*1024;i+=int64_t(gridDim.x)*blockDim.x){
        const int t=i/1024,head=(i%1024)/64,d=i%64;
        if(d>=32)continue;
        const int axis=d/16,pair=d%16;const float position=axis?t%width:t/width;
        const float angle=position/powf(10000.f,float(pair)/16.f);float sine,cosine;sincosf(angle,&sine,&cosine);
        for(int part=0;part<2;++part){
            const float* source=input+int64_t(t)*3072+part*1024+head*64;
            float* destination=(part?k:q)+int64_t(t)*1024+head*64;
            const float a=source[d],b=source[d+32];
            destination[d]=rounded(a*cosine-b*sine,bf16);destination[d+32]=rounded(b*cosine+a*sine,bf16);
        }
        v[int64_t(t)*1024+head*64+d]=input[int64_t(t)*3072+2048+head*64+d];
        v[int64_t(t)*1024+head*64+d+32]=input[int64_t(t)*3072+2048+head*64+d+32];
    }
}
__global__ void attention_kernel(const float* q,const float* k,const float* v,float* out,int n,bool bf16) {
    const int token=blockIdx.x,head=blockIdx.y,lane=threadIdx.x;
    const int64_t offset=int64_t(token)*1024+head*64;
    const float qa=q[offset+lane],qb=q[offset+lane+32];float a=0,b=0,maximum=-INFINITY,denominator=0;
    for(int key=0;key<n;++key){
        const int64_t base=int64_t(key)*1024+head*64;
        float dot=qa*k[base+lane]+qb*k[base+lane+32];
        for(int step=16;step;step>>=1)dot+=__shfl_down_sync(0xffffffff,dot,step);
        const float score=__shfl_sync(0xffffffff,dot,0)*.125f;
        const float next=fmaxf(maximum,score),alpha=maximum==-INFINITY?0.f:expf(maximum-next),probability=expf(score-next);
        a=a*alpha+probability*v[base+lane];b=b*alpha+probability*v[base+lane+32];denominator=denominator*alpha+probability;maximum=next;
    }
    out[offset+lane]=rounded(a/denominator,bf16);out[offset+lane+32]=rounded(b/denominator,bf16);
}
__global__ void unshuffle_kernel(const float* x,float* out,int h,int w,int oh,int ow) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(oh)*ow*9216;i+=int64_t(gridDim.x)*blockDim.x){
        const int cell=i/9216,d=(i%9216)/9,dy=(i%9)/3,dx=i%3;
        const int y=(cell/ow)*3+dy,z=(cell%ow)*3+dx;
        out[i]=y<h&&z<w?x[(int64_t(y)*w+z)*1024+d]:0.f;
    }
}
__global__ void span_kernel(const float* x,const float* start,const float* end,const float* newline,float* out,int h,int w) {
    const int tokens=h*(w+1)+2;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(tokens)*5120;i+=int64_t(gridDim.x)*blockDim.x){
        const int t=i/5120,d=i%5120;
        if(t==0)out[i]=start[d];else if(t==tokens-1)out[i]=end[d];
        else{const int row=(t-1)/(w+1),col=(t-1)%(w+1);out[i]=col==w?newline[d]:x[(int64_t(row)*w+col)*5120+d];}
    }
}
__global__ void bias_kernel(float* x,const float* b,int64_t n,int width,bool bf16) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x)x[i]=rounded(x[i]+(b?b[i%width]:0.f),bf16);
}
__global__ void add_kernel(float* x,const float* y,int64_t n,bool bf16) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x)x[i]=rounded(x[i]+y[i],bf16);
}
__global__ void activate_kernel(const float* gate,const float* up,float* out,int64_t n,bool gelu,bool bf16) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x){
        const float g=gate[i];float value=gelu?.5f*g*(1.f+erff(g*.7071067811865475244f)):g/(1.f+expf(-g));
        value=rounded(value,bf16);if(!gelu)value*=up[i];out[i]=rounded(value,bf16);
    }
}
int grid(int64_t n){return static_cast<int>(std::min<int64_t>((n+255)/256,65535));}
}
cudaError_t vision_qkv_gpu(const float* qkv,float* q,float* k,float* v,int h,int w,bool bf16,cudaStream_t s){
    if(h<1||w<1||h*w>9198)return cudaErrorInvalidValue;qkv_kernel<<<grid(int64_t(h)*w*1024),256,0,s>>>(qkv,q,k,v,h*w,w,bf16);return cudaGetLastError();
}
cudaError_t vision_attention_gpu(const float* q,const float* k,const float* v,float* out,int n,bool bf16,cudaStream_t s){
    if(n<1||n>9198)return cudaErrorInvalidValue;attention_kernel<<<dim3(n,16),32,0,s>>>(q,k,v,out,n,bf16);return cudaGetLastError();
}
cudaError_t vision_unshuffle_gpu(const float* x,float* out,int h,int w,cudaStream_t s){
    if(h<1||w<1)return cudaErrorInvalidValue;unshuffle_kernel<<<grid(int64_t((h+2)/3)*((w+2)/3)*9216),256,0,s>>>(x,out,h,w,(h+2)/3,(w+2)/3);return cudaGetLastError();
}
cudaError_t vision_span_gpu(const float* x,const float* start,const float* end,const float* newline,float* out,int h,int w,cudaStream_t s){
    if(h<1||w<1||h*(w+1)+2>1024)return cudaErrorInvalidValue;span_kernel<<<grid(int64_t(h*(w+1)+2)*5120),256,0,s>>>(x,start,end,newline,out,h,w);return cudaGetLastError();
}
cudaError_t vision_bias_gpu(float* x,const float* bias,int tokens,int width,bool bf16,cudaStream_t s){bias_kernel<<<grid(int64_t(tokens)*width),256,0,s>>>(x,bias,int64_t(tokens)*width,width,bf16);return cudaGetLastError();}
cudaError_t vision_add_gpu(float* x,const float* branch,int64_t n,bool bf16,cudaStream_t s){add_kernel<<<grid(n),256,0,s>>>(x,branch,n,bf16);return cudaGetLastError();}
cudaError_t vision_activate_gpu(const float* g,const float* u,float* out,int64_t n,bool gelu,bool bf16,cudaStream_t s){activate_kernel<<<grid(n),256,0,s>>>(g,u,out,n,gelu,bf16);return cudaGetLastError();}
} // namespace strata::models::deepseek41
