// IQ layouts/codebooks and dequantization: pinned ggml (MIT), as in iq_kernels.cu.
// The fused input reuse follows Strata's paired GU and NVFP4 W4A16 organization.
// Keep DeepSeek BF16 boundaries; neither FP16 nor quantized activations are added.
#include "strata/models/deepseek41/mixed_expert.hpp"
#include "strata/models/deepseek41/kernels.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace strata::models::deepseek41 {
namespace {
std::atomic<bool> enabled{false};
std::atomic<bool> short_enabled{false};
std::atomic<bool> short_vector_enabled{true};
std::atomic<bool> short_warp_enabled{true};
std::atomic<bool> projection_warp_enabled{true};
std::atomic<bool> short_tokens_enabled{true};
std::atomic<bool> short_down_tokens_enabled{true};

__device__ __forceinline__ float bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

template<int Type>
__device__ __forceinline__ float weight(const void* packed,int64_t index) {
    if constexpr(Type==39) {
        const auto* p=static_cast<const uint8_t*>(packed)+(index/32)*17;
        // The table contains E2M1*2, so the exponent is decreased by one.
        constexpr int values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
        const uint32_t bits=p[0]<2?0x00200000u<<p[0]:uint32_t(p[0]-1)<<23;
        const int code=(p[1+index%16]>>((index%32)/16*4))&15;
        return bf16(__uint_as_float(bits)*values[code]);
    } else {
        const int64_t block=index/256;
        const int ib=(index%256)/32,il=(index%32)/8,j=index%8;
        float d;
        uint8_t sign,grid;
        if constexpr(Type==16) {
            const auto& b=static_cast<const block_iq2_xxs*>(packed)[block];
            const uint16_t* q=b.qs+4*ib;
            const uint32_t aux=uint32_t(q[2])|(uint32_t(q[3])<<16);
            d=float(b.d)*(0.5f+(aux>>28))*0.25f;
            sign=ksigns_iq2xs[(aux>>(7*il))&127];
            grid=(iq2xxs_grid[reinterpret_cast<const uint8_t*>(q)[il]]>>(8*j))&255;
        } else if constexpr(Type==17) {
            const auto& b=static_cast<const block_iq2_xs*>(packed)[block];
            const uint16_t q=b.qs[4*ib+il];
            d=float(b.d)*(0.5f+((b.scales[ib]>>(4*(il/2)))&15))*0.25f;
            sign=ksigns_iq2xs[q>>9];
            grid=(iq2xs_grid[q&511]>>(8*j))&255;
        } else if constexpr(Type==22) {
            const auto& b=static_cast<const block_iq2_s*>(packed)[block];
            const int q=b.qs[4*ib+il]|((b.qh[ib]<<(8-2*il))&0x300);
            d=float(b.d)*(0.5f+((b.scales[ib]>>(4*(il/2)))&15))*0.25f;
            sign=b.qs[32+4*ib+il];
            grid=(iq2s_grid[q]>>(8*j))&255;
        } else if constexpr(Type==18) {
            const auto& b=static_cast<const block_iq3_xxs*>(packed)[block];
            const auto* gas=reinterpret_cast<const uint16_t*>(b.qs+64)+2*ib;
            const uint32_t aux=uint32_t(gas[0])|(uint32_t(gas[1])<<16);
            d=float(b.d)*(0.5f+(aux>>28))*0.5f;
            sign=ksigns_iq2xs[(aux>>(7*il))&127];
            grid=(iq3xxs_grid[b.qs[8*ib+2*il+j/4]]>>(8*(j%4)))&255;
        } else {
            static_assert(Type==21);
            const auto& b=static_cast<const block_iq3_s*>(packed)[block];
            const int q=b.qs[8*ib+2*il+j/4]|((b.qh[ib]<<(8-2*il-j/4))&256);
            d=float(b.d)*(1+2*((b.scales[ib/2]>>(4*(ib%2)))&15));
            sign=b.signs[4*ib+il];
            grid=(iq3s_grid[q]>>(8*(j%4)))&255;
        }
        return bf16(d*grid*(sign&kmask_iq2xs[j]?-1.f:1.f));
    }
}

// Decode eight consecutive IQ values once per lane. Scale, sign and codebook
// indices are shared by all eight values, and by every activation column.
template<int Type>
__device__ __forceinline__ void weight8(const void* packed,int64_t index,float (&out)[8]) {
    const int64_t block=index/256;
    const int ib=(index%256)/32,il=(index%32)/8;
    float d; uint8_t sign; uint64_t grid;
    if constexpr(Type==16) {
        const auto& b=static_cast<const block_iq2_xxs*>(packed)[block];
        const uint16_t* q=b.qs+4*ib;
        const uint32_t aux=uint32_t(q[2])|(uint32_t(q[3])<<16);
        d=float(b.d)*(0.5f+(aux>>28))*0.25f;
        sign=ksigns_iq2xs[(aux>>(7*il))&127];
        grid=iq2xxs_grid[reinterpret_cast<const uint8_t*>(q)[il]];
    } else if constexpr(Type==18) {
        const auto& b=static_cast<const block_iq3_xxs*>(packed)[block];
        const auto* gas=reinterpret_cast<const uint16_t*>(b.qs+64)+2*ib;
        const uint32_t aux=uint32_t(gas[0])|(uint32_t(gas[1])<<16);
        d=float(b.d)*(0.5f+(aux>>28))*0.5f;
        sign=ksigns_iq2xs[(aux>>(7*il))&127];
        grid=uint64_t(iq3xxs_grid[b.qs[8*ib+2*il]])|
             (uint64_t(iq3xxs_grid[b.qs[8*ib+2*il+1]])<<32);
    } else {
        static_assert(Type==21);
        const auto& b=static_cast<const block_iq3_s*>(packed)[block];
        const int q0=b.qs[8*ib+2*il]|((b.qh[ib]<<(8-2*il))&256);
        const int q1=b.qs[8*ib+2*il+1]|((b.qh[ib]<<(7-2*il))&256);
        d=float(b.d)*(1+2*((b.scales[ib/2]>>(4*(ib%2)))&15));
        sign=b.signs[4*ib+il];
        grid=uint64_t(iq3s_grid[q0])|(uint64_t(iq3s_grid[q1])<<32);
    }
#pragma unroll
    for(int j=0;j<8;++j)out[j]=bf16(d*int((grid>>(8*j))&255)*(sign&(1<<j)?-1.f:1.f));
}
template<int Type>
__global__ void dequant_vector_kernel(const void* w,uint16_t* output,int64_t n) {
    const int64_t index=(int64_t(blockIdx.x)*128+threadIdx.x)*8;
    if(index>=n)return;
    float values[8];weight8<Type>(w,index,values);
#pragma unroll
    for(int j=0;j<4;++j)reinterpret_cast<uint32_t*>(output)[index/2+j]=
        uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(values[2*j])))|
        (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(values[2*j+1])))<<16);
}
// Retain the fallback's per-thread K order and two-stage shuffle reduction.
template<int Count>
__device__ void sum_columns(float (&values)[Count]) {
    __shared__ float partial[Count][8];
    for(int delta=16;delta;delta>>=1) {
        #pragma unroll
        for(int j=0;j<Count;++j)values[j]+=__shfl_down_sync(0xffffffff,values[j],delta);
    }
    if((threadIdx.x&31)==0) {
        #pragma unroll
        for(int j=0;j<Count;++j)partial[j][threadIdx.x>>5]=values[j];
    }
    __syncthreads();
    #pragma unroll
    for(int j=0;j<Count;++j)values[j]=threadIdx.x<8?partial[j][threadIdx.x]:0.f;
    if(threadIdx.x<32)for(int delta=16;delta;delta>>=1) {
        #pragma unroll
        for(int j=0;j<Count;++j)values[j]+=__shfl_down_sync(0xffffffff,values[j],delta);
    }
}

template<int Type,int N,bool Pair>
__global__ void projection_kernel(const void* w,const void* wu,const float* x,float* y,float* yu,
                                  int rows,int cols,const int* slots) {
    const int row=blockIdx.x;
    float values[N*(Pair?2:1)]{};
    for(int col=threadIdx.x;col<cols;col+=256) {
        const int64_t index=int64_t(row)*cols+col;
        const float wg=weight<Type>(w,index);
        float up=0;
        if constexpr(Pair)up=weight<Type>(wu,index);
        #pragma unroll
        for(int t=0;t<N;++t) {
            const float activation=x[int64_t(t)*cols+col];
            values[t]+=wg*activation;
            if constexpr(Pair)values[N+t]+=up*activation;
        }
    }
    sum_columns(values);
    if(threadIdx.x==0) {
        #pragma unroll
        for(int t=0;t<N;++t) {
            const int destination=slots?slots[t]:t;
            y[int64_t(destination)*rows+row]=bf16(values[t]);
            if constexpr(Pair)yu[int64_t(t)*rows+row]=bf16(values[N+t]);
        }
    }
}

__global__ void activation_kernel(const float* gate,const float* up,const float* routing,
                                   float* activation,int width,int64_t n) {
    for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<n;i+=int64_t(gridDim.x)*256) {
        const float g=fminf(gate[i],10.f),u=fminf(fmaxf(up[i],-10.f),10.f);
        const float sig=g>=0?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));
        const float h=g*sig*u*(routing?routing[i/width]:1.f);
        activation[i]=bf16(h);
    }
}

template<int Type,int N>
__global__ void gu_activation_kernel(const void* gate_weights,const void* up_weights,
    const float* x,const float* routing,float* activation,int rows,int cols) {
    const int row=blockIdx.x;
    float values[2*N]{};
    for(int col=threadIdx.x;col<cols;col+=256) {
        const int64_t index=int64_t(row)*cols+col;
        const float gate=weight<Type>(gate_weights,index),up=weight<Type>(up_weights,index);
        #pragma unroll
        for(int t=0;t<N;++t) {
            const float input=x[int64_t(t)*cols+col];
            values[t]+=gate*input;
            values[N+t]+=up*input;
        }
    }
    sum_columns(values);
    if constexpr(N==1||(N==3&&(Type==16||Type==17))) {
        if(threadIdx.x==0) {
            #pragma unroll
            for(int t=0;t<N;++t) {
                const float g=fminf(bf16(values[t]),10.f);
                const float u=fminf(fmaxf(bf16(values[N+t]),-10.f),10.f);
                const float sig=g>=0?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));
                activation[int64_t(t)*rows+row]=bf16(g*sig*u*(routing?routing[t]:1.f));
            }
        }
    } else if(threadIdx.x<32) {
        float gate=0,up=0;
        #pragma unroll
        for(int t=0;t<N;++t) {
            const float g=__shfl_sync(0xffffffff,values[t],0);
            const float u=__shfl_sync(0xffffffff,values[N+t],0);
            if(threadIdx.x==t){gate=g;up=u;}
        }
        if(threadIdx.x<N) {
            const float g=fminf(bf16(gate),10.f);
            const float u=fminf(fmaxf(bf16(up),-10.f),10.f);
            const float sig=g>=0?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));
            activation[int64_t(threadIdx.x)*rows+row]=bf16(g*sig*u*(routing?routing[threadIdx.x]:1.f));
        }
    }
}

template<int Type>
void launch_gu_activation(const void* g,const void* u,const float* x,const float* routing,
    float* activation,int rows,int cols,int tokens,cudaStream_t stream) {
    #define RUN(N) case N:gu_activation_kernel<Type,N><<<rows,256,0,stream>>>(g,u,x,routing,activation,rows,cols);break
    switch(tokens){RUN(1);RUN(2);RUN(3);RUN(4);RUN(5);RUN(6);RUN(7);RUN(8);}
    #undef RUN
}

// Parameter memory holds only descriptors. Inputs/outputs remain in the existing
// workspace; each group retains its own token count and original K/reduction order.
struct ShortBatch { VerifyExpertGroup group[48]; };
template<int Type,int N,bool Fused>
__global__ void short_gu_kernel(ShortBatch batch,size_t up_offset,const float* x,
    const int* token_rows,const float* routing,float* gate,float* up,float* activation) {
    const auto g=batch.group[blockIdx.y];
    const int row=blockIdx.x;
    const auto* packed=static_cast<const uint8_t*>(g.packed);
    float values[2*N]{};
    for(int col=threadIdx.x;col<5120;col+=256) {
        const int64_t index=int64_t(row)*5120+col;
        const float wg=weight<Type>(packed,index),wu=weight<Type>(packed+up_offset,index);
        #pragma unroll
        for(int t=0;t<N;++t) {
            const float input=x[int64_t(token_rows[g.offset+t])*5120+col];
            values[t]+=wg*input;
            values[N+t]+=wu*input;
        }
    }
    sum_columns(values);
    if constexpr(!Fused) {
        if(threadIdx.x==0) {
            #pragma unroll
            for(int t=0;t<N;++t) {
                gate[int64_t(g.offset+t)*2304+row]=bf16(values[t]);
                up[int64_t(g.offset+t)*2304+row]=bf16(values[N+t]);
            }
        }
    } else if constexpr(N==1||(N==3&&(Type==16||Type==17))) {
        if(threadIdx.x==0) {
            #pragma unroll
            for(int t=0;t<N;++t) {
                const float v=fminf(bf16(values[t]),10.f);
                const float u=fminf(fmaxf(bf16(values[N+t]),-10.f),10.f);
                const float sig=v>=0?1.f/(1.f+expf(-v)):expf(v)/(1.f+expf(v));
                activation[int64_t(g.offset+t)*2304+row]=bf16(v*sig*u*routing[g.offset+t]);
            }
        }
    } else if(threadIdx.x<32) {
        float v=0,u=0;
        #pragma unroll
        for(int t=0;t<N;++t) {
            const float vg=__shfl_sync(0xffffffff,values[t],0);
            const float vu=__shfl_sync(0xffffffff,values[N+t],0);
            if(threadIdx.x==t){v=vg;u=vu;}
        }
        if(threadIdx.x<N) {
            v=fminf(bf16(v),10.f);u=fminf(fmaxf(bf16(u),-10.f),10.f);
            const float sig=v>=0?1.f/(1.f+expf(-v)):expf(v)/(1.f+expf(v));
            activation[int64_t(g.offset+threadIdx.x)*2304+row]=bf16(v*sig*u*routing[g.offset+threadIdx.x]);
        }
    }
}
template<int N>
__global__ void short_activation_kernel(ShortBatch batch,const float* gate,const float* up,
    const float* routing,float* activation) {
    const auto group=batch.group[blockIdx.y];
    const int i=blockIdx.x*256+threadIdx.x;
    if(i>=N*2304)return;
    const int64_t at=int64_t(group.offset)*2304+i;
    const float g=fminf(gate[at],10.f),u=fminf(fmaxf(up[at],-10.f),10.f);
    const float sig=g>=0?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));
    activation[at]=bf16(g*sig*u*routing[group.offset+i/2304]);
}
template<int Type,int N>
__global__ void short_down_kernel(ShortBatch batch,size_t down_offset,const float* activation,
    const int* slots,float* parts) {
    const auto g=batch.group[blockIdx.y];const int row=blockIdx.x;
    const auto* packed=static_cast<const uint8_t*>(g.packed)+down_offset;
    float values[N]{};
    for(int col=threadIdx.x;col<2304;col+=256) {
        const float w=weight<Type>(packed,int64_t(row)*2304+col);
        #pragma unroll
        for(int t=0;t<N;++t)values[t]+=w*activation[int64_t(g.offset+t)*2304+col];
    }
    sum_columns(values);
    if(threadIdx.x==0) {
        #pragma unroll
        for(int t=0;t<N;++t)parts[int64_t(slots[g.offset+t])*5120+row]=bf16(values[t]);
    }
}
// Four adjacent virtual lanes per physical thread; retain each lane's K order
// and the original 256-thread, two-level reduction tree.
template<int Count>
__device__ void short_sum_virtual(float (&values)[Count][4],float (&sums)[Count]) {
    __shared__ float partial[Count][8];
    for(int delta=4;delta;delta>>=1) {
        #pragma unroll
        for(int c=0;c<Count;++c) {
            #pragma unroll
            for(int j=0;j<4;++j)values[c][j]+=__shfl_down_sync(0xffffffff,values[c][j],delta,8);
        }
    }
    if((threadIdx.x&7)==0) {
        #pragma unroll
        for(int c=0;c<Count;++c)partial[c][threadIdx.x>>3]=(values[c][0]+values[c][2])+(values[c][1]+values[c][3]);
    }
    __syncthreads();
    #pragma unroll
    for(int c=0;c<Count;++c)sums[c]=threadIdx.x<8?partial[c][threadIdx.x]:0.f;
    if(threadIdx.x<32)for(int delta=16;delta;delta>>=1) {
        #pragma unroll
        for(int c=0;c<Count;++c)sums[c]+=__shfl_down_sync(0xffffffff,sums[c],delta);
    }
}
template<int Type,bool Fused>
__global__ void short_gu_vector_kernel(ShortBatch batch,size_t up_offset,const float* x,
    const int* token_rows,const float* routing,float* gate,float* up,float* activation) {
    const auto group=batch.group[blockIdx.y];const int row=blockIdx.x;
    const auto* packed=static_cast<const uint8_t*>(group.packed);
    float values[2][4]{};
    for(int col=threadIdx.x*4;col<5120;col+=256) {
        const float4 input=*reinterpret_cast<const float4*>(x+int64_t(token_rows[group.offset])*5120+col);
        const float inputs[4]={input.x,input.y,input.z,input.w};
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int64_t at=int64_t(row)*5120+col+j;
            values[0][j]+=weight<Type>(packed,at)*inputs[j];
            values[1][j]+=weight<Type>(packed+up_offset,at)*inputs[j];
        }
    }
    float sums[2];short_sum_virtual(values,sums);
    if(threadIdx.x==0) {
        const int64_t out=int64_t(group.offset)*2304+row;
        if constexpr(!Fused){gate[out]=bf16(sums[0]);up[out]=bf16(sums[1]);}
        else {
            const float v=fminf(bf16(sums[0]),10.f),u=fminf(fmaxf(bf16(sums[1]),-10.f),10.f);
            const float sig=v>=0?1.f/(1.f+expf(-v)):expf(v)/(1.f+expf(v));
            activation[out]=bf16(v*sig*u*routing[group.offset]);
        }
    }
}
template<int Type>
__global__ void short_down_vector_kernel(ShortBatch batch,size_t down_offset,const float* activation,
    const int* slots,float* parts) {
    const auto group=batch.group[blockIdx.y];const int row=blockIdx.x;
    const auto* packed=static_cast<const uint8_t*>(group.packed)+down_offset;
    float values[1][4]{};
    for(int col=threadIdx.x*4;col<2304;col+=256) {
        const float4 input=*reinterpret_cast<const float4*>(activation+int64_t(group.offset)*2304+col);
        const float inputs[4]={input.x,input.y,input.z,input.w};
        #pragma unroll
        for(int j=0;j<4;++j)values[0][j]+=weight<Type>(packed,int64_t(row)*2304+col+j)*inputs[j];
    }
    float sums[1];short_sum_virtual(values,sums);
    if(threadIdx.x==0)parts[int64_t(slots[group.offset])*5120+row]=bf16(sums[0]);
}
void launch_short_vector(ShortBatch batch,int count,uint32_t gu_type,uint32_t down_type,
    size_t up_offset,size_t down_offset,const float* x,const int* token_rows,const float* routing,
    const int* slots,float* gate,float* up,float* activation,float* parts,cudaStream_t stream) {
    const bool fused=expert_mixed_gu_activation_selected(gu_type,1);
    #define GU(T) case T:if(fused)short_gu_vector_kernel<T,true><<<dim3(2304,count),64,0,stream>>>(batch,up_offset,x,token_rows,routing,gate,up,activation);else short_gu_vector_kernel<T,false><<<dim3(2304,count),64,0,stream>>>(batch,up_offset,x,token_rows,routing,gate,up,activation);break
    switch(gu_type){GU(16);GU(17);GU(18);GU(21);GU(22);GU(39);}
    #undef GU
    if(!fused)short_activation_kernel<1><<<dim3(9,count),256,0,stream>>>(batch,gate,up,routing,activation);
    #define DOWN(T) case T:short_down_vector_kernel<T><<<dim3(5120,count),64,0,stream>>>(batch,down_offset,activation,slots,parts);break
    switch(down_type){DOWN(16);DOWN(17);DOWN(18);DOWN(21);DOWN(22);DOWN(39);}
    #undef DOWN
}

// Eight adjacent virtual lanes share scale/sign/codebook decoding. This helper
// is private to the uncovered N=1 path; existing vector and dequant paths stay.
template<int Type>
__device__ __forceinline__ void short_weight8(const void* packed,int64_t index,float (&out)[8]) {
    if constexpr(Type==16||Type==18||Type==21)weight8<Type>(packed,index,out);
    else if constexpr(Type==39) {
        #pragma unroll
        for(int j=0;j<8;++j)out[j]=weight<Type>(packed,index+j);
    } else {
        const int64_t block=index/256;
        const int ib=(index%256)/32,il=(index%32)/8;
        float d;uint8_t sign;uint64_t grid;
        if constexpr(Type==17) {
            const auto& b=static_cast<const block_iq2_xs*>(packed)[block];
            const uint16_t q=b.qs[4*ib+il];
            d=float(b.d)*(0.5f+((b.scales[ib]>>(4*(il/2)))&15))*0.25f;
            sign=ksigns_iq2xs[q>>9];grid=iq2xs_grid[q&511];
        } else {
            static_assert(Type==22);
            const auto& b=static_cast<const block_iq2_s*>(packed)[block];
            const int q=b.qs[4*ib+il]|((b.qh[ib]<<(8-2*il))&0x300);
            d=float(b.d)*(0.5f+((b.scales[ib]>>(4*(il/2)))&15))*0.25f;
            sign=b.qs[32+4*ib+il];grid=iq2s_grid[q];
        }
        #pragma unroll
        for(int j=0;j<8;++j)out[j]=bf16(d*int((grid>>(8*j))&255)*(sign&(1<<j)?-1.f:1.f));
    }
}

// The first two shuffles reproduce virtual deltas16/8; the register additions
// reproduce4/2/1. Gather the eight old warp totals and retain their old tree.
template<int Count>
__device__ __forceinline__ void short_sum_warp(float (&values)[Count][8],float (&sums)[Count]) {
    const int lane=threadIdx.x&31;
    for(int delta=2;delta;delta>>=1) {
        #pragma unroll
        for(int c=0;c<Count;++c) {
            #pragma unroll
            for(int j=0;j<8;++j)values[c][j]+=__shfl_down_sync(0xffffffff,values[c][j],delta,4);
        }
    }
    #pragma unroll
    for(int c=0;c<Count;++c) {
        const float partial=((values[c][0]+values[c][4])+(values[c][2]+values[c][6]))+
                            ((values[c][1]+values[c][5])+(values[c][3]+values[c][7]));
        const float gathered=__shfl_sync(0xffffffff,partial,(lane&7)*4);
        sums[c]=lane<8?gathered:0.f;
    }
    for(int delta=16;delta;delta>>=1) {
        #pragma unroll
        for(int c=0;c<Count;++c)sums[c]+=__shfl_down_sync(0xffffffff,sums[c],delta);
    }
}
template<int Type,bool Fused>
__global__ void short_gu_warp_kernel(ShortBatch batch,size_t up_offset,const float* x,
    const int* token_rows,const float* routing,float* gate,float* up,float* activation) {
    const auto group=batch.group[blockIdx.y];const int row=blockIdx.x*4+(threadIdx.x>>5);
    const int lane=threadIdx.x&31;
    const auto* packed=static_cast<const uint8_t*>(group.packed);
    float values[2][8]{};
    for(int col=lane*8;col<5120;col+=256) {
        const float* input=x+int64_t(token_rows[group.offset])*5120+col;
        const float4 a=*reinterpret_cast<const float4*>(input),b=*reinterpret_cast<const float4*>(input+4);
        const float inputs[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
        float g[8],u[8];
        short_weight8<Type>(packed,int64_t(row)*5120+col,g);
        short_weight8<Type>(packed+up_offset,int64_t(row)*5120+col,u);
        #pragma unroll
        for(int j=0;j<8;++j){values[0][j]+=g[j]*inputs[j];values[1][j]+=u[j]*inputs[j];}
    }
    float sums[2];short_sum_warp(values,sums);
    if(lane==0) {
        const int64_t out=int64_t(group.offset)*2304+row;
        if constexpr(!Fused){gate[out]=bf16(sums[0]);up[out]=bf16(sums[1]);}
        else {
            const float v=fminf(bf16(sums[0]),10.f),u=fminf(fmaxf(bf16(sums[1]),-10.f),10.f);
            const float sig=v>=0?1.f/(1.f+expf(-v)):expf(v)/(1.f+expf(v));
            activation[out]=bf16(v*sig*u*routing[group.offset]);
        }
    }
}
template<int Type>
__global__ void short_down_warp_kernel(ShortBatch batch,size_t down_offset,const float* activation,
    const int* slots,float* parts) {
    const auto group=batch.group[blockIdx.y];const int row=blockIdx.x*4+(threadIdx.x>>5);
    const int lane=threadIdx.x&31;
    const auto* packed=static_cast<const uint8_t*>(group.packed)+down_offset;
    float values[1][8]{};
    for(int col=lane*8;col<2304;col+=256) {
        const float* input=activation+int64_t(group.offset)*2304+col;
        const float4 a=*reinterpret_cast<const float4*>(input),b=*reinterpret_cast<const float4*>(input+4);
        const float inputs[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
        float weights[8];short_weight8<Type>(packed,int64_t(row)*2304+col,weights);
        #pragma unroll
        for(int j=0;j<8;++j)values[0][j]+=weights[j]*inputs[j];
    }
    float sums[1];short_sum_warp(values,sums);
    if(lane==0)parts[int64_t(slots[group.offset])*5120+row]=bf16(sums[0]);
}
void launch_short_warp(ShortBatch batch,int count,uint32_t gu_type,uint32_t down_type,
    size_t up_offset,size_t down_offset,const float* x,const int* token_rows,const float* routing,
    const int* slots,float* gate,float* up,float* activation,float* parts,cudaStream_t stream) {
    const bool fused=expert_mixed_gu_activation_selected(gu_type,1);
    #define GU(T) case T:if(fused)short_gu_warp_kernel<T,true><<<dim3(576,count),128,0,stream>>>(batch,up_offset,x,token_rows,routing,gate,up,activation);else short_gu_warp_kernel<T,false><<<dim3(576,count),128,0,stream>>>(batch,up_offset,x,token_rows,routing,gate,up,activation);break
    switch(gu_type){GU(16);GU(17);GU(18);GU(21);GU(22);GU(39);}
    #undef GU
    if(!fused)short_activation_kernel<1><<<dim3(9,count),256,0,stream>>>(batch,gate,up,routing,activation);
    #define DOWN(T) case T:short_down_warp_kernel<T><<<dim3(1280,count),128,0,stream>>>(batch,down_offset,activation,slots,parts);break
    switch(down_type){DOWN(16);DOWN(17);DOWN(18);DOWN(21);DOWN(22);DOWN(39);}
    #undef DOWN
}

// A warp retains the eight virtual-lane accumulators of one output row.
// Two token columns share decoding without the register growth of N columns.
template<int Type,int N,bool Pair,int Tile=(N==1?1:2)>
__global__ void projection_warp_kernel(const void* w,const void* wu,const float* x,float* y,float* yu,
    int rows,int cols,const int* slots) {
    const int row=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;
    const int token=blockIdx.y*Tile;
    if(row>=rows)return;
    float values[Tile*(Pair?2:1)][8]{};
    for(int col=lane*8;col<cols;col+=256) {
        float gate[8],up[8];
        short_weight8<Type>(w,int64_t(row)*cols+col,gate);
        if constexpr(Pair)short_weight8<Type>(wu,int64_t(row)*cols+col,up);
        #pragma unroll
        for(int t=0;t<Tile;++t)if(token+t<N) {
            const float* input=x+int64_t(token+t)*cols+col;
            const float4 a=*reinterpret_cast<const float4*>(input),b=*reinterpret_cast<const float4*>(input+4);
            const float v[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
            #pragma unroll
            for(int j=0;j<8;++j) {
                values[t][j]+=gate[j]*v[j];
                if constexpr(Pair)values[Tile+t][j]+=up[j]*v[j];
            }
        }
    }
    float sums[Tile*(Pair?2:1)];short_sum_warp(values,sums);
    if(lane==0) {
        #pragma unroll
        for(int t=0;t<Tile;++t)if(token+t<N) {
            const int destination=slots?slots[token+t]:token+t;
            y[int64_t(destination)*rows+row]=bf16(sums[t]);
            if constexpr(Pair)yu[int64_t(token+t)*rows+row]=bf16(sums[Tile+t]);
        }
    }
}
template<int Type,int N,bool Fused,int Tile=2>
__global__ void short_gu_tokens_kernel(ShortBatch batch,size_t up_offset,const float* x,
    const int* token_rows,const float* routing,float* gate,float* up,float* activation) {
    const auto group=batch.group[blockIdx.y];
    const int row=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31,token=blockIdx.z*Tile;
    const auto* packed=static_cast<const uint8_t*>(group.packed);
    float values[2*Tile][8]{};
    for(int col=lane*8;col<5120;col+=256) {
        float g[8],u[8];
        short_weight8<Type>(packed,int64_t(row)*5120+col,g);
        short_weight8<Type>(packed+up_offset,int64_t(row)*5120+col,u);
        #pragma unroll
        for(int t=0;t<Tile;++t)if(token+t<N) {
            const float* input=x+int64_t(token_rows[group.offset+token+t])*5120+col;
            const float4 a=*reinterpret_cast<const float4*>(input),b=*reinterpret_cast<const float4*>(input+4);
            const float v[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
            #pragma unroll
            for(int j=0;j<8;++j){values[t][j]+=g[j]*v[j];values[Tile+t][j]+=u[j]*v[j];}
        }
    }
    float sums[2*Tile];short_sum_warp(values,sums);
    if(lane==0) {
        #pragma unroll
        for(int t=0;t<Tile;++t)if(token+t<N) {
            const int64_t out=int64_t(group.offset+token+t)*2304+row;
            if constexpr(!Fused){gate[out]=bf16(sums[t]);up[out]=bf16(sums[Tile+t]);}
            else {
                const float v=fminf(bf16(sums[t]),10.f),u=fminf(fmaxf(bf16(sums[Tile+t]),-10.f),10.f);
                const float sig=v>=0?1.f/(1.f+expf(-v)):expf(v)/(1.f+expf(v));
                activation[out]=bf16(v*sig*u*routing[group.offset+token+t]);
            }
        }
    }
}
template<int Type,int N,int Tile=2>
__global__ void short_down_tokens_kernel(ShortBatch batch,size_t down_offset,const float* activation,
    const int* slots,float* parts) {
    const auto group=batch.group[blockIdx.y];
    const int row=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31,token=blockIdx.z*Tile;
    const auto* packed=static_cast<const uint8_t*>(group.packed)+down_offset;
    float values[Tile][8]{};
    for(int col=lane*8;col<2304;col+=256) {
        float weights[8];short_weight8<Type>(packed,int64_t(row)*2304+col,weights);
        #pragma unroll
        for(int t=0;t<Tile;++t)if(token+t<N) {
            const float* input=activation+int64_t(group.offset+token+t)*2304+col;
            const float4 a=*reinterpret_cast<const float4*>(input),b=*reinterpret_cast<const float4*>(input+4);
            const float v[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
            #pragma unroll
            for(int j=0;j<8;++j)values[t][j]+=weights[j]*v[j];
        }
    }
    float sums[Tile];short_sum_warp(values,sums);
    if(lane==0) {
        #pragma unroll
        for(int t=0;t<Tile;++t)if(token+t<N)
            parts[int64_t(slots[group.offset+token+t])*5120+row]=bf16(sums[t]);
    }
}
template<int Type,int N>
void launch_short_down(ShortBatch batch,int count,size_t down_offset,const float* activation,
    const int* slots,float* parts,cudaStream_t stream) {
    if constexpr(N>=2)if(short_down_tokens_enabled.load()&&expert_mixed_short_down_tokens_selected(Type,N,count)&&
        (reinterpret_cast<uintptr_t>(activation)&15)==0) {
        short_down_tokens_kernel<Type,N><<<dim3(1280,count,(N+1)/2),128,0,stream>>>(batch,down_offset,activation,slots,parts);
        return;
    }
    short_down_kernel<Type,N><<<dim3(5120,count),256,0,stream>>>(batch,down_offset,activation,slots,parts);
}
template<int Type,int N>
void launch_short_gu(ShortBatch batch,int count,size_t up_offset,const float* x,
    const int* token_rows,const float* routing,float* gate,float* up,float* activation,bool fused,cudaStream_t stream) {
    if constexpr(N>=2)if(short_tokens_enabled.load()&&expert_mixed_short_tokens_selected(Type,N,count)&&
        (reinterpret_cast<uintptr_t>(x)&15)==0) {
        if(fused)short_gu_tokens_kernel<Type,N,true><<<dim3(576,count,(N+1)/2),128,0,stream>>>(batch,up_offset,x,token_rows,routing,gate,up,activation);
        else short_gu_tokens_kernel<Type,N,false><<<dim3(576,count,(N+1)/2),128,0,stream>>>(batch,up_offset,x,token_rows,routing,gate,up,activation);
        return;
    }
    if(fused)short_gu_kernel<Type,N,true><<<dim3(2304,count),256,0,stream>>>(batch,up_offset,x,token_rows,routing,gate,up,activation);
    else short_gu_kernel<Type,N,false><<<dim3(2304,count),256,0,stream>>>(batch,up_offset,x,token_rows,routing,gate,up,activation);
}
template<int N>
void launch_short(ShortBatch batch,int count,uint32_t gu_type,uint32_t down_type,
    size_t up_offset,size_t down_offset,const float* x,const int* token_rows,const float* routing,
    const int* slots,float* gate,float* up,float* activation,float* parts,cudaStream_t stream) {
    const bool fused=expert_mixed_gu_activation_selected(gu_type,N);
    #define GU(T) case T:launch_short_gu<T,N>(batch,count,up_offset,x,token_rows,routing,gate,up,activation,fused,stream);break
    switch(gu_type){GU(16);GU(17);GU(18);GU(21);GU(22);GU(39);}
    #undef GU
    if(!fused)short_activation_kernel<N><<<dim3((N*2304+255)/256,count),256,0,stream>>>(batch,gate,up,routing,activation);
    #define DOWN(T) case T:launch_short_down<T,N>(batch,count,down_offset,activation,slots,parts,stream);break
    switch(down_type){DOWN(16);DOWN(17);DOWN(18);DOWN(21);DOWN(22);DOWN(39);}
    #undef DOWN
}

// Eight independent 256-value blocks per CTA, coalesced BF16 pair stores.
template<int Type>
__global__ void dequant_kernel(const void* w,uint16_t* output,int64_t n) {
    const int64_t block=int64_t(blockIdx.x)*8+threadIdx.x/32;
    const int lane=threadIdx.x&31;
    if(block*256>=n)return;
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int64_t i=block*256+lane*2+j*64;
        const auto a=__float2bfloat16_rn(weight<Type>(w,i));
        const auto b=__float2bfloat16_rn(weight<Type>(w,i+1));
        reinterpret_cast<uint32_t*>(output)[i/2]=uint32_t(__bfloat16_as_ushort(a))|
            (uint32_t(__bfloat16_as_ushort(b))<<16);
    }
}

template<int Type,bool Pair>
void launch(const void* w,const void* wu,const float* x,float* y,float* yu,
            int rows,int cols,int tokens,const int* slots,cudaStream_t stream) {
    if(projection_warp_enabled.load()&&expert_mixed_projection_warp_selected(Type,rows,cols,tokens,Pair)&&
       (reinterpret_cast<uintptr_t>(x)&15)==0) {
        #define FAST(N) case N:projection_warp_kernel<Type,N,Pair><<<dim3((rows+3)/4,(N+1)/2),128,0,stream>>>(w,wu,x,y,yu,rows,cols,slots);break
        switch(tokens){FAST(1);FAST(2);FAST(3);FAST(4);FAST(5);FAST(6);FAST(7);FAST(8);}
        #undef FAST
        return;
    }
    #define RUN(N) case N:projection_kernel<Type,N,Pair><<<rows,256,0,stream>>>(w,wu,x,y,yu,rows,cols,slots);break
    switch(tokens){RUN(1);RUN(2);RUN(3);RUN(4);RUN(5);RUN(6);RUN(7);RUN(8);}
    #undef RUN
}
template<bool Pair>
cudaError_t dispatch(const void* w,const void* wu,uint32_t type,const float* x,float* y,float* yu,
                     int rows,int cols,int tokens,const int* slots,cudaStream_t stream) {
    if(!w||!x||!y||(Pair&&(!wu||!yu))||rows<1||cols<1||cols%256||tokens<1||tokens>8)
        return cudaErrorInvalidValue;
    #define RUN(T) case T:launch<T,Pair>(w,wu,x,y,yu,rows,cols,tokens,slots,stream);break
    switch(type){RUN(16);RUN(17);RUN(18);RUN(21);RUN(22);RUN(39);default:return cudaErrorInvalidValue;}
    #undef RUN
    return cudaGetLastError();
}
} // namespace

void expert_mixed_fast_set_enabled(bool value){enabled.store(value);}
bool expert_mixed_fast_enabled(){return enabled.load();}
void expert_mixed_short_set_enabled(bool value){short_enabled.store(value);}
bool expert_mixed_short_enabled(){return short_enabled.load();}
void expert_mixed_short_vector_set_enabled(bool value){short_vector_enabled.store(value);}
void expert_mixed_short_warp_set_enabled(bool value){short_warp_enabled.store(value);}
void expert_mixed_projection_warp_set_enabled(bool value){projection_warp_enabled.store(value);}
void expert_mixed_short_tokens_set_enabled(bool value){short_tokens_enabled.store(value);}
void expert_mixed_short_down_tokens_set_enabled(bool value){short_down_tokens_enabled.store(value);}
bool expert_mixed_short_down_tokens_selected(uint32_t type,int tokens,int groups) {
    // A type/N/group bin must pass every real GU/Down pair that uses it.
    if(tokens<2||tokens>8||groups<1||groups>48/tokens)return false;
    struct Choice {uint32_t type;uint64_t masks[7];};
    static constexpr Choice choices[]={
        {16,{33554430ull,131070ull,8190ull,1022ull,510ull,126ull,126ull}},
        {18,{33554430ull,131054ull,8190ull,1022ull,510ull,126ull,118ull}},
        {21,{33554430ull,131070ull,8190ull,1022ull,510ull,126ull,126ull}},
    };
    for(const auto& c:choices)if(c.type==type)return (c.masks[tokens-2]&(uint64_t(1)<<groups))!=0;
    return false;
}
bool expert_mixed_projection_warp_selected(uint32_t type,int rows,int cols,int tokens,bool pair) {
    // Real-weight median ABBA passed all IQ shapes; MXFP4 Down regressed.
    if(tokens<1||tokens>8)return false;
    return pair?(rows==2304&&cols==5120&&(type==16||type==17||type==18||type==22)):
                (rows==5120&&cols==2304&&(type==16||type==18||type==21));
}
bool expert_mixed_short_tokens_selected(uint32_t gu,int tokens,int groups) {
    // Preserve N1. IQ2_XS N5/group1 gained only3.60%; retain its old path.
    if(gu==17&&tokens==5&&groups==1)return false;
    return tokens>=2&&tokens<=8&&groups>=1&&groups*tokens<=48&&
           (gu==16||gu==17||gu==18||gu==22);
}
bool expert_mixed_short_warp_selected(uint32_t gu,uint32_t down,int groups) {
    // The original48 vector bins retain priority. These seven real format pairs
    // cover every independently checked group size1..48 in the new path.
    const bool pair=(gu==16&&(down==16||down==18))||(gu==17&&down==39)||
        (gu==18&&(down==18||down==21))||(gu==22&&(down==21||down==39));
    return pair&&groups>=1&&groups<=48&&!expert_mixed_short_vector_selected(gu,down,groups);
}
bool expert_mixed_short_vector_selected(uint32_t gu,uint32_t down,int groups) {
    if(groups<1||groups>48)return false;
    uint64_t mask=0;
    if(gu==16&&down==16)mask=285942833484152ull;
    if(gu==16&&down==18)mask=285942833484156ull;
    if(gu==17&&down==39)mask=285942833479680ull;
    if(gu==18&&down==21)mask=1090781184ull;
    if(gu==22&&down==21)mask=281544786972672ull;
    if(gu==22&&down==39)mask=285942833483840ull;
    return (mask&(uint64_t(1)<<groups))!=0;
}
bool expert_mixed_short_selected(uint32_t gu,uint32_t down,int tokens,int groups,bool uniform) {
    // Real-weight ABBA includes metadata copies. Zero keeps the cumulative path;
    // uniform means every resident expert is used by every token in this window.
    struct Choice { uint32_t gu,down;uint8_t uniform[8],mixed[8]; };
    static constexpr Choice choices[]={
        {17,39,{3,2,2,1,2,2,0,4},{3,3,3,1,3,3,12,3}},
        {16,16,{4,2,2,2,3,3,3,0},{4,3,4,4,12,12,12,4}},
        {16,18,{2,2,2,2,3,3,2,0},{2,2,3,3,4,4,12,4}},
        {22,39,{3,2,2,2,2,3,3,0},{3,3,3,3,3,3,12,3}},
        {22,21,{2,1,2,2,2,6,0,0},{2,1,3,3,3,3,0,4}},
        {18,21,{2,1,3,3,2,2,0,0},{2,1,3,3,12,0,0,0}},
        {18,18,{2,2,3,2,2,2,0,0},{2,2,4,3,0,0,0,0}}
    };
    if(tokens<1||tokens>8||groups<1||groups>48)return false;
    for(const auto& c:choices)if(c.gu==gu&&c.down==down) {
        const int minimum=(uniform?c.uniform:c.mixed)[tokens-1];
        return minimum&&groups>=minimum;
    }
    return false;
}
bool expert_mixed_type(uint32_t type){
    return type==16||type==17||type==18||type==21||type==22||type==39;
}
cudaError_t expert_mixed_short_gpu(const VerifyExpertGroup* groups,int count,
    uint32_t gu_type,uint32_t down_type,size_t up_offset,size_t down_offset,
    const float* x,const int* token_rows,const float* routing,const int* slots,
    float* gate,float* up,float* activation,float* parts,int tokens,cudaStream_t stream) {
    if(!groups||count<1||count>48||!expert_mixed_type(gu_type)||!expert_mixed_type(down_type)||
       !x||!token_rows||!routing||!slots||!gate||!up||!activation||!parts||tokens<1||tokens>8||
       !up_offset||down_offset<=up_offset)return cudaErrorInvalidValue;
    for(int i=0;i<count;++i)if(!groups[i].packed||groups[i].offset<0||groups[i].tokens<1||
       groups[i].tokens>tokens||groups[i].offset+groups[i].tokens>tokens*6)return cudaErrorInvalidValue;
    for(int n=1;n<=tokens;++n) {
        ShortBatch batch{};int size=0;
        for(int i=0;i<count;++i)if(groups[i].tokens==n)batch.group[size++]=groups[i];
        if(!size)continue;
        if(n==1&&short_vector_enabled.load()&&expert_mixed_short_vector_selected(gu_type,down_type,size)&&((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(activation))&15)==0) {
            launch_short_vector(batch,size,gu_type,down_type,up_offset,down_offset,x,token_rows,routing,slots,gate,up,activation,parts,stream);
            continue;
        }
        if(n==1&&short_warp_enabled.load()&&expert_mixed_short_warp_selected(gu_type,down_type,size)&&((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(activation))&15)==0) {
            launch_short_warp(batch,size,gu_type,down_type,up_offset,down_offset,x,token_rows,routing,slots,gate,up,activation,parts,stream);
            continue;
        }
        #define RUN(N) case N:launch_short<N>(batch,size,gu_type,down_type,up_offset,down_offset,x,token_rows,routing,slots,gate,up,activation,parts,stream);break
        switch(n){RUN(1);RUN(2);RUN(3);RUN(4);RUN(5);RUN(6);RUN(7);RUN(8);}
        #undef RUN
    }
    return cudaGetLastError();
}
cudaError_t expert_mixed_gu_gpu(const void* g,const void* u,uint32_t type,const float* x,
    float* gate,float* up,int rows,int cols,int tokens,cudaStream_t stream) {
    return dispatch<true>(g,u,type,x,gate,up,rows,cols,tokens,nullptr,stream);
}
cudaError_t expert_mixed_gu_activation_gpu(const void* g,const void* u,uint32_t type,const float* x,
    const float* routing,float* activation,int rows,int cols,int tokens,cudaStream_t stream) {
    if(!g||!u||!x||!activation||rows<1||cols<1||cols%256||tokens<1||tokens>8)
        return cudaErrorInvalidValue;
    #define RUN(T) case T:launch_gu_activation<T>(g,u,x,routing,activation,rows,cols,tokens,stream);break
    switch(type){RUN(16);RUN(17);RUN(18);RUN(21);RUN(22);RUN(39);default:return cudaErrorInvalidValue;}
    #undef RUN
    return cudaGetLastError();
}
cudaError_t expert_mixed_projection_gpu(const void* w,uint32_t type,const float* x,
    float* out,int rows,int cols,int tokens,const int* slots,cudaStream_t stream) {
    return dispatch<false>(w,nullptr,type,x,out,nullptr,rows,cols,tokens,slots,stream);
}
cudaError_t expert_mixed_swiglu_gpu(const float* gate,const float* up,const float* routing,
    float* activation,int width,int tokens,cudaStream_t stream) {
    if(!gate||!up||!activation||width<1||tokens<1)return cudaErrorInvalidValue;
    const int64_t n=int64_t(width)*tokens;
    const int blocks=int((n+255)/256<65535?(n+255)/256:65535);
    activation_kernel<<<blocks,256,0,stream>>>(gate,up,routing,activation,width,n);
    return cudaGetLastError();
}
cudaError_t expert_mixed_down_gpu(const void* w,uint32_t type,const float* gate,const float* up,
    const float* routing,float* activation,float* out,int rows,int cols,int tokens,
    const int* slots,cudaStream_t stream) {
    if(!w||!out||!expert_mixed_type(type)||rows<1||cols<1||cols%256||tokens<1||tokens>8)
        return cudaErrorInvalidValue;
    const auto status=expert_mixed_swiglu_gpu(gate,up,routing,activation,cols,tokens,stream);
    if(status!=cudaSuccess)return status;
    return expert_mixed_projection_gpu(w,type,activation,out,rows,cols,tokens,slots,stream);
}
cudaError_t expert_mixed_dequant_gpu(const void* w,uint32_t type,uint16_t* out,
    int64_t n,cudaStream_t stream) {
    if(!w||!out||n<1||n%256)return cudaErrorInvalidValue;
    const unsigned blocks=unsigned((n+2047)/2048);
    #define RUN(T) case T:dequant_kernel<T><<<blocks,256,0,stream>>>(w,out,n);break
    #define VECTOR(T) case T:dequant_vector_kernel<T><<<unsigned((n+1023)/1024),128,0,stream>>>(w,out,n);break
    switch(type){VECTOR(16);RUN(17);VECTOR(18);VECTOR(21);RUN(22);RUN(39);default:return cudaErrorInvalidValue;}
    #undef VECTOR
    #undef RUN
    return cudaGetLastError();
}
} // namespace strata::models::deepseek41
