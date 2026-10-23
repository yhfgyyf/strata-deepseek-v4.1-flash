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
template<int N>
void launch_short(ShortBatch batch,int count,uint32_t gu_type,uint32_t down_type,
    size_t up_offset,size_t down_offset,const float* x,const int* token_rows,const float* routing,
    const int* slots,float* gate,float* up,float* activation,float* parts,cudaStream_t stream) {
    const bool fused=expert_mixed_gu_activation_selected(gu_type,N);
    #define GU(T) case T:if(fused)short_gu_kernel<T,N,true><<<dim3(2304,count),256,0,stream>>>(batch,up_offset,x,token_rows,routing,gate,up,activation);else short_gu_kernel<T,N,false><<<dim3(2304,count),256,0,stream>>>(batch,up_offset,x,token_rows,routing,gate,up,activation);break
    switch(gu_type){GU(16);GU(17);GU(18);GU(21);GU(22);GU(39);}
    #undef GU
    if(!fused)short_activation_kernel<N><<<dim3((N*2304+255)/256,count),256,0,stream>>>(batch,gate,up,routing,activation);
    #define DOWN(T) case T:short_down_kernel<T,N><<<dim3(5120,count),256,0,stream>>>(batch,down_offset,activation,slots,parts);break
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
