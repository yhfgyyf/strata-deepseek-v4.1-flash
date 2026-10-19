#include "strata/models/deepseek41/format_down.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>

namespace strata::models::deepseek41 {
namespace {
struct Q81{half2 ds;int8_t qs[32];};
static_assert(sizeof(Q81)==36);
__device__ float activation(float g,float u,float weight){
    g=fminf(g,10.f);u=fminf(fmaxf(u,-10.f),10.f);
    const float sigmoid=g>=0?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));
    return __bfloat162float(__float2bfloat16_rn(g*sigmoid*u*weight));
}
template<bool Q8> __global__ void prepare(const float* gate,const float* up,const float* routing,void* output,int cols,int count){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const float x=activation(gate[i],up[i],routing?routing[i/cols]:1.f);
    if constexpr(!Q8)static_cast<__nv_bfloat16*>(output)[i]=__float2bfloat16_rn(x);
    else{
        float maximum=fabsf(x),sum=x;
        for(int d=16;d;d>>=1){maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));sum+=__shfl_xor_sync(0xffffffff,sum,d);}
        const float scale=kernels::q8_1_finite(maximum*(1.f/127.f));auto* q=static_cast<Q81*>(output);
        // native_mmvq's quantizer is compiled with --use_fast_math. Match its
        // division at half-integer ties without changing SwiGLU's precise expf.
        const float rounded=roundf(__fdividef(x,scale));
        q[i/32].qs[i%32]=maximum==0?0:static_cast<int8_t>(rounded>127?127:rounded<-127?-127:rounded);
        if(i%32==0)q[i/32].ds=kernels::q8_1_ds(scale,sum);
    }
}
__device__ float q2_bf16(const uint8_t* row,int col){
    const uint8_t* block=row+(col/256)*84;const int i=col%256,scale=block[i/16];
    const int code=(block[16+i/128*32+i%32]>>(i%128/32*2))&3;
    const float value=(__half2float(*reinterpret_cast<const half*>(block+80))*(scale&15))*code-
                      __half2float(*reinterpret_cast<const half*>(block+82))*(scale>>4);
    return __bfloat162float(__float2bfloat16_rn(value));
}
__global__ void q2_down(const uint8_t* packed,const __nv_bfloat16* x,float* output,const int* slots,int rows,int cols){
    const int row=blockIdx.x,token=blockIdx.y;float value=0;
    const uint8_t* weight=packed+int64_t(row)*(cols/256)*84;
    for(int col=threadIdx.x;col<cols;col+=256)value+=q2_bf16(weight,col)*__bfloat162float(x[int64_t(token)*cols+col]);
    for(int d=16;d;d>>=1)value+=__shfl_down_sync(0xffffffff,value,d);
    __shared__ float partial[8];
    if(threadIdx.x%32==0)partial[threadIdx.x/32]=value;
    __syncthreads();value=threadIdx.x<8?partial[threadIdx.x]:0;
    if(threadIdx.x<32)for(int d=16;d;d>>=1)value+=__shfl_down_sync(0xffffffff,value,d);
    if(threadIdx.x==0)output[int64_t(slots?slots[token]:token)*rows+row]=__bfloat162float(__float2bfloat16_rn(value));
}
template<int N,int R,int K=0>
__device__ void q2_verify_values(const uint8_t* packed,const __nv_bfloat16* x,float* output,
                               const int* slots,int rows,int width){
    const int cols=K?K:width;
    const int row0=blockIdx.x*R;float values[R][N]{};
    #pragma unroll
    for(int col=threadIdx.x;col<cols;col+=256){
        float weights[R];
        #pragma unroll
        for(int r=0;r<R;++r)weights[r]=row0+r<rows?q2_bf16(packed+int64_t(row0+r)*(cols/256)*84,col):0.f;
        #pragma unroll
        for(int t=0;t<N;++t){const float input=__bfloat162float(x[int64_t(t)*cols+col]);
            #pragma unroll
            for(int r=0;r<R;++r)values[r][t]+=weights[r]*input;
        }
    }
    __shared__ float partial[R][N][8];
    for(int d=16;d;d>>=1){
        #pragma unroll
        for(int r=0;r<R;++r){
            #pragma unroll
            for(int t=0;t<N;++t)values[r][t]+=__shfl_down_sync(0xffffffff,values[r][t],d);
        }
    }
    if(threadIdx.x%32==0){
        #pragma unroll
        for(int r=0;r<R;++r){
            #pragma unroll
            for(int t=0;t<N;++t)partial[r][t][threadIdx.x/32]=values[r][t];
        }
    }
    __syncthreads();
    #pragma unroll
    for(int r=0;r<R;++r){
        #pragma unroll
        for(int t=0;t<N;++t)values[r][t]=threadIdx.x<8?partial[r][t][threadIdx.x]:0.f;
    }
    if(threadIdx.x<32)for(int d=16;d;d>>=1){
        #pragma unroll
        for(int r=0;r<R;++r){
            #pragma unroll
            for(int t=0;t<N;++t)values[r][t]+=__shfl_down_sync(0xffffffff,values[r][t],d);
        }
    }
    if(threadIdx.x==0){
        #pragma unroll
        for(int r=0;r<R;++r)if(row0+r<rows){
            #pragma unroll
            for(int t=0;t<N;++t)output[int64_t(slots?slots[t]:t)*rows+row0+r]=__bfloat162float(__float2bfloat16_rn(values[r][t]));
        }
    }
}
template<int N,int R,int K=0>
__global__ void q2_verify_down(const uint8_t* packed,const __nv_bfloat16* x,float* output,
                               const int* slots,int rows,int width){
    q2_verify_values<N,R,K>(packed,x,output,slots,rows,width);
}
struct DownBatch { VerifyExpertGroup groups[48]; };
template<int N>
__global__ void q2_grouped_down(DownBatch batch,const __nv_bfloat16* x,float* output,const int* slots){
    const auto g=batch.groups[blockIdx.y];
    q2_verify_values<N,(N<4?2:4),2304>(static_cast<const uint8_t*>(g.packed),x+int64_t(g.offset)*2304,
        output,slots+g.offset,5120,2304);
}
template<bool Q8>
__global__ void prepare_grouped(DownBatch batch,const float* gate,const float* up,const float* routing,void* output){
    const auto g=batch.groups[blockIdx.y];
    const int local=blockIdx.x*blockDim.x+threadIdx.x;if(local>=g.tokens*2304)return;
    const int i=g.offset*2304+local;const float x=activation(gate[i],up[i],routing[i/2304]);
    if constexpr(!Q8)static_cast<__nv_bfloat16*>(output)[i]=__float2bfloat16_rn(x);
    else{
        float maximum=fabsf(x),sum=x;
        for(int d=16;d;d>>=1){maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));sum+=__shfl_xor_sync(0xffffffff,sum,d);}
        const float scale=kernels::q8_1_finite(maximum*(1.f/127.f));auto* q=static_cast<Q81*>(output);
        const float rounded=roundf(__fdividef(x,scale));
        q[i/32].qs[i%32]=maximum==0?0:static_cast<int8_t>(rounded>127?127:rounded<-127?-127:rounded);
        if(i%32==0)q[i/32].ds=kernels::q8_1_ds(scale,sum);
    }
}
template<int N>
void q2_verify_launch_n(const uint8_t* p,const __nv_bfloat16* x,float* out,const int* slots,
                        int rows,int cols,cudaStream_t stream){
    // Reuse each activation across adjacent output rows. Keep each row's
    // column accumulation and 256-thread reduction in their original order.
    constexpr int R=N<4?2:4;
    if(cols==2304)q2_verify_down<N,R,2304><<<(rows+R-1)/R,256,0,stream>>>(p,x,out,slots,rows,cols);
    else q2_verify_down<N,1><<<rows,256,0,stream>>>(p,x,out,slots,rows,cols);
}
void q2_verify_launch(const uint8_t* p,const __nv_bfloat16* x,float* out,const int* slots,
                      int rows,int cols,int tokens,cudaStream_t stream){
    #define VERIFY_DOWN(N) case N:q2_verify_launch_n<N>(p,x,out,slots,rows,cols,stream);break
    switch(tokens){VERIFY_DOWN(2);VERIFY_DOWN(3);VERIFY_DOWN(4);VERIFY_DOWN(5);VERIFY_DOWN(6);VERIFY_DOWN(7);VERIFY_DOWN(8);}
    #undef VERIFY_DOWN
}

__global__ void finish(const float* input,float* output,const int* slots,int rows,int count){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)
        output[int64_t(slots?slots[i/rows]:i/rows)*rows+i%rows]=__bfloat162float(__float2bfloat16_rn(input[i]));
}
}
cudaError_t format_down_grouped_gpu(const VerifyExpertGroup* groups,int count,uint32_t type,
    const float* gate,const float* up,const float* routing,float* output,const int* slots,
    void* scratch,size_t bytes,bool prepared_q2,cudaStream_t stream){
    if(!groups||count<1||count>48||(type!=10&&type!=11)||!routing||!output||!slots||!scratch||!stream||
       (prepared_q2?type!=10:(!gate||!up)))return cudaErrorInvalidValue;
    DownBatch all{};bool used[48]{};int max_rows=0,max_tokens=0;
    for(int i=0;i<count;++i){const auto& g=groups[i];
        if(!g.packed||g.offset<0||g.tokens<1||g.tokens>8||g.offset>48-g.tokens)return cudaErrorInvalidValue;
        for(int t=0;t<g.tokens;++t){if(used[g.offset+t])return cudaErrorInvalidValue;used[g.offset+t]=true;}
        max_rows=std::max(max_rows,g.offset+g.tokens);max_tokens=std::max(max_tokens,g.tokens);all.groups[i]=g;
    }
    if(bytes<size_t(max_rows)*2304*(type==10?2:36)/size_t(type==10?1:32))return cudaErrorInvalidValue;
    if(!prepared_q2){
        if(type==10)prepare_grouped<false><<<dim3(max_tokens*9,count),256,0,stream>>>(all,gate,up,routing,scratch);
        else prepare_grouped<true><<<dim3(max_tokens*9,count),256,0,stream>>>(all,gate,up,routing,scratch);
        const auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    }
    if(type==11){
        kernels::NativeQ3DownGroup native[48]{};
        for(int i=0;i<count;++i)native[i]={groups[i].packed,groups[i].offset,groups[i].tokens};
        kernels::native_q3_k_grouped_down(native,count,scratch,output,slots,stream);return cudaGetLastError();
    }
    for(int n=1;n<=max_tokens;++n){DownBatch batch{};int size=0;
        for(int i=0;i<count;++i)if(groups[i].tokens==n)batch.groups[size++]=groups[i];
        if(!size)continue;
        #define GROUP_Q2(N) case N:q2_grouped_down<N><<<dim3(5120/(N<4?2:4),size),256,0,stream>>>(batch,static_cast<const __nv_bfloat16*>(scratch),output,slots);break
        switch(n){GROUP_Q2(1);GROUP_Q2(2);GROUP_Q2(3);GROUP_Q2(4);GROUP_Q2(5);GROUP_Q2(6);GROUP_Q2(7);GROUP_Q2(8);}
        #undef GROUP_Q2
        const auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    }
    return cudaSuccess;
}
static cudaError_t format_down_impl(uint32_t type,const void* packed,const float* gate,const float* up,const float* routing,
                           float* temporary,float* output,const int* slots,int rows,int cols,int tokens,
                           void* scratch,size_t bytes,cudaStream_t stream,bool batch){
    if((type!=10&&type!=11)||!packed||!gate||!up||!output||!scratch||!stream||rows<1||cols<1||cols%256||tokens<1||tokens>8||
       rows>INT32_MAX/tokens||cols>INT32_MAX/tokens||bytes<(type==10?size_t(tokens)*cols*2:size_t(tokens)*cols/32*36)||
       (type==11&&(!temporary||temporary==output)))return cudaErrorInvalidValue;
    const int count=tokens*cols;
    if(type==10){
        prepare<false><<<(count+255)/256,256,0,stream>>>(gate,up,routing,scratch,cols,count);
        const auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
        if(batch&&tokens>1)q2_verify_launch(static_cast<const uint8_t*>(packed),static_cast<const __nv_bfloat16*>(scratch),output,slots,rows,cols,tokens,stream);
        else q2_down<<<dim3(rows,tokens),256,0,stream>>>(static_cast<const uint8_t*>(packed),static_cast<const __nv_bfloat16*>(scratch),output,slots,rows,cols);
    }else{
        prepare<true><<<(count+255)/256,256,0,stream>>>(gate,up,routing,scratch,cols,count);
        const auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
        // One-token reductions keep ordinary decode and speculative verification arithmetic identical.
        if(batch&&tokens>1&&kernels::native_mmvq_multi_exact())
            kernels::native_q3_k_mmvq(packed,scratch,temporary,cols,rows,tokens,stream);
        else for(int t=0;t<tokens;++t)kernels::native_q3_k_mmvq(packed,static_cast<const uint8_t*>(scratch)+size_t(t)*cols/32*36,temporary+size_t(t)*rows,cols,rows,1,stream);
        finish<<<(tokens*rows+255)/256,256,0,stream>>>(temporary,output,slots,rows,tokens*rows);
    }
    return cudaGetLastError();
}
cudaError_t format_down_gpu(uint32_t type,const void* packed,const float* gate,const float* up,const float* routing,
                           float* temporary,float* output,const int* slots,int rows,int cols,int tokens,
                           void* scratch,size_t bytes,cudaStream_t stream){
    return format_down_impl(type,packed,gate,up,routing,temporary,output,slots,rows,cols,tokens,scratch,bytes,stream,false);
}
cudaError_t format_down_verify_gpu(uint32_t type,const void* packed,const float* gate,const float* up,const float* routing,
                                  float* temporary,float* output,const int* slots,int rows,int cols,int tokens,
                                  void* scratch,size_t bytes,cudaStream_t stream){
    return format_down_impl(type,packed,gate,up,routing,temporary,output,slots,rows,cols,tokens,scratch,bytes,stream,true);
}
}
