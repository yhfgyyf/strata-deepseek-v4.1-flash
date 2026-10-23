// DeepSeek-V4.1 mathematics: deepseek-ai/DeepSeek-V4.1-Flash (MIT).
// Quantization layouts: ggml-org/llama.cpp 3cf03257 (MIT).
#include "strata/models/deepseek41/kernels.hpp"
#include "strata/models/deepseek41/ops.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "deepseek41_rotary_shared.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

namespace strata::models::deepseek41 {
namespace {
__device__ float half_value(const uint8_t* p) { return __half2float(*reinterpret_cast<const __half*>(p)); }
__device__ float weight_value(const uint8_t* p, int type, int64_t i) {
    if (type == 0) return reinterpret_cast<const float*>(p)[i];
    if (type == 1) return __half2float(reinterpret_cast<const __half*>(p)[i]);
    if (type == 30) return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);
    if (type == 2) {
        p += (i / 32) * 18;
        return half_value(p) * (int((p[2 + i % 16] >> ((i % 32) / 16 * 4)) & 15) - 8);
    }
    if (type == 42) {
        p += (i / 64) * 18;
        return half_value(p) * (int((p[2 + (i % 64) / 4] >> (i % 4 * 2)) & 3) - 1);
    }
    if (type == 8) { p += (i / 32) * 34; return half_value(p) * reinterpret_cast<const int8_t*>(p + 2)[i % 32]; }
    if (type == 39) {
        p += (i / 32) * 17;
        constexpr int values[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
        const uint32_t bits = p[0] < 2 ? 0x00200000u << p[0] : uint32_t(p[0] - 1) << 23;
        return __uint_as_float(bits) * values[(p[1 + i % 16] >> ((i % 32) / 16 * 4)) & 15];
    }
    const int index = i % 256;
    if (type == 14) {
        p += (i / 256) * 210;
        const int half=index/128,quarter=index%128/32,lane=index%32;
        const int low=(p[half*64+(quarter%2)*32+lane] >> (quarter/2*4))&15;
        const int high=(p[128+half*32+lane] >> (quarter*2))&3;
        const int scale=reinterpret_cast<const int8_t*>(p+192)[half*8+quarter*2+lane/16];
        return (half_value(p+208)*scale)*((low|(high<<4))-32);
    }
    if (type == 10) {
        p += (i / 256) * 84;
        const int scale = p[index / 16];
        const int code = (p[16 + (index / 128) * 32 + index % 32] >> ((index % 128) / 32 * 2)) & 3;
        return (half_value(p + 80) * (scale & 15)) * code - half_value(p + 82) * (scale >> 4);
    }
    p += (i / 256) * 110;
    const int s = index / 16;
    const int scale = ((p[96 + s % 8] >> (s / 8 * 4)) & 15) | (((p[104 + s % 4] >> (s / 4 * 2)) & 3) << 4);
    const int low = (p[32 + (index / 128) * 32 + index % 32] >> ((index % 128) / 32 * 2)) & 3;
    const int code = low - ((p[index % 32] & (1 << (index / 32))) ? 0 : 4);
    return (half_value(p + 108) * (scale - 32)) * code;
}
bool supported(int type) { return type == 0 || type == 1 || type == 2 || type == 42 || type == 8 || type == 10 || type == 11 || type == 14 || type == 30 || type == 39; }
bool iq_type(int type) { return type==16 || type==17 || type==18 || type==21 || type==22; }
__device__ float sigmoid(float x) { return x >= 0 ? 1.f / (1.f + expf(-x)) : expf(x) / (1.f + expf(x)); }
__device__ float sum_block(float v) {
    __shared__ float sums[8];
    for (int delta = 16; delta; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8 ? sums[threadIdx.x] : 0;
    if (threadIdx.x < 32) for (int delta = 16; delta; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
    return v;
}
template<int TYPE>
__global__ void dq_pairs(const uint8_t* input,uint16_t* out,int64_t n){
 const int lane=threadIdx.x&31;
 const int64_t block=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
 if(block*256>=n)return;
 const uint8_t* p=input+block*(TYPE==10?84:110);
 const float d=half_value(p+(TYPE==10?80:108));
 const float dm=TYPE==10?half_value(p+82):0.f;
 #pragma unroll
 for(int j=0;j<4;++j){
  const int i=lane*2+j*64,group=i/16;
  const int off=(i/128)*32+i%32,shift=(i%128)/32*2;
  const uint16_t bits=*reinterpret_cast<const uint16_t*>(p+(TYPE==10?16:32)+off);
  const int a=(bits>>shift)&3,b=(bits>>(shift+8))&3;
  float x,y;
  if constexpr(TYPE==10){
   const int s=p[group];const float scale=d*(s&15),minimum=dm*(s>>4);
   x=scale*a-minimum;y=scale*b-minimum;
  }else{
   const int s=((p[96+group%8]>>(group/8*4))&15)|(((p[104+group%4]>>(group/4*2))&3)<<4);
   const uint16_t high=*reinterpret_cast<const uint16_t*>(p+i%32);
   const int ca=a-((high&(1<<(i/32)))?0:4);
   const int cb=b-((high&(1<<(i/32+8)))?0:4);
   x=(d*(s-32))*ca;y=(d*(s-32))*cb;
  }
  const uint32_t pair=uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(x)))|
   (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(y)))<<16);
  const int64_t dst=block*256+i;
  if(dst+1<n)*reinterpret_cast<uint32_t*>(out+dst)=pair;
  else if(dst<n)out[dst]=uint16_t(pair);
 }
}
__global__ void dequant_kernel(const uint8_t* p, int type, float* out, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        out[i] = weight_value(p, type, i);
}
__global__ void dequant_bf16_kernel(const uint8_t* p, int type, uint16_t* out, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(weight_value(p, type, i));
}
template<bool round_weights = false>
__global__ void gemv_kernel(const uint8_t* p, int type, const float* x, float* out, int rows, int cols) {
    const int row = blockIdx.x, token = blockIdx.y;
    float sum = 0;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float weight = weight_value(p, type, int64_t(row) * cols + col);
        if constexpr (round_weights) weight = __bfloat162float(__float2bfloat16_rn(weight));
        sum += weight * x[int64_t(token) * cols + col];
    }
    sum = sum_block(sum);
    if (threadIdx.x == 0) out[int64_t(token) * rows + row] = sum;
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
// Q6_K heads retain FP32 activations and decoded weights. Four independent
// warp rows reduce launch/CTA overhead; every token count uses the same order.
template<int NT>
__global__ void q6_projection_warp(const uint8_t* __restrict__ weights,const float* __restrict__ x,
                                 float* __restrict__ out,int rows,int cols,int64_t ldx,int64_t ldy,
                                 bool round_weights,bool round_output) {
    const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;
    if(row>=rows)return;
    const int blocks=cols/256;float sum[NT]{};
    for(int b=0;b<blocks;++b){
        const auto* p=weights+(int64_t(row)*blocks+b)*210;
        const float d=half_value(p+208);
        const int lo[4]={p[lane],p[32+lane],p[64+lane],p[96+lane]};
        const int hi[2]={p[128+lane],p[160+lane]};
#pragma unroll
        for(int g=0;g<8;++g){
            const int q=((lo[(g/4)*2+(g&1)]>>((g%4)/2*4))&15)|(((hi[g/4]>>(2*(g%4)))&3)<<4);
            const int scale=reinterpret_cast<const int8_t*>(p+192)[2*g+lane/16];
            float w=(d*scale)*(q-32);
            if(round_weights)w=__bfloat162float(__float2bfloat16_rn(w));
#pragma unroll
            for(int t=0;t<NT;++t)sum[t]+=w*x[int64_t(t)*ldx+b*256+g*32+lane];
        }
    }
#pragma unroll
    for(int delta=16;delta;delta>>=1){
#pragma unroll
        for(int t=0;t<NT;++t)sum[t]+=__shfl_down_sync(0xffffffff,sum[t],delta);
    }
    if(lane==0){
#pragma unroll
        for(int t=0;t<NT;++t)out[int64_t(t)*ldy+row]=round_output?__bfloat162float(__float2bfloat16_rn(sum[t])):sum[t];
    }
}
cudaError_t q6_projection(const void* weights,const float* x,float* out,int rows,int cols,int tokens,
                          int64_t ldx,int64_t ldy,bool round_weights,bool round_output,cudaStream_t stream){
#define Q6_PROJECTION(N) case N:q6_projection_warp<N><<<(rows+3)/4,128,0,stream>>>(static_cast<const uint8_t*>(weights),x,out,rows,cols,ldx,ldy,round_weights,round_output);break
    switch(tokens){Q6_PROJECTION(1);Q6_PROJECTION(2);Q6_PROJECTION(3);Q6_PROJECTION(4);Q6_PROJECTION(5);Q6_PROJECTION(6);Q6_PROJECTION(7);Q6_PROJECTION(8);default:return cudaErrorInvalidValue;}
#undef Q6_PROJECTION
    return cudaGetLastError();
}
__device__ float bf16(float value){return __bfloat162float(__float2bfloat16_rn(value));}
__global__ void draft_projection5_kernel(const uint8_t* weights,int type,const float* x,float* out,
                                         int cols,int64_t ldx,int64_t ldy,bool round_weights,bool round_output){
    const int row=blockIdx.x;float values[5]{};
    for(int col=threadIdx.x;col<cols;col+=256){
        float weight=weight_value(weights,type,int64_t(row)*cols+col);
        if(round_weights)weight=bf16(weight);
        #pragma unroll
        for(int t=0;t<5;++t)values[t]+=weight*x[int64_t(t)*ldx+col];
    }
    sum_columns(values);
    if(threadIdx.x==0){
        #pragma unroll
        for(int t=0;t<5;++t)out[int64_t(t)*ldy+row]=round_output?bf16(values[t]):values[t];
    }
}
__global__ void draft_projection7_kernel(const uint8_t* weights,int type,const float* x,float* out,
                                         int cols,int64_t ldx,int64_t ldy,bool round_weights,bool round_output){
    const int row=blockIdx.x;float values[7]{};
    for(int col=threadIdx.x;col<cols;col+=256){
        float weight=weight_value(weights,type,int64_t(row)*cols+col);
        if(round_weights)weight=bf16(weight);
        #pragma unroll
        for(int t=0;t<7;++t)values[t]+=weight*x[int64_t(t)*ldx+col];
    }
    sum_columns(values);
    if(threadIdx.x==0){
        #pragma unroll
        for(int t=0;t<7;++t)out[int64_t(t)*ldy+row]=round_output?bf16(values[t]):values[t];
    }
}
#include "deepseek41_draft_warp.cuh"
template<int expert_type>
__global__ void draft_moe_gu_kernel(const uint8_t* experts,uint64_t projection,const uint8_t* shared_gate,
                                    const uint8_t* shared_up,int shared_type,const float* x,const int* ids,
                                    const float* routing,float* activation){
    const int row=blockIdx.x,job=blockIdx.y,token=job/4,slot=job%4;
    const bool shared=slot==3;const int id=shared?0:ids[token*3+slot];
    if(id<0||id>=128){if(!threadIdx.x)activation[int64_t(job)*2304+row]=NAN;return;}
    const auto* gate=shared?shared_gate:experts+uint64_t(id)*3*projection;
    const auto* up=shared?shared_up:gate+projection;const int type=shared?shared_type:expert_type;
    float values[2]{};
    for(int col=threadIdx.x;col<5120;col+=256){const float input=x[int64_t(token)*5120+col];
        values[0]+=bf16(weight_value(gate,type,int64_t(row)*5120+col))*input;
        values[1]+=bf16(weight_value(up,type,int64_t(row)*5120+col))*input;
    }
    sum_columns(values);
    if(threadIdx.x==0){const float g=fminf(bf16(values[0]),10.f),u=fminf(fmaxf(bf16(values[1]),-10.f),10.f);
        activation[int64_t(job)*2304+row]=bf16(g*sigmoid(g)*u*(shared?1.f:routing[token*3+slot]));
    }
}
template<int expert_type>
__global__ void draft_moe_down_kernel(const uint8_t* experts,uint64_t projection,const uint8_t* shared_down,
                                      int shared_type,const int* ids,const float* activation,float* out){
    const int row=blockIdx.x,token=blockIdx.y;
    const int selected[3]={ids[token*3],ids[token*3+1],ids[token*3+2]};
    if(selected[0]<0||selected[0]>=128||selected[1]<0||selected[1]>=128||selected[2]<0||selected[2]>=128){
        if(!threadIdx.x)out[int64_t(token)*5120+row]=NAN;return;
    }
    const uint8_t* weights[4]={experts+(uint64_t(selected[0])*3+2)*projection,
        experts+(uint64_t(selected[1])*3+2)*projection,experts+(uint64_t(selected[2])*3+2)*projection,shared_down};
    float values[4]{};
    for(int col=threadIdx.x;col<2304;col+=256){
        #pragma unroll
        for(int slot=0;slot<4;++slot)values[slot]+=bf16(weight_value(weights[slot],slot==3?shared_type:expert_type,
            int64_t(row)*2304+col))*activation[(int64_t(token)*4+slot)*2304+col];
    }
    sum_columns(values);
    if(threadIdx.x==0){
        int order[3]={0,1,2};
        for(int k=1;k<3;++k)for(int j=k;j&&selected[order[j]]<selected[order[j-1]];--j){
            const int previous=order[j];order[j]=order[j-1];order[j-1]=previous;
        }
        float result=0;for(int k=0;k<3;++k)result+=bf16(values[order[k]]);
        out[int64_t(token)*5120+row]=bf16(result+bf16(values[3]));
    }
}
struct Q81Input {half2 ds;int8_t qs[32];};
template<bool quantized>
__global__ void expert_gu_kernel(const uint8_t* p,int type,const float* x,const Q81Input* xq,
                                  float* gate,float* up,int rows,int cols,int tokens) {
    const int row=blockIdx.x,token=blockIdx.y%tokens,projection=blockIdx.y/tokens;
    const int64_t row_index=int64_t(projection*rows+row)*cols;
    float sum=0;
    if constexpr(!quantized){
        for(int col=threadIdx.x;col<cols;col+=256)
            sum+=__bfloat162float(__float2bfloat16_rn(weight_value(p,type,row_index+col)))*x[int64_t(token)*cols+col];
    }else{
        for(int b=threadIdx.x;b<cols/32;b+=256){
            const auto* block=p+(row_index/256+b/8)*(type==10?84:110);
            const auto& a=xq[int64_t(token)*cols/32+b];const float xd=__low2float(a.ds);
            for(int h=0;h<2;++h){
                const int sub=(b%8)*2+h;int product=0,asum=0;
                #pragma unroll
                for(int j=0;j<16;j+=4){
                    uint32_t codes=0;
                    #pragma unroll
                    for(int v=0;v<4;++v){
                        const int i=sub*16+j+v;
                        int code=(block[(type==10?16:32)+i/128*32+i%32]>>(i%128/32*2))&3;
                        if(type==11 && !(block[i%32]&(1<<(i/32))))code-=4;
                        codes|=uint32_t(uint8_t(code))<<(v*8);
                    }
                    const int aq=*reinterpret_cast<const int*>(a.qs+h*16+j);
                    product=__dp4a(int(codes),aq,product);asum=__dp4a(0x01010101,aq,asum);
                }
                if(type==10)sum+=xd*((half_value(block+80)*(block[sub]&15))*product-half_value(block+82)*(block[sub]>>4)*asum);
                else{
                    const int scale=((block[96+sub%8]>>(sub/8*4))&15)|(((block[104+sub%4]>>(sub/4*2))&3)<<4);
                    sum+=xd*(half_value(block+108)*(scale-32))*product;
                }
            }
        }
    }
    sum=sum_block(sum);
    if(threadIdx.x==0)(projection?up:gate)[int64_t(token)*rows+row]=__bfloat162float(__float2bfloat16_rn(sum));
}
template<int N>
__global__ void verify_projection_kernel(const uint8_t* p,int type,const float* x,float* out,
                                         int cols,int64_t ldx,int64_t ldy){
    const int row=blockIdx.x;float values[N]{};
    for(int col=threadIdx.x;col<cols;col+=256){
        const float weight=weight_value(p,type,int64_t(row)*cols+col);
        #pragma unroll
        for(int t=0;t<N;++t)values[t]+=weight*x[int64_t(t)*ldx+col];
    }
    sum_columns(values);
    if(threadIdx.x==0){
        #pragma unroll
        for(int t=0;t<N;++t)out[int64_t(t)*ldy+row]=values[t];
    }
}
// Q3_K blocks alternate between four-byte and two-byte alignment.
__device__ uint32_t load_quant_four(const uint8_t* p){
    if((reinterpret_cast<uintptr_t>(p)&3)==0)return *reinterpret_cast<const uint32_t*>(p);
    if((reinterpret_cast<uintptr_t>(p)&1)==0)
        return uint32_t(*reinterpret_cast<const uint16_t*>(p))|(uint32_t(*reinterpret_cast<const uint16_t*>(p+2))<<16);
    return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
}
template<int N,bool quantized,int QuantType=0>
__device__ void expert_gu_verify_values(const uint8_t* p,int type,const float* x,const Q81Input* xq,
                                       float* gate,float* up,int rows,int cols,const int* token_rows,int offset){
    const int row=blockIdx.x,projection=blockIdx.y;
    int input_rows[N];
    #pragma unroll
    for(int t=0;t<N;++t)input_rows[t]=token_rows?token_rows[offset+t]:t;
    const int64_t row_index=int64_t(projection*rows+row)*cols;
    float values[N]{};
    if constexpr(!quantized){
        for(int col=threadIdx.x;col<cols;col+=256){
            const float weight=bf16(weight_value(p,type,row_index+col));
            #pragma unroll
            for(int t=0;t<N;++t)values[t]+=weight*x[int64_t(input_rows[t])*cols+col];
        }
    }else{
        constexpr int type=QuantType;
        for(int b=threadIdx.x;b<cols/32;b+=256){
            const auto* block=p+(row_index/256+b/8)*(type==10?84:110);
            for(int h=0;h<2;++h){
                const int sub=(b%8)*2+h;int codes[4];
                #pragma unroll
                for(int j=0;j<16;j+=4){
                    const int i=sub*16+j;
                    const auto* low=block+(type==10?16:32)+i/128*32+i%32;
                    uint32_t code_word=(load_quant_four(low)>>(i%128/32*2))&0x03030303u;
                    if(type==11){
                        const uint32_t high=(load_quant_four(block+i%32)>>(i/32))&0x01010101u;
                        code_word|=(0x01010101u-high)*0xfcu;
                    }
                    codes[j/4]=int(code_word);
                }
                #pragma unroll
                for(int t=0;t<N;++t){
                    const auto& a=xq[int64_t(input_rows[t])*cols/32+b];const float xd=__low2float(a.ds);
                    int product=0,asum=0;
                    #pragma unroll
                    for(int j=0;j<16;j+=4){
                        const int aq=*reinterpret_cast<const int*>(a.qs+h*16+j);
                        product=__dp4a(codes[j/4],aq,product);asum=__dp4a(0x01010101,aq,asum);
                    }
                    // The original type branch rounds the product before adding; prevent new FMA contraction.
                    if(type==10)values[t]=__fadd_rn(values[t],__fmul_rn(xd,(half_value(block+80)*(block[sub]&15))*product-half_value(block+82)*(block[sub]>>4)*asum));
                    else{
                        const int scale=((block[96+sub%8]>>(sub/8*4))&15)|(((block[104+sub%4]>>(sub/4*2))&3)<<4);
                        values[t]=__fadd_rn(values[t],__fmul_rn(xd*(half_value(block+108)*(scale-32)),float(product)));
                    }
                }
            }
        }
    }
    sum_columns(values);
    if(threadIdx.x==0){
        #pragma unroll
        for(int t=0;t<N;++t)(projection?up:gate)[int64_t(offset+t)*rows+row]=bf16(values[t]);
    }
}

template<int N,bool quantized,int QuantType=0>
__global__ void expert_gu_verify_kernel(const uint8_t* p,int type,const float* x,const Q81Input* xq,
                                       float* gate,float* up,int rows,int cols){
    expert_gu_verify_values<N,quantized,QuantType>(p,type,x,xq,gate,up,rows,cols,nullptr,0);
}
struct VerifyExpertBatch { VerifyExpertGroup groups[48]; };
template<int N,bool quantized,int QuantType=0>
__global__ void expert_gu_grouped_verify_kernel(VerifyExpertBatch batch,int type,const float* x,const Q81Input* xq,
                                               const int* token_rows,float* gate,float* up){
    const auto group=batch.groups[blockIdx.z];
    expert_gu_verify_values<N,quantized,QuantType>(static_cast<const uint8_t*>(group.packed),type,x,xq,
        gate,up,2304,5120,token_rows,group.offset);
}

template<int N>
__global__ void q2_gu_activation_grouped(VerifyExpertBatch batch,const Q81Input* xq,
    const int* token_rows,const float* routing,__nv_bfloat16* activation){
    const auto group=batch.groups[blockIdx.y];const auto* p=static_cast<const uint8_t*>(group.packed);
    const int row=blockIdx.x;float values[2*N]{};
    for(int b=threadIdx.x;b<5120/32;b+=256){
        #pragma unroll
        for(int projection=0;projection<2;++projection){
            const auto* block=p+((int64_t(projection*2304+row)*5120)/256+b/8)*84;
            for(int h=0;h<2;++h){const int sub=b%8*2+h;int codes[4];
                #pragma unroll
                for(int j=0;j<16;j+=4){const int i=sub*16+j;
                    codes[j/4]=int((load_quant_four(block+16+i/128*32+i%32)>>(i%128/32*2))&0x03030303u);
                }
                #pragma unroll
                for(int t=0;t<N;++t){const auto& a=xq[int64_t(token_rows[group.offset+t])*(5120/32)+b];
                    const float xd=__low2float(a.ds);int product=0,asum=0;
                    #pragma unroll
                    for(int j=0;j<16;j+=4){const int aq=*reinterpret_cast<const int*>(a.qs+h*16+j);
                        product=__dp4a(codes[j/4],aq,product);asum=__dp4a(0x01010101,aq,asum);
                    }
                    values[projection*N+t]=__fadd_rn(values[projection*N+t],__fmul_rn(xd,
                        (half_value(block+80)*(block[sub]&15))*product-half_value(block+82)*(block[sub]>>4)*asum));
                }
            }
        }
    }
    sum_columns(values);
    if(threadIdx.x==0){
        #pragma unroll
        for(int t=0;t<N;++t){const float g=fminf(bf16(values[t]),10.f),u=fminf(fmaxf(bf16(values[N+t]),-10.f),10.f);
            activation[int64_t(group.offset+t)*2304+row]=__float2bfloat16_rn(g*sigmoid(g)*u*routing[group.offset+t]);
        }
    }
}

__global__ void rms_kernel(const float* x, const float* weight, float* out, int width, float eps) {
    const int token = blockIdx.x;
    float square = 0;
    for (int col = threadIdx.x; col < width; col += blockDim.x) { const float v = x[int64_t(token) * width + col]; square += v * v; }
    square = sum_block(square);
    __shared__ float inverse;
    if (threadIdx.x == 0) inverse = rsqrtf(square / width + eps);
    __syncthreads();
    for (int col = threadIdx.x; col < width; col += blockDim.x)
        out[int64_t(token) * width + col] = x[int64_t(token) * width + col] * inverse * (weight ? weight[col] : 1.f);
}
__global__ void round_kernel(float* x, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        x[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
}
__global__ void to_bf16_kernel(const float* x,uint16_t* out,int64_t n) {
    for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x)
        reinterpret_cast<__nv_bfloat16*>(out)[i]=__float2bfloat16_rn(x[i]);
}
__global__ void from_bf16_kernel(const uint16_t* x,float* out,int64_t n) {
    for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x)
        out[i]=__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x)[i]);
}
__global__ void swiglu_kernel(const float* gate, const float* up, const float* weights, float* out, int width, int64_t n, float limit) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        float g = gate[i], u = up[i];
        if (limit > 0) { g = fminf(g, limit); u = fminf(fmaxf(u, -limit), limit); }
        out[i] = g * sigmoid(g) * u * (weights ? weights[i / width] : 1.f);
    }
}
__global__ void route_kernel(const float* logits, const float* bias, const float* vl, const uint8_t* image,
                             int* ids, float* weights, int experts, int top_k, float scale) {
    // One token per block. Stable ties select the lowest ID, as in the saved native router.
    if (threadIdx.x) return;
    const int token = blockIdx.x;
    float scores[384], selected[6], total = 0;
    const float* correction = vl && image && image[token] ? vl : bias;
    for (int e = 0; e < experts; ++e) {
        const float x = logits[int64_t(token) * experts + e];
        scores[e] = sqrtf(fmaxf(x, 0.f) + log1pf(expf(-fabsf(x))));
    }
    for (int k = 0; k < top_k; ++k) {
        float maximum = -INFINITY; int id = -1;
        for (int e = 0; e < experts; ++e) {
            bool used = false;
            for (int j = 0; j < k; ++j) used |= ids[token * top_k + j] == e;
            const float value = scores[e] + correction[e];
            if (!used && value > maximum) { maximum = value; id = e; }
        }
        ids[token * top_k + k] = id;
        selected[k] = id >= 0 ? scores[id] : 0.f; total += selected[k];
    }
    for (int k = 0; k < top_k; ++k) weights[token * top_k + k] = (selected[k] / (total + 1e-20f)) * scale;
}
__global__ void route_parallel_kernel(const float* logits,const float* bias,const float* vl,const uint8_t* image,
                                       int* ids,float* weights,int experts,int top_k,float scale){
    __shared__ float scores[384],warp_value[16],selected[6],total;
    __shared__ int warp_id[16],winner;
    const int e=threadIdx.x,token=blockIdx.x,lane=e%32,warp=e/32;
    const float* correction=vl && image && image[token]?vl:bias;
    if(e<experts){const float x=logits[int64_t(token)*experts+e];scores[e]=sqrtf(fmaxf(x,0.f)+log1pf(expf(-fabsf(x))));}
    if(e==0)total=0;
    __syncthreads();bool used=false;
    for(int k=0;k<top_k;++k){
        float best=e<experts && !used?scores[e]+correction[e]:-INFINITY;
        int id=best>-INFINITY?e:INT32_MAX; // Match serial rejection of NaN and -Inf candidates.
        if(id==INT32_MAX)best=-INFINITY;
        for(int d=16;d;d>>=1){const float other=__shfl_down_sync(0xffffffff,best,d);const int oid=__shfl_down_sync(0xffffffff,id,d);
            if(other>best || (other==best && oid<id)){best=other;id=oid;}}
        if(lane==0){warp_value[warp]=best;warp_id[warp]=id;}
        __syncthreads();
        if(warp==0){
            best=lane<16?warp_value[lane]:-INFINITY;id=lane<16?warp_id[lane]:INT32_MAX;
            for(int d=16;d;d>>=1){const float other=__shfl_down_sync(0xffffffff,best,d);const int oid=__shfl_down_sync(0xffffffff,id,d);
                if(other>best || (other==best && oid<id)){best=other;id=oid;}}
            if(lane==0){winner=id==INT32_MAX?-1:id;ids[token*top_k+k]=winner;selected[k]=winner<0?0:scores[winner];total+=selected[k];}
        }
        __syncthreads();used|=e==winner;
    }
    if(e<top_k)weights[token*top_k+e]=(selected[e]/(total+1e-20f))*scale;
}
__global__ void hc_project_kernel(const float* x, const float* fn, float* mixes, int hidden, float eps) {
    const int row = blockIdx.x, token = blockIdx.y, width = hidden * 4;
    float sum = 0, dot = 0;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const float value = x[int64_t(token) * width + col]; sum += value * value;
        dot += value * fn[int64_t(row) * width + col];
    }
    sum = sum_block(sum);
    __shared__ float inverse;
    if (!threadIdx.x) inverse = rsqrtf(sum / width + eps);
    __syncthreads();
    dot = sum_block(dot);
    if (!threadIdx.x) mixes[token * 24 + row] = dot * inverse;
}
__global__ void hc_split_kernel(const float* mixes, const float* scale, const float* base,
                                float* pre, float* post, float* comb, float eps, int iterations) {
    if (threadIdx.x) return;
    const int token = blockIdx.x;
    mixes += token * 24; pre += token * 4; post += token * 4; comb += token * 16;
    for (int j = 0; j < 4; ++j) {
        pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps;
        post[j] = 2.f * sigmoid(mixes[4 + j] * scale[1] + base[4 + j]);
        float maximum = -INFINITY, sum = 0;
        for (int k = 0; k < 4; ++k) { comb[j * 4 + k] = mixes[8 + j * 4 + k] * scale[2] + base[8 + j * 4 + k]; maximum = fmaxf(maximum, comb[j * 4 + k]); }
        for (int k = 0; k < 4; ++k) { comb[j * 4 + k] = expf(comb[j * 4 + k] - maximum); sum += comb[j * 4 + k]; }
        for (int k = 0; k < 4; ++k) comb[j * 4 + k] = comb[j * 4 + k] / sum + eps;
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (iteration) for (int j = 0; j < 4; ++j) {
            float sum = eps; for (int k = 0; k < 4; ++k) sum += comb[j * 4 + k];
            for (int k = 0; k < 4; ++k) comb[j * 4 + k] /= sum;
        }
        for (int k = 0; k < 4; ++k) {
            float sum = eps; for (int j = 0; j < 4; ++j) sum += comb[j * 4 + k];
            for (int j = 0; j < 4; ++j) comb[j * 4 + k] /= sum;
        }
    }
}
__global__ void hc_pre_kernel(const float* x, const float* pre, float* out, int hidden, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int token = i / hidden, d = i % hidden;
        float sum = 0; for (int j = 0; j < 4; ++j) sum += pre[token * 4 + j] * x[(int64_t(token) * 4 + j) * hidden + d];
        out[i] = sum;
    }
}
__global__ void hc_post_kernel(const float* x, const float* branch, const float* post, const float* comb,
                              float* out, int hidden, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int token = i / hidden, d = i % hidden;
        float copies[4]; for (int j = 0; j < 4; ++j) copies[j] = x[(int64_t(token) * 4 + j) * hidden + d];
        for (int destination = 0; destination < 4; ++destination) {
            float value = post[token * 4 + destination] * branch[i];
            for (int source = 0; source < 4; ++source) value += comb[token * 16 + source * 4 + destination] * copies[source];
            out[(int64_t(token) * 4 + destination) * hidden + d] = value;
        }
    }
}
__global__ void rotate_kernel(float* x, int heads, int head_dim, int tokens, int64_t start, RotaryFrequencies frequencies, bool inverse) {
    constexpr int dim = 64, pairs = 32;
    const int64_t n = int64_t(tokens) * heads * pairs;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int pair = i % pairs, head = i / pairs % heads, token = i / (pairs * heads);
        const float frequency = frequencies.value[pair];
        const float angle = float(start + token) * frequency * (inverse ? -1.f : 1.f);
        float s, c; sincosf(angle, &s, &c);
        float* p = x + (int64_t(token) * heads + head) * head_dim + head_dim - dim + pair * 2;
        const float a = p[0], b = p[1]; p[0] = a * c - b * s; p[1] = a * s + b * c;
    }
}
__global__ void pool2_kernel(const float* a, const float* b, const float* ga, const float* gb, float* out, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) { const float m = fmaxf(ga[i], gb[i]); const float x = expf(ga[i] - m), y = expf(gb[i] - m); out[i] = (a[i] * x + b[i] * y) / (x + y); }
}
__global__ void engram_gate_kernel(float* h, const float* kv, const float* qw, const float* kw,
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
    hh = sum_block(hh); if (!threadIdx.x) sums[0] = hh; __syncthreads();
    kk = sum_block(kk); if (!threadIdx.x) sums[1] = kk; __syncthreads();
    dot = sum_block(dot);
    if (!threadIdx.x) {
        dot *= rsqrtf(sums[0] / width + 1e-20f) * rsqrtf(sums[1] / width + 1e-20f) * rsqrtf(float(width));
        float input = sqrtf(fmaxf(fabsf(dot), 1e-6f)); if (dot < 0) input = -input;
        gate = image && image[t] ? 0.f : sigmoid(input);
    }
    __syncthreads();
    const float* value = kv + int64_t(t * 5 + 4) * width;
    for (int d = threadIdx.x; d < width; d += blockDim.x) x[d] += gate * value[d];
}
int grid(int64_t n) { return static_cast<int>(std::min<int64_t>((n + 255) / 256, 65535)); }
}
cudaError_t dequant_gpu(const void* p, uint32_t type, float* out, int64_t n, cudaStream_t s) {
    if(type==13 || type==14 || iq_type(type)){
        if(!p || !out || n<=0 || n%256)return cudaErrorInvalidValue;
        kernels::dequant_f32(type,p,0,1,n,out,s);return cudaGetLastError();
    }
    if (!supported(type) || n <= 0 || (type == 10 || type == 11 || type == 14 ? n % 256 : type == 42 ? n % 64 : type == 2 || type == 8 || type == 39 ? n % 32 : 0)) return cudaErrorInvalidValue;
    dequant_kernel<<<grid(n),256,0,s>>>(static_cast<const uint8_t*>(p), type, out, n); return cudaGetLastError();
}
cudaError_t dequant_round_bf16_gpu(const void* p, uint32_t type, float* out, int64_t n, cudaStream_t s) {
    if(type==13 && kernels::dequant_q5_f32_bf16(p,n,out,s))return cudaGetLastError();
    if(type==14 && kernels::dequant_q6_f32_bf16(p,n,out,s))return cudaGetLastError();
    const cudaError_t decoded=dequant_gpu(p,type,out,n,s);
    return decoded==cudaSuccess?round_bf16_gpu(out,n,s):decoded;
}
cudaError_t draft_projection5_gpu(const void* weights,uint32_t type,const float* x,float* out,
                                 int rows,int cols,int64_t ldx,int64_t ldy,bool round_weights,
                                 bool round_output,cudaStream_t stream){
    if(!weights||!x||!out||rows<1||cols<1||ldx<cols||ldy<rows||
       (type!=0&&type!=2&&type!=8&&type!=14&&type!=30)||(type==14?cols%256:(type==2||type==8)&&cols%32))return cudaErrorInvalidValue;
    if(type==14)return q6_projection(weights,x,out,rows,cols,5,ldx,ldy,round_weights,round_output,stream);
    draft_projection5_kernel<<<rows,256,0,stream>>>(static_cast<const uint8_t*>(weights),type,x,out,cols,ldx,ldy,round_weights,round_output);
    return cudaGetLastError();
}
cudaError_t draft_projection7_gpu(const void* weights,uint32_t type,const float* x,float* out,
                                 int rows,int cols,int64_t ldx,int64_t ldy,bool round_weights,
                                 bool round_output,cudaStream_t stream){
    if(!weights||!x||!out||rows<1||cols<1||ldx<cols||ldy<rows||
       (type!=0&&type!=2&&type!=8&&type!=14&&type!=30)||(type==14?cols%256:(type==2||type==8)&&cols%32))return cudaErrorInvalidValue;
    if(type==14)return q6_projection(weights,x,out,rows,cols,7,ldx,ldy,round_weights,round_output,stream);
    draft_projection7_kernel<<<rows,256,0,stream>>>(static_cast<const uint8_t*>(weights),type,x,out,cols,ldx,ldy,round_weights,round_output);
    return cudaGetLastError();
}
cudaError_t draft_fused_moe_gpu(const void* experts,uint32_t expert_type,uint64_t projection,const void* shared_gate,
                               const void* shared_up,const void* shared_down,uint32_t shared_type,
                               const float* x,const int* ids,const float* routing,float* scratch,
                               float* out,int tokens,cudaStream_t stream){
    const uint64_t expected=expert_type==39?uint64_t(5120)*2304/32*17:uint64_t(5120)*2304/256*84;
    if((expert_type!=10&&expert_type!=39)||!experts||projection!=expected||!shared_gate||!shared_up||!shared_down||
       (shared_type!=2&&shared_type!=8&&shared_type!=30)||!x||!ids||!routing||!scratch||!out||tokens<1||tokens>7)return cudaErrorInvalidValue;
    if(expert_type==39 && shared_type==30) {
        draft_warp::draft_gu_warp<<<dim3(576,tokens*4),128,0,stream>>>(static_cast<const uint8_t*>(experts),projection,
            static_cast<const uint8_t*>(shared_gate),static_cast<const uint8_t*>(shared_up),x,ids,routing,scratch);
        const auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
        draft_warp::draft_down_warp<<<dim3(1280,tokens),128,0,stream>>>(static_cast<const uint8_t*>(experts),projection,
            static_cast<const uint8_t*>(shared_down),ids,scratch,out);
        return cudaGetLastError();
    }
    if(expert_type==10)draft_moe_gu_kernel<10><<<dim3(2304,tokens*4),256,0,stream>>>(static_cast<const uint8_t*>(experts),projection,
        static_cast<const uint8_t*>(shared_gate),static_cast<const uint8_t*>(shared_up),shared_type,x,ids,routing,scratch);
    else draft_moe_gu_kernel<39><<<dim3(2304,tokens*4),256,0,stream>>>(static_cast<const uint8_t*>(experts),projection,
        static_cast<const uint8_t*>(shared_gate),static_cast<const uint8_t*>(shared_up),shared_type,x,ids,routing,scratch);
    const auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    if(expert_type==10)draft_moe_down_kernel<10><<<dim3(5120,tokens),256,0,stream>>>(static_cast<const uint8_t*>(experts),projection,
        static_cast<const uint8_t*>(shared_down),shared_type,ids,scratch,out);
    else draft_moe_down_kernel<39><<<dim3(5120,tokens),256,0,stream>>>(static_cast<const uint8_t*>(experts),projection,
        static_cast<const uint8_t*>(shared_down),shared_type,ids,scratch,out);
    return cudaGetLastError();
}
cudaError_t expert_gu_gpu(const void* p,uint32_t type,const float* x,float* gate,float* up,
                          int rows,int cols,int tokens,int mode,void* scratch,cudaStream_t stream) {
    if(!p || !x || !gate || !up || rows<1 || cols<1 || cols%256 || tokens<1 || tokens>8 ||
       (type!=10 && type!=11) || (mode!=1 && mode!=2) || (mode==2 && !scratch))return cudaErrorInvalidValue;
    if(mode==2){
        kernels::native_quantize_q8_1(x,scratch,cols,tokens,stream);
        expert_gu_kernel<true><<<dim3(rows,2*tokens),256,0,stream>>>(static_cast<const uint8_t*>(p),type,x,
            static_cast<const Q81Input*>(scratch),gate,up,rows,cols,tokens);
    }else expert_gu_kernel<false><<<dim3(rows,2*tokens),256,0,stream>>>(static_cast<const uint8_t*>(p),type,x,nullptr,gate,up,rows,cols,tokens);
    return cudaGetLastError();
}
cudaError_t packed_verify_gpu(const void* p,uint32_t type,const float* x,float* out,
                             int rows,int cols,int tokens,int64_t ldx,int64_t ldy,cudaStream_t stream){
    if(!p||!x||!out||rows<1||cols<1||tokens<2||tokens>8||ldx<cols||ldy<rows||
       (type!=0&&type!=30&&type!=8&&type!=14)||(type==14?cols%256:type==8&&cols%32))return cudaErrorInvalidValue;
    if(type==14)return q6_projection(p,x,out,rows,cols,tokens,ldx,ldy,false,false,stream);
    #define VERIFY_PROJECTION(N) case N: verify_projection_kernel<N><<<rows,256,0,stream>>>(static_cast<const uint8_t*>(p),type,x,out,cols,ldx,ldy);break
    switch(tokens){VERIFY_PROJECTION(2);VERIFY_PROJECTION(3);VERIFY_PROJECTION(4);VERIFY_PROJECTION(5);VERIFY_PROJECTION(6);VERIFY_PROJECTION(7);VERIFY_PROJECTION(8);}
    #undef VERIFY_PROJECTION
    return cudaGetLastError();
}
cudaError_t expert_gu_verify_gpu(const void* p,uint32_t type,const float* x,float* gate,float* up,
                                int rows,int cols,int tokens,int mode,void* scratch,cudaStream_t stream){
    if(tokens==1)return expert_gu_gpu(p,type,x,gate,up,rows,cols,tokens,mode,scratch,stream);
    if(!p||!x||!gate||!up||rows<1||cols<1||cols%256||tokens<2||tokens>8||
       (type!=10&&type!=11)||(mode!=1&&mode!=2)||(mode==2&&!scratch))return cudaErrorInvalidValue;
    if(mode==2)kernels::native_quantize_q8_1(x,scratch,cols,tokens,stream);
    #define VERIFY_GU(N) case N: if(mode==2){if(type==10)expert_gu_verify_kernel<N,true,10><<<dim3(rows,2),cols==5120?160:256,0,stream>>>(static_cast<const uint8_t*>(p),type,x,static_cast<const Q81Input*>(scratch),gate,up,rows,cols);else expert_gu_verify_kernel<N,true,11><<<dim3(rows,2),cols==5120?160:256,0,stream>>>(static_cast<const uint8_t*>(p),type,x,static_cast<const Q81Input*>(scratch),gate,up,rows,cols);}else expert_gu_verify_kernel<N,false><<<dim3(rows,2),256,0,stream>>>(static_cast<const uint8_t*>(p),type,x,nullptr,gate,up,rows,cols);break
    switch(tokens){VERIFY_GU(2);VERIFY_GU(3);VERIFY_GU(4);VERIFY_GU(5);VERIFY_GU(6);VERIFY_GU(7);VERIFY_GU(8);}
    #undef VERIFY_GU
    return cudaGetLastError();
}

cudaError_t expert_gu_grouped_verify_gpu(const VerifyExpertGroup* groups,int group_count,uint32_t type,
                                        const float* x,const int* token_rows,float* gate,float* up,
                                        int tokens,int mode,void* scratch,cudaStream_t stream){
    if(!groups||group_count<1||group_count>48||(type!=10&&type!=11)||!x||!token_rows||!gate||!up||
       tokens<1||tokens>8||(mode!=1&&mode!=2)||(mode==2&&!scratch))return cudaErrorInvalidValue;
    for(int i=0;i<group_count;++i)if(!groups[i].packed||groups[i].offset<0||groups[i].offset>48||groups[i].tokens<1||
        groups[i].tokens>tokens||groups[i].offset+groups[i].tokens>48)return cudaErrorInvalidValue;
    if(mode==2)kernels::native_quantize_q8_1(x,scratch,5120,tokens,stream);
    for(int n=1;n<=tokens;++n){
        VerifyExpertBatch batch{};int count=0;
        for(int i=0;i<group_count;++i)if(groups[i].tokens==n)batch.groups[count++]=groups[i];
        if(!count)continue;
        #define VERIFY_GROUP(N) case N:if(mode==2){if(type==10)expert_gu_grouped_verify_kernel<N,true,10><<<dim3(2304,2,count),160,0,stream>>>(batch,type,x,static_cast<const Q81Input*>(scratch),token_rows,gate,up);else expert_gu_grouped_verify_kernel<N,true,11><<<dim3(2304,2,count),160,0,stream>>>(batch,type,x,static_cast<const Q81Input*>(scratch),token_rows,gate,up);}else expert_gu_grouped_verify_kernel<N,false><<<dim3(2304,2,count),256,0,stream>>>(batch,type,x,nullptr,token_rows,gate,up);break
        switch(n){VERIFY_GROUP(1);VERIFY_GROUP(2);VERIFY_GROUP(3);VERIFY_GROUP(4);VERIFY_GROUP(5);VERIFY_GROUP(6);VERIFY_GROUP(7);VERIFY_GROUP(8);}
        #undef VERIFY_GROUP
        const auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    }
    return cudaSuccess;
}

cudaError_t expert_q2_activation_grouped_gpu(const VerifyExpertGroup* groups,int count,
    const float* x,const int* token_rows,const float* routing,uint16_t* activation,
    int tokens,void* scratch,cudaStream_t stream){
    if(!groups||count<1||count>48||!x||!token_rows||!routing||!activation||!scratch||!stream||tokens<1||tokens>8)
        return cudaErrorInvalidValue;
    bool used[48]{};
    for(int i=0;i<count;++i){const auto& g=groups[i];
        if(!g.packed||g.offset<0||g.tokens<1||g.tokens>tokens||g.offset>48-g.tokens)return cudaErrorInvalidValue;
        for(int t=0;t<g.tokens;++t){if(used[g.offset+t])return cudaErrorInvalidValue;used[g.offset+t]=true;}
    }
    kernels::native_quantize_q8_1(x,scratch,5120,tokens,stream);
    for(int n=1;n<=tokens;++n){VerifyExpertBatch batch{};int size=0;
        for(int i=0;i<count;++i)if(groups[i].tokens==n)batch.groups[size++]=groups[i];
        if(!size)continue;
        #define Q2_GU_ACT(N) case N:q2_gu_activation_grouped<N><<<dim3(2304,size),160,0,stream>>>(batch,static_cast<const Q81Input*>(scratch),token_rows,routing,reinterpret_cast<__nv_bfloat16*>(activation));break
        switch(n){Q2_GU_ACT(1);Q2_GU_ACT(2);Q2_GU_ACT(3);Q2_GU_ACT(4);Q2_GU_ACT(5);Q2_GU_ACT(6);Q2_GU_ACT(7);Q2_GU_ACT(8);}
        #undef Q2_GU_ACT
        const auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    }
    return cudaSuccess;
}

cudaError_t dequant_bf16_gpu(const void* p, uint32_t type, uint16_t* out, int64_t n, cudaStream_t s) {
    if(type==13 || type==14 || iq_type(type)){
        if(!p || !out || n<=0 || n%256)return cudaErrorInvalidValue;
        if(type==13||type==14)kernels::dequant_bf16(type,p,0,1,n,out,s);
        else kernels::iq_dequant_bf16(type,p,n,out,s);
        return cudaGetLastError();
    }
    if (!p || !out || n < 1 || !supported(type) ||
        (type == 10 || type == 11 || type == 14 ? n % 256 : type == 42 ? n % 64 : type == 2 || type == 8 || type == 39 ? n % 32 : 0)) return cudaErrorInvalidValue;
    if(expert_dequant_fast_enabled() && type==10)
        dq_pairs<10><<<(n+2047)/2048,256,0,s>>>(static_cast<const uint8_t*>(p),out,n);
    else if(expert_dequant_fast_enabled() && type==11)
        dq_pairs<11><<<(n+2047)/2048,256,0,s>>>(static_cast<const uint8_t*>(p),out,n);
    else dequant_bf16_kernel<<<grid(n),256,0,s>>>(static_cast<const uint8_t*>(p),type,out,n);
    return cudaGetLastError();
}
cudaError_t packed_bf16_gemv_gpu(const void* p, uint32_t type, const float* x, float* out, int rows, int cols, int tokens, cudaStream_t s) {
    if (!p || !x || !out || rows < 1 || cols < 1 || tokens < 1 || tokens > 8 || !supported(type) ||
        (type == 10 || type == 11 || type == 14 ? cols % 256 : type == 42 ? cols % 64 : type == 2 || type == 8 || type == 39 ? cols % 32 : 0)) return cudaErrorInvalidValue;
    if(type==14)return q6_projection(p,x,out,rows,cols,tokens,cols,rows,true,false,s);
    gemv_kernel<true><<<dim3(rows,tokens),256,0,s>>>(static_cast<const uint8_t*>(p),type,x,out,rows,cols); return cudaGetLastError();
}
cudaError_t packed_gemv_gpu(const void* p, uint32_t type, const float* x, float* out, int rows, int cols, int tokens, cudaStream_t s) {
    if (!supported(type) || rows < 1 || cols < 1 || tokens < 1 || tokens > 65535 ||
        (type == 10 || type == 11 || type == 14 ? cols % 256 : type == 42 ? cols % 64 : type == 2 || type == 8 || type == 39 ? cols % 32 : 0)) return cudaErrorInvalidValue;
    if(type==14&&tokens<=8)return q6_projection(p,x,out,rows,cols,tokens,cols,rows,false,false,s);
    gemv_kernel<false><<<dim3(rows,tokens),256,0,s>>>(static_cast<const uint8_t*>(p), type, x, out, rows, cols); return cudaGetLastError();
}
cudaError_t rms_gpu(const float* x, const float* w, float* out, int width, int tokens, float eps, cudaStream_t s) {
    if (width < 1 || tokens < 1 || eps <= 0) return cudaErrorInvalidValue;
    rms_kernel<<<tokens,256,0,s>>>(x,w,out,width,eps); return cudaGetLastError();
}
cudaError_t round_bf16_gpu(float* x, int64_t n, cudaStream_t s) {
    if (n <= 0) return cudaErrorInvalidValue;
    round_kernel<<<grid(n),256,0,s>>>(x,n); return cudaGetLastError();
}
cudaError_t float_to_bf16_gpu(const float* x,uint16_t* out,int64_t n,cudaStream_t s) {
    if (n<=0) return cudaErrorInvalidValue;
    to_bf16_kernel<<<grid(n),256,0,s>>>(x,out,n);return cudaGetLastError();
}
cudaError_t bf16_to_float_gpu(const uint16_t* x,float* out,int64_t n,cudaStream_t s) {
    if (n<=0) return cudaErrorInvalidValue;
    from_bf16_kernel<<<grid(n),256,0,s>>>(x,out,n);return cudaGetLastError();
}
cudaError_t swiglu_gpu(const float* g, const float* u, const float* w, float* out, int width, int tokens, float limit, cudaStream_t s) {
    if (width < 1 || tokens < 1) return cudaErrorInvalidValue;
    swiglu_kernel<<<grid(int64_t(width)*tokens),256,0,s>>>(g,u,w,out,width,int64_t(width)*tokens,limit); return cudaGetLastError();
}
cudaError_t route_gpu(const float* x, const float* b, const float* vl, const uint8_t* image, int* ids, float* weights,
                      int experts, int top_k, int tokens, float scale, cudaStream_t s) {
    if (!((experts == 384 && top_k == 6) || (experts == 128 && top_k == 3)) || tokens < 1) return cudaErrorInvalidValue;
    route_kernel<<<tokens,32,0,s>>>(x,b,vl,image,ids,weights,experts,top_k,scale); return cudaGetLastError();
}
cudaError_t route_parallel_gpu(const float* x,const float* b,const float* vl,const uint8_t* image,
                              int* ids,float* weights,int experts,int top_k,int tokens,float scale,cudaStream_t stream){
    if(!x || !b || !ids || !weights || !((experts==384 && top_k==6)||(experts==128 && top_k==3)) || tokens<1)
        return cudaErrorInvalidValue;
    route_parallel_kernel<<<tokens,512,0,stream>>>(x,b,vl,image,ids,weights,experts,top_k,scale);return cudaGetLastError();
}
cudaError_t hc_project_gpu(const float* x, const float* fn, float* mixes, int hidden, int tokens, float eps, cudaStream_t s) {
    if (hidden < 1 || tokens < 1 || tokens > 65535 || eps <= 0) return cudaErrorInvalidValue;
    hc_project_kernel<<<dim3(24,tokens),256,0,s>>>(x,fn,mixes,hidden,eps); return cudaGetLastError();
}
cudaError_t hc_split_gpu(const float* x, const float* scale, const float* base, float* pre, float* post, float* comb,
                         int tokens, float eps, int iterations, cudaStream_t s) {
    if (tokens < 1 || eps <= 0 || iterations < 1) return cudaErrorInvalidValue;
    return hc_split_warp_gpu(x,scale,base,pre,post,comb,tokens,eps,iterations,s);
}
cudaError_t hc_pre_gpu(const float* x, const float* pre, float* out, int hidden, int tokens, cudaStream_t s) {
    if (hidden < 1 || tokens < 1) return cudaErrorInvalidValue;
    hc_pre_kernel<<<grid(int64_t(hidden)*tokens),256,0,s>>>(x,pre,out,hidden,int64_t(hidden)*tokens); return cudaGetLastError();
}
cudaError_t hc_post_gpu(const float* x, const float* branch, const float* post, const float* comb, float* out,
                        int hidden, int tokens, cudaStream_t s) {
    if (hidden < 1 || tokens < 1) return cudaErrorInvalidValue;
    hc_post_kernel<<<grid(int64_t(hidden)*tokens),256,0,s>>>(x,branch,post,comb,out,hidden,int64_t(hidden)*tokens); return cudaGetLastError();
}
cudaError_t rotate_gpu(float* x, int heads, int head_dim, int tokens, int64_t start, const Geometry& g,
                       bool compressed, bool inverse, cudaStream_t s) {
    if (heads < 1 || g.rope_dim != 64 || head_dim < g.rope_dim || tokens < 1 || start < 0) return cudaErrorInvalidValue;
    // Use the same FP32 frequency table on CPU and GPU; per-device pow rounding grows with position.
    const auto frequencies = rotary_frequencies(g, compressed);
    if(rotary_detail::shared_shape(heads,head_dim,tokens))
        return rotary_detail::launch(x,heads,head_dim,tokens,start,1,frequencies,inverse,s);
    rotate_kernel<<<grid(int64_t(tokens)*heads*g.rope_dim/2),256,0,s>>>(x,heads,head_dim,tokens,start,frequencies,inverse); return cudaGetLastError();
}
cudaError_t pool2_gpu(const float* a, const float* b, const float* ga, const float* gb, float* out, int width, cudaStream_t s) {
    if (width < 1) return cudaErrorInvalidValue;
    pool2_kernel<<<grid(width),256,0,s>>>(a,b,ga,gb,out,width); return cudaGetLastError();
}
cudaError_t engram_gate_gpu(float* h, const float* kv, const float* qw, const float* kw,
                           const uint8_t* image, int tokens, int width, cudaStream_t s) {
    if (tokens < 1 || width < 1) return cudaErrorInvalidValue;
    engram_gate_kernel<<<tokens * 4,256,0,s>>>(h,kv,qw,kw,image,width); return cudaGetLastError();
}
} // namespace strata::models::deepseek41
