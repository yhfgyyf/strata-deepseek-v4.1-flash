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
#include <math_constants.h>
#include <cmath>
#include <type_traits>
#include <atomic>

namespace strata::models::deepseek41 {
namespace {
std::atomic<bool> hc_vector_enabled{true};
std::atomic<bool> hc_reuse_enabled{false};
std::atomic<bool> hc_tile_enabled{true};
// Stable expert order, then original token/slot order; at most48 selected slots.
__global__ void route_group_short_kernel(const int* ids,const float* weights,int* rows,int* slots,float* grouped,int count) {
    __shared__ int selected[48];
    const int i=int(threadIdx.x);
    if(i<count)selected[i]=ids[i];
    __syncthreads();
    if(i>=count)return;
    int at=0;
    for(int j=0;j<count;++j)at+=selected[j]<selected[i]||(selected[j]==selected[i]&&j<i);
    rows[at]=i/6;slots[at]=i;grouped[at]=weights[i];
}

__global__ void route_handoff_short_kernel(const int* ids,const float* weights,int* rows,int* slots,float* grouped,int* handoff,int count) {
    __shared__ int selected[48],invalid[48];
    const int i=int(threadIdx.x);
    float weight=0;
    if(i<count){
        selected[i]=ids[i];weight=weights[i];handoff[i]=selected[i];
        invalid[i]=selected[i]<0||selected[i]>=384||!isfinite(weight)?i:count;
    }
    __syncthreads();
    if(i==0){
        int first=count;
        for(int j=0;j<count;++j)first=min(first,invalid[j]);
        handoff[count]=first==count?-1:first;
    }
    if(i>=count)return;
    int at=0;
    for(int j=0;j<count;++j)at+=selected[j]<selected[i]||(selected[j]==selected[i]&&j<i);
    rows[at]=i/6;slots[at]=i;grouped[at]=weight;
}

__global__ void target_greedy_kernel(const float* logits,int* output,int vocab) {
    const int row=int(blockIdx.x),tid=int(threadIdx.x),lane=tid&31,warp=tid/32;
    float best=-CUDART_INF_F;int id=0x7fffffff,bad=0;
    for(int i=tid;i<vocab;i+=256){const float value=logits[int64_t(row)*vocab+i];
        bad|=isnan(value)||value==CUDART_INF_F;
        if(value>best||(value==best&&i<id)){best=value;id=i;}}
    for(int delta=16;delta;delta/=2){const float v=__shfl_xor_sync(0xffffffff,best,delta);
        const int j=__shfl_xor_sync(0xffffffff,id,delta);bad|=__shfl_xor_sync(0xffffffff,bad,delta);
        if(v>best||(v==best&&j<id)){best=v;id=j;}}
    __shared__ float values[8];__shared__ int ids[8],invalid[8];
    if(!lane){values[warp]=best;ids[warp]=id;invalid[warp]=bad;}__syncthreads();
    if(warp)return;
    best=lane<8?values[lane]:-CUDART_INF_F;id=lane<8?ids[lane]:0x7fffffff;bad=lane<8?invalid[lane]:0;
    for(int delta=16;delta;delta/=2){const float v=__shfl_xor_sync(0xffffffff,best,delta);
        const int j=__shfl_xor_sync(0xffffffff,id,delta);bad|=__shfl_xor_sync(0xffffffff,bad,delta);
        if(v>best||(v==best&&j<id)){best=v;id=j;}}
    if(!lane)output[row]=bad||best==-CUDART_INF_F?-1:id;
}

__global__ void state_copy_kernel(StateCopyBatch batch) {
    int lo=0,hi=batch.count;
    while(lo+1<hi){const int mid=(lo+hi)/2;if(batch.jobs[mid].first_block<=int(blockIdx.x))lo=mid;else hi=mid;}
    const auto job=batch.jobs[lo];
    const unsigned offset=(unsigned(blockIdx.x)-job.first_block)*4096+unsigned(threadIdx.x)*16;
    if(offset<job.bytes)reinterpret_cast<uint4*>(job.destination)[offset/16]=reinterpret_cast<const uint4*>(job.source)[offset/16];
}

__device__ float half_value(const uint8_t* p) { return __half2float(*reinterpret_cast<const __half*>(p)); }
__device__ float q5_value(const uint8_t* p,int index) {
    const int group=index/32,lane=index%32;const uint8_t* scales=p+4;
    const int scale=group<4?scales[group]&63:(scales[group+4]&15)|((scales[group-4]>>6)<<4);
    const int minimum=group<4?scales[group+4]&63:(scales[group+4]>>4)|((scales[group]>>6)<<4);
    const int code=((p[48+(group/2)*32+lane]>>((group&1)*4))&15)|(((p[16+lane]>>group)&1)<<4);
    return (half_value(p)*scale)*code-half_value(p+2)*minimum;
}
__device__ constexpr int iq4_values[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
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
    if (type == 20) { p += (i / 32) * 18; return half_value(p)*iq4_values[(p[2+i%16]>>((i%32)/16*4))&15]; }
    if (type == 39) {
        p += (i / 32) * 17;
        constexpr int values[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
        const uint32_t bits = p[0] < 2 ? 0x00200000u << p[0] : uint32_t(p[0] - 1) << 23;
        return __uint_as_float(bits) * values[(p[1 + i % 16] >> ((i % 32) / 16 * 4)) & 15];
    }
    const int index = i % 256;
    if (type == 13) return q5_value(p+(i/256)*176,index);
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
bool supported(int type) { return type == 0 || type == 1 || type == 2 || type == 42 || type == 8 || type == 10 || type == 11 || type == 13 || type == 14 || type == 20 || type == 30 || type == 39; }
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
// A thread consumes both nibbles from one IQ4_NL byte. Engram's BF16 boundary
// stays fused with decoding, as in the existing Q5/Q6 Engram paths.
__global__ void iq4nl_decode_kernel(const uint8_t* packed,float* out,uint16_t* bf,int64_t n,bool round) {
    const int64_t pair=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=n/2)return;
    const int64_t block=pair/16;const int lane=pair%16;
    const auto* p=packed+block*18;const float scale=half_value(p);const int codes=p[2+lane];
    const float a=scale*iq4_values[codes&15],b=scale*iq4_values[codes>>4];
    const int64_t index=block*32+lane;
    if(bf){bf[index]=__bfloat16_as_ushort(__float2bfloat16_rn(a));bf[index+16]=__bfloat16_as_ushort(__float2bfloat16_rn(b));}
    else {out[index]=round?__bfloat162float(__float2bfloat16_rn(a)):a;out[index+16]=round?__bfloat162float(__float2bfloat16_rn(b)):b;}
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
template<int NT>
__global__ void q5_projection_warp(const uint8_t* __restrict__ weights,const float* __restrict__ x,
                                  float* __restrict__ out,int rows,int cols,int64_t ldx,int64_t ldy,
                                  bool round_weights,bool round_output) {
    const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;
    if(row>=rows)return;
    const int blocks=cols/256;float sum[NT]{};
    for(int b=0;b<blocks;++b){const auto* p=weights+(int64_t(row)*blocks+b)*176;
#pragma unroll
        for(int g=0;g<8;++g){float w=q5_value(p,g*32+lane);
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
    if(!lane){
#pragma unroll
        for(int t=0;t<NT;++t)out[int64_t(t)*ldy+row]=round_output?__bfloat162float(__float2bfloat16_rn(sum[t])):sum[t];
    }
}
// Eight physical lanes per row, four consecutive virtual lanes (elements) per physical lane:
// virtual lane l still owns elements b*256+g*32+l in (b,g) order with one accumulator, and the
// 32-lane shfl_down tree is replayed (deltas 16/8/4 across lanes, 2/1 in registers).
template<int NT>
__global__ void __launch_bounds__(128) q5_v2_kernel(const uint8_t* __restrict__ weights,const float* __restrict__ x,
                                  float* __restrict__ out,int rows,int cols,int64_t ldx,int64_t ldy,
                                  bool round_weights,bool round_output) {
    const int lane=threadIdx.x&31,sub=lane&7;
    const int row=(blockIdx.x*4+threadIdx.x/32)*4+(lane>>3);
    const bool valid=row<rows;
    const int blocks=cols/256;float sum[NT][4]{};
    const uint8_t* p=weights+int64_t(valid?row:0)*blocks*176;
    const float* xl=x+4*sub;
#pragma unroll 2
    for(int b=0;b<blocks;++b,p+=176){
        const uint4 h=__ldg(reinterpret_cast<const uint4*>(p));
        const uint32_t qh=__ldg(reinterpret_cast<const uint32_t*>(p+16)+sub);
        uint32_t qs[4];
#pragma unroll
        for(int k=0;k<4;++k)qs[k]=__ldg(reinterpret_cast<const uint32_t*>(p+48+k*32)+sub);
        const float d=__half2float(__ushort_as_half(uint16_t(h.x))),dm=__half2float(__ushort_as_half(uint16_t(h.x>>16)));
        uint8_t sc[12];
        const uint32_t sw[3]={h.y,h.z,h.w};
#pragma unroll
        for(int i=0;i<12;++i)sc[i]=uint8_t(sw[i/4]>>(8*(i%4)));
#pragma unroll
        for(int g=0;g<8;++g){
            const int scale=g<4?sc[g]&63:(sc[g+4]&15)|((sc[g-4]>>6)<<4);
            const int minimum=g<4?sc[g+4]&63:(sc[g+4]>>4)|((sc[g]>>6)<<4);
            float w[4];
#pragma unroll
            for(int e=0;e<4;++e){
                const int code=((int(qs[g/2]>>(8*e))>>((g&1)*4))&15)|((int(qh>>(8*e))>>g&1)<<4);
                w[e]=(d*scale)*code-dm*minimum;
                if(round_weights)w[e]=__bfloat162float(__float2bfloat16_rn(w[e]));
            }
#pragma unroll
            for(int t=0;t<NT;++t){
                const float4 v=__ldg(reinterpret_cast<const float4*>(xl+int64_t(t)*ldx+b*256+g*32));
                sum[t][0]+=w[0]*v.x;sum[t][1]+=w[1]*v.y;sum[t][2]+=w[2]*v.z;sum[t][3]+=w[3]*v.w;
            }
        }
    }
    // Virtual deltas 16, 8, 4 are physical deltas 4, 2, 1 inside each 8-lane row group.
#pragma unroll
    for(int delta=4;delta;delta>>=1){
#pragma unroll
        for(int t=0;t<NT;++t){
#pragma unroll
            for(int e=0;e<4;++e)sum[t][e]+=__shfl_down_sync(0xffffffff,sum[t][e],delta,8);
        }
    }
    if(sub==0&&valid){
#pragma unroll
        for(int t=0;t<NT;++t){
            const float s=(sum[t][0]+sum[t][2])+(sum[t][1]+sum[t][3]);
            out[int64_t(t)*ldy+row]=round_output?__bfloat162float(__float2bfloat16_rn(s)):s;
        }
    }
}
std::atomic<bool> q5_rows8_enabled{true};
cudaError_t q5_projection(const void* weights,const float* x,float* out,int rows,int cols,int tokens,
                         int64_t ldx,int64_t ldy,bool round_weights,bool round_output,cudaStream_t stream) {
    if(q5_rows8_enabled.load()&&tokens>=1&&tokens<=8&&cols%256==0&&ldx%4==0&&
       ((reinterpret_cast<uintptr_t>(weights)|reinterpret_cast<uintptr_t>(x))&15)==0){
#define Q5_ROWS8(N) case N:q5_v2_kernel<N><<<(rows+15)/16,128,0,stream>>>(static_cast<const uint8_t*>(weights),x,out,rows,cols,ldx,ldy,round_weights,round_output);break
        switch(tokens){Q5_ROWS8(1);Q5_ROWS8(2);Q5_ROWS8(3);Q5_ROWS8(4);Q5_ROWS8(5);Q5_ROWS8(6);Q5_ROWS8(7);Q5_ROWS8(8);}
#undef Q5_ROWS8
        return cudaGetLastError();
    }
#define Q5_PROJECTION(N) case N:q5_projection_warp<N><<<(rows+3)/4,128,0,stream>>>(static_cast<const uint8_t*>(weights),x,out,rows,cols,ldx,ldy,round_weights,round_output);break
    switch(tokens){Q5_PROJECTION(1);Q5_PROJECTION(2);Q5_PROJECTION(3);Q5_PROJECTION(4);Q5_PROJECTION(5);Q5_PROJECTION(6);Q5_PROJECTION(7);Q5_PROJECTION(8);default:return cudaErrorInvalidValue;}
#undef Q5_PROJECTION
    return cudaGetLastError();
}
__device__ float bf16(float value){return __bfloat162float(__float2bfloat16_rn(value));}
__global__ void draft_projection5_kernel(const uint8_t* weights,int type,const float* x,float* out,
                                         int cols,int64_t ldx,int64_t ldy,bool round_weights,bool round_output,
                                         int64_t group_weight_bytes=0,int64_t group_x=0,int64_t group_out=0){
    weights+=blockIdx.y*group_weight_bytes;x+=blockIdx.y*group_x;out+=blockIdx.y*group_out;
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
// FP32/BF16 rows with every weight of a thread loaded before the dot products: removes the
// per-iteration load latency chain. Same thread/column assignment, order and tree as above.
template<int N,int MAXI,bool BF>
__global__ void __launch_bounds__(256) verify_projection_prefetch(const uint8_t* __restrict__ p,const float* __restrict__ x,
                                                                 float* __restrict__ out,int cols,int64_t ldx,int64_t ldy){
    const int row=blockIdx.x;
    using W=typename std::conditional<BF,__nv_bfloat16,float>::type;
    const W* w=reinterpret_cast<const W*>(p)+int64_t(row)*cols;
    W weights[MAXI];
    #pragma unroll
    for(int i=0;i<MAXI;++i){const int col=threadIdx.x+256*i;if(col<cols)weights[i]=__ldg(w+col);}
    float values[N]{};
    #pragma unroll
    for(int i=0;i<MAXI;++i){
        const int col=threadIdx.x+256*i;
        if(col<cols){
            float weight;
            if constexpr(BF)weight=__bfloat162float(weights[i]);else weight=weights[i];
            #pragma unroll
            for(int t=0;t<N;++t)values[t]+=weight*x[int64_t(t)*ldx+col];
        }
    }
    sum_columns(values);
    if(threadIdx.x==0){
        #pragma unroll
        for(int t=0;t<N;++t)out[int64_t(t)*ldy+row]=values[t];
    }
}
std::atomic<bool> projection_prefetch_enabled{true};
template<int N>
bool launch_projection_prefetch(const void* p,uint32_t type,const float* x,float* out,int rows,int cols,int64_t ldx,int64_t ldy,cudaStream_t stream){
    if(!projection_prefetch_enabled.load()||(type!=0&&type!=30)||cols>8192)return false;
    const auto* w=static_cast<const uint8_t*>(p);
    #define VPP(M) do{if(type==30)verify_projection_prefetch<N,M,true><<<rows,256,0,stream>>>(w,x,out,cols,ldx,ldy);\
        else verify_projection_prefetch<N,M,false><<<rows,256,0,stream>>>(w,x,out,cols,ldx,ldy);}while(0)
    const int iters=(cols+255)/256;
    if(iters<=4)VPP(4);else if(iters<=8)VPP(8);else if(iters<=16)VPP(16);else if(iters<=20)VPP(20);else if(iters<=24)VPP(24);else VPP(32);
    #undef VPP
    return true;
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
// Four contiguous scalar lanes per physical thread. Each virtual lane retains
// the original K stride and the original 256-thread reduction tree.
__device__ float hc_sum_virtual256(float (&v)[4]) {
    __shared__ float partial[8];
    #pragma unroll
    for(int delta=4;delta;delta>>=1) {
        #pragma unroll
        for(int j=0;j<4;++j)v[j]+=__shfl_down_sync(0xffffffff,v[j],delta,8);
    }
    const float even=v[0]+v[2],odd=v[1]+v[3];
    if((threadIdx.x&7)==0)partial[threadIdx.x>>3]=even+odd;
    __syncthreads();
    float sum=threadIdx.x<8?partial[threadIdx.x]:0.f;
    if(threadIdx.x<32)for(int delta=16;delta;delta>>=1)
        sum+=__shfl_down_sync(0xffffffff,sum,delta);
    return sum;
}
__global__ void hc_project_vector_kernel(const float* x,const float* fn,float* mixes,int hidden,float eps) {
    const int row=blockIdx.x,token=blockIdx.y,width=hidden*4;
    float square[4]{},dot[4]{};
    for(int col=threadIdx.x*4;col<width;col+=256) {
        const float4 value=*reinterpret_cast<const float4*>(x+int64_t(token)*width+col);
        const float4 weight=*reinterpret_cast<const float4*>(fn+int64_t(row)*width+col);
        const float values[4]={value.x,value.y,value.z,value.w};
        const float weights[4]={weight.x,weight.y,weight.z,weight.w};
        #pragma unroll
        for(int j=0;j<4;++j) {
            square[j]+=values[j]*values[j];
            dot[j]+=values[j]*weights[j];
        }
    }
    const float sum=hc_sum_virtual256(square);
    __shared__ float inverse;
    if(!threadIdx.x)inverse=rsqrtf(sum/width+eps);
    __syncthreads();
    const float product=hc_sum_virtual256(dot);
    if(!threadIdx.x)mixes[token*24+row]=product*inverse;
}

// Four output rows reuse one FP32 input load and one norm. The four virtual
// lanes retain their original K stride, products and 256-lane reduction tree.
__global__ void hc_project_reuse_kernel(const float* x,const float* fn,float* mixes,int hidden,float eps) {
    const int first=blockIdx.x*4,token=blockIdx.y,width=hidden*4;
    float square[4]{},dot[4][4]{};
    for(int col=threadIdx.x*4;col<width;col+=256){
        const float4 value=*reinterpret_cast<const float4*>(x+int64_t(token)*width+col);
        const float values[4]={value.x,value.y,value.z,value.w};
        #pragma unroll
        for(int j=0;j<4;++j)square[j]+=values[j]*values[j];
        #pragma unroll
        for(int row=0;row<4;++row){
            const float4 weight=*reinterpret_cast<const float4*>(fn+int64_t(first+row)*width+col);
            const float weights[4]={weight.x,weight.y,weight.z,weight.w};
            #pragma unroll
            for(int j=0;j<4;++j)dot[row][j]+=values[j]*weights[j];
        }
    }
    const float sum=hc_sum_virtual256(square);
    __shared__ float inverse;
    if(!threadIdx.x)inverse=rsqrtf(sum/width+eps);
    __syncthreads();
    #pragma unroll
    for(int row=0;row<4;++row){
        const float product=hc_sum_virtual256(dot[row]);
        if(!threadIdx.x)mixes[token*24+first+row]=product*inverse;
        __syncthreads(); // All readers finish before the next reduction reuses partial[].
    }
}

// Prefill hc_project, bit-exact with hc_project_vector_kernel: four 64-thread groups own six rows
// each; a block covers TT tokens so x streams from DRAM once and each fn element feeds TT tokens.
// Every (token,row) keeps its 256 virtual-lane chains and the virtual-256 reduction tree.
__device__ __forceinline__ float hc_sum_group64(float (&v)[4],float* partial,int lane,int barrier) {
    #pragma unroll
    for(int delta=4;delta;delta>>=1) {
        #pragma unroll
        for(int j=0;j<4;++j)v[j]+=__shfl_down_sync(0xffffffff,v[j],delta,8);
    }
    const float even=v[0]+v[2],odd=v[1]+v[3];
    if((lane&7)==0)partial[lane>>3]=even+odd;
    asm volatile("bar.sync %0, 64;"::"r"(barrier));
    float sum=lane<8?partial[lane]:0.f;
    if(lane<32)for(int delta=16;delta;delta>>=1)
        sum+=__shfl_down_sync(0xffffffff,sum,delta);
    asm volatile("bar.sync %0, 64;"::"r"(barrier));
    return sum;
}
template<int TT>
__global__ void __launch_bounds__(256) hc_project_tile_kernel(const float* x,const float* fn,float* mixes,int hidden,float eps,int tokens) {
    const int group=threadIdx.x>>6,lane=threadIdx.x&63,width=hidden*4,first=group*6,t0=blockIdx.x*TT;
    __shared__ float partial[4][8];
    const float* xs[TT];
    #pragma unroll
    for(int i=0;i<TT;++i)xs[i]=x+int64_t(min(t0+i,tokens-1))*width;
    const float* fs=fn+int64_t(first)*width;
    float square[TT][4]{},dot[TT][6][4]{};
    for(int col=lane*4;col<width;col+=256) {
        float values[TT][4];
        #pragma unroll
        for(int i=0;i<TT;++i){const float4 v=__ldg(reinterpret_cast<const float4*>(xs[i]+col));values[i][0]=v.x;values[i][1]=v.y;values[i][2]=v.z;values[i][3]=v.w;}
        #pragma unroll
        for(int r=0;r<6;++r){
            const float4 w=__ldg(reinterpret_cast<const float4*>(fs+int64_t(r)*width+col));
            const float weights[4]={w.x,w.y,w.z,w.w};
            #pragma unroll
            for(int i=0;i<TT;++i)
                #pragma unroll
                for(int j=0;j<4;++j)dot[i][r][j]+=values[i][j]*weights[j];
        }
        #pragma unroll
        for(int i=0;i<TT;++i)
            #pragma unroll
            for(int j=0;j<4;++j)square[i][j]+=values[i][j]*values[i][j];
    }
    #pragma unroll
    for(int i=0;i<TT;++i){
        const float sum=hc_sum_group64(square[i],partial[group],lane,1+group);
        const float inverse=rsqrtf(sum/width+eps);
        #pragma unroll
        for(int r=0;r<6;++r){
            const float product=hc_sum_group64(dot[i][r],partial[group],lane,1+group);
            if(!lane&&t0+i<tokens)mixes[(t0+i)*24+first+r]=product*inverse;
        }
    }
}
// Same reduction as hc_sum_virtual256, synchronizing only the two compute warps.
__device__ float hc_sum_virtual256_64(float (&v)[4]) {
    __shared__ float partial[8];
    #pragma unroll
    for(int delta=4;delta;delta>>=1) {
        #pragma unroll
        for(int j=0;j<4;++j)v[j]+=__shfl_down_sync(0xffffffff,v[j],delta,8);
    }
    const float even=v[0]+v[2],odd=v[1]+v[3];
    if((threadIdx.x&7)==0)partial[threadIdx.x>>3]=even+odd;
    asm volatile("bar.sync 1, 64;");
    float sum=threadIdx.x<8?partial[threadIdx.x]:0.f;
    if(threadIdx.x<32)for(int delta=16;delta;delta>>=1)
        sum+=__shfl_down_sync(0xffffffff,sum,delta);
    return sum;
}
// hc_project, bit-exact with hc_project_vector_kernel: 64 compute threads keep the four
// virtual lanes per thread and their 80-step chains; all 256 threads stream x and fn rows into
// shared memory with cp.async (double-buffered 16-step chunks) so many loads are in flight.
__device__ __forceinline__ void hc_cp16(void* dst, const void* src) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src));
}
// hc_split_warp_kernel for one token on lanes 0..15 of a warp (same expressions and shuffles).
__device__ void hc_split_token(const float* mixes,const float* scale,const float* base,float* pre,float* post,float* comb,
                               int token,int lane,float eps,int iterations){
 constexpr unsigned mask=0xffffu;
 const float* x=mixes+token*24;
 if(lane<4){pre[token*4+lane]=sigmoid(__ldcg(x+lane)*scale[0]+base[lane])+eps;post[token*4+lane]=2.f*sigmoid(__ldcg(x+4+lane)*scale[1]+base[4+lane]);}
 float v=__ldcg(x+8+lane)*scale[2]+base[8+lane];const int row=lane/4,col=lane%4;
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
__device__ unsigned hc_split_counters[2][8];
struct HcSplitArgs{const float* scale;const float* base;float* pre;float* post;float* comb;float eps;int iterations;int slot;};
template<int CH,bool SPLIT=false>
__global__ void __launch_bounds__(256) hc_staged_kernel(const float* x,const float* fn,float* mixes,int hidden,float eps,HcSplitArgs split={}) {
    constexpr int STEP = 256, CHUNK = CH * STEP;      // floats per chunk per operand
    extern __shared__ __align__(16) float sm[];      // [2][2][CHUNK]
    const int row=blockIdx.x,token=blockIdx.y,width=hidden*4,tid=threadIdx.x;
    const float* xs=x+int64_t(token)*width; const float* fs=fn+int64_t(row)*width;
    const int chunks=width/CHUNK;
    auto issue=[&](int c,int buf){
        float* dx=sm+(buf*2)*CHUNK; float* df=dx+CHUNK;
        for(int i=tid*4;i<CHUNK;i+=1024){hc_cp16(dx+i,xs+c*CHUNK+i);hc_cp16(df+i,fs+c*CHUNK+i);}
        asm volatile("cp.async.commit_group;");
    };
    float square[4]{},dot[4]{};
    issue(0,0);
    for(int c=0;c<chunks;++c){
        if(c+1<chunks)issue(c+1,(c+1)&1); else asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        if(tid<64){
            const float* dx=sm+((c&1)*2)*CHUNK; const float* df=dx+CHUNK;
#pragma unroll
            for(int k=0;k<CH;++k){
                const float4 value=*reinterpret_cast<const float4*>(dx+k*STEP+tid*4);
                const float4 weight=*reinterpret_cast<const float4*>(df+k*STEP+tid*4);
                const float values[4]={value.x,value.y,value.z,value.w};
                const float weights[4]={weight.x,weight.y,weight.z,weight.w};
#pragma unroll
                for(int j=0;j<4;++j){square[j]+=values[j]*values[j];dot[j]+=values[j]*weights[j];}
            }
        }
        __syncthreads();
    }
    if(tid>=64)return;   // the reductions below use only the first two warps
    const float sum=hc_sum_virtual256_64(square);
    __shared__ float inverse;
    if(!tid)inverse=rsqrtf(sum/width+eps);
    asm volatile("bar.sync 1, 64;");
    const float product=hc_sum_virtual256_64(dot);
    if(!tid)mixes[token*24+row]=product*inverse;
    if constexpr(SPLIT){
        // The last of the token's 24 row blocks runs the split on the published mixes.
        __shared__ bool last;
        if(!tid){__threadfence();last=atomicAdd(&hc_split_counters[split.slot][token],1u)==gridDim.x-1;}
        asm volatile("bar.sync 1, 64;");
        if(last&&tid<16){
            __threadfence();
            if(!tid)hc_split_counters[split.slot][token]=0;
            hc_split_token(mixes,split.scale,split.base,split.pre,split.post,split.comb,token,tid,split.eps,split.iterations);
        }
    }
}
// Decode hc_project + hc_split, bit-exact with hc_staged_kernel<8,true>. Each (token,row) is a fixed tree
// over 256 independent 80-step chains (virtual lane L = 4t+j owns cols L+256s). Block (row, part) owns
// chains [64 part, 64 part + 64) of every token, so each fn element is read once for all tokens; a warp
// holds one 8-thread group of the tree and runs its first stage. The last block finishes the 8-partial
// stage of every (token,row), then the split.
std::atomic<bool> hc_lane_enabled{true};
__device__ unsigned hc_lane_counters[2];
__device__ float hc_lane_partials[2][8*25*8];
__global__ void __launch_bounds__(512) hc_lane_split_kernel(const float* x,const float* fn,float* mixes,int hidden,float eps,
                                                            int tokens,HcSplitArgs split) {
    constexpr int C=64;
    const int row=blockIdx.x,part=blockIdx.y,width=hidden*4;
    const int token=threadIdx.x/C,L=part*C+threadIdx.x%C;
    float* partials=hc_lane_partials[split.slot];
    const float* xs=x+int64_t(token)*width+L;
    const float* fs=fn+int64_t(row)*width+L;
    float sq=0.f,dt=0.f;
    #pragma unroll
    for(int s=0;s<80;++s){
        const float v=__ldg(xs+256*s),w=__ldg(fs+256*s);
        sq+=v*v; dt+=v*w;
    }
    auto first_stage=[](float v){
        v+=__shfl_down_sync(0xffffffff,v,16);
        v+=__shfl_down_sync(0xffffffff,v,8);
        v+=__shfl_down_sync(0xffffffff,v,4);
        const float even=__shfl_sync(0xffffffff,v,0)+__shfl_sync(0xffffffff,v,2);
        const float odd=__shfl_sync(0xffffffff,v,1)+__shfl_sync(0xffffffff,v,3);
        return even+odd;
    };
    const float pd=first_stage(dt),ps=first_stage(sq);
    if((threadIdx.x&31)==0){
        __stcg(partials+(token*25+row)*8+(L>>5),pd);
        if(row==0)__stcg(partials+(token*25+24)*8+(L>>5),ps);
    }
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if(threadIdx.x==0)last=atomicAdd(&hc_lane_counters[split.slot],1u)==gridDim.x*gridDim.y-1;
    __syncthreads();
    if(!last)return;
    __threadfence();
    // hc_sum_virtual256's last stage: lanes 8..31 contribute zeros at deltas 16 and 8.
    auto tree=[](const float* p){
        float q[8];
        #pragma unroll
        for(int i=0;i<8;++i){q[i]=__ldcg(p+i);q[i]+=0.f;q[i]+=0.f;}
        return ((q[0]+q[4])+(q[2]+q[6]))+((q[1]+q[5])+(q[3]+q[7]));
    };
    for(int item=threadIdx.x;item<tokens*24;item+=blockDim.x){
        const int t=item/24,r=item%24;
        const float inverse=rsqrtf(tree(partials+(t*25+24)*8)/width+eps);
        __stcg(mixes+t*24+r,tree(partials+(t*25+r)*8)*inverse);
    }
    __threadfence();
    __syncthreads();
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    if(warp<tokens&&lane<16)
        hc_split_token(mixes,split.scale,split.base,split.pre,split.post,split.comb,warp,lane,split.eps,split.iterations);
    if(threadIdx.x==0)hc_lane_counters[split.slot]=0;
}
std::atomic<bool> hc_staged_enabled{true};
std::atomic<bool> hc_split_fused_enabled{true};
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
    if(type==20){
        if(!p||!out||n<1||n%32)return cudaErrorInvalidValue;
        iq4nl_decode_kernel<<<(n/2+255)/256,256,0,s>>>(static_cast<const uint8_t*>(p),out,nullptr,n,false);
        return cudaGetLastError();
    }
    if(type==13 || type==14 || iq_type(type)){
        if(!p || !out || n<=0 || n%256)return cudaErrorInvalidValue;
        kernels::dequant_f32(type,p,0,1,n,out,s);return cudaGetLastError();
    }
    if (!supported(type) || n <= 0 || (type == 10 || type == 11 || type == 13 || type == 14 ? n % 256 : type == 42 ? n % 64 : type == 2 || type == 8 || type == 20 || type == 39 ? n % 32 : 0)) return cudaErrorInvalidValue;
    dequant_kernel<<<grid(n),256,0,s>>>(static_cast<const uint8_t*>(p), type, out, n); return cudaGetLastError();
}
cudaError_t dequant_round_bf16_gpu(const void* p, uint32_t type, float* out, int64_t n, cudaStream_t s) {
    if(type==20){
        if(!p||!out||n<1||n%32)return cudaErrorInvalidValue;
        iq4nl_decode_kernel<<<(n/2+255)/256,256,0,s>>>(static_cast<const uint8_t*>(p),out,nullptr,n,true);
        return cudaGetLastError();
    }
    if(type==13 && kernels::dequant_q5_f32_bf16(p,n,out,s))return cudaGetLastError();
    if(type==14 && kernels::dequant_q6_f32_bf16(p,n,out,s))return cudaGetLastError();
    const cudaError_t decoded=dequant_gpu(p,type,out,n,s);
    return decoded==cudaSuccess?round_bf16_gpu(out,n,s):decoded;
}
// Q8_0 draft projection (5 or 7 tokens): one warp per row, eight virtual lanes per lane.
// Same per-element value (half(d) * q, optional BF16 round), per-virtual-lane column order and
// the 256-thread sum_columns tree (replayed as in short_sum_warp).
template<int Count>
__device__ __forceinline__ void dp_sum_warp(float (&values)[Count][8], float (&sums)[Count]) {
    const int lane = threadIdx.x & 31;
    for (int delta = 2; delta; delta >>= 1) {
#pragma unroll
        for (int c = 0; c < Count; ++c) {
#pragma unroll
            for (int j = 0; j < 8; ++j) values[c][j] += __shfl_down_sync(0xffffffff, values[c][j], delta, 4);
        }
    }
#pragma unroll
    for (int c = 0; c < Count; ++c) {
        const float partial = ((values[c][0] + values[c][4]) + (values[c][2] + values[c][6])) +
                              ((values[c][1] + values[c][5]) + (values[c][3] + values[c][7]));
        const float gathered = __shfl_sync(0xffffffff, partial, (lane & 7) * 4);
        sums[c] = lane < 8 ? gathered : 0.f;
    }
    for (int delta = 16; delta; delta >>= 1) {
#pragma unroll
        for (int c = 0; c < Count; ++c) sums[c] += __shfl_down_sync(0xffffffff, sums[c], delta);
    }
}
template<int NT>
__global__ void __launch_bounds__(128) dp_q8_warp(const uint8_t* __restrict__ weights, const float* __restrict__ x, float* out,
                                                  int rows, int cols, int64_t ldx, int64_t ldy, bool round_weights, bool round_output) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int bpr = cols / 32;
    const uint8_t* p = weights + (int64_t(row) * bpr + lane / 4) * 34;
    const int off = 2 + (lane & 3) * 8;
    const float* xl = x + lane * 8;
    float values[NT][8]{};
    const int iters = cols / 256;
#pragma unroll 4
    for (int i = 0; i < iters; ++i, p += 8 * 34) {
        const float d = __half2float(*reinterpret_cast<const __half*>(p));
        const uint16_t* q16 = reinterpret_cast<const uint16_t*>(p + off);
        float w[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint16_t u = __ldg(q16 + k);
            w[2 * k] = d * int(int8_t(u & 255));
            w[2 * k + 1] = d * int(int8_t(u >> 8));
        }
        if (round_weights) {
#pragma unroll
            for (int j = 0; j < 8; ++j) w[j] = __bfloat162float(__float2bfloat16_rn(w[j]));
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 a = __ldg(reinterpret_cast<const float4*>(xl + t * ldx + i * 256));
            const float4 b = __ldg(reinterpret_cast<const float4*>(xl + t * ldx + i * 256 + 4));
            const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) values[t][j] += w[j] * v[j];
        }
    }
    float sums[NT];
    dp_sum_warp(values, sums);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) out[int64_t(t) * ldy + row] = round_output ? __bfloat162float(__float2bfloat16_rn(sums[t])) : sums[t];
    }
}
// BF16 rows: eight virtual lanes per lane via one 16-byte load; same order/tree as above.
template<int NT>
__global__ void __launch_bounds__(128) dp_bf16_warp(const uint8_t* __restrict__ weights, const float* __restrict__ x, float* out,
                                                    int rows, int cols, int64_t ldx, int64_t ldy, bool round_output) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wrow = reinterpret_cast<const uint4*>(weights + (int64_t(row) * cols + lane * 8) * 2);
    const float* xl = x + lane * 8;
    float values[NT][8]{};
    const int iters = cols / 256;
#pragma unroll 2
    for (int i = 0; i < iters; ++i) {
        const uint4 u = __ldg(wrow + i * 32);
        const uint32_t ws[4] = {u.x, u.y, u.z, u.w};
        float w[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) { w[2 * k] = __uint_as_float(ws[k] << 16); w[2 * k + 1] = __uint_as_float(ws[k] & 0xffff0000u); }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 a = __ldg(reinterpret_cast<const float4*>(xl + t * ldx + i * 256));
            const float4 b = __ldg(reinterpret_cast<const float4*>(xl + t * ldx + i * 256 + 4));
            const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) values[t][j] += w[j] * v[j];
        }
    }
    float sums[NT];
    dp_sum_warp(values, sums);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) out[int64_t(t) * ldy + row] = round_output ? __bfloat162float(__float2bfloat16_rn(sums[t])) : sums[t];
    }
}
std::atomic<bool> draft_q8_warp_enabled{true};
std::atomic<bool> draft_dedup_enabled{true};
std::atomic<bool> draft_stage_enabled{true};
void draft_stage_set_enabled(bool value){draft_stage_enabled.store(value);}
std::atomic<bool> draft_group_enabled{true};
// Large Q8_0 draft matrices only: small ones keep more CTAs with the block-per-row kernel.
template<int NT>
bool draft_q8_warp(const void* weights,uint32_t type,const float* x,float* out,int rows,int cols,int64_t ldx,int64_t ldy,
                   bool round_weights,bool round_output,cudaStream_t stream){
    if(!draft_q8_warp_enabled.load()||(type!=8&&type!=30)||rows<4096||cols%256||ldx%4||
       (reinterpret_cast<uintptr_t>(x)&15)||(reinterpret_cast<uintptr_t>(weights)&(type==30?15:1)))return false;
    // BF16 values are unchanged by the optional weight rounding.
    if(type==30){dp_bf16_warp<NT><<<(rows+3)/4,128,0,stream>>>(static_cast<const uint8_t*>(weights),x,out,rows,cols,ldx,ldy,round_output);return true;}
    dp_q8_warp<NT><<<(rows+3)/4,128,0,stream>>>(static_cast<const uint8_t*>(weights),x,out,rows,cols,ldx,ldy,round_weights,round_output);
    return true;
}
cudaError_t draft_projection5_gpu(const void* weights,uint32_t type,const float* x,float* out,
                                 int rows,int cols,int64_t ldx,int64_t ldy,bool round_weights,
                                 bool round_output,cudaStream_t stream){
    if(!weights||!x||!out||rows<1||cols<1||ldx<cols||ldy<rows||
       (type!=0&&type!=2&&type!=8&&type!=13&&type!=14&&type!=30)||((type==13||type==14)?cols%256:(type==2||type==8)&&cols%32))return cudaErrorInvalidValue;
    if(type==13)return q5_projection(weights,x,out,rows,cols,5,ldx,ldy,round_weights,round_output,stream);
    if(type==14)return q6_projection(weights,x,out,rows,cols,5,ldx,ldy,round_weights,round_output,stream);
    if(draft_q8_warp<5>(weights,type,x,out,rows,cols,ldx,ldy,round_weights,round_output,stream))return cudaGetLastError();
    draft_projection5_kernel<<<rows,256,0,stream>>>(static_cast<const uint8_t*>(weights),type,x,out,cols,ldx,ldy,round_weights,round_output);
    return cudaGetLastError();
}
// draft_projection5_kernel for FP32/BF16 weights (BF16 rounding of BF16 values is the identity) with
// every weight of a thread loaded first; same order and tree.
template<bool BF>
__global__ void __launch_bounds__(256) draft_projection5_prefetch(const uint8_t* weights,const float* x,float* out,int cols,int64_t ldx,int64_t ldy,
                                                                bool round_output,int64_t group_weight_bytes,int64_t group_x,int64_t group_out){
    weights+=blockIdx.y*group_weight_bytes;x+=blockIdx.y*group_x;out+=blockIdx.y*group_out;
    using W=typename std::conditional<BF,__nv_bfloat16,float>::type;
    const W* w=reinterpret_cast<const W*>(weights)+int64_t(blockIdx.x)*cols;
    W cache[16];
    #pragma unroll
    for(int i=0;i<16;++i){const int col=threadIdx.x+256*i;if(col<cols)cache[i]=__ldg(w+col);}
    float values[5]{};
    #pragma unroll
    for(int i=0;i<16;++i){
        const int col=threadIdx.x+256*i;
        if(col<cols){
            float weight;if constexpr(BF)weight=__bfloat162float(cache[i]);else weight=cache[i];
            #pragma unroll
            for(int t=0;t<5;++t)values[t]+=weight*x[int64_t(t)*ldx+col];
        }
    }
    sum_columns(values);
    if(threadIdx.x==0){
        #pragma unroll
        for(int t=0;t<5;++t)out[int64_t(t)*ldy+blockIdx.x]=round_output?bf16(values[t]):values[t];
    }
}
cudaError_t draft_projection5_grouped_gpu(const void* weights,uint32_t type,const float* x,float* out,
                                         int rows,int cols,int64_t ldx,int64_t ldy,int groups,int64_t group_weight_bytes,
                                         int64_t group_x,int64_t group_out,bool round_weights,bool round_output,cudaStream_t stream){
    if(!draft_group_enabled.load()||!weights||!x||!out||rows<1||cols<1||ldx<cols||ldy<rows||groups<1||groups>65535||
       (type!=0&&type!=30)||rows>=4096)return cudaErrorNotSupported;
    if(cols<=4096&&projection_prefetch_enabled.load()&&(type==30||!round_weights)){
        if(type==30)draft_projection5_prefetch<true><<<dim3(rows,groups),256,0,stream>>>(static_cast<const uint8_t*>(weights),x,out,cols,ldx,ldy,
            round_output,group_weight_bytes,group_x,group_out);
        else draft_projection5_prefetch<false><<<dim3(rows,groups),256,0,stream>>>(static_cast<const uint8_t*>(weights),x,out,cols,ldx,ldy,
            round_output,group_weight_bytes,group_x,group_out);
        return cudaGetLastError();
    }
    draft_projection5_kernel<<<dim3(rows,groups),256,0,stream>>>(static_cast<const uint8_t*>(weights),type,x,out,cols,ldx,ldy,
        round_weights,round_output,group_weight_bytes,group_x,group_out);
    return cudaGetLastError();
}
cudaError_t draft_projection7_gpu(const void* weights,uint32_t type,const float* x,float* out,
                                 int rows,int cols,int64_t ldx,int64_t ldy,bool round_weights,
                                 bool round_output,cudaStream_t stream){
    if(!weights||!x||!out||rows<1||cols<1||ldx<cols||ldy<rows||
       (type!=0&&type!=2&&type!=8&&type!=13&&type!=14&&type!=30)||((type==13||type==14)?cols%256:(type==2||type==8)&&cols%32))return cudaErrorInvalidValue;
    if(type==13)return q5_projection(weights,x,out,rows,cols,7,ldx,ldy,round_weights,round_output,stream);
    if(type==14)return q6_projection(weights,x,out,rows,cols,7,ldx,ldy,round_weights,round_output,stream);
    if(draft_q8_warp<7>(weights,type,x,out,rows,cols,ldx,ldy,round_weights,round_output,stream))return cudaGetLastError();
    draft_projection7_kernel<<<rows,256,0,stream>>>(static_cast<const uint8_t*>(weights),type,x,out,cols,ldx,ldy,round_weights,round_output);
    return cudaGetLastError();
}
cudaError_t draft_fused_moe_gpu(const void* experts,uint32_t expert_type,uint64_t projection,const void* shared_gate,
                               const void* shared_up,const void* shared_down,uint32_t shared_type,
                               const float* x,const int* ids,const float* routing,float* scratch,
                               float* out,int tokens,cudaStream_t stream,float* parts){
    const uint64_t expected=expert_type==39?uint64_t(5120)*2304/32*17:uint64_t(5120)*2304/256*84;
    if((expert_type!=10&&expert_type!=39)||!experts||projection!=expected||!shared_gate||!shared_up||!shared_down||
       (shared_type!=2&&shared_type!=8&&shared_type!=30)||!x||!ids||!routing||!scratch||!out||tokens<1||tokens>7)return cudaErrorInvalidValue;
    if(expert_type==39 && shared_type==30 && parts && draft_dedup_enabled.load()) {
        const auto* e=static_cast<const uint8_t*>(experts);
        int* desc=reinterpret_cast<int*>(parts+int64_t(7)*4*5120);
        draft_dedup::plan<<<1,32,0,stream>>>(ids,routing,tokens,desc);
        auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
        if(draft_stage_enabled.load()){
            draft_dedup::gu<true><<<dim3(576,tokens*3+1),128,0,stream>>>(e,projection,static_cast<const uint8_t*>(shared_gate),
                static_cast<const uint8_t*>(shared_up),x,ids,desc,scratch,tokens);
            error=cudaGetLastError();if(error!=cudaSuccess)return error;
            draft_dedup::down<true><<<dim3(1280,tokens*3+1),128,0,stream>>>(e,projection,static_cast<const uint8_t*>(shared_down),ids,scratch,desc,parts,tokens);
        }else{
            draft_dedup::gu<false><<<dim3(576,tokens*3+1),128,0,stream>>>(e,projection,static_cast<const uint8_t*>(shared_gate),
                static_cast<const uint8_t*>(shared_up),x,ids,desc,scratch,tokens);
            error=cudaGetLastError();if(error!=cudaSuccess)return error;
            draft_dedup::down<false><<<dim3(1280,tokens*3+1),128,0,stream>>>(e,projection,static_cast<const uint8_t*>(shared_down),ids,scratch,desc,parts,tokens);
        }
        error=cudaGetLastError();if(error!=cudaSuccess)return error;
        draft_dedup::combine<<<(tokens*5120+255)/256,256,0,stream>>>(parts,ids,out,tokens);
        return cudaGetLastError();
    }
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
       (type!=0&&type!=30&&type!=8&&type!=13&&type!=14)||((type==13||type==14)?cols%256:type==8&&cols%32))return cudaErrorInvalidValue;
    if(type==13)return q5_projection(p,x,out,rows,cols,tokens,ldx,ldy,false,false,stream);
    if(type==14)return q6_projection(p,x,out,rows,cols,tokens,ldx,ldy,false,false,stream);
    #define VERIFY_PROJECTION(N) case N: if(!launch_projection_prefetch<N>(p,type,x,out,rows,cols,ldx,ldy,stream))verify_projection_kernel<N><<<rows,256,0,stream>>>(static_cast<const uint8_t*>(p),type,x,out,cols,ldx,ldy);break
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
    if(type==20){
        if(!p||!out||n<1||n%32)return cudaErrorInvalidValue;
        iq4nl_decode_kernel<<<(n/2+255)/256,256,0,s>>>(static_cast<const uint8_t*>(p),nullptr,out,n,false);
        return cudaGetLastError();
    }
    if(type==13 || type==14 || iq_type(type)){
        if(!p || !out || n<=0 || n%256)return cudaErrorInvalidValue;
        if(type==13||type==14)kernels::dequant_bf16(type,p,0,1,n,out,s);
        else kernels::iq_dequant_bf16(type,p,n,out,s);
        return cudaGetLastError();
    }
    if (!p || !out || n < 1 || !supported(type) ||
        (type == 10 || type == 11 || type == 13 || type == 14 ? n % 256 : type == 42 ? n % 64 : type == 2 || type == 8 || type == 20 || type == 39 ? n % 32 : 0)) return cudaErrorInvalidValue;
    if(expert_dequant_fast_enabled() && type==10)
        dq_pairs<10><<<(n+2047)/2048,256,0,s>>>(static_cast<const uint8_t*>(p),out,n);
    else if(expert_dequant_fast_enabled() && type==11)
        dq_pairs<11><<<(n+2047)/2048,256,0,s>>>(static_cast<const uint8_t*>(p),out,n);
    else dequant_bf16_kernel<<<grid(n),256,0,s>>>(static_cast<const uint8_t*>(p),type,out,n);
    return cudaGetLastError();
}
cudaError_t packed_bf16_gemv_gpu(const void* p, uint32_t type, const float* x, float* out, int rows, int cols, int tokens, cudaStream_t s) {
    if (!p || !x || !out || rows < 1 || cols < 1 || tokens < 1 || tokens > 8 || !supported(type) ||
        (type == 10 || type == 11 || type == 13 || type == 14 ? cols % 256 : type == 42 ? cols % 64 : type == 2 || type == 8 || type == 20 || type == 39 ? cols % 32 : 0)) return cudaErrorInvalidValue;
    if(type==13)return q5_projection(p,x,out,rows,cols,tokens,cols,rows,true,false,s);
    if(type==14)return q6_projection(p,x,out,rows,cols,tokens,cols,rows,true,false,s);
    gemv_kernel<true><<<dim3(rows,tokens),256,0,s>>>(static_cast<const uint8_t*>(p),type,x,out,rows,cols); return cudaGetLastError();
}
cudaError_t packed_gemv_gpu(const void* p, uint32_t type, const float* x, float* out, int rows, int cols, int tokens, cudaStream_t s) {
    if (!supported(type) || rows < 1 || cols < 1 || tokens < 1 || tokens > 65535 ||
        (type == 10 || type == 11 || type == 13 || type == 14 ? cols % 256 : type == 42 ? cols % 64 : type == 2 || type == 8 || type == 20 || type == 39 ? cols % 32 : 0)) return cudaErrorInvalidValue;
    if(type==13&&tokens<=8)return q5_projection(p,x,out,rows,cols,tokens,cols,rows,false,false,s);
    if(type==14&&tokens<=8)return q6_projection(p,x,out,rows,cols,tokens,cols,rows,false,false,s);
    if(tokens==1&&launch_projection_prefetch<1>(p,type,x,out,rows,cols,cols,rows,s))return cudaGetLastError();
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
cudaError_t state_copy_gpu(const StateCopyBatch& batch,cudaStream_t stream) {
    if(batch.count<1||batch.count>96||batch.blocks<1)return cudaErrorInvalidValue;
    unsigned blocks=0;
    for(int i=0;i<batch.count;++i){const auto& j=batch.jobs[i];
        if(!j.destination||!j.source||!j.bytes||j.bytes>256*1024||j.first_block!=blocks||
           ((reinterpret_cast<uintptr_t>(j.destination)|reinterpret_cast<uintptr_t>(j.source)|j.bytes)&15))return cudaErrorInvalidValue;
        blocks+=(j.bytes+4095)/4096;
    }
    if(blocks!=unsigned(batch.blocks))return cudaErrorInvalidValue;
    state_copy_kernel<<<blocks,256,0,stream>>>(batch);return cudaGetLastError();
}
cudaError_t target_greedy_gpu(const float* logits,int* ids,int vocab,int tokens,cudaStream_t stream) {
    if(!logits||!ids||vocab<1||vocab>129280||tokens<1||tokens>8)return cudaErrorInvalidValue;
    target_greedy_kernel<<<tokens,256,0,stream>>>(logits,ids,vocab);return cudaGetLastError();
}
cudaError_t route_group_short_gpu(const int* ids,const float* weights,int* rows,int* slots,float* grouped,int tokens,cudaStream_t stream) {
    if(!ids||!weights||!rows||!slots||!grouped||tokens<2||tokens>8)return cudaErrorInvalidValue;
    route_group_short_kernel<<<1,64,0,stream>>>(ids,weights,rows,slots,grouped,tokens*6);return cudaGetLastError();
}
cudaError_t route_handoff_short_gpu(const int* ids,const float* weights,int* rows,int* slots,float* grouped,int* handoff,int tokens,cudaStream_t stream) {
    if(!ids||!weights||!rows||!slots||!grouped||!handoff||tokens<1||tokens>8)return cudaErrorInvalidValue;
    route_handoff_short_kernel<<<1,64,0,stream>>>(ids,weights,rows,slots,grouped,handoff,tokens*6);return cudaGetLastError();
}
void hc_project_vector_set_enabled(bool value){hc_vector_enabled.store(value);}
void hc_project_reuse_set_enabled(bool value){hc_reuse_enabled.store(value);}
void hc_project_tile_set_enabled(bool value){hc_tile_enabled.store(value);}
void hc_lane_set_enabled(bool value){hc_lane_enabled.store(value);}
void q5_rows8_set_enabled(bool value){q5_rows8_enabled.store(value);}
void hc_staged_set_enabled(bool value){hc_staged_enabled.store(value);}
void draft_q8_warp_set_enabled(bool value){draft_q8_warp_enabled.store(value);}
void draft_dedup_set_enabled(bool value){draft_dedup_enabled.store(value);}
void projection_prefetch_set_enabled(bool value){projection_prefetch_enabled.store(value);}
void draft_group_set_enabled(bool value){draft_group_enabled.store(value);}
bool draft_group_enabled_value(){return draft_group_enabled.load();}
cudaError_t hc_project_gpu(const float* x, const float* fn, float* mixes, int hidden, int tokens, float eps, cudaStream_t s) {
    if (hidden < 1 || tokens < 1 || tokens > 65535 || eps <= 0) return cudaErrorInvalidValue;
    if(hc_vector_enabled.load()&&hc_staged_enabled.load()&&hidden==5120&&tokens<=8&&((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(fn))&15)==0)
        hc_staged_kernel<8><<<dim3(24,tokens),256,2*2*8*256*4,s>>>(x,fn,mixes,hidden,eps);
    else if(hc_vector_enabled.load()&&hidden==5120&&tokens<=8192&&((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(fn))&15)==0){
        if(hc_tile_enabled.load()&&tokens>=128)
            hc_project_tile_kernel<4><<<(tokens+3)/4,256,0,s>>>(x,fn,mixes,hidden,eps,tokens);
        else if(hc_reuse_enabled.load()&&tokens>=128)
            hc_project_reuse_kernel<<<dim3(6,tokens),64,0,s>>>(x,fn,mixes,hidden,eps);
        else hc_project_vector_kernel<<<dim3(24,tokens),64,0,s>>>(x,fn,mixes,hidden,eps);
    }
    else hc_project_kernel<<<dim3(24,tokens),256,0,s>>>(x,fn,mixes,hidden,eps);
    return cudaGetLastError();
}
cudaError_t hc_project_split_gpu(const float* x, const float* fn, float* mixes, const float* scale, const float* base,
                                 float* pre, float* post, float* comb, int hidden, int tokens, float norm_eps, float hc_eps,
                                 int iterations, int slot, cudaStream_t s) {
    if(hc_split_fused_enabled.load()&&hc_vector_enabled.load()&&hc_staged_enabled.load()&&hidden==5120&&tokens>=1&&tokens<=8&&
       norm_eps>0&&hc_eps>0&&iterations>=1&&(slot==0||slot==1)&&((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(fn))&15)==0){
        if(hc_lane_enabled.load())
            hc_lane_split_kernel<<<dim3(24,4),tokens*64,0,s>>>(x,fn,mixes,hidden,norm_eps,tokens,
                HcSplitArgs{scale,base,pre,post,comb,hc_eps,iterations,slot});
        else hc_staged_kernel<8,true><<<dim3(24,tokens),256,2*2*8*256*4,s>>>(x,fn,mixes,hidden,norm_eps,
            HcSplitArgs{scale,base,pre,post,comb,hc_eps,iterations,slot});
        return cudaGetLastError();
    }
    const auto error=hc_project_gpu(x,fn,mixes,hidden,tokens,norm_eps,s);
    if(error!=cudaSuccess)return error;
    return hc_split_gpu(mixes,scale,base,pre,post,comb,tokens,hc_eps,iterations,s);
}
void hc_split_fused_set_enabled(bool value){hc_split_fused_enabled.store(value);}
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
// Fused production glue: [BF16 -> FP32] -> rotate the last 64 dims of each head -> BF16 round
// -> optional BF16 copy. Rotation math equals rotate_kernel (Shared=false) or
// rotary_detail::rotate_shared (Shared=true), whichever rotate_gpu selects for the shape.
template<bool Shared>
__global__ void rope_round_kernel(float* x, const uint16_t* in_bf, uint16_t* out_bf, int heads, int head_dim, int64_t start,
                                  RotaryFrequencies f, bool inverse, int64_t pairs, bool pre_round, bool keep_float) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < pairs; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t e = 2 * i;
        const int col = int(e % head_dim);
        const int64_t token = e / (int64_t(heads) * head_dim);
        float a, b;
        if (in_bf) { a = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(in_bf)[e]); b = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(in_bf)[e + 1]); }
        else { a = x[e]; b = x[e + 1]; }
        if (pre_round) { a = __bfloat162float(__float2bfloat16_rn(a)); b = __bfloat162float(__float2bfloat16_rn(b)); }
        if (col >= head_dim - 64) {
            const int pair = (col - (head_dim - 64)) / 2;
            float sn, cs;
            if constexpr (Shared) {
                const float angle = float(start + token) * f.value[pair];
                sincosf(angle, &sn, &cs); if (inverse) sn = -sn;
            } else {
                const float angle = float(start + token) * f.value[pair] * (inverse ? -1.f : 1.f);
                sincosf(angle, &sn, &cs);
            }
            const float p0 = a * cs - b * sn, p1 = a * sn + b * cs;
            a = p0; b = p1;
        }
        const __nv_bfloat16 ra = __float2bfloat16_rn(a), rb = __float2bfloat16_rn(b);
        if (keep_float) { x[e] = __bfloat162float(ra); x[e + 1] = __bfloat162float(rb); }
        if (out_bf) { reinterpret_cast<__nv_bfloat16*>(out_bf)[e] = ra; reinterpret_cast<__nv_bfloat16*>(out_bf)[e + 1] = rb; }
    }
}
// rotary_detail::rotate_shared (same per-token sine/cosine table and rotation code) followed by the BF16 round of every
// element of the token and the optional BF16 copy: the prefill shapes' rotate + round (+ convert) in one pass.
__global__ void rope_round_shared_kernel(float* x, const uint16_t* in_bf, uint16_t* out_bf, int heads, int dim, int64_t start,
                                         RotaryFrequencies f, bool inverse, bool pre_round, bool keep_float) {
    __shared__ float sine[32], cosine[32];
    const int token = blockIdx.x;
    if (threadIdx.x < 32) {
        const float angle = float(start + int64_t(token)) * f.value[threadIdx.x];
        float s, c; sincosf(angle, &s, &c);
        sine[threadIdx.x] = inverse ? -s : s; cosine[threadIdx.x] = c;
    }
    __syncthreads();
    const int64_t base = int64_t(token) * heads * dim;
    for (int i = threadIdx.x; i < heads * dim / 2; i += blockDim.x) {
        const int head = i / (dim / 2), pr = i % (dim / 2), col = 2 * pr;
        const int64_t e = base + int64_t(head) * dim + col;
        float a, b;
        if (in_bf) { a = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(in_bf)[e]); b = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(in_bf)[e + 1]); }
        else { a = x[e]; b = x[e + 1]; }
        if (pre_round) { a = __bfloat162float(__float2bfloat16_rn(a)); b = __bfloat162float(__float2bfloat16_rn(b)); }
        if (col >= dim - 64) {
            const int d = (col - (dim - 64)) / 2;
            const float s = sine[d], c = cosine[d];
            const float p0 = a * c - b * s, p1 = a * s + b * c;
            a = p0; b = p1;
        }
        const __nv_bfloat16 ra = __float2bfloat16_rn(a), rb = __float2bfloat16_rn(b);
        if (keep_float) { x[e] = __bfloat162float(ra); x[e + 1] = __bfloat162float(rb); }
        if (out_bf) { reinterpret_cast<__nv_bfloat16*>(out_bf)[e] = ra; reinterpret_cast<__nv_bfloat16*>(out_bf)[e + 1] = rb; }
    }
}
std::atomic<bool> rope_fused_enabled{true};
std::atomic<bool> rope_shared_fused_enabled{true};
void rope_shared_fused_set_enabled(bool value) { rope_shared_fused_enabled.store(value); }
bool rope_shared_fused_enabled_value() { return rope_shared_fused_enabled.load() && rope_fused_enabled.load(); }
cudaError_t rope_round_gpu(float* x, const uint16_t* in_bf, uint16_t* out_bf, int heads, int head_dim, int tokens, int64_t start,
                           const Geometry& g, bool compressed, bool inverse, cudaStream_t s, bool pre_round, bool keep_float) {
    if (!keep_float && !out_bf) return cudaErrorInvalidValue;
    if (!rope_fused_enabled.load()) return cudaErrorNotSupported;
    if (heads < 1 || g.rope_dim != 64 || head_dim < g.rope_dim || head_dim % 2 || tokens < 1 || start < 0) return cudaErrorInvalidValue;
    // rotate_shared's FMA contraction is not reproduced; those shapes keep the separate kernels.
    // rotate_shared shapes (prefill q/kv/index-q, decode kv): rope_round_shared_kernel reproduces rotate_shared bitwise.
    const bool shared = rotary_detail::shared_shape(heads, head_dim, tokens);
    if (shared && !rope_shared_fused_enabled.load()) return cudaErrorNotSupported;
    const auto frequencies = rotary_frequencies(g, compressed);
    const int64_t pairs = int64_t(tokens) * heads * head_dim / 2;
    if (shared) rope_round_shared_kernel<<<tokens, 256, 0, s>>>(x, in_bf, out_bf, heads, head_dim, start, frequencies, inverse, pre_round, keep_float);
    else rope_round_kernel<false><<<grid(pairs), 256, 0, s>>>(x, in_bf, out_bf, heads, head_dim, start, frequencies, inverse, pairs, pre_round, keep_float);
    return cudaGetLastError();
}
void rope_fused_set_enabled(bool value) { rope_fused_enabled.store(value); }
bool rope_fused_enabled_value() { return rope_fused_enabled.load(); }
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
