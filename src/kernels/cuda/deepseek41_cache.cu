// Copyright (c) 2026 FlashInfer contributors.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "strata/models/deepseek41/attention.hpp"
#include <limits>

namespace strata::models::deepseek41 {

constexpr int kDim = 512;
constexpr int kSwaGroup = 32;
constexpr int kGlobalGroup = 16;
constexpr int kSwaRowBytes = 528;
constexpr int kGlobalRowBytes = 288;
constexpr float kFp8Max = 448.0f;
constexpr float kFp4Max = 6.0f;

__device__ __forceinline__ float warp_absmax(float x) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    x = fmaxf(x, __shfl_down_sync(0xffffffff, x, delta));
  }
  return __shfl_sync(0xffffffff, x, 0);
}

__device__ __forceinline__ float ceil_pow2(float x) {
  uint32_t bits = __float_as_uint(x);
  if (bits & 0x007fffffU) bits = (bits + 0x00800000U) & 0x7f800000U;
  return __uint_as_float(bits);
}

__device__ __forceinline__ uint8_t fp8_bits(float x) {
  __nv_fp8_e4m3 v(x);
  return v.__x;
}

// Software E2M1 RNE is used on both Ada and Blackwell.  The alternating
// strict/non-strict boundaries implement ties-to-even for the representable
// magnitudes 0, .5, 1, 1.5, 2, 3, 4, and 6.
__device__ __forceinline__ uint8_t fp4_bits(float x) {
  const bool negative = signbit(x);
  const float a = fminf(fabsf(x), kFp4Max);
  uint8_t code;
  if (a <= 0.25f)
    code = 0;
  else if (a < 0.75f)
    code = 1;
  else if (a <= 1.25f)
    code = 2;
  else if (a < 1.75f)
    code = 3;
  else if (a <= 2.5f)
    code = 4;
  else if (a < 3.5f)
    code = 5;
  else if (a <= 5.0f)
    code = 6;
  else
    code = 7;
  return code | (negative ? 0x8 : 0);
}

template <typename SlotT>
__global__ void pack_swa_kernel(const __nv_bfloat16* __restrict__ kv,
                                const SlotT* __restrict__ slots,
                                uint8_t* __restrict__ cache, int num_tokens,
                                int64_t kv_stride, int64_t cache_nbytes,
                                int page_size, int64_t page_stride,
                                int64_t row_stride) {
  const int token = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  if (token >= num_tokens) return;
  const int64_t slot = static_cast<int64_t>(slots[token]);
  if (slot < 0 || slot / page_size > cache_nbytes / page_stride) return;
  const int64_t row_offset = (slot / page_size) * page_stride +
                             (slot % page_size) * row_stride;
  if (slot < 0 || row_offset < 0 || row_offset > cache_nbytes - kSwaRowBytes) return;

  const int d = group * kSwaGroup + lane;
  const float x = __bfloat162float(kv[token * kv_stride + d]);
  const float amax = warp_absmax(fabsf(x));
  const float scale = ceil_pow2(fmaxf(amax, 1.0e-4f) / kFp8Max);
  uint8_t* row = cache + row_offset;
  const float scaled = __fdiv_rn(x, scale);
  row[d] = fp8_bits(fmaxf(-kFp8Max, fminf(kFp8Max, scaled)));
  if (lane == 0) row[kDim + group] = (__float_as_uint(scale) >> 23) & 0xff;
}

template <typename SlotT>
__global__ void pack_global_kernel(const __nv_bfloat16* __restrict__ kv,
                                   const SlotT* __restrict__ slots,
                                   uint8_t* __restrict__ cache, int num_tokens,
                                   int64_t kv_stride, int64_t cache_nbytes,
                                   int page_size, int64_t page_stride,
                                   int64_t row_stride) {
  const int token = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  if (token >= num_tokens) return;
  const int64_t slot = static_cast<int64_t>(slots[token]);
  if (slot < 0 || slot / page_size > cache_nbytes / page_stride) return;
  const int64_t row_offset = (slot / page_size) * page_stride +
                             (slot % page_size) * row_stride;
  if (slot < 0 || row_offset < 0 || row_offset > cache_nbytes - kGlobalRowBytes) return;

  const int d = group * kGlobalGroup + lane;
  float x = lane < kGlobalGroup
                ? __bfloat162float(kv[token * kv_stride + d])
                : 0.0f;
  const float amax = warp_absmax(fabsf(x));
  const float raw_scale = fmaxf(amax, kFp4Max * 0x1p-9f) / kFp4Max;
  const uint8_t scale_bits = fp8_bits(raw_scale);
  const __nv_fp8_e4m3 scale_fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&scale_bits);
  const float scale = static_cast<float>(scale_fp8);
  uint8_t* row = cache + row_offset;
  if (lane < kGlobalGroup / 2) {
    const int d0 = group * kGlobalGroup + lane * 2;
    const float x0 = __bfloat162float(kv[token * kv_stride + d0]);
    const float x1 = __bfloat162float(kv[token * kv_stride + d0 + 1]);
    row[d0 / 2] = fp4_bits(__fdiv_rn(x0, scale)) |
                  (fp4_bits(__fdiv_rn(x1, scale)) << 4);
  }
  if (lane == 0) row[kDim / 2 + group] = scale_bits;
}

namespace {
bool cache_valid(const CacheLayout& c, int payload) {
  return c.data && c.bytes >= payload && c.page_size > 0 && c.row_stride >= payload &&
         c.row_stride <= std::numeric_limits<int64_t>::max() / c.page_size &&
         c.page_stride >= int64_t(c.page_size) * c.row_stride;
}
__global__ void pack_index_kernel(const float* keys, uint8_t* packed, float* fake) {
  const int row = blockIdx.x, group = blockIdx.y, lane = threadIdx.x;
  const int d = group * 32 + lane;
  const float x = keys[int64_t(row) * 128 + d];
  const float maximum = warp_absmax(fabsf(x));
  const float scale = ceil_pow2(fmaxf(maximum, 6.f * 0x1p-126f) / 6.f);
  const uint8_t code = fp4_bits(__fdiv_rn(x, scale));
  if (fake) {
    constexpr float values[8] = {0,.5f,1,1.5f,2,3,4,6};
    fake[int64_t(row) * 128 + d] = (code & 8 ? -values[code & 7] : values[code & 7]) * scale;
  } else {
    const uint8_t peer = __shfl_xor_sync(0xffffffff,code,1);
    if (!(lane & 1)) packed[int64_t(row) * 68 + d / 2] = code | (peer << 4);
    if (!lane) packed[int64_t(row) * 68 + 64 + group] = (__float_as_uint(scale) >> 23) & 255;
  }
}
}
template<bool GLOBAL>
__global__ void pack_rows_kernel(const __nv_bfloat16* input,const int64_t* slots,int64_t stride,CacheLayout c){
 constexpr int group=GLOBAL?16:32,groups=512/group;
 const int token=blockIdx.x,lane=threadIdx.x%group,warp=threadIdx.x/group;
 const int64_t slot=slots[token];if(slot<0)return;
 const int64_t off=slot/c.page_size*c.page_stride+(slot%c.page_size)*c.row_stride;
 if(off<0||off>c.bytes-(GLOBAL?288:528))return;uint8_t* out=c.data+off;
 for(int g=warp;g<groups;g+=256/group){
  const int d=g*group+lane;float x=__bfloat162float(input[int64_t(token)*stride+d]),m=fabsf(x);
  for(int delta=group/2;delta;delta>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,delta,group));m=__shfl_sync(0xffffffff,m,0,group);
  if constexpr(GLOBAL){
   __nv_fp8_e4m3 s(fmaxf(m,6.f*0x1p-9f)/6.f);float scale=float(s);
   uint8_t v=fp4_bits(__fdiv_rn(x,scale)),peer=__shfl_xor_sync(0xffffffff,v,1,group);
   if(!(lane&1))out[d/2]=v|(peer<<4);if(!lane)out[256+g]=s.__x;
  }else{
   float scale=ceil_pow2(fmaxf(m,1e-4f)/448.f);__nv_fp8_e4m3 v(fmaxf(-448.f,fminf(448.f,__fdiv_rn(x,scale))));out[d]=v.__x;
   if(!lane)out[512+g]=(__float_as_uint(scale)>>23)&255;
  }
 }
}

cudaError_t pack_swa_gpu(const uint16_t* kv,const int64_t* slots,int tokens,int64_t stride,
                        CacheLayout c,cudaStream_t stream) {
  if (!kv || !slots || tokens < 1 || stride < 512 || !cache_valid(c,528)) return cudaErrorInvalidValue;
  if(tokens>=128)pack_rows_kernel<false><<<tokens,256,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(kv),slots,stride,c);
  else pack_swa_kernel<int64_t><<<dim3(tokens,16),32,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(kv),
    slots,c.data,tokens,stride,c.bytes,c.page_size,c.page_stride,c.row_stride);
  return cudaGetLastError();
}
cudaError_t pack_global_gpu(const uint16_t* kv,const int64_t* slots,int tokens,int64_t stride,
                           CacheLayout c,cudaStream_t stream) {
  if (!kv || !slots || tokens < 1 || stride < 512 || !cache_valid(c,288)) return cudaErrorInvalidValue;
  if(tokens>=32)pack_rows_kernel<true><<<tokens,256,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(kv),slots,stride,c);
  else pack_global_kernel<int64_t><<<dim3(tokens,32),32,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(kv),
    slots,c.data,tokens,stride,c.bytes,c.page_size,c.page_stride,c.row_stride);
  return cudaGetLastError();
}
#include "deepseek41_index_fast.cuh"

cudaError_t pack_index_gpu(const float* keys,uint8_t* out,int rows,cudaStream_t stream) {
  if (!keys || !out || rows < 1) return cudaErrorInvalidValue;
  if(rows>=4096)index_vectorized::quant_rows<false><<<(rows+7)/8,256,0,stream>>>(keys,out,nullptr,rows);
  else pack_index_kernel<<<dim3(rows,4),32,0,stream>>>(keys,out,nullptr);
  return cudaGetLastError();
}
cudaError_t fake_quant_index_gpu(float* keys,int rows,cudaStream_t stream) {
  if (!keys || rows < 1) return cudaErrorInvalidValue;
  if(rows>=4096)index_vectorized::quant_rows<true><<<(rows+7)/8,256,0,stream>>>(keys,nullptr,keys,rows);
  else pack_index_kernel<<<dim3(rows,4),32,0,stream>>>(keys,nullptr,keys);
  return cudaGetLastError();
}
} // namespace strata::models::deepseek41

