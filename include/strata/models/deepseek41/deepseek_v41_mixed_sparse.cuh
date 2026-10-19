// Copyright (c) 2026 FlashInfer contributors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#include <cmath>
#include <cstdint>

#include "deepseek_v41_epilogue.cuh"

namespace strata::models::deepseek41::native {

using bf16 = __nv_bfloat16;

constexpr int kV41Dim = 512;
constexpr int kV41SmemDim = 520;
constexpr int kV41HeadTile = 16;
constexpr int kV41CandidateTile = 32;
constexpr int kV41MmaTile = 16;
constexpr int kV41Warps = 8;
constexpr int kV41Threads = kV41Warps * 32;
constexpr int kV41OutputTiles = kV41Dim / kV41MmaTile;
constexpr int kV41SwaPayloadBytes = 528;
constexpr int kV41GlobalPayloadBytes = 288;
constexpr int kV41MaxSplits = 64;
constexpr int kV41ScratchBytes = kV41Warps * 256 * sizeof(float);
constexpr int kV41ScratchFloats = kV41ScratchBytes / sizeof(float);

struct V41AttentionParams {
  const bf16* q;
  const uint8_t* swa_cache;
  const uint8_t* global_cache;
  const void* swa_slots;
  const void* global_slots;
  const float* sinks;
  bf16* partial_out;
  float* partial_lse;
  int64_t q_stride_token;
  int64_t q_stride_head;
  int64_t swa_slots_stride_token;
  int64_t global_slots_stride_token;
  int64_t partial_out_stride_token;
  int64_t partial_out_stride_split;
  int64_t partial_out_stride_head;
  int64_t partial_lse_stride_token;
  int64_t partial_lse_stride_split;
  int64_t partial_lse_stride_head;
  int64_t swa_nbytes;
  int64_t global_nbytes;
  int64_t swa_page_stride;
  int64_t swa_row_stride;
  int64_t global_page_stride;
  int64_t global_row_stride;
  int swa_page_size;
  int global_page_size;
  int swa_topk;
  int global_topk;
  int num_tokens;
  int num_heads;
  int num_splits;
  float sm_scale;
  bool has_global;
  bool unified_tiles;
  V41EpilogueParams epilogue;
};

__device__ __forceinline__ float decode_fp8_e4m3(uint8_t bits) {
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}

__device__ __forceinline__ float decode_e2m1(uint8_t code) {
  const uint32_t magnitude = code & 7;
  uint32_t magnitude_bits;
  if (magnitude == 0) {
    magnitude_bits = 0;
  } else if (magnitude == 1) {
    magnitude_bits = 0x3f000000U;
  } else {
    magnitude_bits = ((magnitude >> 1) + 126U) << 23;
    magnitude_bits |= (magnitude & 1U) << 22;
  }
  return __uint_as_float(magnitude_bits | ((code & 8U) << 28));
}

__device__ __forceinline__ uint32_t load_u32_unaligned(const uint8_t* src) {
  return static_cast<uint32_t>(src[0]) | (static_cast<uint32_t>(src[1]) << 8) |
         (static_cast<uint32_t>(src[2]) << 16) | (static_cast<uint32_t>(src[3]) << 24);
}

__device__ __forceinline__ uint4 load_u128(const uint8_t* src) {
  if ((reinterpret_cast<uintptr_t>(src) & 15U) == 0) {
    return *reinterpret_cast<const uint4*>(src);
  }
  return make_uint4(load_u32_unaligned(src), load_u32_unaligned(src + 4),
                    load_u32_unaligned(src + 8), load_u32_unaligned(src + 12));
}

__device__ __forceinline__ uint8_t extract_u8(const uint4& value, int byte) {
  const uint32_t word = byte < 4 ? value.x : byte < 8 ? value.y : byte < 12 ? value.z : value.w;
  return static_cast<uint8_t>(word >> ((byte & 3) * 8));
}

template <typename SlotT>
__device__ __forceinline__ bool resolve_v41_row(const SlotT* slots, int64_t slot_offset,
                                                int page_size, int64_t page_stride,
                                                int64_t row_stride, int64_t cache_nbytes,
                                                int payload_bytes, int64_t* row_offset) {
  const int64_t slot = static_cast<int64_t>(slots[slot_offset]);
  if (slot < 0 || page_size <= 0 || page_stride <= 0 || row_stride < payload_bytes ||
      cache_nbytes < payload_bytes) {
    *row_offset = 0;
    return false;
  }
  const int64_t page = slot / page_size;
  const int64_t row = slot % page_size;
  if (page > cache_nbytes / page_stride) {
    *row_offset = 0;
    return false;
  }
  const int64_t offset = page * page_stride + row * row_stride;
  if (offset < 0 || offset > cache_nbytes - payload_bytes) {
    *row_offset = 0;
    return false;
  }
  *row_offset = offset;
  return true;
}

template <bool Global, typename SlotT, int Rows = kV41MmaTile, int Threads = kV41Threads>
__device__ __forceinline__ void load_v41_kv_tile(
    const uint8_t* cache, const SlotT* slots, int64_t slots_stride_token, int token,
    int candidate_begin, int topk, int page_size, int64_t page_stride, int64_t row_stride,
    int64_t cache_nbytes, bf16* kv_smem, float* scale_smem, int* valid_smem, int64_t* row_smem) {
  constexpr int kPayload = Global ? kV41GlobalPayloadBytes : kV41SwaPayloadBytes;
  const int tid = threadIdx.x;
  if (tid < Rows) {
    const int candidate = candidate_begin + tid;
    bool valid = candidate < topk;
    int64_t row_offset = 0;
    if (valid) {
      valid =
          resolve_v41_row(slots, static_cast<int64_t>(token) * slots_stride_token + candidate,
                          page_size, page_stride, row_stride, cache_nbytes, kPayload, &row_offset);
    }
    valid_smem[tid] = valid;
    row_smem[tid] = row_offset;
  }
  __syncthreads();

  constexpr int kScaleGroups = Global ? 32 : 16;
  constexpr int kScaleOffset = Global ? 256 : 512;
  for (int linear = tid; linear < Rows * kScaleGroups; linear += Threads) {
    const int row = linear / kScaleGroups;
    const int group = linear % kScaleGroups;
    float scale = 0.0f;
    if (valid_smem[row]) {
      const uint8_t bits = cache[row_smem[row] + kScaleOffset + group];
      if constexpr (Global) {
        scale = decode_fp8_e4m3(bits);
      } else {
        scale = __uint_as_float(static_cast<uint32_t>(bits) << 23);
      }
    }
    scale_smem[linear] = scale;
  }
  __syncthreads();

  if constexpr (Global) {
    constexpr int kChunksPerRow = (kV41Dim / 2) / sizeof(uint4);
    for (int linear = tid; linear < Rows * kChunksPerRow; linear += Threads) {
      const int row = linear / kChunksPerRow;
      const int chunk = linear % kChunksPerRow;
      uint4 packed = make_uint4(0, 0, 0, 0);
      if (valid_smem[row]) {
        packed = load_u128(cache + row_smem[row] + chunk * sizeof(uint4));
      }
#pragma unroll
      for (int byte = 0; byte < static_cast<int>(sizeof(uint4)); ++byte) {
        const uint8_t codes = extract_u8(packed, byte);
        const float scale = scale_smem[row * kScaleGroups + chunk * 2 + byte / 8];
        const __nv_bfloat162 values = __floats2bfloat162_rn(decode_e2m1(codes & 0xF) * scale,
                                                            decode_e2m1(codes >> 4) * scale);
        *reinterpret_cast<__nv_bfloat162*>(kv_smem + row * kV41SmemDim + chunk * 32 + byte * 2) =
            values;
      }
    }
  } else {
    constexpr int kChunksPerRow = kV41Dim / sizeof(uint4);
    for (int linear = tid; linear < Rows * kChunksPerRow; linear += Threads) {
      const int row = linear / kChunksPerRow;
      const int chunk = linear % kChunksPerRow;
      uint4 packed = make_uint4(0, 0, 0, 0);
      if (valid_smem[row]) {
        packed = load_u128(cache + row_smem[row] + chunk * sizeof(uint4));
      }
      const float scale = scale_smem[row * kScaleGroups + chunk / 2];
#pragma unroll
      for (int pair = 0; pair < static_cast<int>(sizeof(uint4)) / 2; ++pair) {
        const __nv_bfloat162 values =
            __floats2bfloat162_rn(decode_fp8_e4m3(extract_u8(packed, pair * 2)) * scale,
                                  decode_fp8_e4m3(extract_u8(packed, pair * 2 + 1)) * scale);
        *reinterpret_cast<__nv_bfloat162*>(kv_smem + row * kV41SmemDim + chunk * 16 + pair * 2) =
            values;
      }
    }
  }
  __syncthreads();
}

template <bool PadQ>
__device__ __forceinline__ void compute_v41_tile(const bf16* q_tile, const bf16* kv_smem,
                                                 float* scratch, bf16* probs, const int* valid,
                                                 float* running_m, float* running_l, float* alpha,
                                                 float (&acc)[kV41OutputTiles], float sm_scale,
                                                 int valid_heads) {
  namespace wmma = nvcuda::wmma;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int head = tid / kV41MmaTile;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> q_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> k_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_frag;
  wmma::fill_fragment(score_frag, 0.0f);
#pragma unroll
  for (int k = 0; k < 64; k += 16) {
    const int k_offset = warp * 64 + k;
    wmma::load_matrix_sync(q_frag, q_tile + k_offset, PadQ ? kV41SmemDim : kV41Dim);
    wmma::load_matrix_sync(k_frag, kv_smem + k_offset, kV41SmemDim);
    wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
  }
  wmma::store_matrix_sync(scratch + warp * 256, score_frag, 16, wmma::mem_row_major);
  __syncthreads();

  float score = 0.0f;
#pragma unroll
  for (int source_warp = 0; source_warp < kV41Warps; ++source_warp) {
    score += scratch[source_warp * 256 + tid];
  }
  __syncthreads();
  scratch[tid] = score * sm_scale;
  __syncthreads();

  if (tid < kV41HeadTile) {
    if (tid < valid_heads) {
      bool has_value = false;
      float tile_max = -CUDART_INF_F;
#pragma unroll
      for (int candidate = 0; candidate < kV41MmaTile; ++candidate) {
        if (valid[candidate]) {
          has_value = true;
          tile_max = fmaxf(tile_max, scratch[tid * 16 + candidate]);
        }
      }
      if (has_value) {
        const float old_m = running_m[tid];
        const float new_m = fmaxf(old_m, tile_max);
        const float scale_old = expf(old_m - new_m);
        float tile_sum = 0.0f;
#pragma unroll
        for (int candidate = 0; candidate < kV41MmaTile; ++candidate) {
          const float probability =
              valid[candidate] ? expf(scratch[tid * 16 + candidate] - new_m) : 0.0f;
          probs[tid * 16 + candidate] = __float2bfloat16_rn(probability);
          tile_sum += probability;
        }
        running_m[tid] = new_m;
        running_l[tid] = running_l[tid] * scale_old + tile_sum;
        alpha[tid] = scale_old;
      } else {
#pragma unroll
        for (int candidate = 0; candidate < kV41MmaTile; ++candidate) {
          probs[tid * 16 + candidate] = __float2bfloat16_rn(0.0f);
        }
        alpha[tid] = 1.0f;
      }
    } else {
#pragma unroll
      for (int candidate = 0; candidate < kV41MmaTile; ++candidate) {
        probs[tid * 16 + candidate] = __float2bfloat16_rn(0.0f);
      }
      alpha[tid] = 1.0f;
    }
  }
  __syncthreads();

  using PFrag = wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major>;
  using VFrag = wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major>;
  using OFrag = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
#pragma unroll
  for (int batch = 0; batch < kV41OutputTiles / kV41Warps; ++batch) {
    const int output_tile = batch * kV41Warps + warp;
    PFrag p_frag;
    VFrag v_frag;
    OFrag out_frag;
    wmma::fill_fragment(out_frag, 0.0f);
    wmma::load_matrix_sync(p_frag, probs, 16);
    wmma::load_matrix_sync(v_frag, kv_smem + output_tile * 16, kV41SmemDim);
    wmma::mma_sync(out_frag, p_frag, v_frag, out_frag);
    wmma::store_matrix_sync(scratch + warp * 256, out_frag, 16, wmma::mem_row_major);
    __syncthreads();
#pragma unroll
    for (int tile_in_batch = 0; tile_in_batch < kV41Warps; ++tile_in_batch) {
      const int acc_tile = batch * kV41Warps + tile_in_batch;
      acc[acc_tile] = acc[acc_tile] * alpha[head] + scratch[tile_in_batch * 256 + tid];
    }
    __syncthreads();
  }
}

template <bool Global, typename SlotT, bool PadQ>
__device__ __forceinline__ void run_v41_tile(
    const bf16* q_tile, const uint8_t* cache, const SlotT* slots, int64_t slots_stride_token,
    int token, int candidate_begin, int topk, int page_size, int64_t page_stride,
    int64_t row_stride, int64_t cache_nbytes, bf16* kv_smem, float* scratch, bf16* probs,
    int* valid_smem, int64_t* row_smem, float* running_m, float* running_l, float* alpha,
    float (&acc)[kV41OutputTiles], float sm_scale, int valid_heads) {
  load_v41_kv_tile<Global>(cache, slots, slots_stride_token, token, candidate_begin, topk,
                           page_size, page_stride, row_stride, cache_nbytes, kv_smem, scratch,
                           valid_smem, row_smem);
  compute_v41_tile<PadQ>(q_tile, kv_smem, scratch, probs, valid_smem, running_m, running_l, alpha,
                         acc, sm_scale, valid_heads);
}

template <typename SwaSlotT, typename GlobalSlotT, bool PadQ, bool FuseEpilogue = false>
__global__ void __launch_bounds__(kV41Threads)
    deepseek_v41_mixed_sparse_main_kernel(V41AttentionParams params) {
  const int token = blockIdx.x;
  const int head_block = blockIdx.y;
  const int split = blockIdx.z;
  if (token >= params.num_tokens || split >= params.num_splits) return;

  const int tid = threadIdx.x;
  const int head_start = head_block * kV41HeadTile;
  const int remaining_heads = params.num_heads - head_start;
  const int valid_heads = remaining_heads > kV41HeadTile ? kV41HeadTile : remaining_heads;

  extern __shared__ __align__(16) uint8_t smem_raw[];
  bf16* q_pad = reinterpret_cast<bf16*>(smem_raw);
  bf16* kv_smem = q_pad + (PadQ ? kV41HeadTile * kV41SmemDim : 0);
  float* scratch = reinterpret_cast<float*>(kv_smem + kV41MmaTile * kV41SmemDim);
  bf16* probs = reinterpret_cast<bf16*>(scratch + kV41ScratchFloats);
  int* valid_smem = reinterpret_cast<int*>(probs + kV41HeadTile * kV41MmaTile);
  int64_t* row_smem = reinterpret_cast<int64_t*>(valid_smem + kV41MmaTile);
  float* running_m = reinterpret_cast<float*>(row_smem + kV41MmaTile);
  float* running_l = running_m + kV41HeadTile;
  float* alpha = running_l + kV41HeadTile;

  const bf16* q_token = params.q + static_cast<int64_t>(token) * params.q_stride_token +
                        static_cast<int64_t>(head_start) * params.q_stride_head;
  if constexpr (PadQ) {
    for (int linear = tid; linear < kV41HeadTile * kV41Dim; linear += kV41Threads) {
      const int head = linear / kV41Dim;
      const int dim = linear % kV41Dim;
      q_pad[head * kV41SmemDim + dim] = head < valid_heads
                                            ? q_token[head * params.q_stride_head + dim]
                                            : __float2bfloat16_rn(0.0f);
    }
    __syncthreads();
  }
  const bf16* q_tile = PadQ ? q_pad : q_token;

  if (tid < kV41HeadTile) {
    running_m[tid] = -CUDART_INF_F;
    running_l[tid] = 0.0f;
    alpha[tid] = 1.0f;
  }
  float acc[kV41OutputTiles];
#pragma unroll
  for (int tile = 0; tile < kV41OutputTiles; ++tile) acc[tile] = 0.0f;
  __syncthreads();

  const auto* swa_slots = static_cast<const SwaSlotT*>(params.swa_slots);
  const auto* global_slots = static_cast<const GlobalSlotT*>(params.global_slots);
  const int swa_tiles = (params.swa_topk + kV41CandidateTile - 1) / kV41CandidateTile;
  const int global_tiles = (params.global_topk + kV41CandidateTile - 1) / kV41CandidateTile;

  if (params.unified_tiles) {
    const int unified_swa_tiles = (params.swa_topk + kV41MmaTile - 1) / kV41MmaTile;
    const int unified_global_tiles = (params.global_topk + kV41MmaTile - 1) / kV41MmaTile;
    if (split < unified_swa_tiles) {
      const int begin = split * kV41MmaTile;
      run_v41_tile<false, SwaSlotT, PadQ>(
          q_tile, params.swa_cache, swa_slots, params.swa_slots_stride_token, token, begin,
          params.swa_topk, params.swa_page_size, params.swa_page_stride, params.swa_row_stride,
          params.swa_nbytes, kv_smem, scratch, probs, valid_smem, row_smem, running_m, running_l,
          alpha, acc, params.sm_scale, valid_heads);
    } else if (params.has_global && split < unified_swa_tiles + unified_global_tiles) {
      const int begin = (split - unified_swa_tiles) * kV41MmaTile;
      run_v41_tile<true, GlobalSlotT, PadQ>(
          q_tile, params.global_cache, global_slots, params.global_slots_stride_token, token, begin,
          params.global_topk, params.global_page_size, params.global_page_stride,
          params.global_row_stride, params.global_nbytes, kv_smem, scratch, probs, valid_smem,
          row_smem, running_m, running_l, alpha, acc, params.sm_scale, valid_heads);
    }
  } else {
    for (int tile = split; tile < swa_tiles; tile += params.num_splits) {
      const int begin = tile * kV41CandidateTile;
      run_v41_tile<false, SwaSlotT, PadQ>(
          q_tile, params.swa_cache, swa_slots, params.swa_slots_stride_token, token, begin,
          params.swa_topk, params.swa_page_size, params.swa_page_stride, params.swa_row_stride,
          params.swa_nbytes, kv_smem, scratch, probs, valid_smem, row_smem, running_m, running_l,
          alpha, acc, params.sm_scale, valid_heads);
      if (begin + kV41MmaTile < params.swa_topk) {
        run_v41_tile<false, SwaSlotT, PadQ>(
            q_tile, params.swa_cache, swa_slots, params.swa_slots_stride_token, token,
            begin + kV41MmaTile, params.swa_topk, params.swa_page_size, params.swa_page_stride,
            params.swa_row_stride, params.swa_nbytes, kv_smem, scratch, probs, valid_smem, row_smem,
            running_m, running_l, alpha, acc, params.sm_scale, valid_heads);
      }
    }
    if (params.has_global) {
      for (int tile = split; tile < global_tiles; tile += params.num_splits) {
        const int begin = tile * kV41CandidateTile;
        run_v41_tile<true, GlobalSlotT, PadQ>(
            q_tile, params.global_cache, global_slots, params.global_slots_stride_token, token,
            begin, params.global_topk, params.global_page_size, params.global_page_stride,
            params.global_row_stride, params.global_nbytes, kv_smem, scratch, probs, valid_smem,
            row_smem, running_m, running_l, alpha, acc, params.sm_scale, valid_heads);
        if (begin + kV41MmaTile < params.global_topk) {
          run_v41_tile<true, GlobalSlotT, PadQ>(
              q_tile, params.global_cache, global_slots, params.global_slots_stride_token, token,
              begin + kV41MmaTile, params.global_topk, params.global_page_size,
              params.global_page_stride, params.global_row_stride, params.global_nbytes, kv_smem,
              scratch, probs, valid_smem, row_smem, running_m, running_l, alpha, acc,
              params.sm_scale, valid_heads);
        }
      }
    }
  }

  if (tid < kV41HeadTile) {
    alpha[tid] = 1.0f;
    if (tid < valid_heads && split == 0) {
      const float sink = params.sinks[head_start + tid];
      if (sink > -CUDART_INF_F) {
        const float old_m = running_m[tid];
        const float new_m = fmaxf(old_m, sink);
        const float scale_old = expf(old_m - new_m);
        running_l[tid] = running_l[tid] * scale_old + expf(sink - new_m);
        running_m[tid] = new_m;
        alpha[tid] = scale_old;
      }
    }
  }
  __syncthreads();

  const int head = tid / kV41MmaTile;
  const int column = tid % kV41MmaTile;
#pragma unroll
  for (int tile = 0; tile < kV41OutputTiles; ++tile) {
    acc[tile] *= alpha[head];
  }
  if (head < valid_heads) {
    const int global_head = head_start + head;
    const float denominator = running_l[head];
    const int64_t out_base = static_cast<int64_t>(token) * params.partial_out_stride_token +
                             static_cast<int64_t>(split) * params.partial_out_stride_split +
                             static_cast<int64_t>(global_head) * params.partial_out_stride_head;
#pragma unroll
    for (int tile = 0; tile < kV41OutputTiles; ++tile) {
      const float value = denominator > 0.0f ? acc[tile] / denominator : 0.0f;
      if constexpr (FuseEpilogue) {
        const float peer = __shfl_xor_sync(0xffffffff, value, 1);
        if ((column & 1) == 0) {
          store_v41_inv_rope_pair(params.epilogue, token, global_head, tile * 16 + column, value,
                                  peer);
        }
      } else {
        params.partial_out[out_base + tile * 16 + column] = __float2bfloat16_rn(value);
      }
    }
    if (column == 0) {
      const int64_t lse_offset = static_cast<int64_t>(token) * params.partial_lse_stride_token +
                                 static_cast<int64_t>(split) * params.partial_lse_stride_split +
                                 static_cast<int64_t>(global_head) * params.partial_lse_stride_head;
      params.partial_lse[lse_offset] =
          denominator > 0.0f ? running_m[head] + logf(denominator) : -CUDART_INF_F;
    }
  }
}

struct V41MergeParams {
  const bf16* partial_out;
  const float* partial_lse;
  bf16* out;
  int64_t partial_out_stride_token;
  int64_t partial_out_stride_split;
  int64_t partial_out_stride_head;
  int64_t partial_lse_stride_token;
  int64_t partial_lse_stride_split;
  int64_t partial_lse_stride_head;
  int64_t out_stride_token;
  int64_t out_stride_head;
  int num_tokens;
  int num_heads;
  int num_splits;
  V41EpilogueParams epilogue;
};

template <bool FuseEpilogue = false>
__global__ void deepseek_v41_mixed_sparse_merge_kernel(V41MergeParams params) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float weights[kV41MaxSplits];
  __shared__ float denominator;

  if (tid == 0) {
    float max_lse = -CUDART_INF_F;
    for (int split = 0; split < params.num_splits; ++split) {
      const int64_t offset = static_cast<int64_t>(token) * params.partial_lse_stride_token +
                             static_cast<int64_t>(split) * params.partial_lse_stride_split +
                             static_cast<int64_t>(head) * params.partial_lse_stride_head;
      max_lse = fmaxf(max_lse, params.partial_lse[offset]);
    }
    float sum = 0.0f;
    if (max_lse > -CUDART_INF_F) {
      for (int split = 0; split < params.num_splits; ++split) {
        const int64_t offset = static_cast<int64_t>(token) * params.partial_lse_stride_token +
                               static_cast<int64_t>(split) * params.partial_lse_stride_split +
                               static_cast<int64_t>(head) * params.partial_lse_stride_head;
        weights[split] = expf(params.partial_lse[offset] - max_lse);
        sum += weights[split];
      }
    } else {
      for (int split = 0; split < params.num_splits; ++split) weights[split] = 0.0f;
    }
    denominator = sum;
  }
  __syncthreads();

  for (int dim = tid; dim < kV41Dim; dim += blockDim.x) {
    float numerator = 0.0f;
    for (int split = 0; split < params.num_splits; ++split) {
      const int64_t offset = static_cast<int64_t>(token) * params.partial_out_stride_token +
                             static_cast<int64_t>(split) * params.partial_out_stride_split +
                             static_cast<int64_t>(head) * params.partial_out_stride_head + dim;
      numerator += __bfloat162float(params.partial_out[offset]) * weights[split];
    }
    const float value = denominator > 0.0f ? numerator / denominator : 0.0f;
    if constexpr (FuseEpilogue) {
      const float peer = __shfl_xor_sync(0xffffffff, value, 1);
      if ((dim & 1) == 0) {
        store_v41_inv_rope_pair(params.epilogue, token, head, dim, value, peer);
      }
    } else {
      params.out[static_cast<int64_t>(token) * params.out_stride_token +
                 static_cast<int64_t>(head) * params.out_stride_head + dim] =
          __float2bfloat16_rn(value);
    }
  }
}

// Unlike WMMA's opaque fragment layout, m16n8k16 defines the two accumulator
// rows per lane. Keep PV in those registers across online softmax updates.
__device__ __forceinline__ void v41_mma_bf16(float* acc, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t v41_load_bf16_pair(const bf16* ptr) {
  if ((reinterpret_cast<uintptr_t>(ptr) & 3U) == 0) {
    return *reinterpret_cast<const uint32_t*>(ptr);
  }
  return static_cast<uint32_t>(__bfloat16_as_ushort(ptr[0])) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(ptr[1])) << 16);
}

__device__ __forceinline__ void v41_load_mma_a(const bf16* matrix, int64_t stride, int column,
                                               int valid_heads, uint32_t* a) {
  const int lane = threadIdx.x & 31;
  const int row = lane >> 2;
  const int col = column + (lane & 3) * 2;
  a[0] = v41_load_bf16_pair(matrix + row * stride + col);
  a[1] = row + 8 < valid_heads ? v41_load_bf16_pair(matrix + (row + 8) * stride + col) : 0;
  a[2] = v41_load_bf16_pair(matrix + row * stride + col + 8);
  a[3] = row + 8 < valid_heads ? v41_load_bf16_pair(matrix + (row + 8) * stride + col + 8) : 0;
}

__device__ __forceinline__ void v41_load_mma_v(const bf16* matrix, int row, int col, uint32_t* b) {
  const bf16* ptr = matrix + (row + (threadIdx.x & 15)) * kV41SmemDim + col;
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(address));
}

constexpr int kV41PrefillRows = 32;
constexpr int kV41PrefillWarps = kV41PrefillRows / 8;
constexpr int kV41PrefillThreads = 32 * kV41PrefillWarps;
constexpr int kV41PrefillOutputTiles = kV41Dim / (kV41PrefillWarps * 8);

struct alignas(16) V41PrefillSmem {
  bf16 kv[kV41PrefillRows * kV41SmemDim];
  alignas(16) uint8_t packed[2][8 * kV41SwaPayloadBytes];
  bf16 probs[kV41HeadTile * kV41PrefillRows];
  float maxima[kV41PrefillWarps * kV41HeadTile];
  float sums[kV41PrefillWarps * kV41HeadTile];
  int valid[kV41PrefillRows];
  int64_t rows[kV41PrefillRows];
};

template <bool Global>
__device__ __forceinline__ void v41_prefetch_stage(const V41AttentionParams& params, int stage,
                                                   V41PrefillSmem& smem) {
  constexpr int payload = Global ? kV41GlobalPayloadBytes : kV41SwaPayloadBytes;
  constexpr int chunks = payload / 16;
  const uint8_t* cache = Global ? params.global_cache : params.swa_cache;
  for (int linear = threadIdx.x; linear < 8 * chunks; linear += kV41PrefillThreads) {
    const int row = stage * 8 + linear / chunks;
    const int chunk = linear % chunks;
    uint8_t* dst = smem.packed[stage & 1] + linear * 16;
    if (smem.valid[row]) {
      const uint8_t* src = cache + smem.rows[row] + chunk * 16;
      if ((reinterpret_cast<uintptr_t>(src) & 15U) == 0) {
        const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(src)
                     : "memory");
      } else {
        *reinterpret_cast<uint4*>(dst) = load_u128(src);
      }
    } else {
      *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <bool Global, typename SlotT>
__device__ __forceinline__ void load_v41_prefill_tile(const V41AttentionParams& params, int token,
                                                      int begin, V41PrefillSmem& smem) {
  constexpr int payload = Global ? kV41GlobalPayloadBytes : kV41SwaPayloadBytes;
  constexpr int chunks = Global ? 16 : 32;
  const auto* slots = static_cast<const SlotT*>(Global ? params.global_slots : params.swa_slots);
  const int topk = Global ? params.global_topk : 128;
  const int tid = threadIdx.x;
  if (tid < kV41PrefillRows) {
    smem.rows[tid] = 0;
    smem.valid[tid] =
        begin + tid < topk &&
        resolve_v41_row(slots,
                        static_cast<int64_t>(token) * (Global ? params.global_slots_stride_token
                                                              : params.swa_slots_stride_token) +
                            begin + tid,
                        Global ? params.global_page_size : 32,
                        Global ? params.global_page_stride : params.swa_page_stride,
                        Global ? params.global_row_stride : params.swa_row_stride,
                        Global ? params.global_nbytes : params.swa_nbytes, payload,
                        &smem.rows[tid]);
  }
  __syncthreads();
  v41_prefetch_stage<Global>(params, 0, smem);
#pragma unroll
  for (int stage = 0; stage < kV41PrefillRows / 8; ++stage) {
    if (stage + 1 < kV41PrefillRows / 8) {
      v41_prefetch_stage<Global>(params, stage + 1, smem);
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();
    for (int linear = tid; linear < 8 * chunks; linear += kV41PrefillThreads) {
      const int row = linear / chunks;
      const int chunk = linear % chunks;
      const uint8_t* src = smem.packed[stage & 1] + row * payload;
      const uint4 packed = *reinterpret_cast<const uint4*>(src + chunk * 16);
      float scales[2];
      if constexpr (Global) {
        scales[0] = decode_fp8_e4m3(src[256 + chunk * 2]);
        scales[1] = decode_fp8_e4m3(src[256 + chunk * 2 + 1]);
      } else {
        scales[0] = __uint_as_float(static_cast<uint32_t>(src[512 + chunk / 2]) << 23);
      }
#pragma unroll
      for (int half = 0; half < (Global ? 4 : 2); ++half) {
        uint32_t words[4];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          float even, odd;
          if constexpr (Global) {
            const uint8_t codes = extract_u8(packed, half * 4 + pair);
            even = decode_e2m1(codes & 0xf) * scales[half / 2];
            odd = decode_e2m1(codes >> 4) * scales[half / 2];
          } else {
            const int byte = half * 8 + pair * 2;
            even = decode_fp8_e4m3(extract_u8(packed, byte)) * scales[0];
            odd = decode_fp8_e4m3(extract_u8(packed, byte + 1)) * scales[0];
          }
          const __nv_bfloat162_raw values = __floats2bfloat162_rn(even, odd);
          words[pair] = static_cast<uint32_t>(values.x) | (static_cast<uint32_t>(values.y) << 16);
        }
        *reinterpret_cast<uint4*>(smem.kv + (stage * 8 + row) * kV41SmemDim +
                                  chunk * (Global ? 32 : 16) + half * 8) =
            make_uint4(words[0], words[1], words[2], words[3]);
      }
    }
    __syncthreads();
  }
}

template <typename SwaSlotT, typename GlobalSlotT, bool HasGlobal, bool FuseEpilogue = false>
__global__ void __launch_bounds__(kV41PrefillThreads)
    deepseek_v41_prefill_kernel(V41AttentionParams params) {
  const int token = blockIdx.x;
  const int head_start = blockIdx.y * kV41HeadTile;
  const int valid_heads = min(kV41HeadTile, params.num_heads - head_start);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = lane >> 2;
  const int col = (lane & 3) * 2;
  extern __shared__ __align__(16) uint8_t storage[];
  auto& smem = *reinterpret_cast<V41PrefillSmem*>(storage);
  const bf16* q = params.q + static_cast<int64_t>(token) * params.q_stride_token +
                  static_cast<int64_t>(head_start) * params.q_stride_head;
#if __CUDA_ARCH__ == 890
  // Ada needs more registers for FP8 decoding. Retain most Q fragments while
  // leaving enough registers for the mixed-cache loader without local spills.
  constexpr int cached_query_tiles = 24;
#else
  constexpr int cached_query_tiles = kV41Dim / 16;
#endif
  uint32_t query_fragments[cached_query_tiles][4];
#pragma unroll
  for (int k = 0; k < cached_query_tiles; ++k) {
    v41_load_mma_a(q, params.q_stride_head, k * 16, valid_heads, query_fragments[k]);
  }
  float acc[kV41PrefillOutputTiles][4] = {};
  float running_m[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float running_l[2] = {};

#pragma unroll 1
  for (int tile = 0;
       tile < 128 / kV41PrefillRows +
                  (HasGlobal ? (params.global_topk + kV41PrefillRows - 1) / kV41PrefillRows : 0);
       ++tile) {
    if constexpr (HasGlobal) {
      if (tile < 128 / kV41PrefillRows) {
        load_v41_prefill_tile<false, SwaSlotT>(params, token, tile * kV41PrefillRows, smem);
      } else {
        load_v41_prefill_tile<true, GlobalSlotT>(
            params, token, (tile - 128 / kV41PrefillRows) * kV41PrefillRows, smem);
      }
    } else {
      load_v41_prefill_tile<false, SwaSlotT>(params, token, tile * kV41PrefillRows, smem);
    }

    float scores[4] = {};
#pragma unroll
    for (int k = 0; k < kV41Dim; k += 16) {
      uint32_t b[2];
      const bf16* key = smem.kv + (warp * 8 + row) * kV41SmemDim + k + col;
      b[0] = v41_load_bf16_pair(key);
      b[1] = v41_load_bf16_pair(key + 8);
      if (k / 16 < cached_query_tiles) {
        v41_mma_bf16(scores, query_fragments[k / 16], b);
      } else {
        uint32_t a[4];
        v41_load_mma_a(q, params.q_stride_head, k, valid_heads, a);
        v41_mma_bf16(scores, a, b);
      }
    }
    float alpha[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int head = row + r * 8;
      float maximum = -CUDART_INF_F;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        float& score = scores[r * 2 + c];
        score = smem.valid[warp * 8 + col + c] ? score * params.sm_scale : -CUDART_INF_F;
        maximum = fmaxf(maximum, score);
      }
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
      if ((lane & 3) == 0) smem.maxima[warp * kV41HeadTile + head] = maximum;
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int head = row + r * 8;
      float maximum = running_m[r];
#pragma unroll
      for (int w = 0; w < kV41PrefillWarps; ++w) {
        maximum = fmaxf(maximum, smem.maxima[w * kV41HeadTile + head]);
      }
      alpha[r] = maximum > -CUDART_INF_F ? expf(running_m[r] - maximum) : 1.0f;
      running_m[r] = maximum;
      float sum = 0.0f;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int candidate = warp * 8 + col + c;
        const float p = smem.valid[candidate] ? expf(scores[r * 2 + c] - maximum) : 0.0f;
        sum += p;
        smem.probs[head * kV41PrefillRows + candidate] = __float2bfloat16_rn(p);
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      sum += __shfl_xor_sync(0xffffffff, sum, 2);
      if ((lane & 3) == 0) smem.sums[warp * kV41HeadTile + head] = sum;
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kV41PrefillWarps; ++w) {
        sum += smem.sums[w * kV41HeadTile + row + r * 8];
      }
      running_l[r] = running_l[r] * alpha[r] + sum;
    }
#pragma unroll
    for (int n = 0; n < kV41PrefillOutputTiles; ++n) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[n][i] *= alpha[i / 2];
    }
#pragma unroll
    for (int k = 0; k < kV41PrefillRows; k += 16) {
      uint32_t a[4];
      v41_load_mma_a(smem.probs, kV41PrefillRows, k, kV41HeadTile, a);
#pragma unroll
      for (int n = 0; n < kV41PrefillOutputTiles; ++n) {
        uint32_t b[2];
        v41_load_mma_v(smem.kv, k, warp * kV41PrefillOutputTiles * 8 + n * 8, b);
        v41_mma_bf16(acc[n], a, b);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int head = row + r * 8;
    if (head >= valid_heads) continue;
    const float sink = params.sinks[head_start + head];
    float scale = 1.0f;
    if (sink > -CUDART_INF_F) {
      const float maximum = fmaxf(running_m[r], sink);
      scale = expf(running_m[r] - maximum);
      running_l[r] = running_l[r] * scale + expf(sink - maximum);
      running_m[r] = maximum;
    }
    const int64_t output_base = static_cast<int64_t>(token) * params.partial_out_stride_token +
                                (head_start + head) * params.partial_out_stride_head;
#pragma unroll
    for (int n = 0; n < kV41PrefillOutputTiles; ++n) {
      float values[2];
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        values[c] = running_l[r] > 0.0f ? acc[n][r * 2 + c] * scale / running_l[r] : 0.0f;
      }
      const int dim = warp * kV41PrefillOutputTiles * 8 + n * 8 + col;
      if constexpr (FuseEpilogue) {
        store_v41_inv_rope_pair(params.epilogue, token, head_start + head, dim, values[0],
                                values[1]);
      } else {
        params.partial_out[output_base + dim] = __float2bfloat16_rn(values[0]);
        params.partial_out[output_base + dim + 1] = __float2bfloat16_rn(values[1]);
      }
    }
    if (warp == 0 && (lane & 3) == 0) {
      const int64_t lse_index = static_cast<int64_t>(token) * params.partial_lse_stride_token +
                                (head_start + head) * params.partial_lse_stride_head;
      params.partial_lse[lse_index] =
          running_l[r] > 0.0f ? running_m[r] + logf(running_l[r]) : -CUDART_INF_F;
    }
  }
}

template <typename SwaSlotT, typename GlobalSlotT, bool FuseEpilogue>
inline cudaError_t launch_v41_attention_impl(const V41AttentionParams& params, bf16* output,
                                             int64_t out_stride_token, int64_t out_stride_head,
                                             cudaStream_t stream) {
  if (params.swa_topk == 128 && params.num_tokens > 32 && params.num_splits == 1) {
    if (params.has_global) {
      deepseek_v41_prefill_kernel<SwaSlotT, GlobalSlotT, true, FuseEpilogue>
          <<<dim3(params.num_tokens, (params.num_heads + 15) / 16), kV41PrefillThreads,
             sizeof(V41PrefillSmem), stream>>>(params);
    } else {
      deepseek_v41_prefill_kernel<SwaSlotT, SwaSlotT, false, FuseEpilogue>
          <<<dim3(params.num_tokens, (params.num_heads + 15) / 16), kV41PrefillThreads,
             sizeof(V41PrefillSmem), stream>>>(params);
    }
    return cudaGetLastError();
  }
  const bool pad_q = params.num_heads == 8;
  const size_t q_bytes = pad_q ? kV41HeadTile * kV41SmemDim * sizeof(bf16) : 0;
  const size_t smem_bytes = q_bytes + kV41MmaTile * kV41SmemDim * sizeof(bf16) + kV41ScratchBytes +
                            kV41HeadTile * kV41MmaTile * sizeof(bf16) + kV41MmaTile * sizeof(int) +
                            kV41MmaTile * sizeof(int64_t) + 3 * kV41HeadTile * sizeof(float);
  const dim3 grid(params.num_tokens, (params.num_heads + kV41HeadTile - 1) / kV41HeadTile,
                  params.num_splits);
  if (pad_q) {
    if (FuseEpilogue && params.num_splits == 1) {
      deepseek_v41_mixed_sparse_main_kernel<SwaSlotT, GlobalSlotT, true, true>
          <<<grid, kV41Threads, smem_bytes, stream>>>(params);
    } else {
      deepseek_v41_mixed_sparse_main_kernel<SwaSlotT, GlobalSlotT, true, false>
          <<<grid, kV41Threads, smem_bytes, stream>>>(params);
    }
  } else {
    if (FuseEpilogue && params.num_splits == 1) {
      deepseek_v41_mixed_sparse_main_kernel<SwaSlotT, GlobalSlotT, false, true>
          <<<grid, kV41Threads, smem_bytes, stream>>>(params);
    } else {
      deepseek_v41_mixed_sparse_main_kernel<SwaSlotT, GlobalSlotT, false, false>
          <<<grid, kV41Threads, smem_bytes, stream>>>(params);
    }
  }
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess || params.num_splits == 1) return status;

  V41MergeParams merge{params.partial_out,
                       params.partial_lse,
                       output,
                       params.partial_out_stride_token,
                       params.partial_out_stride_split,
                       params.partial_out_stride_head,
                       params.partial_lse_stride_token,
                       params.partial_lse_stride_split,
                       params.partial_lse_stride_head,
                       out_stride_token,
                       out_stride_head,
                       params.num_tokens,
                       params.num_heads,
                       params.num_splits,
                       params.epilogue};
  deepseek_v41_mixed_sparse_merge_kernel<FuseEpilogue>
      <<<dim3(params.num_tokens, params.num_heads), 256, 0, stream>>>(merge);
  return cudaGetLastError();
}

template <typename SwaSlotT, typename GlobalSlotT>
inline cudaError_t launch_deepseek_v41_attention(const V41AttentionParams& params, bf16* output,
                                                 int64_t out_stride_token, int64_t out_stride_head,
                                                 cudaStream_t stream) {
  if (params.epilogue.output != nullptr) {
    return launch_v41_attention_impl<SwaSlotT, GlobalSlotT, true>(params, output, out_stride_token,
                                                                  out_stride_head, stream);
  }
  return launch_v41_attention_impl<SwaSlotT, GlobalSlotT, false>(params, output, out_stride_token,
                                                                 out_stride_head, stream);
}

}  // namespace strata::models::deepseek41::native
