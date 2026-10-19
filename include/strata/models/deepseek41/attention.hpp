#pragma once

#include "strata/models/deepseek41/kernels.hpp"
#include <cstdint>

namespace strata::models::deepseek41 {
struct CacheLayout {
    uint8_t* data = nullptr;
    int64_t bytes = 0, page_stride = 0, row_stride = 0;
    int page_size = 0;
};
struct PackedAttention {
    const uint16_t* q = nullptr;
    CacheLayout swa, global;
    const int64_t *swa_slots = nullptr, *global_slots = nullptr;
    const float* sinks = nullptr;
    uint16_t* partial = nullptr;
    float* lse = nullptr;
    uint16_t* output = nullptr;
    int tokens = 0, heads = 64, swa_topk = 128, global_topk = 0, splits = 1;
    int64_t q_token_stride = 64 * 512, q_head_stride = 512;
    // Optional fused inverse RoPE output, [8, total_tokens, 4096], with token_offset preserving a caller's tile.
    const int64_t* positions = nullptr;
    const float* cos_sin = nullptr;
    int64_t cos_positions = 0, total_tokens = 0, token_offset = 0;
    uint16_t* grouped_output = nullptr;
};
cudaError_t pack_swa_gpu(const uint16_t* kv, const int64_t* slots, int tokens, int64_t kv_stride,
                        CacheLayout cache, cudaStream_t stream);
cudaError_t pack_global_gpu(const uint16_t* kv, const int64_t* slots, int tokens, int64_t kv_stride,
                           CacheLayout cache, cudaStream_t stream);
cudaError_t pack_index_gpu(const float* keys, uint8_t* packed, int rows, cudaStream_t stream);
cudaError_t fake_quant_index_gpu(float* queries, int rows, cudaStream_t stream);
cudaError_t packed_attention_gpu(const PackedAttention& params, cudaStream_t stream);
cudaError_t float_attention_gpu(const float* q, const float* swa, const float* global,
                               const int64_t* swa_slots, const int64_t* global_slots, const float* sinks,
                               float* out, int tokens, int heads, int swa_topk, int global_topk,
                               int64_t swa_rows, int64_t global_rows, cudaStream_t stream);
float decode_fp8_cpu(uint8_t bits);
uint8_t encode_fp8_cpu(float value);
float decode_fp4_cpu(uint8_t code);
uint8_t encode_fp4_cpu(float value);
void pack_swa_cpu(const uint16_t* kv, uint8_t* row);
void pack_global_cpu(const uint16_t* kv, uint8_t* row);
void pack_index_cpu(const float* keys, uint8_t* row);
void decode_cache_cpu(const uint8_t* row, bool global, float* values);
} // namespace strata::models::deepseek41
