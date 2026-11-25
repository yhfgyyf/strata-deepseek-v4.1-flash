#pragma once
// Prefill routed experts on fused tensor-core kernels (src/kernels/cuda/deepseek41_fused_moe.cu). The int8 kernels
// round activations and H to int8 per 32 values; the BF16 kernel (fused_moe_set_bf16) multiplies the BF16 values of
// the cuBLAS path with identically dequantized weights, so it differs from that path only in accumulation order.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strata::models::deepseek41 {

constexpr int kFusedMoeBatch = 128;
/// Experts [e0, e1) of one layer; blob[e - e0] is expert e's packed blob on the device, or null to skip it.
struct FusedMoeBatch {
    int e0 = 0, e1 = 0;
    const uint8_t* blob[kFusedMoeBatch] = {};
};
/// Gate/up and down formats covered (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS) on this device.
bool fused_moe_supported(uint32_t gu_type, uint32_t d_type);
/// Bytes of the int8 activations of `rows` tokens (5120 values each) and of H of `rows` routed rows (2304 each).
size_t fused_moe_act_bytes(int64_t rows);
size_t fused_moe_h_bytes(int64_t rows);
/// Force 64- or 128-row work items (0: by rows per expert).
void fused_moe_set_tile(int rows);
/// The int8 kernels without per-MMA branches, with deferred raw combines and dynamic items (byte-identical).
void fused_moe_set_int8_v2(bool enabled);
cudaError_t fused_moe_quantize(const float* x, int tokens, void* xa, cudaStream_t stream);
/// Routed-row offsets per expert (experts + 1 entries, ascending): the work tables of the next runs.
cudaError_t fused_moe_tables(const int* offsets, int experts, cudaStream_t stream);
/// Gate/up + SwiGLU + down of the batch. src[row]: token of routed row, rw[row]: its route weight, dst[row]: its
/// per-slot output row (token * 6 + k) in `parts` [.][5120]. `ha`: fused_moe_h_bytes(rows) scratch.
cudaError_t fused_moe_run(const FusedMoeBatch& batch, uint32_t gu_type, uint32_t d_type, size_t up_offset, size_t down_offset,
                          int64_t rows, const void* xa, const int32_t* src, const float* rw, const int32_t* dst, void* ha,
                          float* parts, cudaStream_t stream);
void fused_moe_set_enabled(bool enabled);
/// BF16 tensor cores with the reference BF16 dequant (activations and H in BF16) instead of int8.
void fused_moe_set_bf16(bool enabled);
bool fused_moe_bf16();
/// Smallest prefill batch (tokens) that takes the fused path.
void fused_moe_set_min_tokens(int tokens);
int fused_moe_min_tokens();
/// Streamed experts before resident ones (default) or after.
void fused_moe_set_cold_first(bool cold_first);
bool fused_moe_enabled();

}  // namespace strata::models::deepseek41
