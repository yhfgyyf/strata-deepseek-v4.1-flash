#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include "strata/models/deepseek41/kernels.hpp"

namespace strata::models::deepseek41 {
// Each host descriptor points to Down weights, with a disjoint row range in
// gate/up/routing and scratch. Actual model shape only; offsets are bounded by 48.
// prepared_q2 skips preparation only when scratch already holds BF16 activations.
cudaError_t format_down_grouped_gpu(const VerifyExpertGroup* groups,int group_count,uint32_t type,
    const float* gate,const float* up,const float* routing,float* output,const int* slots,
    void* scratch,size_t scratch_bytes,bool prepared_q2,cudaStream_t stream);
// Small expert groups, 1..8 tokens. Gate/up already follow the BF16 projection boundary.
// Fuses clamped SwiGLU, routing scale, BF16 boundary and activation conversion.
// Q2_K dequantizes weights to BF16 inside the dot product; Q3_K uses native Q8_1 MMVQ.
// Final BF16 rounding and the scatter to output_slots are part of this operation.
// activation_scratch: tokens*cols*2 (Q2_K), tokens*cols/32*36 (Q3_K).
// Verification variant shares Q2 weights and uses exact native multi-column Q3 MMVQ.
cudaError_t format_down_verify_gpu(uint32_t type,const void* packed,const float* gate,const float* up,
                                  const float* routing_weights,float* temporary_output,float* output,
                                  const int* output_slots,int rows,int cols,int tokens,
                                  void* activation_scratch,size_t scratch_bytes,cudaStream_t stream);
cudaError_t format_down_gpu(uint32_t type,const void* packed,const float* gate,const float* up,
                           const float* routing_weights,float* temporary_output,float* output,
                           const int* output_slots,int rows,int cols,int tokens,
                           void* activation_scratch,size_t scratch_bytes,cudaStream_t stream);
}
