#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstddef>

namespace strata::models::deepseek41 {
// Experimental until both numerical and uncontested performance checks pass.
// Change only while the runtime is drained, like the existing kernel controls.
void expert_mixed_fast_set_enabled(bool enabled);
bool expert_mixed_fast_enabled();
bool expert_mixed_type(uint32_t type);
void expert_mixed_short_set_enabled(bool enabled);
bool expert_mixed_short_enabled();
// Drained benchmark/rollback control for the exact N=1 vector path.
void expert_mixed_short_vector_set_enabled(bool enabled);
bool expert_mixed_short_vector_selected(uint32_t gu,uint32_t down,int groups);
// Drained control for the experimental uncovered N=1 warp path.
void expert_mixed_short_warp_set_enabled(bool enabled);
bool expert_mixed_short_warp_selected(uint32_t gu,uint32_t down,int groups);
// Drained controls for the projection and multi-token GU experiments.
void expert_mixed_projection_warp_set_enabled(bool enabled);
bool expert_mixed_projection_warp_selected(uint32_t type,int rows,int cols,int tokens,bool pair);
void expert_mixed_short_tokens_set_enabled(bool enabled);
bool expert_mixed_short_tokens_selected(uint32_t gu,int tokens,int groups);
// Drained control for the multi-token Down experiment; N1 retains its paths.
void expert_mixed_short_down_tokens_set_enabled(bool enabled);
bool expert_mixed_short_down_tokens_selected(uint32_t type,int tokens,int groups);
struct VerifyExpertGroup;
bool expert_mixed_short_selected(uint32_t gu_type,uint32_t down_type,int tokens,
    int groups,bool uniform);
// The caller owns stable packed blobs and disjoint metadata spans for this call.
// Grouping changes launch geometry only; keep the original per-lane reduction.
cudaError_t expert_mixed_short_gpu(const VerifyExpertGroup* groups,int count,
    uint32_t gu_type,uint32_t down_type,size_t up_offset,size_t down_offset,
    const float* x,const int* token_rows,const float* routing,const int* slots,
    float* gate,float* up,float* activation,float* parts,int tokens,cudaStream_t stream);
// Adopted format/row pairs from the full-weight GU+activation comparison.
inline bool expert_mixed_gu_activation_selected(uint32_t type,int tokens) {
    return (type==22&&tokens==1) || (type==17&&(tokens==1||tokens==3||tokens==8)) ||
           (type==16&&tokens==3);
}

// Same decoded-weight BF16 rounding and FP32 reduction as the current fallback.
cudaError_t expert_mixed_gu_gpu(const void* gate_weights,const void* up_weights,uint32_t type,
    const float* x,float* gate,float* up,int rows,int cols,int tokens,cudaStream_t stream);
// Same GU reductions and BF16 boundaries; produce the routed/clipped activation
// directly, without materializing Gate or Up. Down remains a separate projection.
cudaError_t expert_mixed_gu_activation_gpu(const void* gate_weights,const void* up_weights,uint32_t type,
    const float* x,const float* routing,float* activation,int rows,int cols,int tokens,cudaStream_t stream);
cudaError_t expert_mixed_projection_gpu(const void* weights,uint32_t type,const float* x,
    float* out,int rows,int cols,int tokens,const int* slots,cudaStream_t stream);
cudaError_t expert_mixed_swiglu_gpu(const float* gate,const float* up,const float* routing,
    float* activation,int width,int tokens,cudaStream_t stream);
cudaError_t expert_mixed_down_gpu(const void* weights,uint32_t type,const float* gate,const float* up,
    const float* routing,float* activation,float* out,int rows,int cols,int tokens,
    const int* slots,cudaStream_t stream);
cudaError_t expert_mixed_dequant_gpu(const void* weights,uint32_t type,uint16_t* out,
    int64_t elements,cudaStream_t stream);
} // namespace strata::models::deepseek41
