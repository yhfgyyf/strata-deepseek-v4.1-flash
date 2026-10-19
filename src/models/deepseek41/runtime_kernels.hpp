#pragma once
#include "strata/models/deepseek41/kernels.hpp"
#include "strata/kernels/sampler.hpp"

namespace strata::models::deepseek41 {
cudaError_t init_residual_gpu(const void* embeddings, int type, const int* ids, const float* image_values,
                             const uint8_t* image, float* residual, float* pre, int tokens, cudaStream_t stream);
cudaError_t mean_residual_gpu(const float* residual, float* mean, int tokens, cudaStream_t stream);
cudaError_t gather_rows_gpu(const float* input, const int* rows, float* output, int tokens, int width,
                          cudaStream_t stream);
cudaError_t scatter_expert_gpu(const float* input, const int* slots, float* parts, int tokens,
                             int top_k, cudaStream_t stream);
cudaError_t combine_experts_gpu(const float* parts, const int* ids, const float* shared, float* out,
                               int tokens, int top_k, cudaStream_t stream);
cudaError_t pool_groups_gpu(const float* kv, const float* gate, const float* previous_kv,
                           const float* previous_gate, float* out, int groups, bool starts_odd,
                           cudaStream_t stream);
cudaError_t rotate_strided_gpu(float* x, int heads, int head_dim, int tokens, int64_t start_position,
                              int position_step, const Geometry& g, bool compressed, bool inverse,
                              cudaStream_t stream);
cudaError_t window_slots_gpu(int64_t* slots, int tokens, int previous, cudaStream_t stream);
cudaError_t to_bf16_strided_gpu(const float* input,uint16_t* output,int tokens,int cols,int64_t stride,
                               cudaStream_t stream);
cudaError_t round_bf16_strided_gpu(float* values,int tokens,int cols,int64_t stride,cudaStream_t stream);
cudaError_t draft_markov_row_gpu(const void* weights,int type,const int* previous,const int* inverse,
                                float* out,bool bf16,int* status,cudaStream_t stream);
cudaError_t draft_penalty_counts_gpu(const int* history,int count,const int* inverse,int* counts,cudaStream_t stream);
cudaError_t draft_greedy_gpu(const float* head,const float* bias,const int* ids,const int* inverse,int vocab,
                            const int* history,int history_size,int step,int* chain,int* counts,
                            const kernels::SamplerParams& sampling,int* status,cudaStream_t stream);
} // namespace strata::models::deepseek41
