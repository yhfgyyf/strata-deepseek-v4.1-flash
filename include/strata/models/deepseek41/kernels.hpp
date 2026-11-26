#pragma once

#include "strata/models/deepseek41/model.hpp"
#include <cuda_runtime_api.h>

namespace strata::models::deepseek41 {
// Independent test controls; change only at drained single-slot boundaries.
void scores_fast_set_enabled(bool enabled);
bool scores_fast_enabled();
void expert_dequant_fast_set_enabled(bool enabled);
bool expert_dequant_fast_enabled();
cudaError_t dequant_gpu(const void* packed, uint32_t type, float* output, int64_t elements, cudaStream_t stream);
// Decode into FP32 storage with the production BF16 rounding boundary preserved.
cudaError_t dequant_round_bf16_gpu(const void* packed, uint32_t type, float* output, int64_t elements, cudaStream_t stream);
// Draft Q8_0 projections (rows >= 4096): two rows per warp share the activation loads (bitwise equal).
void draft_q8_rows_set_enabled(bool enabled);
cudaError_t dequant_bf16_gpu(const void* packed, uint32_t type, uint16_t* output, int64_t elements, cudaStream_t stream);
cudaError_t packed_bf16_gemv_gpu(const void* packed, uint32_t type, const float* x, float* out,
                                int rows, int cols, int tokens, cudaStream_t stream);
cudaError_t packed_gemv_gpu(const void* packed, uint32_t type, const float* x, float* out,
                           int rows, int cols, int tokens, cudaStream_t stream);
// FP32 trained confidence projection over pre-norm hidden + previous-token Markov.
cudaError_t draft_confidence_gpu(const float* hidden,const float* markov,const float* weight,
                                 float* probability,float* raw,int tokens,cudaStream_t stream);
// Five draft rows share weight loads while retaining the single-row reduction order.
cudaError_t draft_projection5_gpu(const void* weights,uint32_t type,const float* x,float* out,
                                 int rows,int cols,int64_t ldx,int64_t ldy,bool round_weights,
                                 bool round_output,cudaStream_t stream);
cudaError_t draft_projection7_gpu(const void* weights,uint32_t type,const float* x,float* out,
                                 int rows,int cols,int64_t ldx,int64_t ldy,bool round_weights,
                                 bool round_output,cudaStream_t stream);
// Q2_K or MXFP4 routed experts plus a separately typed shared expert; scratch holds tokens*4*2304 floats.
cudaError_t draft_fused_moe_gpu(const void* experts,uint32_t expert_type,uint64_t projection_bytes,const void* shared_gate,
                               const void* shared_up,const void* shared_down,uint32_t shared_type,
                               const float* x,const int* ids,const float* routing,float* scratch,
                               float* out,int tokens,cudaStream_t stream,float* parts=nullptr);
// Short verification rows share decoded weights and keep the single-row reduction order.
cudaError_t packed_verify_gpu(const void* packed,uint32_t type,const float* x,float* out,
                             int rows,int cols,int tokens,int64_t ldx,int64_t ldy,cudaStream_t stream);
cudaError_t expert_gu_verify_gpu(const void* packed,uint32_t type,const float* x,float* gate,float* up,
                                int rows,int cols,int tokens,int mode,void* scratch,cudaStream_t stream);
// Host descriptors for at most 8 tokens * 6 routed experts; row indices remain on device.
struct VerifyExpertGroup { const void* packed=nullptr; int offset=0,tokens=0; };
cudaError_t expert_gu_grouped_verify_gpu(const VerifyExpertGroup* groups,int group_count,uint32_t type,
                                        const float* x,const int* token_rows,float* gate,float* up,
                                        int tokens,int mode,void* scratch,cudaStream_t stream);
// Q2_K GU2 with the same rounded projection boundary, followed by routed BF16 SwiGLU.
// Output offsets index BF16 rows; scratch contains tokens*5120/32 Q8_1 blocks.
cudaError_t expert_q2_activation_grouped_gpu(const VerifyExpertGroup* groups,int group_count,
    const float* x,const int* token_rows,const float* routing,uint16_t* activation,
    int tokens,void* scratch,cudaStream_t stream);
// Modes 1/2 fuse gate/up and their BF16 output boundary; mode 2 uses integer Q8_1 inputs.
cudaError_t expert_gu_gpu(const void* packed, uint32_t type, const float* x,float* gate,float* up,
                          int rows,int cols,int tokens,int mode,void* scratch,cudaStream_t stream);
cudaError_t rms_gpu(const float* x, const float* weight, float* out, int width, int tokens,
                    float eps, cudaStream_t stream);
cudaError_t round_bf16_gpu(float* x, int64_t elements, cudaStream_t stream);
cudaError_t float_to_bf16_gpu(const float* x, uint16_t* out, int64_t elements, cudaStream_t stream);
cudaError_t bf16_to_float_gpu(const uint16_t* x, float* out, int64_t elements, cudaStream_t stream);
cudaError_t swiglu_gpu(const float* gate, const float* up, const float* routing_weights,
                       float* out, int width, int tokens, float limit, cudaStream_t stream);
cudaError_t route_gpu(const float* logits, const float* bias, const float* vl_bias, const uint8_t* image_mask,
                      int* ids, float* weights, int experts, int top_k, int tokens, float scale, cudaStream_t stream);
struct StateCopyJob {void* destination;const void* source;unsigned bytes,first_block;};
struct StateCopyBatch {StateCopyJob jobs[96]{};int count=0,blocks=0;};
cudaError_t state_copy_gpu(const StateCopyBatch& batch,cudaStream_t stream);
// Lowest ID wins ties; -1 marks a nonfinite or empty row, checked when consumed.
cudaError_t target_greedy_gpu(const float* logits,int* ids,int vocab,int tokens,cudaStream_t stream);
cudaError_t route_group_short_gpu(const int* ids,const float* weights,int* rows,int* slots,float* grouped,int tokens,cudaStream_t stream);
// handoff holds tokens*6 IDs followed by the first invalid slot, or -1.
cudaError_t route_handoff_short_gpu(const int* ids,const float* weights,int* rows,int* slots,float* grouped,int* handoff,int tokens,cudaStream_t stream);
cudaError_t route_parallel_gpu(const float* logits, const float* bias, const float* vl_bias, const uint8_t* image_mask,
                              int* ids,float* weights,int experts,int top_k,int tokens,float scale,cudaStream_t stream);
// Change only while the runtime is drained; exact HC vector candidate.
void hc_project_vector_set_enabled(bool enabled);
void hc_project_reuse_set_enabled(bool enabled);
void hc_project_tile_set_enabled(bool enabled);
void hc_lane_set_enabled(bool enabled);
// Draft MoE routed rows staged in shared memory with vector loads (same products).
void draft_stage_set_enabled(bool enabled);
// Drained control: Q5_K projections with eight lanes per row (same per-element order and tree).
void q5_rows8_set_enabled(bool enabled);
// Drained control: register-resident RMS rows for widths 5120/1280/512 (same order and tree).
void rms_resident_set_enabled(bool enabled);
// Drained control: hc_project with cp.async-staged rows for <=8 tokens (same chains and tree).
void hc_staged_set_enabled(bool enabled);
// Drained control: warp-per-row Q8_0 draft projections for >=4096 rows (same order and tree).
void draft_q8_warp_set_enabled(bool enabled);
// Drained control: de-duplicated MXFP4 draft MoE (parts holds 7*4*5120 floats + 2048 ints; same order and tree).
void draft_dedup_set_enabled(bool enabled);
// Drained control: one launch for the draft grouped output-A projection and batched hidden copies.
void draft_group_set_enabled(bool enabled);
// Drained control: fused Markov row + markov_w2 + greedy draft step.
void draft_chain_set_enabled(bool enabled);
void draft_chain_lane_set_enabled(bool enabled);
// Drained control: register-prefetched FP32/BF16 verify/gemv projections (same order and tree).
void projection_prefetch_set_enabled(bool enabled);
// Production glue: optional BF16 input -> rotate_gpu math -> BF16 round (in place) -> optional BF16 copy.
// cudaErrorNotSupported when disabled (drained control).
void rope_shared_fused_set_enabled(bool enabled);
// pre_round: x is first rounded to BF16 (the producer's deferred round): round -> rotate -> round in one pass.
// keep_float = false: only the BF16 copy is written (x is input only; requires out_bf).
cudaError_t rope_round_gpu(float* x, const uint16_t* in_bf, uint16_t* out_bf, int heads, int head_dim, int tokens, int64_t start,
                           const Geometry& g, bool compressed, bool inverse, cudaStream_t s, bool pre_round = false,
                           bool keep_float = true);
bool rope_shared_fused_enabled_value();
void rope_fused_set_enabled(bool enabled);
bool rope_fused_enabled_value();
bool draft_group_enabled_value();
// Grouped five-row block projection (FP32/BF16, rows<4096); cudaErrorNotSupported when disabled.
cudaError_t draft_projection5_grouped_gpu(const void* weights,uint32_t type,const float* x,float* out,
                                         int rows,int cols,int64_t ldx,int64_t ldy,int groups,int64_t group_weight_bytes,
                                         int64_t group_x,int64_t group_out,bool round_weights,bool round_output,cudaStream_t stream);
cudaError_t hc_project_gpu(const float* residual, const float* fn, float* mixes, int hidden, int tokens,
                           float eps, cudaStream_t stream);
// hc_project + hc_split; for <=8 tokens the split runs in the last projection block (same values).
// slot selects one of two completion-counter sets (concurrent callers must use different slots).
cudaError_t hc_project_split_gpu(const float* x, const float* fn, float* mixes, const float* scale, const float* base,
                                 float* pre, float* post, float* comb, int hidden, int tokens, float norm_eps, float hc_eps,
                                 int iterations, int slot, cudaStream_t s);
void hc_split_fused_set_enabled(bool enabled);
cudaError_t hc_split_gpu(const float* mixes, const float* scale, const float* base, float* pre,
                         float* post, float* comb, int tokens, float eps, int iterations, cudaStream_t stream);
cudaError_t hc_pre_gpu(const float* residual, const float* pre, float* out, int hidden, int tokens, cudaStream_t stream);
cudaError_t hc_post_gpu(const float* residual, const float* branch, const float* post, const float* comb,
                        float* out, int hidden, int tokens, cudaStream_t stream);
cudaError_t rotate_gpu(float* x, int heads, int head_dim, int tokens, int64_t start_position,
                       const Geometry& g, bool compressed, bool inverse, cudaStream_t stream);
cudaError_t pool2_gpu(const float* a, const float* b, const float* ga, const float* gb,
                      float* out, int width, cudaStream_t stream);
cudaError_t engram_gate_gpu(float* residual, const float* kv, const float* q_weight,
                           const float* k_weight, const uint8_t* image, int tokens, int hidden,
                           cudaStream_t stream);
// Same FP32 arithmetic, with the existing terminal BF16 rounding fused.
// rms_bf16_gpu plus the Q8_1 blocks of its output (bitwise native_quantize_q8_1 of out); widths 5120 and 1280.
// cudaErrorNotSupported (nothing launched) otherwise.
cudaError_t rms_bf16_q8_gpu(const float* x, const float* weight, float* out, void* q8, int width, int tokens, float eps,
                            cudaStream_t stream);
cudaError_t rms_bf16_gpu(const float* x, const float* weight, float* out, int width, int tokens,
                    float eps, cudaStream_t stream);
cudaError_t hc_pre_bf16_gpu(const float* residual, const float* pre, float* out, int hidden, int tokens, cudaStream_t stream);
cudaError_t hc_post_bf16_gpu(const float* residual, const float* branch, const float* post, const float* comb,
                        float* out, int hidden, int tokens, cudaStream_t stream);
cudaError_t swiglu_bf16_gpu(const float* gate, const float* up, const float* routing_weights,
                       float* out, int width, int tokens, float limit, cudaStream_t stream);
cudaError_t engram_gate_bf16_gpu(float* residual, const float* kv, const float* q_weight,
                           const float* k_weight, const uint8_t* image, int tokens, int hidden,
                           cudaStream_t stream);
cudaError_t hc_split_warp_gpu(const float* mixes, const float* scale, const float* base, float* pre,
                         float* post, float* comb, int tokens, float eps, int iterations, cudaStream_t stream);
} // namespace strata::models::deepseek41
