#pragma once

#include "strata/models/deepseek41/kernels.hpp"
#include <vector>

namespace strata::models::deepseek41 {
// Scratch is 2304 bytes per query and belongs to the runtime workspace.
using IndexScoreAudit = void (*)(const float*,const float*,const uint8_t*,const uint8_t*,const int32_t*,
                                const float*,int,int,int,int64_t,int,cudaStream_t);
void index_score_set_audit(IndexScoreAudit audit);
cudaError_t index_scores_mxfp4_gpu(const float* q,const float* weights,const uint8_t* packed,const uint8_t* mask,
 const int32_t* blocks,float* scores,uint8_t* scratch,size_t scratch_bytes,int tokens,int rows,int width,
 int64_t start,int ratio,cudaStream_t stream);
struct IndexSortBuffers {
    float* sorted_scores = nullptr;
    int *input_ids = nullptr, *sorted_ids = nullptr;
    int64_t* offsets = nullptr;
    void* temporary = nullptr;
    size_t temporary_bytes = 0;
};
cudaError_t index_scores_gpu(const float* query, const float* weights, const float* keys,
                             const uint8_t* packed_keys, const uint8_t* candidate_mask, float* scores,
                             int tokens, int key_rows, int64_t start_position, int ratio, cudaStream_t stream);
// A null temporary pointer queries the CUB byte requirement, without launching work.
cudaError_t sort_index_gpu(const float* scores, int tokens, int rows, IndexSortBuffers& buffers, cudaStream_t stream);
cudaError_t index_topk_gpu(const IndexSortBuffers& sorted, int64_t* selected, int tokens, int rows,
                          int64_t start_position, int ratio, cudaStream_t stream);
cudaError_t candidate_scores_gpu(const float* scores, float* blocks, int tokens, int key_rows,
                                 int64_t start_position, int ratio, cudaStream_t stream);
cudaError_t candidate_mask_gpu(const IndexSortBuffers& sorted_blocks, uint8_t* block_mask, uint8_t* key_mask,
                               int tokens, int key_rows, int64_t start_position, int ratio, cudaStream_t stream);
// Candidate block IDs are ascending, padded with -1. This keeps stable score ties in original key order.
cudaError_t candidate_blocks_gpu(const IndexSortBuffers& sorted_blocks, int32_t* blocks, int tokens,
                                 int key_rows, cudaStream_t stream);
cudaError_t compact_index_scores_gpu(const float* query, const float* weights, const float* keys,
                                     const uint8_t* packed_keys, const int32_t* blocks, float* scores,
                                     int tokens, int key_rows, int64_t start_position, int ratio, cudaStream_t stream);
cudaError_t compact_index_topk_gpu(const IndexSortBuffers& sorted, const int32_t* blocks, int64_t* selected,
                                   int tokens, int key_rows, int64_t start_position, int ratio, cudaStream_t stream);
std::vector<uint8_t> candidate_mask_cpu(const float* scores, int rows, int visible);
std::vector<int64_t> index_topk_cpu(const float* scores, int rows, int visible, const uint8_t* candidate_mask = nullptr);
} // namespace strata::models::deepseek41
