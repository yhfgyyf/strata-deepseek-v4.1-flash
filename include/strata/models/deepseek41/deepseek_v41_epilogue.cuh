// Copyright (c) 2026 FlashInfer contributors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::models::deepseek41::native {

using bf16 = __nv_bfloat16;

struct V41EpilogueParams {
  const int64_t* positions = nullptr;
  const float* cos_sin = nullptr;
  bf16* output = nullptr;
  int64_t cos_stride = 0;
  int64_t num_positions = 0;
  int64_t total_tokens = 0;
  int64_t token_offset = 0;
  int heads_per_group = 0;
};

__device__ __forceinline__ void store_v41_inv_rope_pair(const V41EpilogueParams& epilogue,
                                                        int64_t local_token, int head, int even_dim,
                                                        float even_value, float odd_value) {
  if (epilogue.output == nullptr) return;

  float even = __bfloat162float(__float2bfloat16_rn(even_value));
  float odd = __bfloat162float(__float2bfloat16_rn(odd_value));
  if (even_dim >= 448) {
    const int64_t position = epilogue.positions[local_token];
    if (position >= 0 && position < epilogue.num_positions) {
      const float* cos_sin = epilogue.cos_sin + position * epilogue.cos_stride;
      const int pair = (even_dim - 448) / 2;
      const float cosine = cos_sin[pair];
      const float sine = cos_sin[32 + pair];
      const float even_cos = __fmul_rn(even, cosine);
      const float odd_sin = __fmul_rn(odd, sine);
      const float odd_cos = __fmul_rn(odd, cosine);
      const float even_sin = __fmul_rn(even, sine);
      even = __fadd_rn(even_cos, odd_sin);
      odd = __fsub_rn(odd_cos, even_sin);
    }
  }

  const int group = head / epilogue.heads_per_group;
  const int head_in_group = head % epilogue.heads_per_group;
  const int64_t output_token = epilogue.token_offset + local_token;
  const int64_t output_offset =
      ((static_cast<int64_t>(group) * epilogue.total_tokens + output_token) *
           epilogue.heads_per_group +
       head_in_group) *
          512 +
      even_dim;
  epilogue.output[output_offset] = __float2bfloat16_rn(even);
  epilogue.output[output_offset + 1] = __float2bfloat16_rn(odd);
}

}  // namespace strata::models::deepseek41::native
