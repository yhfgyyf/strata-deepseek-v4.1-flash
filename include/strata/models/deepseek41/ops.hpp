#pragma once

#include "strata/models/deepseek41/model.hpp"
#include <cstdint>

namespace strata::models::deepseek41 {
struct RotaryFrequencies { float value[32]{}; };
RotaryFrequencies rotary_frequencies(const Geometry& g, bool compressed);
void rms_cpu(const float* x, const float* weight, float* out, int width, float eps);
float swiglu_cpu(float gate, float up, float limit = 10.f);
void route_cpu(const float* logits, const float* bias, const float* vl_bias, bool image,
               int experts, int top_k, int* ids, float* weights, float scale = 1.5f);
void hc_split_cpu(const float* mixes, const float* scale, const float* base,
                  float* pre, float* post, float* comb, float eps = 1e-6f, int iterations = 20);
void hc_project_cpu(const float* residual, const float* fn, float* mixes, int hidden, float eps);
void hc_pre_cpu(const float* residual, const float* pre, float* out, int hidden);
void hc_post_cpu(const float* residual, const float* branch, const float* post, const float* comb,
                 float* out, int hidden);
void rotate_cpu(float* x, int heads, int head_dim, int64_t position, const Geometry& g,
                bool compressed, bool inverse = false);
void pool2_cpu(const float* kv0, const float* kv1, const float* gate0, const float* gate1,
               float* pooled, int width);
} // namespace strata::models::deepseek41
