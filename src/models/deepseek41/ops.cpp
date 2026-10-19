// Model mathematics follow deepseek-ai/DeepSeek-V4.1-Flash, revision dba1be0 (MIT).
#include "strata/models/deepseek41/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace strata::models::deepseek41 {
namespace {
float sigmoid(float x) { return x >= 0 ? 1.f / (1.f + std::exp(-x)) : std::exp(x) / (1.f + std::exp(x)); }
}
void rms_cpu(const float* x, const float* weight, float* out, int width, float eps) {
    if (width <= 0 || eps <= 0) throw std::invalid_argument("DeepSeek RMS geometry");
    float sum = 0;
    for (int i = 0; i < width; ++i) sum += x[i] * x[i];
    const float inverse = 1.f / std::sqrt(sum / width + eps);
    for (int i = 0; i < width; ++i) out[i] = x[i] * inverse * (weight ? weight[i] : 1.f);
}
float swiglu_cpu(float gate, float up, float limit) {
    if (limit > 0) { gate = std::min(gate, limit); up = std::clamp(up, -limit, limit); }
    return (gate * sigmoid(gate)) * up;
}
void route_cpu(const float* logits, const float* bias, const float* vl, bool image,
               int experts, int top_k, int* ids, float* weights, float scale) {
    if (!((experts == 384 && top_k == 6) || (experts == 128 && top_k == 3)))
        throw std::invalid_argument("DeepSeek router shape");
    std::vector<float> scores(experts);
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    for (int e = 0; e < experts; ++e) {
        if (!std::isfinite(logits[e]) || !std::isfinite(bias[e]) || (image && vl && !std::isfinite(vl[e])))
            throw std::invalid_argument("DeepSeek nonfinite router input");
        scores[e] = std::sqrt(std::max(logits[e], 0.f) + std::log1p(std::exp(-std::abs(logits[e]))));
    }
    const float* correction = image && vl ? vl : bias;
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](int a, int b) {
        const float sa = scores[a] + correction[a], sb = scores[b] + correction[b];
        return sa == sb ? a < b : sa > sb;
    });
    float total = 0;
    for (int k = 0; k < top_k; ++k) { ids[k] = order[k]; weights[k] = scores[ids[k]]; total += weights[k]; }
    for (int k = 0; k < top_k; ++k) weights[k] = (weights[k] / (total + 1e-20f)) * scale;
}
void hc_split_cpu(const float* mixes, const float* scale, const float* base,
                  float* pre, float* post, float* comb, float eps, int iterations) {
    if (eps <= 0 || iterations < 1) throw std::invalid_argument("DeepSeek mHC parameters");
    for (int j = 0; j < 4; ++j) {
        pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps;
        post[j] = 2.f * sigmoid(mixes[j + 4] * scale[1] + base[j + 4]);
        float maximum = -std::numeric_limits<float>::infinity(), sum = 0;
        for (int k = 0; k < 4; ++k) {
            comb[j * 4 + k] = mixes[8 + j * 4 + k] * scale[2] + base[8 + j * 4 + k];
            maximum = std::max(maximum, comb[j * 4 + k]);
        }
        for (int k = 0; k < 4; ++k) { comb[j * 4 + k] = std::exp(comb[j * 4 + k] - maximum); sum += comb[j * 4 + k]; }
        for (int k = 0; k < 4; ++k) comb[j * 4 + k] = comb[j * 4 + k] / sum + eps;
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (iteration) for (int j = 0; j < 4; ++j) {
            float sum = eps;
            for (int k = 0; k < 4; ++k) sum += comb[j * 4 + k];
            for (int k = 0; k < 4; ++k) comb[j * 4 + k] /= sum;
        }
        for (int k = 0; k < 4; ++k) {
            float sum = eps;
            for (int j = 0; j < 4; ++j) sum += comb[j * 4 + k];
            for (int j = 0; j < 4; ++j) comb[j * 4 + k] /= sum;
        }
    }
}
void hc_project_cpu(const float* residual, const float* fn, float* mixes, int hidden, float eps) {
    const int width = hidden * 4;
    float sum = 0;
    for (int i = 0; i < width; ++i) sum += residual[i] * residual[i];
    const float inverse = 1.f / std::sqrt(sum / width + eps);
    for (int row = 0; row < 24; ++row) {
        float value = 0;
        for (int i = 0; i < width; ++i) value += fn[row * width + i] * residual[i];
        mixes[row] = value * inverse;
    }
}
void hc_pre_cpu(const float* residual, const float* pre, float* out, int hidden) {
    for (int i = 0; i < hidden; ++i) {
        float value = 0;
        for (int j = 0; j < 4; ++j) value += pre[j] * residual[j * hidden + i];
        out[i] = value;
    }
}
void hc_post_cpu(const float* residual, const float* branch, const float* post, const float* comb,
                 float* out, int hidden) {
    for (int i = 0; i < hidden; ++i) {
        float copies[4];
        for (int j = 0; j < 4; ++j) copies[j] = residual[j * hidden + i];
        // comb is [source,destination]; the reference sums the source axis.
        for (int destination = 0; destination < 4; ++destination) {
            float value = post[destination] * branch[i];
            for (int source = 0; source < 4; ++source) value += comb[source * 4 + destination] * copies[source];
            out[destination * hidden + i] = value;
        }
    }
}
RotaryFrequencies rotary_frequencies(const Geometry& g, bool compressed) {
    const int dim = g.rope_dim;
    if (dim != 64) throw std::invalid_argument("DeepSeek rotary dimension must be 64");
    const float theta = compressed ? g.compress_rope_theta : g.rope_theta;
    const int original = compressed ? g.original_context : 0;
    const float pi = 3.14159265358979323846f;
    const int low = original ? std::max(static_cast<int>(std::floor(dim * std::log(original / (g.beta_fast * 2. * pi)) / (2. * std::log(theta)))), 0) : 0;
    const int high = original ? std::min(static_cast<int>(std::ceil(dim * std::log(original / (g.beta_slow * 2. * pi)) / (2. * std::log(theta)))), dim - 1) : 0;
    RotaryFrequencies frequencies;
    for (int pair = 0; pair < dim / 2; ++pair) {
        float frequency = 1.f / std::pow(theta, float(pair * 2) / dim);
        if (original) {
            const float ramp = std::clamp((pair - low) / std::max(float(high - low), 1e-3f), 0.f, 1.f);
            frequency = frequency / g.rope_factor * ramp + frequency * (1.f - ramp);
        }
        frequencies.value[pair] = frequency;
    }
    return frequencies;
}
void rotate_cpu(float* x, int heads, int head_dim, int64_t position, const Geometry& g, bool compressed, bool inverse) {
    const auto frequencies = rotary_frequencies(g, compressed);
    const int dim = g.rope_dim;
    for (int h = 0; h < heads; ++h) for (int pair = 0; pair < dim / 2; ++pair) {
        const float frequency = frequencies.value[pair];
        const float angle = float(position) * frequency * (inverse ? -1.f : 1.f);
        const float c = std::cos(angle), s = std::sin(angle);
        float* p = x + h * head_dim + head_dim - dim + pair * 2;
        const float a = p[0], b = p[1];
        p[0] = a * c - b * s; p[1] = a * s + b * c;
    }
}
void pool2_cpu(const float* a, const float* b, const float* ga, const float* gb, float* out, int width) {
    for (int i = 0; i < width; ++i) {
        const float maximum = std::max(ga[i], gb[i]);
        const float x = std::exp(ga[i] - maximum), y = std::exp(gb[i] - maximum);
        out[i] = (a[i] * x + b[i] * y) / (x + y);
    }
}
} // namespace strata::models::deepseek41
