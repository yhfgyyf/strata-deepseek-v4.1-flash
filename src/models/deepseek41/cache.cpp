#include "strata/models/deepseek41/attention.hpp"
#include "strata/artifact/dequant.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::models::deepseek41 {
float decode_fp8_cpu(uint8_t bits) {
    const int exponent = (bits >> 3) & 15, mantissa = bits & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    const float value = exponent ? std::ldexp(1.f + mantissa / 8.f, exponent - 7) : std::ldexp(float(mantissa), -9);
    return bits & 128 ? -value : value;
}
uint8_t encode_fp8_cpu(float x) {
    const uint8_t sign = std::signbit(x) ? 128 : 0;
    if (std::isnan(x)) return sign | 127;
    const float a = std::min(std::abs(x), 448.f);
    uint8_t selected = 0;
    float distance = a;
    for (int code = 1; code < 127; ++code) {
        const float error = std::abs(a - decode_fp8_cpu(code));
        if (error < distance || (error == distance && (code & 1) == 0)) { selected = code; distance = error; }
    }
    return sign | selected;
}
float decode_fp4_cpu(uint8_t code) {
    constexpr float values[8] = {0,.5f,1,1.5f,2,3,4,6};
    return code & 8 ? -values[code & 7] : values[code & 7];
}
uint8_t encode_fp4_cpu(float x) {
    const float a = std::min(std::abs(x), 6.f);
    uint8_t code;
    if (a <= .25f) code = 0; else if (a < .75f) code = 1;
    else if (a <= 1.25f) code = 2; else if (a < 1.75f) code = 3;
    else if (a <= 2.5f) code = 4; else if (a < 3.5f) code = 5;
    else if (a <= 5.f) code = 6; else code = 7;
    return code | (std::signbit(x) ? 8 : 0);
}
namespace {
float pow2(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    if (bits & 0x007fffffu) bits = (bits + 0x00800000u) & 0x7f800000u;
    std::memcpy(&value, &bits, 4);
    return value;
}
uint8_t exponent(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    return (bits >> 23) & 255;
}
}
void pack_swa_cpu(const uint16_t* kv, uint8_t* row) {
    for (int group = 0; group < 16; ++group) {
        float maximum = 0;
        for (int i = 0; i < 32; ++i) maximum = std::max(maximum, std::abs(strata::bf16_to_fp32(kv[group * 32 + i])));
        const float scale = pow2(std::max(maximum, 1e-4f) / 448.f);
        for (int i = 0; i < 32; ++i) row[group * 32 + i] = encode_fp8_cpu(strata::bf16_to_fp32(kv[group * 32 + i]) / scale);
        row[512 + group] = exponent(scale);
    }
}
void pack_global_cpu(const uint16_t* kv, uint8_t* row) {
    for (int group = 0; group < 32; ++group) {
        float maximum = 0;
        for (int i = 0; i < 16; ++i) maximum = std::max(maximum, std::abs(strata::bf16_to_fp32(kv[group * 16 + i])));
        const uint8_t bits = encode_fp8_cpu(std::max(maximum, 6.f * 0x1p-9f) / 6.f);
        const float scale = decode_fp8_cpu(bits);
        for (int i = 0; i < 8; ++i) {
            const int d = group * 16 + i * 2;
            row[d / 2] = encode_fp4_cpu(strata::bf16_to_fp32(kv[d]) / scale)
                | (encode_fp4_cpu(strata::bf16_to_fp32(kv[d + 1]) / scale) << 4);
        }
        row[256 + group] = bits;
    }
}
void pack_index_cpu(const float* keys, uint8_t* row) {
    for (int group = 0; group < 4; ++group) {
        float maximum = 0;
        for (int i = 0; i < 32; ++i) maximum = std::max(maximum, std::abs(keys[group * 32 + i]));
        const float scale = pow2(std::max(maximum, 6.f * 0x1p-126f) / 6.f);
        for (int i = 0; i < 16; ++i) {
            const int d = group * 32 + i * 2;
            row[d / 2] = encode_fp4_cpu(keys[d] / scale) | (encode_fp4_cpu(keys[d + 1] / scale) << 4);
        }
        row[64 + group] = exponent(scale);
    }
}
void decode_cache_cpu(const uint8_t* row, bool global, float* values) {
    for (int i = 0; i < 512; ++i) {
        if (global) values[i] = decode_fp4_cpu((row[i / 2] >> (i % 2 * 4)) & 15) * decode_fp8_cpu(row[256 + i / 16]);
        else values[i] = decode_fp8_cpu(row[i]) * std::ldexp(1.f, int(row[512 + i / 32]) - 127);
    }
}
} // namespace strata::models::deepseek41
