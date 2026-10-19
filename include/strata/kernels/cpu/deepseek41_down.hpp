#pragma once
#include "strata/kernels/cpu/native_expert.hpp"

namespace strata::kernels::cpu {
// Configure before constructing a DeepSeek runtime; existing runtimes keep their CPU formats.
void deepseek41_down_fusion_set_enabled(bool enabled);
bool deepseek41_down_fusion_enabled();
bool native_enable_q3_q8_1_down(NativeFmt& format, std::string& error);
void q3_q8_1_quantize(const float* activation, void* output, int elements);
void q3_q8_1_down_rows(const NativeFmt& format, const uint8_t* blob,
                      const void* const* activations, int tokens, float* const* output,
                      int first_row, int end_row);
}
