#include "strata/models/deepseek41/ops.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace d = strata::models::deepseek41;
namespace c = strata::kernels::cpu;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check_expert(int type, int hidden, int ff, float limit, int tokens, bool bf16 = false, bool bf16_down = false, bool bf16_gu = false) {
    c::NativeFmt fmt;
    std::string error;
    require(c::native_fmt(type, type, hidden, ff, fmt, error), error.c_str());
    require(!fmt.bf16_gu && fmt.gu_act == ggml_get_type_traits_cpu(static_cast<ggml_type>(type))->vec_dot_type,
            "native default gate/up activation contract changed");
    if(bf16_gu){
        require(c::native_enable_bf16_gu(fmt,error),error.c_str());
        require(fmt.bf16_gu && fmt.gu_act == GGML_TYPE_BF16 && fmt.act_bytes == size_t(hidden)*2,
                "BF16 gate/up activation geometry");
    }
    fmt.swiglu_limit = limit;
    fmt.bf16_io = bf16;
    if(bf16_down)require(c::native_enable_bf16_down(fmt,error),error.c_str());
    std::mt19937 generator(41 + type);
    std::normal_distribution<float> normal(0.f, 0.15f);
    std::vector<uint8_t> blob(fmt.bytes);
    std::vector<float> row(hidden);
    auto pack = [&](uint8_t* out, int rows, int cols, size_t stride) {
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < cols; ++i) row[i] = normal(generator);
            const auto bytes = ggml_quantize_chunk(static_cast<ggml_type>(type), row.data(), out + r * stride,
                                                    0, 1, cols, nullptr);
            require(bytes == stride, "quantized expert row size");
        }
    };
    pack(blob.data(), ff, hidden, fmt.gu_row);
    pack(blob.data() + fmt.up_off, ff, hidden, fmt.gu_row);
    pack(blob.data() + fmt.down_off, hidden, ff, fmt.d_row);
    std::vector<std::vector<uint8_t>> act(tokens, std::vector<uint8_t>(fmt.act_bytes));
    std::vector<float> input(hidden), routing(tokens), expected(tokens * hidden), got(tokens * hidden);
    c::ExpertJobMulti job;
    job.blob = blob.data(); job.nt = tokens;
    job.ff_weights = limit ? routing.data() : nullptr;
    const auto* traits = ggml_get_type_traits_cpu(static_cast<ggml_type>(type));
    const auto* bfloat = ggml_get_type_traits_cpu(GGML_TYPE_BF16);
    const auto* plain_weight = ggml_get_type_traits(static_cast<ggml_type>(type));
    const auto* down_quant = ggml_get_type_traits_cpu(static_cast<ggml_type>(fmt.d_act));
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < hidden; ++i) input[i] = std::sin((i + t * 17) * 0.013f);
        c::native_quant_act(fmt, input.data(), act[t].data());
        job.nact[t] = act[t].data(); job.out[t] = got.data() + t * hidden;
        std::vector<ggml_bf16_t> reference_input(hidden);
        if(bf16_gu){
            bfloat->from_float(input.data(),reference_input.data(),hidden);
            require(std::memcmp(act[t].data(),reference_input.data(),fmt.act_bytes)==0, "BF16 gate/up input bytes");
        }
        routing[t] = 0.17f + t * 0.11f;
        std::vector<float> activation(ff);
        for (int r = 0; r < ff; ++r) {
            float gate = 0, up = 0;
            if(bf16_gu){
                // Independently decode and round packed weights; never feed BF16 input to the Q3/Q8 dot.
                std::vector<float> decoded(hidden);std::vector<ggml_bf16_t> row_bf(hidden);
                plain_weight->to_float(blob.data()+r*fmt.gu_row,decoded.data(),hidden);
                bfloat->from_float(decoded.data(),row_bf.data(),hidden);
                bfloat->vec_dot(hidden,&gate,0,row_bf.data(),0,reference_input.data(),0,1);
                plain_weight->to_float(blob.data()+fmt.up_off+r*fmt.gu_row,decoded.data(),hidden);
                bfloat->from_float(decoded.data(),row_bf.data(),hidden);
                bfloat->vec_dot(hidden,&up,0,row_bf.data(),0,reference_input.data(),0,1);
            }else{
                traits->vec_dot(hidden, &gate, 0, blob.data() + r * fmt.gu_row, 0, act[t].data(), 0, 1);
                traits->vec_dot(hidden, &up, 0, blob.data() + fmt.up_off + r * fmt.gu_row, 0, act[t].data(), 0, 1);
            }
            if (bf16) { gate = ggml_bf16_to_fp32(ggml_fp32_to_bf16(gate)); up = ggml_bf16_to_fp32(ggml_fp32_to_bf16(up)); }
            if (limit) { gate = std::min(gate, limit); up = std::clamp(up, -limit, limit); }
            activation[r] = (gate / (1.f + std::exp(-gate))) * up;
            if (limit) activation[r] *= routing[t];
            if (bf16) activation[r] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(activation[r]));
        }
        std::vector<uint8_t> quantized(fmt.h_bytes);
        down_quant->from_float(activation.data(), quantized.data(), ff);
        for (int r = 0; r < hidden; ++r) {
            if(bf16_down){
                std::vector<float> decoded(ff);std::vector<ggml_bf16_t> row_bf(ff);
                ggml_get_type_traits(static_cast<ggml_type>(type))->to_float(blob.data()+fmt.down_off+r*fmt.d_row,decoded.data(),ff);
                down_quant->from_float(decoded.data(),row_bf.data(),ff);
                down_quant->vec_dot(ff,&expected[t*hidden+r],0,row_bf.data(),0,quantized.data(),0,1);
            }else traits->vec_dot(ff, &expected[t * hidden + r], 0, blob.data() + fmt.down_off + r * fmt.d_row,
                            0, quantized.data(), 0, 1);
        }
    }
    c::ExpertPool pool(3, false, true);
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::fill(got.begin(), got.end(), NAN);
        pool.run_split_multi_native(fmt, &job, 1);
        double maximum = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            require(std::isfinite(got[i]), "CPU expert unwritten/nonfinite output");
            maximum = std::max(maximum, std::abs(double(got[i]) - expected[i]));
            require(std::abs(got[i] - expected[i]) <= 1e-5f + 1e-4f * std::abs(expected[i]), "CPU expert differs from ggml contract");
        }
        std::cout << "CPU type=" << type << " shape=" << hidden << 'x' << ff << " tokens=" << tokens
                  << " bf16_gu=" << fmt.bf16_gu << " act_bytes=" << fmt.act_bytes << " h_bytes=" << fmt.h_bytes << " max_abs=" << maximum << '\n';
    }
    const auto* plain = ggml_get_type_traits(static_cast<ggml_type>(type));
    std::vector<float> oracle(hidden), scalar(hidden);
    plain->to_float(blob.data(), oracle.data(), hidden);
    int elements = 0, bytes = 0;
    require(strata::block_geometry(type, elements, bytes), "block geometry");
    for (int b = 0; b < hidden / elements; ++b) {
        const auto* p = blob.data() + b * bytes;
        if (type == 10) strata::dequantize_q2_K(p, scalar.data() + b * elements);
        else if (type == 11) strata::dequantize_q3_K(p, scalar.data() + b * elements);
        else if (type == 39) strata::dequantize_mxfp4(p, scalar.data() + b * elements);
        else strata::dequantize_q4_0(p, scalar.data() + b * elements);
    }
    require(std::memcmp(oracle.data(), scalar.data(), hidden * sizeof(float)) == 0, "scalar dequant differs from ggml");
}
}
int main() {
    try {
        require(std::abs(d::swiglu_cpu(20, 20) - d::swiglu_cpu(10, 10)) < 1e-6f, "positive clamp");
        require(d::swiglu_cpu(-20, 20) != d::swiglu_cpu(-10, 10), "gate must not be clamped below");
        float mixes[24]{}, scale[3]{1,1,1}, base[24]{}, pre[4], post[4], comb[16];
        d::hc_split_cpu(mixes, scale, base, pre, post, comb);
        for (int i = 0; i < 4; ++i) {
            require(std::abs(pre[i] - 0.500001f) < 1e-7f && post[i] == 1.f, "neutral mHC coefficients");
            float sum = 0; for (int j = 0; j < 4; ++j) sum += comb[j * 4 + i];
            require(std::abs(sum - 1.f) < 2e-6f, "Sinkhorn column normalization");
        }
        float residual[4]{1,2,3,4}, branch[1]{0}, rotated[4], directed[16]{};
        directed[0 * 4 + 1] = 1.f;
        d::hc_post_cpu(residual, branch, post, directed, rotated, 1);
        require(rotated[1] == 1.f && rotated[0] == 0.f, "mHC source/destination orientation");
        std::vector<float> logits(384, -1000.f), bias(384, 0.f), vl(384, 0.f);
        int ids[6]; float weights[6];
        d::route_cpu(logits.data(), bias.data(), nullptr, false, 384, 6, ids, weights);
        for (int i = 0; i < 6; ++i) require(ids[i] == i && weights[i] == 0.f, "zero-score routing ties");
        logits[17] = 1000; vl[29] = 100;
        d::route_cpu(logits.data(), bias.data(), vl.data(), true, 384, 6, ids, weights);
        require(ids[0] == 29 && weights[0] == 0.f && ids[1] == 17 && weights[1] == 1.5f, "VL bias must select without scaling");
        check_expert(10, 5120, 2304, 10.f, 3);
        check_expert(11, 5120, 2304, 10.f, 3);
        check_expert(39, 5120, 2304, 10.f, 3);
        check_expert(10, 5120, 2304, 10.f, 3, true);
        check_expert(11, 5120, 2304, 10.f, 3, true);
        check_expert(10, 5120, 2304, 10.f, 3, true, true);
        check_expert(11, 5120, 2304, 10.f, 3, true, true);
        check_expert(10, 5120, 2304, 10.f, 3, true, true, true);
        check_expert(11, 5120, 2304, 10.f, 3, true, true, true);
        check_expert(11, 5120, 2304, 10.f, 1, true, true, true);
        c::NativeFmt unsupported;std::string error;
        require(c::native_fmt(2,2,2560,640,unsupported,error),error.c_str());
        const auto original_act=unsupported.gu_act;const auto original_bytes=unsupported.act_bytes;
        require(!c::native_enable_bf16_gu(unsupported,error)&&!unsupported.bf16_gu&&
                unsupported.gu_act==original_act&&unsupported.act_bytes==original_bytes,"BF16 gate/up must reject Qwen formats unchanged");
        c::NativeFmt tailed;
        require(c::native_fmt(11,11,5120,2304,tailed,error),error.c_str());tailed.tail_off=16;
        require(!c::native_enable_bf16_gu(tailed,error)&&!tailed.bf16_gu,"BF16 gate/up must reject scale tails");
        check_expert(2, 2560, 640, 0.f, 3);
        std::cout << "deepseek41_cpu_test: passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
