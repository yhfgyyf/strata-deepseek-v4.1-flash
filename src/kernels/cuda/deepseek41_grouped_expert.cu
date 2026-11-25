// Grouped resident routed-expert decode: per layer one GU+SwiGLU launch and one Down
// launch for each distinct row count. Bit-identical to expert_mixed_short_gpu: virtual
// lane v (0..255) owns columns v, v+256, ... in order; decoded weights are
// bf16_rn(d*grid*sign); FP32 FMA accumulation; the short_sum_warp reduction tree; and the
// same BF16 boundaries for Gate, Up, the routed activation and Down.
#include "strata/models/deepseek41/mixed_expert.hpp"
#include "strata/models/deepseek41/kernels.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "deepseek41_iq2_levels.cuh"

namespace strata::models::deepseek41 {
namespace {
std::atomic<bool> grouped_enabled{true};
std::atomic<bool> grouped_bf16_inputs{true};
constexpr int kMaxGroups = 48;
struct Desc { const uint8_t* packed; int offset, tokens; };
struct Work { Desc g[kMaxGroups]; int first[kMaxGroups + 1]; int count; };

template<int Type> struct Fmt;
template<> struct Fmt<16> { static constexpr int bytes = sizeof(block_iq2_xxs); };
template<> struct Fmt<17> { static constexpr int bytes = sizeof(block_iq2_xs); };
template<> struct Fmt<22> { static constexpr int bytes = sizeof(block_iq2_s); };
template<> struct Fmt<18> { static constexpr int bytes = sizeof(block_iq3_xxs); };

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// IQ2: the three BF16 levels are rounded once per eight weights; a PRMT then selects
// each weight pair. (+-1)*bf16(d*g) == bf16((d*g)*(+-1)) because RN is sign-symmetric.
template<int Type>
__device__ __forceinline__ void decode_iq2(float d, uint32_t sign, uint64_t sel, float (&out)[8]) {
    const __nv_bfloat162 l01 = __floats2bfloat162_rn(d * 8.f, d * 25.f);
    const __nv_bfloat162 l2 = __floats2bfloat162_rn(d * 43.f, 0.f);
    const uint32_t a = *reinterpret_cast<const uint32_t*>(&l01), b = *reinterpret_cast<const uint32_t*>(&l2);
    const uint32_t lo = uint32_t(sel), hi = uint32_t(sel >> 32);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t u = __byte_perm(a, b, (p < 2 ? lo : hi) >> (16 * (p & 1)));
        const float w0 = __uint_as_float(u << 16), w1 = __uint_as_float(u & 0xffff0000u);
        out[2 * p] = (sign >> (2 * p)) & 1 ? -w0 : w0;
        out[2 * p + 1] = (sign >> (2 * p + 1)) & 1 ? -w1 : w1;
    }
}

// Eight weights 32*ib+8*il.. of one 256-value block; same values as short_weight8<Type>.
template<int Type>
__device__ __forceinline__ void decode8(const uint8_t* blk, int ib, int il, float (&out)[8]) {
    if constexpr (Type == 16) {
        const auto& b = *reinterpret_cast<const block_iq2_xxs*>(blk);
        const uint16_t* q = b.qs + 4 * ib;
        const uint32_t aux = uint32_t(q[2]) | (uint32_t(q[3]) << 16);
        const float d = float(b.d) * (0.5f + (aux >> 28)) * 0.25f;
        decode_iq2<Type>(d, ksigns_iq2xs[(aux >> (7 * il)) & 127],
                         iq2_levels::iq2xxs_selectors[reinterpret_cast<const uint8_t*>(q)[il]], out);
    } else if constexpr (Type == 17) {
        const auto& b = *reinterpret_cast<const block_iq2_xs*>(blk);
        const uint16_t q = b.qs[4 * ib + il];
        const float d = float(b.d) * (0.5f + ((b.scales[ib] >> (4 * (il / 2))) & 15)) * 0.25f;
        decode_iq2<Type>(d, ksigns_iq2xs[q >> 9], iq2_levels::iq2xs_selectors[q & 511], out);
    } else if constexpr (Type == 22) {
        const auto& b = *reinterpret_cast<const block_iq2_s*>(blk);
        const int q = b.qs[4 * ib + il] | ((b.qh[ib] << (8 - 2 * il)) & 0x300);
        const float d = float(b.d) * (0.5f + ((b.scales[ib] >> (4 * (il / 2))) & 15)) * 0.25f;
        decode_iq2<Type>(d, b.qs[32 + 4 * ib + il], iq2_levels::iq2s_selectors[q], out);
    } else {
        static_assert(Type == 18);
        const auto& b = *reinterpret_cast<const block_iq3_xxs*>(blk);
        const auto* gas = reinterpret_cast<const uint16_t*>(b.qs + 64) + 2 * ib;
        const uint32_t aux = uint32_t(gas[0]) | (uint32_t(gas[1]) << 16);
        const float d = float(b.d) * (0.5f + (aux >> 28)) * 0.5f;
        const uint32_t sign = ksigns_iq2xs[(aux >> (7 * il)) & 127];
        const uint32_t lo = iq3xxs_grid[b.qs[8 * ib + 2 * il]], hi = iq3xxs_grid[b.qs[8 * ib + 2 * il + 1]];
        // float(byte) exactly as (2^23 + byte) - 2^23, avoiding the conversion pipe.
        const float nd = -d;
        float p[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float g = __uint_as_float(__byte_perm(j < 4 ? lo : hi, 0x4B000000u, 0x7440u | uint32_t(j & 3))) - 8388608.f;
            p[j] = ((sign >> j) & 1 ? nd : d) * g;
        }
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const __nv_bfloat162 h = __floats2bfloat162_rn(p[j], p[j + 1]);
            const uint32_t u = *reinterpret_cast<const uint32_t*>(&h);
            out[j] = __uint_as_float(u << 16);
            out[j + 1] = __uint_as_float(u & 0xffff0000u);
        }
    }
}

// Identical to short_sum_warp in deepseek41_mixed_expert.cu.
template<int Count>
__device__ __forceinline__ void sum_warp(float (&values)[Count][8], float (&sums)[Count]) {
    const int lane = threadIdx.x & 31;
    for (int delta = 2; delta; delta >>= 1) {
#pragma unroll
        for (int c = 0; c < Count; ++c) {
#pragma unroll
            for (int j = 0; j < 8; ++j) values[c][j] += __shfl_down_sync(0xffffffff, values[c][j], delta, 4);
        }
    }
#pragma unroll
    for (int c = 0; c < Count; ++c) {
        const float partial = ((values[c][0] + values[c][4]) + (values[c][2] + values[c][6])) +
                              ((values[c][1] + values[c][5]) + (values[c][3] + values[c][7]));
        const float gathered = __shfl_sync(0xffffffff, partial, (lane & 7) * 4);
        sums[c] = lane < 8 ? gathered : 0.f;
    }
    for (int delta = 16; delta; delta >>= 1) {
#pragma unroll
        for (int c = 0; c < Count; ++c) sums[c] += __shfl_down_sync(0xffffffff, sums[c], delta);
    }
}

__device__ __forceinline__ int find_group(const Work& w) {
    int g = 0;
    while (g + 1 < w.count && w.first[g + 1] <= int(blockIdx.x)) ++g;
    return g;
}

// Eight BF16 values widened exactly to FP32.
__device__ __forceinline__ void load8b(const uint16_t* p, float (&v)[8]) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) { v[2 * k] = __uint_as_float(w[k] << 16); v[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u); }
}
// The routed-expert input is the RMS output with a BF16 boundary, so this copy is exact;
// it halves the activation bytes every GU warp re-reads.
__global__ void to_bf16_kernel4(const float* x, uint16_t* y, int n) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const float4 v = *reinterpret_cast<const float4*>(x + i);
    const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    *reinterpret_cast<uint2*>(y + i) = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}
__device__ __forceinline__ void load8(const float* p, float (&v)[8]) {
    const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

// Gate/Up rows for NT tokens [t0,t0+NT) of one group, then the routed, clipped SwiGLU.
template<int Type, int NT, bool B>
__device__ __forceinline__ void gu_row(const Desc& g, size_t up_offset, int row, int t0, const void* x,
                                       const int* token_rows, const float* routing, void* activation) {
    const int lane = threadIdx.x & 31, ib = lane >> 2, il = lane & 3;
    int64_t xin[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) xin[t] = int64_t(token_rows[g.offset + t0 + t]) * 5120 + lane * 8;
    float values[2 * NT][8]{};
    constexpr int RB = 20 * Fmt<Type>::bytes;
    const uint8_t* gb = g.packed + size_t(row) * RB;
    const uint8_t* ub = g.packed + up_offset + size_t(row) * RB;
#pragma unroll
    for (int i = 0; i < 20; ++i, gb += Fmt<Type>::bytes, ub += Fmt<Type>::bytes) {
        float wg[8], wu[8];
        decode8<Type>(gb, ib, il, wg);
        decode8<Type>(ub, ib, il, wu);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v[8];
            if constexpr (B) load8b(static_cast<const uint16_t*>(x) + xin[t] + i * 256, v);
            else load8(static_cast<const float*>(x) + xin[t] + i * 256, v);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                values[t][j] += wg[j] * v[j];
                values[NT + t][j] += wu[j] * v[j];
            }
        }
    }
    float sums[2 * NT];
    sum_warp(values, sums);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = fminf(bf16r(sums[t]), 10.f);
            const float u = fminf(fmaxf(bf16r(sums[NT + t]), -10.f), 10.f);
            const float sig = v >= 0 ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v));
            const int slot = g.offset + t0 + t;
            const float a = bf16r(v * sig * u * routing[slot]);
            if constexpr (B) static_cast<uint16_t*>(activation)[int64_t(slot) * 2304 + row] = uint16_t(__float_as_uint(a) >> 16);
            else static_cast<float*>(activation)[int64_t(slot) * 2304 + row] = a;
        }
    }
}

template<int Type, int N, bool B>
__global__ void __launch_bounds__(128) grouped_gu_kernel(Work w, size_t up_offset, const void* x, const int* token_rows,
                                                         const float* routing, void* activation) {
    const int gi = find_group(w);
    const Desc g = w.g[gi];
    const int row = (blockIdx.x - w.first[gi]) * 4 + (threadIdx.x >> 5);
    // Up to four token columns per pass keep the accumulators in registers.
    if constexpr (N <= 4) gu_row<Type, N, B>(g, up_offset, row, 0, x, token_rows, routing, activation);
    else {
        constexpr int A = (N + 1) / 2;
        gu_row<Type, A, B>(g, up_offset, row, 0, x, token_rows, routing, activation);
        gu_row<Type, N - A, B>(g, up_offset, row, A, x, token_rows, routing, activation);
    }
}

template<int Type, int N, bool B>
__global__ void __launch_bounds__(128) grouped_down_kernel(Work w, size_t down_offset, const void* activation,
                                                           const int* slots, float* parts) {
    const int gi = find_group(w);
    const Desc g = w.g[gi];
    const int row = (blockIdx.x - w.first[gi]) * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, ib = lane >> 2, il = lane & 3;
    float values[N][8]{};
    const uint8_t* db = g.packed + down_offset + size_t(row) * 9 * Fmt<Type>::bytes;
    const int64_t in = int64_t(g.offset) * 2304 + lane * 8;
#pragma unroll
    for (int i = 0; i < 9; ++i, db += Fmt<Type>::bytes) {
        float wd[8];
        decode8<Type>(db, ib, il, wd);
#pragma unroll
        for (int t = 0; t < N; ++t) {
            float v[8];
            if constexpr (B) load8b(static_cast<const uint16_t*>(activation) + in + t * 2304 + i * 256, v);
            else load8(static_cast<const float*>(activation) + in + t * 2304 + i * 256, v);
#pragma unroll
            for (int j = 0; j < 8; ++j) values[t][j] += wd[j] * v[j];
        }
    }
    float sums[N];
    sum_warp(values, sums);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < N; ++t) parts[int64_t(slots[g.offset + t]) * 5120 + row] = bf16r(sums[t]);
    }
}

// Tensor-core variant (opt-in): the same BF16 weights (decode8) and BF16 activations, FP32
// accumulation by mma.sync m16n8k16 (summation order differs from the FMA path). Every group,
// whatever its token count, uses this arithmetic. A warp owns 16 rows; lane (g,c) supplies
// rows g and g+8 and, per 256-column block, the 64 columns 64c..64c+63 (k is permuted
// consistently between A and B); B column n = g is token n of the group (zero if n >= tokens).
std::atomic<bool> grouped_mma{false};
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// decode8 as four BF16 pairs (low half = lower column). Same bits as packing decode8's FP32
// outputs: negating a BF16-exact value only flips its sign bit.
template<int Type>
__device__ __forceinline__ void decode8_pairs(const uint8_t* blk, int ib, int il, uint32_t (&out)[4]) {
    if constexpr (Type == 18) {
        const auto& b = *reinterpret_cast<const block_iq3_xxs*>(blk);
        const auto* gas = reinterpret_cast<const uint16_t*>(b.qs + 64) + 2 * ib;
        const uint32_t aux = uint32_t(gas[0]) | (uint32_t(gas[1]) << 16);
        const float d = float(b.d) * (0.5f + (aux >> 28)) * 0.5f;
        const uint32_t sign = ksigns_iq2xs[(aux >> (7 * il)) & 127];
        const uint32_t lo = iq3xxs_grid[b.qs[8 * ib + 2 * il]], hi = iq3xxs_grid[b.qs[8 * ib + 2 * il + 1]];
        const float nd = -d;
        float p[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float g = __uint_as_float(__byte_perm(j < 4 ? lo : hi, 0x4B000000u, 0x7440u | uint32_t(j & 3))) - 8388608.f;
            p[j] = ((sign >> j) & 1 ? nd : d) * g;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat162 h = __floats2bfloat162_rn(p[2 * j], p[2 * j + 1]);
            out[j] = *reinterpret_cast<const uint32_t*>(&h);
        }
    } else {
        float d;uint32_t sign;uint64_t sel;
        if constexpr (Type == 16) {
            const auto& b = *reinterpret_cast<const block_iq2_xxs*>(blk);
            const uint16_t* q = b.qs + 4 * ib;
            const uint32_t aux = uint32_t(q[2]) | (uint32_t(q[3]) << 16);
            d = float(b.d) * (0.5f + (aux >> 28)) * 0.25f;
            sign = ksigns_iq2xs[(aux >> (7 * il)) & 127];
            sel = iq2_levels::iq2xxs_selectors[reinterpret_cast<const uint8_t*>(q)[il]];
        } else if constexpr (Type == 17) {
            const auto& b = *reinterpret_cast<const block_iq2_xs*>(blk);
            const uint16_t q = b.qs[4 * ib + il];
            d = float(b.d) * (0.5f + ((b.scales[ib] >> (4 * (il / 2))) & 15)) * 0.25f;
            sign = ksigns_iq2xs[q >> 9];
            sel = iq2_levels::iq2xs_selectors[q & 511];
        } else {
            static_assert(Type == 22);
            const auto& b = *reinterpret_cast<const block_iq2_s*>(blk);
            const int q = b.qs[4 * ib + il] | ((b.qh[ib] << (8 - 2 * il)) & 0x300);
            d = float(b.d) * (0.5f + ((b.scales[ib] >> (4 * (il / 2))) & 15)) * 0.25f;
            sign = b.qs[32 + 4 * ib + il];
            sel = iq2_levels::iq2s_selectors[q];
        }
        const __nv_bfloat162 l01 = __floats2bfloat162_rn(d * 8.f, d * 25.f);
        const __nv_bfloat162 l2 = __floats2bfloat162_rn(d * 43.f, 0.f);
        const uint32_t a = *reinterpret_cast<const uint32_t*>(&l01), bb = *reinterpret_cast<const uint32_t*>(&l2);
        const uint32_t lo = uint32_t(sel), hi = uint32_t(sel >> 32);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t u = __byte_perm(a, bb, (p < 2 ? lo : hi) >> (16 * (p & 1)));
            out[p] = u ^ (((sign >> (2 * p)) & 1) << 15) ^ (((sign >> (2 * p + 1)) & 1) << 31);
        }
    }
}
// Register-fed IQ2_XXS / IQ2_XS decode: each lane loads its 16 qs bytes (qs[8c..8c+7]) of a
// superblock with aligned 32-bit words, and the sign byte is computed (ksigns_iq2xs[i] =
// i | parity(i) << 7) instead of looked up. Values and operation order are those of decode8.
std::atomic<bool> grouped_mma_wide{false};
__device__ __forceinline__ uint32_t ksign(uint32_t i) { return i | ((__popc(i) & 1u) << 7); }
// 16 bytes at a 2-byte-aligned address; the warp shares the alignment (row strides are 4-aligned).
__device__ __forceinline__ void load16(const uint8_t* p, uint32_t (&w)[4]) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    if (!(a & 3)) {
        const uint32_t* q = reinterpret_cast<const uint32_t*>(a);
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = q[i];
    } else {
        const uint32_t* q = reinterpret_cast<const uint32_t*>(a - 2);
        uint32_t t[5];
#pragma unroll
        for (int i = 0; i < 4; ++i) t[i] = q[i];
        t[4] = *reinterpret_cast<const uint16_t*>(a + 14);
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = __funnelshift_r(t[i], t[i + 1], 16);
    }
}
template<int Type> struct LaneBlock { uint32_t qs[4]; float d; uint32_t sc; };
template<int Type>
__device__ __forceinline__ void lane_load(const uint8_t* blk, int c, LaneBlock<Type>& r) {
    load16(blk + 2 + 16 * c, r.qs);
    r.d = __half2float(*reinterpret_cast<const __half*>(blk));
    if constexpr (Type == 17) r.sc = *reinterpret_cast<const uint16_t*>(blk + 66 + 2 * c);
    else r.sc = 0;
}
// Same as decode8<Type>(blk, 2c + i, il) for Type 16/17.
template<int Type>
__device__ __forceinline__ void lane_decode(const LaneBlock<Type>& r, int i, int il, float (&out)[8]) {
    if constexpr (Type == 16) {
        const uint32_t aux = r.qs[2 * i + 1];
        const float d = r.d * (0.5f + (aux >> 28)) * 0.25f;
        const uint32_t grid = (r.qs[2 * i] >> (8 * il)) & 255;
        decode_iq2<Type>(d, ksign((aux >> (7 * il)) & 127), iq2_levels::iq2xxs_selectors[grid], out);
    } else {
        const uint32_t q = (r.qs[2 * i + (il >> 1)] >> (16 * (il & 1))) & 0xffff;
        const uint32_t scale = (r.sc >> (8 * i)) & 255;
        const float d = r.d * (0.5f + ((scale >> (4 * (il / 2))) & 15)) * 0.25f;
        decode_iq2<Type>(d, ksign(q >> 9), iq2_levels::iq2xs_selectors[q & 511], out);
    }
}
// Superblocks sb = first, first+step, ... of rows (ra, rb).
template<int Type, int Blocks, bool Wide = false>
__device__ __forceinline__ void mma_rows(const uint8_t* ra, const uint8_t* rb, const uint16_t* xr, bool valid, int c,
                                         int first, float (&acc)[4]) {
#pragma unroll
    for (int j = 0; j < (Blocks + 3) / 4; ++j) {
        const int sb = first + 4 * j;
        if (sb >= Blocks) break;
        uint4 xb[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) xb[k] = valid ? *reinterpret_cast<const uint4*>(xr + sb * 256 + 8 * k) : make_uint4(0, 0, 0, 0);
        const uint8_t* ba = ra + sb * Fmt<Type>::bytes;
        const uint8_t* bb = rb + sb * Fmt<Type>::bytes;
        if constexpr (Wide && Type == 16) {
            LaneBlock<Type> la, lb;
            lane_load<Type>(ba, c, la);
            lane_load<Type>(bb, c, lb);
#pragma unroll
            for (int g8 = 0; g8 < 8; ++g8) {
                float fa[8], fb[8];
                lane_decode<Type>(la, g8 / 4, g8 % 4, fa);
                lane_decode<Type>(lb, g8 / 4, g8 % 4, fb);
                uint32_t wa[4], wb[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) { wa[q] = pack2(fa[2 * q], fa[2 * q + 1]); wb[q] = pack2(fb[2 * q], fb[2 * q + 1]); }
                const uint32_t xw[4] = {xb[g8].x, xb[g8].y, xb[g8].z, xb[g8].w};
#pragma unroll
                for (int h = 0; h < 2; ++h)
                    mma16816(acc, wa[2 * h], wb[2 * h], wa[2 * h + 1], wb[2 * h + 1], xw[2 * h], xw[2 * h + 1]);
            }
            continue;
        }
#pragma unroll
        for (int g8 = 0; g8 < 8; ++g8) {
            uint32_t wa[4], wb[4];
            if constexpr (Type == 18) {
                decode8_pairs<Type>(ba, 2 * c + g8 / 4, g8 % 4, wa);
                decode8_pairs<Type>(bb, 2 * c + g8 / 4, g8 % 4, wb);
            } else {
                float fa[8], fb[8];
                decode8<Type>(ba, 2 * c + g8 / 4, g8 % 4, fa);
                decode8<Type>(bb, 2 * c + g8 / 4, g8 % 4, fb);
#pragma unroll
                for (int j = 0; j < 4; ++j) { wa[j] = pack2(fa[2 * j], fa[2 * j + 1]); wb[j] = pack2(fb[2 * j], fb[2 * j + 1]); }
            }
            const uint32_t xw[4] = {xb[g8].x, xb[g8].y, xb[g8].z, xb[g8].w};
#pragma unroll
            for (int h = 0; h < 2; ++h)
                mma16816(acc, wa[2 * h], wb[2 * h], wa[2 * h + 1], wb[2 * h + 1], xw[2 * h], xw[2 * h + 1]);
        }
    }
}
// A block owns 16 rows; warp w accumulates superblocks w, w+4, ...; partial tiles are added
// in warp order 0+1+2+3 (fixed, independent of the group's token count).
template<int M>
__device__ __forceinline__ void mma_reduce(float (&acc)[M][4]) {
    __shared__ float partial[3][M][4][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (warp) {
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
            for (int e = 0; e < 4; ++e) partial[warp - 1][m][e][lane] = acc[m][e];
    }
    __syncthreads();
    if (!warp) {
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[m][e] = ((acc[m][e] + partial[0][m][e][lane]) + partial[1][m][e][lane]) + partial[2][m][e][lane];
    }
}
template<int Type, bool Wide>
__global__ void __launch_bounds__(128) mma_gu_kernel(Work w, size_t up_offset, const uint16_t* x, const int* token_rows,
                                                     const float* routing, uint16_t* activation) {
    const int gi = find_group(w);
    const Desc g = w.g[gi];
    const int lane = threadIdx.x & 31, gid = lane >> 2, c = lane & 3, warp = threadIdx.x >> 5;
    const int row = (blockIdx.x - w.first[gi]) * 16 + gid;
    const bool valid = gid < g.tokens;
    constexpr int RB = 20 * Fmt<Type>::bytes;
    const uint16_t* xr = x + (valid ? int64_t(token_rows[g.offset + gid]) * 5120 : 0) + 64 * c;
    const uint8_t* ga = g.packed + size_t(row) * RB;
    float acc[2][4] = {};
    mma_rows<Type, 20, Wide>(ga, ga + 8 * RB, xr, valid, c, warp, acc[0]);
    mma_rows<Type, 20, Wide>(ga + up_offset, ga + up_offset + 8 * RB, xr, valid, c, warp, acc[1]);
    mma_reduce<2>(acc);
    if (warp) return;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int n = 2 * c + (e & 1);
        if (n >= g.tokens) continue;
        const float v = fminf(bf16r(acc[0][e]), 10.f);
        const float u = fminf(fmaxf(bf16r(acc[1][e]), -10.f), 10.f);
        const float sig = v >= 0 ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v));
        const int slot = g.offset + n;
        const float a = bf16r(v * sig * u * routing[slot]);
        activation[int64_t(slot) * 2304 + row + (e >= 2 ? 8 : 0)] = uint16_t(__float_as_uint(a) >> 16);
    }
}
template<int Type, bool Wide>
__global__ void __launch_bounds__(128) mma_down_kernel(Work w, size_t down_offset, const uint16_t* activation, const int* slots,
                                                       float* parts) {
    const int gi = find_group(w);
    const Desc g = w.g[gi];
    const int lane = threadIdx.x & 31, gid = lane >> 2, c = lane & 3, warp = threadIdx.x >> 5;
    const int row = (blockIdx.x - w.first[gi]) * 16 + gid;
    const bool valid = gid < g.tokens;
    const uint16_t* xr = activation + int64_t(g.offset + (valid ? gid : 0)) * 2304 + 64 * c;
    constexpr int RB = 9 * Fmt<Type>::bytes;
    const uint8_t* da = g.packed + down_offset + size_t(row) * RB;
    float acc[1][4] = {};
    mma_rows<Type, 9, Wide>(da, da + 8 * RB, xr, valid, c, warp, acc[0]);
    mma_reduce<1>(acc);
    if (warp) return;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int n = 2 * c + (e & 1);
        if (n < g.tokens) parts[int64_t(slots[g.offset + n]) * 5120 + row + (e >= 2 ? 8 : 0)] = bf16r(acc[0][e]);
    }
}
// ---- Warp-specialized persistent decode MoE (bit-identical products to mma_gu_kernel / mma_down_kernel).
// A work item is one m16 tile of one expert: gu = 8 features (tile rows 0-7 their gate rows, 8-15 their up rows),
// down = 16 output rows. The tile's rows are contiguous, 16-byte aligned spans, so one producer thread streams them
// into a shared-memory ring with cp.async.bulk + mbarrier while groups of four consumer warps compute one item each.
// The four warps of a group take superblocks w, w+4, ... and reduce in warp order as mma_rows / mma_reduce do; every
// output element is the same chain of m16n8k16 products in the same K order. Decode reads each lane's 16 superblock
// bytes once, hoists the per-sub-block levels and takes the sign masks from a shared table (same BF16 bits).
std::atomic<bool> grouped_ws{true};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define WS_ASM(...) do {} while (0)
#else
#define WS_ASM(...) asm volatile(__VA_ARGS__)
#endif
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    WS_ASM("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
    WS_ASM("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    WS_ASM("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
    WS_ASM("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}"
           ::"r"(smem_u32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    WS_ASM("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
           ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ uint32_t lds16(const uint8_t* p) { return *reinterpret_cast<const uint16_t*>(p); }
__device__ __forceinline__ uint32_t lds32u(const uint8_t* p) {   // 2-byte aligned
    return (reinterpret_cast<uintptr_t>(p) & 3) ? (lds16(p) | (lds16(p + 2) << 16)) : *reinterpret_cast<const uint32_t*>(p);
}
// One lane's view of a superblock: sub-blocks 2c and 2c+1.
template<int Type> struct WsLane { uint32_t q[4]; uint32_t s[2]; float d; };
template<int Type>
__device__ __forceinline__ void ws_fetch(const uint8_t* blk, int c, WsLane<Type>& r) {
    r.d = __half2float(*reinterpret_cast<const __half*>(blk));
    if constexpr (Type == 16 || Type == 17) {
        load16(blk + 2 + 16 * c, r.q);
        if constexpr (Type == 17) r.s[0] = lds16(blk + 66 + 2 * c);
    } else if constexpr (Type == 22) {
        r.q[0] = lds32u(blk + 2 + 8 * c); r.q[1] = lds32u(blk + 6 + 8 * c);       // grid low bytes, ib 2c / 2c+1
        r.q[2] = lds32u(blk + 34 + 8 * c); r.q[3] = lds32u(blk + 38 + 8 * c);     // signs
        r.s[0] = lds16(blk + 66 + 2 * c);                                         // qh[2c], qh[2c+1]
        r.s[1] = lds16(blk + 74 + 2 * c);                                         // scales[2c], scales[2c+1]
    }
}
__device__ __forceinline__ void ws_levels(float d, uint32_t& a, uint32_t& b) {
    const __nv_bfloat162 l01 = __floats2bfloat162_rn(d * 8.f, d * 25.f);
    const __nv_bfloat162 l2 = __floats2bfloat162_rn(d * 43.f, 0.f);
    a = *reinterpret_cast<const uint32_t*>(&l01); b = *reinterpret_cast<const uint32_t*>(&l2);
}
// lut[e] in registers: sign bit 2p -> bit 15 and 2p + 1 -> bit 31 of pair p. s * (2^(15-2p) + 2^(30-2p)) places the
// two shifted copies without overlap, so for s < 1024 bits 15 and 31 hold exactly bits 2p and 2p + 1.
__device__ __forceinline__ uint4 ws_mask(uint32_t s) {
    constexpr uint32_t M = 0x80008000u;
    return make_uint4((s * 0x40008000u) & M, (s * 0x10002000u) & M, (s * 0x04000800u) & M, (s * 0x01000200u) & M);
}
// ksign(v) for v < 128 up to bits 8-9, which ws_mask ignores.
__device__ __forceinline__ uint32_t ws_ksign(uint32_t v) { return v | (uint32_t(__popc(v)) << 7); }
__device__ __forceinline__ void ws_perm(uint32_t a, uint32_t b, uint64_t sel, const uint4 m, uint32_t (&out)[4]) {
    const uint32_t lo = uint32_t(sel), hi = uint32_t(sel >> 32);
    out[0] = __byte_perm(a, b, lo) ^ m.x;
    out[1] = __byte_perm(a, b, lo >> 16) ^ m.y;
    out[2] = __byte_perm(a, b, hi) ^ m.z;
    out[3] = __byte_perm(a, b, hi >> 16) ^ m.w;
}
// The four BF16 pairs of decode8_pairs<Type>(blk, 2c + i, il) for il = 0..3 (one sub-block, 32 weights).
// sel: the type's selector table in shared memory; lut: sign XOR masks, entries 0..127 for 7-bit signs with
// llama.cpp's parity bit folded in (ksign), 128..383 for raw 8-bit signs.
template<int Type>
__device__ __forceinline__ void ws_sub(const WsLane<Type>& r, int i, const uint64_t* sel, const uint4* lut, uint32_t (&out)[4][4]) {
    if constexpr (Type == 16) {
        const uint32_t aux = r.q[2 * i + 1], grid = r.q[2 * i];
        uint32_t a, b; ws_levels(r.d * (0.5f + (aux >> 28)) * 0.25f, a, b);
#pragma unroll
        for (int il = 0; il < 4; ++il)
            ws_perm(a, b, sel[__byte_perm(grid, 0, 0x4440 + il)], ws_mask(ws_ksign((aux >> (7 * il)) & 127)), out[il]);
    } else if constexpr (Type == 17) {
        const uint32_t sc = (r.s[0] >> (8 * i)) & 255;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint32_t a, b; ws_levels(r.d * (0.5f + ((sc >> (4 * h)) & 15)) * 0.25f, a, b);
            const uint32_t word = r.q[2 * i + h];
            ws_perm(a, b, sel[word & 511], ws_mask(ws_ksign((word >> 9) & 127)), out[2 * h]);
            ws_perm(a, b, sel[(word >> 16) & 511], ws_mask(ws_ksign(word >> 25)), out[2 * h + 1]);
        }
    } else {
        static_assert(Type == 22);
        const uint32_t sc = (r.s[1] >> (8 * i)) & 255, qh = (r.s[0] >> (8 * i)) & 255;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint32_t a, b; ws_levels(r.d * (0.5f + ((sc >> (4 * h)) & 15)) * 0.25f, a, b);
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int il = 2 * h + k;
                const uint32_t q = __byte_perm(r.q[i], 0, 0x4440 + il) | ((qh << (8 - 2 * il)) & 0x300);
                ws_perm(a, b, sel[q], ws_mask(__byte_perm(r.q[2 + i], 0, 0x4440 + il)), out[il]);
            }
        }
    }
}
// One tile (rows ra = gid, rb = gid + 8) over superblocks first, first+4, ...: mma_rows' product sequence.
template<int Type, int Blocks, bool CG = false>
__device__ __forceinline__ void ws_rows(const uint8_t* ra, const uint8_t* rb, const uint16_t* xr, bool valid, int c, int first,
                                        const uint64_t* sel, const uint4* lut, float (&acc)[4]) {
#pragma unroll 1
    for (int sb = first; sb < Blocks; sb += 4) {
        uint4 xb[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint4* xp = reinterpret_cast<const uint4*>(xr + sb * 256 + 8 * k);
            xb[k] = valid ? (CG ? __ldcg(xp) : *xp) : make_uint4(0, 0, 0, 0);
        }
        const uint8_t* ba = ra + sb * Fmt<Type>::bytes;
        const uint8_t* bb = rb + sb * Fmt<Type>::bytes;
        if constexpr (Type == 18) {
#pragma unroll
            for (int g8 = 0; g8 < 8; ++g8) {
                uint32_t wa[4], wb[4];
                decode8_pairs<Type>(ba, 2 * c + g8 / 4, g8 % 4, wa);
                decode8_pairs<Type>(bb, 2 * c + g8 / 4, g8 % 4, wb);
                const uint32_t xw[4] = {xb[g8].x, xb[g8].y, xb[g8].z, xb[g8].w};
#pragma unroll
                for (int q = 0; q < 2; ++q) mma16816(acc, wa[2 * q], wb[2 * q], wa[2 * q + 1], wb[2 * q + 1], xw[2 * q], xw[2 * q + 1]);
            }
        } else {
            WsLane<Type> la, lb;
            ws_fetch<Type>(ba, c, la);
            ws_fetch<Type>(bb, c, lb);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                uint32_t wa[4][4], wb[4][4];
                ws_sub<Type>(la, i, sel, lut, wa);
                ws_sub<Type>(lb, i, sel, lut, wb);
#pragma unroll
                for (int il = 0; il < 4; ++il) {
                    const uint4 xv = xb[4 * i + il];
                    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                    for (int q = 0; q < 2; ++q)
                        mma16816(acc, wa[il][2 * q], wb[il][2 * q], wa[il][2 * q + 1], wb[il][2 * q + 1], xw[2 * q], xw[2 * q + 1]);
                }
            }
        }
    }
}
__device__ __forceinline__ void ws_reduce(float (&acc)[4], float* part, int wg, int bar) {
    const int lane = threadIdx.x & 31;
    if (wg) {
#pragma unroll
        for (int e = 0; e < 4; ++e) part[((wg - 1) * 4 + e) * 32 + lane] = acc[e];
    }
    asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");
    if (!wg) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
            acc[e] = ((acc[e] + part[(0 * 4 + e) * 32 + lane]) + part[(1 * 4 + e) * 32 + lane]) + part[(2 * 4 + e) * 32 + lane];
    }
    asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");
}
struct WsWork { Desc g[kMaxGroups]; int count; };
template<int Type, bool GU, int T = 1> __host__ __device__ constexpr int ws_slot() { return 16 * T * (GU ? 20 : 9) * Fmt<Type>::bytes; }
template<int Type> __host__ __device__ constexpr int ws_selectors() { return Type == 16 ? 256 : Type == 17 ? 512 : Type == 22 ? 1024 : 0; }
template<int Type, bool GU, int NS, int G, int T = 1>
__host__ __device__ constexpr size_t ws_smem() {
    return size_t(NS) * ws_slot<Type, GU, T>() + 384 * 16 + ws_selectors<Type>() * 8 + size_t(NS) * T * 4 * 4 * 32 * 4 + 2 * NS * 8 + NS * 8;
}
template<int Type, bool GU, int NS, int G, int T = 1>
__global__ void __launch_bounds__(32 * (4 * G + 1), 1) ws_kernel(WsWork w, size_t up_offset, size_t offset, const uint16_t* x,
                                                                 const int* token_rows, const float* routing, const int* slots,
                                                                 uint16_t* activation, float* parts) {
    static_assert(!GU || T == 1, "gu items are one tile");
    constexpr int BLOCKS = GU ? 20 : 9, RB = BLOCKS * Fmt<Type>::bytes, SLOT = ws_slot<Type, GU, T>();
    constexpr int PER = GU ? 2304 / 8 : 5120 / (16 * T);
    static_assert(SLOT % 16 == 0 && (8 * RB) % 16 == 0, "bulk copy granularity");
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* ring = smem;
    uint4* lut = reinterpret_cast<uint4*>(smem + NS * SLOT);
    uint64_t* sel = reinterpret_cast<uint64_t*>(lut + 384);
    float* part = reinterpret_cast<float*>(sel + ws_selectors<Type>());           // [NS][T][4 roles][4][32]
    uint64_t* full = reinterpret_cast<uint64_t*>(part + NS * T * 4 * 4 * 32);
    uint64_t* empty = full + NS;
    unsigned* done = reinterpret_cast<unsigned*>(empty + NS);
    // seq[slot]: the CTA-local item whose load is armed in the slot's current mbarrier phase. A consumer waits for it
    // before the parity wait, so it never tests a phase two ahead (which parity cannot distinguish).
    volatile int* seq = reinterpret_cast<volatile int*>(done + NS);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int items = w.count * PER;
    const int mine = items > int(blockIdx.x) ? (items - 1 - int(blockIdx.x)) / int(gridDim.x) + 1 : 0;
    for (int e = threadIdx.x; e < 384; e += blockDim.x) {
        const uint32_t s = e < 128 ? ksign(uint32_t(e)) : uint32_t(e - 128);
        uint32_t m[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) m[p] = (((s >> (2 * p)) & 1u) << 15) | (((s >> (2 * p + 1)) & 1u) << 31);
        lut[e] = make_uint4(m[0], m[1], m[2], m[3]);
    }
    if constexpr (ws_selectors<Type>() > 0) {
        const uint64_t* table = Type == 16 ? iq2_levels::iq2xxs_selectors : Type == 17 ? iq2_levels::iq2xs_selectors : iq2_levels::iq2s_selectors;
        for (int e = threadIdx.x; e < ws_selectors<Type>(); e += blockDim.x) sel[e] = table[e];
    }
    if (threadIdx.x == 0) {
        for (int s = 0; s < NS; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 4); done[s] = 0; seq[s] = -1; }
        WS_ASM("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp == 4 * G) {
        if (lane == 0) {
            for (int j = 0; j < mine; ++j) {
                const int slot = j % NS, use = j / NS;
                if (use) mbar_wait(empty + slot, (use - 1) & 1);
                const int item = int(blockIdx.x) + j * int(gridDim.x);
                const Desc g = w.g[item / PER];
                uint8_t* dst = ring + slot * SLOT;
                mbar_expect(full + slot, SLOT);
                seq[slot] = j;
                if constexpr (GU) {
                    const size_t f0 = size_t(item % PER) * 8 * RB;
                    bulk_load(dst, g.packed + f0, 8 * RB, full + slot);
                    bulk_load(dst + 8 * RB, g.packed + up_offset + f0, 8 * RB, full + slot);
                } else {
                    bulk_load(dst, g.packed + offset + size_t(item % PER) * SLOT, SLOT, full + slot);
                }
            }
        }
        return;
    }
    const int group = warp >> 2, wg = warp & 3, gid = lane >> 2, c = lane & 3;
    for (int j = group, t = 0; j < mine; j += G, ++t) {
        const int item = int(blockIdx.x) + j * int(gridDim.x);
        const Desc g = w.g[item / PER];
        const bool valid = gid < g.tokens;
        const int slot = j % NS, role = (wg + t) & 3;
        while (seq[slot] != j) __nanosleep(32);
        mbar_wait(full + slot, (j / NS) & 1);
        const uint16_t* xr = GU ? x + (valid ? int64_t(token_rows[g.offset + gid]) * 5120 : 0) + 64 * c
                                : x + int64_t(g.offset + (valid ? gid : 0)) * 2304 + 64 * c;
        float acc[T][4] = {};
#pragma unroll
        for (int tt = 0; tt < T; ++tt) {
            const uint8_t* ra = ring + slot * SLOT + (16 * tt + gid) * RB;
            ws_rows<Type, BLOCKS>(ra, ra + 8 * RB, xr, valid, c, role, sel, lut, acc[tt]);
        }
        // Role r's partial is warp r's partial of mma_rows; the last of the four reduces in role order.
        float* sp = part + slot * T * 4 * 4 * 32;
#pragma unroll
        for (int tt = 0; tt < T; ++tt)
#pragma unroll
            for (int e = 0; e < 4; ++e) sp[((tt * 4 + role) * 4 + e) * 32 + lane] = acc[tt][e];
        __threadfence_block();
        __syncwarp();
        unsigned arrived = 0;
        if (lane == 0) arrived = atomicAdd(done + slot, 1u);
        arrived = __shfl_sync(0xffffffff, arrived, 0);
        if (arrived == 3) {
            __threadfence_block();
#pragma unroll
            for (int tt = 0; tt < T; ++tt) {
                const float* q = sp + tt * 4 * 4 * 32;
                float r[4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    r[e] = ((q[(0 * 4 + e) * 32 + lane] + q[(1 * 4 + e) * 32 + lane]) + q[(2 * 4 + e) * 32 + lane]) + q[(3 * 4 + e) * 32 + lane];
                if constexpr (GU) {
                    const int feature = (item % PER) * 8 + gid;
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int n = 2 * c + e;
                        if (n >= g.tokens) continue;
                        const float v = fminf(bf16r(r[e]), 10.f);
                        const float u = fminf(fmaxf(bf16r(r[e + 2]), -10.f), 10.f);
                        const float sig = v >= 0 ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v));
                        const int slotr = g.offset + n;
                        const float a = bf16r(v * sig * u * routing[slotr]);
                        activation[int64_t(slotr) * 2304 + feature] = uint16_t(__float_as_uint(a) >> 16);
                    }
                } else {
                    const int row = (item % PER) * 16 * T + 16 * tt + gid;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int n = 2 * c + (e & 1);
                        if (n < g.tokens) parts[int64_t(slots[g.offset + n]) * 5120 + row + (e >= 2 ? 8 : 0)] = bf16r(r[e]);
                    }
                }
            }
            if (lane == 0) done[slot] = 0;
            __syncwarp();
        }
        if (lane == 0) mbar_arrive(empty + slot);
    }
}
template<int Type, bool GU, int NS, int G, int T = 1>
void ws_launch(const WsWork& w, size_t up_offset, size_t offset, const uint16_t* x, const int* token_rows, const float* routing,
               const int* slots, uint16_t* activation, float* parts, cudaStream_t stream) {
    static_assert(ws_smem<Type, GU, NS, G, T>() <= 101376, "shared memory");
    static int sms = [] { int d = 0, v = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d); return v; }();
    static bool configured = [] {
        return cudaFuncSetAttribute(ws_kernel<Type, GU, NS, G, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(ws_smem<Type, GU, NS, G, T>())) == cudaSuccess;
    }();
    (void)configured;
    const int items = w.count * (GU ? 2304 / 8 : 5120 / (16 * T));
    ws_kernel<Type, GU, NS, G, T><<<std::min(items, sms), 32 * (4 * G + 1), ws_smem<Type, GU, NS, G, T>(), stream>>>(
        w, up_offset, offset, x, token_rows, routing, slots, activation, parts);
}

// ---- gu and down in one persistent kernel: items [0, count*288) are gu tiles, then count*320 down tiles. The producer
// streams down weights as soon as slots free (they do not depend on gu); a down tile's consumers wait until all 288 gu
// tiles of its expert have published (a monotonic per-launch target), then read H through L2.
__device__ unsigned long long ws_ready[kMaxGroups];
__device__ __forceinline__ const uint64_t* ws_global_selectors(int type) {
    return type == 16 ? iq2_levels::iq2xxs_selectors : type == 17 ? iq2_levels::iq2xs_selectors : iq2_levels::iq2s_selectors;
}
template<int GT, int DT> __host__ __device__ constexpr bool wsf_shared_down() { return DT == GT && ws_selectors<GT>() > 0 && GT != 22; }
template<int GT, int DT, int NS, int G>
__host__ __device__ constexpr size_t wsf_smem() {
    return size_t(NS) * ws_slot<GT, true>() + 384 * 16 + (GT == 22 ? 0 : ws_selectors<GT>() * 8) + size_t(NS) * 4 * 4 * 32 * 4 +
           2 * NS * 8 + NS * 4;
}
template<int GT, int DT, int NS, int G>
__global__ void __launch_bounds__(32 * (4 * G + 1), 1) wsf_kernel(WsWork w, size_t up_offset, size_t down_offset, const uint16_t* x,
                                                                  const int* token_rows, const float* routing, const int* slots,
                                                                  uint16_t* activation, float* parts, unsigned long long target) {
    constexpr int RBG = 20 * Fmt<GT>::bytes, RBD = 9 * Fmt<DT>::bytes, SLOT = ws_slot<GT, true>(), SLOTD = ws_slot<DT, false>();
    static_assert(SLOT >= SLOTD && SLOT % 16 == 0 && SLOTD % 16 == 0 && (8 * RBG) % 16 == 0, "slot layout");
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* ring = smem;
    uint4* lut = reinterpret_cast<uint4*>(smem + NS * SLOT);
    uint64_t* sel = reinterpret_cast<uint64_t*>(lut + 384);
    float* part = reinterpret_cast<float*>(sel + (GT == 22 ? 0 : ws_selectors<GT>()));
    uint64_t* full = reinterpret_cast<uint64_t*>(part + NS * 4 * 4 * 32);
    uint64_t* empty = full + NS;
    unsigned* done = reinterpret_cast<unsigned*>(empty + NS);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ngu = w.count * 288, items = ngu + w.count * 320;
    const int mine = items > int(blockIdx.x) ? (items - 1 - int(blockIdx.x)) / int(gridDim.x) + 1 : 0;
    for (int e = threadIdx.x; e < 384; e += blockDim.x) {
        const uint32_t s = e < 128 ? ksign(uint32_t(e)) : uint32_t(e - 128);
        uint32_t m[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) m[p] = (((s >> (2 * p)) & 1u) << 15) | (((s >> (2 * p + 1)) & 1u) << 31);
        lut[e] = make_uint4(m[0], m[1], m[2], m[3]);
    }
    if constexpr (GT != 22) {
        const uint64_t* table = ws_global_selectors(GT);
        for (int e = threadIdx.x; e < ws_selectors<GT>(); e += blockDim.x) sel[e] = table[e];
    }
    if (threadIdx.x == 0) {
        for (int s = 0; s < NS; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 4); done[s] = 0; }
        WS_ASM("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp == 4 * G) {
        if (lane == 0) {
            for (int j = 0; j < mine; ++j) {
                const int slot = j % NS, use = j / NS;
                if (use) mbar_wait(empty + slot, (use - 1) & 1);
                const int item = int(blockIdx.x) + j * int(gridDim.x);
                uint8_t* dst = ring + slot * SLOT;
                if (item < ngu) {
                    const Desc g = w.g[item / 288];
                    const size_t f0 = size_t(item % 288) * 8 * RBG;
                    mbar_expect(full + slot, SLOT);
                    bulk_load(dst, g.packed + f0, 8 * RBG, full + slot);
                    bulk_load(dst + 8 * RBG, g.packed + up_offset + f0, 8 * RBG, full + slot);
                } else {
                    const int d = item - ngu;
                    const Desc g = w.g[d / 320];
                    mbar_expect(full + slot, SLOTD);
                    bulk_load(dst, g.packed + down_offset + size_t(d % 320) * SLOTD, SLOTD, full + slot);
                }
            }
        }
        return;
    }
    const uint64_t* gsel = GT == 22 ? ws_global_selectors(22) : sel;
    const uint64_t* dsel = wsf_shared_down<GT, DT>() ? sel : ws_global_selectors(DT);
    const int group = warp >> 2, wg = warp & 3, gid = lane >> 2, c = lane & 3;
    for (int j = group, t = 0; j < mine; j += G, ++t) {
        const int item = int(blockIdx.x) + j * int(gridDim.x);
        const bool gu = item < ngu;
        const int gi = gu ? item / 288 : (item - ngu) / 320, tile = gu ? item % 288 : (item - ngu) % 320;
        const Desc g = w.g[gi];
        const bool valid = gid < g.tokens;
        const int slot = j % NS, role = (wg + t) & 3;
        mbar_wait(full + slot, (j / NS) & 1);
        float acc[4] = {};
        if (gu) {
            const uint8_t* ra = ring + slot * SLOT + gid * RBG;
            const uint16_t* xr = x + (valid ? int64_t(token_rows[g.offset + gid]) * 5120 : 0) + 64 * c;
            ws_rows<GT, 20>(ra, ra + 8 * RBG, xr, valid, c, role, gsel, lut, acc);
        } else {
            if (lane == 0) {
                const volatile unsigned long long* r = ws_ready + gi;
                while (*r < target) __nanosleep(64);
            }
            __syncwarp();
            __threadfence();
            const uint8_t* ra = ring + slot * SLOT + gid * RBD;
            const uint16_t* xr = activation + int64_t(g.offset + (valid ? gid : 0)) * 2304 + 64 * c;
            if constexpr (DT == 18) ws_rows<18, 9, true>(ra, ra + 8 * RBD, xr, valid, c, role, nullptr, lut, acc);
            else ws_rows<DT, 9, true>(ra, ra + 8 * RBD, xr, valid, c, role, dsel, lut, acc);
        }
        float* sp = part + slot * 4 * 4 * 32;
#pragma unroll
        for (int e = 0; e < 4; ++e) sp[(role * 4 + e) * 32 + lane] = acc[e];
        __threadfence_block();
        __syncwarp();
        unsigned arrived = 0;
        if (lane == 0) arrived = atomicAdd(done + slot, 1u);
        arrived = __shfl_sync(0xffffffff, arrived, 0);
        if (arrived == 3) {
            __threadfence_block();
            float r[4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
                r[e] = ((sp[(0 * 4 + e) * 32 + lane] + sp[(1 * 4 + e) * 32 + lane]) + sp[(2 * 4 + e) * 32 + lane]) + sp[(3 * 4 + e) * 32 + lane];
            if (gu) {
                const int feature = tile * 8 + gid;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int n = 2 * c + e;
                    if (n >= g.tokens) continue;
                    const float v = fminf(bf16r(r[e]), 10.f);
                    const float u = fminf(fmaxf(bf16r(r[e + 2]), -10.f), 10.f);
                    const float sig = v >= 0 ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v));
                    const int slotr = g.offset + n;
                    const float a = bf16r(v * sig * u * routing[slotr]);
                    activation[int64_t(slotr) * 2304 + feature] = uint16_t(__float_as_uint(a) >> 16);
                }
                __threadfence();
                __syncwarp();
                if (lane == 0) atomicAdd(ws_ready + gi, 1ull);
            } else {
                const int row = tile * 16 + gid;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int n = 2 * c + (e & 1);
                    if (n < g.tokens) parts[int64_t(slots[g.offset + n]) * 5120 + row + (e >= 2 ? 8 : 0)] = bf16r(r[e]);
                }
            }
            if (lane == 0) done[slot] = 0;
            __syncwarp();
        }
        if (lane == 0) mbar_arrive(empty + slot);
    }
}
std::atomic<bool> grouped_ws_fused{false};
template<int GT, int DT, int NS, int G>
void wsf_launch(const WsWork& w, size_t up_offset, size_t down_offset, const uint16_t* x, const int* token_rows, const float* routing,
                const int* slots, uint16_t* activation, float* parts, cudaStream_t stream) {
    static_assert(wsf_smem<GT, DT, NS, G>() <= 101376, "shared memory");
    static int sms = [] { int d = 0, v = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d); return v; }();
    static bool configured = [] {
        return cudaFuncSetAttribute(wsf_kernel<GT, DT, NS, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(wsf_smem<GT, DT, NS, G>())) == cudaSuccess;
    }();
    (void)configured;
    static unsigned long long epoch[16] = {};
    int dev = 0; cudaGetDevice(&dev);
    const unsigned long long target = ++epoch[dev & 15] * 288ull;
    const int items = w.count * (288 + 320);
    wsf_kernel<GT, DT, NS, G><<<std::min(items, sms), 32 * (4 * G + 1), wsf_smem<GT, DT, NS, G>(), stream>>>(
        w, up_offset, down_offset, x, token_rows, routing, slots, activation, parts, target);
}
template<int GT, int NS, int G>
bool wsf_dispatch(uint32_t down_type, const WsWork& w, size_t up_offset, size_t down_offset, const uint16_t* x, const int* token_rows,
                  const float* routing, const int* slots, uint16_t* activation, float* parts, cudaStream_t stream) {
    switch (down_type) {
        case 16: wsf_launch<GT, 16, NS, G>(w, up_offset, down_offset, x, token_rows, routing, slots, activation, parts, stream); return true;
        case 17: wsf_launch<GT, 17, NS, G>(w, up_offset, down_offset, x, token_rows, routing, slots, activation, parts, stream); return true;
        case 18: wsf_launch<GT, 18, NS, G>(w, up_offset, down_offset, x, token_rows, routing, slots, activation, parts, stream); return true;
        case 22: wsf_launch<GT, 22, NS, G>(w, up_offset, down_offset, x, token_rows, routing, slots, activation, parts, stream); return true;
    }
    return false;
}
template<int Type, bool B>
void launch_gu(int n, const Work& w, size_t up_offset, const void* x, const int* token_rows, const float* routing,
               void* activation, cudaStream_t stream) {
    const dim3 grid(w.first[w.count]);
    switch (n) {
#define C(N) case N: grouped_gu_kernel<Type, N, B><<<grid, 128, 0, stream>>>(w, up_offset, x, token_rows, routing, activation); break;
        C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8)
#undef C
    }
}
template<int Type, bool B>
void launch_down(int n, const Work& w, size_t down_offset, const void* activation, const int* slots, float* parts,
                 cudaStream_t stream) {
    const dim3 grid(w.first[w.count]);
    switch (n) {
#define C(N) case N: grouped_down_kernel<Type, N, B><<<grid, 128, 0, stream>>>(w, down_offset, activation, slots, parts); break;
        C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8)
#undef C
    }
}
} // namespace

void expert_grouped_set_enabled(bool value) { grouped_enabled.store(value); }
void expert_grouped_bf16_inputs_set_enabled(bool value) { grouped_bf16_inputs.store(value); }
void expert_grouped_mma_set_enabled(bool value) { grouped_mma.store(value); }
void expert_grouped_mma_wide_set_enabled(bool value) { grouped_mma_wide.store(value); }
void expert_grouped_ws_set_enabled(bool value) { grouped_ws.store(value); }
void expert_grouped_ws_fused_set_enabled(bool value) { grouped_ws_fused.store(value); }
bool expert_grouped_mma_enabled() { return grouped_mma.load(); }
bool expert_grouped_enabled() { return grouped_enabled.load(); }
bool expert_grouped_supported(uint32_t gu, uint32_t down) {
    return (gu == 16 || gu == 17 || gu == 22) && (down == 16 || down == 17 || down == 18 || down == 22);
}

cudaError_t expert_grouped_gpu(const VerifyExpertGroup* groups, int count, uint32_t gu_type, uint32_t down_type,
                               size_t up_offset, size_t down_offset, const float* x, const int* token_rows,
                               const float* routing, const int* slots, float* activation, float* parts, int tokens,
                               void* scratch, size_t scratch_bytes, cudaStream_t stream) {
    if (!groups || count < 1 || count > kMaxGroups || !expert_grouped_supported(gu_type, down_type) || tokens < 1 ||
        tokens > 8 || !x || !token_rows || !routing || !slots || !activation || !parts ||
        ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(activation)) & 15))
        return cudaErrorInvalidValue;
    Work gu[9], dn[9];
    for (int n = 1; n <= 8; ++n) gu[n].count = dn[n].count = 0;
    for (int i = 0; i < count; ++i) {
        const int n = groups[i].tokens;
        if (!groups[i].packed || (reinterpret_cast<uintptr_t>(groups[i].packed) & 1) || n < 1 || n > tokens ||
            groups[i].offset < 0 || groups[i].offset + n > tokens * 6)
            return cudaErrorInvalidValue;
        Work& a = gu[n];
        Work& b = dn[n];
        a.g[a.count] = b.g[b.count] = {static_cast<const uint8_t*>(groups[i].packed), groups[i].offset, n};
        a.first[a.count] = a.count * (2304 / 4);
        b.first[b.count] = b.count * (5120 / 4);
        ++a.count;
        ++b.count;
    }
    const bool b = grouped_bf16_inputs.load() && scratch && scratch_bytes >= size_t(tokens) * 5120 * 2 &&
                   (reinterpret_cast<uintptr_t>(scratch) & 15) == 0;
    if (b && grouped_mma.load()) {
        Work mg, md;
        mg.count = md.count = count;
        for (int i = 0; i < count; ++i) {
            mg.g[i] = md.g[i] = {static_cast<const uint8_t*>(groups[i].packed), groups[i].offset, groups[i].tokens};
            mg.first[i] = i * (2304 / 16);
            md.first[i] = i * (5120 / 16);
        }
        mg.first[count] = count * (2304 / 16);
        md.first[count] = count * (5120 / 16);
        const bool wd = grouped_mma_wide.load();
        to_bf16_kernel4<<<(tokens * 5120 / 4 + 255) / 256, 256, 0, stream>>>(x, static_cast<uint16_t*>(scratch), tokens * 5120);
        auto* act = reinterpret_cast<uint16_t*>(activation);
        static const bool ws_device = [] { int d = 0, v = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, d); return v >= 9; }();
        if (ws_device && grouped_ws.load() && !wd && (up_offset & 15) == 0 && (down_offset & 15) == 0) {
            bool aligned = true;
            for (int i = 0; i < count; ++i) aligned &= (reinterpret_cast<uintptr_t>(groups[i].packed) & 15) == 0;
            if (aligned) {
                WsWork ws;
                ws.count = count;
                for (int i = 0; i < count; ++i) ws.g[i] = mg.g[i];
                const auto* xs = static_cast<const uint16_t*>(scratch);
                cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
                if (grouped_ws_fused.load() && cudaStreamIsCapturing(stream, &capture) == cudaSuccess &&
                    capture == cudaStreamCaptureStatusNone) {
                    bool launched = false;
                    switch (gu_type) {
                        case 16: launched = wsf_dispatch<16, 4, 3>(down_type, ws, up_offset, down_offset, xs, token_rows, routing, slots, act, parts, stream); break;
                        case 17: launched = wsf_dispatch<17, 3, 2>(down_type, ws, up_offset, down_offset, xs, token_rows, routing, slots, act, parts, stream); break;
                        case 22: launched = wsf_dispatch<22, 3, 2>(down_type, ws, up_offset, down_offset, xs, token_rows, routing, slots, act, parts, stream); break;
                    }
                    if (launched) return cudaGetLastError();
                }
                switch (gu_type) {
                    case 16: ws_launch<16, true, 4, 3>(ws, up_offset, 0, xs, token_rows, routing, slots, act, parts, stream); break;
                    case 17: ws_launch<17, true, 3, 2>(ws, up_offset, 0, xs, token_rows, routing, slots, act, parts, stream); break;
                    case 22: ws_launch<22, true, 3, 2>(ws, up_offset, 0, xs, token_rows, routing, slots, act, parts, stream); break;
                }
                switch (down_type) {
                    case 16: ws_launch<16, false, 8, 4, 1>(ws, 0, down_offset, act, token_rows, routing, slots, act, parts, stream); break;
                    case 17: ws_launch<17, false, 7, 4, 1>(ws, 0, down_offset, act, token_rows, routing, slots, act, parts, stream); break;
                    case 18: ws_launch<18, false, 5, 4, 1>(ws, 0, down_offset, act, token_rows, routing, slots, act, parts, stream); break;
                    case 22: ws_launch<22, false, 6, 4, 1>(ws, 0, down_offset, act, token_rows, routing, slots, act, parts, stream); break;
                }
                return cudaGetLastError();
            }
        }
        switch (gu_type) {
#define G(T) case T: if (T == 16 && wd) mma_gu_kernel<T, true><<<mg.first[count], 128, 0, stream>>>(mg, up_offset, static_cast<const uint16_t*>(scratch), token_rows, routing, act); \
                     else mma_gu_kernel<T, false><<<mg.first[count], 128, 0, stream>>>(mg, up_offset, static_cast<const uint16_t*>(scratch), token_rows, routing, act); break;
            G(16) G(17) G(22)
#undef G
        }
        switch (down_type) {
#define D(T) case T: if (T == 16 && wd) mma_down_kernel<T, true><<<md.first[count], 128, 0, stream>>>(md, down_offset, act, slots, parts); \
                     else mma_down_kernel<T, false><<<md.first[count], 128, 0, stream>>>(md, down_offset, act, slots, parts); break;
            D(16) D(17) D(18) D(22)
#undef D
        }
        return cudaGetLastError();
    }
    if (b) to_bf16_kernel4<<<(tokens * 5120 / 4 + 255) / 256, 256, 0, stream>>>(x, static_cast<uint16_t*>(scratch), tokens * 5120);
    const void* gx = b ? scratch : static_cast<const void*>(x);
    for (int n = 1; n <= 8; ++n) {
        if (!gu[n].count) continue;
        gu[n].first[gu[n].count] = gu[n].count * (2304 / 4);
        switch (gu_type) {
#define G(T) case T: if (b) launch_gu<T, true>(n, gu[n], up_offset, gx, token_rows, routing, activation, stream); \
                     else launch_gu<T, false>(n, gu[n], up_offset, gx, token_rows, routing, activation, stream); break;
            G(16) G(17) G(22)
#undef G
        }
    }
    for (int n = 1; n <= 8; ++n) {
        if (!dn[n].count) continue;
        dn[n].first[dn[n].count] = dn[n].count * (5120 / 4);
        switch (down_type) {
#define D(T) case T: if (b) launch_down<T, true>(n, dn[n], down_offset, activation, slots, parts, stream); \
                     else launch_down<T, false>(n, dn[n], down_offset, activation, slots, parts, stream); break;
            D(16) D(17) D(18) D(22)
#undef D
        }
    }
    return cudaGetLastError();
}
} // namespace strata::models::deepseek41
