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
