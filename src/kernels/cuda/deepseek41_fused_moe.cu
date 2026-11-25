// DeepSeek V4.1 prefill routed experts on int8 tensor cores: the load stage, arithmetic and tiling of
// src/prefill/moe_fused_iq.cu (an IQ sub-block decoded to int8 in shared memory, activations as int8 per 32 values,
// mma.sync m16n8k32 / m16n8k16 with the dot read through the 1.5 * 2^23 bias) at this model's geometry: gate/up
// K = 5120 (2 x 2304 rows), down K = 2304 (5120 rows), every gate/up and down format IQ2_XXS / IQ2_XS / IQ2_S /
// IQ3_XXS. The epilogues are DeepSeek's: gate/up rounded to BF16, gate <= 10, |up| <= 10, SiLU times up times the
// route weight, rounded to BF16, then H to int8 per 32 features; down rounded to BF16 into the per-slot rows
// (token * 6 + k) the existing combine reads. Not bit-exact with the BF16 cuBLAS path (int8 activations and H).
#include "strata/models/deepseek41/fused_moe.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <vector>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace strata::models::deepseek41 {
namespace {

constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, {d0, d1, s0, s1}
constexpr int MAGIC = 0x4B400000;
constexpr float MAGICF = 12582912.0f;
constexpr int THREADS = 512;
constexpr int WLD = 80;
constexpr int ASTAGES = 4;
constexpr int TILE_ROWS = 64;
constexpr int K_GU = 5120, K_D = 2304, N_GU = 2 * 2304, N_D = 5120;
__host__ __device__ constexpr int tile_rows(int ww) { return 256 / ww; }
__host__ __device__ constexpr int weight_rows(int ww) { return 64 * ww; }

constexpr int T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ2_S = 22;
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98, "the block layouts this file decodes");
__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : 98;
}
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : 256 * 4;
}
__host__ __device__ constexpr size_t smem_bytes(int t, int ww) {
    return (size_t) 2 * weight_rows(ww) * (WLD + 16) + (size_t) ASTAGES * tile_rows(ww) * AB +
           (size_t) tile_rows(ww) * 4 + grid_bytes(t);
}

struct Tables {
    const int32_t *off, *ts;   // per expert: first row (E + 1), first tile (E + 1)
    const int2* tiles;         // {expert, first row}, in expert order
    const int32_t* ts2;        // BF16 path: first 128-row tile per expert (E + 1)
    const int2* tiles2;        // BF16 path: 128-row tiles {expert, first row}
    int* ctr;                  // BF16 path: this launch's work counter (zeroed by fused_moe_tables)
};

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__global__ void quant_act_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    uint8_t* out = xa + w * AB;
    const int c0 = a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0, c1 = a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0;
    out[lane] = (uint8_t) (int8_t) c0;
    out[32 + lane] = (uint8_t) (int8_t) c1;
    if (lane == 0) *(float4*) (out + 64) = make_float4(a0 / 127.0f, a1 / 127.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(MAGIC));
}
__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC));
}
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
}
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}
// Raw bytes of 32-value sub-block `ib` (0..7) of the 256-value super-block at `bp`.
template <int T> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[4]) {
    if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    }
}
// The sub-block as 32 int8 (natural order) and its scales (s0: values 0-15, s1: 16-31): llama.cpp's d_w.
template <int T>
__device__ __forceinline__ void convert(const uint32_t (&w)[4], const uint8_t* grid, uint32_t (&q)[8], float& s0, float& s1) {
    if constexpr (T == T_IQ2_XXS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
            signed8(e.x, e.y, unpack_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
            signed8(e.x, e.y, unpack_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
            signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    }
}
template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else return iq3xxs_grid;
}

struct Geom {
    size_t up_off, down_off;
    int gu_row, d_row;   // bytes per gate/up row and per down row
};

// One work item = (TR routed rows of expert e, a block of WR weight rows); a persistent grid walks the batch's items.
// GU: gate/up rows of WR / 2 features against the rows' tokens' activations, the SwiGLU epilogue, H as int8 per 32
// features into `out` [row][36 stages]. !GU: down rows against H, BF16-rounded FP32 into parts[slot[row]].
// Warp (wf, wt): weight rows 64 wf.. as four m16 tiles (GU: a tile = 8 features, gate rows as mma rows 0-7 and up
// rows as 8-15, so a lane holds gate and up of one feature), routed rows 16 wt.. as two n8.
template <int WT, bool GU, int WW>
__global__ void __launch_bounds__(THREADS, 1)
fused_kernel(const FusedMoeBatch b, const Geom geo, const Tables tb, const uint8_t* __restrict__ act,
             const int32_t* __restrict__ src, const float* __restrict__ rw, const int32_t* __restrict__ dst,
             uint8_t* __restrict__ out, float* __restrict__ parts) {
    constexpr int TR = tile_rows(WW), WR = weight_rows(WW);
    constexpr int WT_BYTES = WR * WLD, WS_FLOATS = WR * 4, ACT_STAGE = TR * AB;
    constexpr int K = GU ? K_GU : K_D;
    constexpr int NS = K / 64;
    constexpr int NFB = (GU ? N_GU : N_D) / WR;
    constexpr int ACT_LD = NS * AB;
    constexpr int H_LD = (K_D / 64) * AB;
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wt = smem;
    float* ws = (float*) (smem + 2 * WT_BYTES);
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);
    int* srow = (int*) (stages + ASTAGES * ACT_STAGE);
    uint8_t* sgrid = (uint8_t*) (srow + TR);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLD + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    const bool dec = tid < 2 * WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        if ((row0 - tb.off[e]) % TR != 0) continue;
        const uint8_t* blob = b.blob[e - b.e0];
        if (!blob) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        const int rbase = fb * WR;
        const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) +
                                       (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        auto unit = [&](int s) -> const uint8_t* { return wrow + (s >> 2) * BS; };
        auto sub = [&](int s) { return 2 * (s & 3) + uj; };
        constexpr int PF = 2;
        auto prefetch_sb = [&](int sb) {
            if (dec && sb < K / 256) pf_l2(wrow + sb * BS + (uj ? BS - 1 : 0));
        };
#pragma unroll
        for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
        __syncthreads();
        if (tid < TR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;
        __syncthreads();
        auto load_act = [&](int s) {
            uint8_t* st = stages + (s % ASTAGES) * ACT_STAGE;
            for (int c = tid; c < TR * 5; c += THREADS) {
                const int r = c / 5, q = c % 5;
                if (r < nrows) cp16(st + r * AB + q * 16, act + (size_t) srow[r] * (GU ? ACT_LD : H_LD) + s * AB + q * 16);
            }
        };
        auto put = [&](const uint32_t (&raw)[4], int buf) {
            if (!dec) return;
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, q, s0, s1);
            uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLD + 32 * uj);
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) (ws + buf * WS_FLOATS + ur * 4 + 2 * uj) = make_float2(s0, s1);
        };
#pragma unroll
        for (int s = 0; s < ASTAGES - 1; ++s) {
            if (s < NS) load_act(s);
            cp_commit();
        }
        uint32_t raw[4] = {0, 0, 0, 0};
        if (dec) load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (dec) load_unit<WT>(unit(1), sub(1), raw);
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int s = 0; s < NS; ++s) {
            cp_wait<ASTAGES - 2>();
            __syncthreads();
            if (s + ASTAGES - 1 < NS) load_act(s + ASTAGES - 1);
            cp_commit();
            if ((s & 3) == 0) prefetch_sb((s >> 2) + PF);
            if (on0) {
                const uint8_t* W = wt + (s & 1) * WT_BYTES;
                const float* S = ws + (s & 1) * WS_FLOATS;
                const uint8_t* sa = stages + (s % ASTAGES) * ACT_STAGE;
                float2 dx[2][2];
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) dx[n][cc] = *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 64);
if constexpr (K16) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t bq[4];
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a[4];
                        ldsm4(a, W + a_off + 16 * i * WLD + 32 * h);
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
                        const float wa0 = swa.x, wa1 = swa.y, wb0 = swb.x, wb1 = swb.y;
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            int d0[4], d1[4];
                            mma16(d0, a[0], a[1], bq[2 * n]);
                            mma16(d1, a[2], a[3], bq[2 * n + 1]);
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const float v = fmaf(q < 2 ? wa1 : wb1, dotf(d1[q]), (q < 2 ? wa0 : wb0) * dotf(d0[q]));
                                acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                            }
                        }
                    }
                }
} else {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t bq[4];
                    ldsm4(bq, sa + b_off + 32 * h);
                    uint32_t a[4][4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) ldsm4(a[i], W + a_off + 16 * i * WLD + 32 * h);
                    // All products of the half-stage first, then their scales: the mma latency overlaps.
                    int d[4][2][2][4];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            if constexpr (K16) {
                                mma16(d[i][n][0], a[i][0], a[i][1], bq[2 * n]);
                                mma16(d[i][n][1], a[i][2], a[i][3], bq[2 * n + 1]);
                            } else {
                                mma32(d[i][n][0], a[i], bq[2 * n], bq[2 * n + 1]);
                            }
                        }
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
                        const float wa0 = swa.x, wa1 = swa.y, wb0 = swb.x, wb1 = swb.y;
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                if constexpr (K16) {
                                    const float v = fmaf(q < 2 ? wa1 : wb1, dotf(d[i][n][1][q]), (q < 2 ? wa0 : wb0) * dotf(d[i][n][0][q]));
                                    acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                                } else {
                                    const float p = (q < 2 ? wa0 : wb0) * (h ? dx[n][q & 1].y : dx[n][q & 1].x);
                                    acc[i][n][q] = fmaf(p, dotf(d[i][n][0][q]), acc[i][n][q]);
                                }
                            }
                        }
                    }
                }
}
            }
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);
                if (dec && s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
            }
        }
        if (!on0) continue;
        if (GU) {
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    const float route = r < nrows ? rw[row0 + r] : 0.0f;
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float v = fminf(bf16r(acc[i][n][c]), 10.f);
                        const float u = fminf(fmaxf(bf16r(acc[i][n][2 + c]), -10.f), 10.f);
                        const float sig = v >= 0 ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v));
                        h[i] = bf16r(v * sig * u * route);
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * H_LD + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) __float2int_rn(h[i] * inv);
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r >= nrows) continue;
                    float* d = parts + (size_t) dst[row0 + r] * N_D + rbase + 64 * wf + g;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        d[16 * i] = bf16r(acc[i][n][c]);
                        d[16 * i + 8] = bf16r(acc[i][n][2 + c]);
                    }
                }
            }
        }
    }
}

// ---- The same work items on BF16 tensor cores (mma m16n8k16, FP32 accumulate). Weights are dequantized in
// shared memory exactly as dequant_bf16_gpu / expert_mixed_dequant_gpu do (the FP32 product d * scale * grid is
// exact, then one BF16 rounding), activations and H are the BF16 values the cuBLAS path multiplies, so only the
// accumulation order differs from that path. Item: 256 weight rows x 128 routed rows, K in stages of 64; warps
// 4 (rows) x 2 (routed rows), warp tile 64 x 64; two stages (weights decoded by every thread one stage ahead,
// activations by cp.async).
namespace bfp {
constexpr int BK = 64, WR = 256, TR = 128, NFB_GU = N_GU / WR, NFB_D = N_D / WR;
// NW warps: 4 along the weight rows (64 each) x NW / 4 along the routed rows (WTN each); TPR threads decode a row.
constexpr int NW = 16, THR = 32 * NW, WN = NW / 4, WTN = TR / WN, NTN = WTN / 8, NPM = WTN / 16;
constexpr int TPR = THR / WR, CPT = 8 / TPR, CPK = CPT / 4, SBT = CPT / 4;
constexpr int A_BYTES = WR * BK * 2, B_BYTES = TR * BK * 2, STAGE = A_BYTES + B_BYTES;
constexpr size_t SMEM_BASE = (size_t) 2 * STAGE + TR * 4;
__host__ __device__ constexpr bool grid_smem(int t) { return SMEM_BASE + grid_bytes(t) <= 101376; }
__host__ __device__ constexpr size_t smem_bytes(int t) { return SMEM_BASE + (grid_smem(t) ? grid_bytes(t) : 0); }
__device__ __forceinline__ int swz(int row, int ch) { return row * 128 + ((ch ^ (row & 7)) << 4); }
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src),
                 "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
// Eight unsigned grid bytes (lo: values 0-3, hi: 4-7) times the exact FP32 scale s, signs from bits 0-7 of sg,
// rounded to BF16: bf16(s * grid * (+-1)) = the reference dequant. (2^23 + b) * s - 2^23 * s is exact in one FMA.
__device__ __forceinline__ uint4 dq8(uint32_t lo, uint32_t hi, uint32_t sg, float s) {
    const float m = -8388608.0f * s;
    uint32_t o[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t src = p < 2 ? lo : hi;
        const int j = 2 * (p & 1);
        const float f0 = fmaf(__uint_as_float(__byte_perm(src, 0x4B000000u, 0x7540 + j)), s, m);
        const float f1 = fmaf(__uint_as_float(__byte_perm(src, 0x4B000000u, 0x7541 + j)), s, m);
        const uint32_t mask = (sg * ((1u << (15 - 2 * p)) + (1u << (30 - 2 * p)))) & 0x80008000u;
        o[p] = pack_bf16(f0, f1) ^ mask;
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}
// IQ2 grids hold only the bytes 8, 25 and 43: per scale s the three BF16 products are rounded once (exact FP32
// products, as the reference), and a pair of values is one byte permute of them. Selector table entry: per grid
// entry, the four pair selectors (index (b >> 4) & 3 -> bytes 2 i, 2 i + 1 of {bf16(8 s), bf16(25 s) | bf16(43 s), 0}).
__device__ uint2 g_sel_xxs[256], g_sel_xs[512], g_sel_s[1024];
template <int T> __device__ __forceinline__ uint2* sel_dst() {
    if constexpr (T == T_IQ2_XXS) return g_sel_xxs;
    else if constexpr (T == T_IQ2_XS) return g_sel_xs;
    else return g_sel_s;
}
template <int T> __global__ void sel_init_kernel() {
    constexpr int n = grid_bytes(T) / 8;
    const uint64_t* grid = (const uint64_t*) grid_src<T>();
    for (int e = threadIdx.x; e < n; e += blockDim.x) {
        uint32_t sel[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t i0 = ((grid[e] >> (16 * p)) >> 4) & 3, i1 = ((grid[e] >> (16 * p + 8)) >> 4) & 3;
            sel[p] = (2 * i0) | ((2 * i0 + 1) << 4) | ((2 * i1) << 8) | ((2 * i1 + 1) << 12);
        }
        sel_dst<T>()[e] = make_uint2(sel[0] | (sel[1] << 16), sel[2] | (sel[3] << 16));
    }
}
__device__ __forceinline__ uint4 dq8_sel(uint2 sel, uint32_t sg, float s) {
    const uint32_t p01 = pack_bf16(8.0f * s, 25.0f * s), p2 = pack_bf16(43.0f * s, 0.0f);
    uint32_t o[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t sw = p < 2 ? sel.x : sel.y;
        const uint32_t mask = (sg * ((1u << (15 - 2 * p)) + (1u << (30 - 2 * p)))) & 0x80008000u;
        o[p] = __byte_perm(p01, p2, (p & 1) ? sw >> 16 : sw & 0xFFFF) ^ mask;
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}
template <int T> __device__ __forceinline__ uint2 sel8(const uint8_t* grid, uint32_t idx) {
    if constexpr (grid_smem(T)) return reinterpret_cast<const uint2*>(grid)[idx];
    else return __ldg(sel_dst<T>() + idx);
}
template <int T> __device__ __forceinline__ uint32_t grid4(const uint8_t* grid, uint32_t idx) {
    if constexpr (grid_smem(T)) return reinterpret_cast<const uint32_t*>(grid)[idx];
    else return __ldg(reinterpret_cast<const uint32_t*>(grid_src<T>()) + idx);
}
// Values 8 l .. 8 l + 7 of the sub-block whose raw words load_unit<T> returned, as 8 BF16 (one 16-byte chunk).
template <int T> __device__ __forceinline__ uint4 dq_group(const uint32_t (&w)[4], const uint8_t* grid, int l) {
    if constexpr (T == T_IQ2_XXS) {
        const float s = half_at(w[2]) * (0.5f + (float) (w[1] >> 28)) * 0.25f;
        return dq8_sel(sel8<T>(grid, (w[0] >> (8 * l)) & 255), unpack_ksigns(w[1] >> (7 * l)) & 255, s);
    } else if constexpr (T == T_IQ2_XS) {
        const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
        const float s = half_at(w[2]) * (0.5f + (float) ((w[2] >> (16 + 4 * (l >> 1))) & 15)) * 0.25f;
        return dq8_sel(sel8<T>(grid, c & 511), unpack_ksigns(c >> 9) & 255, s);
    } else if constexpr (T == T_IQ2_S) {
        const uint32_t qh = (w[2] >> 16) & 255;
        const float s = half_at(w[2]) * (0.5f + (float) ((w[2] >> (24 + 4 * (l >> 1))) & 15)) * 0.25f;
        const uint32_t gi = ((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300);
        return dq8_sel(sel8<T>(grid, gi), (w[1] >> (8 * l)) & 255, s);
    } else {
        const float s = half_at(w[3]) * (0.5f + (float) (w[2] >> 28)) * 0.5f;
        const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
        return dq8(grid4<T>(grid, i0), grid4<T>(grid, i1), unpack_ksigns(w[2] >> (7 * l)) & 255, s);
    }
}

// The raw 16/8-bit loads of one 32-value sub-block, kept apart until the decode combines them (combining at load
// time would wait for the loads right there).
struct RawH { uint32_t h[7]; };
template <int T> __device__ __forceinline__ void load_halves(const uint8_t* bp, int ib, RawH& r) {
    if constexpr (T == T_IQ2_XXS || T == T_IQ2_XS) {
#pragma unroll
        for (int i = 0; i < 4; ++i) r.h[i] = ld16(bp + 2 + 8 * ib + 2 * i);
        r.h[4] = ld16(bp);
        if constexpr (T == T_IQ2_XS) r.h[5] = bp[66 + ib];
    } else if constexpr (T == T_IQ2_S) {
        r.h[0] = ld16(bp + 2 + 4 * ib); r.h[1] = ld16(bp + 4 + 4 * ib);
        r.h[2] = ld16(bp + 34 + 4 * ib); r.h[3] = ld16(bp + 36 + 4 * ib);
        r.h[4] = ld16(bp); r.h[5] = bp[66 + ib]; r.h[6] = bp[74 + ib];
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) r.h[i] = ld16(bp + 2 + 8 * ib + 2 * i);
        r.h[4] = ld16(bp + 66 + 4 * ib); r.h[5] = ld16(bp + 68 + 4 * ib); r.h[6] = ld16(bp);
    }
}
// The words load_unit<T> would have returned.
template <int T> __device__ __forceinline__ void words(const RawH& r, uint32_t (&w)[4]) {
    w[0] = r.h[0] | (r.h[1] << 16);
    w[1] = r.h[2] | (r.h[3] << 16);
    if constexpr (T == T_IQ2_XXS) { w[2] = r.h[4]; w[3] = 0; }
    else if constexpr (T == T_IQ2_XS) { w[2] = r.h[4] | (r.h[5] << 16); w[3] = 0; }
    else if constexpr (T == T_IQ2_S) { w[2] = r.h[4] | (r.h[5] << 16) | (r.h[6] << 24); w[3] = 0; }
    else { w[2] = r.h[4] | (r.h[5] << 16); w[3] = r.h[6]; }
}

template <int T, bool GU, int NP>
__device__ __forceinline__ void kloop(float (&acc)[4][NTN][4], uint8_t* smem, const uint8_t* sgrid, const int* srow,
                                      const uint8_t* wrow, const __nv_bfloat16* __restrict__ act, int tid, int lane,
                                      int wm, int wn) {
    constexpr int K = GU ? K_GU : K_D, NK = K / BK, BS = block_bytes(T);
    static_assert(NK >= 4 && NK % 2 == 0, "stage pairs");
    const int row = tid / TPR, cbase = (tid % TPR) * CPT;   // this thread decodes chunks cbase .. cbase + CPT - 1
    RawH raw[2][SBT];   // [buffer][sub-block]: stage kb + 1 in buffer (kb + 1) & 1
    auto load_raw = [&](int kb, RawH (&r)[SBT]) {
        const uint8_t* bp = wrow + (kb >> 2) * BS;
#pragma unroll
        for (int q = 0; q < SBT; ++q) load_halves<T>(bp, 2 * (kb & 3) + (cbase >> 2) + q, r[q]);
        if ((kb & 3) == 0 && (kb >> 2) + 2 < K / 256 && (tid % TPR) == 0) pf_l2(bp + 2 * BS);
    };
    constexpr int BCH = TR * 8 / THR;
    int boff[BCH];
#pragma unroll
    for (int j = 0; j < BCH; ++j) {
        const int tok = srow[(tid >> 3) + (THR / 8) * j];
        boff[j] = tok < 0 ? -1 : tok * K + (tid & 7) * 8;
    }
    auto load_b = [&](int kb) {
        uint8_t* sb = smem + (kb & 1) * STAGE + A_BYTES;
#pragma unroll
        for (int j = 0; j < BCH; ++j)
            cp16z(sb + swz((tid >> 3) + (THR / 8) * j, tid & 7), act + (boff[j] < 0 ? 0 : boff[j] + kb * BK), boff[j] >= 0);
    };
    auto put = [&](int kb, int part, const RawH (&r)[SBT]) {   // part 0..3 of this thread's chunks
        uint8_t* sa = smem + (kb & 1) * STAGE;
#pragma unroll
        for (int j = 0; j < CPK; ++j) {
            const int c = cbase + part * CPK + j;
            uint32_t w[4];
            words<T>(r[(c - cbase) >> 2], w);
            *reinterpret_cast<uint4*>(sa + swz(row, c)) = dq_group<T>(w, sgrid, c & 3);
        }
    };
    auto stage = [&](int kb, auto next_tag, RawH (&cur)[SBT]) {
        constexpr bool kNext = decltype(next_tag)::value;
        cp_wait<0>();
        __syncthreads();
        if (kNext) load_b(kb + 1);
        cp_commit();
        const uint8_t* sa = smem + (kb & 1) * STAGE;
        const uint8_t* sb = sa + A_BYTES;
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            if constexpr (NP > 0) {
                uint32_t a[4][4], bq[NP][4];
#pragma unroll
                for (int i = 0; i < 4; ++i) ldsm4(a[i], sa + swz(wm * 64 + i * 16 + (lane & 15), kk * 2 + (lane >> 4)));
#pragma unroll
                for (int j = 0; j < NP; ++j)
                    ldsm4(bq[j], sb + swz(wn * WTN + j * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1)));
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int n = 0; n < 2 * NP; ++n) mma_bf16(acc[i][n], a[i], bq[n >> 1][(n & 1) * 2], bq[n >> 1][(n & 1) * 2 + 1]);
            }
            if (kNext) put(kb + 1, kk, cur);
        }
        if (kb + 3 < NK) load_raw(kb + 3, cur);
    };
    load_raw(0, raw[0]);
    load_b(0);
    cp_commit();
#pragma unroll
    for (int p = 0; p < 4; ++p) put(0, p, raw[0]);
    load_raw(1, raw[1]);
    load_raw(2, raw[0]);
    for (int kb = 0; kb < NK - 2; kb += 2) {
        stage(kb, std::true_type{}, raw[1]);
        stage(kb + 1, std::true_type{}, raw[0]);
    }
    stage(NK - 2, std::true_type{}, raw[1]);
    stage(NK - 1, std::false_type{}, raw[0]);
}

template <int T, bool GU>
__global__ void __launch_bounds__(THR, 1)
bf16_kernel(const FusedMoeBatch b, const Geom geo, const Tables tb, const __nv_bfloat16* __restrict__ act,
            const int32_t* __restrict__ src, const float* __restrict__ rw, const int32_t* __restrict__ dst,
            __nv_bfloat16* __restrict__ hout, float* __restrict__ parts) {
    constexpr int NFB = GU ? NFB_GU : NFB_D;
    extern __shared__ __align__(128) uint8_t smem[];
    int* srow = reinterpret_cast<int*>(smem + 2 * STAGE);
    uint8_t* sgrid = smem + SMEM_BASE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp & 3, wn = warp >> 2;
    const int g = lane >> 2, tig = lane & 3;
    if constexpr (grid_smem(T)) {
        const uint32_t* gs = T != T_IQ3_XXS ? (const uint32_t*) sel_dst<T == T_IQ3_XXS ? T_IQ2_XXS : T>() : (const uint32_t*) grid_src<T>();
        for (int i = tid; i < grid_bytes(T) / 4; i += THR) ((uint32_t*) sgrid)[i] = gs[i];
    }
    const int t0 = tb.ts2[b.e0], nwork = (tb.ts2[b.e1] - t0) * NFB;
    __shared__ int next_item;
    for (;;) {
        __syncthreads();
        if (tid == 0) next_item = atomicAdd(tb.ctr, 1);
        __syncthreads();
        const int w = next_item;
        if (w >= nwork) break;
        const int2 tl = tb.tiles2[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        const uint8_t* blob = b.blob[e - b.e0];
        if (!blob) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        if (tid < TR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;
        __syncthreads();
        // Thread tid decodes weight row tid of the item. GU: 16-row groups = 8 gate rows then the same 8 up rows.
        const int dr = tid / TPR;
        const uint8_t* wrow = GU ? blob + ((dr & 8) ? geo.up_off : 0) +
                                       (size_t) (fb * (WR / 2) + 8 * (dr >> 4) + (dr & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (fb * WR + dr) * geo.d_row;
        float acc[4][NTN][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < NTN; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        const int np = max(0, min(NPM, (nrows - WTN * wn + 15) >> 4));
        switch (np) {
            case 2: kloop<T, GU, 2>(acc, smem, sgrid, srow, wrow, act, tid, lane, wm, wn); break;
            case 1: kloop<T, GU, 1>(acc, smem, sgrid, srow, wrow, act, tid, lane, wm, wn); break;
            default: kloop<T, GU, 0>(acc, smem, sgrid, srow, wrow, act, tid, lane, wm, wn); break;
        }
        if (GU) {
#pragma unroll
            for (int n = 0; n < NTN; ++n)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = WTN * wn + 8 * n + 2 * tig + c;
                    if (r >= nrows) continue;
                    const float route = rw[row0 + r];
                    __nv_bfloat16* o = hout + (size_t) (row0 + r) * K_D + fb * (WR / 2) + wm * 32 + g;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float v = fminf(bf16r(acc[i][n][c]), 10.f);
                        const float u = fminf(fmaxf(bf16r(acc[i][n][2 + c]), -10.f), 10.f);
                        const float sig = v >= 0 ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v));
                        o[8 * i] = __float2bfloat16_rn(v * sig * u * route);
                    }
                }
        } else {
#pragma unroll
            for (int n = 0; n < NTN; ++n)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = WTN * wn + 8 * n + 2 * tig + c;
                    if (r >= nrows) continue;
                    float* d = parts + (size_t) dst[row0 + r] * N_D + fb * WR + wm * 64 + g;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        d[16 * i] = bf16r(acc[i][n][c]);
                        d[16 * i + 8] = bf16r(acc[i][n][2 + c]);
                    }
                }
        }
    }
}

__global__ void to_bf16_rows_kernel(const float* __restrict__ x, int64_t n, __nv_bfloat16* __restrict__ y) {
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const float4 v = *reinterpret_cast<const float4*>(x + i);
    *reinterpret_cast<uint2*>(y + i) = make_uint2(pack_bf16(v.x, v.y), pack_bf16(v.z, v.w));
}
}  // namespace bfp

// ---- int8 v2: the same arithmetic and accumulation order as fused_kernel (outputs byte-identical), with the
// stalls the BF16 work found removed: active n8 tiles dispatched outside the K loop (no per-MMA branches), raw weight
// halves combined at decode time instead of at load time, activation copy offsets computed once per item, and items
// claimed from an atomic counter over tiles of the item's own row count.
namespace i8v2 {
template <int WT, bool GU, int WW> struct Cfg {
    static constexpr int TR = tile_rows(WW), WR = weight_rows(WW);
    static constexpr int WT_BYTES = WR * WLD, WS_FLOATS = WR * 4, ACT_STAGE = TR * AB;
    static constexpr int K = GU ? K_GU : K_D, NS = K / 64, NFB = (GU ? N_GU : N_D) / WR;
    static constexpr int ACT_LD = NS * AB, H_LD = (K_D / 64) * AB, LD = GU ? ACT_LD : H_LD;
    static constexpr int BS = block_bytes(WT);
    static constexpr bool K16 = per16(WT);
    static constexpr int ACH = (TR * 5 + THREADS - 1) / THREADS;
};
template <int WT, bool GU, int WW, int NB>
__device__ __forceinline__ void kloop(float (&acc)[4][2][4], uint8_t* smem, const uint8_t* sgrid, const uint8_t* wrow,
                                      const uint8_t* __restrict__ act, const int (&aoff)[Cfg<WT, GU, WW>::ACH],
                                      const int (&adst)[Cfg<WT, GU, WW>::ACH], int tid, int lane, int wf, int nb0) {
    using C = Cfg<WT, GU, WW>;
    constexpr int WT_BYTES = C::WT_BYTES, WS_FLOATS = C::WS_FLOATS, ACT_STAGE = C::ACT_STAGE, NS = C::NS, BS = C::BS, K = C::K;
    uint8_t* wt = smem;
    float* ws = (float*) (smem + 2 * WT_BYTES);
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);
    const int g = lane >> 2, tig = lane & 3, lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLD + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    const bool dec = tid < 2 * C::WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    auto prefetch_sb = [&](int sb) {
        if (dec && sb < K / 256) pf_l2(wrow + sb * BS + (uj ? BS - 1 : 0));
    };
    constexpr int PF = 2;
#pragma unroll
    for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
    auto load_act = [&](int s) {
        uint8_t* st = stages + (s % ASTAGES) * ACT_STAGE;
#pragma unroll
        for (int j = 0; j < C::ACH; ++j)
            if (aoff[j] >= 0) cp16(st + adst[j], act + aoff[j] + s * AB);
    };
    auto load = [&](int s, bfp::RawH& r) {
        if (dec) bfp::load_halves<WT>(wrow + (s >> 2) * BS, 2 * (s & 3) + uj, r);
    };
    auto put = [&](const bfp::RawH& r, int buf) {
        if (!dec) return;
        uint32_t raw[4], q[8];
        bfp::words<WT>(r, raw);
        float s0, s1;
        convert<WT>(raw, sgrid, q, s0, s1);
        uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLD + 32 * uj);
        d[0] = make_uint4(q[0], q[1], q[2], q[3]);
        d[1] = make_uint4(q[4], q[5], q[6], q[7]);
        *(float2*) (ws + buf * WS_FLOATS + ur * 4 + 2 * uj) = make_float2(s0, s1);
    };
#pragma unroll
    for (int s = 0; s < ASTAGES - 1; ++s) {
        if (s < NS) load_act(s);
        cp_commit();
    }
    bfp::RawH raw{};
    load(0, raw);
    put(raw, 0);
    load(1, raw);
    for (int s = 0; s < NS; ++s) {
        cp_wait<ASTAGES - 2>();
        __syncthreads();
        if (s + ASTAGES - 1 < NS) load_act(s + ASTAGES - 1);
        cp_commit();
        if ((s & 3) == 0) prefetch_sb((s >> 2) + PF);
        if constexpr (NB > 0) {
            const uint8_t* W = wt + (s & 1) * WT_BYTES;
            const float* S = ws + (s & 1) * WS_FLOATS;
            const uint8_t* sa = stages + (s % ASTAGES) * ACT_STAGE;
            float2 dx[NB][2];
#pragma unroll
            for (int n = 0; n < NB; ++n)
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) dx[n][cc] = *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 64);
            if constexpr (C::K16) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t bq[4];
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a[4];
                        ldsm4(a, W + a_off + 16 * i * WLD + 32 * h);
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
                        const float wa0 = swa.x, wa1 = swa.y, wb0 = swb.x, wb1 = swb.y;
#pragma unroll
                        for (int n = 0; n < NB; ++n) {
                            int d0[4], d1[4];
                            mma16(d0, a[0], a[1], bq[2 * n]);
                            mma16(d1, a[2], a[3], bq[2 * n + 1]);
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const float v = fmaf(q < 2 ? wa1 : wb1, dotf(d1[q]), (q < 2 ? wa0 : wb0) * dotf(d0[q]));
                                acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                            }
                        }
                    }
                }
            } else {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t bq[4];
                    ldsm4(bq, sa + b_off + 32 * h);
                    uint32_t a[4][4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) ldsm4(a[i], W + a_off + 16 * i * WLD + 32 * h);
                    int d[4][NB][4];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int n = 0; n < NB; ++n) mma32(d[i][n], a[i], bq[2 * n], bq[2 * n + 1]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
#pragma unroll
                        for (int n = 0; n < NB; ++n)
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const float p = (q < 2 ? swa.x : swb.x) * (h ? dx[n][q & 1].y : dx[n][q & 1].x);
                                acc[i][n][q] = fmaf(p, dotf(d[i][n][q]), acc[i][n][q]);
                            }
                    }
                }
            }
        }
        if (s + 1 < NS) {
            put(raw, (s + 1) & 1);
            if (s + 2 < NS) load(s + 2, raw);
        }
    }
}

template <int WT, bool GU, int WW>
__global__ void __launch_bounds__(THREADS, 1)
fused2_kernel(const FusedMoeBatch b, const Geom geo, const Tables tb, const uint8_t* __restrict__ act,
              const int32_t* __restrict__ src, const float* __restrict__ rw, const int32_t* __restrict__ dst,
              uint8_t* __restrict__ out, float* __restrict__ parts) {
    using C = Cfg<WT, GU, WW>;
    constexpr int TR = C::TR, WR = C::WR, NFB = C::NFB, H_LD = C::H_LD;
    extern __shared__ __align__(16) uint8_t smem[];
    int* srow = (int*) (smem + 2 * C::WT_BYTES + 2 * C::WS_FLOATS * 4 + ASTAGES * C::ACT_STAGE);
    uint8_t* sgrid = (uint8_t*) (srow + TR);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    const int32_t* ts = TR == TILE_ROWS ? tb.ts : tb.ts2;
    const int2* tiles = TR == TILE_ROWS ? tb.tiles : tb.tiles2;
    const int t0 = ts[b.e0], nwork = (ts[b.e1] - t0) * NFB;
    __shared__ int next_item;
    for (;;) {
        __syncthreads();
        if (tid == 0) next_item = atomicAdd(tb.ctr, 1);
        __syncthreads();
        const int w = next_item;
        if (w >= nwork) break;
        const int2 tl = tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        const uint8_t* blob = b.blob[e - b.e0];
        if (!blob) continue;
        const int nrows = min(TR, tb.off[e + 1] - row0);
        const int rbase = fb * WR;
        if (tid < TR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;
        __syncthreads();
        const int ur = tid < 2 * WR ? tid >> 1 : 0;
        const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) +
                                       (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        int aoff[C::ACH], adst[C::ACH];
#pragma unroll
        for (int j = 0; j < C::ACH; ++j) {
            const int c = tid + THREADS * j, r = c / 5, q = c % 5;
            const bool on = c < TR * 5 && r < nrows;
            aoff[j] = on ? srow[r] * C::LD + q * 16 : -1;
            adst[j] = on ? r * AB + q * 16 : 0;
        }
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        const int nb = on0 + on1;
        if (nb == 2) kloop<WT, GU, WW, 2>(acc, smem, sgrid, wrow, act, aoff, adst, tid, lane, wf, nb0);
        else if (nb == 1) kloop<WT, GU, WW, 1>(acc, smem, sgrid, wrow, act, aoff, adst, tid, lane, wf, nb0);
        else kloop<WT, GU, WW, 0>(acc, smem, sgrid, wrow, act, aoff, adst, tid, lane, wf, nb0);
        if (!on0) continue;
        if (GU) {
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    const float route = r < nrows ? rw[row0 + r] : 0.0f;
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float v = fminf(bf16r(acc[i][n][c]), 10.f);
                        const float u = fminf(fmaxf(bf16r(acc[i][n][2 + c]), -10.f), 10.f);
                        const float sig = v >= 0 ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v));
                        h[i] = bf16r(v * sig * u * route);
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * H_LD + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) __float2int_rn(h[i] * inv);
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r >= nrows) continue;
                    float* d = parts + (size_t) dst[row0 + r] * N_D + rbase + 64 * wf + g;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        d[16 * i] = bf16r(acc[i][n][c]);
                        d[16 * i + 8] = bf16r(acc[i][n][2 + c]);
                    }
                }
            }
        }
    }
}
}  // namespace i8v2

struct DevState {
    bool done = false, ok = false, ok_bf16 = false;
    int sms = 0, occ = 1;
    int32_t* tables = nullptr;     // device: off[E+1], ts[E+1], tiles
    int32_t* host = nullptr;       // pinned staging
    int next_ctr = 0;              // BF16 launches since the last counter reset
};
std::mutex g_mu;
DevState g_dev[16];
constexpr int MAX_E = 384, MAX_TILES = 8192 * 6 / TILE_ROWS + MAX_E, MAX_TILES2 = 8192 * 6 / bfp::TR + MAX_E, NCTR = 1024;
constexpr size_t TS2_WORD = 2 * (MAX_E + 1) + 2 * MAX_TILES, TILES2_WORD = (TS2_WORD + (MAX_E + 1) + 1) / 2 * 2,
                 TABLE_WORDS = TILES2_WORD + 2 * MAX_TILES2, CTR_WORD = (TABLE_WORDS + 3) / 4 * 4;

template <int T, bool GU, int WW> bool setup_ww(int& occ) {
    if (cudaFuncSetAttribute(fused_kernel<T, GU, WW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes(T, WW)) !=
        cudaSuccess)
        return false;
    if (cudaFuncSetAttribute(i8v2::fused2_kernel<T, GU, WW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_bytes(T, WW)) !=
        cudaSuccess)
        return false;
    int o = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, fused_kernel<T, GU, WW>, THREADS, smem_bytes(T, WW)) != cudaSuccess || o < 1)
        return false;
    occ = std::min(occ, o);
    return true;
}
template <int T, bool GU> bool setup_bf16() {
    if (cudaFuncSetAttribute(bfp::bf16_kernel<T, GU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bfp::smem_bytes(T)) !=
        cudaSuccess)
        return false;
    int o = 0;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, bfp::bf16_kernel<T, GU>, bfp::THR, bfp::smem_bytes(T)) == cudaSuccess && o >= 1;
}
template <int T> bool setup_bf16_type() { return setup_bf16<T, true>() && setup_bf16<T, false>(); }
template <int T> bool setup_type(int& occ) {
    return setup_ww<T, true, 4>(occ) && setup_ww<T, true, 2>(occ) && setup_ww<T, false, 4>(occ) && setup_ww<T, false, 2>(occ);
}
DevState& state() {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevState& d = g_dev[dev & 15];
    if (d.done) return d;
    d.done = true;
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    if (major < 8) return d;
    int occ = 1 << 20;
    d.ok = setup_type<T_IQ2_XXS>(occ) && setup_type<T_IQ2_XS>(occ) && setup_type<T_IQ2_S>(occ) && setup_type<T_IQ3_XXS>(occ) &&
           cudaMalloc(&d.tables, (CTR_WORD + NCTR) * 4) == cudaSuccess && cudaMallocHost(&d.host, TABLE_WORDS * 4) == cudaSuccess;
    d.occ = d.ok ? occ : 1;
    d.ok_bf16 = d.ok && setup_bf16_type<T_IQ2_XXS>() && setup_bf16_type<T_IQ2_XS>() && setup_bf16_type<T_IQ2_S>() &&
                setup_bf16_type<T_IQ3_XXS>();
    if (d.ok_bf16) {
        bfp::sel_init_kernel<T_IQ2_XXS><<<1, 256>>>();
        bfp::sel_init_kernel<T_IQ2_XS><<<1, 256>>>();
        bfp::sel_init_kernel<T_IQ2_S><<<1, 256>>>();
        d.ok_bf16 = cudaStreamSynchronize(nullptr) == cudaSuccess;
    }
    cudaGetLastError();
    return d;
}
bool covered(uint32_t t) { return t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS; }
std::atomic<int> forced_ww{0};
std::atomic<bool> bf16_mode{false};
bool use_bf16() { return bf16_mode.load() && state().ok_bf16; }
template <bool GU>
void launch_bf16(uint32_t t, unsigned grid, const FusedMoeBatch& b, const Geom& g, const Tables& tb, const __nv_bfloat16* act,
                 const int32_t* src, const float* rw, const int32_t* dst, __nv_bfloat16* h, float* parts, cudaStream_t s) {
    switch (t) {
#define BF(T) case T: bfp::bf16_kernel<T, GU><<<grid, bfp::THR, bfp::smem_bytes(T), s>>>(b, g, tb, act, src, rw, dst, h, parts); break
        BF(T_IQ2_XXS); BF(T_IQ2_XS); BF(T_IQ2_S); default: BF(T_IQ3_XXS);
#undef BF
    }
}

std::atomic<bool> int8_v2{false};
template <int T, bool GU>
void launch(int ww, unsigned grid, const FusedMoeBatch& b, const Geom& g, const Tables& tb, const uint8_t* act,
            const int32_t* src, const float* rw, const int32_t* dst, uint8_t* out, float* parts, cudaStream_t s) {
    if (tb.ctr) {
        if (ww == 4) i8v2::fused2_kernel<T, GU, 4><<<grid, THREADS, smem_bytes(T, 4), s>>>(b, g, tb, act, src, rw, dst, out, parts);
        else i8v2::fused2_kernel<T, GU, 2><<<grid, THREADS, smem_bytes(T, 2), s>>>(b, g, tb, act, src, rw, dst, out, parts);
        return;
    }
    if (ww == 4) fused_kernel<T, GU, 4><<<grid, THREADS, smem_bytes(T, 4), s>>>(b, g, tb, act, src, rw, dst, out, parts);
    else fused_kernel<T, GU, 2><<<grid, THREADS, smem_bytes(T, 2), s>>>(b, g, tb, act, src, rw, dst, out, parts);
}
template <bool GU>
void launch_type(uint32_t t, int ww, unsigned grid, const FusedMoeBatch& b, const Geom& g, const Tables& tb, const uint8_t* act,
                 const int32_t* src, const float* rw, const int32_t* dst, uint8_t* out, float* parts, cudaStream_t s) {
    switch (t) {
        case T_IQ2_XXS: launch<T_IQ2_XXS, GU>(ww, grid, b, g, tb, act, src, rw, dst, out, parts, s); break;
        case T_IQ2_XS: launch<T_IQ2_XS, GU>(ww, grid, b, g, tb, act, src, rw, dst, out, parts, s); break;
        case T_IQ2_S: launch<T_IQ2_S, GU>(ww, grid, b, g, tb, act, src, rw, dst, out, parts, s); break;
        default: launch<T_IQ3_XXS, GU>(ww, grid, b, g, tb, act, src, rw, dst, out, parts, s); break;
    }
}

}  // namespace

bool fused_moe_supported(uint32_t gu_type, uint32_t d_type) { return covered(gu_type) && covered(d_type) && state().ok; }
size_t fused_moe_act_bytes(int64_t rows) { return use_bf16() ? (size_t) rows * K_GU * 2 : (size_t) rows * (K_GU / 64) * AB; }
size_t fused_moe_h_bytes(int64_t rows) { return use_bf16() ? (size_t) rows * K_D * 2 : (size_t) rows * (K_D / 64) * AB; }
void fused_moe_set_bf16(bool enabled) { bf16_mode.store(enabled); }
bool fused_moe_bf16() { return use_bf16(); }
void fused_moe_set_tile(int rows) { forced_ww.store(rows == 64 ? 4 : rows == 128 ? 2 : 0); }
void fused_moe_set_int8_v2(bool enabled) { int8_v2.store(enabled); }

cudaError_t fused_moe_quantize(const float* x, int tokens, void* xa, cudaStream_t stream) {
    if (tokens <= 0) return cudaSuccess;
    if (use_bf16()) {
        const int64_t n = (int64_t) tokens * K_GU;
        bfp::to_bf16_rows_kernel<<<(unsigned) ((n / 4 + 255) / 256), 256, 0, stream>>>(x, n, (__nv_bfloat16*) xa);
        return cudaGetLastError();
    }
    const int64_t nblk = (int64_t) tokens * (K_GU / 64);
    quant_act_kernel<<<(unsigned) ((nblk + 7) / 8), 256, 0, stream>>>(x, nblk, (uint8_t*) xa);
    return cudaGetLastError();
}

cudaError_t fused_moe_tables(const int* offsets, int experts, cudaStream_t stream) {
    DevState& d = state();
    if (!d.ok || experts < 1 || experts > MAX_E) return cudaErrorInvalidValue;
    int32_t* off = d.host;
    int32_t* ts = off + (MAX_E + 1);
    int2* tiles = reinterpret_cast<int2*>(ts + (MAX_E + 1));
    int nt = 0;
    for (int e = 0; e < experts; ++e) {
        off[e] = offsets[e];
        ts[e] = nt;
        for (int r = offsets[e]; r < offsets[e + 1]; r += TILE_ROWS) {
            if (nt == MAX_TILES) return cudaErrorInvalidValue;
            tiles[nt++] = make_int2(e, r);
        }
    }
    off[experts] = offsets[experts];
    ts[experts] = nt;
    int32_t* ts2 = d.host + TS2_WORD;
    int2* tiles2 = reinterpret_cast<int2*>(d.host + TILES2_WORD);
    int nt2 = 0;
    for (int e = 0; e < experts; ++e) {
        ts2[e] = nt2;
        for (int r = offsets[e]; r < offsets[e + 1]; r += bfp::TR) {
            if (nt2 == MAX_TILES2) return cudaErrorInvalidValue;
            tiles2[nt2++] = make_int2(e, r);
        }
    }
    ts2[experts] = nt2;
    d.next_ctr = 0;
    const cudaError_t st = cudaMemsetAsync(d.tables + CTR_WORD, 0, NCTR * 4, stream);
    if (st != cudaSuccess) return st;
    return cudaMemcpyAsync(d.tables, d.host, TABLE_WORDS * 4, cudaMemcpyHostToDevice, stream);
}

cudaError_t fused_moe_run(const FusedMoeBatch& b, uint32_t gu_type, uint32_t d_type, size_t up_offset, size_t down_offset,
                          int64_t rows, const void* xa, const int32_t* src, const float* rw, const int32_t* dst, void* ha,
                          float* parts, cudaStream_t stream) {
    DevState& d = state();
    if (!d.ok || !covered(gu_type) || !covered(d_type) || b.e1 <= b.e0 || b.e1 - b.e0 > kFusedMoeBatch) return cudaErrorInvalidValue;
    if (rows <= 0) return cudaSuccess;
    Tables tb{d.tables, d.tables + (MAX_E + 1), reinterpret_cast<const int2*>(d.tables + 2 * (MAX_E + 1)), d.tables + TS2_WORD,
              reinterpret_cast<const int2*>(d.tables + TILES2_WORD), nullptr};
    const Geom g{up_offset, down_offset, (K_GU / 256) * block_bytes(gu_type), (K_D / 256) * block_bytes(d_type)};
    if (use_bf16()) {
        if (d.next_ctr + 2 > NCTR) {
            const cudaError_t st = cudaMemsetAsync(d.tables + CTR_WORD, 0, NCTR * 4, stream);
            if (st != cudaSuccess) return st;
            d.next_ctr = 0;
        }
        const int64_t items = (rows + bfp::TR - 1) / bfp::TR + (b.e1 - b.e0);
        const unsigned g_gu = (unsigned) std::min<int64_t>(items * bfp::NFB_GU, d.sms), g_d = (unsigned) std::min<int64_t>(items * bfp::NFB_D, d.sms);
        tb.ctr = d.tables + CTR_WORD + d.next_ctr++;
        launch_bf16<true>(gu_type, g_gu, b, g, tb, (const __nv_bfloat16*) xa, src, rw, dst, (__nv_bfloat16*) ha, parts, stream);
        tb.ctr = d.tables + CTR_WORD + d.next_ctr++;
        launch_bf16<false>(d_type, g_d, b, g, tb, (const __nv_bfloat16*) ha, src, rw, dst, nullptr, parts, stream);
        return cudaGetLastError();
    }
    int ww = forced_ww.load();
    if (!ww) {
        const double avg = (double) rows / std::max(b.e1 - b.e0, 1);
        ww = avg > 56.0 ? 2 : 4;   // 128-row items once experts average more than 56 rows (measured: 8192 tokens)
    }
    const bool v2 = int8_v2.load();
    const int tr = v2 ? tile_rows(ww) : TILE_ROWS;
    const int64_t tiles = (rows + tr - 1) / tr + (b.e1 - b.e0);
    const int64_t cap = (int64_t) d.sms * d.occ;
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (N_GU / weight_rows(ww)), cap);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (N_D / weight_rows(ww)), cap);
    if (v2 && d.next_ctr + 2 > NCTR) {
        const cudaError_t st = cudaMemsetAsync(d.tables + CTR_WORD, 0, NCTR * 4, stream);
        if (st != cudaSuccess) return st;
        d.next_ctr = 0;
    }
    if (v2) tb.ctr = d.tables + CTR_WORD + d.next_ctr++;
    launch_type<true>(gu_type, ww, g_gu, b, g, tb, (const uint8_t*) xa, src, rw, dst, (uint8_t*) ha, parts, stream);
    if (v2) tb.ctr = d.tables + CTR_WORD + d.next_ctr++;
    launch_type<false>(d_type, ww, g_d, b, g, tb, (const uint8_t*) ha, src, rw, dst, nullptr, parts, stream);
    return cudaGetLastError();
}

}  // namespace strata::models::deepseek41
namespace strata::models::deepseek41 {
namespace { std::atomic<bool> fused_enabled{false}; std::atomic<int> fused_min_tokens{512}; }
void fused_moe_set_min_tokens(int tokens) { fused_min_tokens.store(tokens); }
int fused_moe_min_tokens() { return fused_min_tokens.load(); }
void fused_moe_set_enabled(bool enabled) { fused_enabled.store(enabled); }
bool fused_moe_enabled() { return fused_enabled.load(); }
}  // namespace strata::models::deepseek41
