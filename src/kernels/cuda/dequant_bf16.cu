// src/kernels/cuda/dequant_bf16.cu - see include/strata/kernels/dequant_bf16.hpp.
//
// Arithmetic transcribed from ggml/src/ggml-quants.c at the pinned llama.cpp (MIT License, Copyright (c) 2023-2026
// The ggml authors): dequantize_row_q2_0/q4_0/q5_0/q8_0/q3_K/q4_K/q5_K/q6_K/iq4_nl/iq4_xs.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <atomic>

namespace strata::kernels {
namespace {

__device__ __forceinline__ float h2f(const uint8_t* p) {
    return __half2float(__ushort_as_half((uint16_t) (p[0] | (p[1] << 8))));
}
__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    // a NaN (a NaN scale in the block) stays a quiet NaN, as in ggml_compute_fp32_to_bf16 and `bf16_from_f32`:
    // the rounding add below would carry it into -0 or inf
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64u);
    u += 0x7fffu + ((u >> 16) & 1u);          // round to nearest even
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ void put(uint16_t* o, int i, float v) { o[i] = f2bf(v); }
struct H16 { uint16_t v; };
__device__ __forceinline__ void put(H16* o, int i, float v) { o[i].v = __half_as_ushort(__float2half_rn(v)); }
__device__ __forceinline__ void put(float* o, int i, float v) { o[i] = v; }

__constant__ int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// One 32-element group `g` (row-major over the whole slice); `out` points at that group's 32 outputs.
template <int TYPE, typename T>
__device__ __forceinline__ void group32(const uint8_t* row_blocks, int gi_in_row, T* out) {
    if constexpr (TYPE == 42) {                                   // Q2_0: 64 per block of 18 B
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 2) * 18;
        const float d = h2f(b);
        const int e0 = (gi_in_row % 2) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            put(out, j, (float) (q - 1) * d);
        }
    } else if constexpr (TYPE == 2) {                              // Q4_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, (float) ((b[2 + j] & 0x0F) - 8) * d);
            put(out, j + 16, (float) ((b[2 + j] >> 4) - 8) * d);
        }
    } else if constexpr (TYPE == 6) {                              // Q5_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 22;
        const float d = h2f(b);
        const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            put(out, j, (float) (((b[6 + j] & 0x0F) | xh0) - 16) * d);
            put(out, j + 16, (float) (((b[6 + j] >> 4) | xh1) - 16) * d);
        }
    } else if constexpr (TYPE == 7) {                              // Q5_1
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 24;
        const float d = h2f(b), m = h2f(b + 2);
        const uint32_t qh = (uint32_t) b[4] | ((uint32_t) b[5] << 8) |
                            ((uint32_t) b[6] << 16) | ((uint32_t) b[7] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) & 1) << 4;
            const int xh1 = ((qh >> (j + 16)) & 1) << 4;
            put(out, j, (float) ((b[8 + j] & 15) | xh0) * d + m);
            put(out, j + 16, (float) ((b[8 + j] >> 4) | xh1) * d + m);
        }
    } else if constexpr (TYPE == 8) {                              // Q8_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 34;
        const float d = h2f(b);
        for (int j = 0; j < 32; ++j) put(out, j, (float) (int8_t) b[2 + j] * d);
    } else if constexpr (TYPE == 20) {                             // IQ4_NL
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, d * (float) kv_iq4nl[b[2 + j] & 0xf]);
            put(out, j + 16, d * (float) kv_iq4nl[b[2 + j] >> 4]);
        }
    } else if constexpr (TYPE == 11) {                             // Q3_K: hmask[32] qs[64] scales[12] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 110;
        const int gi = gi_in_row % 8, n = gi / 4, jj = gi % 4;
        const uint8_t* hm = b;
        const uint8_t* q = b + 32 + n * 32;
        const uint8_t* sc = b + 96;
        const float d_all = h2f(b + 108);
        uint32_t aux[4];
        memcpy(aux, sc, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        const int shift = 2 * jj;
        const uint8_t m = (uint8_t) (1u << (n * 4 + jj));
        for (int t = 0; t < 32; ++t) {
            const int is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
            const float dl = d_all * (float) (scales[is] - 32);
            put(out, t, dl * (float) ((int) ((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4)));
        }
    } else if constexpr (TYPE == 12) {                             // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* q = b + 16 + 32 * j64;
        for (int l = 0; l < 32; ++l) put(out, l, d1 * (float) (hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {                             // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t) (1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, l, d1 * (float) (nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, l, d * (float) sc[is + 2 * qu] * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            put(out, j, dl * (float) kv_iq4nl[qs[j] & 0xf]);
            put(out, j + 16, dl * (float) kv_iq4nl[qs[j] >> 4]);
        }
    }
}

template <int TYPE, typename T>
__global__ void dequant_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                               int64_t groups_per_row, int64_t ld, T* __restrict__ out) {
    const int64_t g = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= rows * groups_per_row) return;
    const int64_t r = g / groups_per_row, gi = g % groups_per_row;
    group32<TYPE>(blocks + (row0 + r) * row_bytes, (int) gi, out + r * ld + gi * 32);
}

// Q6-only port of Strata 9ae93f7b's wide8/dequant_wide_kernel (MIT).
// A block is 210 bytes: read at two-byte alignment, never assume uint64 alignment.
std::atomic<int> q6_mode{2};
__device__ __forceinline__ uint64_t q6_ld8(const uint8_t* p) {
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return uint64_t(q[0]) | (uint64_t(q[1]) << 16) | (uint64_t(q[2]) << 32) | (uint64_t(q[3]) << 48);
}
template <typename T, bool Rounded = false>
__global__ void dequant_q6_wide_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0,
                                     uint32_t rows, uint32_t groups_per_row, int64_t ld, T* __restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t g = i >> 2, r = g / groups_per_row, gi = g - r * groups_per_row;
    if (r >= rows) return;
    const int l0 = 8 * (i & 3), n = (gi % 8) / 4, qu = gi % 4;
    const uint8_t* b = blocks + (row0 + int64_t(r)) * row_bytes + size_t(gi / 8) * 210;
    const uint64_t ql = q6_ld8(b + 64 * n + ((qu & 1) ? 32 : 0) + l0);
    const uint64_t qh = q6_ld8(b + 128 + 32 * n + l0);
    const int8_t* scales = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
    const float ds = h2f(b + 208) * float(scales[l0 / 16 + 2 * qu]);
    __align__(16) T values[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int lo = int(ql >> (8 * k)) & 255, hi = int(qh >> (8 * k)) & 255;
        const int q = (qu >= 2 ? lo >> 4 : lo & 15) | (((hi >> (2 * qu)) & 3) << 4);
        const float value = ds * float(q - 32);
        if constexpr (Rounded) put(values, k, __bfloat162float(__float2bfloat16_rn(value)));
        else put(values, k, value);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + int64_t(r) * ld + int64_t(gi) * 32 + l0);
#pragma unroll
    for (int k = 0; k < int(8 * sizeof(T) / 16); ++k)
        dst[k] = reinterpret_cast<const uint4*>(values)[k];
}

// Q5_K Engram rows: eight contiguous outputs per thread, with the
// same scale/minimum expression as the independent group32 implementation.
template <typename T, bool Rounded = false>
__global__ void dequant_q5_wide_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0,
                                     uint32_t rows, uint32_t groups_per_row, int64_t ld, T* __restrict__ out) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t g=i>>2,r=g/groups_per_row,gi=g-r*groups_per_row;
    if(r>=rows)return;
    const int l0=8*(i&3),group=gi%8;
    const uint8_t* b=blocks+(row0+int64_t(r))*row_bytes+size_t(gi/8)*176;
    const uint64_t ql=q6_ld8(b+48+32*(group/2)+l0),qh=q6_ld8(b+16+l0);
    int sc,mn;scale_min_k4(group,b+4,sc,mn);
    const float ds=h2f(b)*float(sc),minimum=h2f(b+2)*float(mn);
    __align__(16) T values[8];
#pragma unroll
    for(int k=0;k<8;++k){
        const int lo=int(ql>>(8*k))&255,hi=int(qh>>(8*k))&255;
        const int q=((lo>>(4*(group&1)))&15)|(((hi>>group)&1)<<4);
        const float value=ds*float(q)-minimum;
        if constexpr(Rounded)put(values,k,__bfloat162float(__float2bfloat16_rn(value)));
        else put(values,k,value);
    }
    auto* dst=reinterpret_cast<uint4*>(out+int64_t(r)*ld+int64_t(gi)*32+l0);
#pragma unroll
    for(int k=0;k<int(8*sizeof(T)/16);++k)dst[k]=reinterpret_cast<const uint4*>(values)[k];
}

// Q8_0 to FP16 (the prompt path's dense Q8_0 weights before each cuBLAS GEMM): a thread per 8 outputs, one 16-byte
// store each (dequant_kernel's thread wrote 32 halves one by one, 64 bytes apart from its neighbours': ~13% of the
// card's bandwidth).  The same value per element: (float) q * d, rounded to FP16 by __float2half_rn.
__global__ void dequant_q8_0_h16_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                                        int64_t chunks_per_row, int64_t ld, uint16_t* __restrict__ out) {
    const int64_t g = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= rows * chunks_per_row) return;
    const int64_t r = g / chunks_per_row, c = g % chunks_per_row;
    const uint8_t* b = blocks + (row0 + r) * row_bytes + (c >> 2) * 34;   // blocks are 2-byte aligned
    const float d = h2f(b);
    const uint16_t* q2 = reinterpret_cast<const uint16_t*>(b + 2 + (c & 3) * 8);
    uint32_t w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint16_t qq = q2[k];
        const uint32_t lo = __half_as_ushort(__float2half_rn((float) (int8_t) (qq & 0xff) * d));
        const uint32_t hi = __half_as_ushort(__float2half_rn((float) (int8_t) (qq >> 8) * d));
        w[k] = lo | (hi << 16);
    }
    *reinterpret_cast<uint4*>(out + r * ld + c * 8) = make_uint4(w[0], w[1], w[2], w[3]);
}

bool geometry(int type, int& block_elems, int& block_bytes) {
    switch (type) {
    case 2: block_elems = 32; block_bytes = 18; return true;
    case 6: block_elems = 32; block_bytes = 22; return true;
    case 7: block_elems = 32; block_bytes = 24; return true;
    case 8: block_elems = 32; block_bytes = 34; return true;
    case 20: block_elems = 32; block_bytes = 18; return true;
    case 11: block_elems = 256; block_bytes = 110; return true;
    case 12: block_elems = 256; block_bytes = 144; return true;
    case 13: block_elems = 256; block_bytes = 176; return true;
    case 14: block_elems = 256; block_bytes = 210; return true;
    case 23: block_elems = 256; block_bytes = 136; return true;
    case 42: block_elems = 64; block_bytes = 18; return true;
    default: return false;
    }
}

template <typename T>
void launch(int type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, T* out, void* stream,
            int64_t ld = 0) {
    int be = 0, bb = 0;
    if (!geometry(type, be, bb) || cols % be != 0 || rows <= 0) {
        std::fprintf(stderr, "dequant: unsupported type %d or shape %lld x %lld\n", type, (long long) rows,
                     (long long) cols);
        std::exit(1);
    }
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32, total = rows * gpr;
    if (ld <= 0) ld = cols;
    const unsigned grid = (unsigned) ((total + 255) / 256);
    const uint8_t* p = (const uint8_t*) blocks;
    cudaStream_t st = (cudaStream_t) stream;
    if (!std::is_same_v<T,H16> && type==13 && (reinterpret_cast<uintptr_t>(blocks)&1)==0 &&
        (reinterpret_cast<uintptr_t>(out)&15)==0 && (uint64_t(ld)*sizeof(T))%16==0 && total<(1ll<<29)) {
        dequant_q5_wide_kernel<T><<<unsigned((4*total+255)/256),256,0,st>>>(
            p,row_bytes,row0,uint32_t(rows),uint32_t(gpr),ld,out);
        const auto e=cudaGetLastError();
        if(e!=cudaSuccess){std::fprintf(stderr,"dequant Q5 wide launch: %s\n",cudaGetErrorString(e));std::exit(1);}
        return;
    }
    if (!std::is_same_v<T, H16> && type == 14 && q6_mode.load(std::memory_order_relaxed) >= 1 &&
        (reinterpret_cast<uintptr_t>(blocks) & 1) == 0 &&
        (reinterpret_cast<uintptr_t>(out) & 15) == 0 &&
        (uint64_t(ld) * sizeof(T)) % 16 == 0 && total < (1ll << 29)) {
        dequant_q6_wide_kernel<T><<<unsigned((4 * total + 255) / 256), 256, 0, st>>>(
            p, row_bytes, row0, uint32_t(rows), uint32_t(gpr), ld, out);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "dequant Q6 wide launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    if constexpr (sizeof(T) == 2 && !std::is_same_v<T, uint16_t>) {   // H16
        if (type == 8 && (uintptr_t) out % 16 == 0 && ld % 8 == 0 && (uintptr_t) p % 2 == 0 && row_bytes % 2 == 0) {
            const int64_t chunks = rows * (cols / 8);
            dequant_q8_0_h16_kernel<<<(unsigned) ((chunks + 255) / 256), 256, 0, st>>>(
                p, row_bytes, row0, rows, cols / 8, ld, reinterpret_cast<uint16_t*>(out));
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { std::fprintf(stderr, "dequant launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
            return;
        }
    }
#define STRATA_DQ(TY) dequant_kernel<TY, T><<<grid, 256, 0, st>>>(p, row_bytes, row0, rows, gpr, ld, out); break
    switch (type) {
    case 2: STRATA_DQ(2);
    case 6: STRATA_DQ(6);
    case 7: STRATA_DQ(7);
    case 8: STRATA_DQ(8);
    case 11: STRATA_DQ(11);
    case 12: STRATA_DQ(12);
    case 13: STRATA_DQ(13);
    case 14: STRATA_DQ(14);
    case 20: STRATA_DQ(20);
    case 23: STRATA_DQ(23);
    case 42: STRATA_DQ(42);
    }
#undef STRATA_DQ
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "dequant launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace

void dequant_q6_set_mode(int mode) {
    if (mode < 0 || mode > 2) { std::fprintf(stderr, "dequant Q6 mode must be 0, 1 or 2\n"); std::exit(1); }
    q6_mode.store(mode, std::memory_order_relaxed);
}

bool dequant_q5_f32_bf16(const void* blocks,int64_t elements,float* out,void* stream) {
    if(!blocks||!out||elements<=0||elements%256||(reinterpret_cast<uintptr_t>(blocks)&1)!=0||
       (reinterpret_cast<uintptr_t>(out)&15)!=0||elements/32>=(1ll<<29))return false;
    const int64_t groups=elements/32;
    dequant_q5_wide_kernel<float,true><<<unsigned((4*groups+255)/256),256,0,(cudaStream_t)stream>>>(
        static_cast<const uint8_t*>(blocks),elements/256*176,0,1,uint32_t(groups),elements,out);
    return true;
}

bool dequant_q6_f32_bf16(const void* blocks, int64_t elements, float* out, void* stream) {
    if (q6_mode.load(std::memory_order_relaxed) < 2 || !blocks || !out || elements <= 0 || elements % 256 ||
        (reinterpret_cast<uintptr_t>(blocks) & 1) != 0 || (reinterpret_cast<uintptr_t>(out) & 15) != 0 ||
        elements / 32 >= (1ll << 29)) return false;
    const int64_t groups = elements / 32;
    dequant_q6_wide_kernel<float, true><<<unsigned((4 * groups + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
        static_cast<const uint8_t*>(blocks), elements / 256 * 210, 0, 1, uint32_t(groups), elements, out);
    return true;
}

bool dequant_bf16_supported(int ggml_type) noexcept {
    int a, b;
    return geometry(ggml_type, a, b);
}

namespace {
// plan v0.3 P6: the i-quant formats (llama.cpp's dequantizers, iq_kernels.cu)
bool iq_only(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }
}  // namespace

void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream) {
    launch<uint16_t>(ggml_type, blocks, row0, rows, cols, out, stream);
}

void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f16(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream);
}

bool dequant_f16_ld(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, int64_t ld,
                    uint16_t* out, void* stream) {
    int be = 0, bb = 0;
    if (iq_only(ggml_type) || !geometry(ggml_type, be, bb) || cols % be != 0 || rows <= 0 || ld < cols || ld % 8 != 0)
        return false;
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream, ld);
    return true;
}

void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f32(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<float>(ggml_type, blocks, row0, rows, cols, out, stream);
}

}  // namespace strata::kernels
