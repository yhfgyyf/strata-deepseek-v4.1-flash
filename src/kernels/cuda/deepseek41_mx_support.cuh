#pragma once
// Minimal SM120 fragment/MMA helpers adapted from vllm-project/DeepGEMM
// a7a927731b109fdfec770df7c2a5be5d7e2216b5, common/sm120_utils.cuh and mma/sm120.cuh.
// The B64 CuTe swizzle is expressed directly, so no external build dependency is added.
//
// MIT License
// Copyright (c) 2025 DeepSeek
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <cuda_runtime.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#if __CUDACC_VER_MAJOR__ < 13
#error "SM120 block-scaled MMA requires CUDA 13 or newer"
#endif
namespace deep_gemm::sm120 {
template<int Bytes>struct CuTeSwizzle {
    static_assert(Bytes==64,"Only the validated B64 layout is used");
    __device__ __forceinline__ static int apply(int offset){return offset^((offset>>3)&0x30);}
};
template<int Bytes>struct SwizzleContext {
    int row_base_addr,row_xor_bits;
    __device__ __forceinline__ void init(int row,int row_stride){
        row_base_addr=row*row_stride;
        row_xor_bits=CuTeSwizzle<Bytes>::apply(row_base_addr)^row_base_addr;
    }
    __device__ __forceinline__ void* addr(char* tile,int column)const{
        return tile+row_base_addr+(column^row_xor_bits);
    }
};
__device__ __forceinline__ void ldmatrix_x4(uint32_t& a,uint32_t& b,uint32_t& c,uint32_t& d,void* p){
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
        :"=r"(a),"=r"(b),"=r"(c),"=r"(d):"l"(__cvta_generic_to_shared(p)):"memory");
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t& a,uint32_t& b,void* p){
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];\n"
        :"=r"(a),"=r"(b):"l"(__cvta_generic_to_shared(p)):"memory");
}
template<int Bytes>__device__ __forceinline__ void load_a_fragment(
 uint32_t (&frag)[4],char* tile,const SwizzleContext<Bytes>& ctx,int lane,int step,int k){
    ldmatrix_x4(frag[0],frag[1],frag[2],frag[3],ctx.addr(tile,(lane>>4)*16+step*k));
}
template<int Bytes>__device__ __forceinline__ void load_b_fragment_x2(
 uint32_t (&frag)[2],char* tile,const SwizzleContext<Bytes>& ctx,int lane,int step,int k){
    ldmatrix_x2(frag[0],frag[1],ctx.addr(tile,((lane>>3)&1)*16+step*k));
}
} // namespace deep_gemm::sm120
namespace deep_gemm::mma::sm120 {
__device__ __forceinline__ uint16_t extract_sf_pair(uint32_t packed,uint32_t index){
    return static_cast<uint16_t>((packed>>(index*8))&0xffff);
}
__device__ __forceinline__ void fp4_mma_block_scaled(
 float (&d)[4],const uint32_t (&a)[4],const uint32_t (&b)[2],uint16_t sa,uint16_t sb){
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
         "r"(static_cast<uint32_t>(sa)),"n"(static_cast<uint16_t>(0)),"n"(static_cast<uint16_t>(0)),
         "r"(static_cast<uint32_t>(sb)),"n"(static_cast<uint16_t>(0)),"n"(static_cast<uint16_t>(0)));
}
} // namespace deep_gemm::mma::sm120
#endif
