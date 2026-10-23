#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace strata::models::deepseek41 {
constexpr int BOUNDED_MOE_BATCH = 2;
// Host descriptors are copied into kernel arguments, never retained after return.
// gate_bf has room for contiguous Gate/Up (2*5120*2304 elements).
struct BoundedMoeGroup {
    const void *gate, *up, *down;
    uint16_t *gate_bf, *down_bf;
    int offset, tokens;
};
// Caller owns all storage. bx: rows*5120; gu: rows*4608; act: rows*2304
// BF16 elements. ptrs: 512 bytes, 16-byte aligned, borrowed from the existing
// dequant buffer. GEMM does not allocate any workspace.
// bx is reused as Down output only after Gate/Up has consumed the input.
// rows/routing/slots may be null for a shared expert (identity, unit, identity).
// No allocation, persistent expanded weight, or change to residency is allowed.
int bounded_moe_prepare_gpu(const BoundedMoeGroup& group,int gu_type,int down_type,cudaStream_t stream);
int bounded_moe_gpu(const BoundedMoeGroup* groups,int count,int gu_type,int down_type,
                    const float* x,const int* rows,const float* routing,const int* slots,float* out,
                    uint16_t* bx,uint16_t* gu,uint16_t* act,void** ptrs,int row_capacity,
                    cudaStream_t stream,bool prepared=false);
}
