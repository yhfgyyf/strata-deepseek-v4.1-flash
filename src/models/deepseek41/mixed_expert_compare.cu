#include <cuda_runtime.h>
#include <cstdint>

namespace {
__global__ void compare_bits(const uint16_t* a,const uint16_t* b,uint64_t n,unsigned long long* count) {
    unsigned long long different=0;
    for(uint64_t i=uint64_t(blockIdx.x)*256+threadIdx.x;i<n;i+=uint64_t(gridDim.x)*256)
        different+=a[i]!=b[i];
    if(different)atomicAdd(count,different);
}
}
cudaError_t mixed_test_compare_bf16(const uint16_t* a,const uint16_t* b,uint64_t n,
                                  unsigned long long* count,cudaStream_t stream) {
    auto status=cudaMemsetAsync(count,0,sizeof(*count),stream);
    if(status!=cudaSuccess)return status;
    compare_bits<<<4096,256,0,stream>>>(a,b,n,count);
    return cudaGetLastError();
}
