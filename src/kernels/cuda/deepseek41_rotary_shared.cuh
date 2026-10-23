#pragma once
#include "strata/models/deepseek41/ops.hpp"
#include <cuda_runtime.h>

namespace strata::models::deepseek41::rotary_detail {
// Conservative dispatch from the fixed same-contract shape measurements.
static bool shared_shape(int heads,int dim,int tokens) {
    return (heads==1&&dim==512)||
        (heads==32&&dim==128&&(tokens==1||tokens>=128))||
        (heads==64&&dim==512&&tokens>=128);
}
static __global__ void rotate_shared(float* x,int heads,int dim,int64_t start,int step,
                                    RotaryFrequencies f,bool inverse) {
    __shared__ float sine[32],cosine[32];
    const int token=blockIdx.x;
    if(threadIdx.x<32){
        const float angle=float(start+int64_t(token)*step)*f.value[threadIdx.x];
        float s,c;sincosf(angle,&s,&c);
        sine[threadIdx.x]=inverse?-s:s;cosine[threadIdx.x]=c;
    }
    __syncthreads();
    for(int i=threadIdx.x;i<heads*32;i+=blockDim.x){
        const int head=i/32,d=i%32;
        float* p=x+(int64_t(token)*heads+head)*dim+dim-64+d*2;
        const float a=p[0],b=p[1],s=sine[d],c=cosine[d];
        p[0]=a*c-b*s;p[1]=a*s+b*c;
    }
}
static cudaError_t launch(float* x,int heads,int dim,int tokens,int64_t start,int step,
                          RotaryFrequencies f,bool inverse,cudaStream_t stream) {
    rotate_shared<<<tokens,256,0,stream>>>(x,heads,dim,start,step,f,inverse);
    return cudaGetLastError();
}
} // namespace strata::models::deepseek41::rotary_detail
