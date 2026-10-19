#include "strata/models/deepseek41/kernels.hpp"
#include <cuda_runtime.h>

namespace strata::models::deepseek41 {
namespace {
__global__ void confidence_kernel(const float* hidden,const float* markov,const float* weight,
                                  float* probability,float* raw) {
    const int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x/32;
    float sum=0;
    for(int col=threadIdx.x;col<5376;col+=blockDim.x){
        const float x=col<5120?hidden[row*5120+col]:markov[row*256+col-5120];
        sum=fmaf(x,weight[col],sum);
    }
    for(int offset=16;offset;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    __shared__ float partial[8];
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(warp==0){
        sum=lane<8?partial[lane]:0;
        for(int offset=16;offset;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(lane==0){
            if(raw)raw[row]=sum;
            probability[row]=1.0f/(1.0f+expf(-sum));
        }
    }
}
}
cudaError_t draft_confidence_gpu(const float* hidden,const float* markov,const float* weight,
                                 float* probability,float* raw,int tokens,cudaStream_t stream) {
    if(!hidden||!markov||!weight||!probability||tokens<=0)return cudaErrorInvalidValue;
    confidence_kernel<<<tokens,256,0,stream>>>(hidden,markov,weight,probability,raw);
    return cudaGetLastError();
}
}
