#include "strata/models/deepseek41/vision.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>
namespace strata::models::deepseek41 {
__global__ void softmax_rows(float* scores,int cols){
 const int row=blockIdx.x;float* x=scores+int64_t(row)*cols;float m=-INFINITY;
 for(int j=threadIdx.x;j<cols;j+=256)m=fmaxf(m,x[j]);
 __shared__ float sums[8];
 for(int d=16;d;d>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,d));
 if((threadIdx.x&31)==0)sums[threadIdx.x/32]=m;__syncthreads();
 m=threadIdx.x<8?sums[threadIdx.x]:-INFINITY;
 if(threadIdx.x<32)for(int d=16;d;d>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,d));
 if(threadIdx.x==0)sums[0]=m;__syncthreads();m=sums[0];float sum=0;
 for(int j=threadIdx.x;j<cols;j+=256){float v=expf(x[j]-m);x[j]=v;sum+=v;}
 for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
 if((threadIdx.x&31)==0)sums[threadIdx.x/32]=sum;__syncthreads();sum=threadIdx.x<8?sums[threadIdx.x]:0;
 if(threadIdx.x<32)for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
 if(threadIdx.x==0)sums[0]=sum;__syncthreads();sum=sums[0];
 for(int j=threadIdx.x;j<cols;j+=256)x[j]/=sum;
}
cudaError_t vision_attention_tiled_gpu(const float* q,const float* k,const float* v,float* out,float* scratch,int n,int tile,cublasHandle_t handle,cudaStream_t stream){
 if(!q||!k||!v||!out||!scratch||!handle||n<1||n>9198||tile<1||tile>128)return cudaErrorInvalidValue;
 if(cublasSetStream(handle,stream)!=CUBLAS_STATUS_SUCCESS)return cudaErrorUnknown;const float alpha=.125f,one=1.f,zero=0.f;
 for(int begin=0;begin<n;begin+=tile){const int m=(n-begin<tile)?n-begin:tile;
  auto e=cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,64,&alpha,k,1024,64,q+int64_t(begin)*1024,1024,64,&zero,scratch,n,int64_t(n)*m,16);if(e)return cudaErrorUnknown;
  softmax_rows<<<16*m,256,0,stream>>>(scratch,n);
  e=cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,64,m,n,&one,v,1024,64,scratch,n,int64_t(n)*m,&zero,out+int64_t(begin)*1024,1024,64,16);if(e)return cudaErrorUnknown;
 }
 return cudaGetLastError();
}

}
