#pragma once
#include "strata/models/deepseek41/runtime.hpp"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

namespace strata::models::deepseek41 {
cudaError_t vision_qkv_gpu(const float* qkv,float* q,float* k,float* v,int height,int width,
                          bool bf16,cudaStream_t stream);
cudaError_t vision_attention_gpu(const float* q,const float* k,const float* v,float* output,
                                int patches,bool bf16,cudaStream_t stream);
// Scratch holds16*tile*patches F32 values; caller reuses the later FFN gate buffer.
cudaError_t vision_attention_tiled_gpu(const float* q,const float* k,const float* v,float* output,float* scratch,
                                      int patches,int tile,cublasHandle_t blas,cudaStream_t stream);
cudaError_t vision_unshuffle_gpu(const float* input,float* output,int height,int width,cudaStream_t stream);
cudaError_t vision_span_gpu(const float* features,const float* start,const float* end,const float* newline,
                           float* output,int height,int width,cudaStream_t stream);
cudaError_t vision_bias_gpu(float* input,const float* bias,int tokens,int width,bool bf16,cudaStream_t stream);
cudaError_t vision_add_gpu(float* residual,const float* branch,int64_t elements,bool bf16,cudaStream_t stream);
cudaError_t vision_activate_gpu(const float* gate,const float* up,float* output,int64_t elements,
                               bool gelu,bool bf16,cudaStream_t stream);
// Standalone mmproj execution is also used to validate vision before touching the main model.
std::vector<float> encode_vision(const std::string& mmproj,const std::vector<float>& patches,
                                 int height,int width,Arithmetic arithmetic,std::string capture = {},std::string integrity = {});
} // namespace strata::models::deepseek41
