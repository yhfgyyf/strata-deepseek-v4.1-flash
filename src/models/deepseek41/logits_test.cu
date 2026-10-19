#include "runtime_internal.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>

namespace d=strata::models::deepseek41;
namespace {
template<class T>struct Device {
    T* p=nullptr;explicit Device(size_t n){d::gpu_check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    ~Device(){cudaFree(p);}
};
__global__ void fill(uint16_t* weights,float* fp32,float* input,int64_t elements,int tokens,int cols) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=int64_t(gridDim.x)*blockDim.x) {
        const float scale=__half2float(__float2half_rn(ldexpf(1+float((i/32)%31)/32,-10-int((i/32)%7))));
        const float value=scale*(int((i*97)%255)-127);
        fp32[i]=value;reinterpret_cast<__nv_bfloat16*>(weights)[i]=__float2bfloat16_rn(value);
    }
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(tokens)*cols;i+=int64_t(gridDim.x)*blockDim.x)
        input[i]=__bfloat162float(__float2bfloat16_rn(float(int((i*43)%239)-119)*.013337f));
}
void head(int type){
    constexpr int rows=129280,cols=5120,tokens=260;
    d::Stream stream;cublasHandle_t blas=nullptr;
    if(cublasCreate(&blas)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("BLAS creation");
    struct Guard{cublasHandle_t h;~Guard(){cublasDestroy(h);}} guard{blas};
    if(cublasSetStream(blas,stream.value)!=CUBLAS_STATUS_SUCCESS || cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("BLAS setup");
    Device<uint16_t> weights(int64_t(rows)*cols),scratch(int64_t(tokens)*cols);
    Device<float> fp32(int64_t(rows)*cols),input(int64_t(tokens)*cols),original(int64_t(tokens)*rows+64),tile(int64_t(128)*rows+64);
    fill<<<4096,256,0,stream.value>>>(weights.p,fp32.p,input.p,int64_t(rows)*cols,tokens,cols);d::gpu_check(cudaGetLastError());
    d::Matrix matrix{type==30?static_cast<void*>(weights.p):static_cast<void*>(fp32.p),type,rows,cols,uint64_t(rows)*cols};
    auto project=[&](const float* x,float* out,int nt){
        if(type==0)for(int begin=0;begin<rows;begin+=2048)
            d::matrix_product(matrix,x,out+begin,nt,blas,stream.value,scratch.p,d::Arithmetic::production,std::min(2048,rows-begin),cols,rows,uint64_t(begin)*cols,false);
        else d::matrix_product(matrix,x,out,nt,blas,stream.value,scratch.p,d::Arithmetic::production,0,0,0,0,false);
    };
    project(input.p,original.p,tokens);
    std::vector<float> reference(int64_t(tokens)*rows),got(reference.size());
    d::gpu_check(cudaMemcpyAsync(reference.data(),original.p,reference.size()*4,cudaMemcpyDeviceToHost,stream.value));
    for(int first=0;first<tokens;first+=128) {
        const int n=std::min(128,tokens-first);
        project(input.p+int64_t(first)*cols,tile.p,n);
        d::gpu_check(cudaMemcpyAsync(got.data()+int64_t(first)*rows,tile.p,int64_t(n)*rows*4,cudaMemcpyDeviceToHost,stream.value));
    }
    d::gpu_check(cudaStreamSynchronize(stream.value));
    double maximum=0,error=0,norm=0;
    for(size_t i=0;i<got.size();++i){const double delta=double(got[i])-reference[i];if(!std::isfinite(got[i]))throw std::runtime_error("logit tile nonfinite");maximum=std::max(maximum,std::abs(delta));error+=delta*delta;norm+=double(reference[i])*reference[i];}
    const double relative=std::sqrt(error/norm);
    std::cout<<"full-vocab129280 head260 type="<<type<<", tile128+128+4: max_abs="<<maximum<<" rel_l2="<<relative<<'\n';
    if(maximum>1e-4 || relative>1e-5)throw std::runtime_error("tiled head numerical tolerance");
}
}
int main(){try {
    head(30);head(0);
    for(int tokens:{4096,8192})for(bool aliases:{false,true}) {
        d::Workspace w(16ull<<30,tokens,1048576,true,aliases);
        w.engram_input=d::allocate<uint16_t>(*w.arena,uint64_t(tokens)*6144);
        for(auto& bf:w.expert_ahead)bf=d::allocate<uint16_t>(*w.arena,5120*2304);
        if(w.max_logit_tokens!=128)throw std::runtime_error("unbounded logit workspace");
        if(aliases && (w.parts!=w.q || w.group_x!=w.o_low || w.group_gate!=reinterpret_cast<float*>(w.q_bf) ||
                       w.shared_gate!=reinterpret_cast<float*>(w.out_bf) || w.max_bf_input_cols!=8192))
            throw std::runtime_error("phase scratch ownership/bounds");
        std::cout<<"compact-workspace tokens="<<tokens<<" phase_aliases="<<aliases<<" used_bytes="<<w.arena->used()<<" logit_rows="<<w.max_logit_tokens<<'\n';
    }
    std::cout<<"COMPACT_LOGITS_FULL_VOCAB_AND_WORKSPACE_PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
