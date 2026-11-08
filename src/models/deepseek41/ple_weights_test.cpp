#include "runtime_internal.hpp"
#include "strata/models/deepseek41/engram.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace d=strata::models::deepseek41;
namespace {
template<class T>struct Device {
    T* data=nullptr;size_t size;
    explicit Device(size_t n):size(n){d::gpu_check(cudaMalloc(reinterpret_cast<void**>(&data),n*sizeof(T)));}
    explicit Device(const std::vector<T>& v):Device(v.size()){d::gpu_check(cudaMemcpy(data,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){cudaFree(data);}
    std::vector<T> read(){std::vector<T> v(size);d::gpu_check(cudaMemcpy(v.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
float bf(float x){return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));}
void compare(const std::vector<float>& got,const std::vector<float>& ref,bool rounded,const std::string& label){
    need(got.size()==ref.size(),"PLE comparison geometry");double maximum=0,squared=0,norm=0;
    for(size_t i=0;i<got.size();++i){
        need(std::isfinite(got[i])&&std::isfinite(ref[i]),"PLE nonfinite output");
        const double error=std::abs(double(got[i])-ref[i]);maximum=std::max(maximum,error);
        squared+=error*error;norm+=double(ref[i])*ref[i];
        const double limit=rounded?.01*std::max(1.,double(std::abs(ref[i]))):1e-4+2e-5*std::abs(ref[i]);
        need(error<=limit,"PLE independent per-element tolerance");
    }
    const double relative=std::sqrt(squared/std::max(norm,1e-30));need(relative<=(rounded?.003:2e-5),"PLE independent L2 tolerance");
    std::cout<<"{\"case\":\""<<label<<"\",\"bf16\":"<<(rounded?"true":"false")<<",\"max_abs\":"<<maximum<<",\"rel_l2\":"<<relative<<"}\n";
}
void projection(const d::TensorView& tensor,const std::string& label,cublasHandle_t blas){
    const auto& info=*tensor.info;const int cols=info.shape[0],total=info.shape[1],rows=16;
    const auto type=static_cast<ggml_type>(info.type);const auto stride=ggml_row_size(type,cols);
    std::vector<uint8_t> packed(rows*stride);std::vector<float> reference(rows*cols);
    for(int r=0;r<rows;++r){
        const int row=int(int64_t(r)*(total-1)/(rows-1));
        std::memcpy(packed.data()+r*stride,tensor.data+row*stride,stride);
        ggml_get_type_traits(type)->to_float(packed.data()+r*stride,reference.data()+r*cols,cols);
    }
    Device<uint8_t> raw(packed);Device<float> decoded(reference.size());Device<uint16_t> rounded(reference.size());
    d::gpu_check(d::dequant_gpu(raw.data,type,decoded.data,reference.size(),nullptr));
    need(decoded.read()==reference,"PLE real-weight dequant differs from GGML");
    d::gpu_check(d::dequant_bf16_gpu(raw.data,type,rounded.data,reference.size(),nullptr));
    const auto actual=rounded.read();
    for(size_t i=0;i<actual.size();++i)need(actual[i]==ggml_fp32_to_bf16(reference[i]).bits,"PLE real-weight BF16 boundary");
    for(bool production:{false,true})for(int tokens:{1,3,8,9,128})for(int pattern:{0,1,2}){
        std::vector<float> input(tokens*cols),expected(tokens*rows);
        for(size_t i=0;i<input.size();++i){
            const float value=pattern==0?0.f:std::sin(float(i%100003)*.013f)*(pattern==2?128.f:1.f);
            input[i]=production?bf(value):value;
        }
        for(int t=0;t<tokens;++t)for(int r=0;r<rows;++r){
            double sum=0;for(int k=0;k<cols;++k)sum+=double(production?bf(reference[r*cols+k]):reference[r*cols+k])*input[t*cols+k];
            expected[t*rows+r]=production?bf(float(sum)):float(sum);
        }
        Device<float> x(input),out(expected.size());Device<uint16_t> scratch(input.size());
        d::Matrix m{production?static_cast<void*>(rounded.data):decoded.data,production?30:0,rows,cols,uint64_t(rows)*cols};
        d::matrix_product(m,x.data,out.data,tokens,blas,nullptr,scratch.data,production?d::Arithmetic::production:d::Arithmetic::f32,
                          0,0,0,0,true,production&&tokens>1&&tokens<=8);
        compare(out.read(),expected,production,label+"/N"+std::to_string(tokens)+"/input"+std::to_string(pattern));
    }
}
void table(const d::Model& model){
    const auto& t=model.tensor("per_layer_token_embd.weight");
    const auto type=static_cast<ggml_type>(t.info->type);const int cols=t.info->shape[0];const auto stride=ggml_row_size(type,cols);
    const int64_t total=t.info->shape[1];const std::vector<int64_t> rows{0,1,127,128,384006167,384006168,total-2,total-1};
    std::vector<uint8_t> packed(rows.size()*stride);std::vector<float> reference(rows.size()*cols);
    for(size_t r=0;r<rows.size();++r){
        std::memcpy(packed.data()+r*stride,t.data+rows[r]*stride,stride);
        ggml_get_type_traits(type)->to_float(packed.data()+r*stride,reference.data()+r*cols,cols);
    }
    Device<uint8_t> raw(packed);Device<float> output(reference.size());
    d::gpu_check(d::dequant_gpu(raw.data,type,output.data,reference.size(),nullptr));need(output.read()==reference,"PLE IQ4_NL real row dequant");
    for(auto& x:reference)x=bf(x);
    d::gpu_check(d::dequant_round_bf16_gpu(raw.data,type,output.data,reference.size(),nullptr));need(output.read()==reference,"PLE IQ4_NL real row fused BF16");
    std::cout<<"{\"case\":\"IQ4_NL table boundaries\",\"rows\":8,\"exact\":true}\n";
}
void gate(const d::Model& model){
    constexpr int width=5120;
    for(int layer:{1,14})for(int count:{1,3,8,9,128}){
        const auto& t=model.tensor("blk."+std::to_string(layer)+".engram_gate.weight");
        std::vector<float> weights(width*4);std::memcpy(weights.data(),t.data,weights.size()*4);
        std::vector<float> h(count*4*width),kv(count*5*width);std::vector<uint8_t> mask(count);
        for(size_t i=0;i<h.size();++i)h[i]=bf(std::sin(float(i%100003)*.031f));
        for(size_t i=0;i<kv.size();++i)kv[i]=bf(std::cos(float(i%100003)*.019f));
        if(count>1)mask[1]=1;
        for(bool production:{false,true}){
            auto expected=h;d::engram_gate_cpu(expected.data(),kv.data(),weights.data(),nullptr,mask.data(),count,width);
            if(production)for(auto& x:expected)x=bf(x);
            Device<float> residual(h),value(kv),w(weights);Device<uint8_t> image(mask);
            d::gpu_check((production?d::engram_gate_bf16_gpu:d::engram_gate_gpu)(residual.data,value.data,w.data,nullptr,image.data,count,width,nullptr));
            compare(residual.read(),expected,production,"folded gate/L"+std::to_string(layer)+"/N"+std::to_string(count));
        }
    }
}
}
int main(int argc,char** argv){try{
    if(argc!=2)throw std::invalid_argument("usage: deepseek41_ple_weights_test MAIN.gguf");
    d::Model model(argv[1]);need(model.ple_engram(),"PLE checkpoint required");
    cublasHandle_t blas=nullptr;need(cublasCreate(&blas)==CUBLAS_STATUS_SUCCESS,"cuBLAS init");
    struct Cleanup{cublasHandle_t handle;~Cleanup(){cublasDestroy(handle);}}cleanup{blas};
    need(cublasSetMathMode(blas,static_cast<cublasMath_t>(CUBLAS_PEDANTIC_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION))==CUBLAS_STATUS_SUCCESS,"cuBLAS math mode");
    table(model);gate(model);
    for(const std::string name:{"blk.1.engram_kv.weight","blk.14.engram_kv.weight","blk.0.attn_output_a.weight","blk.39.attn_output_a.weight"})
        projection(model.tensor(name),name,blas);
    std::cout<<"{\"status\":\"passed\",\"reference\":\"GGML dequant and FP64 dot; original checkpoint quantization error is not measured\"}\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
