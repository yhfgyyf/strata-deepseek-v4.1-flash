// Standalone numerical experiment. No Runtime, model file or production dispatch is modified.
// Q8_0/Q8_1 are decoded before a COMMON FP32 down product with the same BF16-rounded
// weights. This isolates activation representation error; it is not a new integer dot kernel.
#include "runtime_internal.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace d=strata::models::deepseek41;
namespace c=strata::kernels::cpu;
namespace {
template<class T> struct Device {
    T* p=nullptr; size_t count;
    explicit Device(size_t n):count(n){d::gpu_check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    explicit Device(const std::vector<T>& x):Device(x.size()){set(x);}
    ~Device(){cudaFree(p);}
    void set(const std::vector<T>& x){if(x.size()!=count)throw std::runtime_error("device shape");d::gpu_check(cudaMemcpy(p,x.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read()const{std::vector<T> x(count);d::gpu_check(cudaMemcpy(x.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
float bf(float x){return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));}
struct Metric{double l2=0,maximum=0;};
Metric metric(const std::vector<float>& got,const std::vector<float>& ref){
    if(got.size()!=ref.size())throw std::runtime_error("metric shape");
    double error=0,denom=0,maximum=0;
    for(size_t i=0;i<got.size();++i){
        if(!std::isfinite(got[i])||!std::isfinite(ref[i]))throw std::runtime_error("nonfinite result");
        const double delta=double(got[i])-ref[i];error+=delta*delta;denom+=double(ref[i])*ref[i];maximum=std::max(maximum,std::abs(delta));
    }
    if(!denom)throw std::runtime_error("zero reference norm");
    return {std::sqrt(error/denom),maximum};
}
void record(int type,int seed,int tokens,const char* stage,const std::string& codec,const Metric& value){
    std::cout<<type<<','<<seed<<','<<tokens<<','<<stage<<','<<codec<<','<<value.l2<<','<<value.maximum<<'\n';
}
const std::vector<std::string> codecs={"BF16","Q8_K_CPU","Q8_0_CPU","Q8_1_CPU","Q8_1_GPU"};
std::vector<float> quantized(const std::vector<float>& x,int width,const std::string& codec){
    std::vector<float> out(x.size());
    if(codec=="BF16"){std::transform(x.begin(),x.end(),out.begin(),bf);return out;}
    const int tokens=x.size()/width;
    if(codec=="Q8_1_GPU"){
        static_assert(sizeof(block_q8_1)==36);
        Device<float> input(x);Device<uint8_t> packed(x.size()/32*sizeof(block_q8_1));
        strata::kernels::quantize_q8_1_rows(input.p,tokens,width,packed.p,nullptr);
        const auto data=packed.read();
        for(size_t b=0;b<x.size()/32;++b){
            block_q8_1 block;std::memcpy(&block,data.data()+b*sizeof(block),sizeof(block));
            const float scale=ggml_fp16_to_fp32(block.d);
            for(int j=0;j<32;++j)out[b*32+j]=scale*block.qs[j];
        }
        return out;
    }
    const ggml_type type=codec=="Q8_K_CPU"?GGML_TYPE_Q8_K:codec=="Q8_0_CPU"?GGML_TYPE_Q8_0:GGML_TYPE_Q8_1;
    const auto* traits=ggml_get_type_traits_cpu(type);
    if(!traits||!traits->from_float)throw std::runtime_error("missing native activation quantizer");
    std::vector<uint8_t> packed(ggml_row_size(type,width));
    for(int t=0;t<tokens;++t){
        traits->from_float(x.data()+t*width,packed.data(),width);
        float* dst=out.data()+t*width;
        if(type==GGML_TYPE_Q8_K)dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(packed.data()),dst,width);
        else if(type==GGML_TYPE_Q8_0)dequantize_row_q8_0(reinterpret_cast<const block_q8_0*>(packed.data()),dst,width);
        else for(int b=0;b<width/32;++b){
            block_q8_1 block;std::memcpy(&block,packed.data()+b*sizeof(block),sizeof(block));
            const float scale=ggml_fp16_to_fp32(block.d);
            for(int j=0;j<32;++j)dst[b*32+j]=scale*block.qs[j];
        }
    }
    return out;
}
void run(int type,int seed,int tokens,cublasHandle_t blas){
    constexpr int hidden=5120,ff=2304;
    c::NativeFmt f;std::string error;
    if(!c::native_fmt(type,type,hidden,ff,f,error))throw std::runtime_error(error);
    f.swiglu_limit=10;f.bf16_io=true;
    std::mt19937 rng(seed);std::normal_distribution<float> normal(0,.15f);
    std::vector<uint8_t> blob(f.bytes);std::vector<float> row(hidden);
    auto pack=[&](uint64_t offset,int rows,int cols,uint64_t stride){
        for(int r=0;r<rows;++r){
            for(int i=0;i<cols;++i)row[i]=normal(rng);
            if(ggml_quantize_chunk(static_cast<ggml_type>(type),row.data(),blob.data()+offset+r*stride,0,1,cols,nullptr)!=stride)
                throw std::runtime_error("fixture quantization");
        }
    };
    pack(0,ff,hidden,f.gu_row);pack(f.up_off,ff,hidden,f.gu_row);pack(f.down_off,hidden,ff,f.d_row);
    std::vector<float> x(tokens*hidden),weights(tokens),cpu_h(tokens*ff);
    for(int t=0;t<tokens;++t){weights[t]=.15f+.03f*t;for(int i=0;i<hidden;++i)x[t*hidden+i]=bf(std::sin((i+17*t)*.013f));}
    std::vector<std::vector<uint8_t>> input(tokens,std::vector<uint8_t>(f.act_bytes));
    for(int t=0;t<tokens;++t)c::native_quant_act(f,x.data()+t*hidden,input[t].data());
    for(int start=0;start<tokens;start+=c::MAXT){
        const int count=std::min(c::MAXT,tokens-start);const void* inputs[c::MAXT];float* outputs[c::MAXT];
        for(int t=0;t<count;++t){inputs[t]=input[start+t].data();outputs[t]=cpu_h.data()+(start+t)*ff;}
        c::native_gu_rows(f,blob.data(),inputs,count,outputs,0,ff);
    }
    for(int t=0;t<tokens;++t)for(int i=0;i<ff;++i)cpu_h[t*ff+i]=bf(cpu_h[t*ff+i]*weights[t]);
    Device<uint8_t> db(blob);Device<float> dx(x),dw(weights),dequant(hidden*ff),gate(tokens*ff),up(tokens*ff),act(tokens*ff),output(tokens*hidden);
    Device<uint16_t> weight_bf(hidden*ff),scratch(tokens*hidden);
    auto project=[&](uint64_t offset,int rows,int cols,const float* in,float* out){
        d::gpu_check(d::dequant_gpu(db.p+offset,type,dequant.p,uint64_t(rows)*cols,nullptr));
        d::gpu_check(d::float_to_bf16_gpu(dequant.p,weight_bf.p,uint64_t(rows)*cols,nullptr));
        d::Matrix m{weight_bf.p,30,rows,cols,uint64_t(rows)*cols};
        d::matrix_product(m,in,out,tokens,blas,nullptr,scratch.p,d::Arithmetic::production);
    };
    project(0,ff,hidden,dx.p,gate.p);project(f.up_off,ff,hidden,dx.p,up.p);
    d::gpu_check(d::swiglu_gpu(gate.p,up.p,dw.p,act.p,ff,tokens,10,nullptr));
    d::gpu_check(d::round_bf16_gpu(act.p,uint64_t(tokens)*ff,nullptr));
    const auto gpu_h=act.read();project(f.down_off,hidden,ff,act.p,output.p);const auto production=output.read();
    record(type,seed,tokens,"gate_up","Q8_K_CPU",metric(cpu_h,gpu_h));
    // BF16-rounded down weights are shared by ALL representation comparisons.
    std::vector<float> common_weight(hidden*ff);
    for(int r=0;r<hidden;++r){
        ggml_get_type_traits(static_cast<ggml_type>(type))->to_float(blob.data()+f.down_off+r*f.d_row,row.data(),ff);
        for(int i=0;i<ff;++i)common_weight[r*ff+i]=bf(row[i]);
    }
    Device<float> cw(common_weight),common_input(cpu_h),common_output(tokens*hidden);
    auto down=[&](const std::vector<float>& h){
        common_input.set(h);const float one=1,zero=0;
        d::gpu_check(cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,hidden,tokens,ff,&one,cw.p,ff,common_input.p,ff,&zero,common_output.p,hidden)==CUBLAS_STATUS_SUCCESS?cudaSuccess:cudaErrorUnknown);
        return common_output.read();
    };
    const auto target_down=down(gpu_h);
    auto rounded=target_down;for(auto& v:rounded)v=bf(v);
    record(type,seed,tokens,"common_vs_production","BF16",metric(rounded,production));
    for(const auto& codec:codecs){
        const auto q=quantized(gpu_h,ff,codec),cpu_q=quantized(cpu_h,ff,codec);
        record(type,seed,tokens,"activation_gpu_input",codec,metric(q,gpu_h));
        record(type,seed,tokens,"activation_cpu_input",codec,metric(cpu_q,cpu_h));
        record(type,seed,tokens,"isolated_down",codec,metric(down(q),target_down));
        auto routed=down(cpu_q);for(auto& v:routed)v=bf(v);
        record(type,seed,tokens,"routed_common_down",codec,metric(routed,production));
    }
    // Reproduce the actual existing CPU expert paths, separately from the common-product experiment.
    c::ExpertPool pool(3,false,true);
    for(bool bfloat:{false,true}){
        auto native=f;if(bfloat&&!c::native_enable_bf16_down(native,error))throw std::runtime_error(error);
        std::vector<float> result(tokens*hidden);std::vector<c::ExpertJobMulti> jobs;
        for(int start=0;start<tokens;start+=c::MAXT){
            c::ExpertJobMulti job;job.blob=blob.data();job.nt=std::min(c::MAXT,tokens-start);job.ff_weights=weights.data()+start;
            for(int t=0;t<job.nt;++t){job.nact[t]=input[start+t].data();job.out[t]=result.data()+(start+t)*hidden;}
            jobs.push_back(job);
        }
        pool.run_split_multi_native(native,jobs.data(),jobs.size());for(auto& v:result)v=bf(v);
        record(type,seed,tokens,"routed_native_control",bfloat?"BF16":"Q8_K_CPU",metric(result,production));
    }
    std::cerr<<"completed type="<<type<<" seed="<<seed<<" tokens="<<tokens<<'\n';
}
}
int main(){
    try{
        ggml_cpu_init();cublasHandle_t blas=nullptr;
        if(cublasCreate(&blas)!=CUBLAS_STATUS_SUCCESS||cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("BLAS initialization");
        struct Cleanup{cublasHandle_t h;~Cleanup(){cublasDestroy(h);}} cleanup{blas};
        std::cout<<std::setprecision(12)<<"weight_type,seed,tokens,stage,codec,relative_l2,max_abs\n";
        for(int type:{10,11})for(int offset:{0,1000})for(int tokens:{1,3,12})run(type,41+type+offset,tokens,blas);
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
