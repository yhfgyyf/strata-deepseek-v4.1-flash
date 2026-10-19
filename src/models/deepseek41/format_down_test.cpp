#include "runtime_internal.hpp"
#include "strata/models/deepseek41/format_down.hpp"
#include "strata/kernels/cpu/deepseek41_down.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>

namespace d=strata::models::deepseek41;
namespace c=strata::kernels::cpu;
namespace k=strata::kernels;
namespace {
int numerical_failures=0;
template<class T> struct Device{
    T* p=nullptr;size_t size;
    explicit Device(size_t n):size(n){d::gpu_check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    explicit Device(const std::vector<T>& values):Device(values.size()){set(values);}
    ~Device(){cudaFree(p);}
    void set(const std::vector<T>& values){d::gpu_check(cudaMemcpy(p,values.data(),size*sizeof(T),cudaMemcpyHostToDevice));d::gpu_check(cudaDeviceSynchronize());}
    std::vector<T> read(){d::gpu_check(cudaDeviceSynchronize());std::vector<T> values(size);d::gpu_check(cudaMemcpy(values.data(),p,size*sizeof(T),cudaMemcpyDeviceToHost));return values;}
};
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
float bf(float x){return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));}
double relative(const std::vector<float>& got,const std::vector<float>& reference){
    require(got.size()==reference.size(),"metric shape");double error=0,norm=0;
    for(size_t i=0;i<got.size();++i){require(std::isfinite(got[i])&&std::isfinite(reference[i]),"nonfinite output");
        const double delta=double(got[i])-reference[i];error+=delta*delta;norm+=double(reference[i])*reference[i];}
    return std::sqrt(error/norm);
}
template<class F> double gpu_time(F&& operation,cudaStream_t stream){
    for(int i=0;i<3;++i)operation();
    cudaEvent_t first,last;d::gpu_check(cudaEventCreate(&first));d::gpu_check(cudaEventCreate(&last));
    d::gpu_check(cudaEventRecord(first,stream));for(int i=0;i<20;++i)operation();
    d::gpu_check(cudaEventRecord(last,stream));d::gpu_check(cudaEventSynchronize(last));
    float ms=0;d::gpu_check(cudaEventElapsedTime(&ms,first,last));cudaEventDestroy(first);cudaEventDestroy(last);return ms/20;
}
void run(int type,int seed,int tokens,cublasHandle_t blas,cudaStream_t stream){
    constexpr int hidden=5120,ff=2304;
    c::NativeFmt original;std::string error;require(c::native_fmt(type,type,hidden,ff,original,error),error.c_str());
    original.swiglu_limit=10;original.bf16_io=true;
    std::mt19937 random(seed);std::normal_distribution<float> normal(0,.15f);
    std::vector<uint8_t> blob(original.bytes);std::vector<float> row(hidden);
    auto pack=[&](uint64_t offset,int rows,int cols,uint64_t stride){for(int r=0;r<rows;++r){
        for(int i=0;i<cols;++i)row[i]=normal(random);
        require(ggml_quantize_chunk(static_cast<ggml_type>(type),row.data(),blob.data()+offset+r*stride,0,1,cols,nullptr)==stride,"fixture quantization");}};
    pack(0,ff,hidden,original.gu_row);pack(original.up_off,ff,hidden,original.gu_row);pack(original.down_off,hidden,ff,original.d_row);
    std::vector<float> x(tokens*hidden),routing(tokens);
    for(int t=0;t<tokens;++t){routing[t]=.15f+.03f*t;for(int i=0;i<hidden;++i)x[t*hidden+i]=bf(std::sin((i+17*t)*.013f));}
    std::vector<int> slots(tokens);for(int t=0;t<tokens;++t)slots[t]=2*(tokens-t)-1;
    Device<uint8_t> packed(blob);Device<float> input(x),route(routing),decoded(hidden*ff),gate(tokens*ff),up(tokens*ff),activation(tokens*ff),out(tokens*hidden),temporary(tokens*hidden);
    Device<uint16_t> weights(hidden*ff),bf_input(tokens*hidden);
    Device<int> output_slots(slots);
    std::vector<float> sentinel((tokens*2+1)*hidden,-777);Device<float> parts(sentinel);
    std::vector<uint8_t> canary(tokens*ff*4+64,0xa5);Device<uint8_t> scratch(canary);
    auto project=[&](uint64_t offset,int rows,int cols,const float* values,float* result){
        d::gpu_check(d::dequant_gpu(packed.p+offset,type,decoded.p,uint64_t(rows)*cols,stream));
        d::gpu_check(d::float_to_bf16_gpu(decoded.p,weights.p,uint64_t(rows)*cols,stream));
        d::Matrix matrix{weights.p,30,rows,cols,uint64_t(rows)*cols};
        d::matrix_product(matrix,values,result,tokens,blas,stream,bf_input.p,d::Arithmetic::production);
    };
    project(0,ff,hidden,input.p,gate.p);project(original.up_off,ff,hidden,input.p,up.p);
    auto baseline=[&]{
        d::gpu_check(d::swiglu_gpu(gate.p,up.p,route.p,activation.p,ff,tokens,10,stream));
        d::gpu_check(d::round_bf16_gpu(activation.p,uint64_t(tokens)*ff,stream));
        project(original.down_off,hidden,ff,activation.p,out.p);
        d::gpu_check(d::scatter_expert_gpu(out.p,output_slots.p,parts.p,tokens,6,stream));
    };
    baseline();const auto bfloat=out.read();
    auto unfused=[&]{
        d::gpu_check(d::swiglu_gpu(gate.p,up.p,route.p,activation.p,ff,tokens,10,stream));
        d::gpu_check(d::round_bf16_gpu(activation.p,uint64_t(tokens)*ff,stream));
        if(type==11){
            k::native_quantize_q8_1(activation.p,scratch.p,ff,tokens,stream);
            for(int t=0;t<tokens;++t)k::native_q3_k_mmvq(packed.p+original.down_off,scratch.p+size_t(t)*ff/32*36,out.p+size_t(t)*hidden,ff,hidden,1,stream);
            d::gpu_check(d::round_bf16_gpu(out.p,uint64_t(tokens)*hidden,stream));
        }else project(original.down_off,hidden,ff,activation.p,out.p);
        d::gpu_check(d::scatter_expert_gpu(out.p,output_slots.p,parts.p,tokens,6,stream));
    };
    unfused();const auto reference=out.read();const auto quant_reference=scratch.read();scratch.set(canary);parts.set(sentinel);
    auto fused=[&]{d::gpu_check(d::format_down_gpu(type,packed.p+original.down_off,gate.p,up.p,route.p,temporary.p,parts.p,output_slots.p,hidden,ff,tokens,scratch.p,tokens*ff*4,stream));};
    fused();const auto scattered=parts.read();const auto after=scratch.read();
    std::vector<float> result(tokens*hidden);
    for(int t=0;t<tokens;++t)std::copy_n(scattered.data()+slots[t]*hidden,hidden,result.data()+t*hidden);
    for(int r=0;r<int(sentinel.size()/hidden);++r)if(std::find(slots.begin(),slots.end(),r)==slots.end())
        require(std::equal(scattered.begin()+r*hidden,scattered.begin()+(r+1)*hidden,sentinel.begin()+r*hidden),"scatter wrote an unowned output row");
    const size_t used=type==10?size_t(tokens)*ff*2:size_t(tokens)*ff/32*36;
    require(std::all_of(after.begin()+used,after.end(),[](uint8_t value){return value==0xa5;}),"activation workspace canary");
    const double fusion_error=relative(result,reference),precision_error=relative(result,bfloat);
    std::cerr<<"check type="<<type<<" seed="<<seed<<" tokens="<<tokens<<" fusion="<<fusion_error<<" precision="<<precision_error<<'\n';
    if(type==11&&fusion_error>1e-3){
        size_t scales=0,codes=0,sums=0;
        for(size_t b=0;b<size_t(tokens)*ff/32;++b){
            if(std::memcmp(after.data()+b*36,quant_reference.data()+b*36,2))++scales;
            if(std::memcmp(after.data()+b*36+2,quant_reference.data()+b*36+2,2))++sums;
            for(int j=0;j<32;++j)codes+=after[b*36+4+j]!=quant_reference[b*36+4+j];
        }
        std::cerr<<"Q8_1 different scale blocks="<<scales<<" sum blocks="<<sums<<" codes="<<codes<<'\n';
    }
    require(fusion_error<=1e-3,"fused vs unfused representation exceeds existing operator tolerance");
    require(precision_error<=1e-2,"fused precision vs BF16 exceeds expert tolerance");
    // Serial calls must read the same bytes and use the same reductions as one verification window.
    parts.set(sentinel);
    for(int t=0;t<tokens;++t)d::gpu_check(d::format_down_gpu(type,packed.p+original.down_off,gate.p+t*ff,up.p+t*ff,route.p+t,
        temporary.p+t*hidden,parts.p,output_slots.p+t,hidden,ff,1,scratch.p,tokens*ff*4,stream));
    const auto serial=parts.read();require(std::memcmp(scattered.data(),serial.data(),scattered.size()*4)==0,"ordinary/speculative reduction differs");
    if(tokens==1){
        std::vector<float> guards(7*hidden,-777);Device<float> direct(guards);
        for(int slot:{0,1,5}){
            direct.set(guards);
            d::gpu_check(d::format_down_gpu(type,packed.p+original.down_off,gate.p,up.p,route.p,temporary.p,
                direct.p+slot*hidden,nullptr,hidden,ff,1,scratch.p,tokens*ff*4,stream));
            const auto values=direct.read();
            require(!std::memcmp(values.data()+slot*hidden,result.data(),hidden*4),"direct one-token output differs from indexed scatter");
            for(int row=0;row<7;++row)if(row!=slot)require(std::equal(values.begin()+row*hidden,values.begin()+(row+1)*hidden,guards.begin()+row*hidden),"direct one-token guard overwritten");
        }
    }
    require(d::format_down_gpu(type,packed.p,gate.p,up.p,route.p,temporary.p,parts.p,nullptr,hidden,ff,9,scratch.p,canary.size(),stream)==cudaErrorInvalidValue,"oversized group was accepted");
    require(d::format_down_gpu(type,packed.p,gate.p,up.p,route.p,temporary.p,parts.p,nullptr,hidden,ff,1,scratch.p,1,stream)==cudaErrorInvalidValue,"undersized scratch was accepted");
    auto cpu_format=original;
    // Match GgufExperts' production path. The legacy Q8_K GU probe measures
    // a different activation contract and is retained separately in the bench.
    require(c::native_enable_bf16_gu(cpu_format,error),error.c_str());
    require(type==11?c::native_enable_q3_q8_1_down(cpu_format,error):c::native_enable_bf16_down(cpu_format,error),error.c_str());
    std::vector<float> cpu(tokens*hidden);std::vector<std::vector<uint8_t>> acts(tokens,std::vector<uint8_t>(cpu_format.act_bytes));
    c::ExpertJobMulti job;job.blob=blob.data();job.nt=tokens;job.ff_weights=routing.data();
    for(int t=0;t<tokens;++t){c::native_quant_act(cpu_format,x.data()+t*hidden,acts[t].data());job.nact[t]=acts[t].data();job.out[t]=cpu.data()+t*hidden;}
    c::ExpertPool pool(3,false,true);pool.run_split_multi_native(cpu_format,&job,1);for(auto& value:cpu)value=bf(value);
    const double cpu_gpu=relative(cpu,result),cpu_bfloat=relative(cpu,bfloat);
    std::cerr<<"CPU type="<<type<<" seed="<<seed<<" tokens="<<tokens<<" vs_gpu="<<cpu_gpu<<" vs_bf16="<<cpu_bfloat<<'\n';
    if(cpu_gpu>1e-2||cpu_bfloat>1e-2)++numerical_failures;
    const double a0=gpu_time(baseline,stream),b=gpu_time(fused,stream),a1=gpu_time(baseline,stream);
    std::cout<<type<<','<<seed<<','<<tokens<<','<<fusion_error<<','<<precision_error<<','<<cpu_gpu<<','<<cpu_bfloat<<','<<a0<<','<<b<<','<<a1<<std::endl;
}
void tests(int selected){
    ggml_cpu_init();require(!c::deepseek41_down_fusion_enabled(),"candidate unexpectedly enabled by default");
    c::deepseek41_down_fusion_set_enabled(true);require(c::deepseek41_down_fusion_enabled(),"candidate switch");
    d::Stream stream;cublasHandle_t blas=nullptr;
    require(cublasCreate(&blas)==CUBLAS_STATUS_SUCCESS&&cublasSetStream(blas,stream.value)==CUBLAS_STATUS_SUCCESS&&
        cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH)==CUBLAS_STATUS_SUCCESS,"BLAS setup");
    struct Cleanup{cublasHandle_t h;~Cleanup(){cublasDestroy(h);}} cleanup{blas};
    std::cout<<std::setprecision(12)<<"weight_type,seed,tokens,fused_vs_unfused_l2,gpu_vs_bf16_l2,cpu_vs_gpu_l2,cpu_vs_bf16_l2,A0_ms,B_ms,A1_ms\n";
    for(int type:{10,11})if(!selected||type==selected)for(int offset:{0,1000})for(int tokens:{1,3,8})run(type,41+type+offset,tokens,blas,stream.value);
    require(numerical_failures==0,"mixed CPU/GPU expert exceeds tolerance; all cases are recorded");
}
}
int main(int argc,char** argv){
    int status=0;
    try{tests(argc>1?std::stoi(argv[1]):0);std::cerr<<"all format-down correctness and A/B/A samples completed\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';status=1;}
    // Release only this test process's CUDA context before process teardown.
    const auto cleanup=cudaDeviceReset();if(cleanup!=cudaSuccess){std::cerr<<cudaGetErrorString(cleanup)<<'\n';status=1;}
    return status;
}
