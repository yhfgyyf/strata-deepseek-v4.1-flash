#include "strata/models/deepseek41/kernels.hpp"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

namespace d=strata::models::deepseek41;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
template<class T>struct Device {
    T* p=nullptr;size_t n;
    explicit Device(size_t n):n(n){check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    ~Device(){cudaFree(p);}
    void set(const std::vector<T>& x){need(x.size()==n,"device fixture size");check(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T>x(n);check(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
void run(int type,int rows,int cols,int seed) {
    std::mt19937 random(seed);std::normal_distribution<float> normal(0,.15f);
    const size_t n=size_t(rows)*cols;
    std::vector<float> weights(n);for(auto& x:weights)x=normal(random);
    // Include scale zero, signs, large/small values and BF16 halfway input values.
    for(int i=0;i<cols;++i)weights[i]=0;
    std::vector<uint8_t> packed(ggml_row_size(static_cast<ggml_type>(type),cols)*rows);
    need(ggml_quantize_chunk(static_cast<ggml_type>(type),weights.data(),packed.data(),0,rows,cols,nullptr)==packed.size(),"quantization size");
    Device<uint8_t> dp(packed.size());dp.set(packed);
    Device<float> decoded(n);Device<uint16_t> reference(n+32),fused(n+32);
    std::vector<uint16_t> guard(n+32,0x5a5a);reference.set(guard);fused.set(guard);
    check(d::dequant_gpu(dp.p,type,decoded.p,n,nullptr));
    check(d::float_to_bf16_gpu(decoded.p,reference.p,n,nullptr));
    check(d::dequant_bf16_gpu(dp.p,type,fused.p,n,nullptr));
    need(reference.get()==fused.get(),"fused dequant BF16 bytes/canary differ");
    const auto floats=decoded.get();const auto bf=fused.get();
    for(size_t i=0;i<n;++i) {
        const auto value=ggml_fp32_to_bf16(floats[i]);uint16_t bits;std::memcpy(&bits,&value,sizeof(bits));
        need(bf[i]==bits,"CPU/GPU BF16 rounding differs");
    }
    for(int tokens:{1,3,8}) {
        const int stride=rows+13;
        std::vector<float> input(size_t(tokens)*cols),sentinel(size_t(tokens)*stride+32,-777);
        for(auto& x:input)x=ggml_bf16_to_fp32(ggml_fp32_to_bf16(normal(random)));
        Device<float> dx(input.size()),old(sentinel.size()),now(sentinel.size());dx.set(input);old.set(sentinel);now.set(sentinel);
        for(int t=0;t<tokens;++t) {
            check(d::packed_gemv_gpu(reference.p,30,dx.p+size_t(t)*cols,old.p+size_t(t)*stride,rows,cols,1,nullptr));
            check(d::packed_bf16_gemv_gpu(dp.p,type,dx.p+size_t(t)*cols,now.p+size_t(t)*stride,rows,cols,1,nullptr));
        }
        const auto a=old.get(),b=now.get();
        need(!std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),"direct GEMV reduction/strides/canary differ");
        for(float x:b)need(std::isfinite(x),"GEMV nonfinite");
    }
    std::cout<<"exact type="<<type<<" rows="<<rows<<" cols="<<cols<<" seed="<<seed<<"\n";
}
}
void real(const char* path) {
    d::Model model(path);
    for(int layer:{0,13,14,39})for(int expert:{0,2,128,383}) {
        const auto view=model.expert(layer,expert);
        for(const auto* source:{view.gate,view.up,view.down}) {
            const bool down=source==view.down;const int rows=down?5120:2304,cols=down?2304:5120;const size_t n=size_t(rows)*cols;
            const uint64_t bytes=down?view.blob_bytes-2*view.projection_bytes:view.projection_bytes;
            std::vector<uint8_t> data(source,source+bytes);Device<uint8_t> packed(bytes);packed.set(data);
            Device<float> decoded(n);Device<uint16_t> a(n),b(n);
            check(d::dequant_gpu(packed.p,view.type,decoded.p,n,nullptr));check(d::float_to_bf16_gpu(decoded.p,a.p,n,nullptr));
            check(d::dequant_bf16_gpu(packed.p,view.type,b.p,n,nullptr));const auto old=a.get(),now=b.get();
            size_t bad=0;for(size_t i=0;i<n;++i)bad+=old[i]!=now[i];
            std::cout<<"real layer="<<layer<<" expert="<<expert<<" down="<<down<<" BF16_byte_mismatches="<<bad<<'\n';
            need(!bad,"real GGUF fused weight mismatch");
            std::vector<float> input(size_t(3)*cols);
            for(size_t i=0;i<input.size();++i)input[i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::sin(float(i)*.013f)));
            Device<float> x(input.size()),old_gemv(size_t(3)*rows),new_gemv(size_t(3)*rows);x.set(input);
            check(d::packed_gemv_gpu(a.p,30,x.p,old_gemv.p,rows,cols,3,nullptr));
            check(d::packed_bf16_gemv_gpu(packed.p,view.type,x.p,new_gemv.p,rows,cols,3,nullptr));
            const auto y=old_gemv.get(),z=new_gemv.get();
            need(!std::memcmp(y.data(),z.data(),y.size()*sizeof(float)),"real GGUF direct multi-token GEMV mismatch");
        }
    }
}
int main(int argc,char** argv){try {
    for(int type:{10,11})for(int seed:{41,42}) {
        run(type,17,5120,seed);run(type,2304,5120,seed);run(type,5120,2304,seed);
    }
    if(argc==2)real(argv[1]);
    check(cudaDeviceSynchronize());std::cout<<"DIRECT_BF16_AND_GEMV_EXACT_PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
