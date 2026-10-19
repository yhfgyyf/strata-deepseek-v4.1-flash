// Real dense prefill projections: BF16 expansion/GEMM vs existing integer Q8_1 MMQ.
#include "runtime_internal.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include <cmath>
#include <iomanip>

namespace d=strata::models::deepseek41;
namespace m=strata::prefill::mmq;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class T>struct Device {
    T* p=nullptr;
    explicit Device(size_t n){d::gpu_check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    ~Device(){cudaFree(p);}
};
template<class F>float timing(F call,cudaStream_t stream){
    for(int i=0;i<3;++i)call();cudaEvent_t a,b;d::gpu_check(cudaEventCreate(&a));d::gpu_check(cudaEventCreate(&b));
    d::gpu_check(cudaEventRecord(a,stream));for(int i=0;i<10;++i)call();d::gpu_check(cudaEventRecord(b,stream));
    d::gpu_check(cudaEventSynchronize(b));float ms=0;d::gpu_check(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms/10;
}
void run(const d::TensorView& t,const std::string& name,int tokens,cublasHandle_t blas,cudaStream_t stream){
    const int k=t.info->shape[0],n=t.info->shape[1];need(t.info->type==8 && k%256==0,"real Q8 matrix view");
    if(!m::built() || !m::fits(8,n))throw std::runtime_error("Q8 MMQ unavailable");
    Device<uint8_t> w(t.bytes+65536),q(m::q8_bytes(tokens,k));Device<uint16_t> bf(uint64_t(n)*k),input_bf(uint64_t(tokens)*k);
    Device<float> x(uint64_t(tokens)*k),reference(uint64_t(tokens)*n),candidate(uint64_t(tokens)*n);
    Device<int32_t> ids(tokens),bounds(2);int32_t limits[]={0,tokens};
    std::vector<float> input(uint64_t(tokens)*k);
    for(size_t i=0;i<input.size();++i)input[i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::sin(float(i)*.013f)));
    d::gpu_check(cudaMemset(w.p,0,t.bytes+65536));d::gpu_check(cudaMemcpy(w.p,t.data,t.bytes,cudaMemcpyHostToDevice));
    d::gpu_check(cudaMemcpy(x.p,input.data(),input.size()*4,cudaMemcpyHostToDevice));d::gpu_check(cudaMemcpy(bounds.p,limits,8,cudaMemcpyHostToDevice));
    m::iota(ids.p,tokens,stream);m::Context context;m::Product p;
    p.w=w.p;p.type=8;p.w_rows=n;p.w_cols=k;p.expert_bytes=t.bytes;p.n=1;p.xq=q.p;p.bounds=bounds.p;p.ids=ids.p;
    p.total_rows=tokens;p.max_rows=tokens;p.dst=candidate.p;p.ld_dst=n;
    auto baseline=[&]{
        d::gpu_check(d::dequant_bf16_gpu(w.p,8,bf.p,uint64_t(n)*k,stream));
        d::Matrix matrix{bf.p,30,n,k,uint64_t(n)*k};d::matrix_product(matrix,x.p,reference.p,tokens,blas,stream,input_bf.p,d::Arithmetic::production);
    };
    auto integer=[&]{m::quantize(x.p,nullptr,q.p,8,k,k,tokens,stream);context.run(p,stream);d::gpu_check(d::round_bf16_gpu(candidate.p,uint64_t(tokens)*n,stream));};
    baseline();integer();d::gpu_check(cudaStreamSynchronize(stream));
    // Bound host comparison scratch, including the actual 8192x32768 output.
    std::vector<float> a(1<<20),b(a.size());double error=0,norm=0,maximum=0;
    for(uint64_t offset=0;offset<uint64_t(tokens)*n;offset+=a.size()){
        const size_t count=std::min<uint64_t>(a.size(),uint64_t(tokens)*n-offset);
        d::gpu_check(cudaMemcpy(a.data(),reference.p+offset,count*4,cudaMemcpyDeviceToHost));d::gpu_check(cudaMemcpy(b.data(),candidate.p+offset,count*4,cudaMemcpyDeviceToHost));
        for(size_t j=0;j<count;++j){need(std::isfinite(a[j])&&std::isfinite(b[j]),"finite prefill outputs");const double delta=double(a[j])-b[j];error+=delta*delta;norm+=double(a[j])*a[j];maximum=std::max(maximum,std::abs(delta));}
    }
    const double rel=std::sqrt(error/norm);need(rel<.02,"prefill Q8_1 error budget");
    std::cout<<std::setprecision(12)<<"{\"kind\":\"prefill_projection\",\"name\":\""<<name<<"\",\"tokens\":"<<tokens<<",\"rows\":"<<n<<",\"cols\":"<<k
             <<",\"rel_l2\":"<<rel<<",\"max_abs\":"<<maximum<<",\"bf16_with_dequant_ms\":"<<timing(baseline,stream)<<",\"q8_1_mmq_ms\":"<<timing(integer,stream)
             <<",\"q8_activation_bytes\":"<<m::q8_bytes(tokens,k)<<"}"<<std::endl;
}
}
int main(int argc,char** argv){try{
    need(argc==2,"MODEL");d::Model model(argv[1]);d::Stream stream;cublasHandle_t blas;
    need(cublasCreate(&blas)==CUBLAS_STATUS_SUCCESS,"BLAS");cublasSetStream(blas,stream.value);cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH);
    std::cout<<"{\"kind\":\"supported\",\"q2_k\":"<<(m::supported(10)?"true":"false")<<",\"q3_k\":"<<(m::supported(11)?"true":"false")<<"}\n";
    for(const char* name:{"blk.0.attn_q_b.weight","blk.0.attn_output_b.weight","blk.0.ffn_gate_shexp.weight","blk.0.ffn_down_shexp.weight"})
        for(int tokens:{16,128,8192})run(model.tensor(name),name,tokens,blas,stream.value);
    cublasDestroy(blas);std::cout<<"{\"kind\":\"completed\"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
