#include "runtime_internal.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <random>

namespace d=strata::models::deepseek41;
namespace k=strata::kernels;
namespace c=strata::kernels::cpu;
namespace {
void need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
template<class T>struct Device {
    T* p=nullptr;size_t n;
    explicit Device(size_t count):n(count){d::gpu_check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    explicit Device(const std::vector<T>& x):Device(x.size()){set(x);}
    ~Device(){cudaFree(p);}
    void set(const std::vector<T>& x){
        need(x.size()==n,"fixture shape");d::gpu_check(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));
        // The tested kernels use a nonblocking stream, which does not wait for
        // the default-stream transfer of pageable fixture storage.
        d::gpu_check(cudaStreamSynchronize(nullptr));
    }
    std::vector<T> get(){d::gpu_check(cudaDeviceSynchronize());std::vector<T>x(n);d::gpu_check(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
float bf(float x){return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));}
double error(const std::vector<float>& a,const std::vector<float>& b){
    need(a.size()==b.size(),"metric shape");double e=0,n=0;
    for(size_t i=0;i<a.size();++i){need(std::isfinite(a[i])&&std::isfinite(b[i]),"finite outputs");const double delta=double(a[i])-b[i];e+=delta*delta;n+=double(b[i])*b[i];}
    return std::sqrt(e/std::max(n,1e-30));
}
template<class F>double gpu_ms(F operation,cudaStream_t stream){
    for(int i=0;i<3;++i)operation();cudaEvent_t a,b;d::gpu_check(cudaEventCreate(&a));d::gpu_check(cudaEventCreate(&b));
    d::gpu_check(cudaEventRecord(a,stream));for(int i=0;i<20;++i)operation();d::gpu_check(cudaEventRecord(b,stream));
    d::gpu_check(cudaEventSynchronize(b));float ms=0;d::gpu_check(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms/20;
}
void dense(const uint8_t* weights,int rows,int cols,const std::string& name,cudaStream_t stream){
    std::vector<uint8_t> packed(weights,weights+uint64_t(rows)*cols/32*34);Device<uint8_t> w(packed);
    Device<uint16_t> wb(uint64_t(rows)*cols);Device<float> x(8*cols),a(8*rows),b(8*rows),qout(8*rows);
    Device<uint8_t> q(8ull*cols/32*36);
    std::vector<float> input(8*cols);for(size_t i=0;i<input.size();++i)input[i]=bf(std::sin(float(i)*.013f));x.set(input);
    d::gpu_check(d::dequant_bf16_gpu(w.p,8,wb.p,uint64_t(rows)*cols,stream));
    for(int tokens:{1,2,5,6,8}){
        auto old=[&]{d::gpu_check(d::packed_gemv_gpu(wb.p,30,x.p,a.p,rows,cols,tokens,stream));};
        auto direct=[&]{d::gpu_check(d::packed_bf16_gemv_gpu(w.p,8,x.p,b.p,rows,cols,tokens,stream));};
        auto quant=[&]{k::native_quantize_q8_1(x.p,q.p,cols,tokens,stream);k::native_q8_0_mmvq(w.p,q.p,qout.p,cols,rows,tokens,stream);};
        old();direct();quant();const auto original=a.get(),exact=b.get(),approx=qout.get();
        if(name=="synthetic"){
            strata::core::CapturedGraph graph;std::string why;need(graph.begin(stream,why),why.c_str());quant();need(graph.end(stream,why),why.c_str());
            need(graph.launch(stream,why),why.c_str());auto replay=qout.get();need(!std::memcmp(replay.data(),approx.data(),tokens*rows*4),"Q8 graph first replay exact");
            auto changed=input;for(auto& v:changed)v=-v*.5f;x.set(changed);quant();const auto reference=qout.get();
            need(graph.launch(stream,why),why.c_str());replay=qout.get();need(!std::memcmp(replay.data(),reference.data(),tokens*rows*4),"Q8 graph must reread changing activations");x.set(input);
            std::cout<<"{\"kind\":\"graph\",\"nodes\":"<<graph.nodes()<<",\"dynamic_input_exact\":true}\n";
        }
        need(!std::memcmp(original.data(),exact.data(),size_t(tokens)*rows*4),"packed dense BF16 exact oracle");
        const double rel=error(std::vector<float>(approx.begin(),approx.begin()+tokens*rows),std::vector<float>(original.begin(),original.begin()+tokens*rows));
        need(rel<.02,"dense Q8_1 relative error budget");
        std::cout<<"{\"kind\":\"dense\",\"name\":\""<<name<<"\",\"rows\":"<<rows<<",\"cols\":"<<cols<<",\"tokens\":"<<tokens
                 <<",\"exact\":true,\"q8_1_rel_l2\":"<<rel<<",\"bf16_ms\":"<<gpu_ms(old,stream)<<",\"packed_bf16_ms\":"<<gpu_ms(direct,stream)<<",\"q8_1_ms\":"<<gpu_ms(quant,stream)<<"}\n";
    }
}
void expert(const std::vector<uint8_t>& blob,int type,int rows,int cols,const std::string& name,cudaStream_t stream){
    c::NativeFmt f;std::string why;need(c::native_fmt(type,type,cols,rows,f,why),why.c_str());
    const size_t gu_bytes=ggml_row_size(static_cast<ggml_type>(type),cols)*rows;
    need(blob.size()>=gu_bytes*2,"GU fixture bytes");Device<uint8_t>w(blob);
    std::vector<float> input(8*cols);for(size_t i=0;i<input.size();++i)input[i]=bf(std::sin(float(i)*.013f));Device<float>x(input);
    Device<float> gate(8*rows),up(8*rows),rg(8*rows),ru(8*rows);Device<uint8_t>q(8ull*cols/32*36);
    for(int tokens:{1,2,3,5,6,8}){
        auto old=[&]{
            d::gpu_check(d::packed_bf16_gemv_gpu(w.p,type,x.p,rg.p,rows,cols,tokens,stream));d::gpu_check(d::round_bf16_gpu(rg.p,tokens*rows,stream));
            d::gpu_check(d::packed_bf16_gemv_gpu(w.p+gu_bytes,type,x.p,ru.p,rows,cols,tokens,stream));d::gpu_check(d::round_bf16_gpu(ru.p,tokens*rows,stream));
        };old();const auto g=rg.get(),u=ru.get();const double old_ms=gpu_ms(old,stream);
        for(int mode:{1,2}){
            auto operation=[&]{d::gpu_check(d::expert_gu_gpu(w.p,type,x.p,gate.p,up.p,rows,cols,tokens,mode,q.p,stream));};
            operation();const auto ng=gate.get(),nu=up.get();
            if(mode==1)need(!std::memcmp(g.data(),ng.data(),tokens*rows*4)&&!std::memcmp(u.data(),nu.data(),tokens*rows*4),"fused BF16 GU exact oracle");
            const double eg=error(std::vector<float>(ng.begin(),ng.begin()+tokens*rows),std::vector<float>(g.begin(),g.begin()+tokens*rows));
            const double eu=error(std::vector<float>(nu.begin(),nu.begin()+tokens*rows),std::vector<float>(u.begin(),u.begin()+tokens*rows));
            need(std::max(eg,eu)<.02,"integer GU relative error budget");
            std::cout<<"{\"kind\":\"expert_gpu\",\"name\":\""<<name<<"\",\"type\":"<<type<<",\"mode\":"<<mode<<",\"tokens\":"<<tokens
                     <<",\"exact\":"<<(!std::memcmp(g.data(),ng.data(),tokens*rows*4)&&!std::memcmp(u.data(),nu.data(),tokens*rows*4)?"true":"false")
                     <<",\"gate_rel_l2\":"<<eg<<",\"up_rel_l2\":"<<eu<<",\"reference_ms\":"<<old_ms<<",\"candidate_ms\":"<<gpu_ms(operation,stream)<<"}\n";
        }
    }
    f.bf16_io=true;f.swiglu_limit=10;need(c::native_enable_bf16_gu(f,why),why.c_str());
    std::vector<uint8_t> act(f.act_bytes);c::native_quant_act(f,input.data(),act.data());const void* in[]={act.data()};
    std::vector<float> a(rows),b(rows);float* ap[]={a.data()};float* bp[]={b.data()};
    auto cpu=[&](c::NativeFmt& fmt,const void* const* activation,float* const* output){
        const auto begin=std::chrono::steady_clock::now();c::native_gu_rows(fmt,blob.data(),activation,1,output,0,rows);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    };
    const double baseline=cpu(f,in,ap);need(c::native_enable_q8_0_gu(f,why),why.c_str());act.resize(f.act_bytes);
    c::native_quant_act(f,input.data(),act.data());in[0]=act.data();const double candidate=cpu(f,in,bp);
    need(error(b,a)<.04,"CPU Q8_0 GU activation error budget");
    // Independent scalar double dot on explicitly decoded Q8_0 inputs/weights.
    std::vector<float> xq(cols),decoded(cols),ref(rows);ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(act.data(),xq.data(),cols);
    const auto* wt=ggml_get_type_traits(static_cast<ggml_type>(type));
    for(int r=0;r<rows;++r){
        wt->to_float(blob.data()+r*f.gu_row,decoded.data(),cols);double gv=0,uv=0;
        for(int j=0;j<cols;++j)gv+=double(decoded[j])*xq[j];
        wt->to_float(blob.data()+f.up_off+r*f.gu_row,decoded.data(),cols);for(int j=0;j<cols;++j)uv+=double(decoded[j])*xq[j];
        const float g=std::min(bf(gv),10.f),u=std::clamp(bf(uv),-10.f,10.f);ref[r]=(g/(1.f+std::exp(-g)))*u;
    }
    need(error(b,ref)<.004,"CPU Q8_0 independent dot oracle");
    std::cout<<"{\"kind\":\"expert_cpu\",\"name\":\""<<name<<"\",\"type\":"<<type<<",\"rel_l2\":"<<error(b,a)
             <<",\"independent_rel_l2\":"<<error(b,ref)<<",\"bf16_ms\":"<<baseline<<",\"q8_0_ms\":"<<candidate<<"}\n";
    for(int nt:{2,5,6,8}){
        std::vector<uint8_t> activations(size_t(nt)*f.act_bytes);std::vector<const void*> inputs(nt);
        std::vector<float> actual(size_t(nt)*rows),reference(actual.size());std::vector<float*> outputs(nt);
        for(int t=0;t<nt;++t){
            auto* encoded=activations.data()+size_t(t)*f.act_bytes;c::native_quant_act(f,input.data()+size_t(t)*cols,encoded);
            inputs[t]=encoded;outputs[t]=actual.data()+size_t(t)*rows;
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(encoded,xq.data(),cols);
            for(int r=0;r<rows;++r){
                double gv=0,uv=0;wt->to_float(blob.data()+r*f.gu_row,decoded.data(),cols);for(int j=0;j<cols;++j)gv+=double(decoded[j])*xq[j];
                wt->to_float(blob.data()+f.up_off+r*f.gu_row,decoded.data(),cols);for(int j=0;j<cols;++j)uv+=double(decoded[j])*xq[j];
                const float g=std::min(bf(gv),10.f),u=std::clamp(bf(uv),-10.f,10.f);reference[size_t(t)*rows+r]=(g/(1.f+std::exp(-g)))*u;
            }
        }
        c::native_gu_rows(f,blob.data(),inputs.data(),nt,outputs.data(),0,rows);
        const double rel=error(actual,reference);need(rel<.004,"CPU verification Q8_0 independent dot oracle");
        std::cout<<"{\"kind\":\"expert_cpu_verify\",\"name\":\""<<name<<"\",\"type\":"<<type<<",\"tokens\":"<<nt<<",\"independent_rel_l2\":"<<rel<<"}\n";
    }
}
}
void routing(cudaStream_t stream){
    for(int experts:{128,384})for(int tokens:{1,17,8192}){
        const int top=experts==384?6:3;
        std::vector<float> logits(size_t(tokens)*experts),bias(experts),vl(experts);
        std::vector<uint8_t> mask(tokens);
        for(int e=0;e<experts;++e){bias[e]=float(e/4%7)*.01f;vl[e]=-bias[e];}
        for(int t=0;t<tokens;++t){mask[t]=t%2;for(int e=0;e<experts;++e)logits[size_t(t)*experts+e]=t%3?std::sin(float((t+1)*(e+1))*.017f)*100:0;}
        Device<float> x(logits),b(bias),v(vl),a(size_t(tokens)*top),p(size_t(tokens)*top);Device<uint8_t> m(mask);
        Device<int> ai(size_t(tokens)*top),pi(size_t(tokens)*top);
        auto old=[&]{d::gpu_check(d::route_gpu(x.p,b.p,v.p,m.p,ai.p,a.p,experts,top,tokens,1.5f,stream));};
        auto parallel=[&]{d::gpu_check(d::route_parallel_gpu(x.p,b.p,v.p,m.p,pi.p,p.p,experts,top,tokens,1.5f,stream));};
        old();parallel();need(ai.get()==pi.get() && a.get()==p.get(),"parallel routing stable ties, IDs and weights exact");
        std::cout<<"{\"kind\":\"router\",\"experts\":"<<experts<<",\"tokens\":"<<tokens<<",\"exact\":true,\"reference_ms\":"<<gpu_ms(old,stream)<<",\"parallel_ms\":"<<gpu_ms(parallel,stream)<<"}\n";
    }
}
int main(int argc,char** argv){try{
    std::cout<<std::setprecision(12);d::Stream stream;
    routing(stream.value);
    std::mt19937 rng(416);std::normal_distribution<float> normal(0,.015f);
    for(int type:{8,10,11}){
        const int cols=5120,rows=type==8?1280:2304;std::vector<float> values(size_t(rows)*cols*2);
        for(auto& x:values)x=normal(rng);std::vector<uint8_t> packed(ggml_row_size(static_cast<ggml_type>(type),cols)*rows*2);
        ggml_quantize_chunk(static_cast<ggml_type>(type),values.data(),packed.data(),0,rows*2,cols,nullptr);
        if(type==8)dense(packed.data(),rows,cols,"synthetic",stream.value);else expert(packed,type,rows,cols,"synthetic",stream.value);
    }
    if(argc==2){d::Model model(argv[1]);
        for(int l:{0,14,39}){
            for(const char* name:{"attn_q_a","attn_q_b","attn_output_b","ffn_gate_shexp","ffn_down_shexp"}){
                const std::string key="blk."+std::to_string(l)+"."+name+".weight";const auto& t=model.tensor(key);
                dense(t.data,t.info->shape[1],t.info->shape[0],key,stream.value);
            }
            const auto e=model.expert(l,2);std::vector<uint8_t> gu(e.gate,e.gate+e.projection_bytes);
            gu.insert(gu.end(),e.up,e.up+e.projection_bytes);expert(gu,e.type,2304,5120,"layer"+std::to_string(l),stream.value);
        }
    }
    std::cout<<"{\"kind\":\"completed\"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
