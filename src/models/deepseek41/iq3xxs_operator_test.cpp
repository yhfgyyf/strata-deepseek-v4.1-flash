// Standalone real-weight operators; never constructs Runtime or Session.
#include "runtime_internal.hpp"
#include "strata/models/deepseek41/format_down.hpp"
#include "strata/kernels/cpu/deepseek41_down.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <chrono>

namespace d=strata::models::deepseek41;
namespace {
int failures=0;
void need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
float bf(float x){return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));}
template<class T>struct Device {
    T* p=nullptr;size_t size;
    explicit Device(size_t n):size(n){d::gpu_check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    ~Device(){cudaFree(p);}
    void set(const T* x,size_t n){
        need(n<=size,"upload bound");
        d::gpu_check(cudaMemcpy(p,x,n*sizeof(T),cudaMemcpyHostToDevice));
        // Pageable H2D can return after staging. The test kernels run in a
        // nonblocking stream, so explicitly finish this default-stream upload.
        d::gpu_check(cudaStreamSynchronize(nullptr));
    }
    std::vector<T> get(size_t n=0)const{
        if(!n)n=size;need(n<=size,"download bound");std::vector<T> out(n);
        d::gpu_check(cudaMemcpy(out.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return out;
    }
};
void decode(int type,const void* input,float* out,size_t n) {
    if(type==0){std::memcpy(out,input,n*4);return;}
    const auto* traits=ggml_get_type_traits(static_cast<ggml_type>(type));
    need(traits&&traits->to_float,"independent ggml decoder missing");
    traits->to_float(input,out,n);
}
uint64_t fingerprint(const uint8_t* bytes,size_t n){
    uint64_t h=14695981039346656037ull;
    for(size_t i=0;i<n;++i){h^=bytes[i];h*=1099511628211ull;}return h;
}
struct Metric {
    double maximum=0,relative=0,reference_max=0;size_t nonfinite=0;
    Metric(const std::vector<float>& got,const std::vector<float>& ref){
        need(got.size()==ref.size(),"metric shape");double sum=0,norm=0;
        for(size_t i=0;i<got.size();++i){
            if(!std::isfinite(got[i])||!std::isfinite(ref[i])){++nonfinite;continue;}
            const double delta=double(got[i])-ref[i];maximum=std::max(maximum,std::abs(delta));
            reference_max=std::max(reference_max,std::abs(double(ref[i])));sum+=delta*delta;norm+=double(ref[i])*ref[i];
        }
        relative=std::sqrt(sum/std::max(norm,1e-30));
    }
    void json()const{std::cout<<"\"max_abs\":"<<maximum<<",\"relative_l2\":"<<relative<<",\"reference_max\":"<<reference_max<<",\"nonfinite\":"<<nonfinite;}
};
std::vector<int> positions(int n){
    std::vector<int> out{0,n/2,n-1};std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;
}
void samples(const d::Model& model,const char* arm,cudaStream_t stream){
    size_t tensors=0,cases=0,values=0;
    for(const auto& [name,v]:model.tensors()){
        const int type=v.info->type;
        if(v.role==d::TensorRole::token_map)continue;
        const size_t cols=v.info->shape[0],rows=v.info->elements()/cols;
        for(size_t row:std::set<size_t>{0,rows/2,rows-1}){
            const auto* source=v.data+row*v.row_bytes;
            std::vector<uint8_t> packed(source,source+v.row_bytes);
            std::vector<float> reference(cols);decode(type,packed.data(),reference.data(),cols);
            Device<uint8_t> input(packed.size());input.set(packed.data(),packed.size());
            Device<float> actual(cols+16);Device<uint16_t> actual_bf(cols+16);
            std::vector<float> guard(cols+16,-777);actual.set(guard.data(),guard.size());
            std::vector<uint16_t> bf_guard(cols+16,0x5a5a);actual_bf.set(bf_guard.data(),bf_guard.size());
            d::gpu_check(d::dequant_gpu(input.p,type,actual.p,cols,stream));
            d::gpu_check(d::dequant_bf16_gpu(input.p,type,actual_bf.p,cols,stream));
            d::gpu_check(cudaStreamSynchronize(stream));
            auto got=actual.get();const auto b=actual_bf.get();
            bool canary=true;for(size_t i=cols;i<cols+16;++i)canary&=got[i]==-777&&b[i]==0x5a5a;
            got.resize(cols);Metric error(got,reference);size_t mismatches=0;
            for(size_t i=0;i<cols;++i){const auto expect=ggml_fp32_to_bf16(reference[i]);uint16_t bits;std::memcpy(&bits,&expect,2);mismatches+=b[i]!=bits;}
            const bool pass=!error.nonfinite&&error.maximum<=1e-6&&error.relative<=1e-6&&!mismatches&&canary;
            failures+=!pass;
            std::cout<<"{\"kind\":\"dequant\",\"arm\":\""<<arm<<"\",\"tensor\":\""<<name<<"\",\"type\":"<<type
                <<",\"row\":"<<row<<",\"values\":"<<cols<<",\"sample_fnv1a64\":\""<<std::hex<<fingerprint(packed.data(),packed.size())<<std::dec<<"\",";
            error.json();std::cout<<",\"bf16_mismatches\":"<<mismatches<<",\"canary\":"<<(canary?"true":"false")
                <<",\"pass\":"<<(pass?"true":"false")<<"}\n";
            ++cases;values+=cols;
        }
        ++tensors;
    }
    std::cout<<"{\"kind\":\"dequant_summary\",\"arm\":\""<<arm<<"\",\"tensors\":"<<tensors<<",\"cases\":"<<cases<<",\"values\":"<<values<<"}\n"<<std::flush;
}

struct Projection {
    const d::TensorView& view;
    int rows,cols,type;
    Device<uint8_t> packed;
    Device<uint16_t> weights;
    Projection(const d::TensorView& v,cudaStream_t stream)
        :view(v),rows(v.role==d::TensorRole::routed_expert?v.info->shape[1]:v.info->elements()/v.info->shape[0]),cols(v.info->shape[0]),type(v.info->type),
         packed(v.row_bytes*rows),weights(size_t(rows)*cols){
        // One expert (ID 0) for 3D tensors, the complete matrix for 2D tensors.
        packed.set(v.data,packed.size);
        d::gpu_check(d::dequant_bf16_gpu(packed.p,type,weights.p,weights.size,stream));
        d::gpu_check(cudaStreamSynchronize(stream));
    }
    void run(const float* x,float* out,int tokens,uint16_t* scratch,cublasHandle_t blas,cudaStream_t stream,bool expand=false)const{
        if(expand)d::gpu_check(d::dequant_bf16_gpu(packed.p,type,weights.p,weights.size,stream));
        const d::Matrix matrix{weights.p,30,rows,cols,uint64_t(rows)*cols};
        d::matrix_product(matrix,x,out,tokens,blas,stream,scratch,d::Arithmetic::production);
    }
    std::vector<float> reference(const std::vector<float>& x,int tokens)const{
        std::vector<float> result,row(cols);
        for(int t:positions(tokens))for(int r:positions(rows)){
            decode(type,view.data+size_t(r)*view.row_bytes,row.data(),cols);
            double sum=0;for(int k=0;k<cols;++k)sum+=double(bf(row[k]))*x[size_t(t)*cols+k];
            result.push_back(bf(float(sum)));
        }
        return result;
    }
    std::vector<float> selected(const float* out,int tokens)const{
        std::vector<float> result;
        for(int t:positions(tokens))for(int r:positions(rows)){
            float value;d::gpu_check(cudaMemcpy(&value,out+size_t(t)*rows+r,4,cudaMemcpyDeviceToHost));result.push_back(value);
        }
        return result;
    }
};
template<class F>double time_gpu(F&& work,cudaStream_t stream,int repeats){
    for(int i=0;i<3;++i)work();
    cudaEvent_t a,b;d::gpu_check(cudaEventCreate(&a));d::gpu_check(cudaEventCreate(&b));
    d::gpu_check(cudaEventRecord(a,stream));for(int i=0;i<repeats;++i)work();
    d::gpu_check(cudaEventRecord(b,stream));d::gpu_check(cudaEventSynchronize(b));
    float ms=0;d::gpu_check(cudaEventElapsedTime(&ms,a,b));cudaEventDestroy(a);cudaEventDestroy(b);return ms/repeats;
}
std::string suffix(const std::string& name){
    if(name.rfind("blk.",0)!=0)return name;
    return name.substr(name.find('.',4)+1);
}
void projections(const d::Model& old,const d::Model& now,cublasHandle_t blas,cudaStream_t stream,bool timing){
    std::set<std::string> covered;int cases=0;
    for(const auto& [name,v]:now.tensors()){
        if(v.info->shape.size()<2||v.role==d::TensorRole::engram||v.role==d::TensorRole::token_map)continue;
        std::string old_name=name;
        if(suffix(name)=="engram_gate.weight"||suffix(name)=="engram_q.weight"||suffix(name)=="engram_k.weight")continue; // Separate nonlinear gate test.
        if(!old.ple_engram()&&suffix(name)=="engram_kv.weight")old_name.replace(old_name.find("engram_kv"),9,"engram_wkv");
        if(old.ple_engram()&&suffix(name)=="engram_wkv.weight")old_name.replace(old_name.find("engram_wkv"),10,"engram_kv");
        const auto& ov=old.tensor(old_name);
        need(ov.info->shape[0]==v.info->shape[0]&&ov.info->elements()==v.info->elements(),"paired flattened shape");
        const int rows=v.role==d::TensorRole::routed_expert?v.info->shape[1]:v.info->elements()/v.info->shape[0],cols=v.info->shape[0];
        std::ostringstream key;key<<suffix(name)<<':'<<rows<<':'<<cols<<':'<<ov.info->type<<':'<<v.info->type;
        if(!covered.insert(key.str()).second)continue;
        Projection a(ov,stream),b(v,stream);
        for(int tokens:{1,2,5,8,9,128,512,8192}){
            // Heads at 8192 output rows consume 4.2 GB, still bounded well below VRAM capacity.
            std::vector<float> input(size_t(tokens)*cols);
            for(size_t i=0;i<input.size();++i)input[i]=bf(float(.25*std::sin(double(i%cols)*.013+double(i/cols)*.031)));
            Device<float> x(input.size()),ya(size_t(tokens)*rows),yb(size_t(tokens)*rows);x.set(input.data(),input.size());
            Device<uint16_t> scratch(input.size());
            a.run(x.p,ya.p,tokens,scratch.p,blas,stream);b.run(x.p,yb.p,tokens,scratch.p,blas,stream);
            d::gpu_check(cudaStreamSynchronize(stream));
            const auto av=a.selected(ya.p,tokens),bv=b.selected(yb.p,tokens);
            for(int arm=0;arm<2;++arm){
                const auto reference=(arm?b:a).reference(input,tokens);
                const Metric error(arm?bv:av,reference);
                const bool pass=!error.nonfinite&&error.maximum<=2e-4+.0078125*error.reference_max&&error.relative<=.005;
                failures+=!pass;
                std::cout<<"{\"kind\":\"projection_accuracy\",\"arm\":\""<<(arm?"new":"old")<<"\",\"tensor\":\""<<name
                    <<"\",\"tokens\":"<<tokens<<",\"rows\":"<<rows<<",\"cols\":"<<cols<<",\"reference\":\"CPU ggml decode + BF16 + FP64 dot + BF16\",\"sampled_outputs\":"<<reference.size()<<",";
                error.json();std::cout<<",\"pass\":"<<(pass?"true":"false")<<"}\n";
            }
            const Metric delta(bv,av);
            std::cout<<"{\"kind\":\"checkpoint_and_quantization_delta\",\"tensor\":\""<<name<<"\",\"tokens\":"<<tokens<<",";
            delta.json();std::cout<<"}\n";
            if(timing)for(bool expand:{false,true}){
                const int repeats=tokens<=8?40:tokens<=512?10:3;
                const auto fa=[&]{a.run(x.p,ya.p,tokens,scratch.p,blas,stream,expand);};
                const auto fb=[&]{b.run(x.p,yb.p,tokens,scratch.p,blas,stream,expand);};
                const double a0=time_gpu(fa,stream,repeats),b0=time_gpu(fb,stream,repeats),b1=time_gpu(fb,stream,repeats),a1=time_gpu(fa,stream,repeats);
                std::cout<<"{\"kind\":\"projection_timing\",\"path\":\""<<(expand?"dequant_bf16_plus_projection":"predecoded_bf16_projection")
                    <<"\",\"tensor\":\""<<name<<"\",\"tokens\":"<<tokens<<",\"rows\":"<<rows<<",\"cols\":"<<cols
                    <<",\"old_type\":"<<a.type<<",\"new_type\":"<<b.type<<",\"repeats\":"<<repeats
                    <<",\"A0_ms\":"<<a0<<",\"B0_ms\":"<<b0<<",\"B1_ms\":"<<b1<<",\"A1_ms\":"<<a1<<"}\n";
            }
            ++cases;std::cout<<std::flush;
        }
    }
    std::cout<<"{\"kind\":\"projection_summary\",\"unique_tensor_families\":"<<covered.size()<<",\"cases\":"<<cases<<"}\n";
}

// Independent scalar model of the 32-value activation encoding used by the
// existing Q2/Q3 fused GPU operators. This does not call the CUDA quantizer.
void q81_values(std::vector<float>& values){
    need(values.size()%32==0,"Q8_1 reference blocks");
    for(size_t b=0;b<values.size();b+=32){
        float maximum=0;for(int j=0;j<32;++j)maximum=std::max(maximum,std::abs(values[b+j]));
        const float scale=maximum*(1.f/127.f);
        const float stored=ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
        for(int j=0;j<32;++j)values[b+j]=maximum==0?0:stored*std::clamp(std::round(values[b+j]/scale),-127.f,127.f);
    }
}
struct Expert {
    d::ExpertSlice slice;
    std::vector<uint8_t> host;
    Device<uint8_t> packed;
    Device<uint16_t> weights;
    explicit Expert(d::ExpertSlice s):slice(s),host(s.blob_bytes),packed(s.blob_bytes),weights(5120ull*2304){
        std::memcpy(host.data(),s.gate,s.projection_bytes);
        std::memcpy(host.data()+s.projection_bytes,s.up,s.projection_bytes);
        std::memcpy(host.data()+2*s.projection_bytes,s.down,s.blob_bytes-2*s.projection_bytes);
        packed.set(host.data(),host.size());
    }
    void projection(int type,const uint8_t* data,const float* x,float* y,int rows,int cols,int tokens,
                    uint16_t* scratch,cublasHandle_t blas,cudaStream_t stream){
        if(tokens<=8&&(type==10||type==11||type==39)){
            d::gpu_check(d::packed_bf16_gemv_gpu(data,type,x,y,rows,cols,tokens,stream));
            d::gpu_check(d::round_bf16_gpu(y,size_t(tokens)*rows,stream));return;
        }
        d::gpu_check(d::dequant_bf16_gpu(data,type,weights.p,size_t(rows)*cols,stream));
        d::Matrix matrix{weights.p,30,rows,cols,uint64_t(rows)*cols};
        d::matrix_product(matrix,x,y,tokens,blas,stream,scratch,d::Arithmetic::production,0,0,0,0,true,true);
    }
    void gu(const float* x,float* gate,float* up,int tokens,void* q,uint16_t* scratch,cublasHandle_t blas,cudaStream_t stream){
        if(tokens<=8&&(slice.type==10||slice.type==11))
            d::gpu_check(d::expert_gu_verify_gpu(packed.p,slice.type,x,gate,up,2304,5120,tokens,2,q,stream));
        else{
            projection(slice.type,packed.p,x,gate,2304,5120,tokens,scratch,blas,stream);
            projection(slice.type,packed.p+slice.projection_bytes,x,up,2304,5120,tokens,scratch,blas,stream);
        }
    }
    void down(const float* gate,const float* up,const float* routing,float* activation,float* out,float* temporary,
              int tokens,void* q,uint16_t* scratch,cublasHandle_t blas,cudaStream_t stream){
        const auto* data=packed.p+2*slice.projection_bytes;
        if(tokens<=8&&(slice.down_type==10||slice.down_type==11))
            d::gpu_check(d::format_down_verify_gpu(slice.down_type,data,gate,up,routing,temporary,out,nullptr,
                5120,2304,tokens,q,size_t(tokens)*5120*4,stream));
        else{
            d::gpu_check(d::swiglu_gpu(gate,up,routing,activation,2304,tokens,10,stream));
            d::gpu_check(d::round_bf16_gpu(activation,size_t(tokens)*2304,stream));
            projection(slice.down_type,data,activation,out,5120,2304,tokens,scratch,blas,stream);
        }
    }
    std::vector<float> reference(const std::vector<float>& input,const std::vector<float>& routing,int tokens,bool prefill_reference=false)const{
        std::vector<float> result,row(5120),x(5120),h(2304);
        const bool quant_gu=!prefill_reference&&tokens<=8&&(slice.type==10||slice.type==11);
        const bool quant_down=!prefill_reference&&tokens<=8&&slice.down_type==11;
        for(int t:positions(tokens)){
            std::copy_n(input.data()+size_t(t)*5120,5120,x.data());
            if(quant_gu)q81_values(x);
            for(int r=0;r<2304;++r){
                double gv=0,uv=0;
                decode(slice.type,slice.gate+size_t(r)*slice.gu_row_bytes,row.data(),5120);
                for(int j=0;j<5120;++j)gv+=double(quant_gu?row[j]:bf(row[j]))*x[j];
                decode(slice.type,slice.up+size_t(r)*slice.gu_row_bytes,row.data(),5120);
                for(int j=0;j<5120;++j)uv+=double(quant_gu?row[j]:bf(row[j]))*x[j];
                const float g=std::min(bf(gv),10.f),u=std::clamp(bf(uv),-10.f,10.f);
                h[r]=bf(g/(1.f+std::exp(-g))*u*routing[t]);
            }
            if(quant_down)q81_values(h);
            // All output rows of each selected token, not a three-value norm.
            for(int r=0;r<5120;++r){
                decode(slice.down_type,slice.down+size_t(r)*slice.down_row_bytes,row.data(),2304);
                double sum=0;for(int j=0;j<2304;++j)sum+=double(quant_down?row[j]:bf(row[j]))*h[j];
                result.push_back(bf(sum));
            }
        }
        return result;
    }
    void cpu_check(int layer,const char* arm){
        namespace c=strata::kernels::cpu;
        c::NativeFmt fmt;std::string error;
        need(c::native_fmt(slice.type,slice.down_type,5120,2304,fmt,error),error.c_str());
        fmt.bf16_io=true;fmt.swiglu_limit=10;
        need(c::native_enable_bf16_gu(fmt,error)&&c::native_enable_bf16_down(fmt,error),error.c_str());
        for(int tokens:{1,2,5,8}){
            std::vector<float> x(size_t(tokens)*5120),h(size_t(tokens)*2304),gu(h.size()),out(x.size());
            for(size_t i=0;i<x.size();++i)x[i]=bf(.25f*std::sin(float(i)*.013f));
            for(size_t i=0;i<h.size();++i)h[i]=bf(.15f*std::cos(float(i)*.023f));
            std::vector<uint8_t> ax(size_t(tokens)*fmt.act_bytes),ah(size_t(tokens)*fmt.h_bytes);
            std::vector<const void*> xp(tokens),hp(tokens);std::vector<float*> gp(tokens),op(tokens);
            for(int t=0;t<tokens;++t){
                xp[t]=ax.data()+size_t(t)*fmt.act_bytes;hp[t]=ah.data()+size_t(t)*fmt.h_bytes;
                c::native_quant_act(fmt,x.data()+size_t(t)*5120,const_cast<void*>(xp[t]));
                c::native_quant_h(fmt,h.data()+size_t(t)*2304,const_cast<void*>(hp[t]));
                gp[t]=gu.data()+size_t(t)*2304;op[t]=out.data()+size_t(t)*5120;
            }
            std::vector<float> actual,ref,row(5120);
            for(int r:positions(2304)){
                c::native_gu_rows(fmt,host.data(),xp.data(),tokens,gp.data(),r,r+1);
                for(int t=0;t<tokens;++t){
                    double g=0,u=0;
                    decode(slice.type,slice.gate+size_t(r)*slice.gu_row_bytes,row.data(),5120);
                    for(int j=0;j<5120;++j)g+=double(bf(row[j]))*x[size_t(t)*5120+j];
                    decode(slice.type,slice.up+size_t(r)*slice.gu_row_bytes,row.data(),5120);
                    for(int j=0;j<5120;++j)u+=double(bf(row[j]))*x[size_t(t)*5120+j];
                    const float gc=std::min(bf(g),10.f),uc=std::clamp(bf(u),-10.f,10.f);
                    actual.push_back(gp[t][r]);ref.push_back(gc/(1.f+std::exp(-gc))*uc);
                }
            }
            for(int r:positions(5120)){
                c::native_down_rows(fmt,host.data(),hp.data(),tokens,op.data(),r,r+1);
                decode(slice.down_type,slice.down+size_t(r)*slice.down_row_bytes,row.data(),2304);
                for(int t=0;t<tokens;++t){
                    double sum=0;for(int j=0;j<2304;++j)sum+=double(bf(row[j]))*h[size_t(t)*2304+j];
                    actual.push_back(op[t][r]);ref.push_back(sum);
                }
            }
            const Metric metric(actual,ref);const bool pass=!metric.nonfinite&&metric.relative<=.005&&metric.maximum<=2e-4+.0078125*metric.reference_max;
            failures+=!pass;
            std::cout<<"{\"kind\":\"cpu_real_expert\",\"arm\":\""<<arm<<"\",\"layer\":"<<layer<<",\"tokens\":"<<tokens<<",";
            metric.json();std::cout<<",\"pass\":"<<(pass?"true":"false")<<"}\n";
        }
    }
    void cpu_q3_down_check(int layer,const char* arm){
        if(slice.down_type!=11)return;
        namespace c=strata::kernels::cpu;
        c::NativeFmt fmt;std::string error;
        need(c::native_fmt(slice.type,slice.down_type,5120,2304,fmt,error),error.c_str());
        fmt.bf16_io=true;
        need(c::native_enable_q3_q8_1_down(fmt,error),error.c_str());
        for(int tokens:{1,2,5,8}){
            std::vector<float> h(size_t(tokens)*2304),out(size_t(tokens)*5120);
            for(size_t i=0;i<h.size();++i)h[i]=bf(.15f*std::cos(float(i)*.023f));
            std::vector<uint8_t> encoded(size_t(tokens)*fmt.h_bytes);
            std::vector<const void*> hp(tokens);std::vector<float*> op(tokens);
            for(int t=0;t<tokens;++t){
                hp[t]=encoded.data()+size_t(t)*fmt.h_bytes;op[t]=out.data()+size_t(t)*5120;
                c::native_quant_h(fmt,h.data()+size_t(t)*2304,const_cast<void*>(hp[t]));
            }
            auto quantized=h;q81_values(quantized);
            std::vector<float> actual,ref,row(2304);
            for(int r:positions(5120)){
                c::native_down_rows(fmt,host.data(),hp.data(),tokens,op.data(),r,r+1);
                decode(slice.down_type,slice.down+size_t(r)*slice.down_row_bytes,row.data(),2304);
                for(int t=0;t<tokens;++t){
                    double sum=0;for(int j=0;j<2304;++j)sum+=double(row[j])*quantized[size_t(t)*2304+j];
                    actual.push_back(op[t][r]);ref.push_back(sum);
                }
            }
            const Metric metric(actual,ref);
            const bool pass=!metric.nonfinite&&metric.relative<=.005&&metric.maximum<=2e-4+.0078125*metric.reference_max;
            failures+=!pass;
            std::cout<<"{\"kind\":\"cpu_real_q3_down\",\"arm\":\""<<arm<<"\",\"layer\":"<<layer
                <<",\"tokens\":"<<tokens<<",\"sampled_outputs\":"<<ref.size()<<",";
            metric.json();std::cout<<",\"pass\":"<<(pass?"true":"false")<<"}\n";
        }
    }
};
void cpu_q3_down_paths(const d::Model& old,const d::Model& now){
    std::set<std::array<int,4>> covered;
    for(int layer=0;layer<40;++layer){
        const auto a=old.expert(layer,0),b=now.expert(layer,0);
        if(!covered.insert({int(a.type),int(a.down_type),int(b.type),int(b.down_type)}).second)continue;
        if(a.down_type==11)Expert(a).cpu_q3_down_check(layer,"old");
        if(b.down_type==11)Expert(b).cpu_q3_down_check(layer,"new");
    }
}
void expert_branches(const d::Model& old,const d::Model& now,cublasHandle_t blas,cudaStream_t stream,bool timing){
    std::set<std::array<int,4>> covered;
    for(int layer=0;layer<40;++layer){
        const auto a=old.expert(layer,0),b=now.expert(layer,0);
        if(!covered.insert({int(a.type),int(a.down_type),int(b.type),int(b.down_type)}).second)continue;
        Expert ea(a),eb(b);ea.cpu_check(layer,"old");eb.cpu_check(layer,"new");
        for(int tokens:{1,2,5,8,9,128,512,8192}){
            std::vector<float> input(size_t(tokens)*5120),routing(tokens);
            for(size_t i=0;i<input.size();++i)input[i]=bf(.25f*std::sin(float(i%5120)*.013f+float(i/5120)*.031f));
            for(int t=0;t<tokens;++t)routing[t]=.15f+.01f*(t%7);
            Device<float> x(input.size()),route(tokens),gate(size_t(tokens)*2304),up(gate.size),act(gate.size),
                out(size_t(tokens)*5120),temporary(out.size);
            Device<uint8_t> q(size_t(tokens)*5120*4);Device<uint16_t> scratch(input.size());
            x.set(input.data(),input.size());route.set(routing.data(),routing.size());
            auto gu=[&](Expert& e){e.gu(x.p,gate.p,up.p,tokens,q.p,scratch.p,blas,stream);};
            auto down=[&](Expert& e){e.down(gate.p,up.p,route.p,act.p,out.p,temporary.p,tokens,q.p,scratch.p,blas,stream);};
            auto chain=[&](Expert& e){gu(e);down(e);};
            for(int arm=0;arm<2;++arm){
                auto& e=arm?eb:ea;chain(e);d::gpu_check(cudaStreamSynchronize(stream));
                std::vector<float> actual(size_t(positions(tokens).size())*5120);size_t offset=0;
                for(int t:positions(tokens)){
                    d::gpu_check(cudaMemcpy(actual.data()+offset,out.p+size_t(t)*5120,5120*4,cudaMemcpyDeviceToHost));offset+=5120;
                }
                const auto ref=e.reference(input,routing,tokens);const Metric metric(actual,ref);
                const bool pass=!metric.nonfinite&&metric.relative<=.01&&metric.maximum<=1e-3+.02*metric.reference_max;
                failures+=!pass;
                std::cout<<"{\"kind\":\"expert_branch_accuracy\",\"arm\":\""<<(arm?"new":"old")<<"\",\"layer\":"<<layer
                    <<",\"gu_type\":"<<e.slice.type<<",\"down_type\":"<<e.slice.down_type<<",\"tokens\":"<<tokens
                    <<",\"sampled_outputs\":"<<ref.size()<<",\"reference\":\"independent ggml decode, scalar Q8_1 where dispatched, FP64 dots, BF16 boundaries\",";
                metric.json();std::cout<<",\"pass\":"<<(pass?"true":"false")<<"}\n"<<std::flush;
            }
            if(timing)for(int part=0;part<3;++part){
                const int repeats=tokens<=8?30:tokens<=512?10:3;
                // For Down-only measurements, use identical deterministic Gate/Up inputs.
                if(part==1){
                    std::vector<float> values(gate.size);for(size_t i=0;i<values.size();++i)values[i]=bf(.2f*std::sin(float(i)*.017f));
                    gate.set(values.data(),values.size());up.set(values.data(),values.size());
                }
                auto operation=[&](Expert& e){if(part==0)gu(e);else if(part==1)down(e);else chain(e);};
                const double a0=time_gpu([&]{operation(ea);},stream,repeats),b0=time_gpu([&]{operation(eb);},stream,repeats),
                    b1=time_gpu([&]{operation(eb);},stream,repeats),a1=time_gpu([&]{operation(ea);},stream,repeats);
                std::cout<<"{\"kind\":\"expert_branch_timing\",\"part\":\""<<(part==0?"gate_up":part==1?"swiglu_down":"chain")
                    <<"\",\"layer\":"<<layer<<",\"tokens\":"<<tokens<<",\"old_gu\":"<<a.type<<",\"old_down\":"<<a.down_type
                    <<",\"new_gu\":"<<b.type<<",\"new_down\":"<<b.down_type<<",\"repeats\":"<<repeats
                    <<",\"A0_ms\":"<<a0<<",\"B0_ms\":"<<b0<<",\"B1_ms\":"<<b1<<",\"A1_ms\":"<<a1<<"}\n"<<std::flush;
            }
        }
    }
}
void check_metric(const char* kind,const std::string& name,const char* arm,int tokens,
                  const std::vector<float>& actual,const std::vector<float>& reference){
    const Metric m(actual,reference);
    const bool pass=!m.nonfinite&&m.maximum<=1e-4+2e-4*m.reference_max&&m.relative<=5e-5;
    failures+=!pass;
    std::cout<<"{\"kind\":\""<<kind<<"\",\"tensor\":\""<<name<<"\",\"arm\":\""<<arm
        <<"\",\"tokens\":"<<tokens<<",\"sampled_outputs\":"<<reference.size()<<",";
    m.json();std::cout<<",\"pass\":"<<(pass?"true":"false")<<"}\n";
}
void boundaries(const d::Model& model,cudaStream_t stream){
    std::set<int> covered;
    for(const auto& [name,v]:model.tensors()){
        const int type=v.info->type;
        if(type!=13&&type!=14&&type!=16&&type!=17&&type!=18&&type!=20&&type!=21&&type!=22&&type!=39)continue;
        if(!covered.insert(type).second)continue;
        const size_t count=256,bytes=ggml_row_size(static_cast<ggml_type>(type),count);
        const size_t block=ggml_type_size(static_cast<ggml_type>(type));
        const std::vector<int> scales=type==39?std::vector<int>{0,1,126,127,200,240}:
            std::vector<int>{0x0000,0x8000,0x0001,0x0400,0x3c00,0x7bff,0xbc00};
        for(int scale:scales){
            std::vector<uint8_t> packed(v.data,v.data+bytes);
            for(size_t b=0;b<bytes;b+=block){
                if(type==39)packed[b]=scale;
                else{
                    const uint16_t bits=scale;
                    std::memcpy(packed.data()+b+(type==14?block-2:0),&bits,2);
                }
            }
            std::vector<float> ref(count);decode(type,packed.data(),ref.data(),count);
            Device<uint8_t> input(bytes);input.set(packed.data(),bytes);
            Device<float> output(count+16);Device<uint16_t> output_bf(count+16);
            std::vector<float> guard(count+16,-777);output.set(guard.data(),guard.size());
            std::vector<uint16_t> bguard(count+16,0x5a5a);output_bf.set(bguard.data(),bguard.size());
            d::gpu_check(d::dequant_gpu(input.p,type,output.p,count,stream));
            d::gpu_check(d::dequant_bf16_gpu(input.p,type,output_bf.p,count,stream));
            d::gpu_check(cudaStreamSynchronize(stream));
            auto got=output.get();const auto bits=output_bf.get();bool canary=true;
            for(size_t i=count;i<count+16;++i)canary&=got[i]==-777&&bits[i]==0x5a5a;
            got.resize(count);size_t mismatches=0;
            for(size_t i=0;i<count;++i){
                const auto b=ggml_fp32_to_bf16(ref[i]);uint16_t expected;std::memcpy(&expected,&b,2);
                mismatches+=bits[i]!=expected;
            }
            const Metric metric(got,ref);
            const bool pass=!metric.nonfinite&&metric.maximum<=1e-6&&metric.relative<=1e-6&&!mismatches&&canary;
            failures+=!pass;
            std::cout<<"{\"kind\":\"dequant_boundary\",\"type\":"<<type<<",\"scale_bits\":"<<scale<<",";
            metric.json();std::cout<<",\"bf16_mismatches\":"<<mismatches<<",\"canary\":"<<(canary?"true":"false")
                <<",\"pass\":"<<(pass?"true":"false")<<"}\n";
        }
    }
}
void real_other_ops(const d::Model& old,const d::Model& now,cublasHandle_t blas,cudaStream_t stream){
    // These production paths retain FP32 weights/output. The common BF16
    // projection comparison above is deliberately not their accuracy oracle.
    std::set<std::string> covered;
    for(const auto& [name,v]:now.tensors()){
        const auto role=suffix(name);
        const bool hc=role=="hc_attn_fn.weight"||role=="hc_ffn_fn.weight";
        const bool router=role=="ffn_gate_inp.weight";
        const bool compressor=role=="attn_compressor_kv.weight"||role=="attn_compressor_gate.weight";
        if(!hc&&!router&&!compressor)continue;
        if(compressor&&now.ownership()[std::stoi(name.substr(4))].compression<2)continue;
        const auto& ov=old.tensor(name);
        const std::string key=role+":"+std::to_string(ov.info->type)+":"+std::to_string(v.info->type);
        if(!covered.insert(key).second)continue;
        const int rows=v.info->shape[1],cols=v.info->shape[0];
        Device<uint8_t> pa(ov.row_bytes*rows),pb(v.row_bytes*rows);
        pa.set(ov.data,pa.size);pb.set(v.data,pb.size);
        Device<float> wa(size_t(rows)*cols),wb(wa.size);
        d::gpu_check(d::dequant_gpu(pa.p,ov.info->type,wa.p,wa.size,stream));
        d::gpu_check(d::dequant_gpu(pb.p,v.info->type,wb.p,wb.size,stream));
        d::gpu_check(cudaStreamSynchronize(stream));
        for(int tokens:{1,2,5,8,9,128,512,8192}){
            std::vector<float> input(size_t(tokens)*cols);
            for(size_t i=0;i<input.size();++i)input[i]=bf(.25f*std::sin(float(i%cols)*.013f+float(i/cols)*.031f));
            Device<float> x(input.size()),out(size_t(tokens)*rows);x.set(input.data(),input.size());
            Device<uint16_t> scratch(input.size());
            auto run=[&](int arm){
                auto* w=arm?wb.p:wa.p;
                if(hc)d::gpu_check(d::hc_project_gpu(x.p,w,out.p,5120,tokens,now.geometry().norm_eps,stream));
                else{
                    d::Matrix m{w,0,rows,cols,uint64_t(rows)*cols};
                    d::matrix_product(m,x.p,out.p,tokens,blas,stream,scratch.p,d::Arithmetic::production,0,0,0,0,false,true);
                }
            };
            for(int arm=0;arm<2;++arm){
                run(arm);d::gpu_check(cudaStreamSynchronize(stream));
                const auto& view=arm?v:ov;std::vector<float> ref,actual,row(cols);
                for(int t:positions(tokens)){
                    double norm=0;for(int j=0;j<cols;++j)norm+=double(input[size_t(t)*cols+j])*input[size_t(t)*cols+j];
                    const double factor=hc?1/std::sqrt(norm/cols+now.geometry().norm_eps):1;
                    for(int r:positions(rows)){
                        decode(view.info->type,view.data+size_t(r)*view.row_bytes,row.data(),cols);
                        double dot=0;for(int j=0;j<cols;++j)dot+=double(row[j])*input[size_t(t)*cols+j];
                        ref.push_back(dot*factor);float value;
                        d::gpu_check(cudaMemcpy(&value,out.p+size_t(t)*rows+r,4,cudaMemcpyDeviceToHost));actual.push_back(value);
                    }
                }
                check_metric("real_f32_accuracy",name,arm?"new":"old",tokens,actual,ref);
            }
            const int repeats=tokens<=8?30:tokens<=512?10:3;
            const double a0=time_gpu([&]{run(0);},stream,repeats),b0=time_gpu([&]{run(1);},stream,repeats),
                b1=time_gpu([&]{run(1);},stream,repeats),a1=time_gpu([&]{run(0);},stream,repeats);
            std::cout<<"{\"kind\":\"real_f32_timing\",\"tensor\":\""<<name<<"\",\"tokens\":"<<tokens
                <<",\"A0_ms\":"<<a0<<",\"B0_ms\":"<<b0<<",\"B1_ms\":"<<b1<<",\"A1_ms\":"<<a1<<"}\n"<<std::flush;
        }
    }
    for(int layer:{1,14}){
        const std::string base="blk."+std::to_string(layer)+".";
        const auto& folded=now.tensor(base+(now.ple_engram()?"engram_gate.weight":"engram_q.weight"));
        const auto& nk=now.tensor(base+(now.ple_engram()?"engram_gate.weight":"engram_k.weight"));
        const auto& q=old.tensor(base+(old.ple_engram()?"engram_gate.weight":"engram_q.weight"));const auto& k=old.tensor(base+(old.ple_engram()?"engram_gate.weight":"engram_k.weight"));
        Device<float> qd(20480),kd(20480),fd(20480),nkd(20480);
        std::vector<float> qv(20480),kvv(20480),fv(20480),nkv(20480);
        decode(q.info->type,q.data,qv.data(),20480);decode(k.info->type,k.data,kvv.data(),20480);
        decode(folded.info->type,folded.data,fv.data(),20480);decode(nk.info->type,nk.data,nkv.data(),20480);
        qd.set(qv.data(),20480);kd.set(kvv.data(),20480);fd.set(fv.data(),20480);nkd.set(nkv.data(),20480);
        for(int tokens:{1,2,5,8,9,128,512,8192}){
            std::vector<float> h(size_t(tokens)*20480),kv(size_t(tokens)*25600);std::vector<uint8_t> image(tokens);
            for(size_t i=0;i<h.size();++i)h[i]=bf(.2f*std::sin(float(i)*.017f));
            for(size_t i=0;i<kv.size();++i)kv[i]=bf(.1f*std::cos(float(i)*.021f));
            for(int t=0;t<tokens;++t)image[t]=t%3==1;
            Device<float> source(h.size()),out(h.size()),dkv(kv.size());Device<uint8_t> mask(tokens);
            source.set(h.data(),h.size());dkv.set(kv.data(),kv.size());mask.set(image.data(),image.size());
            auto run=[&](int arm){
                // Include the same D2D reset in both arms so repeated in-place
                // gate calls receive identical inputs instead of accumulating.
                d::gpu_check(cudaMemcpyAsync(out.p,source.p,h.size()*4,cudaMemcpyDeviceToDevice,stream));
                d::gpu_check(d::engram_gate_gpu(out.p,dkv.p,arm?fd.p:qd.p,arm?(now.ple_engram()?nullptr:nkd.p):(old.ple_engram()?nullptr:kd.p),mask.p,tokens,5120,stream));
            };
            for(int arm=0;arm<2;++arm){
                run(arm);d::gpu_check(cudaStreamSynchronize(stream));
                const auto* qw=arm?fv.data():qv.data();
                const auto* kw=arm?(now.ple_engram()?nullptr:nkv.data()):(old.ple_engram()?nullptr:kvv.data());
                std::vector<float> ref,actual;
                for(int t:positions(tokens))for(int c=0;c<4;++c){
                    double hh=0,kk=0,dot=0;
                    for(int j=0;j<5120;++j){
                        const float x=h[size_t(t)*20480+c*5120+j],key=kv[size_t(t)*25600+c*5120+j];
                        hh+=double(x)*x;kk+=double(key)*key;
                        dot+=double(x)*qw[c*5120+j]*(kw?kw[c*5120+j]:1)*key;
                    }
                    dot/=std::sqrt((hh/5120+1e-20)*(kk/5120+1e-20)*5120);
                    const double z=std::copysign(std::sqrt(std::max(std::abs(dot),1e-6)),dot);
                    const double gate=image[t]?0:1/(1+std::exp(-z));
                    std::vector<float> values(5120);
                    d::gpu_check(cudaMemcpy(values.data(),out.p+size_t(t)*20480+c*5120,5120*4,cudaMemcpyDeviceToHost));
                    for(int j=0;j<5120;++j){
                        actual.push_back(values[j]);ref.push_back(h[size_t(t)*20480+c*5120+j]+gate*kv[size_t(t)*25600+20480+j]);
                    }
                }
                check_metric("real_engram_gate_accuracy",base+"engram_gate",arm?"new":"old",tokens,actual,ref);
            }
            const int repeats=tokens<=8?30:tokens<=512?10:3;
            const double a0=time_gpu([&]{run(0);},stream,repeats),b0=time_gpu([&]{run(1);},stream,repeats),
                b1=time_gpu([&]{run(1);},stream,repeats),a1=time_gpu([&]{run(0);},stream,repeats);
            std::cout<<"{\"kind\":\"real_engram_gate_timing\",\"tensor\":\""<<base<<"engram_gate\",\"includes\":\"identical D2D reset plus gate\",\"tokens\":"<<tokens
                <<",\"A0_ms\":"<<a0<<",\"B0_ms\":"<<b0<<",\"B1_ms\":"<<b1<<",\"A1_ms\":"<<a1<<"}\n"<<std::flush;
        }
    }
}
}
int main(int argc,char** argv){
    try{
        need(argc==4,"usage: deepseek41_iq3xxs_operator_test new-first.gguf old.gguf samples|projections|timing|experts|expert-timing|other|cpu-q3");
        std::cout<<std::setprecision(12);
        ggml_cpu_init();
        auto now=d::Model::operator_weights_only(argv[1]);d::Model old(argv[2]);
        need(now->ple_engram()||now->gsq_rco_1017(),"expected PLE or GSQ-RCO1017 schema");
        for(int layer=0;layer<40;++layer){
            const auto first=now->expert(layer,0),last=now->expert(layer,383);
            need(last.gate-first.gate==383*first.projection_bytes,"Gate expert stride");
            need(last.up-first.up==383*first.projection_bytes,"Up expert stride");
            need(last.down-first.down==383*(first.blob_bytes-2*first.projection_bytes),"Down expert stride");
        }
        need(now->token_map()==old.token_map(),"token map changed");
        const auto x=d::engram_hash_spec(*now),y=d::engram_hash_spec(old);
        need(x.multipliers==y.multipliers&&x.primes==y.primes&&x.offsets==y.offsets&&x.pad==y.pad,"Engram hash semantics changed");
        std::cout<<"{\"kind\":\"layout\",\"main_complete\":"<<(now->main_complete()?"true":"false")
            <<",\"new_tensors\":"<<now->tensors().size()<<",\"old_tensors\":"<<old.tensors().size()<<",\"expert_strides_and_engram_hash\":\"exact\"}\n";
        d::Stream stream;cublasHandle_t blas=nullptr;
        need(cublasCreate(&blas)==CUBLAS_STATUS_SUCCESS&&cublasSetStream(blas,stream.value)==CUBLAS_STATUS_SUCCESS&&
             cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH)==CUBLAS_STATUS_SUCCESS,"BLAS setup");
        struct Cleanup{cublasHandle_t handle;~Cleanup(){cublasDestroy(handle);}}cleanup{blas};
        const std::string mode=argv[3];
        if(mode=="samples"){samples(old,"old",stream.value);samples(*now,"new",stream.value);}
        else if(mode=="projections"||mode=="timing")projections(old,*now,blas,stream.value,mode=="timing");
        else if(mode=="experts"||mode=="expert-timing")expert_branches(old,*now,blas,stream.value,mode=="expert-timing");
        else if(mode=="other"){boundaries(*now,stream.value);real_other_ops(old,*now,blas,stream.value);}
        else if(mode=="cpu-q3")cpu_q3_down_paths(old,*now);
        else throw std::runtime_error("unknown mode");
        std::cout<<"{\"kind\":\"summary\",\"failures\":"<<failures<<"}\n";
        return failures?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
