// Paired same-weight expert checks; never constructs Runtime or Session.
#include "runtime_internal.hpp"
#include "strata/models/deepseek41/mixed_expert.hpp"
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

cudaError_t mixed_test_compare_bf16(const uint16_t*,const uint16_t*,uint64_t,
                                  unsigned long long*,cudaStream_t);
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
    std::vector<float> reference(const std::vector<float>& input,const std::vector<float>& routing,int tokens)const{
        std::vector<float> result,row(5120),x(5120),h(2304);
        const bool quant_gu=tokens<=8&&(slice.type==10||slice.type==11);
        const bool quant_down=tokens<=8&&slice.down_type==11;
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
};

struct Candidate:Expert {
    using Expert::Expert;
    void projection(int type,const uint8_t* data,const float* x,float* y,int rows,int cols,int tokens,
                    uint16_t* scratch,cublasHandle_t blas,cudaStream_t stream) {
        if(!d::expert_mixed_type(type)){Expert::projection(type,data,x,y,rows,cols,tokens,scratch,blas,stream);return;}
        if(tokens<=8) {
            d::gpu_check(d::expert_mixed_projection_gpu(data,type,x,y,rows,cols,tokens,nullptr,stream));
            return;
        }
        d::gpu_check(d::expert_mixed_dequant_gpu(data,type,weights.p,size_t(rows)*cols,stream));
        const d::Matrix matrix{weights.p,30,rows,cols,uint64_t(rows)*cols};
        d::matrix_product(matrix,x,y,tokens,blas,stream,scratch,d::Arithmetic::production,0,0,0,0,true,true);
    }
    void gu(const float* x,float* gate,float* up,int tokens,void* q,uint16_t* scratch,
            cublasHandle_t blas,cudaStream_t stream) {
        if(tokens<=8&&d::expert_mixed_type(slice.type)) {
            d::gpu_check(d::expert_mixed_gu_gpu(packed.p,packed.p+slice.projection_bytes,slice.type,
                x,gate,up,2304,5120,tokens,stream));return;
        }
        if(tokens<=8){Expert::gu(x,gate,up,tokens,q,scratch,blas,stream);return;}
        projection(slice.type,packed.p,x,gate,2304,5120,tokens,scratch,blas,stream);
        projection(slice.type,packed.p+slice.projection_bytes,x,up,2304,5120,tokens,scratch,blas,stream);
    }
    void down(const float* gate,const float* up,const float* routing,float* activation,float* out,float* temporary,
              int tokens,void* q,uint16_t* scratch,cublasHandle_t blas,cudaStream_t stream) {
        if(!d::expert_mixed_type(slice.down_type)) {
            Expert::down(gate,up,routing,activation,out,temporary,tokens,q,scratch,blas,stream);return;
        }
        const auto* data=packed.p+2*slice.projection_bytes;
        if(tokens<=8) {
            d::gpu_check(d::expert_mixed_down_gpu(data,slice.down_type,gate,up,routing,activation,out,
                5120,2304,tokens,nullptr,stream));return;
        }
        d::gpu_check(d::expert_mixed_swiglu_gpu(gate,up,routing,activation,2304,tokens,stream));
        projection(slice.down_type,data,activation,out,5120,2304,tokens,scratch,blas,stream);
    }
};

void emit(const char* kind,int layer,int expert,int tokens,const char* part,const std::vector<float>& actual,
          const std::vector<float>& reference,double tolerance,bool chain=false) {
    const Metric m(actual,reference);
    const bool pass=!m.nonfinite&&m.relative<=tolerance&&
        m.maximum<=(chain?1e-3+.02*m.reference_max:2e-4+.0078125*m.reference_max);
    size_t unequal=0;
    for(size_t i=0;i<actual.size();++i)unequal+=std::memcmp(&actual[i],&reference[i],4)!=0;
    failures+=!pass;
    std::cout<<"{\"kind\":\""<<kind<<"\",\"layer\":"<<layer<<",\"expert\":"<<expert
        <<",\"tokens\":"<<tokens<<",\"part\":\""<<part<<"\",\"outputs\":"<<actual.size()<<',';
    m.json();
    std::cout<<",\"bit_unequal\":"<<unequal<<",\"pass\":"<<(pass?"true":"false")<<"}\n";
}

// Sparse independent rows complement complete GPU/GPU output comparison.
// Decode each selected row only once; all selected token dots accumulate FP64.
void independent_projection(const Expert& e,int layer,int expert,int tokens,int role,
                            const std::vector<float>& input,const std::vector<float>& output) {
    const int cols=role==2?2304:5120,rows=role==2?5120:2304;
    const int type=role==2?e.slice.down_type:e.slice.type;
    const uint8_t* packed=role==2?e.slice.down:role==1?e.slice.up:e.slice.gate;
    const size_t row_bytes=role==2?e.slice.down_row_bytes:e.slice.gu_row_bytes;
    const bool a8=tokens<=8&&(role==2?type==11:type==10||type==11);
    auto x=input;
    if(a8)q81_values(x);
    std::vector<float> selected,ref,row(cols);
    for(int sample=0;sample<16;++sample) {
        const int r=sample*(rows-1)/15;
        decode(type,packed+size_t(r)*row_bytes,row.data(),cols);
        for(int t:positions(tokens)) {
            double dot=0;
            for(int k=0;k<cols;++k)dot+=double(a8?row[k]:bf(row[k]))*x[size_t(t)*cols+k];
            ref.push_back(bf(float(dot)));selected.push_back(output[size_t(t)*rows+r]);
        }
    }
    emit("independent_projection",layer,expert,tokens,role==0?"gate":role==1?"up":"down",
         selected,ref,.005);
}

void check_dequant(const Expert& e,int layer,int expert,cudaStream_t stream) {
    constexpr size_t elements=5120ull*2304;
    Device<uint16_t> a(elements+16),b(elements+16);
    Device<unsigned long long> different(1);
    for(int role=0;role<3;++role) {
        const int type=role==2?e.slice.down_type:e.slice.type;
        if(!d::expert_mixed_type(type))continue;
        const auto* weights=e.packed.p+(role==0?0:role==1?e.slice.projection_bytes:2*e.slice.projection_bytes);
        d::gpu_check(cudaMemsetAsync(a.p+elements,0xa5,32,stream));
        d::gpu_check(cudaMemsetAsync(b.p+elements,0xa5,32,stream));
        d::gpu_check(d::dequant_bf16_gpu(weights,type,a.p,elements,stream));
        d::gpu_check(d::expert_mixed_dequant_gpu(weights,type,b.p,elements,stream));
        d::gpu_check(mixed_test_compare_bf16(a.p,b.p,elements,different.p,stream));
        d::gpu_check(cudaStreamSynchronize(stream));
        const auto mismatch=different.get()[0];
        uint16_t guard[16];
        d::gpu_check(cudaMemcpy(guard,b.p+elements,32,cudaMemcpyDeviceToHost));
        bool canary=std::all_of(std::begin(guard),std::end(guard),[](uint16_t v){return v==0xa5a5;});
        size_t cpu_mismatch=0;
        const int cols=role==2?2304:5120,rows=role==2?5120:2304;
        const uint8_t* source=role==2?e.slice.down:role==1?e.slice.up:e.slice.gate;
        const size_t row_bytes=role==2?e.slice.down_row_bytes:e.slice.gu_row_bytes;
        for(int row:positions(rows)) {
            std::vector<float> reference(cols);std::vector<uint16_t> actual(cols);
            decode(type,source+size_t(row)*row_bytes,reference.data(),cols);
            d::gpu_check(cudaMemcpy(actual.data(),b.p+size_t(row)*cols,cols*2,cudaMemcpyDeviceToHost));
            for(int j=0;j<cols;++j) {
                const auto expected=ggml_fp32_to_bf16(reference[j]);uint16_t bits;
                std::memcpy(&bits,&expected,2);cpu_mismatch+=actual[j]!=bits;
            }
        }
        const bool pass=!mismatch&&!cpu_mismatch&&canary;failures+=!pass;
        std::cout<<"{\"kind\":\"dequant_pair\",\"layer\":"<<layer<<",\"expert\":"<<expert
            <<",\"role\":"<<role<<",\"type\":"<<type<<",\"elements\":"<<elements
            <<",\"baseline_bit_mismatches\":"<<mismatch<<",\"independent_bit_mismatches\":"<<cpu_mismatch
            <<",\"canary\":"<<(canary?"true":"false")<<",\"pass\":"<<(pass?"true":"false")<<"}\n";
    }
}

struct Buffers {
    Device<float> x,route,ag,au,bg,bu,act,aout,bout,temporary;
    Device<uint8_t> q;
    Device<uint16_t> scratch;
    explicit Buffers(int n):x(size_t(n)*5120),route(n),ag(size_t(n)*2304),au(ag.size),
        bg(ag.size),bu(ag.size),act(ag.size),aout(size_t(n)*5120),bout(aout.size),
        temporary(aout.size),q(size_t(n)*5120*4),scratch(x.size) {}
};

void exact(const char* part,int layer,int id,int n,const std::vector<float>& got,const std::vector<float>& ref) {
    const Metric metric(got,ref);
    size_t different=0;for(size_t i=0;i<got.size();++i)different+=std::memcmp(&got[i],&ref[i],4)!=0;
    const bool pass=!different&&!metric.nonfinite;failures+=!pass;
    std::cout<<"{\"kind\":\"integration_exact\",\"part\":\""<<part<<"\",\"layer\":"<<layer
        <<",\"expert\":"<<id<<",\"tokens\":"<<n<<",\"outputs\":"<<got.size()
        <<",\"bit_unequal\":"<<different<<",\"nonfinite\":"<<metric.nonfinite<<",\"pass\":"<<(pass?"true":"false")<<"}\n";
}

// Exercise the newly connected runtime branches against their previously
// validated, separate operations. Independent scalar references remain above.
void integration(const d::Model& model,cublasHandle_t blas,cudaStream_t stream) {
    Buffers w(16);
    Device<float> act(16*2304),parts(16*6*5120),gu(16*2304),up(16*2304);
    Device<int> rows(16),slots(16);
    for(int layer:{0,2,5,10,25})for(int id:{0,383}) {
        Candidate e(model.expert(layer,id));
        for(int n=1;n<=9;++n) {
            const bool selected=d::expert_mixed_gu_activation_selected(e.slice.type,n);
            const bool expected=(e.slice.type==22&&n==1)||(e.slice.type==17&&(n==1||n==3||n==8))||
                                (e.slice.type==16&&n==3);
            need(selected==expected,"adopted activation dispatch contract");
            if(!selected&&e.slice.type!=11)continue;
            if(e.slice.type==11&&n>8)continue;
            for(int pattern=0;pattern<3;++pattern) {
                std::vector<float> x(size_t(n)*5120),route(n);
                for(size_t i=0;i<x.size();++i)x[i]=bf(pattern==1?0.f:pattern==2?(i%2?2.f:-16.f):
                    .25f*std::sin(float(i)*.013f+float(id)*.007f));
                for(int i=0;i<n;++i)route[i]=i%3==0?0.f:.2f+float(i)*.01f;
                w.x.set(x.data(),x.size());w.route.set(route.data(),n);
                e.gu(w.x.p,w.ag.p,w.au.p,n,w.q.p,w.scratch.p,blas,stream);
                if(selected) {
                    d::gpu_check(d::expert_mixed_swiglu_gpu(w.ag.p,w.au.p,w.route.p,w.act.p,2304,n,stream));
                    d::gpu_check(d::expert_mixed_gu_activation_gpu(e.packed.p,e.packed.p+e.slice.projection_bytes,
                        e.slice.type,w.x.p,w.route.p,act.p,2304,5120,n,stream));
                    d::gpu_check(cudaStreamSynchronize(stream));
                    exact("gu_activation",layer,id,n,act.get(size_t(n)*2304),w.act.get(size_t(n)*2304));
                    e.down(w.ag.p,w.au.p,w.route.p,w.act.p,w.aout.p,w.temporary.p,n,w.q.p,w.scratch.p,blas,stream);
                    if(e.slice.down_type==11) {
                        strata::kernels::native_quantize_q8_1(act.p,w.q.p,2304,n,stream);
                        strata::kernels::native_q3_k_mmvq(e.packed.p+2*e.slice.projection_bytes,w.q.p,w.bout.p,2304,5120,n,stream);
                        d::gpu_check(d::round_bf16_gpu(w.bout.p,size_t(n)*5120,stream));
                    }else d::gpu_check(d::expert_mixed_projection_gpu(e.packed.p+2*e.slice.projection_bytes,
                        e.slice.down_type,act.p,w.bout.p,5120,2304,n,nullptr,stream));
                    d::gpu_check(cudaStreamSynchronize(stream));
                    exact("gu_activation_down",layer,id,n,w.bout.get(size_t(n)*5120),w.aout.get(size_t(n)*5120));
                }else {
                    std::vector<int> source(3+n),target(n);
                    for(int i=0;i<n;++i){source[3+i]=i;target[i]=(n-1-i)*6+2;}
                    rows.set(source.data(),source.size());slots.set(target.data(),n);
                    // Nonzero row offset validates the concatenated resident-group workspace.
                    const d::VerifyExpertGroup group{e.packed.p,3,n};
                    d::gpu_check(d::expert_gu_grouped_verify_gpu(&group,1,e.slice.type,w.x.p,rows.p,
                        gu.p,up.p,n,2,w.q.p,stream));
                    d::gpu_check(cudaStreamSynchronize(stream));
                    const auto all_g=gu.get(),all_u=up.get();
                    exact("grouped_q3_gate",layer,id,n,{all_g.begin()+3*2304,all_g.begin()+(3+n)*2304},w.ag.get(size_t(n)*2304));
                    exact("grouped_q3_up",layer,id,n,{all_u.begin()+3*2304,all_u.begin()+(3+n)*2304},w.au.get(size_t(n)*2304));
                    d::gpu_check(cudaMemsetAsync(parts.p,0,parts.size*4,stream));
                    d::gpu_check(d::expert_mixed_down_gpu(e.packed.p+2*e.slice.projection_bytes,e.slice.down_type,
                        gu.p+3*2304,up.p+3*2304,w.route.p,act.p,parts.p,5120,2304,n,slots.p,stream));
                    e.down(w.ag.p,w.au.p,w.route.p,w.act.p,w.aout.p,w.temporary.p,n,w.q.p,w.scratch.p,blas,stream);
                    d::gpu_check(cudaStreamSynchronize(stream));
                    const auto scatter=parts.get(size_t(n)*6*5120);std::vector<float> gathered(size_t(n)*5120);
                    for(int i=0;i<n;++i)std::copy_n(scatter.data()+size_t(target[i])*5120,5120,gathered.data()+size_t(i)*5120);
                    exact("grouped_q3_mxfp4_scatter",layer,id,n,gathered,w.aout.get(size_t(n)*5120));
                }
                need(!failures,"integration branch numerical gate");
            }
        }
    }
}

std::vector<float> activation(const std::vector<float>& g,const std::vector<float>& u,
                              const std::vector<float>& route) {
    std::vector<float> h(g.size());
    for(size_t i=0;i<h.size();++i) {
        const float gate=std::min(g[i],10.f),up=std::clamp(u[i],-10.f,10.f);
        const float sigmoid=gate>=0?1.f/(1.f+std::exp(-gate)):std::exp(gate)/(1.f+std::exp(gate));
        h[i]=bf(gate*sigmoid*up*route[i/2304]);
    }
    return h;
}

void check_expert(Expert& a,Candidate& b,int layer,int id,int tokens,Buffers& w,
                  cublasHandle_t blas,cudaStream_t stream,bool full_reference,int pattern=0) {
    std::vector<float> input(size_t(tokens)*5120),route(tokens);
    for(size_t i=0;i<input.size();++i) {
        float value=.25f*std::sin(float(i%5120)*.013f+float(i/5120)*.031f+float(layer*384+id)*.0007f);
        if(pattern==1)value=0;
        if(pattern==2)value=(i%2?1.f:-1.f)*(i%7?2.f:16.f);
        input[i]=bf(value);
    }
    for(int t=0;t<tokens;++t)route[t]=pattern==1?0:.15f+.01f*(t%7);
    w.x.set(input.data(),input.size());w.route.set(route.data(),route.size());
    a.gu(w.x.p,w.ag.p,w.au.p,tokens,w.q.p,w.scratch.p,blas,stream);
    b.gu(w.x.p,w.bg.p,w.bu.p,tokens,w.q.p,w.scratch.p,blas,stream);
    d::gpu_check(cudaStreamSynchronize(stream));
    auto ag=w.ag.get(size_t(tokens)*2304),au=w.au.get(ag.size()),bg=w.bg.get(ag.size()),bu=w.bu.get(ag.size());
    emit("paired",layer,id,tokens,"gate",bg,ag,.001);
    emit("paired",layer,id,tokens,"up",bu,au,.001);
    if(Metric(bu,au).maximum>1.f) {
        std::vector<float> decoded(5120);
        for(size_t index=0;index<bu.size();++index)if(std::abs(bu[index]-au[index])>1.f) {
            const int row=index%2304,token=index/2304;
            decode(b.slice.type,b.slice.up+size_t(row)*b.slice.gu_row_bytes,decoded.data(),5120);
            double dot=0;for(int k=0;k<5120;++k)dot+=double(bf(decoded[k]))*input[size_t(token)*5120+k];
            std::cout<<"{\"kind\":\"failure_oracle\",\"layer\":"<<layer<<",\"expert\":"<<id
                <<",\"tokens\":"<<tokens<<",\"pattern\":"<<pattern<<",\"row\":"<<row<<",\"token\":"<<token
                <<",\"baseline\":"<<au[index]<<",\"candidate\":"<<bu[index]<<",\"cpu_fp64_bf16\":"<<bf(dot)<<"}\n";
        }
    }
    independent_projection(b,layer,id,tokens,0,input,bg);
    independent_projection(b,layer,id,tokens,1,input,bu);
    // The chain consumes each arm's own GU output, preserving error propagation.
    a.down(w.ag.p,w.au.p,w.route.p,w.act.p,w.aout.p,w.temporary.p,tokens,w.q.p,w.scratch.p,blas,stream);
    b.down(w.bg.p,w.bu.p,w.route.p,w.act.p,w.bout.p,w.temporary.p,tokens,w.q.p,w.scratch.p,blas,stream);
    d::gpu_check(cudaStreamSynchronize(stream));
    const auto ao=w.aout.get(size_t(tokens)*5120),bo=w.bout.get(ao.size());
    emit("paired",layer,id,tokens,"chain",bo,ao,.002,true);
    if(full_reference) {
        const auto ref=a.reference(input,route,tokens);
        std::vector<float> selected;
        for(int t:positions(tokens))selected.insert(selected.end(),bo.begin()+size_t(t)*5120,bo.begin()+size_t(t+1)*5120);
        emit("independent_chain",layer,id,tokens,"chain",selected,ref,.01,true);
    }
    // Down alone receives identical deterministic GU values in both arms.
    for(size_t i=0;i<ag.size();++i) {
        ag[i]=bf(pattern==2?(i%2?12.f:-12.f):.2f*std::sin(float(i)*.017f));
        au[i]=bf(pattern==2?(i%3?16.f:-16.f):.3f*std::cos(float(i)*.021f));
        if(pattern==1)ag[i]=au[i]=0;
    }
    w.ag.set(ag.data(),ag.size());w.au.set(au.data(),au.size());
    a.down(w.ag.p,w.au.p,w.route.p,w.act.p,w.aout.p,w.temporary.p,tokens,w.q.p,w.scratch.p,blas,stream);
    b.down(w.ag.p,w.au.p,w.route.p,w.act.p,w.bout.p,w.temporary.p,tokens,w.q.p,w.scratch.p,blas,stream);
    d::gpu_check(cudaStreamSynchronize(stream));
    auto ad=w.aout.get(ao.size()),bd=w.bout.get(ao.size());
    emit("paired",layer,id,tokens,"down",bd,ad,.001);
    independent_projection(b,layer,id,tokens,2,activation(ag,au,route),bd);
    if(tokens<=8&&d::expert_mixed_type(b.slice.down_type)) {
        Device<int> slots(tokens);Device<float> scattered(size_t(tokens*2+1)*5120);
        std::vector<int> index(tokens);std::vector<float> expected(scattered.size,-777.f);
        for(int t=0;t<tokens;++t) {
            index[t]=2*(tokens-1-t)+1;
            std::copy_n(bd.data()+size_t(t)*5120,5120,expected.data()+size_t(index[t])*5120);
        }
        slots.set(index.data(),index.size());
        std::vector<float> clear(scattered.size,-777.f);scattered.set(clear.data(),clear.size());
        d::gpu_check(d::expert_mixed_down_gpu(b.packed.p+2*b.slice.projection_bytes,b.slice.down_type,
            w.ag.p,w.au.p,w.route.p,w.act.p,scattered.p,5120,2304,tokens,slots.p,stream));
        d::gpu_check(cudaStreamSynchronize(stream));
        const auto actual=scattered.get();
        const bool exact=std::memcmp(actual.data(),expected.data(),expected.size()*4)==0;
        failures+=!exact;
        std::cout<<"{\"kind\":\"scatter\",\"layer\":"<<layer<<",\"expert\":"<<id
            <<",\"tokens\":"<<tokens<<",\"pass\":"<<(exact?"true":"false")<<"}\n";
    }
    std::cout<<std::flush;
}

void synthetic(cudaStream_t stream) {
    for(int type:{16,17,18,21,22,39}) {
        const size_t bytes=ggml_row_size(static_cast<ggml_type>(type),256);
        const size_t block=ggml_type_size(static_cast<ggml_type>(type));
        for(int scale: type==39?std::vector<int>{0,1,126,127,200,240}:
                                 std::vector<int>{0,0x8000,1,0x0400,0x3c00,0x7bff,0xbc00}) {
            std::vector<uint8_t> source(bytes);
            for(size_t i=0;i<bytes;++i)source[i]=uint8_t(i*73+29);
            for(size_t i=0;i<bytes;i+=block) {
                if(type==39)source[i]=scale;
                else{const uint16_t bits=scale;std::memcpy(source.data()+i,&bits,2);}
            }
            Device<uint8_t> packed(bytes);Device<uint16_t> out(272);
            packed.set(source.data(),source.size());
            std::vector<uint16_t> guard(272,0x5a5a);out.set(guard.data(),guard.size());
            d::gpu_check(d::expert_mixed_dequant_gpu(packed.p,type,out.p,256,stream));
            d::gpu_check(cudaStreamSynchronize(stream));
            const auto actual=out.get();std::vector<float> decoded(256);
            decode(type,source.data(),decoded.data(),256);
            size_t mismatch=0;
            for(int i=0;i<256;++i) {
                const auto value=ggml_fp32_to_bf16(decoded[i]);uint16_t bits;std::memcpy(&bits,&value,2);
                mismatch+=actual[i]!=bits;
            }
            const bool canary=std::all_of(actual.begin()+256,actual.end(),[](uint16_t x){return x==0x5a5a;});
            const bool pass=!mismatch&&canary;failures+=!pass;
            std::cout<<"{\"kind\":\"synthetic_dequant\",\"type\":"<<type<<",\"scale\":"<<scale
                <<",\"bit_mismatches\":"<<mismatch<<",\"canary\":"<<(canary?"true":"false")
                <<",\"pass\":"<<(pass?"true":"false")<<"}\n";
        }
    }
}
} // namespace

int main(int argc,char** argv) {
    try {
        need(argc>=3&&argc<=5,"usage: deepseek41_mixed_expert_test model.gguf synthetic|integration|representatives|all [first_layer end_layer]");
        std::cout<<std::setprecision(12);
        ggml_cpu_init();
        d::Stream stream;
        cublasHandle_t blas=nullptr;
        need(cublasCreate(&blas)==CUBLAS_STATUS_SUCCESS&&cublasSetStream(blas,stream.value)==CUBLAS_STATUS_SUCCESS&&
             cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH)==CUBLAS_STATUS_SUCCESS,"BLAS setup");
        struct Cleanup{cublasHandle_t handle;~Cleanup(){cublasDestroy(handle);}}cleanup{blas};
        // The cumulative runtime already adopted mixed kernels. This harness
        // still invokes the independent expansion path explicitly as its control.
        d::expert_mixed_fast_set_enabled(false);
        const std::string mode=argv[2];
        if(mode=="synthetic")synthetic(stream.value);
        else if(mode=="integration")integration(*d::Model::operator_weights_only(argv[1]),blas,stream.value);
        else {
            need(mode=="representatives"||mode=="all","unknown mode");
            const auto model=d::Model::operator_weights_only(argv[1]);
            const int first=argc>3?std::stoi(argv[3]):0,end=argc>4?std::stoi(argv[4]):40;
            need(first>=0&&end<=40&&first<end,"layer range");
            std::set<std::pair<int,int>> covered;
            Buffers workspace(mode=="all"?5:512);
            for(int layer=first;layer<end;++layer) {
                const auto shape=model->expert(layer,0);
                if(mode=="representatives"&&!covered.insert({shape.type,shape.down_type}).second)continue;
                const std::vector<int> ids=mode=="all"?[] {std::vector<int> v(384);for(int i=0;i<384;++i)v[i]=i;return v;}():
                    std::vector<int>{0,192,383};
                for(int id:ids) {
                    Expert baseline(model->expert(layer,id));Candidate candidate(model->expert(layer,id));
                    check_dequant(candidate,layer,id,stream.value);
                    const std::vector<int> counts=mode=="all"?std::vector<int>{1,5}:
                        std::vector<int>{1,2,3,4,5,6,7,8,9,32,128,256,512};
                    for(int n:counts) {
                        check_expert(baseline,candidate,layer,id,n,workspace,blas,stream.value,
                                     mode=="representatives"&&(n==1||n==5||n==9));
                        if(failures)throw std::runtime_error("numerical gate failed; stop before the next case");
                    }
                    if(mode=="representatives")for(int pattern:{1,2})
                        for(int n:{1,5,8,9}) {
                            check_expert(baseline,candidate,layer,id,n,workspace,blas,stream.value,false,pattern);
                            if(failures)throw std::runtime_error("boundary numerical gate failed");
                        }
                    std::cout<<"{\"kind\":\"expert_complete\",\"layer\":"<<layer<<",\"expert\":"<<id
                        <<",\"gu_type\":"<<shape.type<<",\"down_type\":"<<shape.down_type<<"}\n"<<std::flush;
                }
            }
        }
        std::cout<<"{\"kind\":\"summary\",\"failures\":"<<failures<<"}\n";
        return failures?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
