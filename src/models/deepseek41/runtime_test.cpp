#include "runtime_internal.hpp"
#include "strata/models/deepseek41/prefill_chunk.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

namespace d = strata::models::deepseek41;
namespace c = strata::kernels::cpu;
namespace {
template<class T> struct Device {
    T* p=nullptr;size_t count;
    explicit Device(size_t n):count(n){d::gpu_check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    explicit Device(const std::vector<T>& x):Device(x.size()){d::gpu_check(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){cudaFree(p);}
    std::vector<T> read()const{std::vector<T> x(count);d::gpu_check(cudaMemcpy(x.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
void exact(const std::vector<float>& a,const std::vector<float>& b,const char* name){
    if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*4))throw std::runtime_error(name);
}
float bf(float x){return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));}
void workspace_auto_budget(){
    for(int context:{8192,32768,131072,524288}){
        d::RuntimeOptions options;options.context=context;
        const uint64_t budget=d::Workspace::bytes_needed(32768,options);
        const int chosen=d::select_prefill_chunk(32768,context,budget,[&](int n){return d::Workspace::bytes_needed(n,options);});
        if(chosen!=std::min(context,32768))throw std::runtime_error("32K workspace plan context ceiling");
    }
    for(int mode=0;mode<5;++mode){
        d::RuntimeOptions options;options.context=1048576;
        options.compact_logits=(mode==1||mode==4);options.compact_workspace=(mode==1||mode==4);
        options.dense_q8=(mode==1||mode==4);options.engram_pipeline=(mode==1||mode==4);options.expert_pipeline=(mode==1||mode==4);
        options.dspark=mode>=3;const bool vision=mode==2;
        const uint64_t budget=d::Workspace::bytes_needed(256,options,vision);
        const int chosen=d::select_prefill_chunk(32768,options.context,budget,[&](int n){return d::Workspace::bytes_needed(n,options,vision);});
        if(chosen!=256)throw std::runtime_error("workspace auto exact-budget selection");
        d::Workspace work(budget,chosen,options.context,options.compact_logits,options.compact_workspace,options.dspark);
        d::WorkspaceLayout layout(work.arena.get());work.extras(layout,options);
        if(vision){d::VisionWork extra(*work.arena);}
        if(options.dspark){d::DraftWork extra(*work.arena);}
        if(work.arena->used()!=budget)throw std::runtime_error("workspace plan differs from actual layout");
    }
}
void helpers(){
    const int t=3,width=5120;
    std::vector<float> table(4*width),pixels(t*width),h(t*4*width),pre(t*4),mean(t*width);
    std::vector<int> ids{2,3,0};std::vector<uint8_t> mask{0,1,0};
    for(size_t i=0;i<table.size();++i)table[i]=(int(i%101)-50)*.01f;
    for(size_t i=0;i<pixels.size();++i)pixels[i]=(int(i%13)-6)*.03f;
    Device<float> dt(table),dp(pixels),dh(h.size()),dr(pre.size()),dm(mean.size());Device<int> di(ids);Device<uint8_t> dx(mask);
    d::gpu_check(d::init_residual_gpu(dt.p,0,di.p,dp.p,dx.p,dh.p,dr.p,t,nullptr));
    for(int token=0;token<t;++token)for(int copy=0;copy<4;++copy)for(int dim=0;dim<width;++dim)
        h[(token*4+copy)*width+dim]=mask[token]?pixels[token*width+dim]:table[ids[token]*width+dim];
    for(int token=0;token<t;++token){pre[token*4]=1;std::copy(h.begin()+token*4*width,h.begin()+(token*4+1)*width,mean.begin()+token*width);}
    exact(dh.read(),h,"initial residual/image copies");exact(dr.read(),pre,"initial carried mHC coefficient");
    d::gpu_check(d::mean_residual_gpu(dh.p,dm.p,t,nullptr));exact(dm.read(),mean,"DSpark mean residual capture");
    // C2 odd partial + five new tokens: compare to a softmax pool over six full inputs.
    std::vector<float> kv(6*512),scores(6*512),pooled(3*512);
    for(size_t i=0;i<kv.size();++i){kv[i]=std::sin(i*.07f);scores[i]=std::cos(i*.031f)*12;}
    Device<float> dk(kv),ds(scores),dout(pooled.size());
    d::gpu_check(d::pool_groups_gpu(dk.p+512,ds.p+512,dk.p,ds.p,dout.p,3,true,nullptr));
    for(int row=0;row<3;++row)for(int dim=0;dim<512;++dim){
        const int a=row*1024+dim,b=a+512;const double mx=std::max(scores[a],scores[b]);
        const double wa=std::exp(scores[a]-mx),wb=std::exp(scores[b]-mx);pooled[row*512+dim]=(kv[a]*wa+kv[b]*wb)/(wa+wb);
    }
    const auto got=dout.read();
    for(size_t i=0;i<got.size();++i)if(std::abs(got[i]-pooled[i])>1e-5+1e-4*std::abs(pooled[i]))throw std::runtime_error("C2 partial chunk pool");
    std::cout<<"runtime initialization, capture and C2 chunk helpers passed\n";
}
void strided(cublasHandle_t blas){
    for(int tokens:{1,3,9}){
        constexpr int rows=3,cols=4,ldx=8,ldy=6;
        std::vector<float> weight(rows*cols),x(tokens*ldx),out(tokens*ldy,-777),expected=out;
        for(int i=0;i<rows*cols;++i)weight[i]=(i-5)*.125f;
        for(int i=0;i<tokens*ldx;++i)x[i]=(i%9-4)*.25f;
        for(int t=0;t<tokens;++t)for(int row=0;row<rows;++row){
            float value=0;for(int col=0;col<cols;++col)value+=weight[row*cols+col]*x[t*ldx+col];expected[t*ldy+row]=value;
        }
        Device<float> dw(weight),dx(x),dy(out);Device<uint16_t> scratch(tokens*ldx);
        d::Matrix m{dw.p,0,rows,cols,rows*cols};
        d::matrix_product(m,dx.p,dy.p,tokens,blas,nullptr,scratch.p,d::Arithmetic::f32,0,ldx,ldy,0,false);
        exact(dy.read(),expected,"strided grouped output or padding");
    }
    std::cout<<"short GEMV and prefill GEMM strided matrix views passed\n";
}
void strided_round(){
    constexpr int cols=1024,stride=8192,offset=3*1024;
    for(int tokens:{1,3,8,9,128,4096}){
        std::vector<float> input(size_t(tokens)*stride+256);
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(i*.013f)*103.17f;
        // Include BF16 rounding ties and signed zeros; both paths must preserve every unowned column.
        const uint32_t bits[]={0x3f808000,0x3f818000,0xbf808000,0x00000000,0x80000000};
        for(size_t i=0;i<sizeof(bits)/sizeof(bits[0]);++i)std::memcpy(input.data()+offset+i,bits+i,4);
        Device<float> expected(input),got(input);
        for(int t=0;t<tokens;++t)d::gpu_check(d::round_bf16_gpu(expected.p+offset+int64_t(t)*stride,cols,nullptr));
        d::gpu_check(d::round_bf16_strided_gpu(got.p+offset,tokens,cols,stride,nullptr));
        exact(got.read(),expected.read(),"strided BF16 round differs or modifies padding");
    }
    if(d::round_bf16_strided_gpu(nullptr,0,cols,stride,nullptr)!=cudaErrorInvalidValue||
       d::round_bf16_strided_gpu(nullptr,1,0,stride,nullptr)!=cudaErrorInvalidValue||
       d::round_bf16_strided_gpu(nullptr,1,cols,cols-1,nullptr)!=cudaErrorInvalidValue)
        throw std::runtime_error("invalid strided BF16 shape accepted");
    std::cout<<"strided BF16 round exact for1/3/8/9/128/4096 tokens, ties and unowned padding passed\n";
}
void workspace_attention_bounds(){
    constexpr int capacity=128,heads=64,width=512;
    d::Workspace w(512ull<<20,capacity,512);
    auto span=[](const void* a,const void* b){
        const auto first=reinterpret_cast<uintptr_t>(a),last=reinterpret_cast<uintptr_t>(b);
        if(last<first)throw std::runtime_error("workspace allocation order");
        return uint64_t(last-first);
    };
    const uint64_t guard_bytes=uint64_t(capacity)*heads*sizeof(float);
    const uint64_t lse_bytes=span(w.lse,w.engram_packed),partial_bytes=span(w.partial_out,w.lse);
    if(lse_bytes<guard_bytes||partial_bytes<uint64_t(8)*heads*width*2||
       span(w.out_bf,w.partial_out)<uint64_t(capacity)*heads*width*2||
       span(w.engram_packed,w.temp_swa)<guard_bytes)throw std::runtime_error("workspace attention allocation span");
    Device<uint8_t> cache(32*528);Device<int64_t> slots(capacity*128);Device<float> sinks(heads);
    d::gpu_check(cudaMemset(cache.p,0,32*528));d::gpu_check(cudaMemset(slots.p,0,uint64_t(capacity)*128*8));
    d::gpu_check(cudaMemset(sinks.p,0,heads*4));d::gpu_check(cudaMemset(w.q_bf,0,uint64_t(capacity)*heads*width*2));
    for(int count:{8,9,32,33,128}){
        const int nt=count<=8?1:count,splits=count<=8?8:1,calls=count<=8?count:1;
        d::gpu_check(cudaMemset(w.engram_packed,0xa5,guard_bytes));
        d::gpu_check(cudaMemset(w.partial_out,0x5a,partial_bytes));
        d::PackedAttention p;p.partial=w.partial_out;p.lse=w.lse;p.sinks=sinks.p;
        p.swa={cache.p,32*528,32*528,528,32};p.tokens=nt;p.splits=splits;p.swa_topk=128;
        for(int call=0;call<calls;++call){
            p.q=w.q_bf+int64_t(call)*heads*width;p.output=w.out_bf+int64_t(call)*heads*width;
            p.swa_slots=slots.p+int64_t(call)*128;d::gpu_check(d::packed_attention_gpu(p,nullptr));
        }
        d::gpu_check(cudaStreamSynchronize(nullptr));
        std::vector<uint8_t> guard(guard_bytes);
        d::gpu_check(cudaMemcpy(guard.data(),w.engram_packed,guard.size(),cudaMemcpyDeviceToHost));
        if(!std::all_of(guard.begin(),guard.end(),[](uint8_t x){return x==0xa5;}))
            throw std::runtime_error("attention LSE overwrote adjacent workspace allocation");
        if(splits==1){
            std::vector<uint8_t> partial(partial_bytes);
            d::gpu_check(cudaMemcpy(partial.data(),w.partial_out,partial.size(),cudaMemcpyDeviceToHost));
            if(!std::all_of(partial.begin(),partial.end(),[](uint8_t x){return x==0x5a;}))
                throw std::runtime_error("single-split attention overwrote partial scratch");
        }
        std::cout<<"workspace attention count="<<count<<" splits="<<splits<<" LSE span="<<lse_bytes
                 <<" adjacent canary="<<guard_bytes<<" intact\n";
    }
}
void compact_workspace_budget(){
    d::Workspace w(2ull<<30,1024,1048576);
    if(w.index_elements> (32ull<<20)/sizeof(float)||w.index_max_queries<1||
       w.index_gpu_bytes>(256ull<<20)||w.arena->used()>(2ull<<30))
        throw std::runtime_error("1M compact index workspace exceeds its bounded region");
    std::cout<<"1M compact workspace used="<<w.arena->used()<<" index="<<w.index_gpu_bytes
             <<" score_elements="<<w.index_elements<<" max_queries="<<w.index_max_queries<<'\n';
    d::gpu_check(cudaMemset(w.scores,0,w.index_elements*sizeof(float)));
    for(int rows:{1,16383,16384,16385,524288,1048576}){
        const int tokens=std::min<int>(w.index_max_queries,w.index_elements/rows);
        w.sort.temporary_bytes=w.sort_capacity;
        d::gpu_check(d::sort_index_gpu(w.scores,tokens,rows,w.sort,nullptr));
        const int blocks=(rows+7)/8;
        d::gpu_check(d::candidate_scores_gpu(w.scores,w.block_scores,tokens,rows,1048576-tokens,1,nullptr));
        w.block_sort.temporary_bytes=w.block_sort_capacity;
        d::gpu_check(d::sort_index_gpu(w.block_scores,tokens,blocks,w.block_sort,nullptr));
        d::gpu_check(cudaDeviceSynchronize());
        int last=-1;
        d::gpu_check(cudaMemcpy(&last,w.sort.sorted_ids+int64_t(tokens)*rows-1,sizeof(last),cudaMemcpyDeviceToHost));
        if(last!=rows-1)throw std::runtime_error("bounded index sort span/tie order");
        std::cout<<"bounded sort rows="<<rows<<" queries="<<tokens<<" passed\n";
    }
}

void routed(int type,int tokens,cublasHandle_t blas){
    constexpr int hidden=5120,ff=2304;
    c::NativeFmt f;std::string error;
    if(!c::native_fmt(type,type,hidden,ff,f,error))throw std::runtime_error(error);
    if(f.bf16_gu)throw std::runtime_error("native BF16 gate/up must remain opt-in");
    if((type==10||type==11)&&!c::native_enable_bf16_gu(f,error))throw std::runtime_error(error);
    if(f.bf16_gu!=(type==10||type==11))throw std::runtime_error("routed BF16 gate/up scope");
    f.swiglu_limit=10;f.bf16_io=true;
    if(!c::native_enable_bf16_down(f,error))throw std::runtime_error(error);
    std::mt19937 rng(41+type);std::normal_distribution<float> normal(0,.15f);
    std::vector<uint8_t> blob(f.bytes);std::vector<float> row(hidden);
    auto pack=[&](uint64_t offset,int rows,int cols,uint64_t stride){
        for(int r=0;r<rows;++r){
            for(int i=0;i<cols;++i)row[i]=normal(rng);
            if(ggml_quantize_chunk(static_cast<ggml_type>(type),row.data(),blob.data()+offset+r*stride,0,1,cols,nullptr)!=stride)
                throw std::runtime_error("expert fixture quantization");
        }
    };
    pack(0,ff,hidden,f.gu_row);pack(f.up_off,ff,hidden,f.gu_row);pack(f.down_off,hidden,ff,f.d_row);
    std::vector<float> x(tokens*hidden),weights(tokens),cpu(tokens*hidden);
    for(int t=0;t<tokens;++t){weights[t]=.15f+.03f*t;for(int dim=0;dim<hidden;++dim)x[t*hidden+dim]=bf(std::sin((dim+17*t)*.013f));}
    std::vector<std::vector<uint8_t>> acts(tokens,std::vector<uint8_t>(f.act_bytes));
    std::vector<c::ExpertJobMulti> jobs;
    for(int start=0;start<tokens;start+=c::MAXT){
        c::ExpertJobMulti job;job.blob=blob.data();job.nt=std::min(c::MAXT,tokens-start);job.ff_weights=weights.data()+start;
        for(int t=0;t<job.nt;++t){c::native_quant_act(f,x.data()+(start+t)*hidden,acts[start+t].data());job.nact[t]=acts[start+t].data();job.out[t]=cpu.data()+(start+t)*hidden;}
        jobs.push_back(job);
    }
    c::ExpertPool pool(3,false,true);pool.run_split_multi_native(f,jobs.data(),jobs.size());
    for(auto& value:cpu)value=bf(value);
    Device<uint8_t> db(blob);Device<float> dx(x),dwgt(weights),dequant(hidden*ff),gate(tokens*ff),up(tokens*ff),act(tokens*ff),out(tokens*hidden);
    Device<uint16_t> weight_bf(hidden*ff),scratch(tokens*hidden);
    auto project=[&](uint64_t offset,int rows,int cols,const float* input,float* output){
        d::gpu_check(d::dequant_gpu(db.p+offset,type,dequant.p,uint64_t(rows)*cols,nullptr));
        d::gpu_check(d::float_to_bf16_gpu(dequant.p,weight_bf.p,uint64_t(rows)*cols,nullptr));
        d::Matrix m{weight_bf.p,30,rows,cols,uint64_t(rows)*cols};
        d::matrix_product(m,input,output,tokens,blas,nullptr,scratch.p,d::Arithmetic::production);
    };
    project(0,ff,hidden,dx.p,gate.p);project(f.up_off,ff,hidden,dx.p,up.p);
    d::gpu_check(d::swiglu_gpu(gate.p,up.p,dwgt.p,act.p,ff,tokens,10,nullptr));
    d::gpu_check(d::round_bf16_gpu(act.p,uint64_t(tokens)*ff,nullptr));
    project(f.down_off,hidden,ff,act.p,out.p);
    const auto gpu=out.read();double error2=0,denominator=0,maximum=0;
    for(size_t i=0;i<gpu.size();++i){
        if(!std::isfinite(gpu[i])||!std::isfinite(cpu[i]))throw std::runtime_error("nonfinite routed expert output");
        const double delta=gpu[i]-cpu[i];error2+=delta*delta;denominator+=double(gpu[i])*gpu[i];maximum=std::max(maximum,std::abs(delta));
    }
    const double l2=std::sqrt(error2/denominator);
    std::cout<<"CPU/GPU routed type="<<type<<" tokens="<<tokens<<" bf16_gu="<<f.bf16_gu<<" relative_l2="<<l2<<" max_abs="<<maximum<<'\n';
    if(l2>1e-2){
        // Isolate down-activation quantization from gate/up and weight rounding.
        std::vector<float> cpu_ff(tokens*ff),decoded(ff),float_down(tokens*hidden);
        for(int start=0;start<tokens;start+=c::MAXT){
            const int count=std::min(c::MAXT,tokens-start);const void* inputs[c::MAXT];float* outputs[c::MAXT];
            for(int t=0;t<count;++t){inputs[t]=acts[start+t].data();outputs[t]=cpu_ff.data()+(start+t)*ff;}
            c::native_gu_rows(f,blob.data(),inputs,count,outputs,0,ff);
        }
        for(int t=0;t<tokens;++t)for(int dim=0;dim<ff;++dim)cpu_ff[t*ff+dim]=bf(cpu_ff[t*ff+dim]*weights[t]);
        const auto* btraits=ggml_get_type_traits_cpu(GGML_TYPE_BF16);
        std::vector<ggml_bf16_t> ff_bf(tokens*ff),weight_row(ff);
        btraits->from_float(cpu_ff.data(),ff_bf.data(),tokens*ff);
        for(int r=0;r<hidden;++r){
            ggml_get_type_traits(static_cast<ggml_type>(type))->to_float(blob.data()+f.down_off+r*f.d_row,decoded.data(),ff);
            btraits->from_float(decoded.data(),weight_row.data(),ff);
            for(int t=0;t<tokens;++t){float v=0;btraits->vec_dot(ff,&v,0,weight_row.data(),0,ff_bf.data()+t*ff,0,1);float_down[t*hidden+r]=bf(v);}
        }
        double improved=0,ff_error=0,ff_denom=0;const auto gpu_ff=act.read();
        for(size_t i=0;i<gpu.size();++i){const double delta=float_down[i]-gpu[i];improved+=delta*delta;}
        for(size_t i=0;i<gpu_ff.size();++i){const double delta=cpu_ff[i]-gpu_ff[i];ff_error+=delta*delta;ff_denom+=double(gpu_ff[i])*gpu_ff[i];}
        std::cout<<"diagnostic gate/up L2="<<std::sqrt(ff_error/ff_denom)<<" BF16 down L2="<<std::sqrt(improved/denominator)<<std::endl;
        throw std::runtime_error("CPU/GPU routed expert exceeds 1 percent relative L2");
    }
}
}
int main(){
    try{
        cublasHandle_t blas=nullptr;if(cublasCreate(&blas)!=CUBLAS_STATUS_SUCCESS||cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("test BLAS");
        struct Cleanup{cublasHandle_t h;~Cleanup(){cublasDestroy(h);}} cleanup{blas};
        helpers();strided(blas);strided_round();workspace_attention_bounds();compact_workspace_budget();workspace_auto_budget();
        for(int type:{10,11,39})for(int tokens:{3,12})routed(type,tokens,blas);
        std::cout<<"deepseek41_runtime_test passed\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
