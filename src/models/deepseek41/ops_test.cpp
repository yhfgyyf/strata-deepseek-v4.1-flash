#include "strata/models/deepseek41/kernels.hpp"
#include "strata/models/deepseek41/ops.hpp"
#include "strata/models/deepseek41/engram.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace d = strata::models::deepseek41;
namespace {
void cuda_check(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
template<class T> struct Device {
    T* p = nullptr; size_t count;
    explicit Device(size_t n) : count(n) { cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    explicit Device(const std::vector<T>& x) : Device(x.size()) { upload(x); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    void upload(const std::vector<T>& x) { cuda_check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> read() { std::vector<T> x(count); cuda_check(cudaMemcpy(x.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost)); return x; }
};
void close(const std::vector<float>& got, const std::vector<float>& ref, const char* label, double atol=1e-5, double rtol=1e-4) {
    if (got.size() != ref.size()) throw std::runtime_error("test shape");
    double maximum = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double error = std::abs(double(got[i]) - ref[i]); maximum = std::max(maximum,error);
        if (!std::isfinite(got[i]) || error > atol + rtol * std::abs(ref[i])) {
            std::cerr << label << " index=" << i << " got=" << got[i] << " ref=" << ref[i] << "\n";
            throw std::runtime_error(label);
        }
    }
    std::cout << label << " max_abs=" << maximum << '\n';
}
}
int main() {
    try {
        std::mt19937 generator(410);
        std::normal_distribution<float> normal(0.f,0.15f);
        const int hidden=5120, tokens=3;
        std::vector<float> input(hidden*tokens), norm(hidden), reference(hidden*tokens);
        for (auto& x : input) x=normal(generator);
        for (auto& x : norm) x=1.f+normal(generator);
        Device<float> dx(input), dw(norm), dout(reference.size());
        cuda_check(d::rms_gpu(dx.p,dw.p,dout.p,hidden,tokens,1e-20f,nullptr));
        for (int t=0;t<tokens;++t) d::rms_cpu(input.data()+t*hidden,norm.data(),reference.data()+t*hidden,hidden,1e-20f);
        close(dout.read(),reference,"RMS 5120");
        for (int type : {0,8,10,11,30,39}) {
            const int rows=17, cols=5120;
            std::vector<float> weights(rows*cols), oracle(rows*cols), gemv_ref(tokens*rows);
            for (auto& x : weights) x=normal(generator);
            std::vector<uint8_t> packed(ggml_row_size(static_cast<ggml_type>(type),cols)*rows);
            ggml_quantize_chunk(static_cast<ggml_type>(type),weights.data(),packed.data(),0,rows,cols,nullptr);
            if (type == 0) oracle = weights;
            else ggml_get_type_traits(static_cast<ggml_type>(type))->to_float(packed.data(),oracle.data(),oracle.size());
            Device<uint8_t> dp(packed); Device<float> dq(oracle.size()), dy(gemv_ref.size());
            cuda_check(d::dequant_gpu(dp.p,type,dq.p,oracle.size(),nullptr));
            close(dq.read(),oracle,("dequant type="+std::to_string(type)).c_str(),0,0);
            cuda_check(d::packed_gemv_gpu(dp.p,type,dx.p,dy.p,rows,cols,tokens,nullptr));
            for(int t=0;t<tokens;++t)for(int r=0;r<rows;++r){double value=0;for(int k=0;k<cols;++k)value+=double(oracle[r*cols+k])*input[t*cols+k];gemv_ref[t*rows+r]=value;}
            close(dy.read(),gemv_ref,("GEMV type="+std::to_string(type)).c_str());
        }
        std::vector<float> gate(2304*tokens), up(gate.size()), route{0.1f,0.3f,0.7f}, activated(gate.size());
        for(size_t i=0;i<gate.size();++i){gate[i]=normal(generator)*200;up[i]=normal(generator)*200;activated[i]=d::swiglu_cpu(gate[i],up[i])*route[i/2304];}
        Device<float> dg(gate), du(up), dr(route), da(activated.size());
        cuda_check(d::swiglu_gpu(dg.p,du.p,dr.p,da.p,2304,tokens,10.f,nullptr));
        close(da.read(),activated,"clamped routed SwiGLU");
        for(const auto& shape : {std::pair{384,6},std::pair{128,3}}){
            const auto [experts,top]=shape;
            std::vector<float> logits(tokens*experts), bias(experts), vl(experts), expected(tokens*top);
            for(auto& x:logits)x=normal(generator)*20;
            for(auto& x:bias)x=normal(generator);
            for(auto& x:vl)x=normal(generator);
            std::vector<uint8_t> image{0,1,0}; std::vector<int> expected_ids(tokens*top);
            Device<float> dl(logits),db(bias),dv(vl),dweights(expected.size());Device<uint8_t>di(image);Device<int>dids(expected_ids.size());
            cuda_check(d::route_gpu(dl.p,db.p,dv.p,di.p,dids.p,dweights.p,experts,top,tokens,1.5f,nullptr));
            for(int t=0;t<tokens;++t)d::route_cpu(logits.data()+t*experts,bias.data(),vl.data(),image[t],experts,top,expected_ids.data()+t*top,expected.data()+t*top);
            if(dids.read()!=expected_ids)throw std::runtime_error("GPU router IDs");
            close(dweights.read(),expected,"router unbiased weights");
            Device<int> parallel_ids(expected_ids.size());Device<float> parallel_weights(expected.size());
            cuda_check(d::route_parallel_gpu(dl.p,db.p,dv.p,di.p,parallel_ids.p,parallel_weights.p,experts,top,tokens,1.5f,nullptr));
            if(parallel_ids.read()!=dids.read() || parallel_weights.read()!=dweights.read())throw std::runtime_error("parallel router must preserve IDs and weight bits");
            for(size_t j=0;j<logits.size();j+=5)logits[j]=std::numeric_limits<float>::quiet_NaN();dl.upload(logits);
            cuda_check(d::route_gpu(dl.p,db.p,dv.p,di.p,dids.p,dweights.p,experts,top,tokens,1.5f,nullptr));
            cuda_check(d::route_parallel_gpu(dl.p,db.p,dv.p,di.p,parallel_ids.p,parallel_weights.p,experts,top,tokens,1.5f,nullptr));
            if(parallel_ids.read()!=dids.read() || parallel_weights.read()!=dweights.read())throw std::runtime_error("parallel routing must skip individual NaN candidates");
            std::fill(logits.begin(),logits.end(),std::numeric_limits<float>::quiet_NaN());dl.upload(logits);
            cuda_check(d::route_parallel_gpu(dl.p,db.p,dv.p,di.p,parallel_ids.p,parallel_weights.p,experts,top,tokens,1.5f,nullptr));
            if(parallel_ids.read()!=std::vector<int>(tokens*top,-1) || parallel_weights.read()!=std::vector<float>(tokens*top,0))throw std::runtime_error("nonfinite routing must reject IDs without reading an invalid score");
        }
        std::vector<float> residual(tokens*4*hidden), fn(24*4*hidden), mixes(tokens*24), coeff_base(24), coeff_scale{0.2f,0.3f,0.1f};
        for(auto& x:residual)x=normal(generator);
        for(auto& x:fn)x=normal(generator)*0.01f;
        for(auto& x:coeff_base)x=normal(generator);
        Device<float>dres(residual),dfn(fn),dmix(mixes.size()),dbase(coeff_base),dscale(coeff_scale),dpre(tokens*4),dpost(tokens*4),dcomb(tokens*16);
        cuda_check(d::hc_project_gpu(dres.p,dfn.p,dmix.p,hidden,tokens,1e-20f,nullptr));
        for(int t=0;t<tokens;++t)d::hc_project_cpu(residual.data()+t*4*hidden,fn.data(),mixes.data()+t*24,hidden,1e-20f);
        close(dmix.read(),mixes,"mHC global projection");
        std::vector<float>pre(tokens*4),post(tokens*4),comb(tokens*16),collapsed(tokens*hidden),merged(residual.size());
        cuda_check(d::hc_split_gpu(dmix.p,dscale.p,dbase.p,dpre.p,dpost.p,dcomb.p,tokens,1e-6f,20,nullptr));
        for(int t=0;t<tokens;++t)d::hc_split_cpu(mixes.data()+t*24,coeff_scale.data(),coeff_base.data(),pre.data()+t*4,post.data()+t*4,comb.data()+t*16);
        close(dpre.read(),pre,"mHC next pre");close(dpost.read(),post,"mHC post");close(dcomb.read(),comb,"mHC Sinkhorn");
        cuda_check(d::hc_pre_gpu(dres.p,dpre.p,dout.p,hidden,tokens,nullptr));
        for(int t=0;t<tokens;++t)d::hc_pre_cpu(residual.data()+t*4*hidden,pre.data()+t*4,collapsed.data()+t*hidden,hidden);
        close(dout.read(),collapsed,"mHC carried pre collapse");
        Device<float>dmerged(merged.size());
        cuda_check(d::hc_post_gpu(dres.p,dx.p,dpost.p,dcomb.p,dmerged.p,hidden,tokens,nullptr));
        for(int t=0;t<tokens;++t)d::hc_post_cpu(residual.data()+t*4*hidden,input.data()+t*hidden,post.data()+t*4,comb.data()+t*16,merged.data()+t*4*hidden,hidden);
        close(dmerged.read(),merged,"mHC delayed residual post");
        std::vector<float> eng_kv(tokens*5*hidden), eng_q(4*hidden), eng_k(4*hidden);
        for(auto& x:eng_kv)x=normal(generator);
        for(auto& x:eng_q)x=normal(generator);
        for(auto& x:eng_k)x=normal(generator);
        std::vector<uint8_t>eng_mask{0,1,0};
        Device<float>dekv(eng_kv),deq(eng_q),dek(eng_k);Device<uint8_t>dem(eng_mask);
        dres.upload(residual);auto eng_ref=residual;
        d::engram_gate_cpu(eng_ref.data(),eng_kv.data(),eng_q.data(),eng_k.data(),eng_mask.data(),tokens,hidden);
        cuda_check(d::engram_gate_gpu(dres.p,dekv.p,deq.p,dek.p,dem.p,tokens,hidden,nullptr));
        close(dres.read(),eng_ref,"Engram per-copy signed sqrt gate and image mask");
        d::Geometry geometry;
        std::vector<float>q(64*512);for(auto& x:q)x=normal(generator);
        for(bool compressed:{false,true})for(int position:{1,128,32767,131071,1048575}){
            auto ref=q;d::rotate_cpu(ref.data(),64,512,position,geometry,compressed);
            Device<float>dq(q);cuda_check(d::rotate_gpu(dq.p,64,512,1,position,geometry,compressed,false,nullptr));
            close(dq.read(),ref,"RoPE H64");
            cuda_check(d::rotate_gpu(dq.p,64,512,1,position,geometry,compressed,true,nullptr));close(dq.read(),q,"inverse RoPE H64");
        }
        cuda_check(cudaDeviceSynchronize());
        std::cout<<"deepseek41_ops_test: passed\n";return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
