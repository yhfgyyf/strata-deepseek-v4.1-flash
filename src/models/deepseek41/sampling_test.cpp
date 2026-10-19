#include "strata/models/deepseek41/runtime.hpp"
#include "strata/core/spec_prob.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace strata::models::deepseek41;
namespace {
void check(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
void rejection(const std::array<int32_t,4>& qids,const std::array<float,4>& qvalues) {
    const std::array<int32_t,4> ids{11,12,13,14};const std::array<double,4> probabilities{.1,.2,.3,.4};
    const strata::core::SpecP target{ids.data(),probabilities.data(),4};
    const strata::core::SpecQ draft{qids.data(),qvalues.data(),4};
    std::array<double,4> qdouble{};for(int i=0;i<4;++i)qdouble[i]=qvalues[i];
    const strata::core::SpecP proposal{qids.data(),qdouble.data(),4};
    std::array<int,4> counts{};constexpr int n=200000;
    for(int i=0;i<n;++i){
        const int proposed=strata::core::spec_sample(proposal,draw_uniform(417,uint64_t(i)|strata::core::kSpecCtrDraft));
        const auto result=strata::core::spec_verify_row(target,proposed,draft,
            draw_uniform(417,uint64_t(i)|strata::core::kSpecCtrAccept),draw_uniform(417,uint64_t(i)|strata::core::kSpecCtrResid));
        bool found=false;for(int j=0;j<4;++j)if(result.token==ids[j]){++counts[j];found=true;}
        check(found,"residual token outside target support");
    }
    for(int i=0;i<4;++i)check(std::abs(double(counts[i])/n-probabilities[i])<.006,"rejection sampling changes target distribution");
}
}
int main(){
    try{
        strata::kernels::SamplerParams params;params.temperature=0;
        const float logits[]{-2,4,4,1};
        auto greedy=distribution(logits,4,params,{});check(greedy.ids==std::vector<int32_t>{1},"greedy tie selects lower ID");
        params.temperature=.7f;params.top_k=3;params.top_p=.8f;params.min_p=.01f;
        auto p=distribution(logits,4,params,{});double sum=0;for(double x:p.probabilities)sum+=x;check(std::abs(sum-1)<1e-12,"normalized post-chain distribution");
        params.penalty_last_n=3;params.penalty_repeat=2;params.penalty_freq=.5f;params.penalty_present=.5f;
        params.temperature=0;auto penalty=distribution(logits,4,params,{1,1,1});check(penalty.ids[0]==2,"penalties applied before pick");
        for(uint64_t i=0;i<1000;++i){const float u=draw_uniform(92,i);check(u>=0&&u<1,"Philox open upper bound");check(u==draw_uniform(92,i),"counter replay");}
        rejection({11,12,13,14},{.6f,.1f,.2f,.1f});rejection({21,22,23,24},{.25f,.25f,.25f,.25f});
        const int32_t id[]{7};const double prob[]{1};const strata::core::SpecP deterministic{id,prob,1};
        const strata::core::SpecQ delta{nullptr,nullptr,-1};
        check(strata::core::spec_verify_row(deterministic,7,delta,.99f,.1f).accepted==1,"certain proposal accepted");
        check(strata::core::spec_verify_row(deterministic,8,delta,.1f,.1f).token==7,"complete rejection uses target residual");
        std::cout<<"sampling chain, Philox replay and stochastic residual distribution PASS\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
