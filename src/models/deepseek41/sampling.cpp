#include "strata/models/deepseek41/runtime.hpp"
#include "strata/core/spec_prob.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace strata::models::deepseek41 {
float draw_uniform(uint64_t seed,uint64_t counter) {
    // Same Philox 4x32-10 counter mapping as Strata's CUDA sampler, including the disjoint spec streams.
    uint32_t c0=counter,c1=counter>>32,c2=seed,c3=seed>>32;
    for(uint32_t i=0;i<10;++i){
        const uint64_t a=uint64_t(0x9E3779B9u)*c0,b=uint64_t(0xBB67AE85u)*c2;
        const uint32_t n0=uint32_t(b>>32)^c1^i,n1=b,n2=uint32_t(a>>32)^c3,n3=a;
        c0=n0;c1=n1;c2=n2;c3=n3;
    }
    return float(c0>>8)*(1.f/16777216.f);
}
void validate_sampling(const kernels::SamplerParams& p) {
    if(p.gumbel)throw std::invalid_argument("DeepSeek Gumbel sampling is not implemented");
    if(!std::isfinite(p.temperature)||p.temperature<0||!std::isfinite(p.top_p)||!std::isfinite(p.min_p)||
       !std::isfinite(p.penalty_repeat)||!std::isfinite(p.penalty_freq)||!std::isfinite(p.penalty_present)||
       p.top_p<=0||p.top_p>1||p.min_p<0||p.min_p>1||p.penalty_repeat<=0)
        throw std::invalid_argument("DeepSeek sampling parameters");
}
Distribution distribution(const float* logits,int vocab,const kernels::SamplerParams& p,const std::vector<int>& history) {
    validate_sampling(p);
    if(vocab<1)throw std::invalid_argument("DeepSeek sampling vocabulary");
    std::vector<double> values(vocab);std::vector<int32_t> ids(vocab);std::iota(ids.begin(),ids.end(),0);
    std::unordered_map<int,int> count;
    const int take=std::min<int>(std::max(0,p.penalty_last_n),history.size());
    for(int i=history.size()-take;i<static_cast<int>(history.size());++i)++count[history[i]];
    for(int i=0;i<vocab;++i){
        if(std::isnan(logits[i])||logits[i]==INFINITY)throw std::runtime_error("DeepSeek nonfinite sampling logit");
        double value=logits[i];const auto it=count.find(i);
        if(it!=count.end()){value=value<=0?value*p.penalty_repeat:value/p.penalty_repeat;value-=it->second*p.penalty_freq+p.penalty_present;}
        values[i]=value;
    }
    auto ordered=[&](int a,int b){return values[a]>values[b]||(values[a]==values[b]&&a<b);};
    const int k=std::min(vocab,p.top_k<=0||p.top_k>64?64:p.top_k);
    std::partial_sort(ids.begin(),ids.begin()+k,ids.end(),ordered);ids.resize(k);
    if(values[ids[0]]==-INFINITY)throw std::runtime_error("DeepSeek empty sampling support");
    if(p.greedy||p.temperature==0){Distribution d;d.ids={ids[0]};d.probabilities={1};d.proposal_probabilities={1};return d;}
    std::vector<double> prob(k);double sum=0;
    for(int i=0;i<k;++i){prob[i]=std::exp(values[ids[i]]-values[ids[0]]);sum+=prob[i];}
    for(auto& x:prob)x/=sum;
    int keep=k;
    if(p.top_p<1){double cumulative=0;keep=0;while(keep<k){cumulative+=prob[keep++];if(cumulative>=p.top_p&&keep>=std::max(1,p.min_keep))break;}}
    if(p.min_p>0)while(keep>std::max(1,p.min_keep)&&prob[keep-1]<p.min_p*prob[0])--keep;
    ids.resize(keep);prob.resize(keep);sum=0;
    for(int i=0;i<keep;++i){prob[i]=std::exp((values[ids[i]]-values[ids[0]])/p.temperature);sum+=prob[i];}
    Distribution d;d.ids=std::move(ids);d.probabilities=std::move(prob);d.proposal_probabilities.resize(keep);
    for(int i=0;i<keep;++i){d.probabilities[i]/=sum;d.proposal_probabilities[i]=d.probabilities[i];}
    return d;
}
int sample(const Distribution& d,float uniform){
    if(d.ids.empty()||d.ids.size()!=d.probabilities.size()||uniform<0||uniform>=1)throw std::invalid_argument("DeepSeek sample support/uniform");
    return core::spec_sample({d.ids.data(),d.probabilities.data(),static_cast<int>(d.ids.size())},uniform);
}
} // namespace strata::models::deepseek41
