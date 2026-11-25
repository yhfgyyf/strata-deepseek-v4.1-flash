#pragma once
#include "strata/models/deepseek41/residency.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::models::deepseek41 {
struct AdaptiveExchange {
    ExpertKey in,out;
    int slot;
    uint64_t offset,bytes,generation=0;
    double gain;
};
inline int adaptive_index(ExpertKey key){return key.first*384+key.second;}
inline int adaptive_group(int layer){return layer<14?0:1;}

// Selection is CPU-only so the same policy can be checked independently of DMA.
// The caller assigns identical IDs only to identical NativeFmt layouts.
// A recent mask selects Maya heat/hysteresis, within Strata's existing DMA budget.
inline std::vector<AdaptiveExchange> choose_adaptive_exchanges(
    const std::array<uint64_t,40>& sizes,const std::vector<double>& heat,
    const std::vector<int32_t>& slots,const std::vector<uint64_t>& offsets,
    bool grouped,uint64_t capacity,uint64_t budget,const std::array<int,40>* layout_groups=nullptr,
    const std::vector<uint8_t>* recent=nullptr){
    constexpr int keys=40*384;
    constexpr auto no_ram=std::numeric_limits<uint64_t>::max();
    if(heat.size()!=keys||slots.size()!=keys||offsets.size()!=keys||(recent&&recent->size()!=keys))
        throw std::invalid_argument("adaptive selection table sizes");
    struct Ranked { double value;ExpertKey key; };
    // Reused per thread: building ~15K ranked entries in fresh vectors each verify window costs as much as ranking.
    thread_local std::array<std::vector<Ranked>,40> candidates,victims;
    for(auto& v:candidates)v.clear();
    for(auto& v:victims)v.clear();
    for(int l=0;l<40;++l)for(int e=0;e<384;++e){
        const int k=l*384+e,g=grouped?(layout_groups?layout_groups->at(l):adaptive_group(l)):l;
        if(g<0||g>=40)throw std::invalid_argument("adaptive format group ID");
        if(!std::isfinite(heat[k])||heat[k]<0)throw std::invalid_argument("adaptive nonfinite/negative heat");
        if(slots[k]>=0)victims[g].push_back({heat[k],{l,e}});
        else if((recent?recent->at(k)!=0:heat[k]>=2.0)&&offsets[k]!=no_ram)candidates[g].push_back({heat[k],{l,e}});
    }
    std::vector<AdaptiveExchange> choices;
    const auto hotter=[](const Ranked& a,const Ranked& b){return a.value!=b.value?a.value>b.value:a.key<b.key;};
    const auto colder=[](const Ranked& a,const Ranked& b){return a.value!=b.value?a.value<b.value:a.key<b.key;};
    for(int g=0;g<40;++g){
        auto& in=candidates[g];auto& out=victims[g];
        // Pair gains in[i]-out[i] are non-increasing in i and at most `capacity` exchanges are kept, so only the
        // leading ranks matter: sort a prefix of `capacity` pairs, extended while the last kept gain ties the next
        // (the final order breaks gain ties by key, which can promote a later pair of equal gain). Same selection
        // as sorting both lists completely, without ranking all ~7.4K resident experts every verify window.
        const size_t n=std::min(in.size(),out.size());
        size_t k=std::min<size_t>(n,std::max<uint64_t>(capacity,1));
        for(;;){
            const size_t ki=std::min(in.size(),k+1),ko=std::min(out.size(),k+1);
            std::partial_sort(in.begin(),in.begin()+ki,in.end(),hotter);
            std::partial_sort(out.begin(),out.begin()+ko,out.end(),colder);
            if(k>=n || in[k-1].value-out[k-1].value!=in[k].value-out[k].value)break;
            k=std::min(n,2*k);
        }
        for(size_t i=0;i<k;++i){
            if(recent?in[i].value<=out[i].value*1.25+1.0:in[i].value<out[i].value+1.5)break;
            const auto a=in[i].key,b=out[i].key;
            if(!sizes[a.first]||sizes[a.first]!=sizes[b.first]||offsets[adaptive_index(b)]!=no_ram)
                throw std::logic_error("adaptive incompatible exchange ownership/size");
            choices.push_back({a,b,slots[adaptive_index(b)],offsets[adaptive_index(a)],sizes[a.first],0,in[i].value-out[i].value});
        }
    }
    std::sort(choices.begin(),choices.end(),[](const auto& a,const auto& b){
        return a.gain!=b.gain?a.gain>b.gain:a.in<b.in;
    });
    std::vector<AdaptiveExchange> result;uint64_t used=0;
    for(const auto& choice:choices){
        if(result.size()==capacity)break;
        if(choice.bytes>(budget-used)/2)continue;
        result.push_back(choice);used+=2*choice.bytes;
    }
    return result;
}
} // namespace strata::models::deepseek41
