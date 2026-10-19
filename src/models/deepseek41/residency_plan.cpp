#include "strata/models/deepseek41/residency.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strata::models::deepseek41 {
PhasePlan make_phase_plan(const std::array<uint64_t,40>& sizes,const std::vector<ExpertKey>& targets,
                          const std::vector<ExpertKey>& prefill_rank,uint64_t limit,int floor) {
    PhasePlan plan;
    if(floor<0||floor>384)throw std::invalid_argument("phase decode core floor");
    std::array<std::vector<int>,40> hot,ranking;
    for(auto bytes:sizes)if(!bytes||bytes%256)throw std::invalid_argument("phase residency expert geometry");
    std::vector<uint8_t> seen(40*384),selected(40*384),cold_seen(40*384);
    for(const auto& [l,e]:targets){
        if(l<0||l>=40||e<0||e>=384||seen[l*384+e])
            throw std::invalid_argument("phase residency target/geometry");
        seen[l*384+e]=1;hot[l].push_back(e);
    }
    // The floor protects existing targets only; sparse global admission never fills a layer.
    std::array<int,40> loans{},protected_count{};
    for(int l=0;l<40;++l)protected_count[l]=std::min<int>(floor,hot[l].size());
    // Targets retain global decode rank. Borrow its eligible back-layer tail, coldest first.
    for(auto it=targets.rbegin();it!=targets.rend();++it){
        const auto [l,e]=*it;
        if(l<20||hot[l].size()<=static_cast<size_t>(protected_count[l]+loans[l]))continue;
        if(sizes[l]>limit-plan.bytes)break;
        selected[l*384+e]=1;++loans[l];plan.bytes+=sizes[l];
    }
    for(const auto& key:targets)if(!selected[key.first*384+key.second])plan.core.push_back(key);
    uint64_t offset=0;
    for(int l=20;l<40;++l)for(int e:hot[l])if(selected[l*384+e]){
        plan.decode.push_back({{l,e},offset,sizes[l],-1});offset+=sizes[l];
    }
    std::vector<ExpertKey> cold_rank;
    auto add=[&](int l,int e){
        if(l<0||l>=40||e<0||e>=384)throw std::invalid_argument("phase prefill profile ID");
        if(l<20&&!seen[l*384+e]&&!cold_seen[l*384+e]){cold_seen[l*384+e]=1;cold_rank.emplace_back(l,e);}
    };
    for(const auto& [l,e]:prefill_rank)add(l,e);
    if(prefill_rank.empty()){
        // Unprofiled legacy selection: roundrobin cold IDs, skipping layers that no longer fit.
        for(int l=0;l<20;++l)for(int e=0;e<384;++e)if(!seen[l*384+e])ranking[l].push_back(e);
        std::array<int,20> count{};bool changed=true;
        while(changed){
            changed=false;
            for(int l=0;l<20;++l)if(count[l]<static_cast<int>(ranking[l].size())&&sizes[l]<=plan.bytes-plan.encoder_bytes){
                ++count[l];plan.encoder_bytes+=sizes[l];changed=true;
            }
        }
        for(int l=0;l<20;++l)ranking[l].resize(count[l]);
    }else for(const auto& [l,e]:cold_rank){
        // Real profiles keep their global cold prefix; partial profiles receive no invented fill.
        if(sizes[l]>plan.bytes-plan.encoder_bytes)break;
        ranking[l].push_back(e);plan.encoder_bytes+=sizes[l];
    }
    offset=0;
    // Layer packing preserves the existing consumer/release deadline contract after global selection.
    for(int l=0;l<20;++l)for(int e:ranking[l]){plan.encoder.push_back({{l,e},offset,sizes[l],l});offset+=sizes[l];}
    size_t first=0;
    for(auto& back:plan.decode){
        while(first<plan.encoder.size()&&plan.encoder[first].offset+plan.encoder[first].bytes<=back.offset)++first;
        for(size_t i=first;i<plan.encoder.size()&&plan.encoder[i].offset<back.offset+back.bytes;++i)
            back.release_after=std::max(back.release_after,plan.encoder[i].key.first);
    }
    return plan;
}
} // namespace strata::models::deepseek41
