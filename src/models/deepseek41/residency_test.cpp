#include "strata/models/deepseek41/residency.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

namespace d=strata::models::deepseek41;
namespace {
void require(bool okay,const char* text){if(!okay)throw std::runtime_error(text);}
void check(uint64_t limit,bool real){
    std::array<uint64_t,40> sizes{};
    for(int l=0;l<40;++l)sizes[l]=real?(l<14?15206400:11612160):(l<14?768:512);
    std::vector<d::ExpertKey> hot,rank;
    for(int e=0;e<80;++e)for(int l=0;l<40;++l)hot.emplace_back(l,e);
    for(int e=383;e>=0;--e)for(int l=0;l<40;++l)rank.emplace_back(l,e);
    const auto p=d::make_phase_plan(sizes,hot,rank,limit);
    require(p.bytes<=limit&&p.encoder_bytes<=p.bytes,"byte budget");
    uint64_t core_bytes=0,loan_bytes=0,temp_bytes=0;std::array<int,40> core{};std::set<d::ExpertKey> keys;
    for(auto key:p.core){require(keys.insert(key).second,"duplicate core");core_bytes+=sizes[key.first];++core[key.first];}
    for(const auto& slot:p.decode){
        require(slot.key.first>=20&&slot.bytes==sizes[slot.key.first]&&slot.offset==loan_bytes,"decode slot order/size");
        require(keys.insert(slot.key).second,"core/loan overlap");loan_bytes+=slot.bytes;
        int last=-1;
        for(const auto& temporary:p.encoder)if(temporary.offset<slot.offset+slot.bytes&&slot.offset<temporary.offset+temporary.bytes)
            last=std::max(last,temporary.key.first);
        require(last==slot.release_after,"early restoration deadline");
    }
    require(keys.size()==hot.size(),"decode target loss");
    for(int l=0;l<40;++l)require(core[l]>=32,"stable core floor");
    for(const auto& slot:p.encoder){
        require(slot.key.first<20&&slot.bytes==sizes[slot.key.first]&&slot.offset==temp_bytes,"encoder slot order/size");
        require(!keys.count(slot.key),"temporary expert loses original RAM complement");temp_bytes+=slot.bytes;
    }
    uint64_t wanted=0;for(auto key:hot)wanted+=sizes[key.first];
    require(core_bytes+loan_bytes==wanted&&loan_bytes==p.bytes&&temp_bytes==p.encoder_bytes,"capacity conservation");
    if(real&&limit>4ull<<30)require(p.bytes>4ull<<30,"64-bit mixed-size loan offsets");
}
bool contains(const std::vector<d::PhaseSlice>& slices,d::ExpertKey key){
    return std::any_of(slices.begin(),slices.end(),[&](const auto& slice){return slice.key==key;});
}
void sparse_and_global_rank(){
    std::array<uint64_t,40> sizes{};sizes.fill(512);
    std::vector<d::ExpertKey> hot;
    for(int e=0;e<32;++e)for(int l=0;l<40;++l)hot.emplace_back(l,e);
    hot.emplace_back(20,32);hot.emplace_back(21,32);
    auto prioritized=d::make_phase_plan(sizes,hot,{{0,200}},512);
    require(prioritized.decode.size()==1&&prioritized.decode[0].key==d::ExpertKey{21,32},"loan still uses layer roundrobin instead of global rank");
    hot.clear();
    for(int l=20;l<24;++l)for(int e=0;e<(l==20?31:l==21?32:l==22?33:1);++e)hot.emplace_back(l,e);
    auto p=d::make_phase_plan(sizes,hot,{{0,200}},4096);
    require(p.decode.size()==1&&p.decode[0].key==d::ExpertKey{22,32},"sparse layers were filled or loan protection changed");
    std::array<int,40> core{};for(auto [l,e]:p.core){(void)e;++core[l];}
    require(core[0]==0&&core[20]==31&&core[21]==32&&core[22]==32&&core[23]==1,"sparse actual core counts");
    require(p.core.size()+p.decode.size()==hot.size(),"sparse targets lost or synthesized");
    p=d::make_phase_plan(sizes,{{20,0},{21,0}},{{0,200}},512,0);
    require(p.decode.size()==1&&p.decode[0].key==d::ExpertKey{21,0},"loan must prefer lowest global rank, not lowest layer");
    sizes[21]=768;
    p=d::make_phase_plan(sizes,{{20,0},{21,0},{20,1}},{{0,200}},1024,0);
    require(p.bytes==512&&p.decode.size()==1&&p.decode[0].key==d::ExpertKey{20,1},"loan skipped a nonfitting rank to borrow a hotter expert");
    p=d::make_phase_plan(sizes,{{20,0}},{{0,200}},512,0);
    require(p.core.empty()&&p.decode.size()==1&&p.encoder.size()==1,"explicit floor0 did not allow an empty stable core");
    p=d::make_phase_plan(sizes,{{0,0},{19,0}},{{0,200}},4096);
    require(p.bytes==0&&p.encoder.empty()&&p.decode.empty()&&p.core.size()==2,"front-only targets must not synthesize a back loan");
    p=d::make_phase_plan(sizes,{},{{0,200}},4096);
    require(p.bytes==0&&p.core.empty()&&p.encoder.empty()&&p.decode.empty(),"zero global targets must stay empty");
}
void prefill_global_prefix_and_layout(){
    std::array<uint64_t,40> sizes{};sizes.fill(512);sizes[20]=1024;sizes[0]=768;sizes[1]=256;
    std::vector<d::ExpertKey> dense;
    for(int e=0;e<32;++e)for(int l=0;l<40;++l)dense.emplace_back(l,e);
    dense.emplace_back(20,32);
    auto p=d::make_phase_plan(sizes,dense,{{19,200},{0,201},{1,202}},1024);
    require(p.encoder.size()==1&&p.encoder[0].key==d::ExpertKey{19,200}&&p.encoder_bytes==512,"prefill must take global cold prefix and stop at first nonfit");
    p=d::make_phase_plan(sizes,{{20,0}},{{19,200}},1024,0);
    require(p.encoder.size()==1&&p.encoder_bytes==512,"nonempty partial prefill rank received invented low-ID fill");
    p=d::make_phase_plan(sizes,{{0,201},{20,0}},{{20,0},{0,201},{19,200},{0,202}},1024,0);
    require(p.encoder.size()==1&&p.encoder[0].key==d::ExpertKey{19,200},"prefill prefix did not filter already-hot and back keys");
    p=d::make_phase_plan(sizes,{{20,0}},{},1024,0);
    require(p.encoder.size()==2&&p.encoder[0].key==d::ExpertKey{0,0}&&p.encoder[1].key==d::ExpertKey{1,0}&&p.encoder_bytes==1024,"empty legacy prefill roundrobin/skip-nonfit selection changed");
    sizes.fill(256);sizes[20]=512;sizes[0]=768;
    p=d::make_phase_plan(sizes,{{20,0},{20,1}},{{19,200},{0,201}},1024,0);
    require(p.encoder.size()==2&&p.encoder[0].key==d::ExpertKey{0,201}&&p.encoder[1].key==d::ExpertKey{19,200},"global selection did not retain layer-packed layout");
    require(p.encoder[0].offset==0&&p.encoder[1].offset==768&&p.encoder_bytes==1024,"unequal temporary offsets");
    require(p.decode.size()==2&&p.decode[0].offset==0&&p.decode[1].offset==512&&p.decode[0].release_after==0&&p.decode[1].release_after==19,"mixed overlapping restoration deadlines");
    require(contains(p.decode,{20,0})&&contains(p.decode,{20,1}),"layer packing changed decode target IDs");
}
template<class Fn>void invalid(Fn fn){
    try{fn();}catch(const std::invalid_argument&){return;}
    throw std::runtime_error("invalid phase input was accepted");
}
void invalid_inputs(){
    std::array<uint64_t,40> sizes{};sizes.fill(512);
    invalid([&]{d::make_phase_plan(sizes,{}, {},4096,-1);});
    invalid([&]{d::make_phase_plan(sizes,{}, {},4096,385);});
    invalid([&]{d::make_phase_plan(sizes,{{40,0}}, {},4096);});
    invalid([&]{d::make_phase_plan(sizes,{{20,384}}, {},4096);});
    invalid([&]{d::make_phase_plan(sizes,{{20,0},{20,0}}, {},4096);});
    invalid([&]{d::make_phase_plan(sizes,{{20,0}},{{0,0},{40,0}},512,0);});
    sizes[0]=257;invalid([&]{d::make_phase_plan(sizes,{{20,0}}, {},4096);});
    sizes[0]=512;sizes[39]=0;invalid([&]{d::make_phase_plan(sizes,{{20,0}}, {},4096);});
}
}
int main(){
    try{
        for(uint64_t n:{uint64_t(0),uint64_t(511),uint64_t(512),uint64_t(1024),uint64_t(32768),uint64_t(1)<<30})check(n,false);
        check(8ull<<30,true);check(7ull<<30,true);
        sparse_and_global_rank();prefill_global_prefix_and_layout();invalid_inputs();
        std::cout<<"global decode/prefill priority, sparse cores, mixed byte caps and restoration deadlines passed\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
