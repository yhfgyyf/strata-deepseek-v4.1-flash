#include "adaptive_policy.hpp"
#include <iostream>
#include <random>
#include <set>
namespace d=strata::models::deepseek41;
void need(bool x,const char* why){if(!x)throw std::runtime_error(why);}
using Key=d::ExpertKey;
struct Fixture {
    std::array<uint64_t,40> sizes{};
    std::vector<double> heat=std::vector<double>(15360);
    std::vector<int32_t> slots=std::vector<int32_t>(15360,-1);
    std::vector<uint64_t> offsets=std::vector<uint64_t>(15360);
    Fixture(){for(int l=0;l<40;++l)sizes[l]=l<14?768:512;for(int k=0;k<15360;++k)offsets[k]=uint64_t(k)*1024;}
    void hot(Key k,int slot,double h=0){slots[d::adaptive_index(k)]=slot;offsets[d::adaptive_index(k)]=UINT64_MAX;heat[d::adaptive_index(k)]=h;}
    auto plan(bool grouped,uint64_t cap=96,uint64_t bytes=128*1024){return d::choose_adaptive_exchanges(sizes,heat,slots,offsets,grouped,cap,bytes);}
};
// Brute-force maximum gain pairs, independent of the production sorted lists.
std::vector<d::AdaptiveExchange> oracle(const Fixture& f,bool grouped){
    std::set<int> used;std::vector<d::AdaptiveExchange> result;
    for(;;){double gain=-1;int a=-1,b=-1;
        for(int i=0;i<15360;++i)if(f.slots[i]<0&&f.heat[i]>=2&&!used.count(i))
        for(int j=0;j<15360;++j)if(f.slots[j]>=0&&!used.count(j)&&
            (grouped?d::adaptive_group(i/384)==d::adaptive_group(j/384):i/384==j/384)){
            const double g=f.heat[i]-f.heat[j];if(g<1.5)continue;
            if(g>gain||(g==gain&&(a<0||i<a||(i==a&&j<b)))){gain=g;a=i;b=j;}
        }
        if(a<0)break;
        result.push_back({{a/384,a%384},{b/384,b%384},f.slots[b],f.offsets[a],f.sizes[a/384],0,gain});
        used.insert(a);used.insert(b);
    }
    return result;
}
int main(){try{
    Fixture f;f.hot({0,7},0);f.hot({14,7},1);f.heat[13*384+7]=10;f.heat[39*384+7]=9;
    need(f.plan(false).empty(),"legacy per-layer quota changed");
    const auto swaps=f.plan(true);need(swaps.size()==2,"zero-quota layers did not acquire group slots");
    need(swaps[0].in==Key{13,7}&&swaps[0].out==Key{0,7}&&swaps[1].in==Key{39,7}&&swaps[1].out==Key{14,7},"full cross-layer key lost");
    need(f.plan(true,96,1535).size()==1&&f.plan(true,96,1535)[0].in.first==39,"budget must skip oversized group swap");
    need(f.plan(true,1).size()==1&&f.plan(true,0).empty(),"slot capacity bound");
    Fixture many;for(int e=0;e<128;++e){many.hot({14,e},e);many.heat[39*384+e]=8;}
    need(many.plan(true,96,256*1024).size()==96,"96-exchange limit was not reached or exceeded");
    f.heat[13*384+7]=1.999;need(f.plan(true).size()==1,"candidate threshold");
    f.heat[13*384+7]=2;f.heat[7]=.5;need(f.plan(true).size()==2,"exact gain threshold");
    f.heat[7]=.5001;need(f.plan(true).size()==1,"below gain threshold");
    f.heat[7]=0;f.sizes[13]=512;bool rejected=false;try{f.plan(true);}catch(const std::logic_error&){rejected=true;}need(rejected,"mixed-size group accepted");
    f.sizes[13]=768;f.heat[7]=std::numeric_limits<double>::quiet_NaN();rejected=false;
    try{f.plan(true);}catch(const std::invalid_argument&){rejected=true;}need(rejected,"NaN accepted");
    std::mt19937 rng(20261018);int checked=0;
    for(int trial=0;trial<80;++trial){Fixture z;int slot=0;
        for(int l:{0,13,14,39})for(int e=0;e<4;++e){z.heat[l*384+e]=double(rng()%100)/4;if(rng()%2)z.hot({l,e},slot++,z.heat[l*384+e]);}
        for(bool grouped:{false,true}){const auto a=z.plan(grouped),b=oracle(z,grouped);need(a.size()==b.size(),"oracle plan length");
            std::set<int> seen;uint64_t bytes=0;
            for(size_t i=0;i<a.size();++i){need(a[i].in==b[i].in&&a[i].out==b[i].out&&a[i].gain==b[i].gain,"oracle plan differs");
                need(seen.insert(d::adaptive_index(a[i].in)).second&&seen.insert(d::adaptive_index(a[i].out)).second,"duplicate endpoint");
                need(d::adaptive_group(a[i].in.first)==d::adaptive_group(a[i].out.first),"cross-format exchange");bytes+=2*a[i].bytes;
            }
            need(bytes<=128*1024,"byte budget");++checked;
        }
    }
    std::cout<<"PASS "<<checked<<" independent randomized oracle comparisons; boundaries, full keys, zero-quota, budget and invalid heat/geometry\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
