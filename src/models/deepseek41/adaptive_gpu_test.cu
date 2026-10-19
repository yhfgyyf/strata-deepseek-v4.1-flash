#include "adaptive_cache.hpp"
#include "adaptive_test_access.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

namespace d=strata::models::deepseek41;
namespace {
__global__ void delay(unsigned long long clocks){const auto start=clock64();while(clock64()-start<clocks){} }
__global__ void checksum(const uint8_t* bytes,uint64_t n,unsigned long long* out){unsigned long long sum=0;for(uint64_t i=0;i<n;++i)sum+=bytes[i];*out=sum;}
void need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
uint8_t value(d::ExpertKey key){return 1+(key.first*17+key.second)%251;}
void run(bool grouped,bool aligned,bool pinned,bool huge=false){
    constexpr auto no_ram=strata::core::detail::kNoCacheComplement;
    std::array<uint64_t,40> sizes{};for(int l=0;l<40;++l)sizes[l]=l<14?768:512;
    std::vector<d::ExpertKey> hot;std::vector<int64_t> slot_sizes;
    for(int e=0;e<2;++e)for(int l=0;l<40;++l){hot.emplace_back(l,e);slot_sizes.push_back(sizes[l]);}
    strata::core::ExpertCache cache;std::string error;
    if(!cache.open_sized(slot_sizes,40,384,error))throw std::runtime_error(error);
    std::vector<uint64_t> offsets(40*384,no_ram);uint64_t bytes=0;
    for(int l=0;l<40;++l)for(int e=2;e<384;++e){offsets[l*384+e]=bytes;bytes+=sizes[l];}
    std::vector<uint8_t> ram(bytes);
    for(int l=0;l<40;++l)for(int e=2;e<384;++e)std::fill_n(ram.data()+offsets[l*384+e],sizes[l],value({l,e}));
    d::Stream main;d::ExecutionStats stats;
    for(size_t i=0;i<hot.size();++i){const auto key=hot[i];std::vector<uint8_t> source(sizes[key.first],value(key));
        need(cache.admit(key.first,key.second)==static_cast<int>(i),"initial cache admission order");
        if(!cache.fill_slot_blocking(i,source.data(),error,sizes[key.first]))throw std::runtime_error(error);
    }
    std::unique_ptr<d::Pinned> pinned_owner;
    if(pinned){pinned_owner=std::make_unique<d::Pinned>(bytes,false,huge);std::memcpy(pinned_owner->data,ram.data(),bytes);}
    auto* host=pinned?pinned_owner->data:ram.data();
    const auto original=hot;constexpr uint64_t budget=128*1024;
    d::AdaptiveCache adaptive(cache,hot,offsets,host,sizes,budget,false,grouped,pinned,huge);
    d::PrefetchBlockBenchmark::aligned(adaptive,aligned);
    const auto initial=strata::models::deepseek41::PrefetchBlockBenchmark::snapshot(adaptive);
    const auto initial_ram=ram;
    need(!cache.replace_checked(0,0,14,383,0,sizes[14]),"cross-size slot publication accepted");
    need(!cache.replace_checked(0,0,0,1,0,sizes[0]),"duplicate resident publication accepted");
    strata::core::DeviceArena result_arena(2*hot.size()*sizeof(unsigned long long));
    auto* result=static_cast<unsigned long long*>(result_arena.alloc(2*hot.size()*sizeof(unsigned long long)));
    std::vector<double> counts(40*384);
    for(int l=0;l<40;++l)for(int e=378;e<384;++e)counts[l*384+e]=4;
    delay<<<1,1,0,main.value>>>(100000000);d::gpu_check(cudaGetLastError());
    for(size_t i=0;i<original.size();++i)checksum<<<1,1,0,main.value>>>(cache.device_slot(i),sizes[original[i].first],result+i);
    if(aligned)for(int w=0;w<4;++w){
        adaptive.observe_window(6,counts);
        adaptive.tick(6,counts,main.value,stats);
        if(w<3)for(auto key:original)need(adaptive.resident(key.first,key.second),"exchange before four actual windows");
    }
    else adaptive.tick(4,counts,main.value,stats);
    need(strata::models::deepseek41::PrefetchBlockBenchmark::stale_guard(adaptive),"stale slot generation accepted");
    // Copy-back alone must keep old experts GPU-visible. Queue another set of
    // readers after the first fence: the subsequent overwrite must fence these too.
    for(const auto key:original)need(adaptive.resident(key.first,key.second),"D2H prematurely evicted GPU expert");
    delay<<<1,1,0,main.value>>>(300000000);d::gpu_check(cudaGetLastError());
    for(size_t i=0;i<original.size();++i)checksum<<<1,1,0,main.value>>>(cache.device_slot(i),sizes[original[i].first],result+original.size()+i);
    bool saved=false;
    for(int tries=0;tries<1000&&!saved;++tries){
        adaptive.progress(main.value,stats);saved=adaptive.cpu(0,0)!=nullptr;
        if(!saved)std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    need(saved,"copy-back did not reach loading state");
    for(const auto key:original){
        const auto* cpu=adaptive.cpu(key.first,key.second);
        need(cpu&&!adaptive.resident(key.first,key.second),"evicted expert has no CPU fallback");
        need(std::all_of(cpu,cpu+sizes[key.first],[&](auto x){return x==value(key);}),"CPU exchange payload differs");
    }
    adaptive.drain(main.value,stats);
    d::gpu_check(cudaStreamSynchronize(main.value));
    std::vector<unsigned long long> sums(2*original.size());
    d::gpu_check(cudaMemcpy(sums.data(),result,sums.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<sums.size();++i){const auto key=original[i%original.size()];need(sums[i]==sizes[key.first]*value(key),"H2D overwrote a live old-expert reader");}
    need(stats.phase_migrations==80,"one-to-one candidate/victim selection did not publish80 swaps");
    need(stats.phase_dynamic_h2d_bytes+stats.phase_dynamic_d2h_bytes<=budget,"migration exceeded total directional byte budget");
    auto verify=[&]{
        uint64_t cold=0;
        for(int l=0;l<40;++l)for(int e=0;e<384;++e){
            const auto at=offsets[l*384+e];const bool resident=adaptive.resident(l,e);
            need((at==no_ram)==resident,"GPU/RAM complement ownership mismatch");
            if(resident){std::vector<uint8_t> expected(sizes[l],value({l,e}));if(!cache.verify_slot(cache.slot_of(l,e),expected.data(),error,sizes[l]))throw std::runtime_error(error);}
            else{++cold;need(std::all_of(adaptive.cpu(l,e),adaptive.cpu(l,e)+sizes[l],[&](auto x){return x==value({l,e});}),"RAM complement bytes were lost");}
        }
        need(cold==40*384-original.size(),"cache population changed");
    };
    verify();
    const auto first_learning=adaptive.learned_heat();
    adaptive.begin_request(main.value,stats);std::fill(counts.begin(),counts.end(),0);
    need(adaptive.learned_heat()==first_learning,"request boundary erased learned startup heat");
    for(const auto key:original)counts[key.first*384+key.second]=8;
    const auto before=stats.phase_dynamic_h2d_bytes+stats.phase_dynamic_d2h_bytes;
    if(aligned)for(int w=0;w<4;++w)adaptive.observe_window(8,counts);
    adaptive.tick(8,counts,main.value,stats);
    // A new prompt boundary drains the in-flight exchange before any prefill reuse.
    adaptive.begin_request(main.value,stats);verify();
    for(size_t i=0;i<counts.size();++i)
        if(!aligned)need(adaptive.learned_heat()[i]==first_learning[i]+counts[i],"cross-request committed learning differs");
        else need(adaptive.learned_heat()[i]>=first_learning[i]+counts[i],"window learning lost observed routes");
    need(stats.phase_dynamic_h2d_bytes+stats.phase_dynamic_d2h_bytes-before<=budget,"second batch exceeded byte limit");
    if(!grouped)need(hot==original,"same-layer exchanges did not restore exact slot ownership");
    strata::models::deepseek41::PrefetchBlockBenchmark::restore(adaptive,initial,main.value,stats,grouped);
    verify();need(hot==original&&offsets==initial.offsets&&std::equal(initial_ram.begin(),initial_ram.end(),host),"A/B/A rewind lost exact initial weight bytes/owners");
    std::cout<<"aligned="<<aligned<<" pinned="<<pinned<<" ";
    std::cout<<(grouped?"grouped ":"same-layer ")<<"adaptive mixed-size exchange: live readers, CPU fallback, ready publication, full RAM/GPU payloads, byte cap, prompt drain and cross-request learning passed\n";
}
}
int main(){try{for(bool grouped:{false,true})for(bool aligned:{false,true})for(bool pinned:{false,true})run(grouped,aligned,pinned);run(true,true,true,true);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
