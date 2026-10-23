#include "host_memory.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
#ifdef __linux__
#include <sys/prctl.h>
#endif
namespace d=strata::models::deepseek41;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
uint8_t pattern(uint64_t index,int generation){return uint8_t((index*131+(index>>11)*17+generation*29)^0xa7);}
void transfer(uint8_t* source,uint8_t* destination,uint64_t bytes){
    uint8_t* device=nullptr;cudaStream_t stream=nullptr;cudaEvent_t done=nullptr;
    check(cudaMalloc(&device,bytes+128));
    struct Cleanup{uint8_t*& device;cudaStream_t& stream;cudaEvent_t& event;
        ~Cleanup(){if(stream)cudaStreamSynchronize(stream);if(event)cudaEventDestroy(event);if(stream)cudaStreamDestroy(stream);if(device)cudaFree(device);}}cleanup{device,stream,done};
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
    for(int generation=0;generation<3;++generation){
        for(uint64_t i=0;i<bytes;++i)source[i]=pattern(i,generation);
        std::fill_n(destination,bytes,uint8_t(0));
        check(cudaMemsetAsync(device,0x5a,bytes+128,stream));
        check(cudaMemcpyAsync(device+64,source,bytes,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(destination,device+64,bytes,cudaMemcpyDeviceToHost,stream));
        check(cudaEventRecord(done,stream));check(cudaEventSynchronize(done));
        for(uint64_t i=0;i<bytes;++i)need(destination[i]==pattern(i,generation),"transport byte mismatch");
        uint8_t guard[128];check(cudaMemcpy(guard,device,64,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(guard+64,device+bytes+64,64,cudaMemcpyDeviceToHost));
        need(std::all_of(guard,guard+128,[](uint8_t b){return b==0x5a;}),"device guard corrupted");
    }
}
void coverage(const d::HostMemoryInfo& info,bool huge,bool blocks=false){
    if(huge)need(info.huge_before==info.mapped && info.huge_after==info.mapped && info.mapped% d::HugeHostRegion::page==0 &&
                 std::string(info.mode)==(blocks?"registered_2m_blocks":"registered_2m"),"actual full 2MiB coverage missing");
    else need(info.huge_before==0 && info.huge_after==0,"fallback still reports huge mapping");
}
void layout_checks(){
    need(d::host_huge_sufficient(98,100,98) && !d::host_huge_sufficient(97,100,98),"98 percent boundary");
    need(d::host_huge_sufficient(99,101,98) && !d::host_huge_sufficient(98,101,98),"fractional threshold rounding");
    need(!d::host_huge_sufficient(99,100,100) && d::host_huge_sufficient(100,100,100),"strict default boundary");
    need(!d::host_huge_sufficient(0,0,98) && !d::host_huge_sufficient(101,100,98),"invalid coverage");
    need(d::host_huge_sufficient(UINT64_MAX,UINT64_MAX,98) &&
        d::host_huge_sufficient(UINT64_MAX,UINT64_MAX,100),"large threshold overflow");
    constexpr uint64_t no_ram=std::numeric_limits<uint64_t>::max();
    std::vector<uint64_t> offsets{0,no_ram,1,no_ram,2,no_ram,3,4,no_ram,no_ram,no_ram,no_ram};
    const auto boundaries=d::make_expert_host_segments({15206400,11612160,15206400},4,offsets);
    need(boundaries==std::vector<uint64_t>({0,31457280,67108864}),"mixed-size block padding");
    need(offsets==std::vector<uint64_t>({0,no_ram,15206400,no_ram,31457280,no_ram,43069440,54681600,no_ram,no_ram,no_ram,no_ram}),"cold expert order or hot omission");
    std::vector<uint64_t> all_hot(8,no_ram);
    need(d::make_expert_host_segments({15206400,11612160},4,all_hot)==std::vector<uint64_t>({0}),"empty block registration");
    std::vector<uint64_t> large(384,0);
    const auto split=d::make_expert_host_segments({15206400},384,large);
    need(split.size()==3,"large block must split below 4GiB");
    for(size_t i=1;i<split.size();++i)need(split[i]%d::expert_host_page==0 && split[i]>split[i-1] &&
        split[i]-split[i-1]<=d::expert_host_registration_limit,"oversized or unaligned registration");
    for(size_t i=0;i<large.size();++i){
        const auto end=std::upper_bound(split.begin(),split.end(),large[i]);
        need(end!=split.end() && large[i]+15206400<=*end,"expert straddles registrations");
        if(i)need(large[i]>=large[i-1]+15206400,"expert blobs overlap");
    }
    // Actual old/new 40-layer blob sizes and the frozen 4645-hot distribution.
    const std::vector<int> hot_count{150,140,138,114,121,111,100,102,83,100,102,99,118,141,106,136,94,114,115,160,106,101,128,142,94,94,107,111,90,98,107,110,113,122,120,116,119,138,147,138};
    for(const auto& sizes:std::vector<std::vector<uint64_t>>{{15206400,15206400,15206400,15206400,15206400,15206400,15206400,15206400,15206400,15206400,15206400,15206400,15206400,15206400,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160,11612160},{13086720,13086720,12349440,12349440,12349440,13824000,13824000,13824000,13824000,13824000,12625920,12625920,12625920,12625920,12625920,15298560,15298560,15298560,15298560,15298560,15298560,15298560,15298560,15298560,15298560,16404480,16404480,16404480,16404480,16404480,16404480,16404480,16404480,16404480,16404480,14100480,14100480,14100480,15206400,14100480}}){
        std::vector<uint64_t> mixed(40*384,0);
        for(int l=0;l<40;++l)for(int e=0;e<hot_count[l];++e)mixed[l*384+e]=no_ram;
        const auto plan=d::make_expert_host_segments(sizes,384,mixed,40);
        need(plan.size()==41,"balanced host segment count");
        for(size_t i=1;i<plan.size();++i)need(plan[i]%d::expert_host_page==0 &&
            plan[i]>plan[i-1] && plan[i]-plan[i-1]<=d::expert_host_registration_limit,"balanced segment span");
        uint64_t previous_end=0;size_t cold=0;
        for(size_t i=0;i<mixed.size();++i){
            if(int(i%384)<hot_count[i/384]){need(mixed[i]==no_ram,"balanced hot omission");continue;}
            const auto end=std::upper_bound(plan.begin(),plan.end(),mixed[i]);
            need(end!=plan.end() && mixed[i]+sizes[i/384]<=*end && mixed[i]>=previous_end,"balanced blob crossing or overlap");
            previous_end=mixed[i]+sizes[i/384];++cold;
        }
        need(cold==10715,"balanced cold coverage");
    }
    for(const auto count:{1u,3u}){
        std::vector<uint64_t> invalid(2,17);const auto before=invalid;bool rejected=false;
        try{d::make_expert_host_segments({d::expert_host_registration_limit},2,invalid,count);}
        catch(const std::invalid_argument&){rejected=true;}
        need(rejected && invalid==before,"balanced infeasible plan changed offsets");
    }
    for(bool oversized:{false,true}){
        std::vector<uint64_t> bad{17};const auto before=bad;bool rejected=false;
        try{d::make_expert_host_segments({oversized?d::expert_host_registration_limit+1:15206400},oversized?1:2,bad);}
        catch(const std::invalid_argument&){rejected=true;}
        need(rejected && bad==before,"invalid layout changed offsets");
    }
    for(const auto& bad:std::vector<std::vector<uint64_t>>{{0},{0,1,2*d::expert_host_page},{0,d::expert_host_page,d::expert_host_page,2*d::expert_host_page}}){
        d::HostArena arena;bool rejected=false;
        try{arena.allocate(2*d::expert_host_page,true,bad);}catch(const std::invalid_argument&){rejected=true;}
        need(rejected && !arena && !arena.pinned(),"invalid registration plan acquired memory");
    }
    std::cout<<"EXPERT_HOST_LAYOUT_ALIGNMENT_OMISSION_LARGE_BLOCK_BOUNDARIES_PASS\n";
}
void block_checks(const std::string& mode){
    const std::vector<uint64_t> sizes{11612160,15206400,11612160};
    std::vector<uint64_t> offsets(3,0);
    const auto boundaries=d::make_expert_host_segments(sizes,1,offsets);
    d::HostArena arena;
    if(mode=="blocks-error"){
        bool rejected=false;
        try{arena.allocate(boundaries.back(),true,boundaries);}catch(const std::runtime_error&){rejected=true;}
        need(rejected && !arena && !arena.pinned(),"fatal partial registration leaked ownership");
        cudaGetLastError();return;
    }
    const bool huge=mode=="blocks";
    arena.allocate(boundaries.back(),mode!="blocks-ordinary",boundaries);
    arena.register_if_needed();arena.register_if_needed();
    need(arena.pinned(),"block arena is not pinned");coverage(arena.memory_info(),huge,true);
    need(arena.memory_info().registration_count==(huge?3:1),"wrong registration count or fallback");
    if(huge)need(arena.memory_info().largest_registration==16777216,"wrong maximum registration span");
    if(mode=="blocks-fail")need(std::string(arena.memory_info().fallback)=="registration","partial failure did not fall back");
    d::Pinned readback(*std::max_element(sizes.begin(),sizes.end()));
    for(size_t i=0;i<sizes.size();++i)transfer(arena.get()+offsets[i],readback.data,sizes[i]);
    std::cout<<"block arena ";arena.memory_info().write(std::cout);std::cout<<'\n';
}
}
int main(int argc,char** argv){try{
    const std::string mode=argc>1?argv[1]:"huge";
    layout_checks();
    if(mode=="layout")return 0;
    if(mode=="disabled" || mode=="blocks-disabled"){
#ifdef __linux__
        need(prctl(PR_SET_THP_DISABLE,1,0,0,0)==0,"disable THP for test process");
#endif
    }
    if(mode.rfind("blocks",0)==0){
        block_checks(mode);check(cudaDeviceSynchronize());
        std::cout<<"HOST_BLOCK_REGISTRATION_BYTES_GUARDS_FALLBACK_PASS "<<mode<<'\n';return 0;
    }
    const bool huge=mode=="huge",prefer=mode!="ordinary";
    const bool fail_arena=mode=="fail-arena",fail_staging=mode=="fail-staging";
    for(uint64_t bytes:{11612160ull,15206400ull,53477376ull}){
        if(!fail_staging){
            d::HostArena arena;arena.allocate(bytes,prefer);arena.register_if_needed();need(arena.pinned(),"arena is not registered");
            coverage(arena.memory_info(),huge);d::Pinned readback(bytes);
            transfer(arena.get(),readback.data,bytes);
            std::cout<<"arena ";arena.memory_info().write(std::cout);std::cout<<'\n';
        }
        if(!fail_arena){
            d::Pinned staging(bytes,false,prefer),readback(bytes);need(staging.pinned,"staging is not pinned");
            coverage(staging.info,huge);transfer(staging.data,readback.data,bytes);
            std::cout<<"staging ";staging.info.write(std::cout);std::cout<<'\n';
        }
        if(fail_arena || fail_staging)break;
    }
    {d::Pinned pageable(4096,true,true);need(!pageable.pinned && std::string(pageable.info.mode)=="pageable","pageable fallback");}
    {d::HugeHostRegion region;d::HostMemoryInfo info;need(!region.allocate(0,info) && !region.get(),"zero region");
     need(!region.allocate(std::numeric_limits<uint64_t>::max(),info) && !region.get(),"overflow region");}
    check(cudaDeviceSynchronize());std::cout<<"HOST_MEMORY_EXACT_BYTES_GUARDS_GENERATIONS_FALLBACK_PASS "<<mode<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
