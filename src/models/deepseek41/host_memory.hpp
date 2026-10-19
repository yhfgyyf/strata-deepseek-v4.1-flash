#pragma once
#include "expert_host_layout.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace strata::models::deepseek41 {
// These owners do not synchronize consumers. Pipelines must drain their streams
// and release source leases before destroying them, as with ordinary pinned RAM.
struct HostMemoryInfo {
    uint64_t logical=0,mapped=0,huge_before=0,huge_after=0,attempted_huge=0;
    uint64_t registration_count=0,largest_registration=0;
    double allocation_ms=0,registration_ms=0;
    const char* mode="ordinary";
    const char* fallback="none";
    void write(std::ostream& out)const {
        out<<"{\"logical_bytes\":"<<logical<<",\"mapped_bytes\":"<<mapped
           <<",\"huge_before_bytes\":"<<huge_before<<",\"huge_after_bytes\":"<<huge_after
           <<",\"attempted_huge_bytes\":"<<attempted_huge
           <<",\"registration_count\":"<<registration_count<<",\"largest_registration_bytes\":"<<largest_registration
           <<",\"allocation_ms\":"<<allocation_ms<<",\"registration_ms\":"<<registration_ms
           <<",\"mode\":\""<<mode<<"\",\"fallback\":\""<<fallback<<"\"}";
    }
};
inline double host_elapsed(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
class HugeHostRegion {
public:
    static constexpr uint64_t page=expert_host_page;
    ~HugeHostRegion(){reset();}
    HugeHostRegion()=default;
    HugeHostRegion(const HugeHostRegion&)=delete;
    uint8_t* get()const{return data_;}
    uint64_t size()const{return bytes_;}
    void reset(){
#ifdef __linux__
        if(reservation_)munmap(reservation_,reserved_);
#endif
        reservation_=nullptr;data_=nullptr;reserved_=bytes_=0;
    }
    bool allocate(uint64_t count,HostMemoryInfo& info){
        if(data_)throw std::logic_error("host region already allocated");
        const auto start=std::chrono::steady_clock::now();
        info.logical=count;
        if(!count || count>SIZE_MAX-4*page){info.fallback="size";return false;}
#ifdef __linux__
        uint64_t pmd_size=0;
        std::ifstream policy("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size");policy>>pmd_size;
        if(pmd_size!=page){info.fallback="pmd_size";return false;}
        bytes_=(count+page-1)/page*page;reserved_=bytes_+3*page;
        void* address=mmap(nullptr,reserved_,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(address==MAP_FAILED){reserved_=bytes_=0;info.fallback="mmap";return false;}
        reservation_=address;
        const auto base=reinterpret_cast<uintptr_t>(address);
        data_=reinterpret_cast<uint8_t*>((base+2*page-1)/page*page);
        if(mprotect(data_,bytes_,PROT_READ|PROT_WRITE) || madvise(data_,bytes_,MADV_HUGEPAGE)){
            info.fallback="madvise";reset();return false;
        }
        // Fault every byte before registration: advice/alignment alone is not proof.
        std::memset(data_,0,bytes_);
        info.mapped=bytes_;info.huge_before=huge_bytes();
        for(int attempt=0;attempt<4&&info.huge_before!=bytes_;++attempt){
            if(attempt)std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const auto before=info.huge_before;
            const auto collapse_start=std::chrono::steady_clock::now();
            // Linux UAPI MADV_COLLAPSE=25 (available since 6.1); this host's
            // userspace headers predate it. Operate only on our unregistered VMA.
            constexpr int collapse_advice=25;
            const int status=madvise(data_,bytes_,collapse_advice);
            const int error=status?errno:0;
            info.huge_before=huge_bytes();
            std::cerr<<"DeepSeek host_memory collapse {\"bytes\":"<<bytes_
                <<",\"huge_before\":"<<before<<",\"huge_after\":"<<info.huge_before
                <<",\"attempt\":"<<attempt+1<<",\"status\":"<<status<<",\"errno\":"<<error
                <<",\"milliseconds\":"<<host_elapsed(collapse_start)<<"}\n";
            if(status&&error!=ENOMEM&&error!=EAGAIN)break;
        }
        info.allocation_ms=host_elapsed(start);
        if(info.huge_before!=bytes_){info.attempted_huge=info.huge_before;info.huge_before=0;info.fallback="incomplete_thp";reset();return false;}
        return true;
#else
        info.fallback="platform";return false;
#endif
    }
    uint64_t huge_bytes()const{
        if(!data_)return 0;
        std::ifstream input("/proc/self/smaps");std::string line;
        bool own=false;uint64_t total=0;uintptr_t mapped_end=0;
        const auto begin=reinterpret_cast<uintptr_t>(data_),end=begin+bytes_;
        while(std::getline(input,line)){
            uintptr_t lo=0,hi=0;char dash=0;
            std::istringstream header(line);
            if(header>>std::hex>>lo>>dash>>hi && dash=='-'){own=lo>=begin && hi<=end;mapped_end=hi;}
            if(own && line.rfind("AnonHugePages:",0)==0){
                uint64_t kib=0;std::istringstream value(line.substr(14));value>>kib;total+=kib*1024;
                // procfs generates smaps lazily. Do not walk unrelated, large
                // pinned mappings after the guarded region has been verified.
                if(mapped_end==end)return total;
            }
        }
        return total;
    }
private:
    void* reservation_=nullptr;
    uint8_t* data_=nullptr;
    uint64_t reserved_=0,bytes_=0;
};
struct HostRegistration {
    void* data;
    uint64_t bytes;
    HostRegistration(void* pointer,uint64_t count):data(pointer),bytes(count){
        const auto error=cudaHostRegister(data,bytes,cudaHostRegisterPortable);
        if(error!=cudaSuccess)throw std::runtime_error(std::string("DeepSeek host registration: ")+cudaGetErrorString(error));
        registered_=true;
    }
    ~HostRegistration(){if(registered_)cudaHostUnregister(data);}
    HostRegistration(const HostRegistration&)=delete;
    static std::unique_ptr<HostRegistration> attempt(void* pointer,uint64_t count){
        auto owner=std::unique_ptr<HostRegistration>(new HostRegistration(pointer,count,Unregistered{}));
        const auto error=cudaHostRegister(pointer,count,cudaHostRegisterPortable);
        if(error==cudaSuccess){owner->registered_=true;return owner;}
        if(error!=cudaErrorMemoryAllocation && error!=cudaErrorNotSupported && error!=cudaErrorInvalidValue)
            throw std::runtime_error(std::string("DeepSeek huge host registration: ")+cudaGetErrorString(error));
        cudaGetLastError();return nullptr;
    }
private:
    struct Unregistered{};
    HostRegistration(void* pointer,uint64_t count,Unregistered):data(pointer),bytes(count){}
    bool registered_=false;
};
inline bool register_huge(HugeHostRegion& region,std::vector<std::unique_ptr<HostRegistration>>& registrations,
                          uint64_t count,HostMemoryInfo& info,const std::vector<uint64_t>& boundaries={}){
    if(!boundaries.empty()){
        if(boundaries.size()<2 || boundaries.front()!=0 || boundaries.back()!=count)
            throw std::invalid_argument("DeepSeek host registration boundaries");
        for(size_t i=1;i<boundaries.size();++i)
            if(boundaries[i]<=boundaries[i-1] || boundaries[i]%HugeHostRegion::page)
                throw std::invalid_argument("DeepSeek host registration alignment");
    }
    if(!region.allocate(count,info))return false;
    const auto start=std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<HostRegistration>> pending;
    try{
        const auto spans=boundaries.empty()?std::vector<uint64_t>{0,region.size()}:boundaries;
        pending.reserve(spans.size()-1);
        for(size_t i=1;i<spans.size();++i){
            auto owner=HostRegistration::attempt(region.get()+spans[i-1],spans[i]-spans[i-1]);
            if(!owner)break;
            pending.push_back(std::move(owner));
        }
        info.registration_ms=host_elapsed(start);
        if(pending.size()==spans.size()-1){
            info.huge_after=region.huge_bytes();
            if(info.huge_after==region.size()){
                for(const auto& owner:pending)info.largest_registration=std::max(info.largest_registration,owner->bytes);
                info.registration_count=pending.size();
                registrations=std::move(pending);
                info.mode=boundaries.empty()?"registered_2m":"registered_2m_blocks";return true;
            }
            info.fallback="post_register_thp";
        }else info.fallback="registration";
    }catch(...){pending.clear();region.reset();throw;}
    // Registration must be released before the backing mapping, including fallback.
    pending.clear();region.reset();info.attempted_huge=info.huge_before;info.huge_before=info.huge_after=0;return false;
}
class HostArena {
public:
    void allocate(uint64_t count,bool prefer_huge,const std::vector<uint64_t>& boundaries={}){
        if(get())throw std::logic_error("expert host arena already allocated");
        info.logical=count;
        if(prefer_huge){
            if(register_huge(huge_,registrations_,count,info,boundaries))return;
            if(!boundaries.empty()){
                std::cerr<<"DeepSeek host_memory rejected ";info.write(std::cerr);std::cerr<<'\n';
                throw std::runtime_error("DeepSeek block-pinned memory requires 100% huge-page registration");
            }
        }
        const auto start=std::chrono::steady_clock::now();
        heap_.reset(new uint8_t[count]);info.allocation_ms+=host_elapsed(start);
        info.mapped=count;info.mode="ordinary";
    }
    void register_if_needed(){
        if(!registrations_.empty())return;
        const auto start=std::chrono::steady_clock::now();
        registrations_.push_back(std::make_unique<HostRegistration>(get(),info.logical));
        info.registration_count=1;info.largest_registration=info.logical;
        info.registration_ms+=host_elapsed(start);info.mode="registered_ordinary";
    }
    uint8_t* get()const{return huge_.get()?huge_.get():heap_.get();}
    explicit operator bool()const{return get()!=nullptr;}
    bool pinned()const{return !registrations_.empty();}
    const HostMemoryInfo& memory_info()const{return info;}
private:
    HostMemoryInfo info;
    std::unique_ptr<uint8_t[]> heap_;
    HugeHostRegion huge_;
    std::vector<std::unique_ptr<HostRegistration>> registrations_;
};
struct Pinned {
    uint8_t* data=nullptr;
    uint64_t bytes=0;
    bool pinned=true;
    std::unique_ptr<uint8_t[]> fallback;
    HostMemoryInfo info;
    explicit Pinned(uint64_t count,bool force_pageable=false,bool prefer_huge=false):bytes(count){
        info.logical=count;
        if(!force_pageable && prefer_huge && register_huge(huge_,registrations_,count,info)){
            data=huge_.get();return;
        }
        const auto start=std::chrono::steady_clock::now();
        const auto error=force_pageable?cudaErrorMemoryAllocation:cudaHostAlloc(reinterpret_cast<void**>(&data),count,cudaHostAllocDefault);
        if(error==cudaErrorMemoryAllocation){
            if(!force_pageable)cudaGetLastError();
            fallback.reset(new uint8_t[count]);data=fallback.get();pinned=false;info.mode="pageable";
            std::cerr<<"DeepSeek pageable staging fallback bytes="<<count<<'\n';
        }else if(error!=cudaSuccess)throw std::runtime_error(std::string("DeepSeek pinned allocation: ")+cudaGetErrorString(error));
        else info.mode="cuda_host_alloc";
        info.mapped=count;info.allocation_ms+=host_elapsed(start);
    }
    ~Pinned(){if(data && pinned && registrations_.empty())cudaFreeHost(data);}
    Pinned(const Pinned&)=delete;
private:
    HugeHostRegion huge_;
    std::vector<std::unique_ptr<HostRegistration>> registrations_;
};
}
