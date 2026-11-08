#pragma once
#include "bounded_gemm_generated.cuh"
#include "bounded_sm89_gemm_generated.cuh"
#include "bounded_sm89_schedule.cuh"
#include <mutex>

// Embedded cubins have no Python/Triton runtime dependency. The matching
// generator, PTX, compiler version and cubin digests are retained beside them.
static cudaError_t launch_bounded_gemm(void** ptrs,const int* rows,int count,int projection,cudaStream_t stream){
    if(projection<0||projection>1)return cudaErrorInvalidValue;
    static cudaLibrary_t libraries[2][2]{};
    static cudaKernel_t kernels[2][2]{};
    static int device_backend[32]{};
    static std::mutex mutex;
    int device;auto error=cudaGetDevice(&device);if(error)return error;
    if(device<0||device>=32)return cudaErrorNotSupported;
    const BoundedCubin* selected=nullptr;
    cudaKernel_t kernel;bool sm89=false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto& backend=device_backend[device];
        if(!backend){
            int major,minor;cudaError_t e;
            e=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);if(e)return e;
            e=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device);if(e)return e;
            backend=major==12&&minor==0?1:major==8&&minor==9?2:-1;
        }
        if(backend<0)return cudaErrorNotSupported;
        sm89=backend==2;
        selected=backend==1?&bounded_cubins[0][projection]:&bounded_sm89_cubins[0][projection];
        const auto& spec=*selected;
        auto& library=libraries[backend-1][projection];auto& fn=kernels[backend-1][projection];
        if(!library){
            auto e=cudaLibraryLoadData(&library,spec.data,nullptr,nullptr,0,nullptr,nullptr,0);if(e)return e;
            e=cudaLibraryGetKernel(&fn,library,spec.name);if(e)return e;
        }
        // The kernel object is shared by devices, but this attribute is per
        // device. Configure every selected device, including the second EP rank.
        if(spec.shared>48*1024){auto e=cudaKernelSetAttributeForDevice(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,spec.shared,device);if(e)return e;}
        kernel=fn;
    }
    const auto& spec=*selected;
    int tiles=0;for(int e=0;e<count;++e)tiles+=(rows[e]+spec.bm-1)/spec.bm;
    const int n=projection==0?4608:5120;tiles*=((n+spec.bn-1)/spec.bn);
    void* wp=ptrs+(projection?3*strata::models::deepseek41::BOUNDED_MOE_BATCH:0);void* xp=ptrs+(projection?4*strata::models::deepseek41::BOUNDED_MOE_BATCH:strata::models::deepseek41::BOUNDED_MOE_BATCH);void* yp=ptrs+(projection?5*strata::models::deepseek41::BOUNDED_MOE_BATCH:2*strata::models::deepseek41::BOUNDED_MOE_BATCH);
    void* nr=ptrs+6*strata::models::deepseek41::BOUNDED_MOE_BATCH;uint64_t scratch=0;
    // The SM89 reduction descriptors reproduce the fixed CUDA 13.2 L20
    // arithmetic contract. All dot products still execute in our grouped MMA.
    int split[2]={1,1},sliced[2]={0,0};
    if(sm89){
        if(count<1||count>2)return cudaErrorInvalidValue;
        for(int e=0;e<count;++e){
            if(rows[e]<1||rows[e]>8192)return cudaErrorInvalidValue;
            const auto* table=projection?grouped_arithmetic_1:grouped_arithmetic_0;
            for(int i=0;;++i)if(rows[e]<=table[i].last){split[e]=table[i].split;sliced[e]=table[i].sliced;break;}
        }
    }
    // Triton3.6 appends two zero-sized global/profile scratch pointers.
    void* args[]={&wp,&xp,&yp,&nr,&count,&scratch,&scratch};
    void* sm89_args[]={&wp,&xp,&yp,&nr,&count,&split[0],&split[1],&sliced[0],&sliced[1],&scratch,&scratch};
    return cudaLaunchKernel(reinterpret_cast<const void*>(kernel),dim3(tiles),dim3(spec.warps*32),sm89?sm89_args:args,spec.shared,stream);
}
