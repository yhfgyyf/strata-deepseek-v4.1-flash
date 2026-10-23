#pragma once
#include "bounded_gemm_generated.cuh"
#include <mutex>

// Embedded cubins have no Python/Triton runtime dependency. The matching
// generator, PTX, compiler version and cubin digests are retained beside them.
static cudaError_t launch_bounded_gemm(void** ptrs,const int* rows,int count,int projection,cudaStream_t stream){
    if(projection<0||projection>1)return cudaErrorInvalidValue;
    static cudaLibrary_t libraries[1][2]{};
    static cudaKernel_t kernels[1][2]{};
    static std::mutex mutex;
    const auto& spec=bounded_cubins[0][projection];
    cudaKernel_t kernel;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto& library=libraries[0][projection];auto& fn=kernels[0][projection];
        if(!library){
            int device,major,minor;auto e=cudaGetDevice(&device);if(e)return e;
            e=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);if(e)return e;
            e=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device);if(e)return e;
            if(major!=12||minor!=0)return cudaErrorNotSupported;
            e=cudaLibraryLoadData(&library,spec.data,nullptr,nullptr,0,nullptr,nullptr,0);if(e)return e;
            e=cudaLibraryGetKernel(&fn,library,spec.name);if(e)return e;
            if(spec.shared>48*1024){e=cudaKernelSetAttributeForDevice(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,spec.shared,device);if(e)return e;}
        }
        kernel=fn;
    }
    int tiles=0;for(int e=0;e<count;++e)tiles+=(rows[e]+spec.bm-1)/spec.bm;
    const int n=projection==0?4608:5120;tiles*=((n+spec.bn-1)/spec.bn);
    void* wp=ptrs+(projection?3*strata::models::deepseek41::BOUNDED_MOE_BATCH:0);void* xp=ptrs+(projection?4*strata::models::deepseek41::BOUNDED_MOE_BATCH:strata::models::deepseek41::BOUNDED_MOE_BATCH);void* yp=ptrs+(projection?5*strata::models::deepseek41::BOUNDED_MOE_BATCH:2*strata::models::deepseek41::BOUNDED_MOE_BATCH);
    void* nr=ptrs+6*strata::models::deepseek41::BOUNDED_MOE_BATCH;uint64_t scratch=0;
    // Triton3.6 appends two zero-sized global/profile scratch pointers.
    void* args[]={&wp,&xp,&yp,&nr,&count,&scratch,&scratch};
    return cudaLaunchKernel(reinterpret_cast<const void*>(kernel),dim3(tiles),dim3(spec.warps*32),args,spec.shared,stream);
}
