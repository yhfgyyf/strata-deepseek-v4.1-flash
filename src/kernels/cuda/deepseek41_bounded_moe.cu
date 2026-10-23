#include "../../models/deepseek41/bounded_moe.hpp"
#include "../../models/deepseek41/runtime_kernels.hpp"
#include "strata/models/deepseek41/mixed_expert.hpp"
#include <cuda_bf16.h>
#include <algorithm>
#include "bounded_gemm_launch.cuh"

namespace strata::models::deepseek41 {
namespace {
constexpr int H=5120,I=2304,W=H*I;
struct Batch {
    BoundedMoeGroup group[BOUNDED_MOE_BATCH];
    int local[BOUNDED_MOE_BATCH];
};
// Dynamic descriptor indexing must stay in kernel parameter memory. Without
// grid_constant NVCC materializes a 104-byte per-thread stack copy on SM120.
__global__ void bounded_gather(const __grid_constant__ Batch batch,const float* x,const int* rows,
                              __nv_bfloat16* bx,__nv_bfloat16* gu,__nv_bfloat16* act,void** ptrs) {
    const int e=blockIdx.y;const auto g=batch.group[e];const int base=batch.local[e];
    if(blockIdx.x==0&&threadIdx.x==0){
        ptrs[e]=g.gate_bf;ptrs[BOUNDED_MOE_BATCH+e]=bx+int64_t(base)*H;
        ptrs[2*BOUNDED_MOE_BATCH+e]=gu+int64_t(base)*2*I;
        ptrs[3*BOUNDED_MOE_BATCH+e]=g.down_bf;ptrs[4*BOUNDED_MOE_BATCH+e]=act+int64_t(base)*I;
        ptrs[5*BOUNDED_MOE_BATCH+e]=bx+int64_t(base)*H;
        reinterpret_cast<int*>(ptrs+6*BOUNDED_MOE_BATCH)[e]=g.tokens;
    }
    for(int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<int64_t(g.tokens)*H;j+=int64_t(gridDim.x)*blockDim.x){
        const int t=j/H,d=j%H;const int r=rows?rows[g.offset+t]:g.offset+t;
        bx[int64_t(base)*H+j]=__float2bfloat16_rn(x[int64_t(r)*H+d]);
    }
}
__global__ void bounded_activation(const __grid_constant__ Batch batch,const __nv_bfloat16* gu,const float* routing,__nv_bfloat16* act) {
    const int e=blockIdx.y;const auto group=batch.group[e];const int base=batch.local[e];
    for(int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<int64_t(group.tokens)*I;j+=int64_t(gridDim.x)*blockDim.x){
        const int t=j/I,d=j%I;const int64_t row=int64_t(base+t)*2*I;
        const float g=fminf(__bfloat162float(gu[row+d]),10.f);
        const float u=fminf(fmaxf(__bfloat162float(gu[row+I+d]),-10.f),10.f);
        const float sigmoid=g>=0?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));
        const float r=routing?routing[group.offset+t]:1.f;
        act[int64_t(base)*I+j]=__float2bfloat16_rn(g*sigmoid*u*r);
    }
}
__global__ void bounded_scatter(const __grid_constant__ Batch batch,const __nv_bfloat16* bx,const int* slots,float* out) {
    const int e=blockIdx.y;const auto group=batch.group[e];const int base=batch.local[e];
    for(int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<int64_t(group.tokens)*H;j+=int64_t(gridDim.x)*blockDim.x){
        const int t=j/H,d=j%H;const int r=slots?slots[group.offset+t]:group.offset+t;
        out[int64_t(r)*H+d]=__bfloat162float(bx[int64_t(base)*H+j]);
    }
}
bool supported(int type){return type==8||type==11||type==16||type==17||type==18||type==21||type==22||type==39;}
int dequant(const void* packed,int type,uint16_t* bf,cudaStream_t stream){
    return expert_mixed_type(type)?expert_mixed_dequant_gpu(packed,type,bf,W,stream):dequant_bf16_gpu(packed,type,bf,W,stream);
}
}
int bounded_moe_prepare_gpu(const BoundedMoeGroup& g,int gu_type,int down_type,cudaStream_t stream){
    if(!supported(gu_type)||!supported(down_type)||!g.gate||!g.up||!g.down||!g.gate_bf||!g.down_bf)return cudaErrorInvalidValue;
    int rc=dequant(g.gate,gu_type,g.gate_bf,stream);if(rc)return rc;
    rc=dequant(g.up,gu_type,g.gate_bf+W,stream);if(rc)return rc;
    return dequant(g.down,down_type,g.down_bf,stream);
}
int bounded_moe_gpu(const BoundedMoeGroup* groups,int count,int gu_type,int down_type,
                    const float* x,const int* rows,const float* routing,const int* slots,float* out,
                    uint16_t* bx,uint16_t* gu,uint16_t* act,void** ptrs,int capacity,
                    cudaStream_t stream,bool prepared) {
    if(count<1||count>BOUNDED_MOE_BATCH||!supported(gu_type)||!supported(down_type)||
       !groups||!x||!out||!bx||!gu||!act||!ptrs)return cudaErrorInvalidValue;
    Batch batch{};int total=0,largest=0;
    for(int e=0;e<count;++e){
        const auto& g=groups[e];if(g.tokens<1||g.tokens>capacity-total||g.offset<0||!g.gate||!g.up||!g.down||!g.gate_bf||!g.down_bf)return cudaErrorInvalidValue;
        batch.group[e]=g;
        // Gate/Up is dead before Down DQ; write Down into its first half.
        // Already-prepared cold batches retain their separate Down bank.
        if(!prepared)batch.group[e].down_bf=g.gate_bf;
        batch.local[e]=total;total+=g.tokens;largest=std::max(largest,g.tokens);
    }
    if(!prepared)for(int e=0;e<count;++e){
        int rc=dequant(groups[e].gate,gu_type,groups[e].gate_bf,stream);if(rc)return rc;
        rc=dequant(groups[e].up,gu_type,groups[e].gate_bf+W,stream);if(rc)return rc;
    }
    auto* xb=reinterpret_cast<__nv_bfloat16*>(bx);auto* gb=reinterpret_cast<__nv_bfloat16*>(gu);auto* ab=reinterpret_cast<__nv_bfloat16*>(act);
    bounded_gather<<<dim3(std::min(4096,(largest*H+255)/256),count),256,0,stream>>>(batch,x,rows,xb,gb,ab,ptrs);
    auto err=cudaGetLastError();if(err)return err;
    int counts[BOUNDED_MOE_BATCH];for(int e=0;e<count;++e)counts[e]=groups[e].tokens;
    err=launch_bounded_gemm(ptrs,counts,count,0,stream);if(err)return err;
    bounded_activation<<<dim3(std::min(4096,(largest*I+255)/256),count),256,0,stream>>>(batch,gb,routing,ab);
    err=cudaGetLastError();if(err)return err;
    if(!prepared)for(int e=0;e<count;++e){const int rc=dequant(groups[e].down,down_type,batch.group[e].down_bf,stream);if(rc)return rc;}
    err=launch_bounded_gemm(ptrs,counts,count,1,stream);if(err)return err;
    bounded_scatter<<<dim3(std::min(4096,(largest*H+255)/256),count),256,0,stream>>>(batch,xb,slots,out);
    return cudaGetLastError();
}
} // namespace strata::models::deepseek41
