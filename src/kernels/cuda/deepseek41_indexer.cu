// DeepSeek-V4.1 index and candidate semantics follow the pinned official model (MIT).
#include "strata/models/deepseek41/indexer.hpp"
#include <cub/device/device_segmented_radix_sort.cuh>
#include <climits>
#include <cuda_bf16.h>
#include <mma.h>

namespace strata::models::deepseek41 {
namespace {
__device__ float fp4(uint8_t code) {
    constexpr float values[8]={0,.5f,1,1.5f,2,3,4,6};
    return code&8?-values[code&7]:values[code&7];
}
__global__ void scores_kernel(const float* queries,const float* weights,const float* keys,const uint8_t* packed,
                              const uint8_t* candidates,const int32_t* blocks,float* scores,int rows,int width,int64_t start,int ratio) {
    const int column=blockIdx.x,token=blockIdx.y,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int block=blocks?blocks[int64_t(token)*2048+column/8]:0;
    const int key=blocks?block*8+column%8:column;
    const int64_t visible=(start+token+1)/ratio;
    if(key<0||key>=rows||key>=visible||(candidates&&!candidates[int64_t(token)*rows+key])){
        if(!threadIdx.x)scores[int64_t(token)*width+column]=-INFINITY;
        return;
    }
    __shared__ float key_values[128],warp_sums[8];
    if(threadIdx.x<128){
        const int d=threadIdx.x;
        if(packed){const uint8_t* p=packed+int64_t(key)*68;
            key_values[d]=fp4((p[d/2]>>(d%2*4))&15)*__uint_as_float(uint32_t(p[64+d/32])<<23);
        }else key_values[d]=keys[int64_t(key)*128+d];
    }
    __syncthreads();
    float sum=0;
    for(int head=warp;head<32;head+=8){
        float dot=0;
        for(int d=lane;d<128;d+=32)dot+=queries[(int64_t(token)*32+head)*128+d]*key_values[d];
        for(int delta=16;delta;delta>>=1)dot+=__shfl_down_sync(0xffffffff,dot,delta);
        if(!lane)sum+=fmaxf(dot,0.f)*weights[token*32+head]*(1.f/64.f);
    }
    if(!lane)warp_sums[warp]=sum;
    __syncthreads();
    if(!threadIdx.x){float total=0;for(int w=0;w<8;++w)total+=warp_sums[w];scores[int64_t(token)*width+column]=total;}
}
__global__ void sort_inputs_kernel(int* ids,int64_t* offsets,int64_t count,int rows,int tokens) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x)ids[i]=i%rows;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<=tokens;i+=gridDim.x*blockDim.x)offsets[i]=int64_t(i)*rows;
}
__global__ void topk_kernel(const float* sorted_scores,const int* sorted_ids,const int32_t* blocks,int64_t* out,
                            int rows,int key_rows,int64_t start,int ratio) {
    const int token=blockIdx.x,i=threadIdx.x;
    __shared__ int positions[512];
    const int64_t visible=(start+token+1)/ratio;
    int id=INT_MAX;
    if(i<rows){const int column=sorted_ids[int64_t(token)*rows+i];
        const int candidate=blocks?blocks[int64_t(token)*2048+column/8]*8+column%8:column;
        if(sorted_scores[int64_t(token)*rows+i]>-INFINITY&&candidate>=0&&candidate<key_rows&&candidate<visible)id=candidate;}
    positions[i]=id;__syncthreads();
    for(int size=2;size<=512;size<<=1)for(int step=size>>1;step;step>>=1){
        const int partner=i^step;
        if(partner>i){const bool ascending=(i&size)==0;
            const int a=positions[i],b=positions[partner];
            if((a>b)==ascending){positions[i]=b;positions[partner]=a;}}
        __syncthreads();
    }
    out[int64_t(token)*512+i]=positions[i]==INT_MAX?-1:positions[i];
}
__global__ void candidate_scores_kernel(const float* scores,float* out,int rows,int blocks,int64_t start,int ratio) {
    const int block=blockIdx.x*blockDim.x+threadIdx.x,token=blockIdx.y;
    if(block>=blocks)return;
    const int64_t visible=min(int64_t(rows),(start+token+1)/ratio);
    float value=-INFINITY;
    for(int i=block*8;i<min((block+1)*8,rows)&&i<visible;++i)value=fmaxf(value,scores[int64_t(token)*rows+i]);
    if(visible>0&&block==(visible-1)/8)value=INFINITY;
    out[int64_t(token)*blocks+block]=value;
}
__global__ void mark_blocks_kernel(const float* scores,const int* ids,uint8_t* selected,int blocks) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x,token=blockIdx.y;
    if(index<min(blocks,2048)&&scores[int64_t(token)*blocks+index]>-INFINITY)
        selected[int64_t(token)*blocks+ids[int64_t(token)*blocks+index]]=1;
}
__global__ void expand_blocks_kernel(const uint8_t* selected,uint8_t* mask,int rows,int blocks,int64_t start,int ratio,int64_t n) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x){
        const int token=i/rows,key=i%rows;
        mask[i]=key<(start+token+1)/ratio&&selected[int64_t(token)*blocks+key/8];
    }
}
__global__ void candidate_blocks_kernel(const float* scores,const int* ids,int32_t* out,int blocks) {
    __shared__ int selected[2048];
    const int token=blockIdx.x;
    for(int i=threadIdx.x;i<2048;i+=blockDim.x)
        selected[i]=i<blocks&&scores[int64_t(token)*blocks+i]>-INFINITY?ids[int64_t(token)*blocks+i]:INT_MAX;
    __syncthreads();
    // Stable score ties were resolved by CUB in original block-ID order. Canonicalise the selected set so
    // compact key columns also have original-ID order when their scores tie at the next index source.
    for(int size=2;size<=2048;size<<=1)for(int step=size>>1;step;step>>=1){
        for(int i=threadIdx.x;i<2048;i+=blockDim.x){const int partner=i^step;
            if(partner>i){const bool ascending=(i&size)==0;const int a=selected[i],b=selected[partner];
                if((a>b)==ascending){selected[i]=b;selected[partner]=a;}}}
        __syncthreads();
    }
    for(int i=threadIdx.x;i<2048;i+=blockDim.x)out[int64_t(token)*2048+i]=selected[i]==INT_MAX?-1:selected[i];
}
__global__ void all_candidates(const float* scores,int* out,int blocks){
    const int token=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
    __shared__ int counts[64],offsets[64];
    for(int i=threadIdx.x;i<2048;i+=256)out[token*2048+i]=-1;
    for(int segment=warp;segment<64;segment+=8){
        const int i=segment*32+lane;
        const unsigned valid=__ballot_sync(0xffffffff,i<blocks&&scores[token*blocks+i]>-INFINITY);
        if(!lane)counts[segment]=__popc(valid);
    }
    __syncthreads();
    if(!threadIdx.x){int sum=0;for(int i=0;i<64;++i){offsets[i]=sum;sum+=counts[i];}}
    __syncthreads();
    for(int segment=warp;segment<64;segment+=8){
        const int i=segment*32+lane;
        const unsigned valid=__ballot_sync(0xffffffff,i<blocks&&scores[token*blocks+i]>-INFINITY);
        if(valid&(1u<<lane))out[token*2048+offsets[segment]+__popc(valid&((1u<<lane)-1))]=i;
    }
}
int grid(int64_t n){return std::min<int64_t>((n+255)/256,65535);}
}
cudaError_t index_scores_gpu(const float* q,const float* weights,const float* keys,const uint8_t* packed,const uint8_t* candidates,
                             float* scores,int tokens,int rows,int64_t start,int ratio,cudaStream_t stream){
    if(!q||!weights||(!keys&&!packed)||!scores||tokens<1||tokens>65535||rows<1||start<0||(ratio!=1&&ratio!=2))return cudaErrorInvalidValue;
    scores_kernel<<<dim3(rows,tokens),256,0,stream>>>(q,weights,keys,packed,candidates,nullptr,scores,rows,rows,start,ratio);
    return cudaGetLastError();
}
cudaError_t sort_index_gpu(const float* scores,int tokens,int rows,IndexSortBuffers& b,cudaStream_t stream){
    if(tokens<1||rows<1)return cudaErrorInvalidValue;
    const int64_t count=int64_t(tokens)*rows;
    if(b.temporary){
        if(!scores||!b.sorted_scores||!b.input_ids||!b.sorted_ids||!b.offsets)return cudaErrorInvalidValue;
        sort_inputs_kernel<<<grid(count),256,0,stream>>>(b.input_ids,b.offsets,count,rows,tokens);
        const auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
    }
    return cub::DeviceSegmentedRadixSort::SortPairsDescending(b.temporary,b.temporary_bytes,scores,b.sorted_scores,
        b.input_ids,b.sorted_ids,count,tokens,b.offsets,b.offsets?b.offsets+1:nullptr,0,32,stream);
}
cudaError_t index_topk_gpu(const IndexSortBuffers& b,int64_t* out,int tokens,int rows,int64_t start,int ratio,cudaStream_t stream){
    if(!b.sorted_scores||!b.sorted_ids||!out||tokens<1||rows<1||start<0||(ratio!=1&&ratio!=2))return cudaErrorInvalidValue;
    topk_kernel<<<tokens,512,0,stream>>>(b.sorted_scores,b.sorted_ids,nullptr,out,rows,rows,start,ratio);return cudaGetLastError();
}
cudaError_t candidate_scores_gpu(const float* scores,float* out,int tokens,int rows,int64_t start,int ratio,cudaStream_t stream){
    if(!scores||!out||tokens<1||tokens>65535||rows<1||start<0||(ratio!=1&&ratio!=2))return cudaErrorInvalidValue;
    const int blocks=(rows+7)/8;
    candidate_scores_kernel<<<dim3((blocks+255)/256,tokens),256,0,stream>>>(scores,out,rows,blocks,start,ratio);return cudaGetLastError();
}
cudaError_t candidate_mask_gpu(const IndexSortBuffers& b,uint8_t* selected,uint8_t* mask,int tokens,int rows,int64_t start,int ratio,cudaStream_t stream){
    if(!b.sorted_scores||!b.sorted_ids||!selected||!mask||tokens<1||tokens>65535||rows<1||start<0||(ratio!=1&&ratio!=2))return cudaErrorInvalidValue;
    const int blocks=(rows+7)/8;
    auto status=cudaMemsetAsync(selected,0,int64_t(tokens)*blocks,stream);if(status!=cudaSuccess)return status;
    mark_blocks_kernel<<<dim3((std::min(blocks,2048)+255)/256,tokens),256,0,stream>>>(b.sorted_scores,b.sorted_ids,selected,blocks);
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    expand_blocks_kernel<<<grid(int64_t(tokens)*rows),256,0,stream>>>(selected,mask,rows,blocks,start,ratio,int64_t(tokens)*rows);
    return cudaGetLastError();
}
cudaError_t candidate_all_blocks_gpu(const float* scores,int32_t* out,int tokens,int blocks,cudaStream_t stream) {
    if(!scores||!out||tokens<1||blocks<1||blocks>2048)return cudaErrorInvalidValue;
    all_candidates<<<tokens,256,0,stream>>>(scores,out,blocks);return cudaGetLastError();
}
cudaError_t candidate_blocks_gpu(const IndexSortBuffers& b,int32_t* out,int tokens,int rows,cudaStream_t stream){
    if(!b.sorted_scores||!b.sorted_ids||!out||tokens<1||rows<1)return cudaErrorInvalidValue;
    candidate_blocks_kernel<<<tokens,256,0,stream>>>(b.sorted_scores,b.sorted_ids,out,(rows+7)/8);
    return cudaGetLastError();
}
cudaError_t compact_index_scores_gpu(const float* q,const float* weights,const float* keys,const uint8_t* packed,
                                     const int32_t* blocks,float* scores,int tokens,int rows,int64_t start,int ratio,cudaStream_t stream){
    if(!q||!weights||(!keys&&!packed)||!blocks||!scores||tokens<1||tokens>65535||rows<1||start<0||
       (ratio!=1&&ratio!=2))return cudaErrorInvalidValue;
    scores_kernel<<<dim3(16384,tokens),256,0,stream>>>(q,weights,keys,packed,nullptr,blocks,scores,rows,16384,start,ratio);
    return cudaGetLastError();
}
cudaError_t compact_index_topk_gpu(const IndexSortBuffers& b,const int32_t* blocks,int64_t* out,
                                   int tokens,int rows,int64_t start,int ratio,cudaStream_t stream){
    if(!b.sorted_scores||!b.sorted_ids||!blocks||!out||tokens<1||rows<1||start<0||
       (ratio!=1&&ratio!=2))return cudaErrorInvalidValue;
    topk_kernel<<<tokens,512,0,stream>>>(b.sorted_scores,b.sorted_ids,blocks,out,16384,rows,start,ratio);
    return cudaGetLastError();
}
} // namespace strata::models::deepseek41

// SM89 BF16 Tensor Core scorer, with exact scalar fallback for unsafe inputs.
namespace strata::models::deepseek41 {
namespace {
__device__ __forceinline__ float ada_fp4(uint8_t c) {
    const unsigned m=c&7;
    const unsigned bits=m==0?0:m==1?0x3f000000u:((m/2+126)<<23)|((m&1)<<22);
    return __uint_as_float(bits|((c&8u)<<28));
}
__global__ void scores_bf16(const float* q,const float* weights,const uint8_t* packed,
                           const uint8_t* candidates,const int32_t* blocks,float* scores,
                           int rows,int width,int64_t start,int ratio) {
    constexpr int tile=64,stride=136;
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,token=blockIdx.y,first=blockIdx.x*tile;
    const int64_t visible=(start+token+1)/ratio;
    __shared__ __align__(32) __nv_bfloat16 matrices[(32+tile)*stride];
    auto* queries=matrices;
    auto* keys=matrices+32*stride;
    auto* dots=reinterpret_cast<float*>(matrices);
    __shared__ int key_ids[tile];
    if(tid<tile){
        const int c=first+tid;
        const int key=c<width?(blocks?blocks[int64_t(token)*2048+c/8]*8+c%8:c):-1;
        key_ids[tid]=key>=0&&key<rows&&key<visible&&(!candidates||candidates[int64_t(token)*rows+key])?key:-1;
    }
    if(!__syncthreads_or(tid<tile&&key_ids[tid]>=0)){
        if(tid<tile&&first+tid<width)scores[int64_t(token)*width+first+tid]=-INFINITY;
        return;
    }
    bool unsafe=false;
    for(int i=tid*4;i<32*128;i+=128*4){
        const float4 values=*reinterpret_cast<const float4*>(q+int64_t(token)*4096+i);
        const uint32_t bits[4]={__float_as_uint(values.x),__float_as_uint(values.y),
                               __float_as_uint(values.z),__float_as_uint(values.w)};
#pragma unroll
        for(int z=0;z<4;++z){
            const unsigned magnitude=bits[z]&0x7fffffffU;
            unsafe|=(bits[z]&0xffffU)!=0 || (magnitude!=0 &&
                (magnitude<0x24800000U || magnitude>0x5a800000U));
        }
        auto* target=reinterpret_cast<uint64_t*>(queries+(i/128)*stride+i%128);
        *target=uint64_t(bits[0]>>16)|(uint64_t(bits[1]>>16)<<16)|
                (uint64_t(bits[2]>>16)<<32)|(uint64_t(bits[3]>>16)<<48);
    }
    for(int r=warp;r<tile;r+=4){
        const int key=key_ids[r],dim=lane*4;
        uint64_t result=0;
        if(key>=0){
            const auto* p=packed+int64_t(key)*68;
            const unsigned exponent=p[64+dim/32];
            unsafe|=exponent<73||exponent>181;
            const unsigned codes=*reinterpret_cast<const uint16_t*>(p+dim/2);
#pragma unroll
            for(int z=0;z<4;++z){
                const unsigned code=(codes>>(4*z))&15,m=code&7;
                unsigned bits=((m/2+126)<<7)|((m&1)<<6);
                bits=m==1?0x3f00U:bits;
                bits=m==0?0U:unsigned(int(bits)+(int(exponent)-127)*128);
                result|=uint64_t((bits|((code&8)<<12))&0xffffU)<<(z*16);
            }
        }
        *reinterpret_cast<uint64_t*>(keys+r*stride+dim)=result;
    }
    if(__syncthreads_or(unsafe)){
        // Same dimension/shuffle tree and eight head groups as the scalar baseline.
        for(int r=warp;r<tile;r+=4){
            const int key=key_ids[r];
            if(key<0){if(!lane&&first+r<width)scores[int64_t(token)*width+first+r]=-INFINITY;continue;}
            const auto* p=packed+int64_t(key)*68;float kv[4];
#pragma unroll
            for(int z=0;z<4;++z){const int d=lane+32*z;kv[z]=ada_fp4((p[d/2]>>(d%2*4))&15)*__uint_as_float(uint32_t(p[64+z])<<23);}
            float sums[8]={};
            for(int h=0;h<32;++h){
                float dot=0;
#pragma unroll
                for(int z=0;z<4;++z)dot+=q[(int64_t(token)*32+h)*128+lane+32*z]*kv[z];
                for(int delta=16;delta;delta>>=1)dot+=__shfl_down_sync(0xffffffff,dot,delta);
                if(!lane)sums[h%8]+=fmaxf(dot,0.f)*weights[token*32+h]*(1.f/64.f);
            }
            if(!lane){float sum=0;for(int w=0;w<8;++w)sum+=sums[w];scores[int64_t(token)*width+first+r]=sum;}
        }
        return;
    }
    namespace wmma=nvcuda::wmma;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2];
    wmma::fill_fragment(acc[0],0.f);wmma::fill_fragment(acc[1],0.f);
#pragma unroll
    for(int k=0;k<128;k+=16){
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
        wmma::load_matrix_sync(a,keys+warp*16*stride+k,stride);
#pragma unroll
        for(int h=0;h<2;++h){
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
            wmma::load_matrix_sync(b,queries+h*16*stride+k,stride);
            wmma::mma_sync(acc[h],a,b,acc[h]);
        }
    }
    __syncthreads(); // Query storage becomes the FP32 score tile only after every warp has read it.
    wmma::store_matrix_sync(dots+warp*16,acc[0],64,wmma::mem_col_major);
    wmma::store_matrix_sync(dots+warp*16+16*64,acc[1],64,wmma::mem_col_major);
    __syncthreads();
    if(tid<tile&&first+tid<width){
        float total=0;
#pragma unroll
        for(int w=0;w<8;++w){
            float sum=0;
#pragma unroll
            for(int h=w;h<32;h+=8)sum+=fmaxf(dots[h*64+tid],0.f)*weights[token*32+h]*(1.f/64.f);
            total+=sum;
        }
        scores[int64_t(token)*width+first+tid]=key_ids[tid]>=0?total:-INFINITY;
    }
}
}
cudaError_t index_scores_bf16_gpu(const float* q,const float* w,const uint8_t* packed,
    const uint8_t* mask,const int32_t* blocks,float* scores,int tokens,int rows,int width,
    int64_t start,int ratio,cudaStream_t stream){
    if(!q||!w||!packed||!scores||tokens<1||tokens>65535||rows<1||start<0||
       (ratio!=1&&ratio!=2)||width!=(blocks?16384:rows))return cudaErrorInvalidValue;
    if((reinterpret_cast<uintptr_t>(q)&15)||(reinterpret_cast<uintptr_t>(packed)&1))
        return blocks?compact_index_scores_gpu(q,w,nullptr,packed,blocks,scores,tokens,rows,start,ratio,stream):
                      index_scores_gpu(q,w,nullptr,packed,mask,scores,tokens,rows,start,ratio,stream);
    scores_bf16<<<dim3((uint64_t(width)+63)/64,tokens),128,0,stream>>>(q,w,packed,mask,blocks,scores,rows,width,start,ratio);
    return cudaGetLastError();
}
}
