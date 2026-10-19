// DeepSeek-V4.1 index and candidate semantics follow the pinned official model (MIT).
#include "strata/models/deepseek41/indexer.hpp"
#include <cub/device/device_segmented_radix_sort.cuh>
#include <climits>

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
