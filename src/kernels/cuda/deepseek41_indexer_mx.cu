// MX64 device body from the validated 2026-10-19 isolated candidate.
// SM120 helper provenance and MIT notice are in deepseek41_mx_support.cuh.
#include "strata/models/deepseek41/indexer.hpp"
#include "deepseek41_mx_support.cuh"
#include <atomic>
#include <cmath>
namespace strata::models::deepseek41 {
namespace {
std::atomic<bool> scores_enabled{true},dequant_enabled{true};
IndexScoreAudit score_audit = nullptr; // Set only while the single-slot runtime is idle.
__device__ __forceinline__ float decode4(uint8_t c){
 constexpr float v[8]={0,.5f,1,1.5f,2,3,4,6};return c&8?-v[c&7]:v[c&7];
}
__device__ __forceinline__ uint8_t encode4(float x){
 const float a=fminf(fabsf(x),6.f);
 const uint8_t c=a<=.25f?0:a<.75f?1:a<=1.25f?2:a<1.75f?3:a<=2.5f?4:a<3.5f?5:a<=5.f?6:7;
 return c|(signbit(x)?8:0);
}
__global__ void prepare_queries(const float* q,uint8_t* payload,uint32_t* sf,uint8_t* unsafe){
 const int row=blockIdx.x,d=threadIdx.x,lane=d&31,group=d>>5;
 const float x=q[int64_t(row)*128+d];
 float maximum=fabsf(x);
 for(int delta=16;delta;delta>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,delta));
 maximum=__shfl_sync(0xffffffff,maximum,0);
 uint32_t bits=__float_as_uint(fmaxf(maximum,6.f*0x1p-126f)/6.f);
 if(bits&0x7fffff)bits=(bits+0x800000)&0x7f800000;
 const float scale=__uint_as_float(bits);
 const uint8_t code=encode4(__fdiv_rn(x,scale));
 const uint8_t peer=__shfl_xor_sync(0xffffffff,code,1);
 if(!(lane&1))payload[int64_t(row)*64+d/2]=code|(peer<<4);
 const float restored=decode4(code)*scale;
 const bool invalid=!isfinite(x)||restored!=x||(x!=0.f&&fabsf(x)<0x1p-126f);
 const bool bad=__any_sync(0xffffffff,invalid);
 if(!lane){
  reinterpret_cast<uint8_t*>(sf)[int64_t(row)*4+group]=(bits>>23)&255;
  unsafe[int64_t(row)*4+group]=bad;
 }
}

template<int TK>
__global__ void scores_mxfp4(const float* raw_q,const uint8_t* q,const uint32_t* sf_q,
 const uint8_t* unsafe_q,const float* weights,const uint8_t* packed,const uint8_t* candidates,
 const int32_t* blocks,float* scores,int rows,int width,int64_t start,int ratio){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
 using namespace deep_gemm;
 const int token=blockIdx.y,first=blockIdx.x*TK,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 const int64_t visible=(start+token+1)/ratio;
 __shared__ __align__(512) char sm_q[32*64],sm_k[TK*64];
 __shared__ uint32_t qs[32],ks[TK];
 __shared__ float dots[32*TK];
 __shared__ int key_ids[TK];
 if(tid<TK){
  const int c=first+tid;
  const int key=c<width?(blocks?blocks[int64_t(token)*2048+c/8]*8+c%8:c):-1;
  key_ids[tid]=key>=0&&key<rows&&key<visible&&(!candidates||candidates[int64_t(token)*rows+key])?key:-1;
 }
 const bool any_valid=__syncthreads_or(tid<TK&&key_ids[tid]>=0);
 if(!any_valid){
  if(tid<TK&&first+tid<width)scores[int64_t(token)*width+first+tid]=-INFINITY;
  return;
 }
 for(int i=tid;i<32*64/16;i+=blockDim.x){
  const uint4 value=*reinterpret_cast<const uint4*>(q+int64_t(token)*32*64+i*16);
  *reinterpret_cast<uint4*>(sm_q+sm120::CuTeSwizzle<64>::apply(i*16))=value;
 }
 if(tid<32)qs[tid]=sf_q[token*32+tid];
 __syncthreads();
 bool unsafe=tid<128&&unsafe_q[token*128+tid];
 for(int i=tid;i<TK*64/4;i+=blockDim.x){
  const int j=i/16,chunk=i%16,key=key_ids[j];
  const uint32_t value=key>=0?*reinterpret_cast<const uint32_t*>(packed+int64_t(key)*68+chunk*4):0;
  *reinterpret_cast<uint32_t*>(sm_k+sm120::CuTeSwizzle<64>::apply(i*4))=value;
  if(!chunk){
   const uint32_t sf=key>=0?*reinterpret_cast<const uint32_t*>(packed+int64_t(key)*68+64):0x7f7f7f7f;
   ks[j]=sf;
   // UE8M0 exponent 0 and FP32 exponent 0 have different meanings. Also
   // retain scalar behavior at subnormal/nonfinite boundaries.
   #pragma unroll
   for(int g=0;g<4;++g){const unsigned e=(sf>>(8*g))&255;unsafe|=e<=1||e>=253;}
  }
 }
 const bool fallback=__syncthreads_or(unsafe);
 if(fallback){
  // One warp/key, same four dimension chunks and shuffle tree as baseline.
  // Head partials preserve the original eight-warp accumulation order.
  for(int j=warp;j<TK;j+=blockDim.x/32){
   const int key=key_ids[j];
   if(key<0){if(!lane&&first+j<width)scores[int64_t(token)*width+first+j]=-INFINITY;continue;}
   const uint8_t* p=packed+int64_t(key)*68;float kval[4];
   #pragma unroll
   for(int z=0;z<4;++z){const int d=lane+32*z;kval[z]=decode4((p[d/2]>>(d%2*4))&15)*__uint_as_float(uint32_t(p[64+z])<<23);}
   float partial[8]={};
   for(int h=0;h<32;++h){
    float dot=0;
    #pragma unroll
    for(int z=0;z<4;++z)dot+=raw_q[(int64_t(token)*32+h)*128+lane+32*z]*kval[z];
    for(int delta=16;delta;delta>>=1)dot+=__shfl_down_sync(0xffffffff,dot,delta);
    if(!lane)partial[h%8]+=fmaxf(dot,0.f)*weights[token*32+h]*(1.f/64.f);
   }
   if(!lane){float total=0;for(int i=0;i<8;++i)total+=partial[i];scores[int64_t(token)*width+first+j]=total;}
  }
  return;
 }
 const int g=lane/4,t=lane%4;
 sm120::SwizzleContext<64> a_ctx;
 a_ctx.init(warp*16+(lane&7)+((lane>>3)&1)*8,64);
 const uint32_t sf_a=ks[warp*16+g+(t&1)*8];
 uint32_t a[2][4];
 #pragma unroll
 for(int z=0;z<2;++z)sm120::load_a_fragment(a[z],sm_k,a_ctx,lane,z,32);
 #pragma unroll
 for(int nt=0;nt<4;++nt){
  sm120::SwizzleContext<64> b_ctx;b_ctx.init(nt*8+(lane&7),64);
  const uint32_t sf_b=qs[nt*8+g];float d[4]={};
  #pragma unroll
  for(int z=0;z<2;++z){
   uint32_t b[2];sm120::load_b_fragment_x2(b,sm_q,b_ctx,lane,z,32);
   mma::sm120::fp4_mma_block_scaled(d,a[z],b,mma::sm120::extract_sf_pair(sf_a,2*z),mma::sm120::extract_sf_pair(sf_b,2*z));
  }
  const int key=warp*16+g,h=nt*8+t*2;
  dots[h*TK+key]=d[0];dots[(h+1)*TK+key]=d[1];
  dots[h*TK+key+8]=d[2];dots[(h+1)*TK+key+8]=d[3];
 }
 __syncthreads();
 if(tid<TK&&first+tid<width){
  float total=0;
  #pragma unroll
  for(int w=0;w<8;++w){
   float sum=0;
   #pragma unroll
   for(int h=w;h<32;h+=8)sum+=fmaxf(dots[h*TK+tid],0.f)*weights[token*32+h]*(1.f/64.f);
   total+=sum;
  }
  scores[int64_t(token)*width+first+tid]=key_ids[tid]>=0?total:-INFINITY;
 }
#endif
}


} // namespace
void scores_fast_set_enabled(bool enabled){scores_enabled.store(enabled,std::memory_order_relaxed);}
bool scores_fast_enabled(){return scores_enabled.load(std::memory_order_relaxed);}
void expert_dequant_fast_set_enabled(bool enabled){dequant_enabled.store(enabled,std::memory_order_relaxed);}
bool expert_dequant_fast_enabled(){return dequant_enabled.load(std::memory_order_relaxed);}
void index_score_set_audit(IndexScoreAudit audit){score_audit=audit;}
cudaError_t index_scores_mxfp4_gpu(const float* q,const float* w,const uint8_t* packed,const uint8_t* mask,
 const int32_t* blocks,float* scores,uint8_t* scratch,size_t scratch_bytes,int tokens,int rows,int width,
 int64_t start,int ratio,cudaStream_t stream){
 if(!q||!w||!packed||!scores||!scratch||tokens<1||tokens>65535||rows<1||start<0||
    (ratio!=1&&ratio!=2)||width!=(blocks?16384:rows)||scratch_bytes<size_t(tokens)*2304)
  return cudaErrorInvalidValue;
 auto* payload=scratch;
 auto* scales=reinterpret_cast<uint32_t*>(scratch+size_t(tokens)*2048);
 auto* unsafe=scratch+size_t(tokens)*2176;
 prepare_queries<<<tokens*32,128,0,stream>>>(q,payload,scales,unsafe);
 auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
 scores_mxfp4<64><<<dim3((width+63)/64,tokens),128,0,stream>>>(q,payload,scales,unsafe,w,packed,mask,blocks,scores,rows,width,start,ratio);
 status=cudaGetLastError();
 if(status==cudaSuccess&&score_audit)score_audit(q,w,packed,mask,blocks,scores,tokens,rows,width,start,ratio,stream);
 return status;
}
} // namespace strata::models::deepseek41
