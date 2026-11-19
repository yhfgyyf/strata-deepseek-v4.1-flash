// Warp-row MXFP4 draft MoE. BF16 operands and boundaries; no additional scratch.
#pragma once
namespace draft_warp {
__device__ float bf16(float x){return __bfloat162float(__float2bfloat16_rn(x));}
__device__ float sigmoid(float x){return x>=0?1.f/(1.f+expf(-x)):expf(x)/(1.f+expf(x));}
__device__ uint16_t mx_bits(uint8_t scale,uint8_t q){
    const int a=q&7;if(!a)return 0;
    if(scale>1&&scale<253)return uint16_t((a==1?(scale-1)*128:scale*128+(a-2)*64)|((q&8)<<12));
    constexpr int v[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    const uint32_t bits=scale<2?0x00200000u<<scale:uint32_t(scale-1)<<23;
    return __bfloat16_as_ushort(__float2bfloat16_rn(__uint_as_float(bits)*v[q]));
}
__device__ float2 weight_pair(const uint8_t* p,bool shared,int64_t i){
    uint32_t bits;
    if(shared)bits=*reinterpret_cast<const uint32_t*>(p+2*i);
    else{
        p+=(i/32)*17;const int shift=(i%32)/16*4,offset=1+i%16;
        bits=uint32_t(mx_bits(p[0],(p[offset]>>shift)&15))|(uint32_t(mx_bits(p[0],(p[offset+1]>>shift)&15))<<16);
    }
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
}
__global__ void __launch_bounds__(128) draft_gu_warp(const uint8_t* experts,uint64_t projection,const uint8_t* sg,const uint8_t* su,
    const float* x,const int* ids,const float* routing,float* activation){
    const int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5),job=blockIdx.y,token=job/4,slot=job%4;
    const bool shared=slot==3;const int id=shared?0:ids[token*3+slot];
    if(id<0||id>=128){if(!lane)activation[int64_t(job)*2304+row]=NAN;return;}
    const uint8_t* gate=shared?sg:experts+uint64_t(id)*3*projection;const uint8_t* up=shared?su:gate+projection;
    float g=0,u=0;
    for(int col=lane*2;col<5120;col+=64){
        const float2 gv=weight_pair(gate,shared,int64_t(row)*5120+col),uv=weight_pair(up,shared,int64_t(row)*5120+col);
        const float2 v=*reinterpret_cast<const float2*>(x+int64_t(token)*5120+col);
        g=fmaf(gv.x,v.x,g);g=fmaf(gv.y,v.y,g);u=fmaf(uv.x,v.x,u);u=fmaf(uv.y,v.y,u);
    }
    for(int delta=16;delta;delta>>=1){g+=__shfl_down_sync(0xffffffff,g,delta);u+=__shfl_down_sync(0xffffffff,u,delta);}
    if(!lane){g=fminf(bf16(g),10.f);u=fminf(fmaxf(bf16(u),-10.f),10.f);activation[int64_t(job)*2304+row]=bf16(g*sigmoid(g)*u*(shared?1.f:routing[token*3+slot]));}
}
__global__ void __launch_bounds__(128) draft_down_warp(const uint8_t* experts,uint64_t projection,const uint8_t* sd,const int* ids,const float* act,float* out){
    const int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5),token=blockIdx.y;
    const int selected[3]={ids[token*3],ids[token*3+1],ids[token*3+2]};
    if(selected[0]<0||selected[0]>=128||selected[1]<0||selected[1]>=128||selected[2]<0||selected[2]>=128){if(!lane)out[int64_t(token)*5120+row]=NAN;return;}
    const uint8_t* w[4]={experts+(uint64_t(selected[0])*3+2)*projection,experts+(uint64_t(selected[1])*3+2)*projection,experts+(uint64_t(selected[2])*3+2)*projection,sd};
    float y[4]{};
    for(int col=lane*2;col<2304;col+=64){
        #pragma unroll
        for(int j=0;j<4;++j){
            const float2 q=weight_pair(w[j],j==3,int64_t(row)*2304+col),x=*reinterpret_cast<const float2*>(act+(int64_t(token)*4+j)*2304+col);
            y[j]=fmaf(q.x,x.x,y[j]);y[j]=fmaf(q.y,x.y,y[j]);
        }
    }
    for(int delta=16;delta;delta>>=1){
        #pragma unroll
        for(int j=0;j<4;++j)y[j]+=__shfl_down_sync(0xffffffff,y[j],delta);
    }
    if(!lane){
        int order[3]={0,1,2};for(int k=1;k<3;++k)for(int j=k;j&&selected[order[j]]<selected[order[j-1]];--j){const int v=order[j];order[j]=order[j-1];order[j-1]=v;}
        float result=0;for(int k=0;k<3;++k)result+=bf16(y[order[k]]);out[int64_t(token)*5120+row]=bf16(result+bf16(y[3]));
    }
}
}
// De-duplicated draft MoE: one block row-group per distinct routed expert (and one for the
// shared expert) reads its weights once for every (token,slot) job routed to it. Per-job
// accumulation order, reduction tree and BF16 boundaries equal draft_gu_warp/draft_down_warp.
#include <cuda_pipeline.h>
namespace draft_dedup {
using draft_warp::bf16;using draft_warp::sigmoid;using draft_warp::weight_pair;
constexpr int kMaxJobs=7;
__device__ __forceinline__ float2 mx_pair(const uint8_t* p,int64_t i){
    p+=(i/32)*17;const int shift=(i%32)/16*4,offset=1+i%16;const uint32_t scale=p[0];
    const uint32_t q0=(p[offset]>>shift)&15,q1=(p[offset+1]>>shift)&15;
    if(scale<2||scale>=253)return weight_pair(p-(i/32)*17,false,i);
    const float s=__uint_as_float((scale-1)<<23);
    const float a=__uint_as_float(__float_as_uint(float((0xC8643210u>>((q0&7)*4))&15)*s)|((q0&8)<<28));
    const float b=__uint_as_float(__float_as_uint(float((0xC8643210u>>((q1&7)*4))&15)*s)|((q1&8)<<28));
    return make_float2(a,b);
}
__device__ __forceinline__ float2 any_pair(const uint8_t* p,bool shared,int64_t i){
    if(shared){const uint32_t bits=*reinterpret_cast<const uint32_t*>(p+2*i);return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));}
    return mx_pair(p,i);
}
// Descriptor per blockIdx.y (kDesc ints): expert (-1 shared, -2 unused), count, jobs, tokens, weight bits.
constexpr int kDesc=80,kDescJobs=21;
struct Jobs{const int* d;__device__ int count()const{return d[1];}__device__ int expert()const{return d[0];}
    __device__ int job(int n)const{return d[2+n];}__device__ int token(int n)const{return d[2+kDescJobs+n];}
    __device__ float weight(int n)const{return __int_as_float(d[2+2*kDescJobs+n]);}};
// y=0 is the shared expert (heaviest, scheduled first), y>0 the (y-1)-th distinct valid id in ascending order.
__global__ void plan(const int* ids,const float* routing,int tokens,int* desc){
    if(threadIdx.x)return;
    int sorted[3*7];int n=0;
    for(int i=0;i<tokens*3;++i){const int id=ids[i];if(id<0||id>=128)continue;bool seen=false;for(int k=0;k<n;++k)seen|=sorted[k]==id;if(!seen)sorted[n++]=id;}
    for(int a=1;a<n;++a)for(int b=a;b&&sorted[b]<sorted[b-1];--b){const int v=sorted[b];sorted[b]=sorted[b-1];sorted[b-1]=v;}
    for(int y=0;y<=tokens*3;++y){
        int* d=desc+y*kDesc;int count=0;
        if(y==0){d[0]=-1;for(int t=0;t<tokens;++t){d[2+count]=t*4+3;d[2+kDescJobs+count]=t;d[2+2*kDescJobs+count]=__float_as_int(1.f);++count;}}
        else if(y<=n){const int e=sorted[y-1];d[0]=e;
            for(int t=0;t<tokens;++t)for(int k=0;k<3;++k)if(ids[t*3+k]==e){d[2+count]=t*4+k;d[2+kDescJobs+count]=t;d[2+2*kDescJobs+count]=__float_as_int(routing[t*3+k]);++count;}}
        else d[0]=-2;
        d[1]=count;
    }
}
__device__ void stage(uint8_t* dst,const uint8_t* src,int bytes,int lane){
    if(bytes%16==0)for(int i=lane;i<bytes/16;i+=32)__pipeline_memcpy_async(dst+i*16,src+i*16,16);
    else for(int i=lane;i<bytes/8;i+=32)__pipeline_memcpy_async(dst+i*8,src+i*8,8);
}
template<int N>
__device__ void gu_rows(const uint8_t* gate,const uint8_t* up,bool shared,int64_t base,const float* x,const Jobs& j,int first,
                        int row,int lane,float* activation){
    float g[N]{},u[N]{};const float* xs[N];
    #pragma unroll
    for(int n=0;n<N;++n)xs[n]=x+int64_t(j.token(first+n))*5120;
    #pragma unroll 4
    for(int col=lane*2;col<5120;col+=64){
        const float2 gv=any_pair(gate,shared,base+col),uv=any_pair(up,shared,base+col);
        #pragma unroll
        for(int n=0;n<N;++n){
            const float2 v=*reinterpret_cast<const float2*>(xs[n]+col);
            g[n]=fmaf(gv.x,v.x,g[n]);g[n]=fmaf(gv.y,v.y,g[n]);u[n]=fmaf(uv.x,v.x,u[n]);u[n]=fmaf(uv.y,v.y,u[n]);
        }
    }
    #pragma unroll
    for(int n=0;n<N;++n)for(int delta=16;delta;delta>>=1){g[n]+=__shfl_down_sync(0xffffffff,g[n],delta);u[n]+=__shfl_down_sync(0xffffffff,u[n],delta);}
    if(!lane){
        #pragma unroll
        for(int n=0;n<N;++n){const float gg=fminf(bf16(g[n]),10.f),uu=fminf(fmaxf(bf16(u[n]),-10.f),10.f);
            activation[int64_t(j.job(first+n))*2304+row]=bf16(gg*sigmoid(gg)*uu*(shared?1.f:j.weight(first+n)));}
    }
}
template<int N>
__device__ void down_rows(const uint8_t* w,bool shared,int64_t base,const float* act,const Jobs& j,int first,int row,int lane,float* parts){
    float y[N]{};const float* as[N];
    #pragma unroll
    for(int n=0;n<N;++n)as[n]=act+int64_t(j.job(first+n))*2304;
    #pragma unroll 4
    for(int col=lane*2;col<2304;col+=64){
        const float2 q=any_pair(w,shared,base+col);
        #pragma unroll
        for(int n=0;n<N;++n){const float2 v=*reinterpret_cast<const float2*>(as[n]+col);
            y[n]=fmaf(q.x,v.x,y[n]);y[n]=fmaf(q.y,v.y,y[n]);}
    }
    #pragma unroll
    for(int n=0;n<N;++n)for(int delta=16;delta;delta>>=1)y[n]+=__shfl_down_sync(0xffffffff,y[n],delta);
    if(!lane){
        #pragma unroll
        for(int n=0;n<N;++n)parts[int64_t(j.job(first+n))*5120+row]=y[n];
    }
}
#define DRAFT_DEDUP_DISPATCH(F,...) do{for(int first=0;first<j.count();first+=kMaxJobs){switch(min(kMaxJobs,j.count()-first)){ \
    case 1:F<1>(__VA_ARGS__);break;case 2:F<2>(__VA_ARGS__);break;case 3:F<3>(__VA_ARGS__);break;case 4:F<4>(__VA_ARGS__);break; \
    case 5:F<5>(__VA_ARGS__);break;case 6:F<6>(__VA_ARGS__);break;default:F<7>(__VA_ARGS__);break;}}}while(0)
__global__ void __launch_bounds__(128,8) gu(const uint8_t* experts,uint64_t projection,const uint8_t* sg,const uint8_t* su,
    const float* x,const int* ids,const int* desc,float* activation,int tokens){
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*4+warp;
    const Jobs j{desc+blockIdx.y*kDesc};if(j.expert()==-2)return;
    if(j.expert()<0){
        for(int t=0;t<tokens;++t)for(int k=0;k<3;++k){const int id=ids[t*3+k];if((id<0||id>=128)&&!lane)activation[int64_t(t*4+k)*2304+row]=NAN;}
        DRAFT_DEDUP_DISPATCH(gu_rows,sg,su,true,int64_t(row)*5120,x,j,first,row,lane,activation);
        return;
    }
    const uint8_t* gate=experts+uint64_t(j.expert())*3*projection+uint64_t(row)*2720;
    DRAFT_DEDUP_DISPATCH(gu_rows,gate,gate+projection,false,0,x,j,first,row,lane,activation);
}
__global__ void __launch_bounds__(128,8) down(const uint8_t* experts,uint64_t projection,const uint8_t* sd,const int* ids,
    const float* act,const int* desc,float* parts,int tokens){
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,row=blockIdx.x*4+warp;
    const Jobs j{desc+blockIdx.y*kDesc};if(j.expert()==-2)return;
    if(j.expert()<0){DRAFT_DEDUP_DISPATCH(down_rows,sd,true,int64_t(row)*2304,act,j,first,row,lane,parts);return;}
    DRAFT_DEDUP_DISPATCH(down_rows,experts+(uint64_t(j.expert())*3+2)*projection+uint64_t(row)*1224,false,0,act,j,first,row,lane,parts);
}
#undef DRAFT_DEDUP_DISPATCH
__global__ void combine(const float* parts,const int* ids,float* out,int tokens){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tokens*5120)return;
    const int token=i/5120,row=i%5120;
    const int selected[3]={ids[token*3],ids[token*3+1],ids[token*3+2]};
    if(selected[0]<0||selected[0]>=128||selected[1]<0||selected[1]>=128||selected[2]<0||selected[2]>=128){out[i]=NAN;return;}
    int order[3]={0,1,2};for(int k=1;k<3;++k)for(int j=k;j&&selected[order[j]]<selected[order[j-1]];--j){const int v=order[j];order[j]=order[j-1];order[j-1]=v;}
    float result=0;for(int k=0;k<3;++k)result+=bf16(parts[(int64_t(token)*4+order[k])*5120+row]);
    out[i]=bf16(result+bf16(parts[(int64_t(token)*4+3)*5120+row]));
}
}
