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
