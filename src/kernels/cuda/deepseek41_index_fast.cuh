// Eight rows per CTA; small batches retain the prior kernel.
namespace index_vectorized {
__device__ uint8_t code4(float x){
    const float a=fminf(fabsf(x),6.f);
    const uint8_t v=a<=.25f?0:a<.75f?1:a<=1.25f?2:a<1.75f?3:a<=2.5f?4:a<3.5f?5:a<=5.f?6:7;
    return v|(signbit(x)?8:0);
}
__device__ float value4(uint8_t x){constexpr float a[8]={0,.5,1,1.5,2,3,4,6};return x&8?-a[x&7]:a[x&7];}
template<bool Fake> __global__ void quant_rows(const float* x,uint8_t* packed,float* fake,int rows){
    const int row=blockIdx.x*8+threadIdx.x/32,group=(threadIdx.x%32)/8,lane=threadIdx.x%8;
    if(row>=rows)return;
    const int at=row*128+group*32+lane*4;
    const float4 v=*reinterpret_cast<const float4*>(x+at);
    float m=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
    for(int d=4;d;d>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,d,8));
    uint32_t bits=__float_as_uint(fmaxf(m,6.f*0x1p-126f)/6.f);
    if(bits&0x7fffff)bits=(bits+0x800000)&0x7f800000;
    const float scale=__uint_as_float(bits);
    const uint8_t c0=code4(__fdiv_rn(v.x,scale)),c1=code4(__fdiv_rn(v.y,scale)),c2=code4(__fdiv_rn(v.z,scale)),c3=code4(__fdiv_rn(v.w,scale));
    if constexpr(Fake)*reinterpret_cast<float4*>(fake+at)=make_float4(value4(c0)*scale,value4(c1)*scale,value4(c2)*scale,value4(c3)*scale);
    else{
        *reinterpret_cast<uint16_t*>(packed+row*68+group*16+lane*2)=uint16_t(c0)|(uint16_t(c1)<<4)|(uint16_t(c2)<<8)|(uint16_t(c3)<<12);
        if(!lane)packed[row*68+64+group]=(bits>>23)&255;
    }
}
}
