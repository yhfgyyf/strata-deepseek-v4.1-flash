#include "strata/kernels/cpu/deepseek41_down.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels::cpu {
namespace {
std::atomic<bool> enabled{false};
float half(const uint8_t* p){uint16_t value;std::memcpy(&value,p,2);return ggml_fp16_to_fp32(value);}
int scale(const uint8_t* p,int group){
    return (((p[96+group%8]>>(group/8*4))&15)|(((p[104+group%4]>>(group/4*2))&3)<<4))-32;
}
int code(const uint8_t* p,int index){
    return ((p[32+(index/128)*32+index%32]>>((index%128)/32*2))&3)-((p[index%32]&(1<<(index/32)))?0:4);
}
void scalar(const uint8_t* row,int blocks,const void* const* activations,int tokens,float* values){
    for(int b=0;b<blocks;++b){
        const uint8_t* weight=row+b*110;
        for(int g=0;g<16;++g){
            const float dw=half(weight+108)*scale(weight,g);
            for(int t=0;t<tokens;++t){
                const uint8_t* q=static_cast<const uint8_t*>(activations[t])+(b*8+g/2)*36;
                int dot=0;for(int j=0;j<16;++j)dot+=code(weight,g*16+j)*reinterpret_cast<const int8_t*>(q+4)[g%2*16+j];
                values[t]+=dw*(half(q)*dot);
            }
        }
    }
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
void avx2(const uint8_t* row,int blocks,const void* const* activations,int tokens,float* values){
    const __m128i low_mask=_mm_set1_epi8(3),four=_mm_set1_epi8(4),ones=_mm_set1_epi16(1),zero=_mm_setzero_si128();
    for(int b=0;b<blocks;++b){
        const uint8_t* weight=row+b*110;
        for(int g=0;g<16;++g){
            const __m128i low=_mm_and_si128(_mm_srl_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(weight+32+g/8*32+g%2*16)),_mm_cvtsi32_si128(g%8/2*2)),low_mask);
            const __m128i high=_mm_loadu_si128(reinterpret_cast<const __m128i*>(weight+g%2*16));
            const __m128i absent=_mm_cmpeq_epi8(_mm_and_si128(high,_mm_set1_epi8(1<<(g/2))),zero);
            const __m128i codes=_mm_sub_epi8(low,_mm_and_si128(absent,four));
            const __m128i magnitude=_mm_abs_epi8(codes);
            const float dw=half(weight+108)*scale(weight,g);
            for(int t=0;t<tokens;++t){
                const uint8_t* q=static_cast<const uint8_t*>(activations[t])+(b*8+g/2)*36;
                const __m128i x=_mm_loadu_si128(reinterpret_cast<const __m128i*>(q+4+g%2*16));
                __m128i dots=_mm_madd_epi16(_mm_maddubs_epi16(magnitude,_mm_sign_epi8(x,codes)),ones);
                dots=_mm_add_epi32(dots,_mm_srli_si128(dots,8));dots=_mm_add_epi32(dots,_mm_srli_si128(dots,4));
                values[t]+=dw*(half(q)*_mm_cvtsi128_si32(dots));
            }
        }
    }
}
#endif
}
void deepseek41_down_fusion_set_enabled(bool value){enabled.store(value,std::memory_order_relaxed);}
bool deepseek41_down_fusion_enabled(){return enabled.load(std::memory_order_relaxed);}
bool native_enable_q3_q8_1_down(NativeFmt& f,std::string& error){
    if(f.d_type!=GGML_TYPE_Q3_K||f.n_ff<1||f.n_ff%256){error="Q8_1 down requires Q3_K whole blocks";return false;}
    f.bf16_down=false;f.d_act=GGML_TYPE_Q8_1;f.h_bytes=ggml_row_size(GGML_TYPE_Q8_1,f.n_ff);return true;
}
void q3_q8_1_quantize(const float* values,void* output,int elements){
    if(!values||!output||elements<1||elements%32)throw std::invalid_argument("Q8_1 activation geometry");
    auto* bytes=static_cast<uint8_t*>(output);
    for(int b=0;b<elements/32;++b){
        float maximum=0,sum=0;for(int j=0;j<32;++j){maximum=std::max(maximum,std::abs(values[b*32+j]));sum+=values[b*32+j];}
        const float d=maximum*(1.f/127.f);const uint16_t scale=ggml_fp32_to_fp16(d),total=ggml_fp32_to_fp16(std::clamp(sum,-65504.f,65504.f));
        std::memcpy(bytes+b*36,&scale,2);std::memcpy(bytes+b*36+2,&total,2);
        for(int j=0;j<32;++j)reinterpret_cast<int8_t*>(bytes+b*36+4)[j]=maximum==0?0:static_cast<int8_t>(std::clamp(std::round(values[b*32+j]/d),-127.f,127.f));
    }
}
void q3_q8_1_down_rows(const NativeFmt& f,const uint8_t* blob,const void* const* activations,int tokens,float* const* output,int first,int end){
    if(f.d_type!=GGML_TYPE_Q3_K||f.d_act!=GGML_TYPE_Q8_1||tokens<1||tokens>8||first<0||end>f.n_embd||first>end)
        throw std::invalid_argument("Q3_K/Q8_1 down geometry");
    for(int r=first;r<end;++r){
        float values[8]={};const uint8_t* row=blob+f.down_off+size_t(r)*f.d_row;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        if(cpu_avx2_ok())avx2(row,f.n_ff/256,activations,tokens,values);else
#endif
        scalar(row,f.n_ff/256,activations,tokens,values);
        for(int t=0;t<tokens;++t)output[t][r]=values[t];
    }
}
}
