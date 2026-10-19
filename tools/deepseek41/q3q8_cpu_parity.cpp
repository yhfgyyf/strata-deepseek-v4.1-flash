// Independent double-precision dot over GGML-decoded Q3_K rows and Q8_1 values.
#include "strata/kernels/cpu/deepseek41_down.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace c=strata::kernels::cpu;
int main(){try{
    ggml_cpu_init();constexpr int hidden=5120,width=2304,rows=64;
    double worst=0;
    for(int seed:{52,1052})for(int tokens:{1,3,8}){
        c::NativeFmt format;std::string error;
        if(!c::native_fmt(GGML_TYPE_Q3_K,GGML_TYPE_Q3_K,hidden,width,format,error)||!c::native_enable_q3_q8_1_down(format,error))throw std::runtime_error(error);
        format.bf16_io=true;std::vector<uint8_t> blob(format.bytes);std::vector<float> row(width),decoded(width);
        std::mt19937 random(seed);std::normal_distribution<float> normal(0,.15f);
        for(int r=0;r<rows;++r){for(float& v:row)v=normal(random);
            if(ggml_quantize_chunk(GGML_TYPE_Q3_K,row.data(),blob.data()+format.down_off+r*format.d_row,0,1,width,nullptr)!=format.d_row)throw std::runtime_error("pack");}
        std::vector<std::vector<uint8_t>> packed(tokens,std::vector<uint8_t>(format.h_bytes));
        std::vector<std::vector<float>> x(tokens,std::vector<float>(width)),got(tokens,std::vector<float>(hidden));
        const void* inputs[8];float* output[8];
        for(int t=0;t<tokens;++t){
            for(int i=0;i<width;++i)row[i]=std::sin((i+t*17)*.031f)*(t+1)*.1f;
            c::native_quant_h(format,row.data(),packed[t].data());inputs[t]=packed[t].data();output[t]=got[t].data();
            for(int b=0;b<width/32;++b){uint16_t scale;std::memcpy(&scale,packed[t].data()+b*36,2);
                const float d=ggml_fp16_to_fp32(scale);for(int j=0;j<32;++j)x[t][b*32+j]=d*reinterpret_cast<const int8_t*>(packed[t].data()+b*36+4)[j];}
        }
        c::native_down_rows(format,blob.data(),inputs,tokens,output,0,rows);
        double error2=0,norm=0;
        for(int r=0;r<rows;++r){
            ggml_get_type_traits(GGML_TYPE_Q3_K)->to_float(blob.data()+format.down_off+r*format.d_row,decoded.data(),width);
            for(int t=0;t<tokens;++t){double expected=0;for(int i=0;i<width;++i)expected+=double(decoded[i])*x[t][i];
                const double delta=got[t][r]-expected;error2+=delta*delta;norm+=expected*expected;}
        }
        const double relative=std::sqrt(error2/norm);if(!std::isfinite(relative)||relative>1e-5)throw std::runtime_error("CPU integer dot differs from independent FP64 oracle");
        worst=std::max(worst,relative);
    }
    std::cout<<"q3_q8_1 CPU parity passed; avx2="<<c::cpu_avx2_ok()<<" max_relative_l2="<<worst<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
