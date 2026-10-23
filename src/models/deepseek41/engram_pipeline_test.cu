#include "runtime_internal.hpp"
#include "ggml.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

namespace d=strata::models::deepseek41;
namespace {
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
__global__ void projection(uint16_t* weights){for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<25600;row+=blockDim.x*gridDim.x)weights[int64_t(row)*6144+row%6144]=0x3f80;}
__global__ void verify(const float* values,int count,int table,int generation,int* failures) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count*25600;i+=blockDim.x*gridDim.x) {
        const int t=i/25600,col=i%25600%6144,head=col/256;
        const int row=(t*7+(table*24+head)*11+generation)%64;
        const float expected=t==1?0:1+(table*5+row)%99;
        if(values[i]!=expected)atomicAdd(failures,1);
    }
}
void run(int type) {
    const auto path=std::filesystem::temp_directory_path()/("strata-engram-pipeline-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove(path,e);}} cleanup{path};
    const int row_bytes=type==13?176:type==14?210:272;
    const std::array<uint64_t,2> offsets{4096,32768+128};std::ofstream file(path,std::ios::binary);std::vector<uint8_t> raw(row_bytes);
    for(int table=0;table<2;++table)for(int row=0;row<64;++row) {
        const float value=1+(table*5+row)%99;
        std::vector<float> values(256,value),decoded(256);
        ggml_quantize_chunk(static_cast<ggml_type>(type),values.data(),raw.data(),0,1,256,nullptr);
        ggml_get_type_traits(static_cast<ggml_type>(type))->to_float(raw.data(),decoded.data(),256);
        for(float x:decoded)need(ggml_bf16_to_fp32(ggml_fp32_to_bf16(x))==value,"fixture BF16 value");
        file.seekp(offsets[table]+row*row_bytes);file.write(reinterpret_cast<const char*>(raw.data()),raw.size());
    }
    file.seekp(57343);file.put(0);file.close();need(bool(file),"Engram fixture");
    d::EngramReader reader(path.string(),offsets,{64,64},1<<20,4,false,false,row_bytes);
    d::Workspace work(512ull<<20,128,512);work.engram_input=d::allocate<uint16_t>(*work.arena,128*6144);
    strata::core::DeviceArena weights_arena(uint64_t(25600)*6144*2+256);
    auto* weights=d::allocate<uint16_t>(weights_arena,uint64_t(25600)*6144);auto* errors=d::allocate<int>(*work.arena,64);
    d::gpu_check(cudaMemset(weights,0,uint64_t(25600)*6144*2));d::gpu_check(cudaMemset(errors,0,256));
    projection<<<128,256>>>(weights);d::gpu_check(cudaDeviceSynchronize());
    d::Matrix matrix{weights,30,25600,6144,uint64_t(25600)*6144};d::Stream main;
    {
        d::EngramPipeline pipeline(work,d::Arithmetic::production,false,true,type);
        for(uint64_t sequence=1;sequence<=12;++sequence) {
            const int count=sequence%3==0?128:sequence%3==1?1:17;
            d::EngramQuery query;query.image.resize(count);if(count>1)query.image[1]=1;query.rows.resize(count*48);
            for(int t=0;t<count;++t)for(int h=0;h<48;++h)query.rows[t*48+h]=(t*7+h*11+sequence)%64;
            auto ticket=reader.submit(query,sequence,true);pipeline.start(ticket,sequence,count,matrix,matrix);
            for(int table=0;table<2;++table) {
                if(sequence%2)std::this_thread::sleep_for(std::chrono::microseconds(300));
                pipeline.wait(table,sequence,sequence,main.value);
                verify<<<128,256,0,main.value>>>(work.engram_kv,count,table,sequence,errors);d::gpu_check(cudaGetLastError());
                pipeline.consume(table,sequence,main.value);
            }
            d::gpu_check(cudaStreamSynchronize(main.value));
        }
    }
    int bad=0;d::gpu_check(cudaMemcpy(&bad,errors,4,cudaMemcpyDeviceToHost));need(!bad,"Engram projection/generation/source reuse corrupted");
    {
        d::EngramPipeline pipeline(work,d::Arithmetic::production,false,true,type);d::EngramQuery query;query.image={0};query.rows.resize(48);
        auto ticket=reader.submit(query,99,true);pipeline.start(ticket,1,1,matrix,matrix);pipeline.wait(0,99,1,main.value);
        // Producer may be waiting for consumed0. Cancellation must wake and join it.
        pipeline.stop();d::gpu_check(cudaStreamSynchronize(main.value));
    }
    {
        d::EngramPipeline pipeline(work,d::Arithmetic::production,false,true,type);d::EngramQuery query;query.image={0};query.rows.resize(48);query.rows[24]=64;
        auto ticket=reader.submit(query,100,true);pipeline.start(ticket,1,1,matrix,matrix);bool failed=false;
        try{pipeline.wait(0,100,1,main.value);}catch(const std::runtime_error&){failed=true;}
        pipeline.stop();need(failed,"Engram I/O failure did not wake consumer");
    }
    std::cout<<"ENGRAM_LAYER_PREPARE_GENERATION_WRAP_IMAGE_TAIL_CANCEL_FAILURE_PASS\n";
}
}
int main(){try{run(8);run(13);run(14);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
