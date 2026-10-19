#include "runtime_internal.hpp"
#include "strata/models/deepseek41/vision.hpp"
#include "strata/models/deepseek41/integrity.hpp"
#include <algorithm>
#include <cmath>

namespace strata::models::deepseek41 {
void VisionWork::init(WorkspaceLayout& layout) {
    auto f=[&](uint64_t count){return layout.alloc<float>(count);};
    constexpr uint64_t p=9198,c=1024;
    patches=f(p*588);h=f(p*1024);norm=f(p*1024);qkv=f(p*3072);
    q=f(p*1024);k=f(p*1024);v=f(p*1024);attention=f(p*1024);branch=f(p*1024);
    gate=f(p*2816);up=f(p*2816);activation=f(p*2816);aligned=f(c*9216);
    projected=f(c*5120);features=f(c*5120);span=f(c*5120);
}
std::vector<float> vision_forward(WeightBank& weights,VisionWork& w,const std::vector<float>& patches,
                                 int height,int width,Arithmetic arithmetic,cublasHandle_t blas,cudaStream_t stream,
                                 uint16_t* bf_input,int chunk,Capture* capture) {
    const int64_t p=int64_t(height)*width,oh=(height+2)/3,ow=(width+2)/3,cells=oh*ow,span=oh*(ow+1)+2;
    if(height<1||width<1||p>9198||span>1024||patches.size()!=static_cast<size_t>(p*588)||chunk<1)
        throw std::invalid_argument("DeepSeek vision patch/grid/token bound");
    if(!std::all_of(patches.begin(),patches.end(),[](float x){return std::isfinite(x)&&x>=-1.01f&&x<=1.01f;}))
        throw std::invalid_argument("DeepSeek vision normalized pixels");
    const bool bf=arithmetic==Arithmetic::production;
    gpu_check(cudaMemcpyAsync(w.patches,patches.data(),patches.size()*4,cudaMemcpyHostToDevice,stream));
    if(bf)gpu_check(round_bf16_gpu(w.patches,patches.size(),stream));
    auto dump=[&](const std::string& name,const float* data,std::vector<int64_t> shape){if(capture)capture->put(name,data,shape,stream);};
    auto linear=[&](const Matrix& m,const float* x,float* output,int tokens,const float* bias=nullptr){
        for(int begin=0;begin<tokens;begin+=chunk){
            const int count=std::min(chunk,tokens-begin);
            matrix_product(m,x+int64_t(begin)*m.cols,output+int64_t(begin)*m.rows,count,blas,stream,bf_input,arithmetic,0,0,0,0,false);
            gpu_check(vision_bias_gpu(output+int64_t(begin)*m.rows,bias,count,m.rows,bf,stream));
        }
    };
    auto bias=[&](const std::string& name){return static_cast<const float*>(weights.at(name).data);};
    Matrix patch=weights.at("v.patch_embd.weight");patch.cols=588;patch.rows=1024;
    linear(patch,w.patches,w.h,p,bias("v.patch_embd.bias"));dump("vision.patch_embed",w.h,{p,1024});
    for(int layer=0;layer<32;++layer){
        const std::string prefix="v.blk."+std::to_string(layer)+".";
        gpu_check(rms_gpu(w.h,bias(prefix+"ln1.weight"),w.norm,1024,p,1e-6f,stream));if(bf)gpu_check(round_bf16_gpu(w.norm,p*1024,stream));
        linear(weights.at(prefix+"attn_qkv.weight"),w.norm,w.qkv,p,bias(prefix+"attn_qkv.bias"));
        gpu_check(vision_qkv_gpu(w.qkv,w.q,w.k,w.v,height,width,bf,stream));
        dump(prefix+"q",w.q,{p,16,64});dump(prefix+"k",w.k,{p,16,64});
        gpu_check(vision_attention_gpu(w.q,w.k,w.v,w.attention,p,bf,stream));
        linear(weights.at(prefix+"attn_out.weight"),w.attention,w.branch,p,bias(prefix+"attn_out.bias"));
        gpu_check(vision_add_gpu(w.h,w.branch,p*1024,bf,stream));
        dump(prefix+"after_attention",w.h,{p,1024});
        gpu_check(rms_gpu(w.h,bias(prefix+"ln2.weight"),w.norm,1024,p,1e-6f,stream));if(bf)gpu_check(round_bf16_gpu(w.norm,p*1024,stream));
        linear(weights.at(prefix+"ffn_gate.weight"),w.norm,w.gate,p);
        linear(weights.at(prefix+"ffn_up.weight"),w.norm,w.up,p);
        gpu_check(vision_activate_gpu(w.gate,w.up,w.activation,p*2816,false,bf,stream));
        linear(weights.at(prefix+"ffn_down.weight"),w.activation,w.branch,p);
        gpu_check(vision_add_gpu(w.h,w.branch,p*1024,bf,stream));dump(prefix+"output",w.h,{p,1024});
    }
    gpu_check(rms_gpu(w.h,bias("v.post_ln.weight"),w.norm,1024,p,1e-6f,stream));if(bf)gpu_check(round_bf16_gpu(w.norm,p*1024,stream));
    dump("vision.features",w.norm,{p,1024});
    gpu_check(vision_unshuffle_gpu(w.norm,w.aligned,height,width,stream));dump("vision.unshuffle",w.aligned,{cells,9216});
    linear(weights.at("mm.1.weight"),w.aligned,w.projected,cells,bias("mm.1.bias"));
    gpu_check(vision_activate_gpu(w.projected,nullptr,w.projected,cells*5120,true,bf,stream));
    linear(weights.at("mm.2.weight"),w.projected,w.features,cells,bias("mm.2.bias"));dump("vision.aligner",w.features,{cells,5120});
    gpu_check(vision_span_gpu(w.features,bias("v.token_embd.img_start"),bias("v.token_embd.img_end"),bias("v.image_newline"),w.span,oh,ow,stream));
    dump("vision.span",w.span,{span,5120});
    std::vector<float> output(span*5120);gpu_check(cudaMemcpyAsync(output.data(),w.span,output.size()*4,cudaMemcpyDeviceToHost,stream));gpu_check(cudaStreamSynchronize(stream));return output;
}
std::vector<float> encode_vision(const std::string& mmproj,const std::vector<float>& patches,int h,int w,Arithmetic arithmetic,std::string capture_path,std::string integrity) {
    require_integrity(integrity,"vision",mmproj);
    auto model=Model::vision_only(mmproj);Stream stream;cublasHandle_t blas=nullptr;
    if(cublasCreate(&blas)!=CUBLAS_STATUS_SUCCESS||cublasSetStream(blas,stream.value)!=CUBLAS_STATUS_SUCCESS||cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH)!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("DeepSeek standalone vision cuBLAS");
    struct Cleanup {cublasHandle_t value;~Cleanup(){cublasDestroy(value);}} cleanup{blas};
    WeightBank weights(*model,arithmetic,stream.value,false,true);core::DeviceArena arena(2ull<<30);
    auto* bf=allocate<uint16_t>(arena,1024*32768);VisionWork work(arena);Capture capture(capture_path);
    return vision_forward(weights,work,patches,h,w,arithmetic,blas,stream.value,bf,1024,&capture);
}
} // namespace strata::models::deepseek41
