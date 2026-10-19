#include "runtime_internal.hpp"
#include "strata/models/deepseek41/integrity.hpp"
#include <iomanip>

// This executable never opens a main GGUF. It exercises the production draft
// forward pass on explicit chronological synthetic hidden captures.
int main(int argc,char** argv) {
    namespace d=strata::models::deepseek41;
    try {
        std::string draft,hidden,capture,integrity,output,vocab,input_vocab,expected_sha;
        int window=128,first=417;int64_t position=131;
        auto arithmetic=d::Arithmetic::f32;
        strata::kernels::SamplerParams sampling{};sampling.greedy=1;sampling.temperature=0;
        for(int i=1;i<argc;++i){const std::string option=argv[i];auto next=[&]{if(i+1==argc)throw std::invalid_argument("draft option value");return std::string(argv[++i]);};
            if(option=="--draft")draft=next();else if(option=="--hidden")hidden=next();else if(option=="--capture")capture=next();
            else if(option=="--integrity")integrity=next();else if(option=="--output")output=next();
            else if(option=="--window")window=std::stoi(next());else if(option=="--position")position=std::stoll(next());
            else if(option=="--first")first=std::stoi(next());
            else if(option=="--draft-sha256")expected_sha=next();
            else if(option=="--draft-vocab")vocab=next();
            else if(option=="--draft-input-vocab")input_vocab=next();
            else if(option=="--arithmetic"){auto a=next();if(a!="f32"&&a!="production")throw std::invalid_argument("draft arithmetic");arithmetic=a=="f32"?d::Arithmetic::f32:d::Arithmetic::production;}
            else throw std::invalid_argument("unknown draft option");
        }
        if(draft.empty()||hidden.empty()||capture.empty()||output.empty()||window<1||window>128||position<window||first<0||first>=129280)
            throw std::invalid_argument("draft input/position/output");
        d::require_integrity(integrity,"draft",draft,expected_sha);
        std::vector<float> values(int64_t(window)*15360);std::ifstream in(hidden,std::ios::binary);
        in.read(reinterpret_cast<char*>(values.data()),values.size()*4);if(!in||in.peek()!=EOF)throw std::runtime_error("draft hidden file size");
        size_t free=0,total=0;d::gpu_check(cudaMemGetInfo(&free,&total));
        if(free<(16ull<<30))throw std::runtime_error("full-resident draft fixture requires 16 GiB free GPU memory");
        auto model=d::Model::draft_only(draft);d::Stream stream;
        cublasHandle_t blas=nullptr;
        if(cublasCreate(&blas)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("draft cuBLAS create");
        struct BlasGuard{cublasHandle_t handle;~BlasGuard(){cublasDestroy(handle);}} guard{blas};
        if(cublasSetStream(blas,stream.value)!=CUBLAS_STATUS_SUCCESS||cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH)!=CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("draft cuBLAS setup");
        d::DraftWeights weights(*model,arithmetic,stream.value,vocab,input_vocab);
        if(weights.bytes()>(11ull<<30))throw std::runtime_error("draft weights exceed 11 GiB");
        d::Workspace work(512ull<<20,128,128);d::DraftWork draft_work(*work.arena,weights.vocab.size());d::Capture tensors(capture);
        d::gpu_check(cudaMemcpyAsync(draft_work.main_hidden,values.data(),values.size()*4,cudaMemcpyHostToDevice,stream.value));
        const std::vector<int> history{128000,417,9863,1151};tensors.tokens(history);
        d::DraftContext context{*model,arithmetic,weights,draft_work,work,blas,stream.value,&tensors,{}};
        const auto proposal=d::draft_forward(context,draft_work.main_hidden,window,position,first,history,sampling);
        std::ofstream out(output);out<<std::setprecision(9)<<"{\"mode\":\"synthetic chronological main hidden\",\"arithmetic\":\""
            <<(arithmetic==d::Arithmetic::f32?"f32":"production")<<"\",\"cache\":\""<<d::draft_cache_label(arithmetic)<<"\",\"window\":"
            <<window<<",\"position\":"<<position<<",\"first\":"<<first<<",\"weights_bytes\":"<<weights.bytes()<<",\"workspace_bytes\":"
            <<work.arena->used()<<",\"workspace_reserved\":"<<work.arena->capacity()<<",\"vocab_size\":"<<weights.vocab.size()
            <<",\"host_expert_bytes\":"<<0<<",\"expert_h2d_bytes\":"<<0<<",\"tokens\":[";
        for(size_t i=0;i<proposal.tokens.size();++i){if(i)out<<',';out<<proposal.tokens[i];}
        out<<"],\"confidence\":[";for(size_t i=0;i<proposal.confidence.size();++i){if(i)out<<',';out<<proposal.confidence[i];}out<<"]}\n";
        if(!out)throw std::runtime_error("draft report write");
        std::cout<<"draft_tokens="<<proposal.tokens.size()<<" weights_bytes="<<weights.bytes()<<" workspace_bytes="<<work.arena->used()<<'\n';
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
