// Calibration only: static residency, independent sessions, no throughput claim.
#include "strata/models/deepseek41/runtime.hpp"
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace d=strata::models::deepseek41;
namespace {void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}}
int main(int argc,char** argv){try {
    need(argc==6,"MODEL PROOF PROMPTS PROFILE NEW.jsonl");
    std::ifstream input(argv[3]);need(bool(input),"calibration inputs");
    std::ifstream prior(argv[5]);need(!prior,"calibration output already exists");
    d::RuntimeOptions o;o.integrity_file=argv[2];o.profile=argv[4];o.context=1048576;o.chunk=4096;o.slots=1;
    o.hot_front=74662871040ull;o.hot_back=0;o.work=3328ull<<20;o.state=1024ull<<20;o.gpu_headroom=700ull<<20;
    o.compact_logits=true;o.expert_bf16=true;o.expert_direct_gemv=true;o.engram_pipeline=true;
    o.compact_workspace=true;
    o.adaptive_experts=false;o.dspark=false;
    const bool eight_k=std::getenv("DS41_CALIB_EIGHT_K")!=nullptr;
    if(eight_k){
        o.chunk=8192;o.work=5888ull<<20;o.hot_front=80603840512ull;
        o.expert_pipeline=true;o.pcie_frac=1;o.dense_q8=true;o.dense_q8_1=true;o.expert_gu_mode=2;
        o.decode_graph=true;o.parallel_router=true;o.engram_continuous=true;o.engram_aio=true;
    }
    std::ofstream out(argv[5]);need(bool(out),"calibration report");out<<std::setprecision(17);
    d::Runtime runtime(argv[1],o);out<<"{\"kind\":\"configuration\",\"calibration_only\":true,\"adaptive\":false,\"cache_reused_between_samples\":true,\"context\":1048576,\"chunk\":"<<o.chunk<<",\"draft_loaded\":false,\"maximum\":128}\n";
    out<<"{\"kind\":\"ledger\",\"value\":";runtime.write_ledger(out);out<<"}\n";out.flush();
    strata::kernels::SamplerParams sampling;sampling.temperature=0;sampling.greedy=true;sampling.top_k=1;sampling.top_p=1;sampling.seed=1234;
    std::string line;int samples=0;
    while(std::getline(input,line)) {
        if(line.empty())continue;std::istringstream row(line);std::string id;int count=0;need(bool(row>>id>>count) && count==(eight_k?8192:4096) && id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos,"calibration record");
        std::vector<int> ids(count);for(auto& t:ids)need(bool(row>>t) && t>=0 && t<129280,"calibration token");row>>std::ws;need(row.eof(),"extra calibration fields");
        auto session=runtime.create_session();auto logits=session->prefill(ids,true);
        const auto result=session->generate(logits,128,{1},sampling,false);const auto st=session->stats();
        need(!result.empty() && result.size()<=128 && st.decoded_tokens==result.size()-1,"calibration physical rows");
        need(st.expert_cpu+st.expert_gpu==6*(st.encoder_token_layers+st.decoder_token_layers),"calibration expert coverage");
        if(eight_k)need(st.expert_cpu==0,"calibration all-GPU expert path");
        out<<"{\"kind\":\"sample\",\"id\":\""<<id<<"\",\"emitted_tokens\":"<<result.size()<<",\"decoded_tokens\":"<<st.decoded_tokens<<"}\n";
        out<<"{\"kind\":\"heat\",\"sample\":\""<<id<<"\",\"value\":";runtime.write_expert_heat(out);out<<"}\n";out.flush();need(bool(out),"calibration write");
        std::cerr<<"calibration completed "<<++samples<<'\n';
    }
    need(input.eof() && samples>0,"calibration end");out<<"{\"kind\":\"completed\",\"samples\":"<<samples<<"}\n";out.flush();need(bool(out),"calibration flush");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
