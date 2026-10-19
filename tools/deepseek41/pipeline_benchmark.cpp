// Native, private benchmark. One process owns exactly one fresh Runtime/request.
#include "strata/models/deepseek41/runtime.hpp"
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace d=strata::models::deepseek41;
namespace {
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void memory(std::ostream& out,const char* stage){
    size_t free=0,total=0;need(cudaDeviceSynchronize()==cudaSuccess && cudaMemGetInfo(&free,&total)==cudaSuccess,"GPU memory");
    out<<"{\"kind\":\"memory\",\"stage\":\""<<stage<<"\",\"free_bytes\":"<<free<<",\"total_bytes\":"<<total<<"}\n";out.flush();
}
void logits_file(std::ofstream& out,const std::vector<float>& logits){
    need(logits.size()==129280,"logits shape");out.write(reinterpret_cast<const char*>(logits.data()),logits.size()*4);need(bool(out),"logits write");
}
}
int main(int argc,char** argv){try {
    need(argc==9,"MODEL PROOF PROMPTS PROFILE NEW.jsonl FEATURES ADAPTIVE MODE(free/force:FILE/prefill/decode)");
    need(!std::filesystem::exists(argv[5]),"output already exists");
    std::ifstream input(argv[3]);std::string id;int n=0;need(bool(input>>id>>n) && n>0 && n<=8192,"prompt header");
    std::vector<int> prompt(n);for(auto& token:prompt)need(bool(input>>token) && token>=0 && token<129280,"input token");
    input>>std::ws;need(input.eof(),"extra prompt records");
    const unsigned features=std::stoul(argv[6]);std::string mode=argv[8],force_path;
    if(mode.rfind("prefill-force:",0)==0){force_path=mode.substr(14);mode="prefill";}
    else if(mode.rfind("decode-force:",0)==0){force_path=mode.substr(13);mode="decode";}
    else if(mode.rfind("force:",0)==0)force_path=mode.substr(6);
    const bool forced=!force_path.empty();
    std::vector<int> trajectory;
    if(forced){std::ifstream f(force_path);int token;while(f>>token){need(token>=0 && token<129280,"forced token");trajectory.push_back(token);}need(f.eof() && trajectory.size()==128,"forced trajectory length");}
    else need(mode=="free" || mode=="prefill" || mode=="decode","benchmark mode");
    d::RuntimeOptions options;options.integrity_file=argv[2];options.profile=argv[4];options.context=1048576;options.chunk=4096;options.slots=1;
    if(const char* chunk=std::getenv("DS41_BENCH_CHUNK"))options.chunk=std::stoi(chunk);
    options.hot_front=74662871040ull;options.hot_back=0;options.work=6144ull<<20;options.state=1024ull<<20;
    options.gpu_headroom=700ull<<20;options.engram_cache=8192ull<<20;options.migration_bytes=2048ull<<20;
    if(const char* work=std::getenv("DS41_BENCH_WORK_MIB"))options.work=std::stoull(work)<<20;
    if(const char* hot=std::getenv("DS41_BENCH_HOT_BYTES"))options.hot_front=std::stoull(hot);
    options.dspark=false;options.trace=mode=="prefill" || mode=="decode";
    if(options.trace)options.gpu_headroom=0; // Profiler-only admission gate; fixed actual allocation.
    need(std::string(argv[7])=="0" || std::string(argv[7])=="1","adaptive flag");options.adaptive_experts=std::string(argv[7])=="1";
    options.engram_pipeline=features&1;options.expert_bf16=features&2;options.expert_direct_gemv=features&4;
    options.expert_pipeline=features&8;options.adaptive_layer_progress=features&16;options.pcie_frac=(features&32)?.7:0;
    options.compact_logits=features&64;
    options.compact_workspace=features&128;
    if(const char* v=std::getenv("DS41_BENCH_PCIE_FRAC"))options.pcie_frac=std::stod(v);
    options.dense_q8=std::getenv("DS41_BENCH_DENSE_Q8")!=nullptr;
    options.dense_q8_1=std::getenv("DS41_BENCH_DENSE_Q8_1")!=nullptr;
    if(const char* v=std::getenv("DS41_BENCH_EXPERT_GU"))options.expert_gu_mode=std::stoi(v);
    options.cpu_gu_q8_0=std::getenv("DS41_BENCH_CPU_Q8_0")!=nullptr;
    options.decode_graph=std::getenv("DS41_BENCH_GRAPH")!=nullptr;
    options.decode_raw_cache=std::getenv("DS41_BENCH_RAW_CACHE")!=nullptr;
    options.pinned_experts=std::getenv("DS41_BENCH_PINNED_EXPERTS")!=nullptr;
    options.prefill_expert_lookahead=std::getenv("DS41_BENCH_EXPERT_LOOKAHEAD")!=nullptr;
    options.engram_continuous=std::getenv("DS41_BENCH_ENGRAM_CONTINUOUS")!=nullptr;
    options.parallel_router=std::getenv("DS41_BENCH_PARALLEL_ROUTER")!=nullptr;
    options.engram_aio=std::getenv("DS41_BENCH_ENGRAM_AIO")!=nullptr;
    if(const char* capture=std::getenv("DS41_BENCH_LAYERS"))options.capture_directory=capture;
    if(std::getenv("DS41_BENCH_LAYER_FILTER")) {
        for(int layer=0;layer<40;++layer)options.capture_stages.push_back("l_last-"+std::to_string(layer));
        for(int layer:{1,14})for(const char* prefix:{"engram_embd-","engram_kv-"})options.capture_stages.push_back(prefix+std::to_string(layer));
    }
    std::ofstream out(argv[5]);need(bool(out),"report open");out<<std::setprecision(17);
    out<<"{\"kind\":\"configuration\",\"features\":"<<features<<",\"context\":"<<options.context<<",\"chunk\":"<<options.chunk
       <<",\"actual_input\":"<<n<<",\"maximum\":128,\"fresh_runtime\":true,\"speculative\":false,\"draft_loaded\":false,\"gpu_headroom_bytes\":"<<options.gpu_headroom
       <<",\"pcie_frac\":"<<options.pcie_frac<<",\"dense_q8\":"<<options.dense_q8<<",\"dense_q8_1\":"<<options.dense_q8_1
       <<",\"expert_gu_mode\":"<<options.expert_gu_mode<<",\"cpu_gu_q8_0\":"<<options.cpu_gu_q8_0
       <<",\"decode_graph\":"<<options.decode_graph<<",\"engram_continuous\":"<<options.engram_continuous
       <<",\"decode_raw_cache\":"<<options.decode_raw_cache
       <<",\"pinned_experts\":"<<options.pinned_experts<<",\"prefill_expert_lookahead\":"<<options.prefill_expert_lookahead
       <<",\"parallel_router\":"<<options.parallel_router
       <<",\"engram_aio\":"<<options.engram_aio
       <<",\"adaptive\":"<<(options.adaptive_experts?"true":"false")<<",\"forced\":"<<(forced?"true":"false")<<"}\n";
    memory(out,"before_runtime");d::Runtime runtime(argv[1],options);memory(out,"after_runtime");
    out<<"{\"kind\":\"ledger\",\"value\":";runtime.write_ledger(out);out<<"}\n";
    out<<"{\"kind\":\"initial_heat\",\"value\":";runtime.write_expert_heat(out);out<<"}\n";out.flush();
    out<<"{\"kind\":\"expert_bytes\",\"value\":[";bool comma=false;
    for(auto bytes:runtime.model().expert_bytes()){if(comma)out<<',';comma=true;out<<bytes;}out<<"]}\n";
    auto session=runtime.create_session();memory(out,"after_session");
    if(mode=="prefill")need(cudaProfilerStart()==cudaSuccess,"prefill profile start");
    auto logits=session->prefill(prompt,true);
    if(mode=="prefill")need(cudaProfilerStop()==cudaSuccess,"prefill profile stop");
    memory(out,"after_prefill");
    out<<"{\"kind\":\"prefill_ledger\",\"value\":";runtime.write_ledger(out);out<<"}\n";
    const auto prefill_stats=session->stats();
    out<<"{\"kind\":\"prefill_stats\",\"engram_wait_ms\":"<<prefill_stats.engram_wait_ms<<",\"engram_layer_wait_ms\":["<<prefill_stats.engram_layer_wait_ms[0]<<','<<prefill_stats.engram_layer_wait_ms[1]<<"]}\n";
    out<<"{\"kind\":\"prefill_heat\",\"value\":";runtime.write_expert_heat(out);out<<"}\n";
    size_t emitted=0;const auto generation_start=std::chrono::steady_clock::now();
    auto token=[&](int id){
        if(mode=="decode" && emitted==64)need(cudaProfilerStop()==cudaSuccess,"decode profile stop");
        const auto st=session->stats();
        out<<"{\"kind\":\"token\",\"offset\":"<<emitted++<<",\"id\":"<<id<<",\"decoded_tokens\":"<<st.decoded_tokens<<",\"cpu_expert_tokens\":"<<st.expert_cpu
           <<",\"decode_ms\":"<<st.decode_ms<<",\"migrations\":"<<st.phase_migrations<<",\"demand_expert_tokens\":"<<st.demand_expert_tokens<<",\"demand_expert_bytes\":"<<st.demand_expert_bytes
           <<",\"raw_cache_hits\":"<<st.raw_cache_hits<<"}\n";out.flush();
    };
    strata::kernels::SamplerParams sampling;sampling.temperature=0;sampling.greedy=true;sampling.top_k=1;sampling.top_p=1;sampling.seed=1234;
    if(mode=="decode")need(cudaProfilerStart()==cudaSuccess,"decode profile start");
    std::vector<int> result;
    if(forced) {
        std::ofstream binary(std::string(argv[5])+".logits.f32",std::ios::binary);need(bool(binary),"forced logits open");
        logits_file(binary,logits);token(trajectory[0]);
        for(size_t i=1;i<trajectory.size();++i){logits=session->forward({trajectory[i-1]});logits_file(binary,logits);token(trajectory[i]);}
        result=trajectory;
    } else result=session->generate(logits,128,{1},sampling,false,token);
    const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-generation_start).count();
    const auto st=session->stats();need(result.size()==emitted && emitted==128,"complete128 output required");
    need(st.kv_only_tokens==prompt.size() && st.replay_tokens==std::min<size_t>(128,prompt.size()) && st.decoded_tokens==127,"physical token counters");
    need(st.encoder_token_layers==20*(prompt.size()+127) && st.decoder_token_layers==20*(std::min<size_t>(128,prompt.size())+127),"layer coverage");
    need(st.expert_cpu+st.expert_gpu==6*(st.encoder_token_layers+st.decoder_token_layers),"expert coverage");
    need(st.proposed_tokens==0 && st.draft_ms==0 && st.phase_ready_waits==0,"unexpected speculation/residency wait");
    memory(out,"after_generation");
    out<<"{\"kind\":\"sample\",\"emitted_tokens\":"<<emitted<<",\"decoded_tokens\":"<<st.decoded_tokens<<",\"prefill_ms\":"<<st.prefill_ms<<",\"decode_ms\":"<<st.decode_ms
       <<",\"generation_wall_ms\":"<<wall<<",\"cpu_expert_tokens\":"<<st.expert_cpu<<",\"gpu_expert_tokens\":"<<st.expert_gpu<<",\"h2d_expert_bytes\":"<<st.h2d_expert_bytes
       <<",\"demand_expert_tokens\":"<<st.demand_expert_tokens<<",\"demand_expert_bytes\":"<<st.demand_expert_bytes
       <<",\"raw_cache_hits\":"<<st.raw_cache_hits<<",\"raw_cache_d2d_bytes\":"<<st.raw_cache_d2d_bytes
       <<",\"direct_expert_bytes\":"<<st.direct_expert_bytes<<",\"prefetch_expert_bytes\":"<<st.prefetch_expert_bytes<<",\"prefetch_unused_bytes\":"<<st.prefetch_unused_bytes<<",\"prefetch_pruned_bytes\":"<<st.prefetch_pruned_bytes
       <<",\"engram_wait_ms\":"<<st.engram_wait_ms<<",\"engram_layer_wait_ms\":["<<st.engram_layer_wait_ms[0]<<','<<st.engram_layer_wait_ms[1]
       <<"],\"migration_h2d_bytes\":"<<st.phase_dynamic_h2d_bytes<<",\"migration_d2h_bytes\":"<<st.phase_dynamic_d2h_bytes<<",\"migrations\":"<<st.phase_migrations
       <<",\"shared_graph_captures\":"<<st.shared_graph_captures<<",\"shared_graph_launches\":"<<st.shared_graph_launches<<"}\n";
    out<<"{\"kind\":\"heat\",\"value\":";runtime.write_expert_heat(out);out<<"}\n";
    out<<"{\"kind\":\"final_ledger\",\"value\":";runtime.write_ledger(out);out<<"}\n{\"kind\":\"completed\"}\n";
    out.flush();need(bool(out),"report flush");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
