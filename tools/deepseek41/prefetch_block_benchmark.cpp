// Isolated private five-block diagnostic, never a generation/serving entrypoint.
#include "expert_pipeline.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace strata::models::deepseek41 {
namespace {
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::duration elapsed){return std::chrono::duration<double,std::milli>(elapsed).count();}
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void memory(std::ostream& out,const char* label){
    size_t free=0,total=0;gpu_check(cudaMemGetInfo(&free,&total));
    out<<"{\"kind\":\"memory\",\"stage\":\""<<label<<"\",\"free_bytes\":"<<free<<",\"total_bytes\":"<<total<<"}\n";
}
struct BlockRange {
#if defined(STRATA_PREFILL_LEDGER_NVTX)
    nvtxRangeId_t id;
    BlockRange(){id=nvtxRangeStartA("ds41.five_blocks");}
    ~BlockRange(){nvtxRangeEnd(id);}
#endif
};
}
struct PrefetchBlockBenchmark {
    static void warmup(Runtime& runtime,const std::vector<int>& counts,std::ostream& out){
        auto& r=*runtime.impl_;auto& w=*r.work;
        std::lock_guard<std::mutex> execution(r.execution_mutex);
        need(!r.prefetch_active && !r.expert_generation && !r.active_sessions,"warmup must precede requests and expert prefetch");
        prefill::LedgerScope trace(r.options.trace,"ds41.gemm_warmup");
        const auto started=Clock::now();
        gpu_check(cudaStreamSynchronize(r.stream.value));
        memory(out,"before_gemm_warmup");
        // Scratch only: no model weights, expert heat, route or Session state.
        gpu_check(cudaMemsetAsync(w.x,0,uint64_t(w.max_tokens)*5120*sizeof(float),r.stream.value));
        gpu_check(cudaMemsetAsync(w.expert_gate,0,5120ull*2304*sizeof(uint16_t),r.stream.value));
        gpu_check(cudaMemsetAsync(w.expert_up,0,5120ull*2304*sizeof(uint16_t),r.stream.value));
        gpu_check(cudaMemsetAsync(w.expert_down,0,5120ull*2304*sizeof(uint16_t),r.stream.value));
        Matrix gate{w.expert_gate,30,2304,5120,5120ull*2304};
        Matrix up{w.expert_up,30,2304,5120,5120ull*2304};
        Matrix down{w.expert_down,30,5120,2304,5120ull*2304};
        for(int nt:counts){
            need(nt>8 && nt<=w.max_tokens,"warmup GEMM token bound");
            r.matmul(gate,w.x,w.group_gate,nt,w);
            r.matmul(up,w.x,w.group_up,nt,w);
            r.matmul(down,w.group_gate,w.group_out,nt,w);
        }
        gpu_check(cudaStreamSynchronize(r.stream.value));
        memory(out,"after_gemm_warmup");
        out<<"{\"kind\":\"gemm_warmup\",\"shapes\":"<<counts.size()<<",\"gemm_calls\":"<<3*counts.size()
           <<",\"wall_ms\":"<<milliseconds(Clock::now()-started)<<",\"before_expert_prefetch\":true}\n";
    }
    static void run(Runtime& runtime,Session& session,const std::vector<int>& prompt,std::ostream& out,const std::string& hidden,Clock::time_point main_started){
        auto& r=*runtime.impl_;auto& s=*session.impl_;
        std::lock_guard<std::mutex> execution(r.execution_mutex);
        need(r.options.pinned_experts && r.host_ring->pinned,"pinned expert source/staging required");
        need(r.ring_slots==128 && r.options.pcie_frac==1 && !r.options.adaptive_experts,"five-block resource contract");
        s.ced=true;s.original_prompt=prompt.size();s.last_encoder_chunk=false;
        if(r.decode_raw_cache)r.decode_raw_cache->clear(r.upload.value);
        r.experts->begin_prefill(prompt.size(),r.options.chunk,true,r.stream.value,s.counters);
        if(r.options.engram_continuous)s.prefetch_engram(prompt,0);
        s.init(prompt,0);
        cudaEvent_t begin=nullptr,end=nullptr;
        gpu_check(cudaEventCreate(&begin));
        try{gpu_check(cudaEventCreate(&end));}catch(...){cudaEventDestroy(begin);throw;}
        struct Events {cudaEvent_t a,b;~Events(){cudaEventDestroy(a);cudaEventDestroy(b);}} events{begin,end};
        Clock::time_point started,completed;
        std::unique_ptr<BlockRange> phase;
        BlockBenchmarkHooks hooks;
        hooks.attention_begin=[&]{
            gpu_check(cudaStreamSynchronize(r.stream.value));
            memory(out,"before_attention");
            if(r.options.trace)phase=std::make_unique<BlockRange>();
            started=Clock::now();gpu_check(cudaEventRecord(begin,r.stream.value));
        };
        hooks.last_block_end=[&]{
            gpu_check(cudaEventRecord(end,r.stream.value));gpu_check(cudaEventSynchronize(end));
            completed=Clock::now();phase.reset();
        };
        s.layers(prompt.size(),0,0,5,true,&hooks);
        const auto drained=Clock::now();
        float device_ms=0;gpu_check(cudaEventElapsedTime(&device_ms,begin,end));
        need(s.counters.encoder_token_layers==5*prompt.size() && !s.counters.decoder_token_layers,"exactly five blocks");
        need(!s.counters.expert_cpu && s.counters.expert_gpu==30*prompt.size(),"all expert arithmetic GPU");
        memory(out,"after_blocks");
        out<<"{\"kind\":\"sample\",\"blocks\":5,\"input_tokens\":"<<prompt.size()
           <<",\"attention_to_block4_wall_ms\":"<<milliseconds(completed-started)<<",\"attention_to_block4_cuda_ms\":"<<device_ms
           <<",\"main_entry_to_block4_wall_ms\":"<<milliseconds(completed-main_started)
           <<",\"remaining_prefetch_drain_ms\":"<<milliseconds(drained-completed)
           <<",\"attention_to_drain_ms\":"<<milliseconds(drained-started)
           <<",\"expert_gpu_tokens\":"<<s.counters.expert_gpu<<",\"expert_cpu_tokens\":"<<s.counters.expert_cpu
           <<",\"h2d_expert_bytes\":"<<s.counters.h2d_expert_bytes<<",\"direct_expert_bytes\":"<<s.counters.direct_expert_bytes
           <<",\"prefetch_expert_bytes\":"<<s.counters.prefetch_expert_bytes<<",\"prefetch_unused_bytes\":"<<s.counters.prefetch_unused_bytes
           <<",\"prefetch_pruned_bytes\":"<<s.counters.prefetch_pruned_bytes<<",\"engram_wait_ms\":"<<s.counters.engram_wait_ms<<"}\n";
        out<<"{\"kind\":\"heat\",\"value\":";runtime.write_expert_heat(out);out<<"}\n";
        // Independent source/reader checks and numerical comparisons use this
        // full tensor on the server. Its D2H and file I/O are outside the timer.
        std::ofstream tensor(hidden,std::ios::binary);need(bool(tensor),"hidden output open");
        const uint64_t bytes=uint64_t(prompt.size())*20480*sizeof(float),chunk=8ull<<20;
        Pinned host(chunk);need(host.pinned,"hidden D2H needs pinned memory");
        for(uint64_t offset=0;offset<bytes;offset+=chunk){
            const auto count=std::min(chunk,bytes-offset);
            gpu_check(cudaMemcpyAsync(host.data,reinterpret_cast<const uint8_t*>(s.work.h)+offset,count,cudaMemcpyDeviceToHost,r.stream.value));
            gpu_check(cudaStreamSynchronize(r.stream.value));
            tensor.write(reinterpret_cast<const char*>(host.data),count);need(bool(tensor),"hidden output write");
        }
        out<<"{\"kind\":\"hidden_tensor\",\"bytes\":"<<bytes<<",\"shape\":["<<prompt.size()<<",4,5120]}\n";
        out<<"{\"kind\":\"completed\"}\n";out.flush();need(bool(out),"statistics output");
    }
};
}
int main(int argc,char** argv){try{
    namespace d=strata::models::deepseek41;
    const auto main_started=std::chrono::steady_clock::now();
    if(argc!=8 && argc!=9)throw std::runtime_error("MODEL PROOF INPUT PROFILE NEW.jsonl demand/lookahead trace0/1 [GEMM_COUNTS]");
    const bool lookahead=std::string(argv[6])=="lookahead",trace=std::string(argv[7])=="1";
    if(!lookahead && std::string(argv[6])!="demand")throw std::invalid_argument("five-block mode");
    if(std::string(argv[7])!="0" && !trace)throw std::invalid_argument("trace flag");
    if(std::filesystem::exists(argv[5]) || std::filesystem::exists(std::string(argv[5])+".hidden.f32"))throw std::runtime_error("output exists");
    std::ifstream input(argv[3]);std::string id;int count=0;
    if(!(input>>id>>count) || count!=8192)throw std::runtime_error("expected actual8192 input");
    std::vector<int> tokens(count);for(auto& token:tokens)if(!(input>>token) || token<0 || token>=129280)throw std::runtime_error("invalid input token");
    input>>std::ws;if(!input.eof())throw std::runtime_error("extra input records");
    std::vector<int> warmup_counts;
    if(argc==9){
        std::ifstream shapes(argv[8]);int nt=0;
        if(!shapes)throw std::runtime_error("warmup counts open");
        while(shapes>>nt){
            if(nt<=8 || nt>8192 || (!warmup_counts.empty() && nt<=warmup_counts.back()))throw std::runtime_error("warmup counts must be sorted unique GEMM sizes");
            warmup_counts.push_back(nt);
        }
        if(!shapes.eof() || warmup_counts.empty())throw std::runtime_error("invalid warmup counts");
    }
    d::RuntimeOptions options;options.integrity_file=argv[2];options.profile=argv[4];options.context=1048576;options.chunk=8192;options.slots=1;
    options.hot_front=80603840512ull;options.hot_back=0;options.work=5888ull<<20;options.state=1024ull<<20;
    options.gpu_headroom=trace?0:700ull<<20;options.engram_cache=8192ull<<20;options.adaptive_experts=false;options.pcie_frac=1;
    options.dspark=false;options.trace=trace;options.engram_pipeline=true;options.expert_bf16=true;options.expert_direct_gemv=true;
    options.expert_pipeline=true;options.compact_logits=true;options.compact_workspace=true;options.dense_q8=true;options.dense_q8_1=true;
    options.expert_gu_mode=2;options.decode_graph=true;options.decode_raw_cache=true;options.pinned_experts=true;
    options.prefill_expert_lookahead=lookahead;options.engram_continuous=true;options.parallel_router=true;options.engram_aio=true;
    std::ofstream out(argv[5]);if(!out)throw std::runtime_error("report open");out<<std::setprecision(17);
    out<<"{\"kind\":\"configuration\",\"input\":8192,\"chunk\":8192,\"context\":1048576,\"blocks\":[0,1,2,3,4],\"pinned_experts\":true,\"ring_slots\":128,\"pcie_frac\":1,\"prefill_expert_lookahead\":"<<lookahead<<",\"trace\":"<<trace<<",\"adaptive\":false,\"engram_pipeline\":true,\"boundary\":\"first Attention to block4 final residual; prefetch starts inside; no decode\"}\n";
    d::Runtime runtime(argv[1],options);
    out<<"{\"kind\":\"initial_ledger\",\"value\":";runtime.write_ledger(out);out<<"}\n";
    out<<"{\"kind\":\"initial_heat\",\"value\":";runtime.write_expert_heat(out);out<<"}\n";
    out<<"{\"kind\":\"runtime_initialization\",\"main_entry_to_runtime_ready_ms\":"<<d::milliseconds(std::chrono::steady_clock::now()-main_started)<<"}\n";
    if(!warmup_counts.empty())d::PrefetchBlockBenchmark::warmup(runtime,warmup_counts,out);
    auto session=runtime.create_session();
    d::PrefetchBlockBenchmark::run(runtime,*session,tokens,out,std::string(argv[5])+".hidden.f32",main_started);
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
