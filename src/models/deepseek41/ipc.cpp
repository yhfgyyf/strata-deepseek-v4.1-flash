#include "strata/models/deepseek41/runtime.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace strata::models::deepseek41 {
namespace {
struct ImageRequest { int64_t begin=0; int height=0,width=0; std::string path; };
struct Request {
    uint64_t id=0;
    int maximum=0,original_prompt=0,prefix_load_tokens=0,prefix_save_tokens=0;
    bool ced=true,spec=false;
    kernels::SamplerParams sampling;
    std::vector<int> tokens;
    std::vector<ImageRequest> images;
    std::string prefix_load,prefix_save;
    std::atomic<bool> cancelled{false},done{false};
    Session* session=nullptr; // protected by the control mutex
};
std::string line() {
    std::string value;
    if(!std::getline(std::cin,value)||value.size()>20u*1024*1024)throw std::runtime_error("DeepSeek IPC input ended or exceeds its bound");
    return value;
}
std::shared_ptr<Request> parse(std::istringstream& header,int context) {
    auto r=std::make_shared<Request>();int count=0,nimages=0,ced=0,spec=0;
    auto& p=r->sampling;
    // Parsing via double also accepts finite FP32 subnormals, which float stream extraction may reject.
    double temperature=0,top_p=0,min_p=0,repeat=0,frequency=0,presence=0;
    if(!(header>>r->id>>count>>r->maximum>>ced>>spec>>temperature>>p.top_k>>top_p>>min_p>>p.seed
         >>repeat>>frequency>>presence>>p.penalty_last_n>>nimages
         >>std::quoted(r->prefix_load)>>std::quoted(r->prefix_save)>>r->original_prompt))throw std::invalid_argument("DeepSeek IPC GEN header");
    p.temperature=temperature;p.top_p=top_p;p.min_p=min_p;p.penalty_repeat=repeat;
    p.penalty_freq=frequency;p.penalty_present=presence;
    r->prefix_load_tokens=r->prefix_load.empty()?0:r->original_prompt;
    r->prefix_save_tokens=r->prefix_save.empty()?0:r->original_prompt;
    header>>std::ws;
    if(!header.eof()){
        if(!(header>>r->prefix_load_tokens>>r->prefix_save_tokens))throw std::invalid_argument("DeepSeek IPC prefix bounds");
        header>>std::ws;if(!header.eof())throw std::invalid_argument("DeepSeek IPC unexpected GEN fields");
    }
    if(!r->id||count<1||count>context||r->maximum<0||r->maximum>context-count||
       r->original_prompt<1||r->original_prompt>count||(ced!=0&&ced!=1)||(spec!=0&&spec!=1)||
       nimages<0||nimages>64)throw std::invalid_argument("DeepSeek IPC GEN bounds");
    if(r->prefix_load_tokens<0||r->prefix_load_tokens>r->original_prompt||
       r->prefix_save_tokens<0||r->prefix_save_tokens>r->original_prompt||
       (r->prefix_load.empty()!=(r->prefix_load_tokens==0))||
       (r->prefix_save.empty()!=(r->prefix_save_tokens==0))||
       (r->prefix_save_tokens&&r->prefix_save_tokens<r->prefix_load_tokens))
        throw std::invalid_argument("DeepSeek IPC prefix bounds");
    if(nimages&&r->prefix_save_tokens&&r->prefix_save_tokens!=r->original_prompt)
        throw std::invalid_argument("DeepSeek partial prefix checkpoint does not support images");
    if(!std::isfinite(p.temperature)||p.temperature<0||!std::isfinite(p.top_p)||p.top_p<=0||p.top_p>1||
       !std::isfinite(p.min_p)||p.min_p<0||p.min_p>1||!std::isfinite(p.penalty_repeat)||p.penalty_repeat<=0||
       !std::isfinite(p.penalty_freq)||!std::isfinite(p.penalty_present))throw std::invalid_argument("DeepSeek IPC sampling bounds");
    r->ced=ced;r->spec=spec;r->tokens.resize(count);
    std::istringstream tokens(line());
    for(auto& id:r->tokens)if(!(tokens>>id)||id<0||id>=129280)throw std::invalid_argument("DeepSeek IPC token IDs");
    tokens>>std::ws;if(!tokens.eof())throw std::invalid_argument("DeepSeek IPC token count");
    for(int i=0;i<nimages;++i){
        ImageRequest image;std::istringstream input(line());
        if(!(input>>image.begin>>image.height>>image.width>>std::quoted(image.path)))throw std::invalid_argument("DeepSeek IPC image");
        input>>std::ws;
        if(!input.eof()||image.begin<0||image.begin>=r->original_prompt||image.height<1||image.width<1||
           int64_t(image.height)*image.width>9198||image.path.empty())throw std::invalid_argument("DeepSeek IPC image bounds");
        r->images.push_back(std::move(image));
    }
    return r;
}
}
int serve(Runtime& runtime,const RuntimeOptions& options) {
    std::mutex control,output;
    std::vector<std::shared_ptr<Request>> jobs(options.slots);
    std::vector<std::thread> workers(options.slots);
    auto say=[&](const std::string& message){std::lock_guard<std::mutex> lock(output);std::cout<<message<<std::endl;};
    auto error=[&](uint64_t id,const std::string& message){
        std::string safe=message;for(char& c:safe)if(c=='\n'||c=='\r')c=' ';
        say("E "+std::to_string(id)+" "+safe);
    };
    const auto* eos=runtime.model().file().get("tokenizer.ggml.eos_token_id");
    if(!eos)throw std::runtime_error("DeepSeek IPC EOS metadata");
    std::vector<int> eos_ids{static_cast<int>(eos->u)};
    say("READY "+std::to_string(options.context)+" "+std::to_string(options.slots));
    int result=0;
    try {
        std::string command;
        while(std::getline(std::cin,command)){
            if(command.size()>8192)throw std::invalid_argument("DeepSeek IPC control line bound");
            std::istringstream input(command);std::string kind;input>>kind;
            if(kind=="QUIT")break;
            if(kind=="CANCEL"){
                uint64_t id=0;if(!(input>>id))throw std::invalid_argument("DeepSeek IPC CANCEL ID");
                std::lock_guard<std::mutex> lock(control);
                for(const auto& job:jobs)if(job&&job->id==id&&!job->done){job->cancelled=true;if(job->session)job->session->cancel();}
                continue;
            }
            if(kind!="GEN")throw std::invalid_argument("DeepSeek IPC command");
            auto request=parse(input,options.context);int slot=-1;
            {
                std::lock_guard<std::mutex> lock(control);
                bool duplicate=false;
                for(const auto& job:jobs)if(job&&job->id==request->id&&!job->done)duplicate=true;
                if(duplicate){error(request->id,"duplicate active request ID");continue;}
                for(int i=0;i<options.slots;++i)if(!jobs[i]||jobs[i]->done){slot=i;break;}
            }
            if(slot<0){error(request->id,"all DeepSeek request slots are busy");continue;}
            if(workers[slot].joinable())workers[slot].join();
            {std::lock_guard<std::mutex> lock(control);jobs[slot]=request;}
            workers[slot]=std::thread([&,request]{
                std::unique_ptr<Session> session;
                try {
                    const std::string marker="ds41.request."+std::to_string(request->id);
                    strata::prefill::LedgerScope scope(options.trace,marker.c_str());
                    if(request->spec&&!options.dspark)throw std::invalid_argument("DSpark was not configured");
                    session=runtime.create_session();
                    {
                        std::lock_guard<std::mutex> lock(control);request->session=session.get();
                        if(request->cancelled)session->cancel();
                    }
                    std::vector<ImageEmbedding> images;
                    for(const auto& image:request->images){
                        if(request->cancelled)throw std::runtime_error("DeepSeek request cancelled");
                        std::vector<float> patches(int64_t(image.height)*image.width*588);
                        std::ifstream file(image.path,std::ios::binary);
                        file.read(reinterpret_cast<char*>(patches.data()),patches.size()*4);
                        if(!file||file.peek()!=EOF)throw std::runtime_error("DeepSeek image patch file size");
                        images.push_back(runtime.encode_image(patches,image.height,image.width,image.begin));
                    }
                    using Clock=std::chrono::steady_clock;
                    const auto prefill_begin=Clock::now();
                    if(!request->prefix_load.empty())session->load_prefix(request->prefix_load,
                        std::vector<int>(request->tokens.begin(),request->tokens.begin()+request->prefix_load_tokens));
                    say("R "+std::to_string(request->id)+" "+std::to_string(session->position()));
                    const bool split=request->prefix_save_tokens&&request->prefix_save_tokens<request->original_prompt;
                    if(split){
                        const std::vector<int> prefix(request->tokens.begin(),request->tokens.begin()+request->prefix_save_tokens);
                        session->resume_history(prefix,request->prefix_save_tokens,request->ced,images,request->prefix_save);
                    }
                    auto logits=session->resume_history(request->tokens,request->original_prompt,request->ced,images,
                        split?std::string{}:request->prefix_save);
                    const auto generation_begin=Clock::now();
                    session->generate(logits,request->maximum,eos_ids,request->sampling,request->spec,[&](int token){
                        if(request->cancelled)throw std::runtime_error("DeepSeek request cancelled");
                        say("T "+std::to_string(request->id)+" "+std::to_string(token));
                    });
                    const auto cleanup_begin=Clock::now();
                    const auto stats=session->stats();
                    std::ostringstream done;done<<std::setprecision(12)<<"D "<<request->id<<' '<<stats.prefill_ms<<' '<<stats.decode_ms<<' '
                        <<stats.proposed_tokens<<' '<<stats.accepted_tokens<<' '<<stats.engram_wait_ms<<' '<<stats.expert_gpu<<' '
                        <<stats.expert_cpu<<' '<<stats.h2d_expert_bytes<<' '<<stats.encoder_token_layers<<' '<<stats.decoder_token_layers
                        <<' '<<stats.kv_only_tokens<<' '<<stats.replay_tokens<<' '<<stats.draft_ms<<' '<<stats.verify_ms
                        <<' '<<stats.phase_borrow_bytes<<' '<<stats.phase_backup_bytes<<' '<<stats.phase_encoder_h2d_bytes
                        <<' '<<stats.phase_restore_h2d_bytes<<' '<<stats.phase_replay_reuse_bytes
                        <<' '<<stats.phase_dynamic_h2d_bytes<<' '<<stats.phase_dynamic_d2h_bytes<<' '<<stats.phase_migrations
                        <<' '<<stats.phase_ready_waits<<' '<<stats.phase_prepare_ms<<' '<<stats.phase_host_wait_ms
                        <<' '<<stats.phase_restore_submit_ms<<' '<<stats.phase_dynamic_ms;
                    // Release the slot before publishing DONE: the next request may arrive immediately.
                    {std::lock_guard<std::mutex> lock(control);request->session=nullptr;}
                    session.reset();runtime.save_learned_profile();
                    const auto cleanup_end=Clock::now();
                    done<<' '<<std::chrono::duration<double,std::milli>(generation_begin-prefill_begin).count()
                        <<' '<<std::chrono::duration<double,std::milli>(cleanup_begin-generation_begin).count()
                        <<' '<<std::chrono::duration<double,std::milli>(cleanup_end-cleanup_begin).count();
                    request->done=true;say(done.str());
                }catch(const std::exception& e){
                    {std::lock_guard<std::mutex> lock(control);request->session=nullptr;}
                    session.reset();request->done=true;error(request->id,e.what());
                }
            });
        }
    }catch(const std::exception& e){error(0,e.what());result=1;}
    {
        std::lock_guard<std::mutex> lock(control);
        for(const auto& job:jobs)if(job&&!job->done){job->cancelled=true;if(job->session)job->session->cancel();}
    }
    for(auto& worker:workers)if(worker.joinable())worker.join();
    return result;
}
} // namespace strata::models::deepseek41
