#include "strata/models/deepseek41/runtime.hpp"
#include "strata/models/deepseek41/main.hpp"
#include "strata/models/deepseek41/prefill_chunk.hpp"
#include "strata/kernels/cpu/deepseek41_down.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace strata::models::deepseek41 {
int serve(Runtime& runtime,const RuntimeOptions& options);
namespace {
uint64_t parse_bytes(const std::string& value,unsigned shift) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("byte budget must be an unsigned integer");
    const uint64_t units=std::stoull(value);
    if (units > (std::numeric_limits<uint64_t>::max() >> shift))
        throw std::overflow_error("byte budget overflow");
    return units << shift;
}
std::vector<int> parse_tokens(std::string text) {
    for(char& c:text)if(c==','||c=='['||c==']')c=' ';
    std::istringstream input(text);std::vector<int> ids;int token;
    while(input>>token){if(token<0||token>=129280)throw std::invalid_argument("token ID outside vocabulary");ids.push_back(token);}
    if(!input.eof()||ids.empty())throw std::invalid_argument("invalid token list");
    return ids;
}
void stats(const ExecutionStats& s) {
    std::cout<<"{\"encoder_token_layers\":"<<s.encoder_token_layers<<",\"decoder_token_layers\":"<<s.decoder_token_layers
        <<",\"kv_only_tokens\":"<<s.kv_only_tokens<<",\"replay_tokens\":"<<s.replay_tokens<<",\"decoded_tokens\":"<<s.decoded_tokens
        <<",\"gpu_expert_tokens\":"<<s.expert_gpu<<",\"cpu_expert_tokens\":"<<s.expert_cpu<<",\"expert_h2d_bytes\":"<<s.h2d_expert_bytes
        <<",\"prefill_ms\":"<<s.prefill_ms<<",\"decode_ms\":"<<s.decode_ms<<",\"engram_wait_ms\":"<<s.engram_wait_ms
        <<",\"proposed_tokens\":"<<s.proposed_tokens<<",\"accepted_tokens\":"<<s.accepted_tokens<<",\"draft_ms\":"<<s.draft_ms<<",\"verify_ms\":"<<s.verify_ms
        <<",\"verify_calls\":"<<s.verify_calls<<",\"verify_rows\":"<<s.verify_rows
        <<",\"rejection_replay_calls\":"<<s.rejection_replay_calls<<",\"rejection_replay_rows\":"<<s.rejection_replay_rows
        <<",\"draft_gpu_rounds\":"<<s.draft_gpu_rounds<<",\"draft_context_rows\":"<<s.draft_context_rows
        <<",\"draft_context_projection_rows\":"<<s.draft_context_projection_rows
        <<",\"draft_fused_moe_calls\":"<<s.draft_fused_moe_calls<<",\"draft_batch_projection_calls\":"<<s.draft_batch_projection_calls
        <<",\"phase_borrow_bytes\":"<<s.phase_borrow_bytes<<",\"phase_backup_bytes\":"<<s.phase_backup_bytes
        <<",\"phase_encoder_h2d_bytes\":"<<s.phase_encoder_h2d_bytes<<",\"phase_restore_h2d_bytes\":"<<s.phase_restore_h2d_bytes
        <<",\"phase_replay_reuse_bytes\":"<<s.phase_replay_reuse_bytes<<",\"phase_dynamic_h2d_bytes\":"<<s.phase_dynamic_h2d_bytes
        <<",\"phase_dynamic_d2h_bytes\":"<<s.phase_dynamic_d2h_bytes<<",\"phase_migrations\":"<<s.phase_migrations
        <<",\"phase_ready_waits\":"<<s.phase_ready_waits<<",\"phase_prepare_ms\":"<<s.phase_prepare_ms
        <<",\"phase_host_wait_ms\":"<<s.phase_host_wait_ms<<",\"phase_restore_submit_ms\":"<<s.phase_restore_submit_ms
        <<",\"phase_dynamic_ms\":"<<s.phase_dynamic_ms<<"}\n";
}
}
int main(int argc,char** argv) {
    try {
        RuntimeOptions options;std::string model,draft,vision,logits_path,prefix_save,prefix_load,heat_path;std::vector<int> tokens;
        bool ced=true,resident=false;int max_new=0;int64_t original_prompt=0;kernels::SamplerParams sampling;sampling.temperature=0;
        bool hot_front_set=false,hot_back_set=false;
        for(int i=1;i<argc;++i){
            const std::string a=argv[i];auto next=[&]{if(i+1>=argc)throw std::invalid_argument(a+" requires a value");return std::string(argv[++i]);};
            if(a=="--help"||a=="-h"){
                std::cout<<"strata deepseek41 --model MAIN.gguf --tokens IDS [--mode full|ced] [--arithmetic f32|production]\n"
                    <<"  --context N --chunk N|auto[:N] --slots N --expert-mode hybrid|gpu|cpu --max-new N\n"
                    <<"  --prefill aliases --chunk; auto selects within --work-mib, up to 32768 tokens\n"
                    <<"  --spec --draft FILE --draft-sha256 SHA256 --draft-vocab IDS.bin --draft-input-vocab IDS.bin\n"
                    <<"  --no-dspark-prefix-commit --no-dense-q8-batch --no-dspark-gpu-greedy --no-dspark-incremental-kv\n"
                    <<"  --no-dspark-fused-moe --no-dspark-batch-projection --no-target-verify-batch\n"
                    <<"  --capture DIRECTORY --logits FILE --prefix-save FILE --prefix-load FILE\n"
                    <<"  --original-prompt N --trace\n"
                    <<"  --profile FILE --hot-auto --hot-front-bytes N --hot-front-gib N --hot-back-gib N\n"
                    <<"  --work-mib N --state-mib N --gpu-headroom-mib N\n"
                    <<"  --adaptive-experts --adaptive-layer-progress --migration-mib N (single-slot hybrid)\n"
                    <<"  --adaptive-expert-policy legacy|maya --adaptive-expert-scope layer|format-group --expert-profile-save FILE (optional, separate from --profile)\n"
                    <<"  --engram-pipeline --expert-bf16 --expert-direct-gemv --expert-pipeline --pcie-frac F\n"
                    <<"  --compact-logits --compact-workspace\n"
                    <<"  --dense-q8 --dense-q8-1 --expert-gu-mode 0|1|2 --cpu-gu-q8-0\n"
                    <<"  --decode-shared-graph --decode-raw-cache --engram-continuous --engram-aio --parallel-router\n"
                    <<"  --pinned-experts --prefill-expert-lookahead\n";return 0;
            }
            else if(a=="--model")model=next();else if(a=="--draft")draft=next();else if(a=="--mmproj")vision=next();
            else if(a=="--tokens")tokens=parse_tokens(next());
            else if(a=="--tokens-file"){std::ifstream file(next());if(!file)throw std::runtime_error("token file");tokens=parse_tokens(std::string(std::istreambuf_iterator<char>(file),{}));}
            else if(a=="--mode"){const auto value=next();if(value!="full"&&value!="ced")throw std::invalid_argument("mode");ced=value=="ced";}
            else if(a=="--arithmetic"){const auto value=next();if(value!="f32"&&value!="production")throw std::invalid_argument("arithmetic");options.arithmetic=value=="f32"?Arithmetic::f32:Arithmetic::production;}
            else if(a=="--expert-mode"){const auto value=next();if(value!="hybrid"&&value!="gpu"&&value!="cpu")throw std::invalid_argument("expert mode");options.experts=value=="hybrid"?ExpertMode::hybrid:value=="cpu"?ExpertMode::cpu:ExpertMode::gpu;}
            else if(a=="--context")options.context=std::stoi(next());
            else if(a=="--chunk"||a=="--prefill"){
                const auto chunk=parse_prefill_chunk(next());options.chunk=chunk.tokens;options.chunk_auto=chunk.automatic;
            }
            else if(a=="--slots")options.slots=std::stoi(next());else if(a=="--cpu-workers")options.cpu_workers=std::stoi(next());
            else if(a=="--hot-front-bytes"){options.hot_front=parse_bytes(next(),0);hot_front_set=true;}
            else if(a=="--hot-front-gib"){options.hot_front=parse_bytes(next(),30);hot_front_set=true;}
            else if(a=="--hot-back-gib"){options.hot_back=parse_bytes(next(),30);hot_back_set=true;}
            else if(a=="--hot-auto")options.hot_auto=true;
            else if(a=="--work-mib")options.work=parse_bytes(next(),20);
            else if(a=="--state-mib")options.state=parse_bytes(next(),20);
            else if(a=="--gpu-headroom-mib")options.gpu_headroom=parse_bytes(next(),20);
            else if(a=="--engram-cache-mib")options.engram_cache=parse_bytes(next(),20);
            else if(a=="--profile")options.profile=next();else if(a=="--capture")options.capture_directory=next();
            else if(a=="--integrity")options.integrity_file=next();
            else if(a=="--trace")options.trace=true;
            else if(a=="--original-prompt")original_prompt=std::stoll(next());
            else if(a=="--phase-residency")options.phase_residency=true;
            else if(a=="--adaptive-experts")options.adaptive_experts=true;
            else if(a=="--adaptive-expert-scope"){
                const auto scope=next();if(scope!="layer"&&scope!="format-group")throw std::invalid_argument("adaptive expert scope");
                options.adaptive_groups=scope=="format-group";
            }
            else if(a=="--adaptive-expert-policy"){
                const auto policy=next();if(policy!="legacy"&&policy!="maya")throw std::invalid_argument("adaptive expert policy");
                options.adaptive_maya=policy=="maya";
            }
            else if(a=="--expert-profile-save")options.expert_profile_save=next();
            else if(a=="--engram-pipeline")options.engram_pipeline=true;
            else if(a=="--compact-logits")options.compact_logits=true;
            else if(a=="--compact-workspace")options.compact_workspace=true;
            else if(a=="--dense-q8")options.dense_q8=true;
            else if(a=="--dense-q8-1")options.dense_q8_1=true;
            else if(a=="--expert-gu-mode")options.expert_gu_mode=std::stoi(next());
            else if(a=="--cpu-gu-q8-0")options.cpu_gu_q8_0=true;
            else if(a=="--decode-shared-graph")options.decode_graph=true;
            else if(a=="--decode-raw-cache")options.decode_raw_cache=true;
            else if(a=="--pinned-experts")options.pinned_experts=true;
            else if(a=="--no-huge-pinned")options.huge_pinned_memory=false;
            else if(a=="--prefill-expert-lookahead")options.prefill_expert_lookahead=true;
            else if(a=="--engram-continuous")options.engram_continuous=true;
            else if(a=="--parallel-router")options.parallel_router=true;
            else if(a=="--engram-aio")options.engram_aio=true;
            else if(a=="--expert-bf16")options.expert_bf16=true;
            else if(a=="--expert-direct-gemv")options.expert_direct_gemv=true;
            else if(a=="--expert-pipeline")options.expert_pipeline=true;
            else if(a=="--adaptive-layer-progress")options.adaptive_layer_progress=true;
            else if(a=="--pcie-frac") {
                const auto value=next();size_t used=0;options.pcie_frac=std::stod(value,&used);
                if(used!=value.size() || !std::isfinite(options.pcie_frac) || options.pcie_frac<0 || options.pcie_frac>1)
                    throw std::invalid_argument("pcie-frac must be between 0 and 1");
            }
            else if(a=="--expert-down-fusion")kernels::cpu::deepseek41_down_fusion_set_enabled(true);
            else if(a=="--borrow-gib")options.borrow_bytes=parse_bytes(next(),30);
            else if(a=="--migration-mib")options.migration_bytes=parse_bytes(next(),20);
            else if(a=="--prefill-profile")options.prefill_profile=next();
            else if(a=="--expert-heat")heat_path=next();
            else if(a=="--logits")logits_path=next();else if(a=="--max-new")max_new=std::stoi(next());
            else if(a=="--prefix-save")prefix_save=next();else if(a=="--prefix-load")prefix_load=next();
            else if(a=="--draft-sha256")options.draft_sha256=next();
            else if(a=="--vision-sha256")options.vision_sha256=next();
            else if(a=="--draft-vocab")options.draft_vocab=next();
            else if(a=="--draft-input-vocab")options.draft_input_vocab=next();
            else if(a=="--no-dspark-prefix-commit")options.dspark_prefix_commit=false;
            else if(a=="--no-dense-q8-batch")options.dense_q8_batch=false;
            else if(a=="--no-target-verify-batch")options.target_verify_batch=false;
            else if(a=="--no-dspark-gpu-greedy")options.dspark_gpu_greedy=false;
            else if(a=="--no-dspark-incremental-kv")options.dspark_incremental_kv=false;
            else if(a=="--no-dspark-fused-moe")options.dspark_fused_moe=false;
            else if(a=="--no-dspark-batch-projection")options.dspark_batch_projection=false;
            else if(a=="--spec")options.dspark=true;else if(a=="--serve")resident=true;
            else if(a=="--temperature")sampling.temperature=std::stof(next());else if(a=="--top-k")sampling.top_k=std::stoi(next());
            else if(a=="--top-p")sampling.top_p=std::stof(next());else if(a=="--min-p")sampling.min_p=std::stof(next());
            else if(a=="--seed")sampling.seed=std::stoull(next());
            else throw std::invalid_argument("unknown DeepSeek option: "+a);
        }
        if(model.empty()||(!resident&&tokens.empty())||max_new<0||tokens.size()+max_new>static_cast<size_t>(options.context))throw std::invalid_argument("model/tokens/context/max-new");
        if(!resident){
            if(!original_prompt)original_prompt=tokens.size();
            if(original_prompt<1||original_prompt>static_cast<int64_t>(tokens.size()))throw std::invalid_argument("original prompt boundary");
        }
        if(options.hot_auto&&(options.profile.empty()||options.arithmetic==Arithmetic::f32))
            throw std::invalid_argument("hot-auto requires production arithmetic and a profile");
        if(!options.work||!options.state)throw std::invalid_argument("work/state budgets must be positive");
        if(options.arithmetic==Arithmetic::f32){options.experts=ExpertMode::gpu;options.hot_front=options.hot_back=0;}
        else if(options.dspark&&!options.hot_auto){if(!hot_front_set)options.hot_front=34ull<<30;if(!hot_back_set)options.hot_back=14ull<<30;}
        Runtime runtime(model,options,draft,vision);
        if(resident){runtime.write_ledger(std::cerr);std::cerr<<'\n';return serve(runtime,options);}
        runtime.write_ledger(std::cout);std::cout<<'\n';
        auto session=runtime.create_session();
        if(!prefix_load.empty())session->load_prefix(prefix_load,std::vector<int>(tokens.begin(),tokens.begin()+original_prompt));
        auto output=session->resume_history(tokens,original_prompt,ced,{},prefix_save);
        if(!logits_path.empty()){std::ofstream out(logits_path,std::ios::binary);out.write(reinterpret_cast<const char*>(output.data()),output.size()*4);if(!out)throw std::runtime_error("logit file");}
        const auto* eos=runtime.model().file().get("tokenizer.ggml.eos_token_id");
        std::vector<int> eos_ids;if(eos)eos_ids.push_back(eos->u);
        session->generate(output,max_new,eos_ids,sampling,options.dspark,[](int id){std::cout<<"T "<<id<<std::endl;});
        stats(session->stats());
        runtime.save_learned_profile();
        if(!heat_path.empty()){std::ofstream out(heat_path);runtime.write_expert_heat(out);out<<'\n';if(!out)throw std::runtime_error("expert heat file");}
        return 0;
    }catch(const std::exception& error){std::cerr<<"DeepSeek: "<<error.what()<<'\n';return 1;}
}
} // namespace strata::models::deepseek41
#ifdef STRATA_DEEPSEEK41_STANDALONE
int main(int argc,char** argv){return strata::models::deepseek41::main(argc,argv);}
#endif
