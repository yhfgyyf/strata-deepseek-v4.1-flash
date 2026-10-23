#pragma once

#include "strata/models/deepseek41/model.hpp"
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "strata/kernels/sampler.hpp"

namespace strata::models::deepseek41 {
enum class Arithmetic { f32, production };
enum class ExpertMode { hybrid, gpu, cpu };
struct RuntimeOptions {
    Arithmetic arithmetic = Arithmetic::production;
    ExpertMode experts = ExpertMode::hybrid;
    int context = 32768, chunk = 1024, cpu_workers = 0, slots = 1;
    bool chunk_auto = false; // chunk is the ceiling; resolved once within the existing work budget.
    uint64_t hot_front = 40ull << 30, hot_back = 16ull << 30;
    bool hot_auto = false; // Profile prefix uses free GPU bytes after subsequent regions and headroom.
    uint64_t engram_cache = 8ull << 30, ram_experts = 144ull << 30;
    uint64_t ring = 2ull << 30, work = 6ull << 30, state = 2ull << 30;
    uint64_t gpu_headroom = 700ull << 20, host_headroom = 16ull << 30;
    std::string profile, capture_directory, integrity_file;
    bool dspark = false, trace = false;
    std::string draft_sha256, vision_sha256, draft_vocab, draft_input_vocab;
    bool dspark_prefix_commit = true, dense_q8_batch = true;
    bool target_verify_batch = true; // Exact multi-row projections and expert operations, 2..8 tokens.
    bool dspark_gpu_greedy = true, dspark_incremental_kv = true;
    bool dspark_fused_moe = true, dspark_batch_projection = true;
    bool dspark_adaptive_verification = false;
    int dspark_draft_tokens = 5; // Supported proposal lengths: 5 or 7; verify uses one extra anchor row.
    // Calibrated wall-clock costs for one draft and verification of 1..8 rows.
    double dspark_draft_cost_ms = 0;
    std::array<double,8> dspark_verify_cost_ms{};
    bool phase_residency = false;
    bool adaptive_experts = false; // Single-slot hybrid decode: asynchronous RAM/GPU exchanges.
    bool adaptive_groups = false; // Q3_K layers 0..13 / Q2_K layers 14..39 compete within each group.
    std::string expert_profile_save; // Opt-in: atomically save learned startup ranking after a successful request.
    uint64_t borrow_bytes = 8ull << 30, migration_bytes = 64ull << 20;
    std::string prefill_profile;
    // Independently selectable experiments; existing execution remains the control.
    bool engram_pipeline = false, expert_bf16 = false, expert_direct_gemv = false;
    bool expert_pipeline = false, adaptive_layer_progress = false;
    double pcie_frac = 0;
    bool compact_logits = false;
    bool compact_workspace = false; // Main-model phase scratch reuse; DSpark reserves its wider FC input.
    bool dense_q8 = false, dense_q8_1 = false; // Packed attention/shared weights; optional short Q8_1 inputs.
    bool shared_q8_pair = false; // Shared Gate/Up reuse one Q8_1 input and an exact paired projection.
    int expert_gu_mode = 0; // 0: separate BF16 GEMVs; 1: fused BF16; 2: fused integer Q8_1.
    bool cpu_gu_q8_0 = false, decode_graph = false, engram_continuous = false;
    bool parallel_router = false;
    bool engram_aio = false;
    bool decode_raw_cache = false;
    bool pinned_experts = false, prefill_expert_lookahead = false;
    bool huge_pinned_memory = true;
    std::vector<std::string> capture_stages; // Empty preserves the complete diagnostic capture.
};
struct ImageEmbedding {
    int64_t begin = 0;
    // Complete image span, including learned start/end/newline vectors.
    std::vector<float> values;
    int tokens() const { return static_cast<int>(values.size() / 5120); }
};
struct ExecutionStats {
    uint64_t expert_gpu = 0, expert_cpu = 0, h2d_expert_bytes = 0;
    uint64_t encoder_token_layers = 0, decoder_token_layers = 0, kv_only_tokens = 0;
    uint64_t replay_tokens = 0, decoded_tokens = 0;
    double prefill_ms = 0, decode_ms = 0, engram_wait_ms = 0;
    double draft_ms = 0, verify_ms = 0;
    uint64_t proposed_tokens = 0, accepted_tokens = 0;
    uint64_t verify_calls = 0, verify_rows = 0, rejection_replay_calls = 0, rejection_replay_rows = 0;
    uint64_t confidence_rounds = 0, adaptive_available_tokens = 0, adaptive_trimmed_tokens = 0;
    std::array<uint64_t,8> adaptive_length_histogram{};
    uint64_t draft_gpu_rounds = 0, draft_context_rows = 0, draft_context_projection_rows = 0;
    uint64_t draft_fused_moe_calls = 0, draft_batch_projection_calls = 0;
    uint64_t phase_borrow_bytes=0, phase_backup_bytes=0, phase_encoder_h2d_bytes=0, phase_restore_h2d_bytes=0;
    uint64_t phase_replay_reuse_bytes=0, phase_dynamic_h2d_bytes=0, phase_dynamic_d2h_bytes=0, phase_migrations=0;
    uint64_t phase_ready_waits=0;
    double phase_prepare_ms=0, phase_host_wait_ms=0, phase_restore_submit_ms=0, phase_dynamic_ms=0;
    uint64_t demand_expert_tokens=0,demand_expert_bytes=0;
    uint64_t cold_expert_groups=0,demand_expert_groups=0;
    uint64_t raw_cache_hits=0,raw_cache_d2d_bytes=0;
    uint64_t direct_expert_bytes=0,prefetch_expert_bytes=0,prefetch_unused_bytes=0,prefetch_pruned_bytes=0;
    std::array<double,2> engram_layer_wait_ms{};
    uint64_t shared_graph_captures=0,shared_graph_launches=0;
};
struct Distribution {
    std::vector<int32_t> ids;
    std::vector<double> probabilities;
    std::vector<float> proposal_probabilities;
};
Distribution distribution(const float* logits,int vocab,const kernels::SamplerParams& params,const std::vector<int>& history);
void validate_sampling(const kernels::SamplerParams& params);
float draw_uniform(uint64_t seed,uint64_t counter);
int sample(const Distribution& probabilities,float uniform);
struct DraftProposal {
    std::vector<int> tokens;
    std::vector<Distribution> distributions;
    std::vector<float> confidence;
};
class Session;
class Runtime {
public:
    Runtime(const std::string& main, const RuntimeOptions& options,
            const std::string& draft = {}, const std::string& vision = {});
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    std::unique_ptr<Session> create_session();
    void write_ledger(std::ostream& out) const;
    void write_expert_heat(std::ostream& out) const;
    void save_learned_profile();
    const Model& model() const;
    ImageEmbedding encode_image(const std::vector<float>& patches,int height,int width,int64_t begin);
private:
    friend class Session;
    friend struct PrefetchBlockBenchmark;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class Session {
public:
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    // Prompt boundary is fixed by this call. Generated suffixes always execute all 40 layers.
    std::vector<float> prefill(const std::vector<int>& tokens, bool ced,
                               const std::vector<ImageEmbedding>& images = {});
    // Restore the fixed original prompt, then execute every generated suffix token through all 40 layers.
    std::vector<float> resume_history(const std::vector<int>& tokens, int64_t original_prompt_length, bool ced,
                                     const std::vector<ImageEmbedding>& images = {}, const std::string& checkpoint_path = {});
    std::vector<float> forward(const std::vector<int>& tokens);
    std::vector<float> decode(int token) { return forward({token}); }
    int64_t position() const;
    int64_t original_prompt_length() const;
    ExecutionStats stats() const;
    void cancel();
    // Transactions retain SWA/partial/history and owner watermarks; appended global rows are discarded on restore.
    void begin_transaction();
    void rollback_transaction();
    void commit_transaction();
    // Keep an already-verified prefix, including its anchor, without another forward.
    void commit_verified_prefix(int keep,const std::vector<float>& verified_logits);
    void save_prefix(const std::string& path) const;
    void load_prefix(const std::string& path, const std::vector<int>& expected_tokens);
    DraftProposal propose(int first_token,const kernels::SamplerParams& sampling);
    std::vector<int> generate(const std::vector<float>& initial_logits,int maximum,const std::vector<int>& eos,
                              const kernels::SamplerParams& sampling,bool speculative,
                              const std::function<void(int)>& on_token = {});
private:
    friend class Runtime;
    friend struct PrefetchBlockBenchmark;
    struct Impl;
    explicit Session(Runtime::Impl& runtime);
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::models::deepseek41
