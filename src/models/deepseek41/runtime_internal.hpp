#pragma once
#include "strata/models/deepseek41/runtime.hpp"
#include "strata/models/deepseek41/attention.hpp"
#include "strata/models/deepseek41/engram.hpp"
#include "strata/models/deepseek41/indexer.hpp"
#include "strata/models/deepseek41/residency.hpp"
#include "strata/core/device.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/graph.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "runtime_kernels.hpp"
#include "host_memory.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <array>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <mutex>
#include <condition_variable>
#include <future>
#include <thread>

namespace strata::models::deepseek41 {
inline void gpu_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("DeepSeek CUDA: ") + cudaGetErrorString(error));
}
inline uint64_t aligned(uint64_t n) { return (n + 255) / 256 * 256; }
template<class T> T* allocate(core::DeviceArena& arena, uint64_t count) {
    if (count > UINT64_MAX / sizeof(T)) throw std::overflow_error("DeepSeek device allocation");
    return static_cast<T*>(arena.alloc(count * sizeof(T)));
}
// The same allocation sequence serves byte planning and actual arena carving.
// Planning returns null pointers; it must never execute a kernel or do pointer arithmetic.
struct WorkspaceLayout {
    core::DeviceArena* arena;
    uint64_t bytes;
    explicit WorkspaceLayout(core::DeviceArena* device=nullptr):arena(device),bytes(device?device->used():0){}
    template<class T> T* alloc(uint64_t count) {
        if(!count)return nullptr;
        if(count>UINT64_MAX/sizeof(T)||bytes>UINT64_MAX-255)throw std::overflow_error("DeepSeek workspace size");
        const uint64_t size=count*sizeof(T),offset=aligned(bytes);
        if(size>UINT64_MAX-offset)throw std::overflow_error("DeepSeek workspace size");
        bytes=offset+size;
        return arena?static_cast<T*>(arena->alloc(size)):nullptr;
    }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { gpu_check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)); }
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
    Stream(const Stream&) = delete;
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { gpu_check(cudaEventCreateWithFlags(&value,cudaEventDisableTiming)); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
};
struct Matrix {
    void* data = nullptr;
    int type = 0;
    int64_t rows = 0, cols = 0;
    uint64_t elements = 0;
    bool packed_bf16 = false;
};
class WeightBank {
public:
    WeightBank(const Model& model, Arithmetic arithmetic, cudaStream_t stream, bool draft = false, bool vision = false,
               bool dense_q8 = false,const std::vector<int32_t>* draft_vocab = nullptr,
               const std::vector<int32_t>* draft_input_vocab = nullptr);
    const Matrix& at(const std::string& name) const { return matrices_.at(name); }
    uint64_t bytes() const { return arena_->used(); }
private:
    std::unique_ptr<core::DeviceArena> arena_;
    std::map<std::string,Matrix> matrices_;
};
void matrix_product(const Matrix& weight,const float* input,float* output,int tokens,cublasHandle_t blas,
                    cudaStream_t stream,uint16_t* bf_input,Arithmetic arithmetic,int rows = 0,
                    int64_t ldx = 0,int64_t ldy = 0,uint64_t weight_offset = 0,bool round_output = true,bool verify_batch = false,
                    int bf_scratch_cols = 0);
struct VisionWork {
    float *patches,*h,*norm,*qkv,*q,*k,*v,*attention,*branch,*gate,*up,*activation,*aligned,*projected,*features,*span;
    explicit VisionWork(core::DeviceArena& arena){WorkspaceLayout layout(&arena);init(layout);}
    explicit VisionWork(WorkspaceLayout& layout){init(layout);}
private:
    void init(WorkspaceLayout& layout);
};
struct Capture;
class PhaseCache;
class AdaptiveCache;
class ExpertPipeline;
struct DraftWork {
    float *main_hidden,*main_x,*h,*pre,*attn_pre,*post,*comb,*mixes,*x,*qr,*q,*kv,*o_low,*branch;
    float *seed_kv,*shared_gate,*shared_up,*shared_act,*shared_out,*route_logits,*route_weights,*parts,*mean;
    float *markov,*bias,*confidence_input,*confidence,*head;
    float* inject_hidden;
    uint8_t* cache;
    uint16_t *q_bf,*kv_bf,*out_bf,*partial;
    float* lse;
    int *ids,*routes;
    int *greedy_ids,*penalty_counts;
    int64_t *slots,*writes;
    explicit DraftWork(core::DeviceArena& arena,int vocab=129280){WorkspaceLayout layout(&arena);init(layout,vocab);}
    explicit DraftWork(WorkspaceLayout& layout,int vocab=129280){init(layout,vocab);}
private:
    void init(WorkspaceLayout& layout,int vocab);
};
struct DraftVocab {
    std::vector<int32_t> ids, inverse;
    explicit DraftVocab(const std::string& path);
    int size()const{return static_cast<int>(ids.size());}
};
class DraftWeights {
public:
    DraftWeights(const Model& model,Arithmetic arithmetic,cudaStream_t stream,const std::string& vocab_path={},const std::string& input_vocab_path={});
    const Matrix& tensor(int layer,const std::string& suffix)const{return dense.at("blk."+std::to_string(layer)+"."+suffix+".weight");}
    const uint8_t* expert(int layer,int id)const;
    uint32_t expert_type(int layer)const{return expert_types_.at(layer);}
    uint64_t projection_bytes(int layer)const{return projection_bytes_.at(layer);}
    uint64_t bytes()const{return dense.bytes()+experts_->used();}
    DraftVocab vocab,input_vocab;
    WeightBank dense;
    int *gpu_vocab_ids=nullptr,*gpu_vocab_inverse=nullptr,*gpu_input_inverse=nullptr;
private:
    std::unique_ptr<core::DeviceArena> experts_;
    std::array<uint64_t,3> layer_offsets_{},projection_bytes_{};
    std::array<uint32_t,3> expert_types_{};
};
struct Workspace;
void draft_matrix_product(const Matrix& m,const float* x,float* out,int tokens,cublasHandle_t blas,
                          cudaStream_t stream,Workspace& work,Arithmetic arithmetic,int rows=0,
                          int64_t ldx=0,int64_t ldy=0,uint64_t offset=0,bool round_output=true);
std::vector<float> vision_forward(WeightBank& weights,VisionWork& work,const std::vector<float>& patches,
                                 int height,int width,Arithmetic arithmetic,cublasHandle_t blas,cudaStream_t stream,
                                 uint16_t* bf_input,int chunk,Capture* capture);
// Adapt GGUF's three disjoint expert slices to Strata's contiguous native blob contract.
class GgufExperts final : public core::ExpertSource {
public:
    GgufExperts(const Model& model, const RuntimeOptions& options, cudaStream_t upload);
    ~GgufExperts();
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    const uint8_t* blob_stable(int64_t layer, int64_t expert) override;
    const uint8_t* pinned_blob(int layer,int expert);
    uint64_t pinned_bytes()const{return ram_.pinned()?ram_bytes_:0;}
    const HostMemoryInfo& host_memory_info()const{return ram_.memory_info();}
    bool transient(int64_t layer, int64_t expert) const override;
    const uint8_t* gpu(int layer,int expert,cudaStream_t stream,ExecutionStats& stats,int phase);
    bool copy_blob(int64_t layer,int64_t expert,uint8_t* out) override;
    const kernels::cpu::NativeFmt& format(int layer) const { return formats_[layer]; }
    uint64_t ram_bytes() const;
    uint64_t gpu_bytes() const;
    uint64_t backup_bytes()const;
    uint64_t phase_staging_bytes()const;
    bool begin_prefill(int remaining,int chunk,bool ced,cudaStream_t stream,ExecutionStats& stats);
    void encoder_layer_done(int layer,cudaStream_t stream,ExecutionStats& stats);
    void finish_prefill(cudaStream_t stream,ExecutionStats& stats);
    void observe_routes(int layer,const int* ids,int tokens,int phase);
    void observe_physical_routes(int layer,const int* ids,int tokens);
    void observe_decode_window(int tokens);
    void adjust_decode(int tokens,cudaStream_t stream,ExecutionStats& stats);
    void progress_decode(cudaStream_t stream,ExecutionStats& stats);
    int until_adjustment()const;
    std::vector<double> heat_snapshot()const;
    void restore_heat(const std::vector<double>& snapshot);
    void write_heat(std::ostream& out)const;
    void write_layout(std::ostream& out)const;
    void save_learned_profile(const std::string& path,cudaStream_t stream);
    std::vector<ExpertKey> hot_pairs() const;
    uint64_t adaptive_staging_bytes()const;
    void release_mapped(int layer, int expert) const;
private:
    friend struct PrefetchBlockBenchmark;
    const Model& model_;
    core::ExpertCache cache_;
    std::array<kernels::cpu::NativeFmt,40> formats_;
    std::vector<std::pair<int32_t,int32_t>> hot_;
    std::vector<uint64_t> offsets_;
    HostArena ram_;
    uint64_t ram_bytes_ = 0;
    std::vector<uint8_t> assembled_;
    std::unique_ptr<PhaseCache> phase_;
    std::unique_ptr<AdaptiveCache> adaptive_;
    std::array<std::vector<double>,3> route_counts_;
    std::array<uint64_t,3> resident_hits_{};
    std::vector<double> physical_counts_;
    std::map<std::string,int> fds_;
};
struct Capture {
    std::filesystem::path root;
    std::ofstream manifest;
    uint64_t serial = 0;
    std::vector<std::string> stages;
    explicit Capture(const std::string& path);
    void put(const std::string& name, const float* data, const std::vector<int64_t>& shape, cudaStream_t stream);
    void put_host(const std::string& name,const std::vector<float>& values,const std::vector<int64_t>& shape);
    void tokens(const std::vector<int>& ids);
    bool enabled() const { return manifest.is_open(); }
};
// Draft-only bridge: chronological hidden captures are supplied by the caller.
// This does not establish which main-model layers should produce those captures.
struct Workspace;
struct DraftKvCache {
    std::array<uint8_t*,3> packed{};
    std::array<float*,3> values{};
    int64_t position = -1;
    const DraftWeights* weights = nullptr;
    explicit DraftKvCache(core::DeviceArena& arena,bool floating);
};
struct DraftContext {
    const Model& model;
    Arithmetic arithmetic;
    DraftWeights& weights;
    DraftWork& draft;
    Workspace& work;
    cublasHandle_t blas;
    cudaStream_t stream;
    Capture* capture = nullptr;
    std::function<void()> check_cancel;
    bool gpu_greedy = false;
    ExecutionStats* stats = nullptr;
    DraftKvCache* kv_cache = nullptr;
    bool fused_moe = false, batch_projection = false, adaptive_confidence = false;
    int draft_tokens = 5;
};
const char* draft_cache_label(Arithmetic arithmetic);
DraftProposal draft_forward(DraftContext& context,const float* hidden,int window,int64_t position,
                            int first,const std::vector<int>& history,const kernels::SamplerParams& sampling);
struct LocalCache {
    uint8_t* packed = nullptr;
    float* values = nullptr;
    int64_t position = -1;
    int count = 0;
};
struct GlobalCache {
    uint8_t *packed = nullptr, *index_packed = nullptr;
    float *values = nullptr, *index_values = nullptr;
    float *partial_kv = nullptr, *partial_gate = nullptr;
    int64_t position = 0, rows = 0;
    bool partial = false;
};
struct Workspace {
    std::unique_ptr<core::DeviceArena> arena;
    int max_tokens = 0, max_rows = 0, max_logit_tokens = 0,max_bf_input_cols = 32768;
    float *h = nullptr, *pre = nullptr, *attn_pre = nullptr, *post = nullptr, *comb = nullptr, *mixes = nullptr;
    float *x = nullptr, *branch = nullptr, *qr = nullptr, *q = nullptr, *kv = nullptr, *latent = nullptr;
    float *comp_raw = nullptr, *comp_gate = nullptr, *index_q = nullptr, *index_w = nullptr, *index_k = nullptr;
    float *engram_emb = nullptr, *engram_kv = nullptr, *image_values = nullptr, *o_low = nullptr;
    float *shared_gate = nullptr, *shared_up = nullptr, *shared_act = nullptr, *shared_out = nullptr;
    float *parts = nullptr, *group_x = nullptr, *group_gate = nullptr, *group_up = nullptr, *group_act = nullptr, *group_out = nullptr;
    float* logits = nullptr;
    float *dequant = nullptr, *scores = nullptr, *block_scores = nullptr;
    uint16_t *expert_gate = nullptr, *expert_up = nullptr, *expert_down = nullptr, *bf_input = nullptr;
    uint16_t* engram_input = nullptr; // Dedicated auxiliary-stream scratch, charged to the work arena.
    uint16_t* dense_weight = nullptr;
    uint8_t* dense_input_q8 = nullptr;
    std::array<uint16_t*,3> expert_ahead{};
    uint16_t *q_bf = nullptr, *kv_bf = nullptr, *out_bf = nullptr, *partial_out = nullptr;
    float* lse = nullptr;
    uint8_t *engram_packed = nullptr, *temp_swa = nullptr, *image = nullptr;
    int32_t* candidate_blocks = nullptr;
    float* temp_swa_float = nullptr;
    int *ids = nullptr, *group_rows = nullptr, *group_slots = nullptr, *route_ids = nullptr;
    int* route_handoff = nullptr;
    float *route_weights = nullptr, *group_weights = nullptr;
    int64_t *swa_slots = nullptr, *global_slots = nullptr, *write_slots = nullptr;
    IndexSortBuffers sort, block_sort;
    size_t sort_capacity = 0, block_sort_capacity = 0;
    uint64_t index_elements = 0, index_gpu_bytes = 0;
    int index_max_queries = 0;
    uint8_t* index_mx = nullptr;
    std::unique_ptr<Pinned> host;
    float *host_x = nullptr, *host_parts = nullptr, *host_weights = nullptr;
    int* host_ids = nullptr;
    int *host_group_rows = nullptr, *host_group_slots = nullptr;
    float* host_group_weights = nullptr;
    // Graphs own only static weight/workspace addresses, never Session/KV state.
    // Destruction precedes the arena and pinned host buffers they reference.
    std::array<std::array<std::array<core::CapturedGraph,8>,40>,2> decode_segments;
    std::array<std::array<std::array<bool,8>,40>,2> decode_segment_disabled{};
    std::array<uint64_t,2> segment_captures{},segment_replays{},segment_fallbacks{},segment_nodes{};
    uint64_t segment_cache_bytes=0,segment_capture_ns=0;
    Workspace(uint64_t budget, int tokens, int context,bool compact_logits = false,bool compact_workspace = false,bool dspark = false);
    static uint64_t bytes_needed(int tokens,const RuntimeOptions& options,bool vision=false);
    void extras(WorkspaceLayout& layout,const RuntimeOptions& options);
private:
    Workspace()=default;
    void layout(WorkspaceLayout& memory,int tokens,int context,bool compact_logits,bool compact_workspace,bool dspark);
};
class EngramPipeline {
public:
    EngramPipeline(Workspace& work,Arithmetic arithmetic,bool trace = false,bool huge_pinned = true,uint32_t packed_type = 8);
    ~EngramPipeline();
    void start(std::shared_ptr<EngramBatch> ticket,uint64_t sequence,int count,const Matrix& first,const Matrix& second);
    void wait(int table,uint64_t generation,uint64_t sequence,cudaStream_t consumer);
    void consume(int table,uint64_t sequence,cudaStream_t consumer);
    void stop();
private:
    Workspace& work_;
    Arithmetic arithmetic_;
    bool trace_;
    uint32_t packed_type_;
    uint64_t row_bytes_;
    Stream stream_;
    std::array<Event,2> ready_,consumed_;
    Event copied_;
    std::unique_ptr<Pinned> host_;
    cublasHandle_t blas_ = nullptr;
    std::shared_ptr<EngramBatch> ticket_;
    std::thread worker_;
    std::array<Matrix,2> matrices_;
    int count_=0,device_id_=0;
    bool pending_=false,working_=false;
    void run();
    std::mutex mutex_;
    std::condition_variable changed_;
    std::array<uint64_t,2> issued_{},used_{};
    uint64_t sequence_ = 0;
    bool stopped_ = false;
    std::exception_ptr error_;
};
class DecodeRawCache;
struct Runtime::Impl {
    Model model;
    RuntimeOptions options;
    Stream stream, upload, dequant_stream;
    Stream hc_stream;
    Event hc_ready,hc_done;
    cublasHandle_t blas = nullptr;
    std::unique_ptr<WeightBank> weights;
    std::unique_ptr<GgufExperts> experts;
    std::unique_ptr<EngramReader> engram;
    std::unique_ptr<kernels::cpu::ExpertPool> pool;
    std::unique_ptr<Workspace> work;
    std::array<std::array<core::CapturedGraph,8>,40> shared_graphs;
    std::unique_ptr<WeightBank> vision_weights;
    std::unique_ptr<VisionWork> vision_work;
    std::unique_ptr<DraftWeights> draft_weights;
    std::unique_ptr<DraftWork> draft_work;
    unsigned decode_segment_mask = 0; // Experimental bits: input=1, output/router=2.
    int active_sessions = 0;
    std::unique_ptr<core::DeviceArena> ring;
    std::unique_ptr<Pinned> host_ring;
    std::unique_ptr<ExpertPipeline> expert_pipeline;
    std::unique_ptr<DecodeRawCache> decode_raw_cache;
    uint64_t expert_generation = 0;
    // Test-only observation at shared completion / host-plan handoff; normally empty.
    std::function<void()> moe_shared_host_ready;
    bool prefetch_active=false;
    size_t prefetch_cursor=0,prefetch_total=0;
    std::vector<uint64_t> prefetch_sizes;
    std::array<std::array<size_t,384>,40> prefetch_positions;
    std::array<size_t,41> prefetch_offsets{};
    std::array<Event,2> ring_ready, ring_consumed;
    std::array<bool,2> ring_used{};
    std::array<Event,2> expert_ready,expert_used;
    Event expert_begin,route_ready;
    uint64_t ring_slot_bytes = 0;
    int ring_slots = 0;
    std::mutex execution_mutex;
    Impl(const std::string& main,const RuntimeOptions& options,const std::string& draft,const std::string& vision);
    ~Impl();
    void matmul(const Matrix& weight,const float* x,float* out,int tokens,Workspace& work,
                int rows = 0,int64_t ldx = 0,int64_t ldy = 0,uint64_t weight_offset = 0,bool round_output = true);
    void round(float* x,uint64_t n);
    const Matrix& tensor(int layer,const std::string& suffix) const {
        const auto name=model.ple_engram()&&suffix=="engram_wkv"?"engram_kv":suffix;
        return weights->at("blk." + std::to_string(layer) + "." + name + ".weight");
    }
    void moe(int layer,int tokens,Workspace& work,ExecutionStats& stats,int phase,std::vector<int>* deferred_routes=nullptr,bool routes_ready=false);
    bool short_gpu_moe(int tokens,int phase) const;
    bool compact_routes(int tokens,int phase) const;
    void route(int layer,int tokens,Workspace& work,int phase);
    void decode_segment(int part,int layer,int tokens,Workspace& work,bool eligible,const std::function<void()>& body);
    void begin_expert_prefetch(int first,int last,ExecutionStats& stats);
    void prune_expert_prefetch(int layer,const std::array<bool,384>& selected,ExecutionStats& stats);
    void retire_expert_prefetch(size_t until,ExecutionStats& stats);
    void end_expert_prefetch();
    void cancel_expert_prefetch();
};
// Diagnostic-only callbacks; ordinary request execution passes no hooks.
struct BlockBenchmarkHooks {
    std::function<void()> attention_begin,last_block_end;
};
struct Session::Impl {
    Runtime::Impl& runtime;
    Workspace& work;
    std::unique_ptr<core::DeviceArena> state;
    std::array<LocalCache,40> local;
    std::array<GlobalCache,40> global;
    int64_t position = 0, original_prompt = 0, encoder_position = 0, decoder_position = 0;
    uint64_t generation = 1, batch_epoch = 0, index_epoch = 0, candidate_epoch = 0;
    int index_source = -1;
    bool ced = false, transaction_active = false;
    bool last_encoder_chunk = false;
    std::atomic<bool> cancelled{false};
    EngramHistory history;
    std::vector<int> tokens;
    std::vector<ImageEmbedding> images;
    std::vector<float> last_logits;
    bool greedy_capture=false;
    std::vector<int> last_greedy;
    float *boundary_h = nullptr, *boundary_pre = nullptr;
    float *draft_hidden = nullptr, *snapshot_hidden = nullptr;
    int64_t draft_capture_position = 0;
    std::unique_ptr<DraftKvCache> draft_kv;
    uint8_t* snapshot_local = nullptr;
    float* snapshot_partial = nullptr;
    float* verified_partial = nullptr;
    std::array<int,40> partial_owner{};
    std::array<std::vector<int>,40> transaction_routes;
    bool transaction_routes_deferred = false;
    int64_t boundary_begin = 0, boundary_end = 0;
    static constexpr int replay_capacity = 1152;
    ExecutionStats counters;
    Capture capture;
    std::shared_ptr<EngramBatch> engram_ticket;
    std::shared_ptr<EngramBatch> engram_lookahead;
    EngramHistory lookahead_after;
    int64_t lookahead_start = -1;
    std::unique_ptr<EngramPipeline> engram_pipeline;
    struct Snapshot;
    std::unique_ptr<Snapshot> transaction;
    Impl(Runtime::Impl& runtime);
    ~Impl();
    void reset_local(int first,int64_t at);
    void check_cancel();
    void prefetch_engram(const std::vector<int>& prompt,int64_t start);
    void init(const std::vector<int>& ids,int64_t start);
    void image_mask(int count,int64_t start);
    void save_boundary(int count,int64_t start);
    void load_boundary(int count,int64_t start);
    void capture_hidden(int layer,int count,int64_t start);
    void eng(int layer,int count);
    void compression(int layer,int count,int64_t start,bool only);
    void capture_partial(int layer,int count,int64_t start);
    void commit_routes(int keep);
    void attention_input(int layer,int count,const std::function<void()>& before_q = {});
    void attention_output(int layer,int count,bool with_routes);
    void attention(int layer,int count,int64_t start);
    void layers(int count,int64_t start,int first,int last,bool prefill = false,const BlockBenchmarkHooks* hooks = nullptr);
    std::vector<float> logits(int count,bool all = false);
    void dump(const std::string& name,const float* data,const std::vector<int64_t>& shape);
};
} // namespace strata::models::deepseek41
