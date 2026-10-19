#pragma once
#include "runtime_internal.hpp"
#include "adaptive_policy.hpp"
#include <future>

namespace strata::models::deepseek41 {
// Single-slot decode exchanges. Only the inference thread changes residency/offsets;
// the worker copies host bytes, and the independent CUDA stream moves expert blobs.
class AdaptiveCache {
public:
    static uint64_t staging_bytes(const std::array<uint64_t,40>& sizes,uint64_t budget);
    AdaptiveCache(core::ExpertCache& cache,std::vector<ExpertKey>& hot,std::vector<uint64_t>& offsets,
                  uint8_t* ram,const std::array<uint64_t,40>& sizes,uint64_t budget,bool trace,bool grouped=false,
                  bool pinned_ram=false,bool huge_pinned=false);
    ~AdaptiveCache();
    bool resident(int layer,int expert)const;
    const uint8_t* cpu(int layer,int expert)const;
    void tick(int tokens,const std::vector<double>& counts,cudaStream_t readers,ExecutionStats& stats);
    // Actual target forward work, including rejected verification rows. Never rolled back.
    void observe_window(int tokens,const std::vector<double>& counts);
    // Progress existing committed work at a drained CPU layer boundary; never select or observe routes.
    void progress(cudaStream_t readers,ExecutionStats& stats);
    void begin_request(cudaStream_t readers,ExecutionStats& stats);
    void drain(cudaStream_t readers,ExecutionStats& stats);
    uint64_t staging_bytes()const{return staging_->bytes;}
    void write_status(std::ostream& out)const;
    const std::vector<double>& learned_heat()const{return learned_;}
private:
    friend struct PrefetchBlockBenchmark;
    enum class State { idle,saving,loading,committing };
    struct Swap : AdaptiveExchange {
        uint8_t* source=nullptr;
        uint8_t* saved=nullptr;
        size_t spare=0;
        bool direct=false,rotate=false;
    };
    core::ExpertCache& cache_;
    std::vector<ExpertKey>& hot_;
    std::vector<uint64_t>& offsets_;
    uint8_t* ram_;
    std::array<uint64_t,40> sizes_;
    uint64_t budget_,stride_,capacity_,rows_=0,since_pick_=0,batches_=0,published_=0;
    bool trace_,grouped_,pinned_ram_,rotation_eligible_=false,aligned_=true;
    uint64_t windows_=0,direct_bytes_=0,rotated_=0,avoided_prepare_=0,avoided_commit_=0;
    State state_=State::idle;
    Stream copies_;
    Event readers_done_,copies_done_;
    std::unique_ptr<Pinned> staging_;
    std::vector<double> heat_,previous_,learned_;
    std::vector<uint64_t> generations_;
    std::vector<const uint8_t*> overrides_;
    // Non-owning views. Original RAM/staging allocations remain alive until all readers drain.
    std::vector<uint8_t*> sources_;
    std::array<std::vector<uint8_t*>,2> spares_;
    std::vector<uint8_t> unavailable_;
    std::vector<Swap> swaps_;
    // Enabled only by the controlled test harness: inverse exchanges restore
    // exact GPU slots AND RAM owners between A/B/A arms without re-registering RAM.
    bool record_exchanges_=false;
    std::vector<Swap> committed_exchanges_;
    std::future<double> host_job_;
    double host_copy_ms_=0;
    uint8_t* saved(size_t i)const{return staging_->data+i*stride_;}
    uint8_t* incoming(size_t i)const{return staging_->data+(capacity_+i)*stride_;}
    void advance(cudaStream_t readers,ExecutionStats& stats,bool wait);
    void choose(cudaStream_t readers,ExecutionStats& stats);
    void validate(const Swap& exchange)const;
    void observe(int tokens,const std::vector<double>& counts,bool window);
    void commit_owners();
};
} // namespace strata::models::deepseek41
