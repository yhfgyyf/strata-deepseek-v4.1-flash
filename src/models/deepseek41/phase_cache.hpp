#pragma once
#include "runtime_internal.hpp"

namespace strata::models::deepseek41 {
// Single-request candidate: the stable core stays in ExpertCache. This owns
// only the explicitly borrowed byte region and ordinary-RAM decode backups.
class PhaseCache {
public:
    PhaseCache(PhasePlan plan,std::function<const uint8_t*(ExpertKey)> cold,cudaStream_t upload,uint64_t migration,
               bool pageable=false,bool trace=false);
    ~PhaseCache();
    const PhasePlan& plan()const{return plan_;}
    uint64_t gpu_bytes()const{return plan_.bytes;}
    uint64_t backup_bytes()const{return plan_.bytes;}
    uint64_t staging_bytes()const{return staging_->bytes;}
    uint64_t pinned_staging_bytes()const{return staging_->pinned?staging_->bytes:0;}
    void initialize(size_t slot,const uint8_t* source);
    const uint8_t* cpu(ExpertKey key)const;
    const uint8_t* gpu(ExpertKey key,cudaStream_t stream,ExecutionStats& stats,int phase);
    bool begin_prefill(int remaining,int chunk,bool ced,cudaStream_t stream,ExecutionStats& stats);
    void encoder_layer_done(int layer,cudaStream_t stream,ExecutionStats& stats);
    void finish_prefill(cudaStream_t stream,ExecutionStats& stats);
    void observe(int layer,const int* ids,int tokens,int phase);
    void adjust_decode(int tokens,cudaStream_t stream,ExecutionStats& stats);
    int until_adjustment()const;
    std::vector<double> snapshot()const;
    void restore_snapshot(const std::vector<double>& saved);
    void write_heat(std::ostream& out)const;
private:
    struct Slot {
        ExpertKey key{-1,-1};
        bool pending=false,initialized=false,restored=false,replay_used=false;
        uint64_t moved_at=0;
        Event ready;
    };
    PhasePlan plan_;
    std::function<const uint8_t*(ExpertKey)> cold_;
    cudaStream_t upload_;
    uint64_t migration_=0,stage_stride_=0,stage_index_=0,decode_rows_=0,adjustment_rows_=0;
    bool borrowed_=false,trace_=false;
    std::unique_ptr<core::DeviceArena> arena_;
    std::unique_ptr<uint8_t[]> backup_;
    std::unique_ptr<Pinned> staging_;
    std::array<Event,8> stage_ready_;
    std::array<bool,8> stage_used_{};
    std::array<Event,20> consumed_;
    std::array<bool,20> released_{};
    Event boundary_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<int> original_,resident_,temporary_;
    std::vector<uint8_t> backup_valid_,core_;
    std::array<std::vector<double>,3> counts_;
    std::vector<double> heat_;
    std::vector<uint64_t> last_move_;
    uint8_t* address(size_t slot)const;
    uint8_t* stage(ExecutionStats& stats,size_t& index);
    void protect_readers(cudaStream_t stream);
    void backup(size_t slot,ExecutionStats& stats);
    const uint8_t* host(ExpertKey key)const;
    void upload(size_t slot,ExpertKey key,ExecutionStats& stats,bool restore);
    void restore_slot(size_t slot,ExecutionStats& stats);
};
} // namespace strata::models::deepseek41
