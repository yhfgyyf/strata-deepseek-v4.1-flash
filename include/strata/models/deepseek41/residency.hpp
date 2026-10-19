#pragma once
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::models::deepseek41 {
using ExpertKey = std::pair<int32_t,int32_t>;
struct PhaseSlice {
    ExpertKey key;
    uint64_t offset=0,bytes=0;
    // Highest encoder layer whose temporary bytes intersect this decode slot.
    // -1 means the slot is never overwritten by the prefill loan.
    int release_after=-1;
};
struct PhasePlan {
    std::vector<ExpertKey> core;
    std::vector<PhaseSlice> decode,encoder;
    uint64_t bytes=0,encoder_bytes=0;
};
PhasePlan make_phase_plan(const std::array<uint64_t,40>& sizes,const std::vector<ExpertKey>& decode_targets,
                          const std::vector<ExpertKey>& prefill_rank,uint64_t byte_limit,int core_floor=32);
} // namespace strata::models::deepseek41
