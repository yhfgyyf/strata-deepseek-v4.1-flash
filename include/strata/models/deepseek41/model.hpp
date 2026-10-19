#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <array>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::models::deepseek41 {

struct Geometry {
    int layers = 40, hidden = 5120, experts = 384, top_k = 6, ff = 2304;
    int heads = 64, head_dim = 512, rope_dim = 64, q_rank = 1280;
    int o_groups = 8, o_rank = 1024, hc = 4, sinkhorn_iters = 20;
    int index_heads = 32, index_dim = 128, index_top_k = 512, window = 128;
    int vocab = 129280, candidate_blocks = 2048, candidate_block_size = 8;
    float norm_eps = 1e-20f, hc_eps = 1e-6f, route_scale = 1.5f, clamp = 10.f;
    float rope_theta = 10000.f, compress_rope_theta = 160000.f, rope_factor = 16.f;
    float beta_fast = 32.f, beta_slow = 1.f;
    int original_context = 65536, context = 1048576;
};

struct LayerOwnership {
    int layer = 0, compression = 0, local_owner = 0, global_owner = -1, index_source = -1;
    bool owns_global() const { return global_owner == layer; }
    bool computes_index() const { return index_source == layer; }
};

std::array<LayerOwnership, 40> layer_ownership();

enum class TensorRole { dense, routed_expert, engram, token_map, vision, draft_head };
const char* role_name(TensorRole role);

struct TensorView {
    const TensorInfo* info = nullptr;
    const uint8_t* data = nullptr;
    uint64_t bytes = 0, row_bytes = 0;
    TensorRole role = TensorRole::dense;
};

struct ExpertSlice {
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    uint32_t type = 0;
    uint64_t gu_row_bytes = 0, down_row_bytes = 0, projection_bytes = 0, blob_bytes = 0;
};

struct HotAllocation {
    std::array<int, 40> count{};
    uint64_t front_bytes = 0, back_bytes = 0, host_bytes = 0;
};
HotAllocation allocate_hot_experts(const std::array<uint64_t, 40>& bytes_per_expert,
                                   uint64_t front_budget, uint64_t back_budget, int floor = 32);
struct RankedHotAllocation {
    std::vector<std::pair<int32_t, int32_t>> experts;
    uint64_t bytes = 0;
};
// Preserve rank priority: stop at the first blob that does not fit, without per-layer floors.
RankedHotAllocation allocate_ranked_hot_experts(const std::array<uint64_t, 40>& bytes_per_expert,
    const std::vector<std::pair<int32_t, int32_t>>& ranked, uint64_t capacity);
uint64_t available_hot_expert_bytes(uint64_t free, uint64_t ring, uint64_t work,
                                  uint64_t state, uint64_t headroom);

// GGUF mappings stay read-only. Views and slices are valid for the lifetime of this model.
class Model {
public:
    Model(const std::string& main, const std::string& draft = {}, const std::string& vision = {});
    static std::unique_ptr<Model> vision_only(const std::string& path);
    static std::unique_ptr<Model> draft_only(const std::string& path);
    bool has_vision() const { return vision_ != nullptr; }
    bool has_draft() const { return draft_ != nullptr; }
    const Geometry& geometry() const { return geometry_; }
    const auto& ownership() const { return ownership_; }
    const auto& tensors() const { return tensors_; }
    const TensorView& tensor(const std::string& name, bool draft = false, bool vision = false) const;
    ExpertSlice expert(int layer, int expert, bool draft = false) const;
    std::array<uint64_t, 40> expert_bytes() const;
    const std::vector<int32_t>& token_map() const { return token_map_; }
    const GgufFile& file(bool draft = false, bool vision = false) const;
    void write_inventory(std::ostream& out) const;

private:
    Model() = default;
    using Views = std::map<std::string, TensorView>;
    void load_main();
    void load_draft();
    void load_vision();
    void validate_file(const GgufFile& file, Views& views, bool draft, bool vision);
    Geometry geometry_;
    std::array<LayerOwnership, 40> ownership_ = layer_ownership();
    std::unique_ptr<GgufFile> main_, draft_, vision_;
    Views tensors_, draft_tensors_, vision_tensors_;
    std::vector<int32_t> token_map_;
};

} // namespace strata::models::deepseek41
