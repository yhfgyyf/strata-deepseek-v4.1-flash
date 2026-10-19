#include "strata/models/deepseek41/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <ostream>
#include <set>
#include <stdexcept>

namespace strata::models::deepseek41 {
namespace {
const MetaValue& meta(const GgufFile& f, const std::string& key) {
    const auto* v = f.get(key);
    if (!v) throw std::runtime_error("DeepSeek: missing metadata " + key);
    return *v;
}
uint64_t integer(const GgufFile& f, const std::string& key) {
    const auto& v = meta(f, key);
    if (v.type != MetaType::U32 && v.type != MetaType::U64 && v.type != MetaType::I32)
        throw std::runtime_error("DeepSeek: expected integer metadata " + key);
    return v.u;
}
void equal(const GgufFile& f, const std::string& key, uint64_t expected) {
    if (integer(f, key) != expected)
        throw std::runtime_error("DeepSeek: unsupported " + key + ", expected " + std::to_string(expected));
}
float number(const GgufFile& f, const std::string& key) {
    const auto& v = meta(f, key);
    if ((v.type != MetaType::F32 && v.type != MetaType::F64) || !std::isfinite(v.f))
        throw std::runtime_error("DeepSeek: expected finite float metadata " + key);
    return static_cast<float>(v.f);
}
std::vector<uint64_t> array(const GgufFile& f, const std::string& key) {
    const auto& v = meta(f, key);
    if (v.type != MetaType::ARRAY || v.count != v.items.size())
        throw std::runtime_error("DeepSeek: missing or truncated array " + key);
    std::vector<uint64_t> a;
    for (const auto& item : v.items) a.push_back(item.u);
    return a;
}
void array_equal(const GgufFile& f, const std::string& key, const std::vector<uint64_t>& expected) {
    if (array(f, key) != expected) throw std::runtime_error("DeepSeek: unsupported " + key);
}
std::string block(int l, const char* suffix) { return "blk." + std::to_string(l) + "." + suffix; }
std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 32) {
            const char hex[] = "0123456789abcdef";
            out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15];
        } else out += static_cast<char>(c);
    }
    return out + '"';
}
uint64_t checked_add(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b) throw std::overflow_error("DeepSeek: byte count overflow");
    return a + b;
}
}

std::array<LayerOwnership, 40> layer_ownership() {
    std::array<LayerOwnership, 40> result{};
    const std::array<int, 4> owners{2, 8, 14, 20};
    const std::array<int, 8> sources{2, 8, 14, 20, 24, 28, 32, 36};
    int owner = -1, source = -1;
    for (int l = 0; l < 40; ++l) {
        if (std::find(owners.begin(), owners.end(), l) != owners.end()) owner = l;
        if (std::find(sources.begin(), sources.end(), l) != sources.end()) source = l;
        result[l] = {l, l < 2 ? 0 : l < 20 ? 2 : 1, l, owner, source};
    }
    return result;
}

const char* role_name(TensorRole r) {
    switch (r) {
    case TensorRole::dense: return "dense";
    case TensorRole::routed_expert: return "routed_expert";
    case TensorRole::engram: return "engram_ssd";
    case TensorRole::token_map: return "token_map_i32";
    case TensorRole::vision: return "vision";
    case TensorRole::draft_head: return "draft_head";
    }
    throw std::runtime_error("DeepSeek: invalid tensor role");
}

uint64_t available_hot_expert_bytes(uint64_t free, uint64_t ring, uint64_t work,
                                  uint64_t state, uint64_t headroom) {
    const uint64_t reserved = checked_add(checked_add(ring, work), checked_add(state, headroom));
    if (free <= reserved)
        throw std::invalid_argument("DeepSeek: no GPU expert capacity after ring/work/state/headroom");
    return free - reserved;
}

RankedHotAllocation allocate_ranked_hot_experts(const std::array<uint64_t, 40>& sizes,
    const std::vector<std::pair<int32_t, int32_t>>& ranked, uint64_t capacity) {
    if (!capacity || ranked.empty()) throw std::invalid_argument("DeepSeek: profile and expert capacity must be nonempty");
    std::array<uint64_t, 40> aligned_sizes{};
    for (int l = 0; l < 40; ++l) {
        if (!sizes[l]) throw std::invalid_argument("DeepSeek: zero expert bytes");
        if (sizes[l] > uint64_t{INT64_MAX} - 255)
            throw std::overflow_error("DeepSeek: aligned expert slot exceeds signed cache capacity");
        aligned_sizes[l] = (sizes[l] + 255) & ~uint64_t{255};
    }
    // Validate the entire supplied ranking, including entries after the capacity boundary.
    std::array<bool, 40 * 384> seen{};
    for (const auto& [l, e] : ranked) {
        if (l < 0 || l >= 40 || e < 0 || e >= 384 || seen[l * 384 + e])
            throw std::invalid_argument("DeepSeek: invalid or duplicate profile expert");
        seen[l * 384 + e] = true;
    }
    RankedHotAllocation allocation;
    for (const auto& pair : ranked) {
        const uint64_t bytes = aligned_sizes[pair.first];
        if (bytes > capacity - allocation.bytes) break;
        if (bytes > uint64_t{INT64_MAX} - allocation.bytes)
            throw std::overflow_error("DeepSeek: expert cache exceeds signed capacity");
        allocation.bytes += bytes; allocation.experts.push_back(pair);
    }
    return allocation;
}

HotAllocation allocate_hot_experts(const std::array<uint64_t, 40>& sizes,
                                   uint64_t front, uint64_t back, int floor) {
    if (floor < 1 || floor > 384 || front <= back)
        throw std::invalid_argument("DeepSeek: front budget must exceed back budget and floor must be 1..384");
    HotAllocation allocation;
    for (int segment = 0; segment < 2; ++segment) {
        const int begin = segment * 20, end = begin + 20;
        const uint64_t budget = segment ? back : front;
        uint64_t used = 0;
        for (int l = begin; l < end; ++l) {
            if (!sizes[l] || sizes[l] > budget / static_cast<uint64_t>(floor))
                throw std::invalid_argument("DeepSeek: expert floor does not fit segment budget");
            used = checked_add(used, sizes[l] * floor);
            allocation.count[l] = floor;
        }
        if (used > budget) throw std::invalid_argument("DeepSeek: expert floor does not fit segment budget");
        while (true) {
            int chosen = -1;
            for (int l = begin; l < end; ++l)
                if (allocation.count[l] < 384 && sizes[l] <= budget - used &&
                    (chosen == -1 || allocation.count[l] < allocation.count[chosen])) chosen = l;
            if (chosen == -1) break;
            ++allocation.count[chosen];
            used += sizes[chosen];
        }
        (segment ? allocation.back_bytes : allocation.front_bytes) = used;
    }
    for (int l = 0; l < 40; ++l) {
        if (sizes[l] > std::numeric_limits<uint64_t>::max() / 384)
            throw std::overflow_error("DeepSeek: expert byte count overflow");
        allocation.host_bytes = checked_add(allocation.host_bytes, sizes[l] * (384 - allocation.count[l]));
    }
    return allocation;
}

Model::Model(const std::string& main, const std::string& draft, const std::string& vision) {
    main_ = std::make_unique<GgufFile>(main);
    load_main();
    if (!draft.empty()) { draft_ = std::make_unique<GgufFile>(draft); load_draft(); }
    if (!vision.empty()) { vision_ = std::make_unique<GgufFile>(vision); load_vision(); }
}

void Model::load_main() {
    const auto& f = *main_;
    if (meta(f, "general.architecture").s != "deepseek41")
        throw std::runtime_error("DeepSeek: main GGUF must have architecture deepseek41");
    const std::string p = "deepseek41.";
    for (const auto& [key, expected] : std::vector<std::pair<std::string, uint64_t>>{
        {"block_count",40}, {"embedding_length",5120}, {"embedding_length_out",20480},
        {"expert_count",384}, {"expert_used_count",6}, {"expert_feed_forward_length",2304},
        {"expert_shared_count",1}, {"expert_gating_func",4}, {"attention.head_count",64},
        {"attention.head_count_kv",1}, {"attention.key_length",512}, {"attention.value_length",512},
        {"rope.dimension_count",64}, {"attention.q_lora_rank",1280}, {"attention.sliding_window",128},
        {"attention.output_group_count",8}, {"attention.output_lora_rank",1024},
        {"attention.indexer.head_count",32}, {"attention.indexer.key_length",128},
        {"attention.indexer.top_k",512}, {"attention.indexer.candidate_source_layer",20},
        {"attention.indexer.candidate_top_k_blocks",2048}, {"attention.indexer.candidate_block_size",8},
        {"hyper_connection.count",4}, {"hyper_connection.sinkhorn_iterations",20},
        {"engram.ngram_size",4}, {"engram.head_count",8}, {"engram.head_dim",256},
        {"engram.compressed_vocab_size",99092}}) equal(f, p + key, expected);
    array_equal(f, p + "attention.kv_source_layers", {2,8,14,20});
    array_equal(f, p + "attention.index_source_layers", {2,8,14,20,24,28,32,36});
    array_equal(f, p + "engram.layers", {1,14});
    std::vector<uint64_t> ratios;
    for (const auto& l : ownership_) ratios.push_back(l.compression);
    array_equal(f, p + "attention.compress_ratios", ratios);
    if (meta(f, p + "attention.q_norm").u || !meta(f, p + "hyper_connection.single_pass").u ||
        !meta(f, p + "expert_weights_norm").u)
        throw std::runtime_error("DeepSeek: unsupported norm or mHC semantics");
    geometry_.norm_eps = number(f, p + "attention.layer_norm_rms_epsilon");
    geometry_.hc_eps = number(f, p + "hyper_connection.epsilon");
    geometry_.route_scale = number(f, p + "expert_weights_scale");
    geometry_.rope_theta = number(f, p + "rope.freq_base");
    geometry_.compress_rope_theta = number(f, p + "attention.compress_rope_freq_base");
    geometry_.rope_factor = number(f, p + "rope.scaling.factor");
    geometry_.beta_fast = number(f, p + "rope.scaling.yarn_beta_fast");
    geometry_.beta_slow = number(f, p + "rope.scaling.yarn_beta_slow");
    geometry_.original_context = static_cast<int>(integer(f, p + "rope.scaling.original_context_length"));
    geometry_.context = static_cast<int>(integer(f, p + "context_length"));
    if (geometry_.norm_eps <= 0 || geometry_.hc_eps <= 0 || geometry_.route_scale <= 0 ||
        geometry_.rope_theta <= 0 || geometry_.compress_rope_theta <= 0 || geometry_.rope_factor < 1 ||
        geometry_.beta_fast <= geometry_.beta_slow || geometry_.original_context <= 0 || geometry_.context <= 0)
        throw std::runtime_error("DeepSeek: invalid numerical metadata");
    for (const auto& suffix : {"swiglu_clamp_exp", "swiglu_clamp_shexp"}) {
        const auto& v = meta(f, p + suffix);
        if (v.count != 40 || v.items.size() != 40 ||
            std::any_of(v.items.begin(), v.items.end(), [](const auto& x) { return x.f != 10.; }))
            throw std::runtime_error("DeepSeek: unsupported SwiGLU clamp");
    }
    validate_file(f, tensors_, false, false);
    const auto& token = tensors_.at("engram_token_map.weight");
    token_map_.resize(geometry_.vocab);
    std::memcpy(token_map_.data(), token.data, token.bytes);
    for (int32_t id : token_map_)
        if (id < 0 || id >= 99092) throw std::runtime_error("DeepSeek: Engram token map out of range");
    const auto multipliers = array(f, p + "engram.multipliers");
    const auto primes = array(f, p + "engram.primes");
    const auto offsets = array(f, p + "engram.offsets");
    const auto rows = array(f, p + "engram.num_embeddings");
    if (multipliers.size() != 8 || primes.size() != 48 || offsets.size() != 48 || rows.size() != 2)
        throw std::runtime_error("DeepSeek: invalid Engram hash metadata lengths");
    for (uint64_t multiplier : multipliers)
        if (!multiplier || !(multiplier & 1) || multiplier > uint64_t(INT64_MAX) / 99091)
            throw std::runtime_error("DeepSeek: invalid Engram multiplier");
    for (int e = 0; e < 2; ++e) {
        uint64_t end = 0;
        for (int h = 0; h < 24; ++h) {
            const int index = e * 24 + h;
            if (!primes[index] || offsets[index] != end)
                throw std::runtime_error("DeepSeek: invalid Engram hash bucket");
            end = checked_add(end, primes[index]);
        }
        if (end != rows[e] || tensor(block(e ? 14 : 1, "engram_embd.weight")).info->shape[1] != rows[e])
            throw std::runtime_error("DeepSeek: Engram hash buckets exceed table");
    }
    if (integer(f, p + "engram.pad_token_id") >= token_map_.size())
        throw std::runtime_error("DeepSeek: Engram padding token out of range");
}

void Model::load_draft() {
    const auto& f = *draft_;
    if (meta(f, "general.architecture").s != "dflash")
        throw std::runtime_error("DeepSeek: DSpark GGUF must have architecture dflash");
    const std::string p = "dflash.";
    for (const auto& [key, expected] : std::vector<std::pair<std::string, uint64_t>>{
        {"block_count",3}, {"embedding_length",5120}, {"expert_count",128}, {"expert_used_count",3},
        {"attention.head_count",64}, {"attention.key_length",512}, {"attention.q_lora_rank",1280},
        {"expert_feed_forward_length",2304}, {"hyper_connection.count",4}, {"block_size",5}})
        equal(f, p + key, expected);
    array_equal(f, p + "attention.compress_ratios", {0,0,0});
    // Store these raw IDs. Their capture mapping is established by the P8 tensor oracle, never by subtracting one.
    array_equal(f, p + "target_layers", {38,39,40});
    validate_file(f, draft_tensors_, true, false);
}

void Model::load_vision() {
    const auto& f = *vision_;
    if (meta(f, "general.architecture").s != "clip" || meta(f, "clip.projector_type").s != "deepseek4v")
        throw std::runtime_error("DeepSeek: mmproj must be the deepseek4v vision encoder");
    for (const auto& [key, expected] : std::vector<std::pair<std::string, uint64_t>>{
        {"block_count",32}, {"embedding_length",1024}, {"feed_forward_length",2816},
        {"attention.head_count",16}, {"patch_size",14}, {"projection_dim",5120}, {"projector.scale_factor",3}})
        equal(f, "clip.vision." + key, expected);
    validate_file(f, vision_tensors_, false, true);
}
std::unique_ptr<Model> Model::vision_only(const std::string& path) {
    auto model=std::unique_ptr<Model>(new Model());
    model->vision_=std::make_unique<GgufFile>(path);model->load_vision();return model;
}
std::unique_ptr<Model> Model::draft_only(const std::string& path) {
    auto model=std::unique_ptr<Model>(new Model());
    model->draft_=std::make_unique<GgufFile>(path);model->load_draft();return model;
}

void Model::validate_file(const GgufFile& f, Views& views, bool draft, bool vision) {
    const uint64_t available = f.file_size() - f.data_start();
    for (const auto& t : f.tensors()) {
        const uint64_t bytes = tensor_payload_bytes(t);
        if (!bytes || t.offset > available || bytes > available - t.offset || t.offset % f.alignment())
            throw std::runtime_error("DeepSeek: invalid tensor span/type/alignment: " + t.name);
        views.emplace(t.name, TensorView{&t, f.tensor_data(t), bytes,
            bytes / (t.elements() / t.shape[0]), TensorRole::dense});
    }
    // Overlapping tensors are never valid role slices, even if each individually fits the file.
    std::vector<const TensorView*> ordered;
    for (const auto& [name, v] : views) { (void) name; ordered.push_back(&v); }
    std::sort(ordered.begin(), ordered.end(), [](auto* a, auto* b) { return a->info->offset < b->info->offset; });
    for (size_t i = 1; i < ordered.size(); ++i)
        if (ordered[i - 1]->bytes > ordered[i]->info->offset - ordered[i - 1]->info->offset)
            throw std::runtime_error("DeepSeek: overlapping tensor " + ordered[i]->info->name);
    std::set<std::string> consumed;
    auto require = [&](const std::string& name, std::vector<uint64_t> shape, uint32_t type,
                       TensorRole role = TensorRole::dense) {
        auto it = views.find(name);
        if (it == views.end()) throw std::runtime_error("DeepSeek: missing tensor " + name);
        if (it->second.info->shape != shape || (it->second.info->type != type && !(draft&&type==8&&it->second.info->type==2)))
            throw std::runtime_error("DeepSeek: tensor shape/type mismatch: " + name);
        it->second.role = role;
        consumed.insert(name);
    };
    if (vision) {
        for (int l = 0; l < 32; ++l) {
            const std::string p = "v.blk." + std::to_string(l) + ".";
            require(p + "attn_out.bias", {1024}, 0, TensorRole::vision);
            require(p + "attn_out.weight", {1024,1024}, 30, TensorRole::vision);
            require(p + "attn_qkv.bias", {3072}, 0, TensorRole::vision);
            require(p + "attn_qkv.weight", {1024,3072}, 30, TensorRole::vision);
            require(p + "ffn_gate.weight", {1024,2816}, 30, TensorRole::vision);
            require(p + "ffn_up.weight", {1024,2816}, 30, TensorRole::vision);
            require(p + "ffn_down.weight", {2816,1024}, 30, TensorRole::vision);
            require(p + "ln1.weight", {1024}, 0, TensorRole::vision);
            require(p + "ln2.weight", {1024}, 0, TensorRole::vision);
        }
        require("mm.1.weight", {9216,5120}, 30, TensorRole::vision);
        require("mm.2.weight", {5120,5120}, 30, TensorRole::vision);
        require("mm.1.bias", {5120}, 0, TensorRole::vision);
        require("mm.2.bias", {5120}, 0, TensorRole::vision);
        require("v.post_ln.weight", {1024}, 0, TensorRole::vision);
        require("v.patch_embd.bias", {1024}, 0, TensorRole::vision);
        require("v.patch_embd.weight", {14,14,3,1024}, 0, TensorRole::vision);
        for (const char* name : {"v.token_embd.img_end", "v.token_embd.img_start", "v.image_newline"})
            require(name, {5120}, 0, TensorRole::vision);
    } else {
        const int layers = draft ? 3 : 40;
        const uint64_t experts = draft ? 128 : 384;
        for (int l = 0; l < layers; ++l) {
            const uint32_t type = draft ? views.at(block(l,"ffn_gate_exps.weight")).info->type : l < 14 ? 11 : 10;
            if(draft&&type!=39&&type!=10&&type!=11&&type!=42)throw std::runtime_error("DeepSeek: unsupported draft expert type");
            for (const char* name : {"ffn_gate_exps.weight", "ffn_up_exps.weight"})
                require(block(l, name), {5120,2304,experts}, type, TensorRole::routed_expert);
            require(block(l, "ffn_down_exps.weight"), {2304,5120,experts}, type, TensorRole::routed_expert);
            require(block(l, "attn_sinks.weight"), {64}, 0);
            require(block(l, "attn_kv_a_norm.weight"), {512}, 0);
            require(block(l, "attn_q_a_norm.weight"), {1280}, 0);
            require(block(l, "attn_kv.weight"), {5120,512}, 8);
            require(block(l, "attn_output_a.weight"), {4096,8192}, 8);
            require(block(l, "attn_output_b.weight"), {8192,5120}, 8);
            require(block(l, "attn_q_a.weight"), {5120,1280}, 8);
            require(block(l, "attn_q_b.weight"), {1280,32768}, 8);
            require(block(l, "attn_norm.weight"), {5120}, 0);
            require(block(l, "ffn_norm.weight"), {5120}, 0);
            require(block(l, "exp_probs_b.bias"), {experts}, 0);
            if (!draft) require(block(l, "exp_probs_b_vl.bias"), {experts}, 0);
            require(block(l, "ffn_gate_inp.weight"), {5120,experts}, 0);
            for (const char* name : {"ffn_gate_shexp.weight", "ffn_up_shexp.weight"})
                require(block(l, name), {5120,2304}, 8);
            require(block(l, "ffn_down_shexp.weight"), {2304,5120}, 8);
            for (const char* stage : {"attn", "ffn"}) {
                const std::string p = "hc_" + std::string(stage);
                require(block(l, (p + "_base.weight").c_str()), {24}, 0);
                require(block(l, (p + "_fn.weight").c_str()), {20480,24}, 0);
                require(block(l, (p + "_scale.weight").c_str()), {3}, 0);
            }
            if (!draft && ownership_[l].owns_global()) {
                require(block(l, "attn_compressor_norm.weight"), {512}, 0);
                require(block(l, "attn_compressor_kv.weight"), {5120,512}, 8);
                if (ownership_[l].compression > 1)
                    require(block(l, "attn_compressor_gate.weight"), {5120,512}, 8);
                require(block(l, "indexer.k_norm.weight"), {128}, 0);
                require(block(l, "indexer.attn_k.weight"), {512,128}, 8);
            }
            if (!draft && ownership_[l].computes_index()) {
                require(block(l, "indexer.proj.weight"), {5120,32}, 8);
                require(block(l, "indexer.attn_q_b.weight"), {1280,4096}, 8);
            }
            if (!draft && (l == 1 || l == 14)) {
                require(block(l, "engram_embd.weight"), {256,l == 1 ? 384006168u : 384016682u}, 8, TensorRole::engram);
                require(block(l, "engram_q.weight"), {5120,4}, 0);
                require(block(l, "engram_k.weight"), {5120,4}, 0);
                require(block(l, "engram_wkv.weight"), {6144,25600}, 8);
            }
        }
        require("token_embd.weight", {5120,129280}, 8);
        require("output.weight", {5120,129280}, 8);
        require("output_norm.weight", {5120}, 0);
        if (!draft) require("engram_token_map.weight", {129280}, 26, TensorRole::token_map);
        else {
            require("enc.output_norm.weight", {5120}, 0, TensorRole::draft_head);
            require("fc.weight", {15360,5120}, 8, TensorRole::draft_head);
            require("conf_proj.weight", {5376,1}, 0, TensorRole::draft_head);
            for (const char* name : {"markov_w1.weight", "markov_w2.weight"})
                require(name, {256,129280}, 8, TensorRole::draft_head);
        }
    }
    if (consumed.size() != views.size())
        for (const auto& [name, view] : views) {
            (void) view;
            if (!consumed.count(name)) throw std::runtime_error("DeepSeek: unrecognized tensor " + name);
        }
}

const TensorView& Model::tensor(const std::string& name, bool draft, bool vision) const {
    const auto& views = vision ? vision_tensors_ : draft ? draft_tensors_ : tensors_;
    const auto it = views.find(name);
    if (it == views.end()) throw std::runtime_error("DeepSeek: tensor unavailable " + name);
    return it->second;
}
const GgufFile& Model::file(bool draft, bool vision) const {
    const auto* f = vision ? vision_.get() : draft ? draft_.get() : main_.get();
    if (!f) throw std::runtime_error("DeepSeek: requested artifact not loaded");
    return *f;
}
ExpertSlice Model::expert(int layer, int id, bool draft) const {
    if (layer < 0 || layer >= (draft ? 3 : 40) || id < 0 || id >= (draft ? 128 : 384))
        throw std::out_of_range("DeepSeek: expert index out of range");
    const uint64_t count = draft ? 128 : 384;
    const auto& gate = tensor(block(layer, "ffn_gate_exps.weight"), draft);
    const auto& up = tensor(block(layer, "ffn_up_exps.weight"), draft);
    const auto& down = tensor(block(layer, "ffn_down_exps.weight"), draft);
    const uint64_t stride = gate.bytes / count;
    return {gate.data + stride * id, up.data + stride * id, down.data + stride * id,
            gate.info->type, gate.row_bytes, down.row_bytes, stride, stride * 3};
}
std::array<uint64_t, 40> Model::expert_bytes() const {
    std::array<uint64_t, 40> sizes{};
    for (int l = 0; l < 40; ++l) sizes[l] = expert(l, 0).blob_bytes;
    return sizes;
}
void Model::write_inventory(std::ostream& out) const {
    out << "{\"architecture\":\"deepseek41\",\"geometry\":{\"layers\":40,\"hidden\":5120,\"heads\":64},\"files\":[";
    bool first_file = true;
    for (int kind = 0; kind < 3; ++kind) {
        const auto* f = kind == 2 ? vision_.get() : kind == 1 ? draft_.get() : main_.get();
        if (!f) continue;
        if (!first_file) out << ',';
        first_file = false;
        const auto& views = kind == 2 ? vision_tensors_ : kind == 1 ? draft_tensors_ : tensors_;
        out << "{\"path\":" << json_string(f->path()) << ",\"bytes\":" << f->file_size()
            << ",\"data_start\":" << f->data_start() << ",\"tensors\":[";
        bool first = true;
        for (const auto& [name, v] : views) {
            if (!first) out << ',';
            first = false;
            out << "{\"name\":" << json_string(name) << ",\"role\":" << json_string(role_name(v.role))
                << ",\"type\":" << v.info->type << ",\"offset\":" << v.info->offset
                << ",\"bytes\":" << v.bytes << ",\"row_bytes\":" << v.row_bytes << ",\"shape\":[";
            for (size_t d = 0; d < v.info->shape.size(); ++d) {
                if (d) out << ',';
                out << v.info->shape[d];
            }
            out << "]}";
        }
        out << "]}";
    }
    out << "],\"ownership\":[";
    for (const auto& l : ownership_) {
        if (l.layer) out << ',';
        out << "{\"layer\":" << l.layer << ",\"compression\":" << l.compression
            << ",\"local_owner\":" << l.local_owner << ",\"global_owner\":" << l.global_owner
            << ",\"index_source\":" << l.index_source << '}';
    }
    out << "]}\n";
}
} // namespace strata::models::deepseek41
