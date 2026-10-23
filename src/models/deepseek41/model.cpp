#include "strata/models/deepseek41/model.hpp"
#include "strata/models/deepseek41/iq3xxs_digests.hpp"
#include "strata/models/deepseek41/q2k_xl_digests.hpp"

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
    if (v.type != MetaType::U16 && v.type != MetaType::U32 && v.type != MetaType::U64 && v.type != MetaType::I32)
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
    open_main(main, false);
    load_main();
    if (!draft.empty()) { draft_ = std::make_unique<GgufFile>(draft); load_draft(); }
    if (!vision.empty()) { vision_ = std::make_unique<GgufFile>(vision); load_vision(); }
}

void Model::open_main(const std::string& path, bool operator_only) {
    const auto paths = gguf_split_paths(path, !operator_only);
    main_ = std::make_unique<GgufFile>(paths.front());
    ple_engram_ = main_->get("deepseek41.ple.layers") != nullptr;
    if (paths.size() > 1) {
        equal(*main_, "split.no", 0);
        equal(*main_, "split.count", paths.size());
    }
    for (size_t i = 1; i < paths.size(); ++i) {
        if (!std::filesystem::exists(paths[i])) {
            if (!operator_only || !ple_engram_ || i + 1 != paths.size())
                throw std::runtime_error("DeepSeek: missing non-Engram shard " + paths[i]);
            main_complete_ = false;
            continue;
        }
        auto shard = std::make_unique<GgufFile>(paths[i]);
        equal(*shard, "split.no", i);
        equal(*shard, "split.count", paths.size());
        equal(*shard, "split.tensors.count", integer(*main_, "split.tensors.count"));
        main_shards_.push_back(std::move(shard));
    }
}

std::unique_ptr<Model> Model::operator_weights_only(const std::string& path) {
    auto model = std::unique_ptr<Model>(new Model);
    model->open_main(path, true);
    model->load_main();
    return model;
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
        {"engram.compressed_vocab_size",99092}}) equal(f, p + key, expected);
    equal(f, p + (ple_engram_ ? "ple.ngram_size" : "engram.ngram_size"), 4);
    equal(f, p + (ple_engram_ ? "ple.heads_per_ngram" : "engram.head_count"), 8);
    equal(f, p + (ple_engram_ ? "embedding_length_per_layer_input" : "engram.head_dim"), 256);
    array_equal(f, p + "attention.kv_source_layers", {2,8,14,20});
    array_equal(f, p + (ple_engram_ ? "attention.indexer.source_layers" : "attention.index_source_layers"), {2,8,14,20,24,28,32,36});
    array_equal(f, p + (ple_engram_ ? "ple.layers" : "engram.layers"), {1,14});
    std::vector<uint64_t> ratios;
    for (const auto& l : ownership_) ratios.push_back(l.compression);
    const auto actual_ratios = array(f, p + "attention.compress_ratios");
    if (ple_engram_ && actual_ratios.size() == 43) ratios.insert(ratios.end(), 3, 0);
    if (actual_ratios != ratios) throw std::runtime_error("DeepSeek: unsupported compression ratios");
    // The PLE converter's DeepSeek41 graph explicitly omits per-head Q norm
    // and implements single-pass mHC (llama.cpp PR30817, b72050d).
    const auto* qnorm = f.get(p + "attention.q_norm");
    const auto* single = f.get(p + "hyper_connection.single_pass");
    if ((!ple_engram_ && (!qnorm || !single)) || (qnorm && qnorm->u) || (single && !single->u) ||
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
    token_map_.resize(geometry_.vocab);
    if (ple_engram_) {
        const auto ids = array(f, p + "engram.token_map");
        if (ids.size() != token_map_.size()) throw std::runtime_error("DeepSeek: Engram token map size");
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] >= 99092) throw std::runtime_error("DeepSeek: Engram token map out of range");
            token_map_[i] = static_cast<int32_t>(ids[i]);
        }
    } else {
        const auto& token = tensors_.at("engram_token_map.weight");
        std::memcpy(token_map_.data(), token.data, token.bytes);
    }
    for (int32_t id : token_map_)
        if (id < 0 || id >= 99092) throw std::runtime_error("DeepSeek: Engram token map out of range");
    const auto multipliers = array(f, p + (ple_engram_ ? "ple.layer_multipliers" : "engram.multipliers"));
    const auto primes = array(f, p + (ple_engram_ ? "ple.head_vocab_sizes" : "engram.primes"));
    const auto offsets = array(f, p + (ple_engram_ ? "ple.head_offsets" : "engram.offsets"));
    const auto rows = ple_engram_ ? std::vector<uint64_t>{384006168,384016682} : array(f, p + "engram.num_embeddings");
    if (multipliers.size() != 8 || primes.size() != 48 || offsets.size() != 48 || rows.size() != 2)
        throw std::runtime_error("DeepSeek: invalid Engram hash metadata lengths");
    for (uint64_t multiplier : multipliers)
        if (!multiplier || !(multiplier & 1) || multiplier > uint64_t(INT64_MAX) / 99091)
            throw std::runtime_error("DeepSeek: invalid Engram multiplier");
    for (int e = 0; e < 2; ++e) {
        const uint64_t base = ple_engram_ && e ? rows[0] : 0;
        uint64_t end = base;
        for (int h = 0; h < 24; ++h) {
            const int index = e * 24 + h;
            if (!primes[index] || offsets[index] != end)
                throw std::runtime_error("DeepSeek: invalid Engram hash bucket");
            end = checked_add(end, primes[index]);
        }
        if (end - base != rows[e] || (!ple_engram_ && tensor(block(e ? 14 : 1, "engram_embd.weight")).info->shape[1] != rows[e]))
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
    const auto targets = array(f, p + "target_layers");
    draft_shared_embeddings_ = targets == std::vector<uint64_t>{37,38,39};
    if (draft_shared_embeddings_) {
        // The 1017 converter writes attention-input layer IDs directly and
        // omits the target's shared embedding/head. Both formats capture 37..39.
        if (!main_ || !ple_engram_)
            throw std::runtime_error("DeepSeek: 1017 DSpark requires its PLE main model");
    } else {
        array_equal(f, p + "target_layers", {38,39,40});
        if (main_ && ple_engram_)
            throw std::runtime_error("DeepSeek: legacy DSpark does not match the PLE main model");
    }
    validate_file(f, draft_tensors_, true, false);
}

void Model::load_vision() {
    const auto& f = *vision_;
    const auto& projector = meta(f, "clip.projector_type").s;
    if (meta(f, "general.architecture").s != "clip" || (projector != "deepseek4v" && projector != "deepseek41v"))
        throw std::runtime_error("DeepSeek: unsupported vision encoder");
    if (main_ && (projector == "deepseek41v") != ple_engram_)
        throw std::runtime_error("DeepSeek: vision checkpoint does not match the main model");
    for (const auto& [key, expected] : std::vector<std::pair<std::string, uint64_t>>{
        {"block_count",32}, {"embedding_length",1024}, {"feed_forward_length",2816},
        {"attention.head_count",16}, {"patch_size",14}, {"projection_dim",5120}, {"projector.scale_factor",3}})
        equal(f, "clip.vision." + key, expected);
    validate_file(f, vision_tensors_, false, true);
}
std::string Model::checkpoint_identity() const {
    if (!main_ || !main_complete_) throw std::runtime_error("DeepSeek: complete main checkpoint required");
    if (!ple_engram_) return "11f46543370256ef616b6e458b6950e148625b5f8b7545173d124445d3393c53";
    std::string identity;
    for (const auto* sha : engram_type()==13 ? q2k_xl_sha256 : iq3xxs_sha256) {
        if (!identity.empty()) identity += ':';
        identity += sha;
    }
    return identity;
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
    const auto collect = [&](const GgufFile& shard) {
    const uint64_t available = shard.file_size() >= shard.data_start() ? shard.file_size() - shard.data_start() : 0;
    std::vector<const TensorInfo*> ordered;
    for (const auto& t : shard.tensors()) {
        const uint64_t bytes = tensor_payload_bytes(t);
        if (!bytes || t.offset > available || bytes > available - t.offset || t.offset % shard.alignment())
            throw std::runtime_error("DeepSeek: invalid tensor span/type/alignment: " + t.name);
        if (!views.emplace(t.name, TensorView{&t, shard.tensor_data(t), bytes,
            bytes / (t.elements() / t.shape[0]), TensorRole::dense, &shard}).second)
            throw std::runtime_error("DeepSeek: duplicate tensor " + t.name);
        ordered.push_back(&t);
    }
    // Overlapping tensors are never valid role slices, even if each individually fits the file.
    std::sort(ordered.begin(), ordered.end(), [](auto* a, auto* b) { return a->offset < b->offset; });
    for (size_t i = 1; i < ordered.size(); ++i)
        if (tensor_payload_bytes(*ordered[i - 1]) > ordered[i]->offset - ordered[i - 1]->offset)
            throw std::runtime_error("DeepSeek: overlapping tensor " + ordered[i]->name);
    };
    collect(f);
    if (draft && draft_shared_embeddings_) {
        for (const char* name : {"token_embd.weight", "output.weight"}) {
            if (views.count(name)) throw std::runtime_error("DeepSeek: 1017 draft duplicates shared target weights");
            views.emplace(name, tensor(name));
        }
    }
    if (!draft && !vision) {
        for (const auto& shard : main_shards_) collect(*shard);
        if (!main_shards_.empty() &&
            views.size() + (main_complete_ ? 0 : 1) != integer(f, "split.tensors.count"))
            throw std::runtime_error("DeepSeek: incomplete split tensor directory");
    }
    std::set<std::string> consumed;
    auto require = [&](const std::string& name, std::vector<uint64_t> shape, uint32_t type,
                       TensorRole role = TensorRole::dense) {
        auto it = views.find(name);
        if (it == views.end()) throw std::runtime_error("DeepSeek: missing tensor " + name);
        const bool bf16_dense = !draft && !vision && ple_engram_ && type==8 && it->second.info->type==30 &&
            (name.find("indexer.")!=std::string::npos || name.find("attn_compressor_")!=std::string::npos);
        const bool bf16_markov = draft && draft_shared_embeddings_ && type==8 && it->second.info->type==30 &&
            (name=="markov_w1.weight" || name=="markov_w2.weight");
        const auto suffix=name.substr(name.find('.',4)+1);
        const bool q6_dense=ple_engram_ && !vision && type==8 && it->second.info->type==14 &&
            (name=="output.weight" || (!draft && (suffix=="attn_q_a.weight" || suffix=="attn_q_b.weight" ||
             suffix=="attn_kv.weight" || suffix=="attn_output_a.weight" || suffix=="attn_output_b.weight" ||
             suffix=="ffn_gate_shexp.weight" || suffix=="ffn_up_shexp.weight" || suffix=="ffn_down_shexp.weight")));
        const bool q5_engram=!draft && !vision && ple_engram_ && type==14 && it->second.info->type==13 &&
            name=="per_layer_token_embd.weight";
        if (it->second.info->shape != shape || (it->second.info->type != type && !bf16_dense && !bf16_markov && !q6_dense && !q5_engram && !(draft&&type==8&&it->second.info->type==2)))
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
            const uint32_t type = draft || ple_engram_ ? views.at(block(l,"ffn_gate_exps.weight")).info->type : l < 14 ? 11 : 10;
            if(draft&&type!=39&&type!=10&&type!=11&&type!=42)throw std::runtime_error("DeepSeek: unsupported draft expert type");
            if(!draft && type!=10 && type!=11 && type!=16 && type!=17 && type!=18 && type!=21 && type!=22)
                throw std::runtime_error("DeepSeek: unsupported main Gate/Up type");
            for (const char* name : {"ffn_gate_exps.weight", "ffn_up_exps.weight"})
                require(block(l, name), {5120,2304,experts}, type, TensorRole::routed_expert);
            const uint32_t down_type = !draft && ple_engram_ ? views.at(block(l,"ffn_down_exps.weight")).info->type : type;
            if(!draft && down_type!=10 && down_type!=11 && down_type!=16 && down_type!=18 && down_type!=21 && down_type!=39)
                throw std::runtime_error("DeepSeek: unsupported main Down type");
            require(block(l, "ffn_down_exps.weight"), {2304,5120,experts}, down_type, TensorRole::routed_expert);
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
                if (ple_engram_) {
                    require(block(l, "engram_gate.weight"), {5120,4}, 0);
                    require(block(l, "engram_kv.weight"), {6144,25600}, 14);
                } else {
                require(block(l, "engram_embd.weight"), {256,l == 1 ? 384006168u : 384016682u}, 8, TensorRole::engram);
                require(block(l, "engram_q.weight"), {5120,4}, 0);
                require(block(l, "engram_k.weight"), {5120,4}, 0);
                require(block(l, "engram_wkv.weight"), {6144,25600}, 8);
                }
            }
        }
        require("token_embd.weight", {5120,129280}, 8);
        require("output.weight", {5120,129280}, 8);
        require("output_norm.weight", {5120}, 0);
        if (!draft) {
            if (ple_engram_) {
                if (main_complete_) require("per_layer_token_embd.weight", {256,768022850}, 14, TensorRole::engram);
                else if (views.count("per_layer_token_embd.weight"))
                    throw std::runtime_error("DeepSeek: missing shard is not the Engram table");
            } else require("engram_token_map.weight", {129280}, 26, TensorRole::token_map);
        }
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
    const uint64_t down_stride = down.bytes / count;
    return {gate.data + stride * id, up.data + stride * id, down.data + down_stride * id,
            gate.info->type, down.info->type, gate.row_bytes, down.row_bytes, stride, stride * 2 + down_stride};
}
std::array<uint64_t, 40> Model::expert_bytes() const {
    std::array<uint64_t, 40> sizes{};
    for (int l = 0; l < 40; ++l) sizes[l] = expert(l, 0).blob_bytes;
    return sizes;
}
void Model::write_inventory(std::ostream& out) const {
    out << "{\"architecture\":\"deepseek41\",\"main_complete\":" << (main_complete_?"true":"false")
        << ",\"geometry\":{\"layers\":40,\"hidden\":5120,\"heads\":64},\"files\":[";
    bool first_file = true;
    std::vector<std::pair<const GgufFile*,const Views*>> files{{main_.get(),&tensors_}};
    for(const auto& shard:main_shards_)files.push_back({shard.get(),&tensors_});
    files.push_back({draft_.get(),&draft_tensors_});
    files.push_back({vision_.get(),&vision_tensors_});
    for (const auto& [f,collection] : files) {
        if (!f) continue;
        if (!first_file) out << ',';
        first_file = false;
        const auto& views = *collection;
        out << "{\"path\":" << json_string(f->path()) << ",\"bytes\":" << f->file_size()
            << ",\"data_start\":" << f->data_start() << ",\"tensors\":[";
        bool first = true;
        for (const auto& [name, v] : views) {
            if(v.file!=f)continue;
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
