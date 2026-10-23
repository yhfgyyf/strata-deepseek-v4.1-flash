#pragma once

#include "strata/models/deepseek41/model.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::models::deepseek41 {
struct EngramHashSpec {
    std::array<uint64_t, 8> multipliers{};
    std::array<uint64_t, 48> primes{}, offsets{};
    int32_t pad = 0;
};
struct EngramHistory {
    // Most recent compressed ID first. Images remain DEAD and stop lookbacks.
    std::array<int32_t, 3> recent{-1, -1, -1};
    int64_t position = 0;
};
struct EngramQuery {
    std::vector<uint64_t> rows; // [token, layer(1/14), 24]
    std::vector<uint8_t> image;
    EngramHistory after;
};
EngramHashSpec engram_hash_spec(const Model& model);
EngramQuery engram_hash(const EngramHashSpec& spec, const std::vector<int32_t>& token_map,
                       const std::vector<int>& tokens, const std::vector<uint8_t>& image,
                       const EngramHistory& before);

struct EngramReadStats {
    uint64_t batches = 0, logical_bytes = 0, disk_bytes = 0, reads = 0, page_hits = 0;
    uint64_t cache_owned_bytes = 0;
    double p95_us = 0;
    // Split-table reads only; host wall includes lookup/cache/I/O/gather work.
    std::array<uint64_t,2> layer_disk_bytes{},layer_reads{},layer_page_hits{};
    std::array<double,2> layer_wall_us{};
    std::array<double,2> layer_cache_fill_us{}; // Optional maintenance after immutable table publication.
};
struct EngramBatch {
    uint64_t generation = 0;
    std::atomic<bool> cancelled{false};
    void cancel() { cancelled.store(true); }
    // Wait retains in-flight storage. A cancelled batch throws instead of publishing stale rows.
    const std::vector<uint8_t>& wait();
    // Same immutable interleaved storage; only the requested table's rows are ready.
    const std::vector<uint8_t>& wait_layer(int table);
private:
    friend class EngramReader;
    struct State;
    std::shared_ptr<State> state;
};
class EngramReader {
public:
    EngramReader(const std::string& path, const std::array<uint64_t, 2>& table_offsets,
                 const std::array<uint64_t, 2>& table_rows, uint64_t cache_budget,
                 size_t queue_limit = 4,bool continuous_reads = false,bool native_aio = false,
                 uint64_t packed_row_bytes = 272);
    explicit EngramReader(const Model& model, uint64_t cache_budget,bool continuous_reads = false,bool native_aio = false);
    ~EngramReader();
    EngramReader(const EngramReader&) = delete;
    EngramReader& operator=(const EngramReader&) = delete;
    std::shared_ptr<EngramBatch> submit(EngramQuery query, uint64_t generation, bool split_layers = false);
    EngramReadStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
void engram_gate_cpu(float* residual, const float* kv, const float* q_weight,
                     const float* k_weight, const uint8_t* image, int tokens, int hidden = 5120);
} // namespace strata::models::deepseek41
