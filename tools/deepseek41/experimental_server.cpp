#include "adaptive_test_access.hpp"
#include "strata/kernels/cpu/deepseek41_down.hpp"
#include "strata/models/deepseek41/mixed_expert.hpp"
#include "strata/artifact/gguf_split.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace d = strata::models::deepseek41;
namespace strata::models::deepseek41 { int serve(Runtime&, const RuntimeOptions&); }

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: strata-deepseek41-experimental MODEL DRAFT VISION_OR_DASH PROOF PROFILE HEAT COSTS INPUT_VOCAB OUTPUT_VOCAB [HOT_BYTES]\n";
            return 0;
        }
        if (argc != 10 && argc != 11) throw std::invalid_argument("expected nine asset paths and optional hot bytes; use --help");
        const std::string model = argv[1];
        const bool iq = strata::gguf_split_paths(model).size() == 7;
        const std::string vision = std::string(argv[3]) == "-" ? "" : argv[3];
        d::RuntimeOptions o;
        o.integrity_file = argv[4];
        o.draft_sha256 = iq ? "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a" :
                             "9269e73cbd04c493d2510d3513da85429441949f6f10ed8130392ae0de2a949b";
        if (iq) o.vision_sha256 = "9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9";
        o.draft_input_vocab = argv[8];
        o.draft_vocab = argv[9];
        o.dspark = true; o.context = 1048576; o.chunk = 8192; o.slots = 1;
        o.hot_front = iq ? 64ull << 30 : 72596914176ull; o.hot_back = 0; o.hot_auto = false;
        if (argc == 11) {
            const std::string value = argv[10];
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("HOT_BYTES must be a positive integer");
            o.hot_front = std::stoull(value);
            if (!o.hot_front) throw std::invalid_argument("HOT_BYTES must be positive");
        }
        o.work = 6720ull << 20; o.state = 1024ull << 20; o.gpu_headroom = 700ull << 20;
        o.profile = argv[5];
        o.engram_cache = 8192ull << 20; o.migration_bytes = 3072ull << 20;
        o.ram_experts = (iq ? 160ull : 144ull) << 30; o.host_headroom = 16ull << 30;
        o.experts = d::ExpertMode::hybrid; o.adaptive_experts = true; o.adaptive_groups = true;
        o.engram_pipeline = true; o.expert_bf16 = true; o.expert_direct_gemv = true;
        o.expert_pipeline = true; o.adaptive_layer_progress = true; o.pcie_frac = 1;
        o.compact_logits = true; o.compact_workspace = true; o.dense_q8 = true; o.dense_q8_1 = true;
        o.expert_gu_mode = 2; o.decode_graph = true; o.decode_raw_cache = true;
        o.pinned_experts = true; o.prefill_expert_lookahead = true; o.huge_pinned_memory = true;
        o.engram_continuous = true; o.parallel_router = true; o.engram_aio = true; o.trace = false;
        o.dspark_prefix_commit = true; o.dense_q8_batch = true; o.dspark_gpu_greedy = true;
        o.dspark_incremental_kv = true; o.dspark_draft_tokens = 5; o.shared_q8_pair = true;
        o.dspark_adaptive_verification = true; o.dspark_fused_moe = true;
        o.dspark_batch_projection = true; o.target_verify_batch = true;
        o.dspark_draft_cost_ms = 7.659439;
        std::ifstream costs(argv[7]);
        double unused = 0;
        if (!(costs >> unused)) throw std::runtime_error("cost header");
        for (double& value : o.dspark_verify_cost_ms)
            if (!(costs >> value) || value <= 0) throw std::runtime_error("verify costs");
        strata::kernels::cpu::deepseek41_down_fusion_set_enabled(true);
        setenv("STRATA_DS41_MAIN_MOE_FUSION", "2", 1);
        d::scores_fast_set_enabled(true); d::expert_dequant_fast_set_enabled(true);
        d::expert_mixed_fast_set_enabled(iq);
        d::expert_mixed_short_set_enabled(iq);
        // Match the established experiment's process-owned runtime lifetime.
        auto* runtime = new d::Runtime(model, o,
            argv[2], vision);
        using Access = d::PrefetchBlockBenchmark;
        const auto& host = Access::rt(*runtime).experts->host_memory_info();
        host.write(std::cerr); std::cerr << '\n';
        const unsigned huge_minimum = d::block_huge_min_percent();
        if ((std::string(host.mode) != "registered_2m_blocks" &&
             std::string(host.mode) != "registered_blocks_mixed_pages") || host.registration_count != 40 ||
            host.required_huge_percent != huge_minimum || !host.logical ||
            !d::host_huge_sufficient(host.huge_before, host.mapped, huge_minimum) ||
            !d::host_huge_sufficient(host.huge_after, host.mapped, huge_minimum) ||
            Access::rt(*runtime).experts->hot_pairs().empty() || Access::rt(*runtime).experts->gpu_bytes() > o.hot_front)
            throw std::runtime_error("block-pinned memory/coverage/residency gate");
        Access::startup_heat(Access::cache(*runtime), argv[6], false);
        Access::aligned(Access::cache(*runtime), false);
        runtime->write_ledger(std::cerr); std::cerr << '\n';
        size_t free = 0, total = 0; d::gpu_check(cudaMemGetInfo(&free, &total));
        if (free < (700ull << 20)) throw std::runtime_error("GPU free-memory guard");
        std::cerr << "Experimental service: context=1048576 chunk=8192 work_mib=6720 vision=F32 block_pinned=40 huge_min_percent="
                  << huge_minimum << " draft=5 adaptive=1 shared_pair=1 main_fusion=2 scores_fast=1 dq_fast=1 mixed_fast=" << iq
                  << " iq_short_batch=" << iq << " hot_bytes=" << o.hot_front << " free=" << free << '\n';
        const int result = d::serve(*runtime, o);
        std::cerr << "Experimental IPC finished code=" << result << std::endl;
        std::_Exit(result);
    } catch (const std::exception& error) {
        std::cerr << "Experimental service: " << error.what() << std::endl;
        return 1;
    }
}
