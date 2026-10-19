// Offline, teacher-forced scoring through the real Session API. Inputs and logits stay on the server.
#include "strata/models/deepseek41/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace d = strata::models::deepseek41;
namespace {
constexpr int vocab = 129280;
struct Score { double nll; int rank, argmax; };
Score score(const float* logits, int size, int target) {
    if (target < 0 || target >= size || size < 1) throw std::invalid_argument("score target/size");
    const auto best = std::max_element(logits, logits + size);
    const double maximum = *best;
    double total = 0;
    int rank = 1;
    for (int i = 0; i < size; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite scoring logits");
        total += std::exp(double(logits[i]) - maximum);
        rank += logits[i] > logits[target];
    }
    return {std::log(total) + maximum - logits[target], rank, int(best - logits)};
}
void self_test() {
    const float equal[]{0, 0}, shifted[]{1000, 1000}, unequal[]{0, float(std::log(3.0))};
    const auto a = score(equal, 2, 0), b = score(shifted, 2, 1), c = score(unequal, 2, 1);
    if (std::abs(a.nll - std::log(2.0)) > 1e-12 || std::abs(a.nll - b.nll) > 1e-12 ||
        std::abs(c.nll + std::log(0.75)) > 1e-7 || a.rank != 1 || c.rank != 1 || c.argmax != 1)
        throw std::runtime_error("NLL normalization/shift/rank self-test failed");
    const float invalid[]{0, std::numeric_limits<float>::quiet_NaN()};
    bool rejected = false;
    try { score(invalid, 2, 0); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("nonfinite NLL self-test failed");
}
struct Sample { std::string id; std::vector<int> prompt, continuation; };
std::vector<Sample> read_samples(const std::string& path, int context) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open tokenized scoring input");
    std::vector<Sample> samples;
    std::set<std::string> ids;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::istringstream row(line);
        Sample sample; int prompt = 0, continuation = 0;
        if (!(row >> sample.id >> prompt >> continuation) || sample.id.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.") != std::string::npos ||
            !ids.insert(sample.id).second || prompt < 1 || continuation < 1 || prompt > context || continuation > context - prompt)
            throw std::invalid_argument("invalid/duplicate sample ID or prompt/continuation bounds");
        sample.prompt.resize(prompt); sample.continuation.resize(continuation);
        for (auto* sequence : {&sample.prompt, &sample.continuation})
            for (int& id : *sequence)
                if (!(row >> id) || id < 0 || id >= vocab) throw std::invalid_argument("scoring token ID/count");
        row >> std::ws;
        if (!row.eof()) throw std::invalid_argument("extra scoring input fields");
        samples.push_back(std::move(sample));
    }
    if (!input.eof() || samples.empty()) throw std::invalid_argument("empty or unreadable scoring input");
    return samples;
}
}
int main(int argc, char** argv) {
    std::string model, input, output;
    d::RuntimeOptions options;
    bool ced = false, test_only = false;int score_step=8;
    options.context = 32768; options.slots = 1;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&] { if (i + 1 >= argc) throw std::invalid_argument(arg + " requires a value"); return std::string(argv[++i]); };
            if (arg == "--self-test") test_only = true;
            else if (arg == "--model") model = next();
            else if (arg == "--integrity") options.integrity_file = next();
            else if (arg == "--input") input = next();
            else if (arg == "--output") output = next();
            else if (arg == "--context") options.context = std::stoi(next());
            else if (arg == "--chunk") options.chunk = std::stoi(next());
            else if (arg == "--score-step") score_step = std::stoi(next());
            else if (arg == "--cpu-workers") options.cpu_workers = std::stoi(next());
            else if (arg == "--hot-front-gib") options.hot_front = std::stoull(next()) << 30;
            else if (arg == "--hot-back-gib") options.hot_back = std::stoull(next()) << 30;
            else if (arg == "--engram-cache-mib") options.engram_cache = std::stoull(next()) << 20;
            else if (arg == "--profile") options.profile = next();
            else if (arg == "--hot-front-bytes") options.hot_front = std::stoull(next());
            else if (arg == "--work-mib") options.work = std::stoull(next()) << 20;
            else if (arg == "--state-mib") options.state = std::stoull(next()) << 20;
            else if (arg == "--gpu-headroom-mib") options.gpu_headroom = std::stoull(next()) << 20;
            else if (arg == "--engram-pipeline") options.engram_pipeline = true;
            else if (arg == "--expert-bf16") options.expert_bf16 = true;
            else if (arg == "--expert-direct-gemv") options.expert_direct_gemv = true;
            else if (arg == "--expert-pipeline") options.expert_pipeline = true;
            else if (arg == "--adaptive-experts") options.adaptive_experts = true;
            else if (arg == "--adaptive-layer-progress") options.adaptive_layer_progress = true;
            else if (arg == "--migration-mib") options.migration_bytes = std::stoull(next()) << 20;
            else if (arg == "--pcie-frac") options.pcie_frac = std::stod(next());
            else if (arg == "--compact-logits") options.compact_logits = true;
            else if (arg == "--compact-workspace") options.compact_workspace = true;
            else if (arg == "--dense-q8") options.dense_q8 = true;
            else if (arg == "--dense-q8-1") options.dense_q8_1 = true;
            else if (arg == "--expert-gu-mode") options.expert_gu_mode = std::stoi(next());
            else if (arg == "--cpu-gu-q8-0") options.cpu_gu_q8_0 = true;
            else if (arg == "--decode-shared-graph") options.decode_graph = true;
            else if (arg == "--decode-raw-cache") options.decode_raw_cache = true;
            else if (arg == "--pinned-experts") options.pinned_experts = true;
            else if (arg == "--prefill-expert-lookahead") options.prefill_expert_lookahead = true;
            else if (arg == "--engram-continuous") options.engram_continuous = true;
            else if (arg == "--parallel-router") options.parallel_router = true;
            else if (arg == "--engram-aio") options.engram_aio = true;
            else if (arg == "--mode") { const auto value = next(); if (value != "full" && value != "ced") throw std::invalid_argument("mode"); ced = value == "ced"; }
            else if (arg == "--expert-mode") { const auto value = next(); if (value != "gpu" && value != "hybrid") throw std::invalid_argument("expert mode"); options.experts = value == "gpu" ? d::ExpertMode::gpu : d::ExpertMode::hybrid; }
            else if (arg == "--arithmetic") { const auto value = next(); if (value != "f32" && value != "production") throw std::invalid_argument("arithmetic"); options.arithmetic = value == "f32" ? d::Arithmetic::f32 : d::Arithmetic::production; }
            else if (arg == "--help") {
                std::cout << "score-runtime --model MAIN --integrity PROOF --input TOKENS --output NEW.jsonl [--mode full|ced] [--arithmetic production|f32] [--expert-mode hybrid|gpu] [--context N] [--chunk N]\n"
                          << "Input: one sample_id prompt_count continuation_count prompt_IDs continuation_IDs per line.\n";
                return 0;
            } else throw std::invalid_argument("unknown scoring option: " + arg);
        }
        self_test();
        if (test_only) { std::cout << "NLL_SELF_TEST_OK\n"; return 0; }
        if (model.empty() || options.integrity_file.empty() || input.empty() || output.empty() ||
            options.context < 2 || options.chunk < 1 || score_step<1 || score_step>8 || std::filesystem::exists(output))
            throw std::invalid_argument("model/proof/input/new output/context/chunk required");
        const auto samples = read_samples(input, options.context);
        if (options.arithmetic == d::Arithmetic::f32) { options.experts = d::ExpertMode::gpu; options.hot_front = options.hot_back = 0; }
        // Exclusive output creation is supplied by the owning controller's private directory.
        std::ofstream report(output);
        if (!report) throw std::runtime_error("cannot create scoring output");
        report << std::setprecision(17);
        report << "{\"kind\":\"configuration\",\"mode\":\"" << (ced ? "ced" : "full") << "\",\"arithmetic\":\""
               << (options.arithmetic == d::Arithmetic::f32 ? "f32" : "production") << "\",\"expert_mode\":\""
               << (options.experts == d::ExpertMode::gpu ? "gpu" : "hybrid") << "\",\"context\":" << options.context
               << ",\"chunk\":" << options.chunk << ",\"score_step\":"<<score_step<<",\"samples\":" << samples.size() << ",\"phase_residency\":false,\"speculative\":false}\n";
        d::Runtime runtime(model, options);
        report << "{\"kind\":\"ledger\",\"value\":"; runtime.write_ledger(report); report << "}\n";
        size_t positions = 0; double total_nll = 0;
        for (const auto& sample : samples) {
            auto session = runtime.create_session();
            auto logits = session->prefill(sample.prompt, ced);
            if (logits.size() != vocab) throw std::runtime_error("prefill scoring shape");
            double sample_nll = 0;
            auto record = [&](size_t at, const float* row) {
                const int target = sample.continuation[at]; const auto value = score(row, vocab, target);
                sample_nll += value.nll;
                report << "{\"kind\":\"position\",\"sample\":\"" << sample.id << "\",\"offset\":" << at
                       << ",\"absolute_target_position\":" << sample.prompt.size() + at << ",\"target\":" << target
                       << ",\"nll\":" << value.nll << ",\"rank\":" << value.rank << ",\"argmax\":" << value.argmax << "}\n";
            };
            record(0, logits.data());
            for (size_t begin = 0; begin + 1 < sample.continuation.size();) {
                const size_t count = std::min<size_t>({size_t(score_step), size_t(options.chunk), sample.continuation.size() - 1 - begin});
                const std::vector<int> ids(sample.continuation.begin() + begin, sample.continuation.begin() + begin + count);
                const auto rows = session->forward(ids);
                if (rows.size() != count * vocab) throw std::runtime_error("teacher-forced scoring row count");
                for (size_t i = 0; i < count; ++i) record(begin + i + 1, rows.data() + i * vocab);
                begin += count;
            }
            if (session->position() != int64_t(sample.prompt.size() + sample.continuation.size() - 1) ||
                session->original_prompt_length() != int64_t(sample.prompt.size())) throw std::runtime_error("scoring original prompt/position changed");
            const auto stats = session->stats();
            report << "{\"kind\":\"sample\",\"id\":\"" << sample.id << "\",\"positions\":" << sample.continuation.size()
                   << ",\"nll_sum\":" << sample_nll << ",\"mean_nll\":" << sample_nll / sample.continuation.size()
                   << ",\"encoder_token_layers\":" << stats.encoder_token_layers << ",\"decoder_token_layers\":" << stats.decoder_token_layers
                   << ",\"kv_only_tokens\":" << stats.kv_only_tokens << ",\"replay_tokens\":" << stats.replay_tokens
                   << ",\"decoded_tokens\":" << stats.decoded_tokens << ",\"cpu_expert_tokens\":" << stats.expert_cpu
                   << ",\"gpu_expert_tokens\":" << stats.expert_gpu << ",\"prefill_ms\":" << stats.prefill_ms
                   << ",\"decode_ms\":" << stats.decode_ms << ",\"shared_graph_captures\":"<<stats.shared_graph_captures
                   << ",\"shared_graph_launches\":"<<stats.shared_graph_launches<<"}\n";
            report.flush(); if (!report) throw std::runtime_error("scoring report write failed");
            total_nll += sample_nll; positions += sample.continuation.size();
        }
        const double mean_nll = total_nll / positions, perplexity = std::exp(mean_nll);
        report << "{\"kind\":\"completed\",\"positions\":" << positions << ",\"nll_sum\":" << total_nll
               << ",\"mean_nll\":" << mean_nll << ",\"perplexity\":";
        if (std::isfinite(perplexity)) report << perplexity; else report << "null";
        report << ",\"perplexity_overflow\":" << (std::isfinite(perplexity) ? "false" : "true") << "}\n";
        report.flush(); report.close();
        if (!report) throw std::runtime_error("scoring report finish failed");
        return 0;
    } catch (const std::exception& error) { std::cerr << "Scoring: " << error.what() << '\n'; return 1; }
}
