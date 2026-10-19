// Real-model Session checks, linked separately against existing native libraries.
// Only JSON and temporary prefix fixtures are written; captures are disabled.
#include "strata/models/deepseek41/runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace d = strata::models::deepseek41;
namespace fs = std::filesystem;
namespace {
constexpr int vocab = 129280, prompt_size = 129;
constexpr double max_abs_limit = 1e-3, rmse_limit = 1e-4;
using Values = std::vector<float>;
std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}
std::string number(double value) {
    std::ostringstream out; out << std::setprecision(17) << value; return out.str();
}
void require(bool okay, const std::string& message) {
    if (!okay) throw std::runtime_error(message);
}
std::vector<int> tokens(int count) {
    std::vector<int> result(count); std::iota(result.begin(), result.end(), 1); return result;
}
std::vector<int> suffix(int count = 19) {
    auto ids = tokens(prompt_size + count); return {ids.begin() + prompt_size, ids.end()};
}
Values last_row(const Values& values) {
    require(!values.empty() && values.size() % vocab == 0, "invalid logit row count");
    return {values.end() - vocab, values.end()};
}
std::string stats_json(const d::ExecutionStats& s) {
    std::ostringstream out;
    out << "{\"encoder_token_layers\":" << s.encoder_token_layers << ",\"decoder_token_layers\":" << s.decoder_token_layers
        << ",\"kv_only_tokens\":" << s.kv_only_tokens << ",\"replay_tokens\":" << s.replay_tokens
        << ",\"decoded_tokens\":" << s.decoded_tokens << '}';
    return out.str();
}
struct Report {
    fs::path output;
    std::vector<std::string> records;
    std::string selected = "full,ced,resume,prefix,transaction,cancel";
    void add(const std::string& record) { records.push_back(record); }
    void write(const std::string& status, const std::string& reason = {}) const {
        std::ofstream out(output, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write JSON report");
        out << "{\"status\":" << quote(status) << ",\"scope\":\"same-model F32 Session checks; no independent oracle or serving claim\""
            << ",\"requested_cases\":" << quote(selected)
            << ",\"fixture\":{\"tokens\":\"1..N\",\"context\":512,\"slots\":1,\"chunk\":3,\"full_chunks\":[9,3,1],\"phase_residency\":false}"
            << ",\"limits\":{\"max_abs\":0.001,\"rmse\":0.0001},\"records\":[";
        for (size_t i = 0; i < records.size(); ++i) out << (i ? ",\n" : "\n") << records[i];
        out << "\n]";
        if (!reason.empty()) out << ",\"reason\":" << quote(reason);
        out << "}\n"; require(bool(out), "cannot finish JSON report");
    }
};
void compare(Report& report, const std::string& name, const Values& actual, const Values& reference) {
    require(!actual.empty() && actual.size() == reference.size() && actual.size() % vocab == 0,
            name + ": logit shapes differ");
    bool passed = true;
    for (size_t row = 0; row < actual.size() / vocab; ++row) {
        bool finite = true; double maximum = 0, squared = 0, norm = 0;
        int arg_actual = 0, arg_reference = 0;
        for (int col = 0; col < vocab; ++col) {
            const size_t i = row * vocab + col;
            const double got = actual[i], want = reference[i];
            if (!std::isfinite(got) || !std::isfinite(want)) { finite = false; continue; }
            const double error = got - want;
            maximum = std::max(maximum, std::abs(error)); squared += error * error; norm += want * want;
            if (got > actual[row * vocab + arg_actual]) arg_actual = col;
            if (want > reference[row * vocab + arg_reference]) arg_reference = col;
        }
        const double rmse = std::sqrt(squared / vocab);
        const bool okay = finite && maximum <= max_abs_limit && rmse <= rmse_limit;
        report.add("{\"kind\":\"logits\",\"name\":" + quote(name) + ",\"row\":" + std::to_string(row) +
            ",\"values\":" + std::to_string(vocab) + ",\"finite\":" + (finite ? "true" : "false") +
            ",\"max_abs\":" + (finite ? number(maximum) : "null") + ",\"rmse\":" + (finite ? number(rmse) : "null") +
            ",\"relative_l2\":" + (finite ? number(std::sqrt(squared) / std::max(std::sqrt(norm), 1e-30)) : "null") +
            ",\"argmax_actual\":" + (finite ? std::to_string(arg_actual) : "null") +
            ",\"argmax_reference\":" + (finite ? std::to_string(arg_reference) : "null") +
            ",\"argmax_equal\":" + (finite && arg_actual == arg_reference ? "true" : "false") +
            ",\"passed\":" + (okay ? "true}" : "false}"));
        passed &= okay;
    }
    require(passed, name + ": finite/max_abs/RMSE gate failed");
}
void state(Report& report, const std::string& name, const d::Session& s, int position, int original) {
    const bool okay = s.position() == position && s.original_prompt_length() == original;
    report.add("{\"kind\":\"state\",\"name\":" + quote(name) + ",\"position\":" + std::to_string(s.position()) +
        ",\"original_prompt\":" + std::to_string(s.original_prompt_length()) +
        ",\"expected_position\":" + std::to_string(position) + ",\"expected_original_prompt\":" + std::to_string(original) +
        ",\"passed\":" + (okay ? "true}" : "false}"));
    require(okay, name + ": position/original boundary mismatch");
}
void counts(Report& report, const std::string& name, const d::Session& session,
            uint64_t encoder, uint64_t decoder, uint64_t kv, uint64_t replay, uint64_t decoded) {
    const auto s = session.stats();
    const bool okay = s.encoder_token_layers == encoder && s.decoder_token_layers == decoder &&
        s.kv_only_tokens == kv && s.replay_tokens == replay && s.decoded_tokens == decoded;
    report.add("{\"kind\":\"counts\",\"name\":" + quote(name) + ",\"actual\":" + stats_json(s) +
        ",\"expected\":{\"encoder_token_layers\":" + std::to_string(encoder) +
        ",\"decoder_token_layers\":" + std::to_string(decoder) + ",\"kv_only_tokens\":" + std::to_string(kv) +
        ",\"replay_tokens\":" + std::to_string(replay) + ",\"decoded_tokens\":" + std::to_string(decoded) +
        "},\"passed\":" + (okay ? "true}" : "false}"));
    require(okay, name + ": execution counts mismatch");
}
template<class F> void rejected(Report& report, const std::string& name, const std::string& expected, F operation) {
    std::string reason;
    try { operation(); } catch (const std::exception& e) { reason = e.what(); }
    const bool okay = !reason.empty() && reason.find(expected) != std::string::npos;
    report.add("{\"kind\":\"rejection\",\"name\":" + quote(name) + ",\"reason\":" + quote(reason) +
        ",\"expected_fragment\":" + quote(expected) + ",\"passed\":" + (okay ? "true}" : "false}"));
    require(okay, name + ": expected rejection was absent or different");
}
d::RuntimeOptions options(const std::string& integrity, int chunk = 3) {
    d::RuntimeOptions result;
    result.arithmetic = d::Arithmetic::f32; result.experts = d::ExpertMode::gpu;
    result.context = 512; result.chunk = chunk; result.slots = 1;
    result.hot_front = result.hot_back = 0; result.phase_residency = false; result.integrity_file = integrity;
    return result;
}
void ledger(Report& report, const std::string& name, const d::Runtime& runtime) {
    std::ostringstream out; runtime.write_ledger(out);
    report.add("{\"kind\":\"ledger\",\"name\":" + quote(name) + ",\"value\":" + out.str() + '}');
}
Values advance(Report& report, const std::string& name, d::Session& session,
               const std::vector<int>& ids, int step, int original) {
    Values rows; const int base = session.position();
    for (size_t start = 0; start < ids.size(); start += step) {
        const size_t count = std::min<size_t>(step, ids.size() - start);
        auto result = session.forward({ids.begin() + start, ids.begin() + start + count});
        require(result.size() == count * vocab, name + ": forward did not return every suffix row");
        rows.insert(rows.end(), result.begin(), result.end());
        state(report, name + ".after_" + std::to_string(start + count), session, base + start + count, original);
    }
    return rows;
}
struct Scratch {
    fs::path path;
    explicit Scratch(const fs::path& parent) {
        path = parent / ("runtime-state-prefix-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(fs::create_directory(path), "cannot create private prefix fixture directory");
        fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace);
    }
    ~Scratch() { std::error_code ignored; fs::remove_all(path, ignored); }
};
void full(Report& report, const std::string& model, const std::string& integrity) {
    Values reference, blocked_rows;
    {
        d::Runtime runtime(model, options(integrity, 9)); ledger(report, "full.chunk9", runtime);
        auto s = runtime.create_session(); reference = s->prefill(tokens(9), false);
        state(report, "full.chunk9", *s, 9, 9); counts(report, "full.chunk9", *s, 180, 180, 0, 0, 0);
        compare(report, "full.reference_finite", reference, reference);
    }
    {
        d::Runtime runtime(model, options(integrity)); ledger(report, "full.chunk3", runtime);
        auto s = runtime.create_session(); auto result = s->prefill(tokens(9), false);
        compare(report, "full.chunk3_vs_chunk9", result, reference);
        state(report, "full.chunk3", *s, 9, 9); counts(report, "full.chunk3", *s, 180, 180, 0, 0, 0);
        s.reset(); s = runtime.create_session(); s->prefill(tokens(1), false);
        auto ids = tokens(9); const std::vector<int> rest(ids.begin() + 1, ids.end());
        blocked_rows = advance(report, "full.forward3", *s, rest, 3, 1);
        compare(report, "full.forward3_vs_chunk9", last_row(blocked_rows), reference);
        counts(report, "full.forward3", *s, 180, 180, 0, 0, 8);
    }
    {
        d::Runtime runtime(model, options(integrity, 1)); ledger(report, "full.chunk1", runtime);
        auto s = runtime.create_session(); auto result = s->prefill(tokens(9), false);
        compare(report, "full.chunk1_vs_chunk9", result, reference); state(report, "full.chunk1", *s, 9, 9);
        counts(report, "full.chunk1", *s, 180, 180, 0, 0, 0);
        s.reset(); s = runtime.create_session(); s->prefill(tokens(1), false);
        auto ids = tokens(9); const std::vector<int> rest(ids.begin() + 1, ids.end());
        auto rows = advance(report, "full.serial", *s, rest, 1, 1);
        compare(report, "full.serial_vs_forward3_all_rows", rows, blocked_rows);
        compare(report, "full.serial_vs_chunk9", last_row(rows), reference);
        counts(report, "full.serial", *s, 180, 180, 0, 0, 8);
    }
}
void ced(Report& report, d::Runtime& runtime) {
    auto s = runtime.create_session(); auto result = s->prefill(tokens(prompt_size), true);
    compare(report, "ced.finite", result, result); state(report, "ced", *s, 129, 129);
    counts(report, "ced", *s, 2580, 2560, 129, 128, 0);
}
void resume(Report& report, d::Runtime& runtime) {
    auto s = runtime.create_session(); s->prefill(tokens(prompt_size), true);
    auto reference = advance(report, "resume.serial", *s, suffix(), 1, prompt_size);
    counts(report, "resume.serial", *s, 2960, 2940, 129, 128, 19);
    s.reset(); s = runtime.create_session(); s->prefill(tokens(prompt_size), true);
    auto blocked = advance(report, "resume.forward3", *s, suffix(), 3, prompt_size);
    compare(report, "resume.forward3_vs_serial", blocked, reference);
    counts(report, "resume.forward3", *s, 2960, 2940, 129, 128, 19);
    s.reset(); s = runtime.create_session(); auto result = s->resume_history(tokens(148), 129, true);
    compare(report, "resume.public_api_vs_serial_final", result, last_row(reference));
    state(report, "resume.public_api", *s, 148, 129);
    counts(report, "resume.public_api", *s, 2960, 2940, 129, 128, 19);
}
void prefix(Report& report, d::Runtime& runtime, const fs::path& parent) {
    Scratch scratch(parent); const auto path = (scratch.path / "p129.prefix").string();
    auto s = runtime.create_session(); auto prompt_logits = s->prefill(tokens(prompt_size), true); s->save_prefix(path);
    rejected(report, "prefix.no_overwrite", "destination exists", [&] { s->save_prefix(path); });
    auto reference = advance(report, "prefix.cold", *s, suffix(), 3, prompt_size);
    counts(report, "prefix.cold", *s, 2960, 2940, 129, 128, 19);
    rejected(report, "prefix.no_generated_suffix", "completed original prompt", [&] { s->save_prefix((scratch.path / "suffix.prefix").string()); });
    s.reset(); s = runtime.create_session(); s->load_prefix(path, tokens(prompt_size)); state(report, "prefix.loaded", *s, 129, 129);
    rejected(report, "prefix.nonempty_load", "empty session", [&] { s->load_prefix(path, tokens(prompt_size)); });
    rejected(report, "prefix.wrong_ced_mode", "CED mode mismatch", [&] { s->prefill(tokens(prompt_size), false); });
    compare(report, "prefix.prompt_logits", s->prefill(tokens(prompt_size), true), prompt_logits);
    auto restored = advance(report, "prefix.restored", *s, suffix(), 3, prompt_size);
    compare(report, "prefix.all_suffix_rows", restored, reference); counts(report, "prefix.restored", *s, 380, 2940, 0, 128, 19);
    s.reset(); s = runtime.create_session(); s->load_prefix(path, tokens(prompt_size));
    compare(report, "prefix.resume_api_final", s->resume_history(tokens(148), 129, true), last_row(reference));
    state(report, "prefix.resume_api", *s, 148, 129); counts(report, "prefix.resume_api", *s, 380, 2940, 0, 128, 19);
    s.reset(); s = runtime.create_session(); auto wrong = tokens(prompt_size); wrong.back() = 0;
    rejected(report, "prefix.wrong_tokens", "token identity", [&] { s->load_prefix(path, wrong); });
    state(report, "prefix.after_wrong_tokens", *s, 0, 0);
    const auto truncated = scratch.path / "truncated.prefix"; { std::ofstream out(truncated, std::ios::binary); out << 'x'; }
    rejected(report, "prefix.truncated", "truncated prefix", [&] { s->load_prefix(truncated.string(), tokens(prompt_size)); });
    state(report, "prefix.after_truncated", *s, 0, 0); s->load_prefix(path, tokens(prompt_size));
    compare(report, "prefix.after_failures_final", s->resume_history(tokens(148), 129, true), last_row(reference));
    state(report, "prefix.after_failures", *s, 148, 129); counts(report, "prefix.after_failures", *s, 380, 2940, 0, 128, 19);
}
void transaction(Report& report, d::Runtime& runtime, const fs::path& parent) {
    const auto all = suffix(9); const std::vector<int> first(all.begin(), all.begin() + 5), pair(all.begin() + 6, all.begin() + 8);
    auto s = runtime.create_session(); s->prefill(tokens(prompt_size), true);
    const auto reference5 = s->forward(first), reference6 = s->decode(all[5]);
    const auto reference8 = s->forward(pair), reference9 = s->decode(all[8]);
    s.reset(); s = runtime.create_session(); s->prefill(tokens(prompt_size), true);
    std::ostringstream heat; runtime.write_expert_heat(heat); const auto before = s->stats(); s->begin_transaction();
    rejected(report, "transaction.nested", "nested transaction", [&] { s->begin_transaction(); });
    Scratch scratch(parent);
    rejected(report, "transaction.prefix_save", "completed original prompt", [&] { s->save_prefix((scratch.path / "active.prefix").string()); });
    compare(report, "transaction.tentative5", s->forward(first), reference5);
    state(report, "transaction.tentative", *s, 134, 129); s->rollback_transaction(); state(report, "transaction.rolled_back", *s, 129, 129);
    std::ostringstream restored_heat; runtime.write_expert_heat(restored_heat); const bool heat_equal = heat.str() == restored_heat.str();
    report.add("{\"kind\":\"rollback\",\"routing_heat_restored\":" + std::string(heat_equal ? "true" : "false") +
        ",\"stats_before\":" + stats_json(before) + ",\"stats_after\":" + stats_json(s->stats()) + ",\"stats_are_physical_work\":true}");
    require(heat_equal, "transaction: rejected work changed committed routing heat");
    compare(report, "transaction.replayed5", s->forward(first), reference5);
    compare(report, "transaction.after_rollback6", s->decode(all[5]), reference6);
    s->begin_transaction(); compare(report, "transaction.commit_rows", s->forward(pair), reference8); s->commit_transaction();
    state(report, "transaction.committed", *s, 137, 129);
    compare(report, "transaction.after_commit9", s->decode(all[8]), reference9); state(report, "transaction.final", *s, 138, 129);
    rejected(report, "transaction.no_snapshot", "no transaction", [&] { s->rollback_transaction(); });
}
void cancel(Report& report, d::Runtime& runtime) {
    auto s = runtime.create_session(); const auto reference = s->prefill(tokens(9), false); const auto next = s->decode(10);
    s.reset(); s = runtime.create_session(); s->prefill(tokens(9), false); s->cancel();
    rejected(report, "cancel.before_forward", "request cancelled", [&] { s->decode(10); });
    state(report, "cancel.position_unchanged", *s, 9, 9); s.reset(); s = runtime.create_session();
    compare(report, "cancel.new_session_prompt", s->prefill(tokens(9), false), reference);
    compare(report, "cancel.new_session_decode", s->decode(10), next); state(report, "cancel.reentry", *s, 10, 9);
}
} // namespace

int main(int argc, char** argv) {
    Report report; std::string model, integrity; bool initialized = false;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&] { require(i + 1 < argc, arg + " requires a value"); return std::string(argv[++i]); };
            if (arg == "--help") { std::cout << "check-runtime-state --model MAIN.gguf --integrity PROOF --output NEW.json [--cases full,ced,resume,prefix,transaction,cancel]\n"; return 0; }
            if (arg == "--model") model = next(); else if (arg == "--integrity") integrity = next();
            else if (arg == "--output") report.output = next(); else if (arg == "--cases") report.selected = next();
            else throw std::invalid_argument("unknown option: " + arg);
        }
        require(!model.empty() && !integrity.empty() && !report.output.empty(), "model, integrity and output are required");
        require(!fs::exists(report.output), "report output already exists");
        const auto parent = report.output.has_parent_path() ? report.output.parent_path() : fs::path(".");
        require(fs::is_directory(parent), "report parent directory must already exist");
        const std::set<std::string> allowed{"full", "ced", "resume", "prefix", "transaction", "cancel"};
        std::set<std::string> selected; std::istringstream names(report.selected); std::string name;
        while (std::getline(names, name, ',')) require(allowed.count(name) && selected.insert(name).second, "unknown or duplicate case: " + name);
        require(!selected.empty(), "at least one case is required"); initialized = true; report.write("running");
        auto run = [&](const std::string& key, const auto& operation) {
            operation(); report.add("{\"kind\":\"case\",\"name\":" + quote(key) + ",\"status\":\"passed\"}"); report.write("running");
        };
        if (selected.count("full")) run("full", [&] { full(report, model, integrity); });
        if (selected.size() > selected.count("full")) {
            d::Runtime runtime(model, options(integrity)); ledger(report, "shared.chunk3", runtime);
            if (selected.count("ced")) run("ced", [&] { ced(report, runtime); });
            if (selected.count("resume")) run("resume", [&] { resume(report, runtime); });
            if (selected.count("prefix")) run("prefix", [&] { prefix(report, runtime, parent); });
            if (selected.count("transaction")) run("transaction", [&] { transaction(report, runtime, parent); });
            if (selected.count("cancel")) run("cancel", [&] { cancel(report, runtime); });
        }
        report.write("passed"); std::cout << "{\"status\":\"passed\",\"output\":" << quote(report.output.string()) << "}\n"; return 0;
    } catch (const std::exception& error) {
        if (initialized) {
            try { report.write("failed", error.what()); } catch (const std::exception& write_error) { std::cerr << write_error.what() << '\n'; }
        }
        std::cerr << "check-runtime-state: " << error.what() << '\n'; return 1;
    }
}
