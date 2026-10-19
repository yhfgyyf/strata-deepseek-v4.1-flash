// Standalone control of the real PhaseCache consumer fence, using tiny synthetic blobs.
// The positive and private negative builds use this identical source. No device asserts.
#include "phase_cache.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace d = strata::models::deepseek41;
namespace {
using Clock = std::chrono::steady_clock;
struct Gate {
    std::atomic<bool> entered{false}, open{false}, timed_out{false};
    std::atomic<bool> watchdog_timeout{false}, scheduled_release{false};
};
void CUDART_CB hold_gate(void* opaque) {
    auto& gate = *static_cast<Gate*>(opaque);
    gate.entered = true;
    const auto deadline = Clock::now() + std::chrono::seconds(3);
    while (!gate.open.load()) {
        if (Clock::now() >= deadline) { gate.timed_out = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
__global__ void checksum(const uint8_t* bytes, uint64_t size, unsigned long long* output) {
    unsigned long long sum = 0;
    for (uint64_t i = 0; i < size; ++i) sum += bytes[i];
    *output = sum;
}
struct Check { std::string name; int code; bool cleanup; };
std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32) out << '?';
        else out << c;
    }
    return out.str() + '"';
}
uint8_t pattern(d::ExpertKey key) { return 1 + (key.first * 17 + key.second) % 251; }
}

int main(int argc, char** argv) {
    bool negative = false, specified = false, pageable = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if ((option == "--expect-positive" || option == "--expect-negative") && !specified) {
            negative = option == "--expect-negative"; specified = true;
        } else if (option == "--pageable" && !pageable) pageable = true;
        else { std::cerr << "invalid phase reader control arguments\n"; return 2; }
    }
    if (!specified || (negative && pageable)) {
        std::cerr << "use --expect-positive [--pageable] or --expect-negative (pinned)\n";
        return 2;
    }
    std::vector<Check> checks;
    std::string error;
    bool cuda_failure = false, cleanup_ok = true, reader_pending_before = false;
    bool reader_pending_after = false, restore_before_reader = false, window_checked = false;
    bool sum_read = false, checksum_queued = false;
    uint64_t expected_sum = 0, actual_sum = 0;
    uint64_t front_bytes = 0, borrow_bytes = 0, restore_bytes_before_release = 0;
    int front_expert = -1;
    int reader_before = -1, reader_after = -1, restore_observed = -1;
    Gate control;
    auto checked = [&](cudaError_t code, const char* name) {
        checks.push_back({name, static_cast<int>(code), false});
        if (code != cudaSuccess) { cuda_failure = true; throw std::runtime_error(name); }
    };
    auto cleaned = [&](cudaError_t code, const char* name) {
        checks.push_back({name, static_cast<int>(code), true});
        if (code != cudaSuccess) { cuda_failure = true; cleanup_ok = false; }
    };
    try {
        // Stream/event constructors precede all queued work. Every allocation is before the gate.
        d::Stream main_stream, upload_stream, gate_stream;
        d::Event gate_ready, reader_done, restore_done;
        std::map<d::ExpertKey, std::vector<uint8_t>> cold;
        std::unique_ptr<d::PhaseCache> cache;
        std::unique_ptr<strata::core::DeviceArena> result;
        unsigned long long* device_sum = nullptr;
        std::thread watchdog;
        d::ExecutionStats stats;
        try {
            std::array<uint64_t, 40> sizes{};
            for (int l = 0; l < 40; ++l) sizes[l] = l < 14 ? 768 : 512;
            std::vector<d::ExpertKey> hot;
            for (int e = 0; e < 33; ++e) for (int l = 0; l < 40; ++l) hot.emplace_back(l, e);
            auto plan = d::make_phase_plan(sizes, hot, {}, 1536);
            for (const auto& slice : plan.encoder)
                cold.emplace(slice.key, std::vector<uint8_t>(slice.bytes, pattern(slice.key)));
            cache = std::make_unique<d::PhaseCache>(plan, [&](d::ExpertKey key) {
                const auto it = cold.find(key); return it == cold.end() ? nullptr : it->second.data();
            }, upload_stream.value, 0, pageable);
            for (size_t i = 0; i < plan.decode.size(); ++i) {
                const auto& slice = plan.decode[i];
                std::vector<uint8_t> bytes(slice.bytes, pattern(slice.key));
                cache->initialize(i, bytes.data());
                checked(cudaStreamSynchronize(upload_stream.value), "initialize-upload-sync");
            }
            if (!cache->begin_prefill(4096, 64, true, main_stream.value, stats))
                throw std::runtime_error("synthetic long prefill did not borrow");
            checked(cudaStreamSynchronize(main_stream.value), "prepare-main-sync");
            checked(cudaStreamSynchronize(upload_stream.value), "prepare-upload-sync");
            const auto front = std::find_if(plan.encoder.begin(), plan.encoder.end(),
                                           [](const d::PhaseSlice& slice) { return slice.key.first == 0; });
            if (front == plan.encoder.end()) throw std::runtime_error("missing layer-zero temporary slice");
            const auto* pointer = cache->gpu(front->key, main_stream.value, stats, 0);
            if (!pointer) throw std::runtime_error("missing temporary reader pointer");
            front_expert = front->key.second;
            front_bytes = front->bytes;
            borrow_bytes = plan.bytes;
            expected_sum = uint64_t(pattern(front->key)) * front->bytes;
            result = std::make_unique<strata::core::DeviceArena>(sizeof(unsigned long long));
            device_sum = static_cast<unsigned long long*>(result->alloc(sizeof(unsigned long long)));
            // Resolve lazy kernel/module loading before holding any stream behind the host gate.
            checksum<<<1, 1, 0, main_stream.value>>>(pointer, front->bytes, device_sum);
            checked(cudaGetLastError(), "warm-checksum-launch");
            checked(cudaStreamSynchronize(main_stream.value), "warm-checksum-sync");
            const auto release_after = Clock::now() + (pageable ? std::chrono::milliseconds(100)
                                                               : std::chrono::milliseconds(1000));
            watchdog = std::thread([&, release_after] {
                while (!control.open.load()) {
                    if (Clock::now() >= release_after) {
                        if (pageable) control.scheduled_release = true;
                        else control.watchdog_timeout = true;
                        control.open = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
            checked(cudaLaunchHostFunc(gate_stream.value, hold_gate, &control), "launch-host-gate");
            checked(cudaEventRecord(gate_ready.value, gate_stream.value), "record-gate-ready");
            checked(cudaStreamWaitEvent(main_stream.value, gate_ready.value, 0), "wait-host-gate");
            checksum<<<1, 1, 0, main_stream.value>>>(pointer, front->bytes, device_sum);
            checked(cudaGetLastError(), "launch-checksum");
            checksum_queued = true;
            checked(cudaEventRecord(reader_done.value, main_stream.value), "record-reader-done");
            reader_before = static_cast<int>(cudaEventQuery(reader_done.value));
            if (reader_before != static_cast<int>(cudaErrorNotReady)) {
                if (reader_before != static_cast<int>(cudaSuccess)) checked(static_cast<cudaError_t>(reader_before), "query-reader-before");
                throw std::runtime_error("actual checksum consumer was not pending before restoration");
            }
            reader_pending_before = true;
            cache->encoder_layer_done(0, main_stream.value, stats);
            restore_bytes_before_release = stats.phase_restore_h2d_bytes;
            if (!restore_bytes_before_release) throw std::runtime_error("no restoration targeted the pending reader's layer");
            checked(cudaEventRecord(restore_done.value, upload_stream.value), "record-restoration-done");
            const auto observation_end = Clock::now() + std::chrono::milliseconds(20);
            do {
                restore_observed = static_cast<int>(cudaEventQuery(restore_done.value));
                reader_after = static_cast<int>(cudaEventQuery(reader_done.value));
                for (auto code : {restore_observed, reader_after})
                    if (code != static_cast<int>(cudaSuccess) && code != static_cast<int>(cudaErrorNotReady))
                        checked(static_cast<cudaError_t>(code), "query-observation");
                reader_pending_after = reader_after == static_cast<int>(cudaErrorNotReady);
                restore_before_reader = restore_before_reader ||
                    (restore_observed == static_cast<int>(cudaSuccess) && reader_pending_after);
                window_checked = window_checked || reader_pending_after;
                if (restore_before_reader || !reader_pending_after) break;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            } while (Clock::now() < observation_end);
        } catch (const std::exception& caught) {
            error = caught.what();
        } catch (...) {
            error = "unknown experiment exception";
        }
        // No exception may escape while a held gate or queued GPU reader owns test memory.
        control.open = true;
        if (watchdog.joinable()) watchdog.join();
        cleaned(cudaStreamSynchronize(main_stream.value), "drain-main-before-restore");
        cleaned(cudaStreamSynchronize(upload_stream.value), "drain-upload-before-restore");
        cleaned(cudaStreamSynchronize(gate_stream.value), "drain-gate-before-restore");
        if (cache) {
            try { cache->finish_prefill(main_stream.value, stats); }
            catch (const std::exception& caught) {
                cleanup_ok = false;
                checks.push_back({"finish-prefill-exception", -1, true});
                if (error.empty()) error = caught.what();
            }
        }
        cleaned(cudaStreamSynchronize(main_stream.value), "drain-main");
        cleaned(cudaStreamSynchronize(upload_stream.value), "drain-upload");
        cleaned(cudaStreamSynchronize(gate_stream.value), "drain-gate");
        cleaned(cudaDeviceSynchronize(), "device-sync-with-resources-alive");
        if (checksum_queued && device_sum && cleanup_ok) {
            const auto code = cudaMemcpy(&actual_sum, device_sum, sizeof(actual_sum), cudaMemcpyDeviceToHost);
            cleaned(code, "read-checksum");
            sum_read = code == cudaSuccess;
        }
        // cache, staging, result arena, events and streams are destroyed only after the drains.
    } catch (const std::exception& caught) {
        if (error.empty()) error = caught.what();
        cuda_failure = true;
        cleaned(cudaDeviceSynchronize(), "outer-exception-device-sync");
    } catch (...) {
        error = "unknown resource exception";
        cuda_failure = true;
        cleaned(cudaDeviceSynchronize(), "outer-exception-device-sync");
    }
    cleaned(cudaDeviceSynchronize(), "device-sync-after-resource-scope");
    cleaned(cudaDeviceReset(), "device-reset-after-resource-scope");
    const bool gate_ok = !control.timed_out && !control.watchdog_timeout;
    const bool witness = reader_pending_before && (pageable || window_checked);
    const bool valid = error.empty() && !cuda_failure && cleanup_ok && gate_ok && witness && sum_read;
    const bool expected_failure = valid && negative && restore_before_reader && actual_sum != expected_sum;
    const bool positive_pass = valid && !negative && !restore_before_reader && actual_sum == expected_sum;
    const int exit_code = expected_failure ? 42 : positive_pass ? 0 : 1;
    const char* status = expected_failure ? "expected_missing_consumer_wait_detected" :
                         positive_pass ? "positive_passed" :
                         (cuda_failure || !cleanup_ok) ? "cuda_or_cleanup_failed" : "failed_or_inconclusive";
    std::cout << std::boolalpha << "{\"status\":" << json_string(status)
              << ",\"expect_negative\":" << negative << ",\"pageable\":" << pageable
              << ",\"reader_pending_before\":" << reader_pending_before
              << ",\"reader_pending_after\":" << reader_pending_after
              << ",\"reader_before_code\":" << reader_before << ",\"reader_after_code\":" << reader_after
              << ",\"restore_observed_code\":" << restore_observed
              << ",\"restoration_completed_before_reader\":" << restore_before_reader
              << ",\"window_checked\":" << window_checked << ",\"gate_entered\":" << control.entered.load()
              << ",\"gate_timed_out\":" << control.timed_out.load()
              << ",\"watchdog_timeout\":" << control.watchdog_timeout.load()
              << ",\"pageable_scheduled_release\":" << control.scheduled_release.load()
              << ",\"checksum_read\":" << sum_read << ",\"expected_sum\":" << expected_sum
              << ",\"front_expert\":" << front_expert << ",\"front_bytes\":" << front_bytes
              << ",\"borrow_bytes\":" << borrow_bytes << ",\"restore_bytes_before_release\":" << restore_bytes_before_release
              << ",\"actual_sum\":" << actual_sum << ",\"cuda_failure\":" << cuda_failure
              << ",\"cleanup_ok\":" << cleanup_ok << ",\"error\":" << json_string(error)
              << ",\"exit_code\":" << exit_code << ",\"cuda_checks\":[";
    for (size_t i = 0; i < checks.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"name\":" << json_string(checks[i].name) << ",\"code\":" << checks[i].code
                  << ",\"cleanup\":" << checks[i].cleanup << '}';
    }
    std::cout << "]}\n" << std::flush;
    return exit_code;
}
