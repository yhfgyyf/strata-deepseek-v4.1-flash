#include "runtime_internal.hpp"
#include "expert_pipeline.hpp"
#include "bounded_moe.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include "strata/models/deepseek41/format_down.hpp"
#include "strata/models/deepseek41/mixed_expert.hpp"
#include "strata/models/deepseek41/fused_moe.hpp"
#include "strata/kernels/cpu/deepseek41_down.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <functional>
#include <cstring>
#include <cstdlib>

namespace strata::models::deepseek41 {
std::atomic<bool> fused_cold_first{true};
void fused_moe_set_cold_first(bool value){fused_cold_first.store(value);}
namespace {
std::atomic<bool> route_group_enabled{true};
std::atomic<bool> shared_early_all{true};
// Each retained format/row bucket passed independent ABBA confirmation.
bool shared_early_selected(int gu_type,int down_type,int tokens){
    unsigned mask=0;
    if(gu_type==16&&down_type==16)mask=0x110u;
    else if(gu_type==16&&down_type==18)mask=0x1d4u;
    else if(gu_type==17&&down_type==39)mask=0x8u;
    else if(gu_type==18&&down_type==18)mask=0x4eu;
    else if(gu_type==18&&down_type==21)mask=0x12eu;
    else if(gu_type==22&&down_type==21)mask=0x4u;
    else if(gu_type==22&&down_type==39)mask=0x104u;
    return tokens>=1&&tokens<=8&&(mask&(1u<<tokens));
}
}
void route_group_set_enabled(bool value){route_group_enabled.store(value);}
void shared_early_all_set_enabled(bool value){shared_early_all.store(value);}
bool Runtime::Impl::short_gpu_moe(int tokens,int phase) const {
    return phase==2&&tokens>=1&&tokens<=8&&options.slots==1&&options.arithmetic==Arithmetic::production&&
        (options.experts==ExpertMode::gpu||(options.experts==ExpertMode::hybrid&&options.pcie_frac==1));
}
bool Runtime::Impl::compact_routes(int tokens,int phase) const {
    return options.route_compact_handoff&&!options.decode_cpu_misses&&short_gpu_moe(tokens,phase)&&options.decode_graph&&
        (decode_segment_mask&2)&&((1u<<tokens)&0x1b4u)&&route_group_enabled.load();
}

// ---- Route-prediction diagnostic (STRATA_DS41_ROUTE_PREDICT=1, decode only; changes no result). Predictors:
// 0 = layer L's MoE input through layer L+1's router, 1 = the same through layer L+2's router, 2 = the mHC estimate
// rms(hc_pre(h, attn_pre_L)) at layer L's attention input. Scored at layer L's MoE against its cold experts.
namespace {
struct RoutePredict {
    bool on = [] { const char* v = std::getenv("STRATA_DS41_ROUTE_PREDICT"); return v && v[0] == '1'; }();
    float* dev = nullptr;
    std::array<std::vector<float>, 40> bias;
    std::array<std::array<std::vector<std::array<int, 12>>, 40>, 3> pred;   // [kind][layer][token] top-12 ids
    std::array<std::array<uint64_t, 3>, 3> hit{};   // [kind][K 6/8/12] cold experts covered
    std::array<std::array<uint64_t, 3>, 3> extra{}; // non-resident predicted but unused
    uint64_t cold = 0, needed = 0, layers = 0;
    std::array<uint64_t, 3> all{};                   // routed (token,expert) recall at K = 6
};
RoutePredict& rp() { static RoutePredict r; return r; }
}
bool route_predict_on() { return rp().on; }
void Runtime::Impl::route_predict(int kind, int layer, const float* x, int tokens, Workspace& w) {
    auto& r = rp();
    if (!r.on || layer < 0 || layer >= 40 || tokens < 1 || tokens > 8) return;
    if (!r.dev) gpu_check(cudaMalloc(&r.dev, 8 * 384 * sizeof(float)));
    if (r.bias[layer].empty()) {
        r.bias[layer].resize(384);
        const auto& b = weights->at("blk." + std::to_string(layer) + ".exp_probs_b.bias");
        gpu_check(cudaMemcpy(r.bias[layer].data(), b.data, 384 * 4, cudaMemcpyDeviceToHost));
    }
    matmul(tensor(layer, "ffn_gate_inp"), x, r.dev, tokens, w, 0, 0, 0, 0, false);
    std::vector<float> logits(size_t(tokens) * 384);
    gpu_check(cudaMemcpyAsync(logits.data(), r.dev, logits.size() * 4, cudaMemcpyDeviceToHost, stream.value));
    gpu_check(cudaStreamSynchronize(stream.value));
    auto& out = r.pred[kind][layer];
    out.assign(tokens, {});
    for (int t = 0; t < tokens; ++t) {
        std::vector<std::pair<float, int>> s(384);
        for (int e = 0; e < 384; ++e) {
            const float v = logits[size_t(t) * 384 + e];
            s[e] = {std::sqrt(std::max(v, 0.f) + std::log1p(std::exp(-std::fabs(v)))) + r.bias[layer][e], e};
        }
        std::partial_sort(s.begin(), s.begin() + 12, s.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        for (int k = 0; k < 12; ++k) out[t][k] = s[k].second;
    }
}
static void route_predict_score(int layer, const int* ids, int tokens, const std::vector<int>& cold_experts,
                                const std::function<bool(int)>& resident) {
    auto& r = rp();
    if (!r.on) return;
    r.cold += cold_experts.size();
    ++r.layers;
    std::array<bool, 384> needed{};
    for (int i = 0; i < tokens * 6; ++i) needed[ids[i]] = true;
    for (int kind = 0; kind < 3; ++kind) {
        const auto& p = r.pred[kind][layer];
        if (int(p.size()) != tokens) continue;
        const int ks[3] = {6, 8, 12};
        for (int ki = 0; ki < 3; ++ki) {
            std::array<bool, 384> pick{};
            for (int t = 0; t < tokens; ++t) for (int k = 0; k < ks[ki]; ++k) pick[p[t][k]] = true;
            for (int e : cold_experts) r.hit[kind][ki] += pick[e];
            for (int e = 0; e < 384; ++e) if (pick[e] && !needed[e] && !resident(e)) ++r.extra[kind][ki];
        }
        for (int t = 0; t < tokens; ++t) for (int k = 0; k < 6; ++k) {
            const int e = ids[t * 6 + k];
            r.all[kind] += std::find(p[t].begin(), p[t].begin() + 6, e) != p[t].begin() + 6;
        }
    }
    r.needed += tokens * 6;
    for (auto& k : r.pred) k[layer].clear();
    if (r.layers % 4000 == 0) {
        std::fprintf(stderr, "route-predict layers=%llu cold=%llu routes=%llu", (unsigned long long)r.layers, (unsigned long long)r.cold,
                     (unsigned long long)r.needed);
        const char* names[3] = {"next", "next2", "mhc"};
        for (int kind = 0; kind < 3; ++kind)
            std::fprintf(stderr, " | %s route-recall@6 %.3f cold-recall@6/8/12 %.3f/%.3f/%.3f extra/layer %.2f/%.2f/%.2f", names[kind],
                         double(r.all[kind]) / std::max<uint64_t>(1, r.needed), double(r.hit[kind][0]) / std::max<uint64_t>(1, r.cold),
                         double(r.hit[kind][1]) / std::max<uint64_t>(1, r.cold), double(r.hit[kind][2]) / std::max<uint64_t>(1, r.cold),
                         double(r.extra[kind][0]) / r.layers, double(r.extra[kind][1]) / r.layers, double(r.extra[kind][2]) / r.layers);
        std::fprintf(stderr, "\n");
    }
}

// ---- Speculative cold-expert uploads (see SpecPrefetch). Results are unchanged: the real route decides every expert;
// a predicted expert found in the private buffer is read from there instead of being uploaded on demand.
std::atomic<bool> spec_prefetch_enabled{true};
void spec_prefetch_set_enabled(bool value) { spec_prefetch_enabled.store(value); }
std::atomic<bool> route_side_enabled{true};
void route_side_set_enabled(bool value) { route_side_enabled.store(value); }
SpecPrefetch* Runtime::Impl::spec_state() {
    static const bool on = [] { const char* v = std::getenv("STRATA_DS41_SPEC_PREFETCH"); return v && v[0] == '1'; }();
    if (!on || !spec_prefetch_enabled.load()) return nullptr;
    if (!spec) {
        spec = std::make_unique<SpecPrefetch>();
        auto env = [](const char* n, double d) { const char* v = std::getenv(n); return v ? std::atof(v) : d; };
        spec->threshold = env("STRATA_DS41_SPEC_THRESHOLD", 0.5);
        spec->rank = int(env("STRATA_DS41_SPEC_RANK", 2));
        spec->per_layer = int(env("STRATA_DS41_SPEC_PER_LAYER", 3));
        spec->nslots = int(env("STRATA_DS41_SPEC_SLOTS", 12));
        for (int l = 0; l < 40; ++l) spec->stride = std::max<uint64_t>(spec->stride, experts->format(l).bytes);
        spec->stride = (spec->stride + 255) / 256 * 256;
        size_t free = 0, total = 0;
        gpu_check(cudaMemGetInfo(&free, &total));
        if (free < spec->nslots * spec->stride + (64ull << 20) || cudaMalloc(&spec->base, spec->nslots * spec->stride) != cudaSuccess) {
            cudaGetLastError(); spec->failed = true;
            std::fprintf(stderr, "spec-prefetch: buffer unavailable (free %zu MB)\n", free >> 20);
        } else {
            gpu_check(cudaMalloc(&spec->dx, 8 * 5120 * 4)); gpu_check(cudaMalloc(&spec->dlogits, 8 * 384 * 4));
            gpu_check(cudaMalloc(&spec->dweights, 8 * 6 * 4)); gpu_check(cudaMalloc(&spec->dids, 8 * 6 * 4));
            gpu_check(cudaMallocHost(&spec->hids, 8 * 6 * 4));
            spec->key.assign(spec->nslots, -1); spec->age.assign(spec->nslots, 0); spec->used_valid.assign(spec->nslots, false);
            for (int i = 0; i < spec->nslots; ++i) { spec->copied.push_back(std::make_unique<Event>()); spec->used.push_back(std::make_unique<Event>()); }
            std::fprintf(stderr, "spec-prefetch: %d slots x %llu bytes, rank<%d, <=%d per layer, threshold %.2f\n", spec->nslots,
                         (unsigned long long)spec->stride, spec->rank, spec->per_layer, spec->threshold);
        }
    }
    return spec->failed ? nullptr : spec.get();
}
void Runtime::Impl::spec_predict(int layer, int tokens, Workspace& w, const float* h, const float* attn_pre, const uint8_t* image) {
    SpecPrefetch* sp = spec_state();
    if (!sp || !sp->active() || tokens < 1 || tokens > 8) return;
    const auto& g = model.geometry();
    gpu_check(hc_pre_bf16_gpu(h, attn_pre, sp->dx, 5120, tokens, stream.value));
    gpu_check(rms_bf16_gpu(sp->dx, static_cast<const float*>(tensor(layer, "ffn_norm").data), sp->dx, 5120, tokens, g.norm_eps, stream.value));
    matmul(tensor(layer, "ffn_gate_inp"), sp->dx, sp->dlogits, tokens, w, 0, 0, 0, 0, false);
    const auto& bias = weights->at("blk." + std::to_string(layer) + ".exp_probs_b.bias");
    const auto* text_bias = static_cast<const float*>(bias.data);
    const float* image_bias = text_bias;
    if (gsq_vision_bias) image_bias = static_cast<const float*>(gsq_vision_bias->base()) + layer * 384;
    else if (!model.gsq_rco_1017()) image_bias = static_cast<const float*>(weights->at("blk." + std::to_string(layer) + ".exp_probs_b_vl.bias").data);
    gpu_check(route_parallel_gpu(sp->dlogits, text_bias, image_bias, image, sp->dids, sp->dweights, 384, 6, tokens, 1.5f, stream.value));
    gpu_check(cudaMemcpyAsync(sp->hids, sp->dids, tokens * 6 * sizeof(int), cudaMemcpyDeviceToHost, stream.value));
    gpu_check(cudaEventRecord(sp->ready.value, stream.value));
    sp->pred_layer = layer; sp->pred_tokens = tokens; ++sp->predictions;
}
void Runtime::Impl::spec_issue(int layer, int tokens) {
    SpecPrefetch* sp = spec.get();
    if (!sp || sp->failed || sp->pred_layer != layer) return;
    sp->pred_layer = -1;
    gpu_check(cudaEventSynchronize(sp->ready.value));
    const int nt = std::min(tokens, sp->pred_tokens);
    int taken = 0;
    for (int k = 0; k < std::min(sp->rank, 6) && taken < sp->per_layer; ++k)
        for (int t = 0; t < nt && taken < sp->per_layer; ++t) {
            const int e = sp->hids[t * 6 + k];
            if (e < 0 || e >= 384) continue;
            const int key = layer * 384 + e;
            if (experts->gpu_resident(layer, e) || (decode_raw_cache && decode_raw_cache->peek(key))) continue;
            if (std::find(sp->key.begin(), sp->key.end(), key) != sp->key.end()) continue;
            const uint8_t* source = experts->pinned_blob(layer, e);
            if (!source) continue;
            int slot = 0;
            for (int i = 1; i < sp->nslots; ++i) if (sp->age[i] < sp->age[slot]) slot = i;
            if (sp->used_valid[slot]) gpu_check(cudaStreamWaitEvent(sp->stream.value, sp->used[slot]->value, 0));
            gpu_check(cudaMemcpyAsync(sp->base + uint64_t(slot) * sp->stride, source, experts->format(layer).bytes, cudaMemcpyHostToDevice,
                                      sp->stream.value));
            gpu_check(cudaEventRecord(sp->copied[slot]->value, sp->stream.value));
            sp->key[slot] = key; sp->age[slot] = ++sp->clock; ++sp->issued; ++taken;
        }
}
const uint8_t* Runtime::Impl::spec_find(int key) {
    SpecPrefetch* sp = spec.get();
    if (!sp || sp->failed) return nullptr;
    const auto at = std::find(sp->key.begin(), sp->key.end(), key);
    if (at == sp->key.end()) return nullptr;
    const int slot = int(at - sp->key.begin());
    gpu_check(cudaStreamWaitEvent(stream.value, sp->copied[slot]->value, 0));
    sp->age[slot] = ++sp->clock; sp->pending.push_back(slot); ++sp->hits;
    return sp->base + uint64_t(slot) * sp->stride;
}
void Runtime::Impl::route(int layer,int tokens,Workspace& w,int phase) {
    matmul(tensor(layer,"ffn_gate_inp"),w.x,w.branch,tokens,w,0,0,0,0,false);
    const auto& bias=weights->at("blk."+std::to_string(layer)+".exp_probs_b.bias");
    const auto* text_bias = static_cast<const float*>(bias.data);
    const float* image_bias = text_bias;
    if (gsq_vision_bias)
        image_bias = static_cast<const float*>(gsq_vision_bias->base()) + layer * 384;
    else if (!model.gsq_rco_1017())
        image_bias = static_cast<const float*>(weights->at("blk."+std::to_string(layer)+".exp_probs_b_vl.bias").data);
    gpu_check((options.parallel_router?route_parallel_gpu:route_gpu)(w.branch,text_bias,image_bias,w.image,
        w.route_ids,w.route_weights,384,6,tokens,1.5f,stream.value));
    static const bool side_env=[]{const char* v=std::getenv("STRATA_DS41_ROUTE_SIDE");return !v||v[0]!='0';}();
    const bool side_enabled=side_env&&route_side_enabled.load();
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    gpu_check(cudaStreamIsCapturing(stream.value,&capture));
    const bool side=side_enabled&&phase==2&&capture==cudaStreamCaptureStatusNone;
    // A route without its MoE (an exception unwound between them) leaves the side copy pending: finish it first.
    if(route_side_pending){gpu_check(cudaEventSynchronize(route_ready.value));route_side_pending=false;}
    const cudaStream_t copies=side?route_stream.value:stream.value;
    auto fork=[&]{if(side){gpu_check(cudaEventRecord(route_done.value,stream.value));gpu_check(cudaStreamWaitEvent(copies,route_done.value,0));}};
    if(compact_routes(tokens,phase)){
        gpu_check(route_handoff_short_gpu(w.route_ids,w.route_weights,w.group_rows,w.group_slots,w.group_weights,w.route_handoff,tokens,stream.value));
        fork();
        gpu_check(cudaMemcpyAsync(w.host_ids,w.route_handoff,(tokens*6+1)*sizeof(int),cudaMemcpyDeviceToHost,copies));
    }else{
        if(route_group_enabled.load()&&tokens>1&&tokens<=8)
            gpu_check(route_group_short_gpu(w.route_ids,w.route_weights,w.group_rows,w.group_slots,w.group_weights,tokens,stream.value));
        fork();
        gpu_check(cudaMemcpyAsync(w.host_ids,w.route_ids,tokens*6*sizeof(int),cudaMemcpyDeviceToHost,copies));
        gpu_check(cudaMemcpyAsync(w.host_weights,w.route_weights,tokens*6*sizeof(float),cudaMemcpyDeviceToHost,copies));
    }
    // The host reads host_ids/host_weights only after route_ready; the main stream never rewrites the copied device
    // buffers before the host has synchronized on it (the next writer is the next layer's router).
    if(side){gpu_check(cudaEventRecord(route_ready.value,copies));route_side_pending=true;}
    if(phase==2&&route_predict_on()){route_predict(0,layer+1,w.x,tokens,w);route_predict(1,layer+2,w.x,tokens,w);}
}
void Runtime::Impl::moe(int layer,int tokens,Workspace& w,ExecutionStats& stats,int phase,std::vector<int>* deferred_routes,bool routes_ready) {
    prefill::LedgerScope moe_trace(options.trace,"ds41.moe");
    const bool prefill=phase!=2;
    const bool cpu_misses=options.decode_cpu_misses&&!prefill;
    const bool mixed_fast=options.arithmetic==Arithmetic::production&&expert_mixed_fast_enabled();
    if(decode_raw_cache && !prefill && tokens>8)decode_raw_cache->clear(upload.value);
    const auto& f=experts->format(layer);
    const bool compact=compact_routes(tokens,phase);
    // Submission order only: the shared expert and the routed experts write disjoint buffers.
    const bool early_shared=options.shared_early_submit&&short_gpu_moe(tokens,phase)&&options.decode_graph&&
        (shared_early_all.load()||shared_early_selected(f.gu_type,f.d_type,tokens));
    // The FFN RMS kernel already wrote x's Q8_1 blocks (kept slot 0) for this layer.
    const bool kept_x_q8=phase==2&&tokens>=2&&tokens<=8&&q8_source[0]==w.x&&q8_tokens[0]==tokens&&w.dense_input_q8_x;
    auto shared=[&]{
        const auto& gate=tensor(layer,"ffn_gate_shexp");
        const auto& up=tensor(layer,"ffn_up_shexp");
        // Keep the validated shared path: the experimental grouped Q8 chain
        // has no material speed gain and two activation-reference failures.
        if(options.shared_q8_pair && options.arithmetic==Arithmetic::production && tokens<=8 &&
           options.dense_q8_1 && (tokens==1||options.dense_q8_batch) &&
           gate.packed_bf16 && up.packed_bf16 && gate.type==8 && up.type==8 &&
           gate.rows==2304 && up.rows==2304 && gate.cols==5120 && up.cols==5120){
            kernels::native_quantize_q8_1(w.x,w.dense_input_q8,5120,tokens,stream.value);
            if(!kernels::native_mmvq_pair(8,gate.data,up.data,w.dense_input_q8,w.shared_gate,w.shared_up,
                                         5120,2304,tokens,stream.value)){
                kernels::native_q8_0_mmvq(gate.data,w.dense_input_q8,w.shared_gate,5120,2304,tokens,stream.value);
                kernels::native_q8_0_mmvq(up.data,w.dense_input_q8,w.shared_up,5120,2304,tokens,stream.value);
            }
            gpu_check(round_bf16_strided_gpu(w.shared_gate,tokens,2304,2304,stream.value));
            gpu_check(round_bf16_strided_gpu(w.shared_up,tokens,2304,2304,stream.value));
        }else if(options.arithmetic==Arithmetic::production && tokens>=2 && tokens<=8 && options.dense_q8_1 && options.dense_q8_batch &&
                 gate.packed_bf16 && up.packed_bf16 && gate.type==13 && up.type==13 && gate.rows==2304 && up.rows==2304 &&
                 gate.cols==5120 && up.cols==5120 &&
                 (kept_x_q8||(kernels::native_quantize_q8_1(w.x,w.dense_input_q8,5120,tokens,stream.value),true),
                  kernels::native_mmvq_pair(13,gate.data,up.data,kept_x_q8?w.dense_input_q8_x:w.dense_input_q8,w.shared_gate,w.shared_up,5120,2304,tokens,stream.value,
                                            kernels::native_mmvq_round2_enabled()))){
            // One launch for both Q5_K projections (matmul's quantize, mmvq and round, per matrix); the BF16 rounding
            // is in the pair's store when shared_pair_round is set.
            if(!kernels::native_mmvq_round2_enabled()){
                gpu_check(round_bf16_strided_gpu(w.shared_gate,tokens,2304,2304,stream.value));
                gpu_check(round_bf16_strided_gpu(w.shared_up,tokens,2304,2304,stream.value));
            }
        }else{
            matmul(gate,w.x,w.shared_gate,tokens,w);
            const bool reuse=options.shared_q8_reuse&&options.arithmetic==Arithmetic::production&&
                tokens>=1&&tokens<=8&&options.dense_q8_1&&(tokens==1||options.dense_q8_batch)&&
                gate.packed_bf16&&up.packed_bf16&&gate.type==13&&up.type==13&&
                gate.rows==2304&&up.rows==2304&&gate.cols==5120&&up.cols==5120;
            matmul(up,w.x,w.shared_up,tokens,w,0,0,0,0,true,reuse);
        }
        gpu_check((options.arithmetic==Arithmetic::production?swiglu_bf16_gpu:swiglu_gpu)(w.shared_gate,w.shared_up,nullptr,w.shared_act,2304,tokens,10.f,stream.value));
        matmul(tensor(layer,"ffn_down_shexp"),w.shared_act,w.shared_out,tokens,w);
    };
    uint64_t shared_captures=0,shared_launches=0;
    auto submit_shared=[&]{
        prefill::LedgerScope shared_trace(options.trace,"ds41.moe.shared.submit");
        if(options.decode_graph && phase==2 && tokens<=8){
            auto& graph=shared_graphs[layer][tokens-1];std::string error;
            if(!graph.valid()){
                if(!graph.begin(stream.value,error))throw std::runtime_error(error);
                try{shared();}catch(...){graph.end(stream.value,error);throw;}
                if(!graph.end(stream.value,error))throw std::runtime_error(error);
                ++shared_captures;
            }
            if(!graph.launch(stream.value,error))throw std::runtime_error(error);
            ++shared_launches;
        }else shared();
    };
    // The event excludes the later shared work. Scratch remains ordered on the
    // same stream; the CPU may plan ownership while shared kernels execute.
    struct SharedFailureGuard {
        cudaStream_t stream;bool active=false;int exceptions=std::uncaught_exceptions();
        ~SharedFailureGuard(){if(active&&std::uncaught_exceptions()>exceptions)cudaStreamSynchronize(stream);}
    } shared_failure{stream.value};
    if(!routes_ready)route(layer,tokens,w,phase);
    const bool cpu_path=options.arithmetic==Arithmetic::production &&
        (options.experts==ExpertMode::cpu || (options.experts==ExpertMode::hybrid && !prefill && (cpu_misses||tokens<=kernels::cpu::MAXT)));
    if(cpu_path && (cpu_misses||options.pcie_frac<1)){
        const uint64_t bytes=int64_t(tokens)*5120*4;
        gpu_check(cudaMemcpyAsync(w.host_x,w.x,bytes,cudaMemcpyDeviceToHost,stream.value));
        stats.cpu_input_d2h_bytes+=bytes;
    }
    if(phase==2&&spec)spec_issue(layer,tokens);
    if(early_shared){
        if(!route_side_pending)gpu_check(cudaEventRecord(route_ready.value,stream.value));
        shared_failure.active=true;
        submit_shared();
        prefill::LedgerScope wait_trace(options.trace,"ds41.moe.route.wait");
        gpu_check(cudaEventSynchronize(route_ready.value));
    }else{
        gpu_check(cudaStreamSynchronize(stream.value));
        if(route_side_pending)gpu_check(cudaEventSynchronize(route_ready.value));
    }
    route_side_pending=false;
    const int invalid_route=compact?w.host_ids[tokens*6]:-1;
    if(compact&&(invalid_route<-1||invalid_route>=tokens*6))throw std::runtime_error("DeepSeek invalid route status");
    for(int i=0;i<tokens*6;++i)if(w.host_ids[i]<0||w.host_ids[i]>=384||
        (compact?i==invalid_route:!std::isfinite(w.host_weights[i]))){
        // Validate before updating counters or indexing expert tables. Preserve
        // enough context to distinguish a router failure from cache ownership.
        std::vector<float> logits(tokens*384),input(tokens*5120);
        gpu_check(cudaMemcpy(logits.data(),w.branch,logits.size()*4,cudaMemcpyDeviceToHost));
        gpu_check(cudaMemcpy(input.data(),w.x,input.size()*4,cudaMemcpyDeviceToHost));
        const auto bad=[](const auto& values){return std::count_if(values.begin(),values.end(),[](float x){return !std::isfinite(x);});};
        throw std::runtime_error("DeepSeek invalid route layer="+std::to_string(layer)+" rows="+std::to_string(tokens)+
            " route="+std::to_string(i)+" expert="+std::to_string(w.host_ids[i])+" nonfinite_input="+std::to_string(bad(input))+
            " nonfinite_logits="+std::to_string(bad(logits)));
    }
    stats.shared_graph_captures+=shared_captures;stats.shared_graph_launches+=shared_launches;
    if(phase==2)experts->observe_physical_routes(layer,w.host_ids,tokens);
    if(deferred_routes)deferred_routes->insert(deferred_routes->end(),w.host_ids,w.host_ids+tokens*6);
    else experts->observe_routes(layer,w.host_ids,tokens,phase);
    struct Group { int expert=0,offset=0; std::vector<int> rows,slots; std::vector<float> weights; };
    std::array<Group,384> groups;
    for(int e=0;e<384;++e)groups[e].expert=e;
    for(int t=0;t<tokens;++t)for(int k=0;k<6;++k){
        const int e=w.host_ids[t*6+k];if(e<0||e>=384)throw std::runtime_error("DeepSeek routed expert ID");
        groups[e].rows.push_back(t);groups[e].slots.push_back(t*6+k);
        if(!compact)groups[e].weights.push_back(w.host_weights[t*6+k]);
    }
    const bool gpu_metadata=compact||(route_group_enabled.load()&&tokens>1&&tokens<=8);
    int metadata_offset=0;
    for(auto& g:groups){
        g.offset=metadata_offset;
        if(!gpu_metadata){
            std::copy(g.rows.begin(),g.rows.end(),w.host_group_rows+metadata_offset);
            std::copy(g.slots.begin(),g.slots.end(),w.host_group_slots+metadata_offset);
            std::copy(g.weights.begin(),g.weights.end(),w.host_group_weights+metadata_offset);
        }
        metadata_offset+=g.rows.size();
    }
    const char* fusion_env=std::getenv("STRATA_DS41_MAIN_MOE_FUSION");
    const int fusion=fusion_env&&fusion_env[0]=='2'?2:fusion_env&&fusion_env[0]=='1'?1:0;
    const bool grouped_down=fusion&&options.target_verify_batch&&phase==2&&tokens<=8&&
        options.arithmetic==Arithmetic::production&&options.expert_gu_mode&&
        (f.gu_type==10||f.gu_type==11)&&(f.d_type==10||f.d_type==11)&&
        kernels::cpu::deepseek41_down_fusion_enabled()&&w.max_tokens>=tokens*6;
    const bool single=options.expert_gu_mode && tokens==1&&!grouped_down;
    const bool short_possible=mixed_fast&&expert_mixed_short_enabled()&&phase==2&&tokens<=8&&
        options.target_verify_batch&&options.expert_direct_gemv&&expert_mixed_type(f.gu_type)&&
        expert_mixed_type(f.d_type)&&w.max_tokens>=tokens*6;
    if(!single&&!gpu_metadata){
        gpu_check(cudaMemcpyAsync(w.group_rows,w.host_group_rows,metadata_offset*sizeof(int),cudaMemcpyHostToDevice,stream.value));
        gpu_check(cudaMemcpyAsync(w.group_slots,w.host_group_slots,metadata_offset*sizeof(int),cudaMemcpyHostToDevice,stream.value));
        gpu_check(cudaMemcpyAsync(w.group_weights,w.host_group_weights,metadata_offset*sizeof(float),cudaMemcpyHostToDevice,stream.value));
    }
    std::vector<const Group*> cold,cpu;
    std::vector<std::pair<const Group*,const uint8_t*>> resident;
    const bool cache_active=decode_raw_cache && phase==2 && tokens<=8 && !cpu_misses;
    static FILE* trace_file=[]{const char* p=std::getenv("STRATA_DS41_ROUTE_TRACE");return p?std::fopen(p,"a"):nullptr;}();
    std::array<char,384> trace_class{};
    if(trace_file&&phase==2)for(const auto& g:groups)if(!g.rows.empty())
        trace_class[g.expert]=experts->gpu_resident(layer,g.expert)?'h':(cache_active&&decode_raw_cache->peek(layer*384+g.expert))?'r':'c';
    for(const auto& g:groups)if(!g.rows.empty()) {
        if(options.experts==ExpertMode::cpu)cpu.push_back(&g);
        else if(const auto* blob=experts->gpu(layer,g.expert,stream.value,stats,phase))resident.push_back({&g,blob});
        else if(const auto* cached=cache_active?decode_raw_cache->find(layer*384+g.expert):nullptr){
            resident.push_back({&g,cached});
            stats.raw_cache_hits+=g.rows.size();stats.demand_expert_tokens+=g.rows.size();++stats.cold_expert_groups;
        }else if(const auto* speculated=phase==2&&spec?spec_find(layer*384+g.expert):nullptr){
            resident.push_back({&g,speculated});
        }else if(cpu_path)cpu.push_back(&g);
        else cold.push_back(&g);
    }
    // STRATA_DS41_ROUTE_TRACE=path: per decode layer "layer expert:class ..." (h hot, r raw LRU, c cold) for offline
    // residency studies. Routing ids only.
    if(phase==2&&trace_file){
        std::fprintf(trace_file,"%d %d",layer,tokens);
        for(const auto& g:groups)if(!g.rows.empty())std::fprintf(trace_file," %d:%c%zu",g.expert,trace_class[g.expert],g.rows.size());
        std::fprintf(trace_file," |%zu\n",cold.size());
    }
    if(cpu_path && !cpu_misses && options.pcie_frac>0) {
        const size_t take=cpu.size()*static_cast<unsigned>(std::lround(options.pcie_frac*256))/256;
        for(size_t i=0;i<take;++i)stats.demand_expert_tokens+=cpu[i]->rows.size();
        stats.demand_expert_bytes+=take*f.bytes;stats.cold_expert_groups+=take;stats.demand_expert_groups+=take;
        cold.insert(cold.end(),cpu.begin(),cpu.begin()+take);cpu.erase(cpu.begin(),cpu.begin()+take);
    }
    if(!prefill&&route_predict_on()){
        std::vector<int> cold_ids;for(const auto* g:cold)cold_ids.push_back(g->expert);
        static std::vector<bool> hot=[&]{std::vector<bool> v(40*384);for(const auto& k:experts->hot_pairs())v[k.first*384+k.second]=true;return v;}();
        route_predict_score(layer,w.host_ids,tokens,cold_ids,[&](int e){return hot[layer*384+e]||
            (decode_raw_cache&&decode_raw_cache->peek(layer*384+e));});
    }
    if(phase==2&&spec_state()){
        auto& sp=*spec;const size_t misses=cold.size()+(cpu_misses?0:0);
        sp.ema=0.99*sp.ema+0.01*double(misses);sp.cold+=misses;++sp.layers;sp.active_layers+=sp.active();
        if(sp.layers%4000==0)std::fprintf(stderr,"spec-prefetch layers=%llu active=%.2f predictions=%llu issued=%llu hits=%llu cold=%llu ema=%.2f\n",
            (unsigned long long)sp.layers,double(sp.active_layers)/sp.layers,(unsigned long long)sp.predictions,(unsigned long long)sp.issued,
            (unsigned long long)sp.hits,(unsigned long long)sp.cold,sp.ema);
    }
    if(compact&&!cpu.empty())throw std::logic_error("DeepSeek compact route reached CPU experts");
    // Route observation already counted true residency. Demand offload is kept
    // separate and never contributes to that hit counter.
    if(prefetch_active && !prefill)throw std::logic_error("prefill stream reached decode");
    if(prefetch_active){
        std::array<bool,384> selected{};
        for(const auto* group:cold)selected[group->expert]=true;
        prune_expert_prefetch(layer,selected,stats);
    }
    const bool issued=prefetch_active || (expert_pipeline && !cold.empty() &&
        (options.expert_pipeline || (!prefill && cpu_path && options.pcie_frac>0)));
    struct FailureGuard {
        ExpertPipeline* pipeline;cudaStream_t dequant,reader;int exceptions=std::uncaught_exceptions();
        ~FailureGuard(){if(pipeline && std::uncaught_exceptions()>exceptions){pipeline->stop();cudaStreamSynchronize(dequant);cudaStreamSynchronize(reader);}}
    } failure{issued?expert_pipeline.get():nullptr,dequant_stream.value,stream.value};
    if(issued && !prefetch_active) {
        const size_t issuer_slots=cache_active?8:0;
        std::vector<ExpertPipeline::Upload> direct;
        if(options.pinned_experts)for(const auto* group:cold){
            const auto* source=experts->pinned_blob(layer,group->expert);
            if(!source){direct.clear();break;}
            direct.push_back({source,f.bytes,options.trace?layer:-1,group->expert});
        }
        if(!direct.empty()){
            expert_pipeline->start(++expert_generation,std::move(direct),issuer_slots);
            stats.direct_expert_bytes+=cold.size()*f.bytes;
        }else expert_pipeline->start(++expert_generation,cold.size(),f.bytes,[&](size_t i,uint8_t* host) {
            prefill::LedgerScope fill_trace(options.trace,"ds41.expert.issuer.hostfill");
            if(experts->transient(layer,cold[i]->expert))experts->copy_blob(layer,cold[i]->expert,host);
            else std::memcpy(host,experts->blob_stable(layer,cold[i]->expert),f.bytes);
            if(options.arithmetic==Arithmetic::f32)experts->release_mapped(layer,cold[i]->expert);
        },issuer_slots);
        stats.h2d_expert_bytes+=cold.size()*f.bytes;
    }
    // Two existing expert banks bound every batch. Activation rows also fit
    // the original per-expert workspace; no residency or arena allocation changes.
    constexpr uint64_t expert_elements=5120ull*2304;
    // Only the measured full prefill chunk uses this path; smaller batches regress.
    const bool bounded=(bounded_moe_supported||(bounded_moe_sm89&&options.sm89_grouped_moe)) && mixed_fast && options.expert_bf16 && options.expert_pipeline && tokens==8192 &&
        // These two pairs fail the whole-layer accuracy gate; keep567.
        !(f.gu_type==17 && f.d_type==39) && !(f.gu_type==11 && f.d_type==11) &&
        (expert_mixed_type(f.gu_type)||f.gu_type==11) && (expert_mixed_type(f.d_type)||f.d_type==11) &&
        w.expert_up==w.expert_gate+expert_elements && w.expert_ahead[0] &&
        w.expert_ahead[1]==w.expert_ahead[0]+expert_elements;
    const int bounded_capacity=2;
    auto batch_group=[&](const Group& g,const uint8_t* blob,int slot){
        uint16_t* gate=slot==0?w.expert_gate:w.expert_ahead[0];
        uint16_t* down=slot==0?w.expert_down:w.expert_ahead[2];
        return BoundedMoeGroup{blob,blob+f.up_off,blob+f.down_off,gate,down,g.offset,static_cast<int>(g.rows.size())};
    };
    auto run_batch=[&](const BoundedMoeGroup* batch,int count,bool prepared=false){
        const int rc=bounded_moe_gpu(batch,count,f.gu_type,f.d_type,w.x,w.group_rows,w.group_weights,w.group_slots,w.parts,
            w.bf_input,reinterpret_cast<uint16_t*>(w.group_gate),reinterpret_cast<uint16_t*>(w.group_act),
            reinterpret_cast<void**>(w.dequant),w.max_tokens,stream.value,prepared);
        if(rc)throw std::runtime_error("DeepSeek bounded grouped MoE error="+std::to_string(rc));
        for(int i=0;i<count;++i)stats.expert_gpu+=batch[i].tokens;
    };
    if(!early_shared){
        submit_shared();
        stats.shared_graph_captures+=shared_captures;stats.shared_graph_launches+=shared_launches;
    }
    if(moe_shared_host_ready)moe_shared_host_ready();
    // Opt-in tensor-core decode: cold (streamed) experts use the same grouped MMA arithmetic as
    // resident ones, so a token's result does not depend on residency.
    const bool mma_decode=short_possible&&expert_grouped_enabled()&&expert_grouped_mma_enabled()&&
        expert_grouped_supported(f.gu_type,f.d_type)&&options.arithmetic==Arithmetic::production;
    bool metadata_on_device=!single||gpu_metadata;
    auto upload_single_metadata=[&]{
        if(metadata_on_device)return;
        gpu_check(cudaMemcpyAsync(w.group_rows,w.host_group_rows,metadata_offset*sizeof(int),cudaMemcpyHostToDevice,stream.value));
        gpu_check(cudaMemcpyAsync(w.group_slots,w.host_group_slots,metadata_offset*sizeof(int),cudaMemcpyHostToDevice,stream.value));
        gpu_check(cudaMemcpyAsync(w.group_weights,w.host_group_weights,metadata_offset*sizeof(float),cudaMemcpyHostToDevice,stream.value));
        metadata_on_device=true;
    };
    auto compute=[&](const Group& g,const uint8_t* blob,const std::array<uint16_t*,3>* prepared=nullptr){
        const int nt=g.rows.size();
        if(mma_decode&&!prepared&&nt<=8){
            upload_single_metadata();
            const VerifyExpertGroup one{blob,g.offset,nt};
            gpu_check(expert_grouped_gpu(&one,1,f.gu_type,f.d_type,f.up_off,f.down_off,w.x,w.group_rows,w.group_weights,
                w.group_slots,w.group_act,w.parts,tokens,w.group_gate,size_t(w.max_tokens)*2304*sizeof(float),stream.value));
            stats.expert_gpu+=nt;return;
        }
        if(!single)gpu_check(gather_rows_gpu(w.x,w.group_rows+g.offset,w.group_x,nt,5120,stream.value));
        const float* input=single?w.x:w.group_x;
        const float* routing=single?w.route_weights+g.slots[0]:w.group_weights+g.offset;
        const int* slots=single?nullptr:w.group_slots+g.offset;
        float* output=single?w.parts+int64_t(g.slots[0])*5120:w.parts;
        int projection_index=0;
        auto reused_gu=[&](const uint16_t* weights,float* output,int index){
            if(!options.moe_input_reuse||nt<=8||index>1||options.arithmetic!=Arithmetic::production)return false;
            if(index==0)gpu_check(float_to_bf16_gpu(input,w.bf_input,int64_t(nt)*5120,stream.value));
            const float alpha=1,beta=0;
            const auto status=cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,2304,nt,5120,&alpha,
                weights,CUDA_R_16BF,5120,w.bf_input,CUDA_R_16BF,5120,&beta,output,CUDA_R_32F,2304,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
            if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("DeepSeek reused GU multiplication status="+std::to_string(status));
            round(output,int64_t(nt)*2304);return true;
        };
        auto projection=[&](const uint8_t* packed,int rows,int cols,uint16_t* bf,const float* input,float* output){
            const int index=projection_index++;
            const int type=index==2?f.d_type:f.gu_type;
            const bool packed_supported=type==10||type==11||type==39;
            if(prepared) {
                if(reused_gu((*prepared)[index],output,index))return;
                Matrix matrix{(*prepared)[index],30,rows,cols,uint64_t(rows)*cols};
                matmul(matrix,input,output,nt,w);return;
            }
            if(mixed_fast && options.expert_direct_gemv && nt<=8 && expert_mixed_type(type)) {
                gpu_check(expert_mixed_projection_gpu(packed,type,input,output,rows,cols,nt,nullptr,stream.value));return;
            }
            if(options.arithmetic==Arithmetic::production && options.expert_direct_gemv && nt<=8 && packed_supported) {
                gpu_check(packed_bf16_gemv_gpu(packed,type,input,output,rows,cols,nt,stream.value));
                round(output,int64_t(nt)*rows);return;
            }
            if(options.arithmetic==Arithmetic::f32 && nt<=8 && packed_supported) {
                gpu_check(packed_gemv_gpu(packed,type,input,output,rows,cols,nt,stream.value));return;
            }
            Matrix matrix;matrix.rows=rows;matrix.cols=cols;matrix.elements=int64_t(rows)*cols;
            if(options.arithmetic==Arithmetic::production){
                if(options.expert_bf16)gpu_check((mixed_fast&&expert_mixed_type(type)?expert_mixed_dequant_gpu:dequant_bf16_gpu)(
                    packed,type,bf,matrix.elements,stream.value));
                else {
                    gpu_check(dequant_gpu(packed,type,w.dequant,matrix.elements,stream.value));
                    gpu_check(float_to_bf16_gpu(w.dequant,bf,matrix.elements,stream.value));
                }
                matrix.data=bf;matrix.type=30;
            }else {
                gpu_check(dequant_gpu(packed,type,w.dequant,matrix.elements,stream.value));matrix.data=w.dequant;
            }
            if(matrix.type==30&&reused_gu(static_cast<const uint16_t*>(matrix.data),output,index))return;
            matmul(matrix,input,output,nt,w);
        };
        if(mixed_fast && options.expert_direct_gemv && !prepared &&
           expert_mixed_gu_activation_selected(f.gu_type,nt) &&
           (expert_mixed_type(f.d_type)||(f.d_type==11&&w.dense_input_q8))){
            gpu_check(expert_mixed_gu_activation_gpu(blob,blob+f.up_off,f.gu_type,input,routing,
                w.group_act,2304,5120,nt,stream.value));
            if(expert_mixed_type(f.d_type))
                gpu_check(expert_mixed_projection_gpu(blob+f.down_off,f.d_type,w.group_act,output,
                    5120,2304,nt,slots,stream.value));
            else{
                kernels::native_quantize_q8_1(w.group_act,w.dense_input_q8,2304,nt,stream.value);
                kernels::native_q3_k_mmvq(blob+f.down_off,w.dense_input_q8,w.group_out,2304,5120,nt,stream.value);
                round(w.group_out,int64_t(nt)*5120);
                if(slots)gpu_check(scatter_expert_gpu(w.group_out,slots,output,nt,6,stream.value));
                else gpu_check(cudaMemcpyAsync(output,w.group_out,int64_t(nt)*5120*4,cudaMemcpyDeviceToDevice,stream.value));
            }
            stats.expert_gpu+=nt;return;
        }
        if(mixed_fast && options.expert_direct_gemv && nt<=8 && !prepared && expert_mixed_type(f.gu_type)){
            gpu_check(expert_mixed_gu_gpu(blob,blob+f.up_off,f.gu_type,input,w.group_gate,w.group_up,2304,5120,nt,stream.value));
            projection_index=2;
        }else if(options.expert_gu_mode && nt<=8 && !prepared && (f.gu_type==10 || f.gu_type==11)){
            gpu_check((options.target_verify_batch&&phase==2?expert_gu_verify_gpu:expert_gu_gpu)(blob,f.gu_type,input,w.group_gate,w.group_up,2304,5120,nt,
                options.expert_gu_mode,w.group_act,stream.value));projection_index=2;
        }else{
            projection(blob,2304,5120,w.expert_gate,input,w.group_gate);
            projection(blob+f.up_off,2304,5120,w.expert_up,input,w.group_up);
        }
        if(options.arithmetic==Arithmetic::production&&nt<=8&&(f.d_type==10||f.d_type==11)&&kernels::cpu::deepseek41_down_fusion_enabled()){
            gpu_check((options.target_verify_batch&&phase==2?format_down_verify_gpu:format_down_gpu)(f.d_type,blob+f.down_off,w.group_gate,w.group_up,routing,
                w.group_out,output,slots,5120,2304,nt,w.group_act,size_t(nt)*2304*4,stream.value));
            stats.expert_gpu+=nt;return;
        }
        if(mixed_fast && options.expert_direct_gemv && nt<=8 && !prepared && expert_mixed_type(f.d_type)){
            gpu_check(expert_mixed_down_gpu(blob+f.down_off,f.d_type,w.group_gate,w.group_up,routing,w.group_act,
                output,5120,2304,nt,slots,stream.value));
            stats.expert_gpu+=nt;return;
        }
        if(mixed_fast && expert_mixed_type(f.d_type))
            gpu_check(expert_mixed_swiglu_gpu(w.group_gate,w.group_up,routing,w.group_act,2304,nt,stream.value));
        else{
            gpu_check((options.arithmetic==Arithmetic::production?swiglu_bf16_gpu:swiglu_gpu)(w.group_gate,w.group_up,routing,w.group_act,2304,nt,10.f,stream.value));
        }
        projection(blob+f.down_off,5120,2304,w.expert_down,w.group_act,single?output:w.group_out);
        if(!single)gpu_check(scatter_expert_gpu(w.group_out,slots,output,nt,6,stream.value));stats.expert_gpu+=nt;
    };
    const bool grouped_verify=grouped_down||(options.target_verify_batch&&phase==2&&tokens>1&&tokens<=8&&
        options.arithmetic==Arithmetic::production&&options.expert_gu_mode&&
        (f.gu_type==10||f.gu_type==11)&&
        ((f.d_type==10||f.d_type==11)&&kernels::cpu::deepseek41_down_fusion_enabled()||
         mixed_fast&&expert_mixed_type(f.d_type))&&w.max_tokens>=tokens*6);
    const bool short_uniform=std::all_of(resident.begin(),resident.end(),[&](const auto& item){return item.first->rows.size()==static_cast<size_t>(tokens);});
    // The grouped kernels match the short path bit for bit and are faster at every
    // measured group count, so they replace the per-bucket selection tables.
    const bool grouped=short_possible&&expert_grouped_enabled()&&expert_grouped_supported(f.gu_type,f.d_type)&&
        !resident.empty()&&resident.size()<=48;
    const bool short_batch=grouped||(short_possible&&expert_mixed_short_selected(f.gu_type,f.d_type,tokens,resident.size(),short_uniform));
    // Prefill routed experts on the fused tensor-core kernels (not bit-exact). The activations use the
    // idle group_x scratch and H the idle attention BF16 storage behind group_gate (compact workspace only).
    const bool fused=prefill&&tokens>=fused_moe_min_tokens()&&fused_moe_enabled()&&options.arithmetic==Arithmetic::production&&
        options.compact_workspace&&cpu.empty()&&(cold.empty()||issued)&&!gpu_metadata&&!single&&
        fused_moe_supported(f.gu_type,f.d_type)&&fused_moe_act_bytes(tokens)<=uint64_t(tokens)*8192*4&&
        fused_moe_h_bytes(metadata_offset)<=uint64_t(tokens)*32768*2;
    void* fused_xa=w.group_x;void* fused_h=w.group_gate;
    auto fused_run=[&](const FusedMoeBatch& batch,int64_t rows){
        gpu_check(fused_moe_run(batch,f.gu_type,f.d_type,f.up_off,f.down_off,rows,fused_xa,w.group_rows,w.group_weights,
            w.group_slots,fused_h,w.parts,stream.value));
        stats.expert_gpu+=rows;
    };
    // Resident experts; with streamed experts they run after them by default, so ring slots return before the
    // resident products and the issuer streams the next layer during them.
    auto fused_resident=[&]{
        for(int first=0;first<384;first+=kFusedMoeBatch){
            FusedMoeBatch batch;batch.e0=first;batch.e1=std::min(384,first+kFusedMoeBatch);int64_t rows=0;
            for(const auto& item:resident){const int e=item.first->expert;
                if(e>=batch.e0&&e<batch.e1){batch.blob[e-first]=item.second;rows+=item.first->rows.size();}}
            if(rows)fused_run(batch,rows);
        }
    };
    if(fused){
        prefill::LedgerScope fused_trace(options.trace,"ds41.moe.fused");
        gpu_check(fused_moe_quantize(w.x,tokens,fused_xa,stream.value));
        std::array<int,385> offsets{};
        for(int e=0;e<384;++e)offsets[e]=groups[e].offset;
        offsets[384]=metadata_offset;
        gpu_check(fused_moe_tables(offsets.data(),384,stream.value));
        if(!issued||!fused_cold_first.load())fused_resident();
    }else if(short_batch){
        upload_single_metadata();
        std::array<VerifyExpertGroup,48> batch{};int count=0;
        for(const auto& item:resident){const auto& g=*item.first;
            if(count==48||g.offset+g.rows.size()>static_cast<size_t>(w.max_tokens))throw std::logic_error("IQ short group workspace");
            batch[count++]={item.second,g.offset,static_cast<int>(g.rows.size())};
        }
        // Gate/Up scratch is unused by the fused grouped path; it holds the BF16 input copy.
        if(grouped)gpu_check(expert_grouped_gpu(batch.data(),count,f.gu_type,f.d_type,f.up_off,f.down_off,
            w.x,w.group_rows,w.group_weights,w.group_slots,w.group_act,w.parts,tokens,
            w.group_gate,size_t(w.max_tokens)*2304*sizeof(float),stream.value));
        else gpu_check(expert_mixed_short_gpu(batch.data(),count,f.gu_type,f.d_type,f.up_off,f.down_off,
            w.x,w.group_rows,w.group_weights,w.group_slots,w.group_gate,w.group_up,w.group_act,w.parts,tokens,stream.value));
        for(const auto& item:resident)stats.expert_gpu+=item.first->rows.size();
    }else if(grouped_verify&&!resident.empty()){
        // All GU inputs refer to the original token rows, so quantize once per layer.
        // Group outputs occupy disjoint row spans until every resident Down has consumed them.
        std::array<VerifyExpertGroup,48> batch{};int count=0;
        for(const auto& item:resident){const auto& g=*item.first;
            if(count==48||g.offset+g.rows.size()>static_cast<size_t>(w.max_tokens))throw std::logic_error("verify group workspace");
            batch[count++]={item.second,g.offset,static_cast<int>(g.rows.size())};
        }
        const bool fused_activation=grouped_down&&fusion==2&&f.gu_type==10&&f.d_type==10&&options.expert_gu_mode==2;
        if(fused_activation)gpu_check(expert_q2_activation_grouped_gpu(batch.data(),count,w.x,w.group_rows,w.group_weights,
            reinterpret_cast<uint16_t*>(w.group_act),tokens,w.group_x,stream.value));
        else gpu_check(expert_gu_grouped_verify_gpu(batch.data(),count,f.gu_type,w.x,w.group_rows,
            w.group_gate,w.group_up,tokens,options.expert_gu_mode,w.group_x,stream.value));
        if(grouped_down){
            for(int i=0;i<count;++i)batch[i].packed=static_cast<const uint8_t*>(batch[i].packed)+f.down_off;
            gpu_check(format_down_grouped_gpu(batch.data(),count,f.d_type,w.group_gate,w.group_up,w.group_weights,
                w.parts,w.group_slots,w.group_act,size_t(w.max_tokens)*2304*4,fused_activation,stream.value));
        }else for(const auto& item:resident){const auto& g=*item.first;const int nt=g.rows.size();
            if(mixed_fast&&expert_mixed_type(f.d_type))
                gpu_check(expert_mixed_down_gpu(item.second+f.down_off,f.d_type,
                    w.group_gate+int64_t(g.offset)*2304,w.group_up+int64_t(g.offset)*2304,w.group_weights+g.offset,
                    w.group_act,w.parts,5120,2304,nt,w.group_slots+g.offset,stream.value));
            else gpu_check(format_down_verify_gpu(f.d_type,item.second+f.down_off,
                    w.group_gate+int64_t(g.offset)*2304,w.group_up+int64_t(g.offset)*2304,w.group_weights+g.offset,
                    w.group_out,w.parts,w.group_slots+g.offset,5120,2304,nt,w.group_act,size_t(nt)*2304*4,stream.value));
        }
        for(const auto& item:resident)stats.expert_gpu+=item.first->rows.size();
    }else if(bounded){
        std::array<BoundedMoeGroup,BOUNDED_MOE_BATCH> batch{};int count=0,rows=0;
        auto flush=[&]{if(count)run_batch(batch.data(),count);count=rows=0;};
        for(const auto& item:resident){const auto& g=*item.first;
            if(g.rows.size()<=8){flush();compute(g,item.second);continue;}
            if(count==bounded_capacity || rows+g.rows.size()>static_cast<size_t>(w.max_tokens))flush();
            batch[count]=batch_group(g,item.second,count);++count;rows+=g.rows.size();
        }
        flush();
    }else for(const auto& item:resident)compute(*item.first,item.second);
    if(fused && issued){
        // Streamed experts in batches of up to 16 ring leases (ascending expert order), released after their products.
        constexpr size_t lease_batch=16;
        std::vector<size_t> leases;FusedMoeBatch batch;int64_t rows=0;
        auto flush=[&]{
            if(leases.empty())return;
            fused_run(batch,rows);
            for(size_t sequence:leases)expert_pipeline->release(sequence,stream.value);
            leases.clear();batch=FusedMoeBatch{};rows=0;
        };
        for(size_t i=0;i<cold.size();++i){const auto& g=*cold[i];
            if(!leases.empty()&&(leases.size()==lease_batch||g.expert-batch.e0>=kFusedMoeBatch))flush();
            size_t sequence=i;
            if(prefetch_active){sequence=prefetch_positions[layer][g.expert];retire_expert_prefetch(sequence,stats);}
            const uint8_t* blob;
            {prefill::LedgerScope wait_trace(options.trace,"ds41.expert.issuer.wait");blob=expert_pipeline->wait(sequence,stream.value);}
            if(leases.empty())batch.e0=g.expert;
            batch.e1=g.expert+1;batch.blob[g.expert-batch.e0]=blob;rows+=g.rows.size();leases.push_back(sequence);
            if(prefetch_active)prefetch_cursor=sequence+1;
        }
        flush();
        if(fused_cold_first.load())fused_resident();
        if(prefetch_active)retire_expert_prefetch(prefetch_offsets[layer+1],stats);
        else expert_pipeline->finish();
    }else if(issued && bounded){
        std::array<BoundedMoeGroup,BOUNDED_MOE_BATCH> batch{};int count=0,rows=0;
        auto flush=[&]{if(count)run_batch(batch.data(),count,true);count=rows=0;};
        for(size_t i=0;i<cold.size();++i){const auto& g=*cold[i];
            if(g.rows.size()<=8 || count==bounded_capacity || rows+g.rows.size()>static_cast<size_t>(w.max_tokens))flush();
            size_t sequence=i;
            if(prefetch_active){sequence=prefetch_positions[layer][g.expert];retire_expert_prefetch(sequence,stats);}
            const uint8_t* blob;
            {prefill::LedgerScope wait_trace(options.trace,"ds41.expert.issuer.wait");blob=expert_pipeline->wait(sequence,stream.value);}
            if(g.rows.size()<=8)compute(g,blob);
            else{
                batch[count]=batch_group(g,blob,count);
                const int rc=bounded_moe_prepare_gpu(batch[count],f.gu_type,f.d_type,stream.value);
                if(rc)throw std::runtime_error("DeepSeek bounded dequant error="+std::to_string(rc));
                rows+=g.rows.size();++count;
            }
            // Return each raw lease after DQ, before requesting the next one.
            // This stays bounded even when a ring has fewer slots than a batch.
            expert_pipeline->release(sequence,stream.value);
            if(prefetch_active)prefetch_cursor=sequence+1;
        }
        flush();
        if(prefetch_active)retire_expert_prefetch(prefetch_offsets[layer+1],stats);
        else expert_pipeline->finish();
    }else if(issued) {
        // The first auxiliary dequant waits for resident readers of the shared
        // BF16 buffers. Subsequent two-buffer reuse waits for that buffer only.
        gpu_check(cudaEventRecord(expert_begin.value,stream.value));
        gpu_check(cudaStreamWaitEvent(dequant_stream.value,expert_begin.value,0));
        std::array<bool,2> used{};
        for(size_t i=0;i<cold.size();++i) {
            const int bank=i%2,nt=cold[i]->rows.size();
            size_t sequence=i;
            if(prefetch_active){
                sequence=prefetch_positions[layer][cold[i]->expert];
                retire_expert_prefetch(sequence,stats);
            }
            const bool expanded=options.expert_pipeline && options.arithmetic==Arithmetic::production &&
                !(nt<=8 && (options.expert_direct_gemv || kernels::cpu::deepseek41_down_fusion_enabled()));
            if(expanded) {
                auto bf=bank?w.expert_ahead:std::array<uint16_t*,3>{w.expert_gate,w.expert_up,w.expert_down};
                if(used[bank])gpu_check(cudaStreamWaitEvent(dequant_stream.value,expert_used[bank].value,0));
                const uint8_t* blob;
                {prefill::LedgerScope wait_trace(options.trace,"ds41.expert.issuer.wait");blob=expert_pipeline->wait(sequence,dequant_stream.value);}
                const uint64_t offsets[3]={0,f.up_off,f.down_off};
                for(int p=0;p<3;++p) {
                    const auto type=p==2?f.d_type:f.gu_type;
                    if(options.expert_bf16)gpu_check((mixed_fast&&expert_mixed_type(type)?expert_mixed_dequant_gpu:dequant_bf16_gpu)(
                        blob+offsets[p],type,bf[p],5120*2304,dequant_stream.value));
                    else {
                        gpu_check(dequant_gpu(blob+offsets[p],p==2?f.d_type:f.gu_type,w.dequant,5120*2304,dequant_stream.value));
                        gpu_check(float_to_bf16_gpu(w.dequant,bf[p],5120*2304,dequant_stream.value));
                    }
                }
                // Fully expanded path has no further raw readers.
                expert_pipeline->release(sequence,dequant_stream.value);
                gpu_check(cudaEventRecord(expert_ready[bank].value,dequant_stream.value));
                gpu_check(cudaStreamWaitEvent(stream.value,expert_ready[bank].value,0));
                compute(*cold[i],blob,&bf);
                gpu_check(cudaEventRecord(expert_used[bank].value,stream.value));used[bank]=true;
            } else {
                const uint8_t* blob;
                {prefill::LedgerScope wait_trace(options.trace,"ds41.expert.issuer.wait");blob=expert_pipeline->wait(sequence,stream.value);}
                if(cache_active){
                    prefill::LedgerScope retain_trace(options.trace,"ds41.expert.rawcache.retain");
                    blob=decode_raw_cache->retain(layer*384+cold[i]->expert,blob,f.bytes);stats.raw_cache_d2d_bytes+=f.bytes;
                }
                compute(*cold[i],blob);
                // Direct GEMV/fused-down still read packed weights through compute.
                expert_pipeline->release(sequence,stream.value);
            }
            if(prefetch_active)prefetch_cursor=sequence+1;
        }
        if(prefetch_active)retire_expert_prefetch(prefetch_offsets[layer+1],stats);
        else expert_pipeline->finish();
    }else if(!cold.empty()) {
        const int bank_slots=ring_slots/2;
        const size_t batches=(cold.size()+bank_slots-1)/bank_slots;
        auto prepare=[&](size_t batch){
            const int bank=batch%2;
            if(ring_used[bank]) {
                // Host memory can be rewritten after DMA; device slots wait for their last consumer.
                gpu_check(cudaEventSynchronize(ring_ready[bank].value));
                gpu_check(cudaStreamWaitEvent(upload.value,ring_consumed[bank].value,0));
            }
            const size_t first=batch*bank_slots,last=std::min(cold.size(),first+bank_slots);
            for(size_t i=first;i<last;++i){
                const uint64_t offset=(uint64_t(bank)*bank_slots+i-first)*ring_slot_bytes;
                uint8_t* host=host_ring->data+offset;
                if(experts->transient(layer,cold[i]->expert))experts->copy_blob(layer,cold[i]->expert,host);
                else std::memcpy(host,experts->blob_stable(layer,cold[i]->expert),f.bytes);
                gpu_check(cudaMemcpyAsync(static_cast<uint8_t*>(ring->base())+offset,host,f.bytes,cudaMemcpyHostToDevice,upload.value));
                stats.h2d_expert_bytes+=f.bytes;
                if(options.arithmetic==Arithmetic::f32)experts->release_mapped(layer,cold[i]->expert);
            }
            gpu_check(cudaEventRecord(ring_ready[bank].value,upload.value));ring_used[bank]=true;
        };
        prepare(0);
        for(size_t batch=0;batch<batches;++batch){
            const int bank=batch%2;
            gpu_check(cudaStreamWaitEvent(stream.value,ring_ready[bank].value,0));
            const size_t first=batch*bank_slots,last=std::min(cold.size(),first+bank_slots);
            for(size_t i=first;i<last;++i){
                const uint64_t offset=(uint64_t(bank)*bank_slots+i-first)*ring_slot_bytes;
                compute(*cold[i],static_cast<const uint8_t*>(ring->base())+offset);
            }
            gpu_check(cudaEventRecord(ring_consumed[bank].value,stream.value));
            if(batch+1<batches)prepare(batch+1);
        }
    }
    if(spec&&!spec->pending.empty()){
        for(int slot:spec->pending){gpu_check(cudaEventRecord(spec->used[slot]->value,stream.value));spec->used_valid[slot]=true;}
        spec->pending.clear();
    }
    if(!cpu.empty()) {
        prefill::LedgerScope cpu_trace(options.trace,"ds41.cpu.experts");
        // One pool batch has at most eight tokens per expert and 96 independent experts.
        struct CpuJob { kernels::cpu::ExpertJobMulti job; std::vector<uint8_t> act,owned; std::vector<float> weights; };
        std::vector<CpuJob> jobs;
        auto drain=[&]{
            std::vector<kernels::cpu::ExpertJobMulti> native;native.reserve(jobs.size());
            for(auto& item:jobs)native.push_back(item.job);
            pool->run_split_multi_native(f,native.data(),native.size());
            for(auto& item:jobs)for(int t=0;t<item.job.nt;++t){
                float* output=item.job.out[t];
                for(int d=0;d<5120;++d)output[d]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(output[d]));
            }
            jobs.clear();
        };
        for(const Group* group:cpu)for(size_t begin=0;begin<group->rows.size();begin+=kernels::cpu::MAXT){
            if(jobs.size()==kernels::cpu::ExpertPool::kMaxSplitMulti)drain();
            CpuJob item;item.job.nt=std::min<size_t>(kernels::cpu::MAXT,group->rows.size()-begin);
            if(experts->transient(layer,group->expert)) { item.owned.resize(f.bytes);experts->copy_blob(layer,group->expert,item.owned.data());item.job.blob=item.owned.data(); }
            else item.job.blob=experts->blob_stable(layer,group->expert);
            item.act.resize(item.job.nt*f.act_bytes);item.weights.resize(item.job.nt);
            for(int t=0;t<item.job.nt;++t){
                const int row=group->rows[begin+t],slot=group->slots[begin+t];
                kernels::cpu::native_quant_act(f,w.host_x+int64_t(row)*5120,item.act.data()+t*f.act_bytes);
                item.job.nact[t]=item.act.data()+t*f.act_bytes;item.job.out[t]=w.host_parts+int64_t(slot)*5120;
                item.weights[t]=group->weights[begin+t];
            }
            item.job.ff_weights=item.weights.data();stats.expert_cpu+=item.job.nt;jobs.push_back(std::move(item));
        }
        if(!jobs.empty())drain();
        for(const Group* group:cpu)for(int slot:group->slots){
            gpu_check(cudaMemcpyAsync(w.parts+int64_t(slot)*5120,w.host_parts+int64_t(slot)*5120,5120*4,cudaMemcpyHostToDevice,stream.value));
            stats.cpu_output_h2d_bytes+=5120*4;
        }
    }
    gpu_check(combine_experts_gpu(w.parts,w.route_ids,w.shared_out,w.branch,tokens,6,stream.value));round(w.branch,int64_t(tokens)*5120);
}
} // namespace strata::models::deepseek41
