#include "strata/models/deepseek41/model.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using namespace strata::models::deepseek41;
    int failures = 0;
    auto check = [&](bool value, const char* message) {
        if (!value) { std::cerr << message << '\n'; ++failures; }
    };
    const auto owners = layer_ownership();
    int globals = 0, sources = 0;
    for (int l = 0; l < 40; ++l) {
        check(owners[l].local_owner == l, "SWA must remain private to its layer");
        check(owners[l].global_owner <= l && owners[l].index_source <= l, "future owner/source");
        check(owners[l].compression == (l < 2 ? 0 : l < 20 ? 2 : 1), "compression ratio");
        globals += owners[l].owns_global(); sources += owners[l].computes_index();
    }
    check(globals == 4 && sources == 8, "global/index source counts");
    check(owners[39].global_owner == 20 && owners[39].index_source == 36, "decoder reuse");
    std::array<uint64_t, 40> sizes{};
    for (int l = 0; l < 40; ++l) sizes[l] = l < 14 ? 15206400 : 11612160;
    const uint64_t gib = uint64_t{1} << 30;
    for (const auto& budgets : {std::pair{40 * gib, 16 * gib}, std::pair{34 * gib, 14 * gib}}) {
        const auto allocation = allocate_hot_experts(sizes, budgets.first, budgets.second);
        check(allocation.front_bytes <= budgets.first && allocation.back_bytes <= budgets.second, "budget overflow");
        check(allocation.front_bytes > allocation.back_bytes, "front allocation preference");
        uint64_t total = allocation.host_bytes + allocation.front_bytes + allocation.back_bytes;
        check(total == 197685411840ull, "GPU plus RAM must cover every expert exactly once");
        for (int n : allocation.count) check(n >= 32 && n <= 384, "per-layer coverage");
    }
    try { (void) allocate_hot_experts(sizes, gib, gib / 2); check(false, "impossible floor accepted"); }
    catch (const std::invalid_argument&) {}
    auto invalid = [&](auto call, const char* message) {
        try { call(); check(false,message); } catch (const std::invalid_argument&) {}
    };
    auto overflow = [&](auto call, const char* message) {
        try { call(); check(false,message); } catch (const std::overflow_error&) {}
    };
    std::array<uint64_t,40> variable{};variable.fill(257);variable[3]=256;variable[11]=513;
    const std::vector<std::pair<int32_t,int32_t>> rank{{3,7},{3,8},{11,12},{0,1}};
    auto prefix=allocate_ranked_hot_experts(variable,rank,512);
    check(prefix.experts==std::vector<std::pair<int32_t,int32_t>>{{3,7},{3,8}}&&prefix.bytes==512,
          "global ranking must permit concentrated and zero-hot layers");
    prefix=allocate_ranked_hot_experts(variable,rank,1279);
    check(prefix.experts.size()==2&&prefix.bytes==512,"first no-fit must stop, without skipping to cheaper expert");
    prefix=allocate_ranked_hot_experts(variable,rank,1280);
    check(prefix.experts.size()==3&&prefix.bytes==1280,"256-byte aligned exact capacity boundary");
    prefix=allocate_ranked_hot_experts(variable,{{11,0},{3,0}},512);
    check(prefix.experts.empty()&&prefix.bytes==0,"first ranked expert no-fit must keep prefix empty");
    invalid([&]{(void)allocate_ranked_hot_experts(variable,rank,0);},"zero capacity accepted");
    invalid([&]{(void)allocate_ranked_hot_experts(variable,{},512);},"empty profile accepted");
    invalid([&]{(void)allocate_ranked_hot_experts(variable,{{11,0},{3,0},{3,0}},1);},"duplicate beyond capacity boundary accepted");
    invalid([&]{(void)allocate_ranked_hot_experts(variable,{{40,0}},512);},"invalid profile layer accepted");
    invalid([&]{(void)allocate_ranked_hot_experts(variable,{{0,-1}},512);},"invalid profile expert accepted");
    auto bad_size=variable;bad_size[39]=0;
    invalid([&]{(void)allocate_ranked_hot_experts(bad_size,rank,512);},"zero byte expert layer accepted");
    bad_size=variable;bad_size[39]=std::numeric_limits<uint64_t>::max();
    overflow([&]{(void)allocate_ranked_hot_experts(bad_size,rank,512);},"aligned expert byte overflow accepted");
    bad_size.fill(uint64_t{INT64_MAX}-255);
    overflow([&]{(void)allocate_ranked_hot_experts(bad_size,{{0,0},{0,1}},UINT64_MAX);},"signed cache capacity overflow accepted");
    check(available_hot_expert_bytes(1024,128,256,128,128)==384,"auto capacity must subtract only subsequent reserves");
    invalid([&]{(void)available_hot_expert_bytes(640,128,256,128,128);},"empty auto capacity accepted");
    invalid([&]{(void)available_hot_expert_bytes(639,128,256,128,128);},"reserve underflow accepted");
    overflow([&]{(void)available_hot_expert_bytes(UINT64_MAX,UINT64_MAX,1,0,0);},"reserve overflow accepted");
    int elements = 0, bytes = 0;
    check(strata::block_geometry(26, elements, bytes) && elements == 1 && bytes == 4, "I32 block geometry");
    check(strata::block_geometry(39, elements, bytes) && elements == 32 && bytes == 17, "MXFP4 block geometry");
    strata::TensorInfo overflowing{"bad", {5120, uint64_t{1} << 63}, 11, 0};
    check(strata::tensor_payload_bytes(overflowing) == 0, "tensor overflow must be rejected");
    if (!failures) std::cout << "deepseek41_contract_test: passed\n";
    return failures ? 1 : 0;
}
