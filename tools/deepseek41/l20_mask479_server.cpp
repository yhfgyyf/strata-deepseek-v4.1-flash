// Use the validated mask479 libraries through the original production IPC server.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/models/deepseek41/mixed_expert.hpp"
#define main strata_original_main
#include "experimental_server.cpp"
#undef main
int main(int argc, char** argv) {
    strata::kernels::native_mmvq_l20_q5_n6_set_enabled(true);
    strata::kernels::native_q5_output_a_set_enabled(true);
    strata::kernels::native_mmvq_l20_q5_extra_set_enabled(true);
    strata::models::deepseek41::expert_mixed_l20_n1_warp_set_enabled(true);
    strata::models::deepseek41::expert_mixed_l20_short_tile_set_enabled(true);
    std::cerr << "L20 cumulative mask479: Q5 N6, N1 warp, Q5 output-A, short GU N3, shared GU N6 enabled\n";
    return strata_original_main(argc, argv);
}
