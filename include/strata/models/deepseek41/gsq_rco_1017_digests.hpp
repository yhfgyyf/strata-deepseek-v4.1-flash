#pragma once
#include <array>
namespace strata::models::deepseek41 {
// ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF, IQ2_XS.
// Publisher revision cb02422c900df489cea5833e8a92171224059b3c.
inline constexpr std::array<const char*,5> gsq_rco_1017_sha256{
    "4c65307992ab586c527ca1f6f64062aa96a33b23c02b7fed103f45f75e854798",
    "933461a974fcccee741bf2f9798201e72ab4aa3f8a952be56f849896ca6ea30e",
    "bbf91fa346a85ebaf32dd4fd5c456b534d780d381e54f46ac90783a481373240",
    "e5b2f8781fab688cd4bc3bc7a69651b5dbb90ed94965657c6bead8bb4774f3cf",
    "eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a"
};
inline constexpr const char* gsq_rco_1017_engram_file="DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf";
}
