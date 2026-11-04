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
// AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF, IQ2_XS.
// Revision b8bca19979431f96126ceac8cbe0ee7160a28f8a changes only shard four.
inline constexpr const char* gsq_rco_1017_uncensored_shard4_sha256=
    "18305a5afbf8087e1f88a9f2978cc74650357f482629d346349ed374cb7c8451";
inline constexpr const char* gsq_rco_1017_engram_file="DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf";
inline constexpr const char* gsq_vision_bias_sha256="710b11267b4db392d1fdb558f36ba8e0e91a91f4e8180c953c912e18eacf843d";
}
