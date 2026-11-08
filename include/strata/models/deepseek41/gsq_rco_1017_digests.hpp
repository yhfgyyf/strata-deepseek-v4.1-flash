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
// Publisher's PLE-compatible six-shard layout, verified from ModelScope 2026-11-05.
// The last shard is the renamed/linked IQ4_NL Engram table, with its own header.
inline constexpr std::array<const char*,6> gsq_rco_1017_ple_sha256{
    "0182a2a1f42d319a0dbc6e34aa82a063880c069fe3aafbbea86018f386e64e5f",
    "5c88039fa640460a9fc3465a6dfcd162a4083dd738d55e5b52e025be14555a08",
    "0fddebfd05f0a36cfd107ba5dfd2547e64df61adbab39560a5a5f5521ca099a5",
    "c577a43c6e6e615838a1655a0b162541edeac6f36542f08c4a9593a70dcaf500",
    "851d8605bf504962054d17ea8a0d9e2e0a228d085d396bd865a4de9f16d90fbd",
    "3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3"
};
// AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF, IQ2_XS.
// Revision b8bca19979431f96126ceac8cbe0ee7160a28f8a changes only shard four.
inline constexpr const char* gsq_rco_1017_uncensored_shard4_sha256=
    "18305a5afbf8087e1f88a9f2978cc74650357f482629d346349ed374cb7c8451";
inline constexpr const char* gsq_rco_1017_engram_file="DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf";
inline constexpr const char* gsq_vision_bias_sha256="710b11267b4db392d1fdb558f36ba8e0e91a91f4e8180c953c912e18eacf843d";
}
