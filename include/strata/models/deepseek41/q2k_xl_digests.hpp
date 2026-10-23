#pragma once
#include <array>

namespace strata::models::deepseek41 {
// unsloth/DeepSeek-V4.1-Flash-1017-GGUF, UD-Q2_K_XL, split order.
// Publisher manifest saved on 2026-10-20; all seven files fully verified on 2026-10-22.
inline constexpr std::array<const char*,7> q2k_xl_sha256{{
    "0c2a2031bd93ebd60aa6a4f3912775ba3990b5f39b24c6edf9ea029d2f387a28",
    "605972fece1b43277b9ff86a2cd79c9002ff424caefc507f20b6ccd47ae4ab39",
    "1cb58786245bd8c38478174eecc648b12691bb79c2bb2f446c135ab00263580e",
    "7b753295dd32534dfea52b2e9b5c4741b8007a955e38702ee37034271df7526e",
    "336be674937fe29683e7aeee7c6af32f896b864931a6c916907b78e0f3692de4",
    "192694fb09612e1200cce9ae35f3645ad39cf8aa43e94417070af9c4f4ecb43b",
    "a653c5309b7c74a6c41580aaa8c2d2bbdae0342253bca2cd3fc9e69d05f3d0eb"
}};
} // namespace strata::models::deepseek41
