#pragma once
#include <string>

namespace strata::models::deepseek41 {
// A proof is produced after a stable full SHA-256 read. A changed file
// invalidates it before any GPU weights are allocated.
void require_integrity(const std::string& proof, const std::string& kind, const std::string& path, const std::string& expected_override = {});
} // namespace strata::models::deepseek41
