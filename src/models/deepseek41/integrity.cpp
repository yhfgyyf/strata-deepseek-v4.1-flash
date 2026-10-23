#include "strata/models/deepseek41/integrity.hpp"
#include "strata/models/deepseek41/iq3xxs_digests.hpp"
#include "strata/models/deepseek41/q2k_xl_digests.hpp"
#include "strata/artifact/gguf_split.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <sys/stat.h>

namespace strata::models::deepseek41 {
namespace {
void require_file(const std::string& proof,const std::string& kind,const std::string& path,const std::string& expected) {
    if (expected.empty() || path.empty() || proof.empty()) throw std::invalid_argument("DeepSeek full-hash proof is required");
    struct stat state{};
    if (stat(path.c_str(), &state) || !S_ISREG(state.st_mode)) throw std::runtime_error("DeepSeek weight file is not regular");
#ifdef __APPLE__
    const uint64_t modified = uint64_t(state.st_mtimespec.tv_sec) * 1000000000 + state.st_mtimespec.tv_nsec;
    const uint64_t changed = uint64_t(state.st_ctimespec.tv_sec) * 1000000000 + state.st_ctimespec.tv_nsec;
#else
    const uint64_t modified = uint64_t(state.st_mtim.tv_sec) * 1000000000 + state.st_mtim.tv_nsec;
    const uint64_t changed = uint64_t(state.st_ctim.tv_sec) * 1000000000 + state.st_ctim.tv_nsec;
#endif
    const std::array<uint64_t, 5> actual{uint64_t(state.st_dev), uint64_t(state.st_ino), uint64_t(state.st_size), modified, changed};
    std::ifstream input(proof);
    std::string header;
    std::getline(input, header);
    if (header != "STRATA_DSV41_SHA256_V1") throw std::runtime_error("DeepSeek full-hash proof header is missing");
    std::string name, hash;
    std::array<uint64_t, 5> identity{};
    bool found = false;
    while (input >> name >> hash >> identity[0] >> identity[1] >> identity[2] >> identity[3] >> identity[4]) {
        if (name != kind) continue;
        if (found || hash != expected || identity != actual)
            throw std::runtime_error("DeepSeek " + kind + " full-hash proof is stale or does not match the approved weights");
        found = true;
    }
    if (!input.eof() || !found) throw std::runtime_error("DeepSeek " + kind + " full-hash proof is missing or invalid");
}
}
void require_integrity(const std::string& proof, const std::string& kind, const std::string& path, const std::string& expected_override) {
    std::string expected = kind == "main" ? "11f46543370256ef616b6e458b6950e148625b5f8b7545173d124445d3393c53" :
        kind == "draft" ? "d5de12076cac2378cf8cdbf0853d8e0afb0a0cd06e6ef2829d0aceee69ff94a9" :
        kind == "vision" ? "bd8a7593fc3721fad8ef84689beeb65437cd7a75326f005d6f741521a872d09e" : "";
    if(!expected_override.empty()){
        if((kind!="draft"&&kind!="vision")||expected_override.size()!=64||expected_override.find_first_not_of("0123456789abcdef")!=std::string::npos)
            throw std::invalid_argument("DeepSeek sidecar digest must be lowercase SHA256");
        expected=expected_override;
    }
    if(kind=="main"){
        const auto shards=gguf_split_paths(path);
        if(shards.size()>1){
            if(shards.size()!=iq3xxs_sha256.size())throw std::invalid_argument("DeepSeek unapproved main shard count");
            // Both checkpoints have the same metadata-only first shard. Select a
            // complete approved family from shard two, then verify every member.
            const std::array<const char*,7>* family=nullptr;
            std::ifstream input(proof);std::string line;
            while(std::getline(input,line)){
                std::istringstream row(line);std::string name,hash;row>>name>>hash;
                if(name!="main.2")continue;
                if(family)throw std::runtime_error("DeepSeek duplicate main.2 proof");
                if(hash==iq3xxs_sha256[1])family=&iq3xxs_sha256;
                else if(hash==q2k_xl_sha256[1])family=&q2k_xl_sha256;
                else throw std::runtime_error("DeepSeek unapproved main checkpoint family");
            }
            if(!family)throw std::runtime_error("DeepSeek main checkpoint family proof is missing");
            for(size_t i=0;i<shards.size();++i)
                require_file(proof,"main."+std::to_string(i+1),shards[i],(*family)[i]);
            return;
        }
    }
    require_file(proof,kind,path,expected);
}
} // namespace strata::models::deepseek41
