#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::models::deepseek41 {
inline constexpr int PREFILL_CHUNK_MAX=32768;
struct PrefillChunk {
    int tokens;
    bool automatic;
};

inline PrefillChunk parse_prefill_chunk(const std::string& value) {
    if(value=="auto")return {PREFILL_CHUNK_MAX,true};
    const bool automatic=value.rfind("auto:",0)==0;
    const std::string number=automatic?value.substr(5):value;
    const char* error="DeepSeek chunk must be 1..32768, auto, or auto:N (N a multiple of 256 in 256..32768)";
    if(number.empty())throw std::invalid_argument(error);
    int tokens=0;
    for(char c:number){
        if(c<'0'||c>'9'||tokens>(PREFILL_CHUNK_MAX-(c-'0'))/10)throw std::invalid_argument(error);
        tokens=tokens*10+(c-'0');
    }
    if(tokens<1||tokens>PREFILL_CHUNK_MAX||(automatic&&(tokens<256||tokens%256)))throw std::invalid_argument(error);
    return {tokens,automatic};
}

// Query the real layout without allocating device buffers. A descending scan
// does not assume that the CUDA library's scratch sizes are monotonic.
template<class BytesNeeded>
int select_prefill_chunk(int ceiling,int context,uint64_t budget,BytesNeeded bytes_needed) {
    if(ceiling<1||ceiling>PREFILL_CHUNK_MAX||context<1||!budget)throw std::invalid_argument("DeepSeek auto chunk bounds");
    int chunk=std::min(ceiling,context);
    if(chunk>=256)chunk=chunk/256*256;
    for(;chunk>0;chunk=chunk>256?chunk-256:0)
        if(bytes_needed(std::max(128,chunk))<=budget)return chunk;
    throw std::runtime_error("DeepSeek auto prefill cannot fit the minimum chunk in --work-mib");
}
} // namespace strata::models::deepseek41
