#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::models::deepseek41 {
inline constexpr uint64_t expert_host_page=2ull<<20;
// Keep each registration below the slow high-offset region observed on the
// target host. This is a layout policy, not a CUDA API size restriction.
inline constexpr uint64_t expert_host_registration_limit=(4ull<<30)-expert_host_page;

// Repack a validated cold complement in layer/expert order. Every nonempty
// layer starts a new registration; larger layers split only between blobs.
// An explicit segment count instead balances cold bytes across layers.
// Returned boundaries include zero and the padded arena size. All-hot layers
// produce no empty registrations. Leave offsets unchanged if planning fails.
inline std::vector<uint64_t> make_expert_host_segments(const std::vector<uint64_t>& layer_bytes,
                                                      std::size_t experts_per_layer,
                                                      std::vector<uint64_t>& offsets,
                                                      std::size_t segment_count=0){
    if(!experts_per_layer || layer_bytes.size()>std::numeric_limits<std::size_t>::max()/experts_per_layer ||
       offsets.size()!=layer_bytes.size()*experts_per_layer)
        throw std::invalid_argument("DeepSeek expert host layout dimensions");
    for(auto bytes:layer_bytes)if(!bytes || bytes>expert_host_registration_limit)
        throw std::invalid_argument("DeepSeek expert blob exceeds host registration span");
    auto next=offsets;
    std::vector<uint64_t> boundaries{0};
    uint64_t bytes=0;
    auto close_segment=[&]{
        if(bytes==boundaries.back())return;
        if(bytes>std::numeric_limits<uint64_t>::max()-(expert_host_page-1))
            throw std::overflow_error("DeepSeek expert host alignment overflow");
        bytes=(bytes+expert_host_page-1)/expert_host_page*expert_host_page;
        boundaries.push_back(bytes);
    };
    if(segment_count){
        std::vector<std::size_t> cold;
        uint64_t remaining=0;
        for(std::size_t index=0;index<next.size();++index){
            if(next[index]==std::numeric_limits<uint64_t>::max())continue;
            const auto blob=layer_bytes[index/experts_per_layer];
            if(remaining>std::numeric_limits<uint64_t>::max()-blob)
                throw std::overflow_error("DeepSeek expert host size overflow");
            remaining+=blob;cold.push_back(index);
        }
        if(cold.size()<segment_count)
            throw std::invalid_argument("DeepSeek fewer cold experts than host segments");
        std::size_t cursor=0;
        for(std::size_t left=segment_count;left;--left){
            const auto target=remaining/left+(remaining%left!=0);
            uint64_t payload=0;
            while(cursor<cold.size() && cold.size()-cursor>=left){
                const auto index=cold[cursor];
                const auto blob=layer_bytes[index/experts_per_layer];
                if(payload && (payload>=target || payload>expert_host_registration_limit-blob))break;
                if(bytes>std::numeric_limits<uint64_t>::max()-blob)
                    throw std::overflow_error("DeepSeek expert host size overflow");
                next[index]=bytes;bytes+=blob;payload+=blob;remaining-=blob;++cursor;
            }
            if(!payload)throw std::invalid_argument("DeepSeek empty balanced host segment");
            close_segment();
        }
        if(cursor!=cold.size())throw std::invalid_argument("DeepSeek cold experts exceed requested host segments");
        offsets=std::move(next);
        return boundaries;
    }
    for(std::size_t layer=0;layer<layer_bytes.size();++layer){
        const uint64_t blob=layer_bytes[layer];
        for(std::size_t expert=0;expert<experts_per_layer;++expert){
            const auto index=layer*experts_per_layer+expert;
            if(next[index]==std::numeric_limits<uint64_t>::max())continue;
            if(bytes-boundaries.back()>expert_host_registration_limit-blob)close_segment();
            if(bytes>std::numeric_limits<uint64_t>::max()-blob)
                throw std::overflow_error("DeepSeek expert host size overflow");
            next[index]=bytes;bytes+=blob;
        }
        close_segment();
    }
    offsets=std::move(next);
    return boundaries;
}
}
