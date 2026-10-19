#pragma once

#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

namespace strata::models::deepseek41 {
// Single-request specialization of the upstream expected-output / step-cost rule.
// Confidence is conditional on the preceding draft prefix, so survival is cumulative.
inline int adaptive_verify_count(std::span<const float> confidence,int maximum,
                                 double draft_ms,std::span<const double> verify_ms) {
    if(maximum<0||maximum>7||verify_ms.size()<=static_cast<size_t>(maximum)||confidence.size()<static_cast<size_t>(maximum)||
       !std::isfinite(draft_ms)||draft_ms<0)
        throw std::invalid_argument("DSpark adaptive verification input/cost");
    for(int k=0;k<=maximum;++k)
        if(!std::isfinite(verify_ms[k])||verify_ms[k]<=0)
            throw std::invalid_argument("DSpark verification costs must be calibrated and positive");
    int best=0;double survival=1,expected=1,best_rate=1/(draft_ms+verify_ms[0]);
    for(int k=1;k<=maximum;++k){
        const double p=confidence[k-1];
        if(!std::isfinite(p)||p<0||p>1)
            throw std::invalid_argument("DSpark confidence must be a finite probability");
        survival*=p;expected+=survival;
        const double rate=expected/(draft_ms+verify_ms[k]);
        if(rate>best_rate){best_rate=rate;best=k;}
    }
    return best;
}
}
