#include "strata/models/deepseek41/indexer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace strata::models::deepseek41 {
std::vector<uint8_t> candidate_mask_cpu(const float* scores,int rows,int visible) {
    const int blocks=(rows+7)/8;
    std::vector<float> maximum(blocks,-INFINITY);
    std::vector<int> indices(blocks);
    std::iota(indices.begin(),indices.end(),0);
    for(int i=0;i<std::min(rows,visible);++i)maximum[i/8]=std::max(maximum[i/8],scores[i]);
    if(visible>0)maximum[(std::min(rows,visible)-1)/8]=INFINITY;
    const int selected=std::min(blocks,2048);
    std::partial_sort(indices.begin(),indices.begin()+selected,indices.end(),[&](int a,int b){
        return maximum[a]==maximum[b]?a<b:maximum[a]>maximum[b];});
    std::vector<uint8_t> mask(rows,0);
    for(int i=0;i<selected;++i)if(maximum[indices[i]]>-INFINITY)
        for(int j=indices[i]*8;j<std::min((indices[i]+1)*8,std::min(rows,visible));++j)mask[j]=1;
    return mask;
}
std::vector<int64_t> index_topk_cpu(const float* scores,int rows,int visible,const uint8_t* candidate) {
    std::vector<int> indices;
    for(int i=0;i<std::min(rows,visible);++i)if((!candidate||candidate[i])&&scores[i]>-INFINITY)indices.push_back(i);
    const int selected=std::min<int>(512,indices.size());
    std::partial_sort(indices.begin(),indices.begin()+selected,indices.end(),[&](int a,int b){
        return scores[a]==scores[b]?a<b:scores[a]>scores[b];});
    indices.resize(selected);
    std::sort(indices.begin(),indices.end());
    std::vector<int64_t> result(512,-1);
    std::copy(indices.begin(),indices.end(),result.begin());
    return result;
}
} // namespace strata::models::deepseek41
