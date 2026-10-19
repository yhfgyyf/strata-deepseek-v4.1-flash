#include "strata/models/deepseek41/indexer.hpp"
#include "strata/models/deepseek41/attention.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace d = strata::models::deepseek41;
namespace {
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class T>struct Device{
    T* p=nullptr;size_t n;
    explicit Device(size_t count):n(count){check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    explicit Device(const std::vector<T>& values):Device(values.size()){check(cudaMemcpy(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){cudaFree(p);}
    Device(const Device&)=delete;
    std::vector<T>read(){std::vector<T>values(n);check(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return values;}
};
struct Sorter{
    Device<float>scores;Device<int>input,output;Device<int64_t>offsets;Device<uint8_t>temporary;
    d::IndexSortBuffers buffers;
    static size_t bytes(int tokens,int rows){d::IndexSortBuffers b;check(d::sort_index_gpu(nullptr,tokens,rows,b,nullptr));return b.temporary_bytes;}
    Sorter(int tokens,int rows):scores(int64_t(tokens)*rows),input(int64_t(tokens)*rows),output(int64_t(tokens)*rows),offsets(tokens+1),temporary(bytes(tokens,rows)){
        buffers={scores.p,input.p,output.p,offsets.p,temporary.p,temporary.n};
    }
    void run(const float* source,int tokens,int rows){buffers.temporary_bytes=temporary.n;check(d::sort_index_gpu(source,tokens,rows,buffers,nullptr));}
};
void selection(int rows,int tokens,int64_t start,int ratio){
    const int blocks=(rows+7)/8;
    std::vector<float>scores(int64_t(tokens)*rows,-INFINITY);
    for(int t=0;t<tokens;++t)for(int k=0;k<rows;++k)if(k<(start+t+1)/ratio)scores[t*rows+k]=float(k);
    Device<float>source(scores),block_scores(tokens*blocks);Sorter sort(tokens,rows),block_sort(tokens,blocks);
    Device<uint8_t>block_mask(tokens*blocks),mask(tokens*rows);Device<int64_t>top(tokens*512);
    check(d::candidate_scores_gpu(source.p,block_scores.p,tokens,rows,start,ratio,nullptr));
    block_sort.run(block_scores.p,tokens,blocks);
    check(d::candidate_mask_gpu(block_sort.buffers,block_mask.p,mask.p,tokens,rows,start,ratio,nullptr));
    const auto actual_mask=mask.read();
    for(int t=0;t<tokens;++t){const auto expected=d::candidate_mask_cpu(scores.data()+t*rows,rows,(start+t+1)/ratio);
        require(std::equal(expected.begin(),expected.end(),actual_mask.begin()+t*rows),"candidate mask/partial latest block mismatch");}
    sort.run(source.p,tokens,rows);check(d::index_topk_gpu(sort.buffers,top.p,tokens,rows,start,ratio,nullptr));
    const auto selected=top.read();
    for(int t=0;t<tokens;++t){const auto expected=d::index_topk_cpu(scores.data()+t*rows,rows,(start+t+1)/ratio);
        require(std::equal(expected.begin(),expected.end(),selected.begin()+t*512),"sorted causal top512 mismatch");}
    std::cout<<"selection rows="<<rows<<" tokens="<<tokens<<" ratio="<<ratio<<" passed\n";
}
void compact_selection(int rows,int tokens,int64_t start,int ratio,bool ties){
    const int blocks=(rows+7)/8;
    std::vector<float> values(int64_t(tokens)*rows,-INFINITY);
    for(int t=0;t<tokens;++t)for(int k=0;k<std::min<int64_t>(rows,(start+t+1)/ratio);++k)
        values[int64_t(t)*rows+k]=ties?(k&1?-0.f:0.f):float((k*17+t*13)%31-15);
    Device<float> source(values),bs(tokens*blocks);Sorter bsort(tokens,blocks);
    Device<int32_t> ids(tokens*2048);
    check(d::candidate_scores_gpu(source.p,bs.p,tokens,rows,start,ratio,nullptr));
    bsort.run(bs.p,tokens,blocks);check(d::candidate_blocks_gpu(bsort.buffers,ids.p,tokens,rows,nullptr));
    const auto candidates=ids.read();std::vector<float> compact(tokens*16384,-INFINITY);
    for(int t=0;t<tokens;++t){
        const int visible=std::min<int64_t>(rows,(start+t+1)/ratio);
        const auto expected_mask=d::candidate_mask_cpu(values.data()+int64_t(t)*rows,rows,visible);
        std::vector<uint8_t> got(rows,0);int previous=-1;bool padded=false;
        for(int i=0;i<2048;++i){const int block=candidates[t*2048+i];
            if(block<0){padded=true;continue;}
            require(!padded&&block>previous&&block<blocks,"compact block IDs are not canonical/unique");previous=block;
            for(int j=0;j<8;++j){const int key=block*8+j;
                if(key<visible){got[key]=1;compact[t*16384+i*8+j]=values[int64_t(t)*rows+key];}}
        }
        require(got==expected_mask,"compact candidates changed causal/newest/tied block selection");
    }
    Device<float> scores(compact);Sorter sorter(tokens,16384);Device<int64_t> output(tokens*512);
    sorter.run(scores.p,tokens,16384);check(d::compact_index_topk_gpu(sorter.buffers,ids.p,output.p,tokens,rows,start,ratio,nullptr));
    const auto actual=output.read();
    for(int t=0;t<tokens;++t){const int visible=std::min<int64_t>(rows,(start+t+1)/ratio);
        const auto mask=d::candidate_mask_cpu(values.data()+int64_t(t)*rows,rows,visible);
        const auto expected=d::index_topk_cpu(values.data()+int64_t(t)*rows,rows,visible,mask.data());
        require(std::equal(expected.begin(),expected.end(),actual.begin()+t*512),"compact top512 changed stable ties or IDs");}
    std::cout<<"compact selection rows="<<rows<<" tokens="<<tokens<<" ratio="<<ratio<<" ties="<<ties<<" exact\n";
}
void scoring(bool packed){
    constexpr int tokens=3,rows=1159;
    const int64_t start=1156;
    std::vector<float>q(tokens*32*128),keys(rows*128),weights(tokens*32);
    for(size_t i=0;i<q.size();++i)q[i]=std::sin(float(i)*.011f);
    for(size_t i=0;i<keys.size();++i)keys[i]=std::cos(float(i)*.021f);
    for(size_t i=0;i<weights.size();++i)weights[i]=std::sin(float(i)*.1f);
    Device<float>dq(q),dk(keys),dw(weights),scores(tokens*rows);Device<uint8_t>pk(rows*68);
    if(packed){check(d::fake_quant_index_gpu(dq.p,tokens*32,nullptr));check(d::pack_index_gpu(dk.p,pk.p,rows,nullptr));q=dq.read();
        const auto bytes=pk.read();for(int r=0;r<rows;++r)for(int k=0;k<128;++k){const auto* p=bytes.data()+r*68;
            keys[r*128+k]=d::decode_fp4_cpu((p[k/2]>>(k%2*4))&15)*std::ldexp(1.f,int(p[64+k/32])-127);}}
    check(d::index_scores_gpu(dq.p,dw.p,packed?nullptr:dk.p,packed?pk.p:nullptr,nullptr,scores.p,tokens,rows,start,1,nullptr));
    const auto got=scores.read();double maximum=0;
    for(int t=0;t<tokens;++t)for(int k=0;k<rows;++k){if(k>=start+t+1){require(got[t*rows+k]==-INFINITY,"future index key visible");continue;}
        double expected=0;for(int h=0;h<32;++h){double dot=0;for(int i=0;i<128;++i)dot+=double(q[(t*32+h)*128+i])*keys[k*128+i];
            expected+=std::max(dot,0.)*weights[t*32+h]/64.;}
        const double error=std::abs(got[t*rows+k]-expected);maximum=std::max(maximum,error);
        require(error<=1e-5+1e-4*std::abs(expected),"index score arithmetic mismatch");}
    std::cout<<"index scoring packed="<<packed<<" max_abs="<<maximum<<'\n';
    Device<float> bs(tokens*((rows+7)/8)),compact(tokens*16384);Sorter bsort(tokens,(rows+7)/8);
    Device<int32_t> blocks(tokens*2048);
    check(d::candidate_scores_gpu(scores.p,bs.p,tokens,rows,start,1,nullptr));bsort.run(bs.p,tokens,(rows+7)/8);
    check(d::candidate_blocks_gpu(bsort.buffers,blocks.p,tokens,rows,nullptr));
    check(d::compact_index_scores_gpu(dq.p,dw.p,packed?nullptr:dk.p,packed?pk.p:nullptr,blocks.p,compact.p,tokens,rows,start,1,nullptr));
    const auto cb=blocks.read();const auto cs=compact.read();
    for(int t=0;t<tokens;++t)for(int c=0;c<16384;++c){const int b=cb[t*2048+c/8],key=b*8+c%8;
        const float expected=b>=0&&key<rows&&key<start+t+1?got[t*rows+key]:-INFINITY;
        require(std::memcmp(&expected,&cs[t*16384+c],sizeof(float))==0,"compact scoring changed exact reduction or causal mask");}
    std::cout<<"compact scoring packed="<<packed<<" bitwise exact\n";
}
}
int main(){try{
    for(int rows:{1,2,3,127,128,129,511,512,513,1159,16383,16384,16385,32768})selection(rows,3,std::max(rows-3,0),1);
    selection(513,3,1023,2);
    for(int rows:{1,127,128,511,512,513,16383,16384,16385,32768})
        for(bool ties:{false,true})compact_selection(rows,3,std::max(rows-3,0),1,ties);
    compact_selection(16393,3,32781,2,true);
    compact_selection(1048576,3,1048573,1,false);compact_selection(1048576,3,1048573,1,true);
    scoring(false);scoring(true);check(cudaDeviceSynchronize());
    std::cout<<"deepseek41_indexer_test: passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
