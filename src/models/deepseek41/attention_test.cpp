#include "strata/models/deepseek41/attention.hpp"
#include "strata/models/deepseek41/ops.hpp"
#include "strata/artifact/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace d = strata::models::deepseek41;
namespace {
void check(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
template<class T> struct Device {
    T* p=nullptr; size_t count;
    explicit Device(size_t n):count(n){check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    explicit Device(const std::vector<T>& v):Device(v.size()){upload(v);}
    ~Device(){cudaFree(p);}
    Device(const Device&)=delete;
    void upload(const std::vector<T>& v){check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read(){std::vector<T> v(count);check(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
uint16_t bf16(float x){uint32_t bits;std::memcpy(&bits,&x,4);bits+=0x7fff+((bits>>16)&1);return bits>>16;}
void require(bool ok,const char* label){if(!ok)throw std::runtime_error(label);}
void compare(const std::vector<float>& got,const std::vector<float>& reference,const char* label,bool packed){
    require(got.size()==reference.size(),"attention test shape");
    double numerator=0,denominator=0,maximum=0;
    for(size_t i=0;i<got.size();++i){
        require(std::isfinite(got[i]),"nonfinite attention output");
        const double error=double(got[i])-reference[i];
        numerator+=error*error;denominator+=double(reference[i])*reference[i];maximum=std::max(maximum,std::abs(error));
        if(!packed)require(std::abs(error)<=1e-5+1e-4*std::abs(reference[i]),label);
    }
    const double relative=std::sqrt(numerator/(denominator+1e-30));
    if(packed)require(relative<=1e-2,label);
    std::cout<<label<<" relative_L2="<<relative<<" max_abs="<<maximum<<'\n';
}
std::vector<float> reference_attention(const std::vector<float>& q,const std::vector<float>& swa,
                                      const std::vector<float>& global,const std::vector<int64_t>& sw_slots,
                                      const std::vector<int64_t>& global_slots,const std::vector<float>& sinks,
                                      int tokens,int sw_topk,int global_topk){
    std::vector<float> output(int64_t(tokens)*64*512);
    for(int t=0;t<tokens;++t)for(int h=0;h<64;++h){
        std::vector<const float*> values;std::vector<double> scores;
        double maximum=sinks[h];
        for(int candidate=0;candidate<sw_topk+global_topk;++candidate){
            const bool glob=candidate>=sw_topk;
            const auto& slots=glob?global_slots:sw_slots;
            const auto& cache=glob?global:swa;
            const int width=glob?global_topk:sw_topk,index=glob?candidate-sw_topk:candidate;
            const int64_t slot=slots[t*width+index];
            if(slot<0||slot>=int64_t(cache.size()/512))continue;
            const float* row=cache.data()+slot*512;double dot=0;
            for(int i=0;i<512;++i)dot+=double(q[(int64_t(t)*64+h)*512+i])*row[i];
            dot*=1./std::sqrt(512.);values.push_back(row);scores.push_back(dot);maximum=std::max(maximum,dot);
        }
        if(maximum==-INFINITY)continue;
        double denominator=std::exp(double(sinks[h])-maximum);
        for(auto& score:scores){score=std::exp(score-maximum);denominator+=score;}
        for(int i=0;i<512;++i){double numerator=0;for(size_t r=0;r<values.size();++r)numerator+=scores[r]*values[r][i];
            output[(int64_t(t)*64+h)*512+i]=numerator/denominator;}
    }
    return output;
}
std::vector<float> from_bf16(const std::vector<uint16_t>& x){std::vector<float> f(x.size());for(size_t i=0;i<x.size();++i)f[i]=strata::bf16_to_fp32(x[i]);return f;}

void scenario(int tokens,int global_topk,int splits,bool fused){
    constexpr int sw_rows=256,global_rows=513,sw_topk=128;
    const int64_t sw_stride=576,global_stride=320,sw_page_stride=32*sw_stride+256,global_page_stride=64*global_stride+256;
    const int64_t sw_bytes=8*sw_page_stride,global_bytes=9*global_page_stride;
    std::mt19937 generator(410+tokens);std::normal_distribution<float> normal(0.f,0.2f);
    std::vector<uint16_t> sw_kv(sw_rows*512),global_kv(global_rows*512),q(tokens*64*512);
    for(auto& x:sw_kv)x=bf16(normal(generator));
    for(auto& x:global_kv)x=bf16(normal(generator));
    for(auto& x:q)x=bf16(normal(generator));
    // Include ties, zero scales, negative zero and saturated finite values in cache pack fixtures.
    for(int i=0;i<32;++i)sw_kv[i]=bf16(i%2?-0.f:0.f);
    for(int i=0;i<16;++i)global_kv[i]=bf16(0.f);
    std::vector<int64_t> sw_insert(sw_rows),global_insert(global_rows);std::iota(sw_insert.begin(),sw_insert.end(),0);std::iota(global_insert.begin(),global_insert.end(),0);
    Device<uint16_t>d_sw_kv(sw_kv),d_global_kv(global_kv);Device<int64_t>d_sw_insert(sw_insert),d_global_insert(global_insert);
    Device<uint8_t>d_sw(sw_bytes),d_global(global_bytes);
    check(cudaMemset(d_sw.p,0xa5,sw_bytes));check(cudaMemset(d_global.p,0xa5,global_bytes));
    d::CacheLayout sw{d_sw.p,sw_bytes,sw_page_stride,sw_stride,32},glob{d_global.p,global_bytes,global_page_stride,global_stride,64};
    check(d::pack_swa_gpu(d_sw_kv.p,d_sw_insert.p,sw_rows,512,sw,nullptr));
    check(d::pack_global_gpu(d_global_kv.p,d_global_insert.p,global_rows,512,glob,nullptr));
    auto sw_data=d_sw.read(),global_data=d_global.read();
    std::vector<float> sw_plain(sw_rows*512),global_plain(global_rows*512);
    for(int r=0;r<sw_rows;++r){uint8_t expected[528];d::pack_swa_cpu(sw_kv.data()+r*512,expected);
        const int64_t offset=(r/32)*sw_page_stride+(r%32)*sw_stride;
        require(std::memcmp(expected,sw_data.data()+offset,528)==0,"SWA packed bytes differ");
        d::decode_cache_cpu(expected,false,sw_plain.data()+r*512);}
    for(int r=0;r<global_rows;++r){uint8_t expected[288];d::pack_global_cpu(global_kv.data()+r*512,expected);
        const int64_t offset=(r/64)*global_page_stride+(r%64)*global_stride;
        require(std::memcmp(expected,global_data.data()+offset,288)==0,"global packed bytes differ");
        d::decode_cache_cpu(expected,true,global_plain.data()+r*512);}
    std::vector<int64_t>sw_slots(tokens*sw_topk,-1),global_slots(tokens*global_topk,-1);
    for(int t=0;t<tokens;++t){for(int j=0;j<128;++j)sw_slots[t*128+j]=t+70+j;
        for(int j=0;j<global_topk;++j)if(j<global_rows&&j<19+t*11)global_slots[t*global_topk+j]=j;}
    sw_slots[0]=-1;sw_slots[1]=sw_rows+10;
    std::vector<float>sinks(64);for(int h=0;h<64;++h)sinks[h]=h%5==0?-INFINITY:float(h)*0.01f;
    const auto q_plain=from_bf16(q);
    const auto reference=reference_attention(q_plain,sw_plain,global_plain,sw_slots,global_slots,sinks,tokens,sw_topk,global_topk);
    // Token padding is supported; contiguous 512-value head stride is required by the native MMA tile.
    const int64_t q_stride=64*512+64;
    std::vector<uint16_t>q_padded(tokens*q_stride,0);for(int t=0;t<tokens;++t)std::copy_n(q.data()+t*64*512,64*512,q_padded.data()+t*q_stride);
    Device<uint16_t>dq(q_padded),dout(tokens*64*512),dpartial(int64_t(tokens)*splits*64*512);
    Device<float>dsinks(sinks),dlse(int64_t(tokens)*splits*64);
    Device<int64_t>dsw_slots(sw_slots),dglobal_slots(std::max<size_t>(global_slots.size(),1));
    if(!global_slots.empty())dglobal_slots.upload(global_slots);
    d::PackedAttention p;p.q=dq.p;p.swa=sw;p.global=glob;p.swa_slots=dsw_slots.p;p.global_slots=dglobal_slots.p;
    p.sinks=dsinks.p;p.partial=dpartial.p;p.lse=dlse.p;p.output=dout.p;p.tokens=tokens;p.global_topk=global_topk;p.splits=splits;p.q_token_stride=q_stride;
    auto invalid=p;invalid.q_head_stride=528;require(d::packed_attention_gpu(invalid,nullptr)==cudaErrorInvalidValue,"unsupported head stride must be refused");
    check(d::packed_attention_gpu(p,nullptr));
    const auto native_bytes=dout.read();
    const auto native=from_bf16(native_bytes);compare(native,reference,"native packed H64",true);
    if(tokens<=8 && splits==8){
        Device<uint16_t>serial(int64_t(tokens)*64*512+128);
        check(cudaMemset(serial.p,0x35,serial.count*2));
        for(int token=0;token<tokens;++token){
            auto single=p;single.tokens=1;single.q=p.q+int64_t(token)*q_stride;
            single.swa_slots=p.swa_slots+int64_t(token)*sw_topk;
            single.global_slots=global_topk?p.global_slots+int64_t(token)*global_topk:nullptr;
            single.output=serial.p+64+int64_t(token)*64*512;
            check(d::packed_attention_gpu(single,nullptr));
        }
        const auto serial_bytes=serial.read();
        require(std::equal(native_bytes.begin(),native_bytes.end(),serial_bytes.begin()+64),"batched attention changed decode bytes");
        require(std::all_of(serial_bytes.begin(),serial_bytes.begin()+64,[](uint16_t x){return x==0x3535;}) &&
                std::all_of(serial_bytes.end()-64,serial_bytes.end(),[](uint16_t x){return x==0x3535;}),"attention output guard");
    }
    Device<float>dqf(q_plain),dswf(sw_plain),dglobf(global_plain),dfout(reference.size());
    check(d::float_attention_gpu(dqf.p,dswf.p,dglobf.p,dsw_slots.p,dglobal_slots.p,dsinks.p,dfout.p,tokens,64,128,global_topk,sw_rows,global_rows,nullptr));
    compare(dfout.read(),reference,"float H64",false);
    if(fused){
        constexpr int table_rows=256;std::vector<float>table(table_rows*64);std::vector<int64_t>positions(tokens);
        const auto frequencies=d::rotary_frequencies(d::Geometry{},global_topk>0);
        for(int pos=0;pos<table_rows;++pos)for(int pair=0;pair<32;++pair){const float angle=pos*frequencies.value[pair];table[pos*64+pair]=std::cos(angle);table[pos*64+32+pair]=std::sin(angle);}
        for(int t=0;t<tokens;++t)positions[t]=128+t;
        const int capacity=tokens+3,offset=2;
        Device<float>dtable(table);Device<int64_t>dpositions(positions);Device<uint16_t>dgroup(int64_t(8)*capacity*4096);
        check(cudaMemset(dgroup.p,0x33,dgroup.count*2));
        p.positions=dpositions.p;p.cos_sin=dtable.p;p.cos_positions=table_rows;p.total_tokens=capacity;p.token_offset=offset;p.grouped_output=dgroup.p;
        auto alias=p;alias.grouped_output=const_cast<uint16_t*>(p.q);require(d::packed_attention_gpu(alias,nullptr)==cudaErrorInvalidValue,"fused output alias must be refused");
        check(d::packed_attention_gpu(p,nullptr));
        const auto grouped=dgroup.read();auto expected=native;
        for(int t=0;t<tokens;++t)d::rotate_cpu(expected.data()+t*64*512,64,512,positions[t],d::Geometry{},global_topk>0,true);
        std::vector<float>got(expected.size());
        for(int t=0;t<tokens;++t)for(int head=0;head<64;++head)for(int dim=0;dim<512;++dim)
            got[(int64_t(t)*64+head)*512+dim]=strata::bf16_to_fp32(grouped[((int64_t(head/8)*capacity+offset+t)*8+head%8)*512+dim]);
        compare(got,expected,"fused H64 inverse RoPE 8 groups",true);
        for(int g=0;g<8;++g)for(int t:{0,1,capacity-1})for(int i=0;i<4096;++i)
            require(grouped[(int64_t(g)*capacity+t)*4096+i]==0x3333,"fused output overwrote another token tile");
    }
}
void large_offset(){
    const int64_t slot=((int64_t{1}<<32)+1024+527)/528,offset=slot*528;
    const int64_t bytes=offset+528;
    Device<uint8_t>cache(bytes);Device<uint16_t>kv(std::vector<uint16_t>(512,bf16(.25f)));
    Device<int64_t>slots(std::vector<int64_t>{slot});
    d::CacheLayout layout{cache.p,bytes,32*528,528,32};
    check(d::pack_swa_gpu(kv.p,slots.p,1,512,layout,nullptr));
    Device<uint16_t>q(std::vector<uint16_t>(64*512,bf16(.1f))),out(64*512),scratch(64*512);
    Device<float>sinks(std::vector<float>(64,0.f)),lse(64);
    d::PackedAttention p;p.q=q.p;p.swa=layout;p.swa_slots=slots.p;p.swa_topk=1;p.sinks=sinks.p;p.output=out.p;p.partial=scratch.p;p.lse=lse.p;p.tokens=1;
    check(d::packed_attention_gpu(p,nullptr));
    const float qv=strata::bf16_to_fp32(bf16(.1f)),kvv=.25f;
    const float score=qv*kvv*std::sqrt(512.f),probability=1.f/(1.f+std::exp(-score));
    std::vector<float>expected(64*512,probability*kvv);compare(from_bf16(out.read()),expected,"cache offset above 4 GiB",true);
}
}
int main(){try{
    for(int tokens=1;tokens<=8;++tokens){scenario(tokens,0,8,false);scenario(tokens,512,8,false);}
    for(int splits:{1,2,64}){scenario(1,512,splits,true);scenario(3,0,splits,false);}
    scenario(35,512,1,true);scenario(35,0,1,true);scenario(35,512,2,false);
    std::vector<float>keys(7*128);for(size_t i=0;i<keys.size();++i)keys[i]=std::sin(float(i)*.37f);
    Device<float>dk(keys);Device<uint8_t>packed(7*68);check(d::pack_index_gpu(dk.p,packed.p,7,nullptr));
    const auto bytes=packed.read();for(int row=0;row<7;++row){uint8_t expected[68];d::pack_index_cpu(keys.data()+row*128,expected);
        require(std::memcmp(expected,bytes.data()+row*68,68)==0,"index packed cache bytes");}
    large_offset();check(cudaDeviceSynchronize());std::cout<<"deepseek41_attention_test: passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
