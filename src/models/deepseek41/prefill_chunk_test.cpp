#include "strata/models/deepseek41/prefill_chunk.hpp"
#include <iostream>
#include <vector>

namespace d=strata::models::deepseek41;
namespace {
void require(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
void parsing(){
    for(int value:{1,128,1024,8192,32768}){
        const auto chunk=d::parse_prefill_chunk(std::to_string(value));
        require(chunk.tokens==value&&!chunk.automatic,"numeric chunk changed");
    }
    const auto automatic=d::parse_prefill_chunk("auto");
    require(automatic.tokens==32768&&automatic.automatic,"auto ceiling");
    for(int value:{256,4096,8192,32768}){
        const auto chunk=d::parse_prefill_chunk("auto:"+std::to_string(value));
        require(chunk.tokens==value&&chunk.automatic,"explicit auto ceiling");
    }
    for(const char* value:{"","0","-1","32769","99999999999999999999","8192x"," 8192","1.5","Auto","auto:","auto:128","auto:257","auto:32769"}){
        bool rejected=false;try{d::parse_prefill_chunk(value);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"invalid chunk accepted");
    }
}
void selection(){
    auto bytes=[](int tokens){return uint64_t(tokens)*1000+5000;};
    for(int context:{8192,32768,131072,524288}){
        const int expected=std::min(context,32768);
        require(d::select_prefill_chunk(32768,context,bytes(32768),bytes)==expected,"context/ceiling selection");
    }
    require(d::select_prefill_chunk(32768,524288,bytes(7168),bytes)==7168,"work budget fallback");
    require(d::select_prefill_chunk(8192,524288,bytes(32768),bytes)==8192,"explicit auto cap ignored");
    require(d::select_prefill_chunk(32768,8193,bytes(32768),bytes)==8192,"256-token alignment");
    std::vector<int> queried;
    require(d::select_prefill_chunk(32768,17,bytes(128),[&](int n){queried.push_back(n);return bytes(n);})==17,"short context chunk");
    require(queried==std::vector<int>{128},"minimum workspace capacity");
    bool refused=false;try{d::select_prefill_chunk(32768,524288,bytes(256)-1,bytes);}catch(const std::runtime_error&){refused=true;}
    require(refused,"impossible budget silently accepted");
    require(d::select_prefill_chunk(1024,1024,1,[](int n){return uint64_t(n==768?1:2);})==768,"nonmonotonic library scratch");
    bool propagated=false;
    try{d::select_prefill_chunk(32768,524288,1,[](int)->uint64_t{throw std::logic_error("query failed");});}
    catch(const std::logic_error&){propagated=true;}
    require(propagated,"library query failure treated as insufficient memory");
}
}
int main(){
    try{parsing();selection();std::cout<<"DeepSeek prefill auto parser/budget/context/boundaries PASS\n";return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
