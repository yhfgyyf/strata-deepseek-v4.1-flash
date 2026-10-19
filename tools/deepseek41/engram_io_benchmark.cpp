// Real immutable row-query replay, without loading expert weights or executing a model.
#include "strata/models/deepseek41/engram.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace d=strata::models::deepseek41;
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("MODEL ONE_PROMPT.tokens");
    d::Model model(argv[1]);std::ifstream f(argv[2]);std::string id;int n;
    if(!(f>>id>>n)||n<1||n>8192)throw std::runtime_error("prompt shape");
    std::vector<int> tokens(n);for(int& token:tokens)if(!(f>>token))throw std::runtime_error("prompt tokens");
    f>>std::ws;if(!f.eof())throw std::runtime_error("extra input");
    const auto query=d::engram_hash(d::engram_hash_spec(model),model.token_map(),tokens,{},{});
    std::vector<uint8_t> reference;
    for(int arm:{0,1,2,2,1,0}){
        d::EngramReader reader(model,8192ull<<20,arm!=0,arm==2);
        const auto start=std::chrono::steady_clock::now();
        const auto batch=reader.submit(query,1,true);batch->wait_layer(0);
        const double first=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        const auto& rows=batch->wait();const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(reference.empty())reference=rows;
        else if(reference.size()!=rows.size() || std::memcmp(reference.data(),rows.data(),rows.size()))throw std::runtime_error("real Engram row bytes differ");
        const auto st=reader.stats();
        std::cout<<std::setprecision(12)<<"{\"continuous\":"<<arm<<",\"actual_input\":"<<n<<",\"first_ready_ms\":"<<first<<",\"wall_ms\":"<<wall
                 <<",\"exact\":true,\"logical_bytes\":"<<st.logical_bytes<<",\"disk_bytes\":"<<st.disk_bytes<<",\"reads\":"<<st.reads
                 <<",\"table_wall_us\":["<<st.layer_wall_us[0]<<','<<st.layer_wall_us[1]<<"],\"table_disk_bytes\":["<<st.layer_disk_bytes[0]<<','<<st.layer_disk_bytes[1]<<"]}"<<std::endl;
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
