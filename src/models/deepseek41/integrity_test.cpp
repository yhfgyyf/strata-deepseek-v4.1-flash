#include "strata/models/deepseek41/integrity.hpp"
#include "strata/models/deepseek41/iq3xxs_digests.hpp"
#include "strata/models/deepseek41/q2k_xl_digests.hpp"
#include "strata/artifact/gguf_split.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

using strata::models::deepseek41::require_integrity;
int main() {
    try {
        const auto root=std::filesystem::temp_directory_path()/
            ("strata-ds41-proof-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(root);
        struct Cleanup { std::filesystem::path root; ~Cleanup(){std::filesystem::remove_all(root);} } cleanup{root};
        const auto weight=root/"weight",proof=root/"proof";
        std::ofstream(weight)<<"fixture";
        struct stat s{};if(stat(weight.c_str(),&s))throw std::runtime_error("fixture stat");
#ifdef __APPLE__
        const auto modified=uint64_t(s.st_mtimespec.tv_sec)*1000000000+s.st_mtimespec.tv_nsec;
        const auto changed=uint64_t(s.st_ctimespec.tv_sec)*1000000000+s.st_ctimespec.tv_nsec;
#else
        const auto modified=uint64_t(s.st_mtim.tv_sec)*1000000000+s.st_mtim.tv_nsec;
        const auto changed=uint64_t(s.st_ctim.tv_sec)*1000000000+s.st_ctim.tv_nsec;
#endif
        auto write=[&](bool correct){
            std::ofstream out(proof);out<<"STRATA_DSV41_SHA256_V1\nmain "
                <<(correct?"11f46543370256ef616b6e458b6950e148625b5f8b7545173d124445d3393c53":"wrong")<<' '
                <<s.st_dev<<' '<<s.st_ino<<' '<<s.st_size<<' '<<modified<<' '<<changed<<'\n';
        };
        auto refuses=[&](const std::string& kind){
            try{require_integrity(proof.string(),kind,weight.string());}
            catch(const std::exception&){return;}
            throw std::runtime_error("invalid proof was accepted");
        };
        refuses("main");write(false);refuses("main");write(true);
        // This tests trusted proof parsing/freshness. The tiny fixture is not an actual model hash claim.
        require_integrity(proof.string(),"main",weight.string());refuses("draft");
        std::ofstream(weight,std::ios::app)<<"changed";refuses("main");
        const auto first=(root/"fixture-00001-of-00007.gguf").string();
        const auto shards=strata::gguf_split_paths(first,false);
        for(const auto& path:shards)std::ofstream(path)<<"fixture";
        auto split_proof=[&](int count,bool q2=false,bool mixed=false){
            std::ofstream out(proof);out<<"STRATA_DSV41_SHA256_V1\n";
            for(int i=0;i<count;++i){
                struct stat q{};if(stat(shards[i].c_str(),&q))throw std::runtime_error("split fixture stat");
#ifdef __APPLE__
                const auto mt=uint64_t(q.st_mtimespec.tv_sec)*1000000000+q.st_mtimespec.tv_nsec;
                const auto ct=uint64_t(q.st_ctimespec.tv_sec)*1000000000+q.st_ctimespec.tv_nsec;
#else
                const auto mt=uint64_t(q.st_mtim.tv_sec)*1000000000+q.st_mtim.tv_nsec;
                const auto ct=uint64_t(q.st_ctim.tv_sec)*1000000000+q.st_ctim.tv_nsec;
#endif
                out<<"main."<<i+1<<' '<<(q2&&!(mixed&&i==3)?strata::models::deepseek41::q2k_xl_sha256[i]:strata::models::deepseek41::iq3xxs_sha256[i])<<' '
                    <<q.st_dev<<' '<<q.st_ino<<' '<<q.st_size<<' '<<mt<<' '<<ct<<'\n';
            }
        };
        auto split_refuses=[&]{
            try{require_integrity(proof.string(),"main",first);}
            catch(const std::exception&){return;}
            throw std::runtime_error("incomplete or stale split proof accepted");
        };
        split_proof(6);split_refuses();
        split_proof(7);require_integrity(proof.string(),"main",shards[3]);
        split_proof(7,true);require_integrity(proof.string(),"main",shards[3]);
        split_proof(7,true,true);split_refuses();
        split_proof(7,true);
        std::ofstream(shards.back(),std::ios::app)<<"changed";split_refuses();
        split_proof(7);
        std::filesystem::remove(shards.back());split_refuses();
        std::cout<<"DeepSeek proof identity and refusal passed\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
