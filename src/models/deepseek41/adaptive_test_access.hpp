#pragma once
#include "adaptive_cache.hpp"
#include "expert_pipeline.hpp"
#include <iomanip>
#include <cstring>

namespace strata::models::deepseek41 {
// Test-only access: no runtime control command can rewind weights or learning.
struct PrefetchBlockBenchmark {
    struct Snapshot {
        std::vector<ExpertKey> hot;
        std::vector<uint64_t> offsets;
        std::vector<double> heat,previous,learned;
        uint64_t rows,since,batches,published;
        double host_ms;
        std::vector<uint8_t*> sources;
        std::array<std::vector<uint8_t*>,2> spares;
        uint64_t windows,direct,rotated,prepare,commit;
    };
    static Runtime::Impl& rt(Runtime& r){return *r.impl_;}
    static Session::Impl& session_impl(Session& s){return *s.impl_;}
    static void verify_session(Session& session){
        const auto& s=*session.impl_;
        if(s.transaction_active||s.transaction_routes_deferred||s.position!=s.encoder_position||
           s.position!=s.decoder_position||s.position!=s.history.position||s.tokens.size()!=size_t(s.position)||
           s.draft_capture_position!=s.position)throw std::logic_error("adaptive session watermark");
        for(int l=0;l<40;++l){
            if(s.local[l].position!=s.position)throw std::logic_error("adaptive local KV watermark");
            if(s.runtime.model.ownership()[l].owns_global()&&s.global[l].position!=s.position)
                throw std::logic_error("adaptive global KV watermark");
        }
    }
    static AdaptiveCache& cache(Runtime& r){return *rt(r).experts->adaptive_;}
    static Snapshot snapshot(AdaptiveCache& a){
        if(a.state_!=AdaptiveCache::State::idle)throw std::logic_error("snapshot requires idle adaptive cache");
        a.record_exchanges_=true;a.committed_exchanges_.clear();
        return {a.hot_,a.offsets_,a.heat_,a.previous_,a.learned_,a.rows_,a.since_pick_,a.batches_,a.published_,a.host_copy_ms_,a.sources_,a.spares_,a.windows_,a.direct_bytes_,a.rotated_,a.avoided_prepare_,a.avoided_commit_};
    }
    static void restore(AdaptiveCache& a,const Snapshot& s,cudaStream_t readers,ExecutionStats& stats,bool grouped){
        a.drain(readers,stats);gpu_check(cudaDeviceSynchronize());
        for(auto it=a.committed_exchanges_.rbegin();it!=a.committed_exchanges_.rend();++it){
            const auto& x=*it;
            if(a.hot_.at(x.slot)!=x.in||a.offsets_.at(adaptive_index(x.out))!=x.offset||
               a.offsets_.at(adaptive_index(x.in))!=core::detail::kNoCacheComplement)
                throw std::logic_error("rewind journal ownership mismatch");
            if(a.sources_[adaptive_index(x.out)]!=(x.rotate?x.saved:x.source)||a.sources_[adaptive_index(x.in)])
                throw std::logic_error("rewind source owner mismatch");
            if(x.rotate){
                auto& spare=a.spares_[adaptive_group(x.in.first)].at(x.spare);
                if(spare!=x.source)throw std::logic_error("rewind spare owner mismatch");
                gpu_check(cudaMemcpyAsync(x.source,a.cache_.device_slot(x.slot),x.bytes,cudaMemcpyDeviceToHost,a.copies_.value));
                gpu_check(cudaMemcpyAsync(a.cache_.device_slot(x.slot),x.saved,x.bytes,cudaMemcpyHostToDevice,a.copies_.value));
                gpu_check(cudaStreamSynchronize(a.copies_.value));spare=x.saved;
            }else{
                // No test scratch may alias a rotated live expert.
                std::vector<uint8_t> incoming(x.bytes);
                gpu_check(cudaMemcpy(incoming.data(),a.cache_.device_slot(x.slot),x.bytes,cudaMemcpyDeviceToHost));
                gpu_check(cudaMemcpyAsync(a.cache_.device_slot(x.slot),x.source,x.bytes,cudaMemcpyHostToDevice,a.copies_.value));
                gpu_check(cudaStreamSynchronize(a.copies_.value));std::memcpy(x.source,incoming.data(),x.bytes);
            }
            a.sources_[adaptive_index(x.in)]=x.source;a.sources_[adaptive_index(x.out)]=nullptr;
            if(!a.cache_.replace_checked(x.in.first,x.in.second,x.out.first,x.out.second,x.slot,x.bytes))
                throw std::logic_error("rewind slot publication rejected");
            a.hot_[x.slot]=x.out;++a.generations_[x.slot];
            a.offsets_[adaptive_index(x.out)]=core::detail::kNoCacheComplement;
            a.offsets_[adaptive_index(x.in)]=x.offset;
        }
        a.committed_exchanges_.clear();
        if(a.hot_!=s.hot||a.offsets_!=s.offsets||a.sources_!=s.sources||a.spares_!=s.spares)throw std::logic_error("rewind did not restore exact slot/RAM ownership");
        a.heat_=s.heat;a.previous_=s.previous;a.learned_=s.learned;
        a.rows_=s.rows;a.since_pick_=s.since;a.batches_=s.batches;a.published_=s.published;a.host_copy_ms_=s.host_ms;
        a.grouped_=grouped;a.windows_=s.windows;a.direct_bytes_=s.direct;a.rotated_=s.rotated;
        a.avoided_prepare_=s.prepare;a.avoided_commit_=s.commit;
        if(std::any_of(a.overrides_.begin(),a.overrides_.end(),[](auto p){return p!=nullptr;})||
           std::any_of(a.unavailable_.begin(),a.unavailable_.end(),[](auto n){return n!=0;}))
            throw std::logic_error("rewind left transient owners");
    }
    static void reset(Runtime& runtime,const Snapshot& initial,bool grouped,bool aligned=true){
        auto& r=rt(runtime);ExecutionStats unused;
        if(r.expert_pipeline)r.expert_pipeline->finish();
        restore(cache(runtime),initial,r.stream.value,unused,grouped);
        if(r.decode_raw_cache)r.decode_raw_cache->clear(r.upload.value);
        gpu_check(cudaDeviceSynchronize());
        for(auto& counts:r.experts->route_counts_)std::fill(counts.begin(),counts.end(),0);
        r.experts->resident_hits_.fill(0);
        std::fill(r.experts->physical_counts_.begin(),r.experts->physical_counts_.end(),0);
        cache(runtime).aligned_=aligned;
        r.options.adaptive_groups=grouped;
    }
    static void state(AdaptiveCache& a,std::ostream& out){
        out<<std::setprecision(17)<<"{\"hot\":[";
        for(size_t i=0;i<a.hot_.size();++i){if(i)out<<',';out<<'['<<a.hot_[i].first<<','<<a.hot_[i].second<<']';}out<<']';
        auto array=[&](const char* name,const auto& values){out<<",\""<<name<<"\":[";for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<values[i];}out<<']';};
        array("ram_offsets",a.offsets_);array("heat",a.heat_);array("previous",a.previous_);array("learned",a.learned_);
        out<<",\"rows\":"<<a.rows_<<",\"since_pick\":"<<a.since_pick_<<",\"pending\":"<<a.swaps_.size()<<"}\n";
    }
    static void verify_weight_samples(Runtime& runtime,std::ostream& out,const std::string& label){
        auto& r=rt(runtime);auto& a=cache(runtime);
        if(a.state_!=AdaptiveCache::State::idle)throw std::logic_error("weight sampling requires drained exchanges");
        gpu_check(cudaDeviceSynchronize());
        constexpr size_t sample=64,per_expert=9*sample,total=40*384*per_expert;
        if(total>a.stride_)throw std::logic_error("weight sample staging capacity");
        std::vector<uint8_t> expected(total),observed(total);auto* actual=observed.data();
        for(int l=0;l<40;++l)for(int e=0;e<384;++e){
            const auto original=r.model.expert(l,e);const auto& f=r.experts->format(l);
            const uint8_t* sources[]={original.gate,original.up,original.down};
            const uint64_t starts[]={0,f.up_off,f.down_off},sizes[]={f.up_off,f.up_off,f.bytes-f.down_off};
            const int key=l*384+e,slot=a.cache_.slot_of(l,e);
            if(slot<0&&a.offsets_.at(key)==core::detail::kNoCacheComplement)
                throw std::logic_error("weight sampling found missing RAM owner");
            const auto* current=slot>=0?a.cache_.device_slot(slot):a.cpu(l,e);
            for(int matrix=0;matrix<3;++matrix){
                const uint64_t offsets[]={0,(sizes[matrix]/2/256)*256,sizes[matrix]-sample};
                for(int part=0;part<3;++part){
                    const size_t index=key*per_expert+(matrix*3+part)*sample;
                    std::memcpy(expected.data()+index,sources[matrix]+offsets[part],sample);
                    const auto* from=current+starts[matrix]+offsets[part];
                    if(slot>=0)gpu_check(cudaMemcpyAsync(actual+index,from,sample,cudaMemcpyDeviceToHost,a.copies_.value));
                    else std::memcpy(actual+index,from,sample);
                }
            }
        }
        gpu_check(cudaStreamSynchronize(a.copies_.value));
        size_t mismatch=0;while(mismatch<total&&expected[mismatch]==actual[mismatch])++mismatch;
        out<<"{\"kind\":\"weight_samples\",\"label\":\""<<label<<"\",\"experts\":15360,\"sampled_bytes\":"<<total
           <<",\"exact\":"<<(mismatch==total?"true":"false")<<",\"first_mismatch_expert\":"<<(mismatch==total?-1:int(mismatch/per_expert))<<"}\n";out.flush();
        if(mismatch!=total)throw std::runtime_error("GPU/RAM expert sample differs from source GGUF");
    }
    static void startup_heat(AdaptiveCache& a,const std::string& path,bool save){
        if(a.state_!=AdaptiveCache::State::idle)throw std::logic_error("startup heat requires idle cache");
        if(save){std::ofstream out(path);out<<std::setprecision(17);for(const auto& values:{a.heat_,a.learned_})for(double v:values)out<<v<<'\n';if(!out)throw std::runtime_error("startup heat write");}
        else{std::ifstream in(path);for(auto* values:{&a.heat_,&a.learned_})for(double& v:*values)if(!(in>>v)||!std::isfinite(v)||v<0)throw std::runtime_error("startup heat read");std::string extra;if(in>>extra)throw std::runtime_error("startup heat trailing data");}
    }
    static void learning_boundary(AdaptiveCache& a){
        if(a.state_!=AdaptiveCache::State::idle)throw std::logic_error("learning boundary requires drained cache");
        std::fill(a.previous_.begin(),a.previous_.end(),0);
        a.rows_=a.windows_=a.since_pick_=a.batches_=a.published_=0;
        a.direct_bytes_=a.rotated_=a.avoided_prepare_=a.avoided_commit_=0;a.host_copy_ms_=0;
    }
    static void aligned(AdaptiveCache& a,bool value){
        if(a.state_!=AdaptiveCache::State::idle||!a.committed_exchanges_.empty())throw std::logic_error("test alignment requires restored cache");
        a.aligned_=value;
    }
    static bool stale_guard(AdaptiveCache& a){
        if(a.swaps_.empty())return false;
        auto& generation=a.generations_[a.swaps_.front().slot];++generation;
        bool rejected=false;try{a.validate(a.swaps_.front());}catch(const std::logic_error&){rejected=true;}
        --generation;return rejected;
    }
};
} // namespace strata::models::deepseek41
