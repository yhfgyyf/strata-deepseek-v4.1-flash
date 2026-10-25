#include "expert_pipeline.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

namespace d=strata::models::deepseek41;
namespace {
void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
__global__ void check_raw(const uint8_t* raw,uint8_t expected,int bytes,int* failures) {
    for(int i=threadIdx.x;i<bytes;i+=blockDim.x)if(raw[i]!=expected)atomicAdd(failures,1);
}
__global__ void delay_reader(unsigned long long cycles) {
    const auto begin=clock64();while(clock64()-begin<cycles){}
}
void cancel_request(d::ExpertPipeline& pipeline,cudaStream_t dequant,cudaStream_t reader) {
#ifdef DS41_TEST_LEGACY_CANCEL
    // The deployed C37 cleanup used terminal stop followed by these fences.
    pipeline.stop();d::gpu_check(cudaStreamSynchronize(dequant));d::gpu_check(cudaStreamSynchronize(reader));
#else
    pipeline.cancel(dequant,reader);
#endif
}
void cancelled_generation_reuse() {
    constexpr int slots=4,stride=4096,payload=4000,jobs=17;
    strata::core::DeviceArena arena(slots*stride+256);auto* raw=static_cast<uint8_t*>(arena.alloc(slots*stride));
    auto* failures=static_cast<int*>(arena.alloc(256));d::Pinned scratch(slots*stride),source(jobs*stride);
    d::Stream upload,dequant,reader;d::ExpertPipeline pipeline(raw,scratch,stride,slots,upload.value);
    d::gpu_check(cudaMemset(raw,0xa5,slots*stride));d::gpu_check(cudaMemset(failures,0,256));d::gpu_check(cudaStreamSynchronize(nullptr));
    uint64_t generation=0;std::thread::id issuer;bool same_issuer=true;
    auto fill=[&](uint64_t g,size_t i,uint8_t* target) {
        const auto current=std::this_thread::get_id();
        if(issuer==std::thread::id{})issuer=current;else same_issuer&=issuer==current;
        std::memset(target,int(g*13+i),payload);
    };
    for(int direct=0;direct<2;++direct)for(int boundary=0;boundary<4;++boundary) {
        const auto cancelled=++generation;
        if(direct){
            std::vector<d::ExpertPipeline::Upload> plan;
            for(int i=0;i<jobs;++i){std::memset(source.data+i*stride,int(cancelled*13+i),payload);plan.push_back({source.data+i*stride,payload});}
            pipeline.start(cancelled,std::move(plan));
        }else pipeline.start(cancelled,jobs,payload,[&,cancelled](size_t i,uint8_t* target){fill(cancelled,i,target);});
        if(boundary==1) {
            pipeline.wait(slots-1,reader.value); // All raw slots issued, none returned.
        }else if(boundary==2) {
            const auto* first=pipeline.wait(0,dequant.value);
            delay_reader<<<1,1,0,dequant.value>>>(500000);
            check_raw<<<1,256,0,dequant.value>>>(first,uint8_t(cancelled*13),payload,failures);
            pipeline.wait(0,reader.value);
            delay_reader<<<1,1,0,reader.value>>>(500000);
            check_raw<<<1,256,0,reader.value>>>(first,uint8_t(cancelled*13),payload,failures);
            d::gpu_check(cudaGetLastError()); // Cancel with both readers in flight and slot unreturned.
        }else if(boundary==3) {
            for(int i=0;i<jobs;++i){pipeline.wait(i,reader.value);pipeline.release(i,reader.value);}
            pipeline.finish(); // Also cover idle cancellation after a completed batch.
        }
        cancel_request(pipeline,dequant.value,reader.value);
        std::memset(source.data,0xee,source.bytes); // Cancel must release all direct-source DMA leases.
        const auto next=++generation;
        pipeline.start(next,jobs,payload,[&,next](size_t i,uint8_t* target){fill(next,i,target);});
        for(int i=0;i<jobs;++i){
            const auto* data=pipeline.wait(i,reader.value);
            check_raw<<<1,256,0,reader.value>>>(data,uint8_t(next*13+i),payload,failures);
            d::gpu_check(cudaGetLastError());pipeline.release(i,reader.value);
        }
        pipeline.finish();d::gpu_check(cudaStreamSynchronize(reader.value));
        need(same_issuer,"cancellation replaced the issuer thread");
    }
    int bad=0;d::gpu_check(cudaMemcpy(&bad,failures,4,cudaMemcpyDeviceToHost));need(!bad,"cancel/reuse payload mismatch");
    std::vector<uint8_t> guard(slots*stride);d::gpu_check(cudaMemcpy(guard.data(),raw,guard.size(),cudaMemcpyDeviceToHost));
    for(int s=0;s<slots;++s)for(int i=payload;i<stride;++i)need(guard[s*stride+i]==0xa5,"cancel/reuse raw guard overwritten");
    std::cout<<"CANCEL_REUSE_8_BOUNDARIES_SAME_ISSUER_EXACT_BYTES_GUARDS_PASS\n";
}

void run(int producer_delay,int consumer_delay) {
    constexpr int slots=4,stride=4096,payload=4000,jobs=17; // > 4 wraps, non-full tail.
    strata::core::DeviceArena arena(slots*stride+256);auto* raw=static_cast<uint8_t*>(arena.alloc(slots*stride));
    auto* failures=static_cast<int*>(arena.alloc(256));d::gpu_check(cudaMemset(raw,0xa5,slots*stride));d::gpu_check(cudaMemset(failures,0,256));
    d::gpu_check(cudaStreamSynchronize(nullptr)); // Initialize before either nonblocking stream.
    d::Pinned host(slots*stride);d::Stream upload,reader;
    d::ExpertPipeline pipeline(raw,host,stride,slots,upload.value);
    for(uint64_t generation=1;generation<=3;++generation) {
        pipeline.start(generation,jobs,payload,[=](size_t i,uint8_t* target) {
            if(producer_delay && i%3==0)std::this_thread::sleep_for(std::chrono::microseconds(producer_delay));
            std::memset(target,int(generation*31+i),payload);
        });
        for(int i=0;i<jobs;++i) {
            if(consumer_delay && i%2==0)std::this_thread::sleep_for(std::chrono::microseconds(consumer_delay));
            const auto* blob=pipeline.wait(i,reader.value);
            check_raw<<<1,256,0,reader.value>>>(blob,uint8_t(generation*31+i),payload,failures);
            d::gpu_check(cudaGetLastError());
            // A second reader models a direct packed down projection. Raw must
            // not be retired at the earlier gate/up read.
            check_raw<<<1,256,0,reader.value>>>(blob,uint8_t(generation*31+i),payload,failures);
            pipeline.release(i,reader.value);
        }
        pipeline.finish();d::gpu_check(cudaStreamSynchronize(reader.value));
    }
    int bad=0;d::gpu_check(cudaMemcpy(&bad,failures,4,cudaMemcpyDeviceToHost));need(!bad,"ring generation/raw reader corrupted");
    std::vector<uint8_t> contents(slots*stride);d::gpu_check(cudaMemcpy(contents.data(),raw,contents.size(),cudaMemcpyDeviceToHost));
    for(int slot=0;slot<slots;++slot)for(int i=payload;i<stride;++i)need(contents[slot*stride+i]==0xa5,"raw guard overwritten");
}
void failure_and_cancel() {
    strata::core::DeviceArena arena(4*4096);auto* raw=arena.alloc(4*4096);d::Pinned host(4*4096);d::Stream upload,reader;
    {
        d::ExpertPipeline pipeline(raw,host,4096,4,upload.value);
        pipeline.start(1,20,4000,[](size_t i,uint8_t* target){if(i==2)throw std::runtime_error("injected producer failure");std::memset(target,1,4000);});
        bool failed=false;
        try {for(int i=0;i<20;++i){pipeline.wait(i,reader.value);pipeline.release(i,reader.value);}pipeline.finish();}
        catch(const std::runtime_error&){failed=true;}
        pipeline.stop();need(failed,"producer error failed to wake consumer");
    }
    {
        d::ExpertPipeline pipeline(raw,host,4096,4,upload.value);
        pipeline.start(2,20,4000,[](size_t,uint8_t* target){std::memset(target,2,4000);});
        pipeline.wait(0,reader.value); // Cancellation before returning the first slot.
        pipeline.stop();d::gpu_check(cudaStreamSynchronize(reader.value));
    }
}
void raw_cache_reuse_and_prefill_fence(){
    constexpr int stride=4096,payload=4000;
    strata::core::DeviceArena arena(12*stride+256);auto* raw=static_cast<uint8_t*>(arena.alloc(12*stride));
    auto* failures=static_cast<int*>(arena.alloc(256));d::gpu_check(cudaMemset(raw,0xa5,12*stride));d::gpu_check(cudaMemset(failures,0,256));
    d::gpu_check(cudaStreamSynchronize(nullptr)); // Initialize before either nonblocking stream.
    d::Stream reader,overwrite;d::DecodeRawCache cache(raw+8*stride,stride,4,reader.value);
    auto put=[&](int key,uint8_t value){
        d::gpu_check(cudaMemsetAsync(raw,value,payload,reader.value));
        const auto* held=cache.retain(key,raw,payload);
        check_raw<<<1,256,0,reader.value>>>(held,value,payload,failures);
        d::gpu_check(cudaMemsetAsync(raw,0xee,payload,reader.value));
        check_raw<<<1,256,0,reader.value>>>(cache.find(key),value,payload,failures);
    };
    for(int key=0;key<4;++key)put(key,uint8_t(key+1));
    need(cache.find(0)!=nullptr,"raw cache hit missing");put(384,17);
    need(cache.find(1)==nullptr && cache.find(0)!=nullptr && cache.find(384)!=nullptr,"raw cache LRU or layer identity");
    for(int key=385;key<410;++key)put(key,uint8_t(key));
    d::gpu_check(cudaStreamSynchronize(reader.value));
    std::vector<uint8_t> bytes(12*stride);d::gpu_check(cudaMemcpy(bytes.data(),raw,bytes.size(),cudaMemcpyDeviceToHost));
    for(int slot=1;slot<8;++slot)for(int i=0;i<stride;++i)need(bytes[slot*stride+i]==0xa5,"raw cache overlapped issuer scratch");
    for(int slot=8;slot<12;++slot)for(int i=payload;i<stride;++i)need(bytes[slot*stride+i]==0xa5,"raw cache tail guard");
    // The prefill writer is another stream. clear must order it after the
    // outstanding decode reader before permitting whole-ring reuse.
    check_raw<<<1,256,0,reader.value>>>(cache.find(409),uint8_t(409),payload,failures);
    cache.clear(overwrite.value);need(!cache.find(409),"raw cache survived prefill invalidation");
    d::gpu_check(cudaMemsetAsync(raw,0,12*stride,overwrite.value));d::gpu_check(cudaStreamSynchronize(overwrite.value));
    put(409,42);d::gpu_check(cudaStreamSynchronize(reader.value));
    int bad=0;d::gpu_check(cudaMemcpy(&bad,failures,4,cudaMemcpyDeviceToHost));need(!bad,"raw cache bytes/lifetime corrupted");
}
void verification_cache_reserved_slots(){
    constexpr int slots=12,stride=4096,payload=4000;
    strata::core::DeviceArena arena(slots*stride+256);auto* raw=static_cast<uint8_t*>(arena.alloc(slots*stride));
    auto* failures=static_cast<int*>(arena.alloc(256));d::Pinned source(73*stride),scratch(slots*stride);
    d::Stream upload,reader;d::ExpertPipeline pipeline(raw,scratch,stride,slots,upload.value);
    d::DecodeRawCache cache(raw+8*stride,stride,slots-8,reader.value);
    d::gpu_check(cudaMemset(raw,0xa5,slots*stride));d::gpu_check(cudaMemset(failures,0,256));
    d::gpu_check(cudaStreamSynchronize(nullptr)); // Initialize before either nonblocking stream.
    uint64_t generation=0;
    for(bool direct:{false,true})for(int tokens:{1,2,5,6,8,0,8,1}){
        ++generation;const bool full_ring=tokens==0;const int jobs=full_ring?73:tokens*6;
        std::array<const uint8_t*,4> held{};
        if(full_ring)cache.clear(upload.value);
        else {
            cache.clear(upload.value);
            for(int i=0;i<4;++i){
                d::gpu_check(cudaMemsetAsync(raw,17+i,payload,reader.value));
                held[i]=cache.retain(900+i,raw,payload);
            }
            d::gpu_check(cudaStreamSynchronize(reader.value));
        }
        const size_t span=full_ring?0:8;
        if(direct){
            std::vector<d::ExpertPipeline::Upload> plan;
            for(int i=0;i<jobs;++i){std::memset(source.data+i*stride,int(generation*31+i),payload);plan.push_back({source.data+i*stride,payload});}
            pipeline.start(generation,std::move(plan),span);
        }else pipeline.start(generation,jobs,payload,[=](size_t i,uint8_t* target){std::memset(target,int(generation*31+i),payload);},span);
        // Fill the first scratch cycle before checking the retained tail.
        pipeline.wait(std::min(jobs,full_ring?slots:8)-1,reader.value);
        if(!full_ring)for(int i=0;i<4;++i)check_raw<<<1,256,0,reader.value>>>(held[i],17+i,payload,failures);
        for(int i=0;i<jobs;++i){
            const auto* blob=pipeline.wait(i,reader.value);
            need(blob==raw+(i%(full_ring?slots:8))*stride,"issuer escaped active slot span");
            if(!full_ring)blob=cache.retain(i,blob,payload);
            check_raw<<<1,256,0,reader.value>>>(blob,uint8_t(generation*31+i),payload,failures);
            delay_reader<<<1,1,0,reader.value>>>(1000);
            check_raw<<<1,256,0,reader.value>>>(blob,uint8_t(generation*31+i),payload,failures);
            pipeline.release(i,reader.value);
        }
        pipeline.finish();std::memset(source.data,0x55,source.bytes);
        d::gpu_check(cudaStreamSynchronize(reader.value));
        bool rejected=false;try{pipeline.release(jobs-1,reader.value);}catch(const std::runtime_error&){rejected=true;}
        need(rejected,"verification duplicate return accepted");
    }
    for(size_t span:{size_t(1),size_t(slots+1)}){
        bool rejected=false;try{pipeline.start(generation+1,1,payload,[](size_t,uint8_t*){},span);}catch(const std::invalid_argument&){rejected=true;}
        need(rejected,"invalid active slot capacity accepted");
    }
    int bad=0;d::gpu_check(cudaMemcpy(&bad,failures,4,cudaMemcpyDeviceToHost));need(!bad,"verification cache/raw lifetime corrupted");
    std::vector<uint8_t> bytes(slots*stride);d::gpu_check(cudaMemcpy(bytes.data(),raw,bytes.size(),cudaMemcpyDeviceToHost));
    for(int slot=0;slot<slots;++slot)for(int i=payload;i<stride;++i)need(bytes[slot*stride+i]==0xa5,"verification raw tail guard");
    std::cout<<"VERIFY_1_2_5_6_8_TOKENS_CACHE_WRAP_FULL_RING_TRANSITION_PASS\n";
}
void direct_variable_sources(){
    constexpr int slots=128,stride=4096,jobs=307;
    strata::core::DeviceArena arena(slots*stride+256);auto* raw=static_cast<uint8_t*>(arena.alloc(slots*stride));
    auto* failures=static_cast<int*>(arena.alloc(256));d::Pinned source(jobs*stride),scratch(slots*stride);
    need(source.pinned && scratch.pinned,"direct source test needs pinned memory");
    d::Stream upload,reader;d::ExpertPipeline pipeline(raw,scratch,stride,slots,upload.value);
    d::gpu_check(cudaMemset(raw,0xa5,slots*stride));d::gpu_check(cudaMemset(failures,0,256));
    d::gpu_check(cudaStreamSynchronize(nullptr)); // Initialize before either nonblocking stream.
    for(uint64_t generation=1;generation<=3;++generation){
        std::vector<d::ExpertPipeline::Upload> plan;
        for(int i=0;i<jobs;++i){
            const int bytes=i%2?2816:4000;std::memset(source.data+i*stride,int(generation*31+i),bytes);
            plan.push_back({source.data+i*stride,uint64_t(bytes)});
        }
        pipeline.start(generation,std::move(plan));
        for(int i=0;i<jobs;++i){
            const auto* blob=pipeline.wait(i,reader.value);
            // Non-routed jobs are retired without arithmetic, like cross-layer prefill.
            if(i%5)check_raw<<<1,256,0,reader.value>>>(blob,uint8_t(generation*31+i),i%2?2816:4000,failures);
            pipeline.release(i,reader.value);
        }
        pipeline.finish();
        // finish is the source lease boundary, including before GPU consumers drain.
        std::memset(source.data,0x55,source.bytes);
        d::gpu_check(cudaStreamSynchronize(reader.value));
    }
    int bad=0;d::gpu_check(cudaMemcpy(&bad,failures,4,cudaMemcpyDeviceToHost));need(!bad,"direct variable source bytes/lease");
    std::vector<uint8_t> bytes(slots*stride);d::gpu_check(cudaMemcpy(bytes.data(),raw,bytes.size(),cudaMemcpyDeviceToHost));
    for(int slot=0;slot<slots;++slot)for(int i=4000;i<stride;++i)need(bytes[slot*stride+i]==0xa5,"direct raw tail guard");
    bool invalid=false;
    try{pipeline.start(4,std::vector<d::ExpertPipeline::Upload>{{source.data,stride+1}});}catch(const std::invalid_argument&){invalid=true;}
    need(invalid,"oversized direct source accepted");
    pipeline.start(4,std::vector<d::ExpertPipeline::Upload>(jobs,{source.data,4000}));
    pipeline.wait(0,reader.value);pipeline.stop(); // Cancellation with a bounded producer blocked on reuse.
    d::gpu_check(cudaStreamSynchronize(reader.value));
}
void real_direct_sources(const char* path){
    d::Model model(path);
    const auto sizes=model.expert_bytes();const auto stride=*std::max_element(sizes.begin(),sizes.end());
    strata::core::DeviceArena arena(2*stride);auto* raw=static_cast<uint8_t*>(arena.alloc(2*stride));
    d::Pinned scratch(2*stride);d::Stream upload,reader;d::ExpertPipeline pipeline(raw,scratch,stride,2,upload.value);
    uint64_t generation=0;
    for(int layer:{0,13,14,39}){
        const auto expert=model.expert(layer,17);
        std::vector<uint8_t> source(expert.blob_bytes),readback(expert.blob_bytes);
        std::memcpy(source.data(),expert.gate,expert.projection_bytes);
        std::memcpy(source.data()+expert.projection_bytes,expert.up,expert.projection_bytes);
        std::memcpy(source.data()+2*expert.projection_bytes,expert.down,expert.projection_bytes);
        d::HostRegistration registration(source.data(),source.size());
        pipeline.start(++generation,std::vector<d::ExpertPipeline::Upload>{{source.data(),source.size()}});
        pipeline.wait(0,reader.value);cancel_request(pipeline,reader.value,reader.value);
        pipeline.start(++generation,std::vector<d::ExpertPipeline::Upload>{{source.data(),source.size()}});
        const auto* held=pipeline.wait(0,reader.value);
        d::gpu_check(cudaMemcpyAsync(readback.data(),held,readback.size(),cudaMemcpyDeviceToHost,reader.value));
        pipeline.release(0,reader.value);pipeline.finish();d::gpu_check(cudaStreamSynchronize(reader.value));
        need(readback==source,"actual GGUF G/U/D direct transfer mismatch");
        std::cout<<"REAL_DIRECT_BYTES layer="<<layer<<" type="<<expert.type<<" bytes="<<source.size()<<" mismatches=0\n";
    }
}
void discard_unissued_direct(){
    constexpr int slots=4,stride=4096,payload=4000,jobs=17;
    strata::core::DeviceArena arena(slots*stride+256);auto* raw=static_cast<uint8_t*>(arena.alloc(slots*stride));
    auto* failures=static_cast<int*>(arena.alloc(256));d::Pinned source(jobs*stride),scratch(slots*stride);
    d::Stream upload,reader;d::ExpertPipeline pipeline(raw,scratch,stride,slots,upload.value);
    d::gpu_check(cudaMemset(raw,0xa5,slots*stride));d::gpu_check(cudaMemset(failures,0,256));
    d::gpu_check(cudaStreamSynchronize(nullptr)); // Initialize before either nonblocking stream.
    auto plan=[&](int generation){
        std::vector<d::ExpertPipeline::Upload> uploads;
        for(int i=0;i<jobs;++i){
            std::memset(source.data+i*stride,generation*31+i,payload);
            uploads.push_back({source.data+i*stride,payload});
        }
        return uploads;
    };
    auto consume=[&](int sequence,int generation){
        const auto* blob=pipeline.wait(sequence,reader.value);
        check_raw<<<1,256,0,reader.value>>>(blob,uint8_t(generation*31+sequence),payload,failures);
        d::gpu_check(cudaGetLastError());pipeline.release(sequence,reader.value);
    };
    pipeline.start(1,plan(1));pipeline.wait(slots-1,reader.value);
    // All physical slots are owned. The issuer cannot claim sequence4 yet;
    // discard must wake it even while the next raw slot is still occupied.
    bool invalid=false;
    try{pipeline.discard({8,jobs});}catch(const std::out_of_range&){invalid=true;}
    need(invalid,"out-of-range discard accepted");
    std::vector<size_t> unwanted{0}; // Already claimed; must remain consumable.
    for(int i=slots;i<jobs;++i)if(i!=8)unwanted.push_back(i);
    const auto discarded=pipeline.discard(unwanted);
    need(discarded.size()==12 && std::find(discarded.begin(),discarded.end(),0)==discarded.end(),"discard claimed/pending boundary");
    need(pipeline.discard(unwanted).empty(),"discard not idempotent");
    invalid=false;try{pipeline.wait(4,reader.value);}catch(const std::logic_error&){invalid=true;}
    need(invalid,"discarded wait did not fail immediately");
    for(int i=0;i<slots;++i)consume(i,1);
    consume(8,1);pipeline.finish(); // Final planned job16 was discarded.
    std::memset(source.data,0x55,source.bytes);d::gpu_check(cudaStreamSynchronize(reader.value));
    int empty_generations=0;
    for(int generation=2;generation<=9;++generation){
        pipeline.start(generation,plan(generation));
        std::vector<size_t> all(jobs);for(int i=0;i<jobs;++i)all[i]=i;
        const auto dropped=pipeline.discard(all);
        std::vector<bool> skipped(jobs);for(size_t i:dropped)skipped[i]=true;
        if(dropped.size()==jobs)++empty_generations;
        for(int i=0;i<jobs;++i)if(!skipped[i])consume(i,generation);
        pipeline.finish();std::memset(source.data,0x55,source.bytes);
        d::gpu_check(cudaStreamSynchronize(reader.value));
    }
    int bad=0;d::gpu_check(cudaMemcpy(&bad,failures,4,cudaMemcpyDeviceToHost));need(!bad,"discard source lease/kept bytes");
    std::vector<uint8_t> guards(slots*stride);d::gpu_check(cudaMemcpy(guards.data(),raw,guards.size(),cudaMemcpyDeviceToHost));
    for(int slot=0;slot<slots;++slot)for(int i=payload;i<stride;++i)need(guards[slot*stride+i]==0xa5,"discard raw guard overwritten");
    pipeline.start(10,plan(10));pipeline.wait(0,reader.value);
    std::vector<size_t> remaining;for(int i=1;i<jobs;++i)remaining.push_back(i);
    pipeline.discard(remaining);pipeline.stop();d::gpu_check(cudaStreamSynchronize(reader.value));
    std::cout<<"DISCARD_PENDING_TAIL_REUSE_RAW_BYTES_PASS empty_generations="<<empty_generations<<'\n';
}
void unused_reclaim_without_compute_dependency(){
    constexpr int slots=4,stride=4096,payload=4000,jobs=17;
    strata::core::DeviceArena arena(slots*stride+256);auto* raw=static_cast<uint8_t*>(arena.alloc(slots*stride));
    auto* failures=static_cast<int*>(arena.alloc(256));d::Pinned source(jobs*stride),scratch(slots*stride);
    d::Stream upload,reader;d::Event delayed;
    d::ExpertPipeline pipeline(raw,scratch,stride,slots,upload.value);
    d::gpu_check(cudaMemset(raw,0xa5,slots*stride));d::gpu_check(cudaMemset(failures,0,256));
    d::gpu_check(cudaStreamSynchronize(nullptr)); // Initialize before either nonblocking stream.
    // Warm the test kernel before the measured dependency check, including its
    // module load. A delayed unrelated compute stream must not hold raw slots.
    delay_reader<<<1,1,0,reader.value>>>(1);d::gpu_check(cudaGetLastError());d::gpu_check(cudaStreamSynchronize(reader.value));
    auto plan=[&](int generation){
        std::vector<d::ExpertPipeline::Upload> uploads;
        for(int i=0;i<jobs;++i){std::memset(source.data+i*stride,generation*31+i,payload);uploads.push_back({source.data+i*stride,payload});}
        return uploads;
    };
    delay_reader<<<1,1,0,reader.value>>>(1000000000ull);d::gpu_check(cudaGetLastError());
    d::gpu_check(cudaEventRecord(delayed.value,reader.value));
    pipeline.start(1,plan(1));
    for(int i=0;i<jobs;++i)pipeline.retire_unused(i);
    pipeline.finish();std::memset(source.data,0x55,source.bytes);
    const auto pending=cudaEventQuery(delayed.value);
    need(pending==cudaErrorNotReady,"unused retirement waited for unrelated compute");
    d::gpu_check(cudaStreamSynchronize(reader.value));
    bool rejected=false;try{pipeline.retire_unused(jobs);}catch(const std::out_of_range&){rejected=true;}
    need(rejected,"unused out-of-range accepted");
    rejected=false;try{pipeline.retire_unused(jobs-1);}catch(const std::runtime_error&){rejected=true;}
    need(rejected,"unused duplicate accepted");
    rejected=false;try{pipeline.retire_unused(0);}catch(const std::runtime_error&){rejected=true;}
    need(rejected,"unused stale wrap accepted");
    for(int generation=2;generation<=4;++generation){
        pipeline.start(generation,plan(generation));
        for(int i=0;i<jobs;++i){
            if(i%3==0)pipeline.retire_unused(i);
            else{
                const auto* blob=pipeline.wait(i,reader.value);
                check_raw<<<1,256,0,reader.value>>>(blob,uint8_t(generation*31+i),payload,failures);
                check_raw<<<1,256,0,reader.value>>>(blob,uint8_t(generation*31+i),payload,failures);
                d::gpu_check(cudaGetLastError());pipeline.release(i,reader.value);
            }
        }
        pipeline.finish();std::memset(source.data,0x55,source.bytes);d::gpu_check(cudaStreamSynchronize(reader.value));
    }
    int bad=0;d::gpu_check(cudaMemcpy(&bad,failures,4,cudaMemcpyDeviceToHost));need(!bad,"mixed unused/selected raw reader lifetime");
    std::vector<uint8_t> guards(slots*stride);d::gpu_check(cudaMemcpy(guards.data(),raw,guards.size(),cudaMemcpyDeviceToHost));
    for(int slot=0;slot<slots;++slot)for(int i=payload;i<stride;++i)need(guards[slot*stride+i]==0xa5,"unused raw tail guard");
    std::cout<<"UNUSED_UPLOAD_STREAM_RECLAIM_DELAYED_READER_WRAP_SOURCE_LEASE_PASS\n";
}
}
int main(int argc,char** argv){try {
    cancelled_generation_reuse();
    run(1000,0);run(0,1000);run(100,300);failure_and_cancel();raw_cache_reuse_and_prefill_fence();verification_cache_reserved_slots();direct_variable_sources();discard_unissued_direct();unused_reclaim_without_compute_dependency();
    if(argc==2)real_direct_sources(argv[1]);
    std::cout<<"EXPERT_ISSUER_GENERATION_WRAP_DELAY_RAW_GUARD_CANCEL_FAILURE_PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
