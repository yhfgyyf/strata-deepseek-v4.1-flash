#pragma once
#include "runtime_internal.hpp"

namespace strata::models::deepseek41 {
// Decode readers and retain copies use one ordered stream. The first eight
// ring slots remain issuer scratch; the remaining slots hold immutable blobs.
class DecodeRawCache {
public:
    DecodeRawCache(void* data,uint64_t stride,int slots,cudaStream_t stream);
    const uint8_t* find(int key);
    const uint8_t* retain(int key,const uint8_t* source,uint64_t bytes);
    void clear(cudaStream_t overwrite);
    int slots()const{return keys_.size();}
private:
    uint8_t* data_;uint64_t stride_,clock_=0;
    cudaStream_t stream_;Event consumed_;
    std::vector<int> keys_;std::vector<uint64_t> ages_;
};

// One sequential issuer, bounded by returned raw slots. Every event is paired
// with a host-published (generation, sequence), including the initial generation.
class ExpertPipeline {
public:
    struct Upload {const uint8_t* source;uint64_t bytes;int trace_layer=-1,trace_expert=-1;};
    ExpertPipeline(void* device,Pinned& host,uint64_t stride,int slots,cudaStream_t upload);
    ~ExpertPipeline();
    void start(uint64_t generation,size_t jobs,uint64_t bytes,std::function<void(size_t,uint8_t*)> fill,size_t active_slots = 0);
    // Sources must be pinned and immutable through finish(). Variable sizes
    // permit one bounded ring to stream experts from several layers.
    void start(uint64_t generation,std::vector<Upload> uploads,size_t active_slots = 0);
    // Discard only direct jobs not yet claimed by the issuer. Returned indices
    // never acquire a raw slot; claimed/in-flight copies retain their lease.
    std::vector<size_t> discard(const std::vector<size_t>& sequences);
    const uint8_t* wait(size_t sequence,cudaStream_t reader);
    void release(size_t sequence,cudaStream_t last_reader);
    // No GPU consumer may read this job. Keep copy-ready/reuse fences on the
    // upload stream so retiring unused speculation cannot stall compute.
    void retire_unused(size_t sequence);
    void finish();
    // Called by the execution owner after abandoning this generation. Drain
    // every reader before reusing slots; retain the issuer thread and affinity.
    void cancel(cudaStream_t dequant,cudaStream_t reader);
    void stop();
private:
    struct Slot { Event copied,used;uint64_t generation=0;size_t sequence=0;bool issued=false,returned=false; };
    uint8_t* device_;
    Pinned& host_;
    uint64_t stride_,generation_=0;
    // A short verification window can stream more than eight cold experts
    // through eight scratch slots, preserving the raw cache in the ring tail.
    size_t jobs_=0,active_slots_=0;
    cudaStream_t upload_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::thread worker_;
    std::function<void(size_t,uint8_t*)> fill_;
    uint64_t bytes_=0;
    std::vector<Upload> uploads_;
    size_t claimed_=0,last_uploaded_=0;
    bool have_uploaded_=false;
    int device_id_=0;
    std::exception_ptr error_;
    bool stopped_=false,pending_=false,working_=false,cancelling_=false;
    struct Diagnostics {
        bool enabled=false,reported=false;
        size_t owned=0,peak=0,submitted=0,returned=0,copy_pending=0,discarded=0;
        int64_t begin=0;
        std::vector<std::pair<int64_t,int64_t>> producer_waits,consumer_waits;
    } diagnostics_;
    bool diagnostics_requested_=false;
    void begin_diagnostics();
    void report_diagnostics();
    void run();
};
} // namespace strata::models::deepseek41
