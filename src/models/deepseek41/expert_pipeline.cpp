#include "expert_pipeline.hpp"
#include <algorithm>
#include <limits>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include "strata/prefill/ledger_scope.hpp"

namespace strata::models::deepseek41 {
namespace {
int64_t ring_now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
}
DecodeRawCache::DecodeRawCache(void* data,uint64_t stride,int slots,cudaStream_t stream)
    :data_(static_cast<uint8_t*>(data)),stride_(stride),stream_(stream),keys_(std::max(0,slots),-1),ages_(std::max(0,slots)) {
    if(!data || !stride || slots<1)throw std::invalid_argument("decode raw cache capacity");
}
const uint8_t* DecodeRawCache::find(int key){
    const auto at=std::find(keys_.begin(),keys_.end(),key);
    if(key<0 || at==keys_.end())return nullptr;
    const size_t slot=at-keys_.begin();ages_[slot]=++clock_;return data_+slot*stride_;
}
const uint8_t* DecodeRawCache::retain(int key,const uint8_t* source,uint64_t bytes){
    if(key<0 || !source || !bytes || bytes>stride_)throw std::invalid_argument("decode raw cache blob");
    if(const auto* present=find(key))return present;
    const size_t slot=std::min_element(ages_.begin(),ages_.end())-ages_.begin();
    gpu_check(cudaMemcpyAsync(data_+slot*stride_,source,bytes,cudaMemcpyDeviceToDevice,stream_));
    keys_[slot]=key;ages_[slot]=++clock_;return data_+slot*stride_;
}
void DecodeRawCache::clear(cudaStream_t overwrite){
    // Before a prefill issuer may overwrite the whole ring, fence every main
    // stream reader/retain, including a cancelled decode generation.
    gpu_check(cudaEventRecord(consumed_.value,stream_));
    gpu_check(cudaStreamWaitEvent(overwrite,consumed_.value,0));
    std::fill(keys_.begin(),keys_.end(),-1);std::fill(ages_.begin(),ages_.end(),0);clock_=0;
}

ExpertPipeline::ExpertPipeline(void* device,Pinned& host,uint64_t stride,int slots,cudaStream_t upload)
    :device_(static_cast<uint8_t*>(device)),host_(host),stride_(stride),upload_(upload) {
    if(!device || !host.pinned || !stride || slots<2 || uint64_t(slots)*stride>host.bytes)
        throw std::invalid_argument("expert issuer ring capacity/pinning");
    active_slots_=slots;
    for(int i=0;i<slots;++i)slots_.push_back(std::make_unique<Slot>());
    diagnostics_requested_=std::getenv("DS41_RING_DIAGNOSTICS")!=nullptr;
    gpu_check(cudaGetDevice(&device_id_));worker_=std::thread([this]{run();});
}
ExpertPipeline::~ExpertPipeline(){stop();}
void ExpertPipeline::start(uint64_t generation,size_t jobs,uint64_t bytes,std::function<void(size_t,uint8_t*)> fill,size_t active_slots) {
    finish();
    std::lock_guard<std::mutex> lock(mutex_);
    if(stopped_ || !generation || generation<=generation_ || !jobs || bytes>stride_)throw std::invalid_argument("expert issuer generation/bytes");
    const size_t span=active_slots?active_slots:slots_.size();
    if(span<2 || span>slots_.size())throw std::invalid_argument("expert issuer active slot capacity");
    for(const auto& slot:slots_)if(slot->issued && !slot->returned)throw std::logic_error("expert issuer unreturned generation");
    active_slots_=span;
    generation_=generation;jobs_=jobs;bytes_=bytes;fill_=std::move(fill);uploads_.clear();claimed_=0;have_uploaded_=false;begin_diagnostics();pending_=true;changed_.notify_all();
}
void ExpertPipeline::start(uint64_t generation,std::vector<Upload> uploads,size_t active_slots) {
    finish();
    for(const auto& job:uploads)if(!job.source || !job.bytes || job.bytes>stride_)throw std::invalid_argument("expert direct upload source/bytes");
    std::lock_guard<std::mutex> lock(mutex_);
    if(stopped_ || !generation || generation<=generation_ || uploads.empty())throw std::invalid_argument("expert direct upload generation/jobs");
    const size_t span=active_slots?active_slots:slots_.size();
    if(span<2 || span>slots_.size())throw std::invalid_argument("expert issuer active slot capacity");
    for(const auto& slot:slots_)if(slot->issued && !slot->returned)throw std::logic_error("expert issuer unreturned generation");
    active_slots_=span;
    generation_=generation;jobs_=uploads.size();bytes_=0;fill_={};uploads_=std::move(uploads);claimed_=0;have_uploaded_=false;begin_diagnostics();pending_=true;changed_.notify_all();
}
std::vector<size_t> ExpertPipeline::discard(const std::vector<size_t>& sequences) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(stopped_ || uploads_.empty())throw std::logic_error("expert discard requires active direct uploads");
    if(error_)std::rethrow_exception(error_);
    for(size_t sequence:sequences)if(sequence>=jobs_)throw std::out_of_range("expert discard sequence");
    std::vector<size_t> discarded;discarded.reserve(sequences.size());
    for(size_t sequence:sequences)if(sequence>=claimed_ && uploads_[sequence].bytes){
        uploads_[sequence].bytes=0;discarded.push_back(sequence);
    }
    if(diagnostics_.enabled)diagnostics_.discarded+=discarded.size();
    changed_.notify_all();return discarded;
}
void ExpertPipeline::begin_diagnostics(){
    diagnostics_={};diagnostics_.enabled=diagnostics_requested_ && jobs_>active_slots_;
    if(diagnostics_.enabled){
        diagnostics_.begin=ring_now();diagnostics_.producer_waits.reserve(jobs_);diagnostics_.consumer_waits.reserve(jobs_);
    }
}
void ExpertPipeline::report_diagnostics(){
    auto& d=diagnostics_;if(!d.enabled || d.reported)return;
    d.reported=true;int64_t producer=0,consumer=0,overlap=0,maximum=0;
    for(const auto& w:d.producer_waits){producer+=w.second-w.first;maximum=std::max(maximum,w.second-w.first);}
    for(const auto& w:d.consumer_waits)consumer+=w.second-w.first;
    size_t i=0,j=0;
    while(i<d.producer_waits.size() && j<d.consumer_waits.size()){
        const auto a=d.producer_waits[i],b=d.consumer_waits[j];
        overlap+=std::max<int64_t>(0,std::min(a.second,b.second)-std::max(a.first,b.first));
        if(a.second<b.second)++i;else ++j;
    }
    std::cerr<<"DS41_RING {\"generation\":"<<generation_<<",\"slots\":"<<active_slots_<<",\"jobs\":"<<jobs_
        <<",\"peak_host_owned\":"<<d.peak<<",\"submitted\":"<<d.submitted<<",\"returned\":"<<d.returned
        <<",\"discarded_jobs\":"<<d.discarded<<",\"producer_full_waits\":"<<d.producer_waits.size()<<",\"producer_full_wait_ns\":"<<producer
        <<",\"producer_max_full_wait_ns\":"<<maximum<<",\"consumer_submit_waits\":"<<d.consumer_waits.size()
        <<",\"consumer_submit_wait_ns\":"<<consumer<<",\"both_host_wait_ns\":"<<overlap
        <<",\"copy_not_ready_at_cpu_arrival\":"<<d.copy_pending<<",\"generation_wall_ns\":"<<ring_now()-d.begin<<"}\n";
}
void ExpertPipeline::run(){
    try{gpu_check(cudaSetDevice(device_id_));}catch(...){std::lock_guard<std::mutex> lock(mutex_);error_=std::current_exception();changed_.notify_all();return;}
    for(;;){
        uint64_t generation,bytes;size_t jobs;std::function<void(size_t,uint8_t*)> fill;
        {std::unique_lock<std::mutex> lock(mutex_);changed_.wait(lock,[&]{return stopped_ || pending_;});if(stopped_)return;
         generation=generation_;jobs=jobs_;bytes=bytes_;fill=std::move(fill_);pending_=false;working_=true;}
        try {
            for(size_t i=0;i<jobs;++i) {
                auto& slot=*slots_[i%active_slots_];
                bool reuse=false;
                Upload direct{nullptr,0};
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    const bool discarded=!uploads_.empty() && !uploads_[i].bytes;
                    const int64_t blocked=diagnostics_.enabled && !discarded && slot.issued && !slot.returned?ring_now():0;
                    changed_.wait(lock,[&]{return stopped_ || cancelling_ || (!uploads_.empty() && !uploads_[i].bytes) || !slot.issued || slot.returned;});
                    if(blocked)diagnostics_.producer_waits.emplace_back(blocked,ring_now());
                    if(stopped_)return;
                    if(cancelling_)break;
                    claimed_=i+1;
                    if(!uploads_.empty()){
                        direct=uploads_[i];
                        if(!direct.bytes)continue;
                    }
                    reuse=slot.issued;
                }
                if(reuse) {
                    // release() recorded used before publishing returned. The old
                    // copied wait was already enqueued before release could run.
                    gpu_check(cudaStreamWaitEvent(upload_,slot.used.value,0));
                    gpu_check(cudaEventSynchronize(slot.copied.value));
                }
                const uint64_t offset=(i%active_slots_)*stride_;
                const uint8_t* source=host_.data+offset;
                uint64_t payload=bytes;
                if(!direct.source)fill(i,host_.data+offset);
                else {source=direct.source;payload=direct.bytes;}
                { std::lock_guard<std::mutex> lock(mutex_);if(stopped_)return;if(cancelling_)break; }
                const std::string label=direct.trace_layer<0?"":"ds41.expert.h2d.layer."+std::to_string(direct.trace_layer)+
                    ".expert."+std::to_string(direct.trace_expert)+".generation."+std::to_string(generation)+".sequence."+std::to_string(i);
                {prefill::LedgerScope trace(direct.trace_layer>=0,label.c_str());
                 gpu_check(cudaMemcpyAsync(device_+offset,source,payload,cudaMemcpyHostToDevice,upload_));}
                gpu_check(cudaEventRecord(slot.copied.value,upload_));
                { std::lock_guard<std::mutex> lock(mutex_);
                  slot.generation=generation;slot.sequence=i;slot.issued=true;slot.returned=false;
                  last_uploaded_=i;have_uploaded_=true;
                  if(diagnostics_.enabled){++diagnostics_.submitted;diagnostics_.peak=std::max(diagnostics_.peak,++diagnostics_.owned);} }
                changed_.notify_all();
            }
        } catch(...) { std::lock_guard<std::mutex> lock(mutex_);error_=std::current_exception();changed_.notify_all(); }
        {std::lock_guard<std::mutex> lock(mutex_);working_=false;changed_.notify_all();}
    }
}
const uint8_t* ExpertPipeline::wait(size_t sequence,cudaStream_t reader) {
    if(sequence>=jobs_)throw std::out_of_range("expert issuer sequence");
    auto& slot=*slots_[sequence%active_slots_];
    std::unique_lock<std::mutex> lock(mutex_);
    if(!uploads_.empty() && !uploads_[sequence].bytes)throw std::logic_error("expert wait for discarded upload");
    const int64_t blocked=diagnostics_.enabled && !(slot.issued && slot.generation==generation_ && slot.sequence==sequence)?ring_now():0;
    changed_.wait(lock,[&]{return stopped_ || error_ || (slot.issued && slot.generation==generation_ && slot.sequence==sequence);});
    if(blocked)diagnostics_.consumer_waits.emplace_back(blocked,ring_now());
    if(error_)std::rethrow_exception(error_);
    if(stopped_)throw std::runtime_error("expert issuer cancelled");
    if(diagnostics_.enabled){
        const auto ready=cudaEventQuery(slot.copied.value);
        if(ready==cudaErrorNotReady)++diagnostics_.copy_pending;else gpu_check(ready);
    }
    gpu_check(cudaStreamWaitEvent(reader,slot.copied.value,0));
    return device_+(sequence%active_slots_)*stride_;
}
void ExpertPipeline::release(size_t sequence,cudaStream_t last_reader) {
    auto& slot=*slots_.at(sequence%active_slots_);
    std::lock_guard<std::mutex> lock(mutex_);
    if(!slot.issued || slot.generation!=generation_ || slot.sequence!=sequence || slot.returned)
        throw std::runtime_error("expert issuer stale/double return");
    gpu_check(cudaEventRecord(slot.used.value,last_reader));slot.returned=true;changed_.notify_all();
    if(diagnostics_.enabled){++diagnostics_.returned;--diagnostics_.owned;}
}
void ExpertPipeline::retire_unused(size_t sequence) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(sequence>=jobs_)throw std::out_of_range("expert unused sequence");
        const auto& slot=*slots_[sequence%active_slots_];
        if(slot.issued && slot.generation==generation_ &&
           (slot.sequence>sequence || (slot.sequence==sequence && slot.returned)))
            throw std::runtime_error("expert unused stale/double return");
    }
    // Publication still precedes the event wait. The same upload stream orders
    // copied -> used -> reuse; finish retains the immutable pinned source lease.
    wait(sequence,upload_);
    release(sequence,upload_);
}
void ExpertPipeline::finish() {
    std::unique_lock<std::mutex> lock(mutex_);changed_.wait(lock,[&]{return stopped_ || error_ || (!working_ && !pending_);});
    if(error_)std::rethrow_exception(error_);
    if(stopped_)throw std::runtime_error("expert issuer stopped");
    // Adaptive RAM ownership may change immediately after this layer. The
    // direct pinned sources must be released only after their final DMA read,
    // rather than merely after the issuer has submitted that read.
    if(!uploads_.empty() && have_uploaded_) {
        const auto copied=slots_[last_uploaded_%active_slots_]->copied.value;
        lock.unlock();gpu_check(cudaEventSynchronize(copied));lock.lock();
    }
    report_diagnostics();
}
void ExpertPipeline::cancel(cudaStream_t dequant,cudaStream_t reader) {
    {
        std::unique_lock<std::mutex> lock(mutex_);cancelling_=true;changed_.notify_all();
        changed_.wait(lock,[&]{return stopped_ || (!working_ && (!pending_ || error_));});
    }
    // Cleanup can run while unwinding a cancelled request. Preserve a CUDA
    // failure for the next caller instead of throwing from PrefetchGuard.
    const auto uploaded=cudaStreamSynchronize(upload_);
    const auto dequantized=cudaStreamSynchronize(dequant);
    const auto consumed=cudaStreamSynchronize(reader);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto failure=uploaded!=cudaSuccess?uploaded:(dequantized!=cudaSuccess?dequantized:consumed);
    if(failure!=cudaSuccess && !error_)error_=std::make_exception_ptr(std::runtime_error(cudaGetErrorString(failure)));
    if(stopped_ || error_)return;
    for(auto& slot:slots_){slot->issued=false;slot->returned=false;slot->generation=0;slot->sequence=0;}
    uploads_.clear();fill_={};jobs_=0;claimed_=0;have_uploaded_=false;pending_=false;
    diagnostics_={};cancelling_=false;
}
void ExpertPipeline::stop() {
    { std::lock_guard<std::mutex> lock(mutex_);stopped_=true; }
    changed_.notify_all();if(worker_.joinable())worker_.join();
    cudaStreamSynchronize(upload_);
}
void Runtime::Impl::begin_expert_prefetch(int first,int last,ExecutionStats& stats) {
    if(prefetch_active || first<0 || last>40 || first>=last)throw std::logic_error("expert prefetch layer range/state");
    std::array<std::array<bool,384>,40> resident{};
    for(const auto& key:experts->hot_pairs())resident[key.first][key.second]=true;
    std::vector<ExpertPipeline::Upload> jobs;prefetch_sizes.clear();
    for(int layer=first;layer<last;++layer){
        prefetch_offsets[layer]=jobs.size();
        prefetch_positions[layer].fill(std::numeric_limits<size_t>::max());
        for(int expert=0;expert<384;++expert)if(!resident[layer][expert]){
            const auto* source=experts->pinned_blob(layer,expert);
            if(!source)throw std::runtime_error("expert prefetch needs stable pinned cold weights");
            const uint64_t bytes=experts->format(layer).bytes;
            prefetch_positions[layer][expert]=jobs.size();
            jobs.push_back({source,bytes,options.trace?layer:-1,expert});prefetch_sizes.push_back(bytes);
            stats.h2d_expert_bytes+=bytes;stats.direct_expert_bytes+=bytes;stats.prefetch_expert_bytes+=bytes;
        }
    }
    prefetch_offsets[last]=jobs.size();prefetch_cursor=0;prefetch_total=jobs.size();
    if(jobs.empty())return;
    prefill::LedgerScope trace(options.trace,"ds41.expert.prefetch.start");
    expert_pipeline->start(++expert_generation,std::move(jobs));prefetch_active=true;
}
void Runtime::Impl::retire_expert_prefetch(size_t until,ExecutionStats& stats) {
    if(!prefetch_active || until<prefetch_cursor || until>prefetch_total)throw std::logic_error("expert prefetch cursor");
    while(prefetch_cursor<until){
        if(!prefetch_sizes[prefetch_cursor]){++prefetch_cursor;continue;}
        // An unselected expert still owns a copy event. Its raw slot may be
        // returned only after that copy; it never contributes to the result.
        prefill::LedgerScope trace(options.trace,"ds41.expert.prefetch.unused");
        expert_pipeline->retire_unused(prefetch_cursor);
        stats.prefetch_unused_bytes+=prefetch_sizes[prefetch_cursor++];
    }
}
void Runtime::Impl::prune_expert_prefetch(int layer,const std::array<bool,384>& selected,ExecutionStats& stats) {
    std::vector<size_t> unused;
    for(int expert=0;expert<384;++expert){
        const size_t sequence=prefetch_positions[layer][expert];
        if(!selected[expert] && sequence!=std::numeric_limits<size_t>::max())unused.push_back(sequence);
    }
    for(size_t sequence:expert_pipeline->discard(unused)){
        const uint64_t bytes=prefetch_sizes[sequence];prefetch_sizes[sequence]=0;
        stats.h2d_expert_bytes-=bytes;stats.direct_expert_bytes-=bytes;stats.prefetch_expert_bytes-=bytes;
        stats.prefetch_pruned_bytes+=bytes;
        const std::string label=options.trace?"ds41.expert.prefetch.pruned."+std::to_string(sequence):"";
        prefill::LedgerScope trace(options.trace,label.c_str());
    }
}
void Runtime::Impl::end_expert_prefetch() {
    if(!prefetch_active)return;
    if(prefetch_cursor!=prefetch_total)throw std::logic_error("expert prefetch incomplete retirement");
    expert_pipeline->finish();prefetch_active=false;prefetch_sizes.clear();
}
void Runtime::Impl::cancel_expert_prefetch() {
    if(!prefetch_active)return;
    expert_pipeline->cancel(dequant_stream.value,stream.value);
    prefetch_active=false;prefetch_sizes.clear();
    std::cerr<<"DeepSeek expert prefetch cancelled generation="<<expert_generation<<'\n';
}
} // namespace strata::models::deepseek41
