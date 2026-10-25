#include "runtime_internal.hpp"
#include "strata/prefill/ledger_scope.hpp"
#include <cstring>

namespace strata::models::deepseek41 {
namespace {std::atomic<bool> engram_short_enabled{true};}
void engram_short_set_enabled(bool value){engram_short_enabled.store(value);}
EngramPipeline::EngramPipeline(Workspace& work,Arithmetic arithmetic,bool trace,bool huge_pinned,uint32_t packed_type)
    : work_(work),arithmetic_(arithmetic),trace_(trace),packed_type_(packed_type),row_bytes_(packed_type==13?176:packed_type==14?210:272) {
    if(packed_type!=8&&packed_type!=13&&packed_type!=14)throw std::invalid_argument("Engram pipeline type");
    host_=std::make_unique<Pinned>(uint64_t(work.max_tokens)*24*row_bytes_,false,huge_pinned);
    if (!work.engram_input || !host_->pinned) throw std::runtime_error("Engram pipeline requires private BF16/pinned scratch");
    std::cerr<<"DeepSeek host_memory engram ";host_->info.write(std::cerr);std::cerr<<'\n';
    if (cublasCreate(&blas_)!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Engram cuBLAS creation");
    if (cublasSetStream(blas_,stream_.value)!=CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(blas_,CUBLAS_PEDANTIC_MATH)!=CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(blas_);blas_=nullptr;throw std::runtime_error("Engram cuBLAS initialization");
    }
    try{gpu_check(cudaGetDevice(&device_id_));worker_=std::thread([this]{run();});}
    catch(...){cublasDestroy(blas_);blas_=nullptr;throw;}
}
EngramPipeline::~EngramPipeline() { stop();if(blas_)cublasDestroy(blas_); }
void EngramPipeline::stop() {
    { std::lock_guard<std::mutex> lock(mutex_);stopped_=true;if(ticket_)ticket_->cancel(); }
    changed_.notify_all();
    if(worker_.joinable())worker_.join();
    cudaStreamSynchronize(stream_.value);
}
void EngramPipeline::start(std::shared_ptr<EngramBatch> ticket,uint64_t sequence,int count,const Matrix& first,const Matrix& second) {
    {std::unique_lock<std::mutex> lock(mutex_);
     if(stopped_ || !ticket || sequence<=sequence_ || count<1 || count>work_.max_tokens)throw std::runtime_error("Engram pipeline generation/count");
     if(sequence_ && used_[1]!=sequence_)throw std::runtime_error("Engram previous generation not consumed");
     changed_.wait(lock,[&]{return stopped_ || error_ || (!working_ && !pending_);});
     if(error_)std::rethrow_exception(error_);if(stopped_)throw std::runtime_error("Engram pipeline stopped");}
    // All old waits were enqueued by consume before a new job can be started.
    if(sequence_) {
        if(used_[1]!=sequence_)throw std::runtime_error("Engram previous generation not consumed");
        gpu_check(cudaStreamWaitEvent(stream_.value,consumed_[1].value,0));
        gpu_check(cudaStreamSynchronize(stream_.value));
    }
    { std::lock_guard<std::mutex> lock(mutex_);
      if(stopped_ || !ticket || sequence<=sequence_)throw std::runtime_error("Engram pipeline generation order");
      if(error_)std::rethrow_exception(error_);
      ticket_=ticket;sequence_=sequence;issued_={};used_={};matrices_={first,second};count_=count;pending_=true; }
    changed_.notify_all();
}
void EngramPipeline::run(){
    try{gpu_check(cudaSetDevice(device_id_));}catch(...){std::lock_guard<std::mutex> lock(mutex_);error_=std::current_exception();changed_.notify_all();return;}
    for(;;){
        std::shared_ptr<EngramBatch> ticket;uint64_t sequence;int count;std::array<Matrix,2> matrices;
        {std::unique_lock<std::mutex> lock(mutex_);changed_.wait(lock,[&]{return stopped_ || pending_;});if(stopped_)return;
         ticket=ticket_;sequence=sequence_;count=count_;matrices=matrices_;pending_=false;working_=true;}
        try {
            for(int table=0;table<2;++table) {
                const auto& data=ticket->wait_layer(table);
                if(table) {
                    std::unique_lock<std::mutex> lock(mutex_);
                    changed_.wait(lock,[&]{return stopped_ || used_[0]==sequence;});
                    if(stopped_)return;
                    // record-before-publish and acquire-before-wait: never an old event.
                    gpu_check(cudaStreamWaitEvent(stream_.value,consumed_[0].value,0));
                    lock.unlock();
                    // A queued device wait alone does not protect the pinned CPU source.
                    gpu_check(cudaEventSynchronize(copied_.value));
                }
                { std::lock_guard<std::mutex> lock(mutex_);if(stopped_)return; }
                prefill::LedgerScope trace(trace_,table?"ds41.engram.prepare14":"ds41.engram.prepare1");
                if(data.size()!=uint64_t(count)*48*row_bytes_)throw std::runtime_error("Engram packed batch size");
                for(int t=0;t<count;++t)std::memcpy(host_->data+uint64_t(t)*24*row_bytes_,data.data()+(uint64_t(t)*48+table*24)*row_bytes_,24*row_bytes_);
                gpu_check(cudaMemcpyAsync(work_.engram_packed,host_->data,uint64_t(count)*24*row_bytes_,cudaMemcpyHostToDevice,stream_.value));
                gpu_check(cudaEventRecord(copied_.value,stream_.value));
                if(arithmetic_==Arithmetic::production)
                    gpu_check(dequant_round_bf16_gpu(work_.engram_packed,packed_type_,work_.engram_emb,int64_t(count)*6144,stream_.value));
                else gpu_check(dequant_gpu(work_.engram_packed,packed_type_,work_.engram_emb,int64_t(count)*6144,stream_.value));
                matrix_product(matrices[table],work_.engram_emb,work_.engram_kv,count,blas_,stream_.value,work_.engram_input,arithmetic_,0,0,0,0,true,
                    engram_short_enabled.load()&&arithmetic_==Arithmetic::production&&count>1&&count<=8);
                gpu_check(cudaEventRecord(ready_[table].value,stream_.value));
                { std::lock_guard<std::mutex> lock(mutex_);issued_[table]=sequence; }
                changed_.notify_all();
            }
        } catch(...) { std::lock_guard<std::mutex> lock(mutex_);error_=std::current_exception();changed_.notify_all(); }
        {std::lock_guard<std::mutex> lock(mutex_);working_=false;changed_.notify_all();}
    }
}
void EngramPipeline::wait(int table,uint64_t generation,uint64_t sequence,cudaStream_t consumer) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock,[&]{return stopped_ || error_ || issued_[table]==sequence;});
    if(error_)std::rethrow_exception(error_);
    if(stopped_ || sequence!=sequence_ || ticket_->generation!=generation)throw std::runtime_error("Engram stale/cancelled prepare");
    gpu_check(cudaStreamWaitEvent(consumer,ready_[table].value,0));
}
void EngramPipeline::consume(int table,uint64_t sequence,cudaStream_t consumer) {
    gpu_check(cudaEventRecord(consumed_[table].value,consumer));
    { std::lock_guard<std::mutex> lock(mutex_);
      if(sequence!=sequence_ || issued_[table]!=sequence)throw std::runtime_error("Engram consumed generation");
      used_[table]=sequence; }
    changed_.notify_all();
}
} // namespace strata::models::deepseek41
