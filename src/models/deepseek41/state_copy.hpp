// Bounded, independent copies from a transaction snapshot. Large hidden copies
// retain cudaMemcpyAsync. Call flush before the transaction's existing sync.
class StateCopies {
    bool enabled_;cudaStream_t stream_;StateCopyBatch batch_{};
public:
    StateCopies(bool enabled,cudaStream_t stream):enabled_(enabled),stream_(stream){}
    void flush(){if(batch_.count){gpu_check(state_copy_gpu(batch_,stream_));batch_={};}}
    void add(void* destination,const void* source,size_t bytes) {
        if(!bytes)return;
        if(!enabled_||bytes>256*1024||((reinterpret_cast<uintptr_t>(destination)|reinterpret_cast<uintptr_t>(source)|bytes)&15)){
            flush();gpu_check(cudaMemcpyAsync(destination,source,bytes,cudaMemcpyDeviceToDevice,stream_));return;
        }
        if(batch_.count==96)flush();
        batch_.jobs[batch_.count++]={destination,source,static_cast<unsigned>(bytes),static_cast<unsigned>(batch_.blocks)};
        batch_.blocks+=(bytes+4095)/4096;
    }
};
