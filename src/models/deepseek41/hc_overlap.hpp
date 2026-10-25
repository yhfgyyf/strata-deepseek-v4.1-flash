#pragma once

// Fork and rejoin inside a single captured segment. Both streams only read h;
// the caller must join before any writer or consumer of the new HC buffers.
class HcOverlap {
    cudaStream_t main_,side_;
    cudaEvent_t done_;
    bool active_,recorded_=false;
public:
    HcOverlap(bool active,cudaStream_t main,cudaStream_t side,cudaEvent_t ready,cudaEvent_t done)
        :main_(main),side_(side),done_(done),active_(active) {
        if(active_){gpu_check(cudaEventRecord(ready,main_));gpu_check(cudaStreamWaitEvent(side_,ready,0));}
    }
    cudaStream_t stream() const{return active_?side_:main_;}
    void produced(){if(active_){gpu_check(cudaEventRecord(done_,side_));recorded_=true;}}
    void join(){if(active_){if(!recorded_)produced();gpu_check(cudaStreamWaitEvent(main_,done_,0));active_=false;}}
    ~HcOverlap(){if(active_){if(!recorded_)cudaEventRecord(done_,side_);cudaStreamWaitEvent(main_,done_,0);}}
    HcOverlap(const HcOverlap&)=delete;
};
