#pragma once
// Helper threads spawned by the request thread inherit its CPU mask. The servers pin the
// request thread to one CPU, so a memcpy-heavy helper would time-slice against it. Capture
// the mask on the spawning thread and apply it first thing in the helper.
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif
namespace strata::models::deepseek41 {
struct HelperAffinity {
#if defined(__linux__)
    cpu_set_t mask{};
    bool valid=false;
#endif
    // Call on the spawning thread: all online CPUs except the caller's (unchanged if that leaves none).
    static HelperAffinity away_from_caller(){
        HelperAffinity a;
#if defined(__linux__)
        cpu_set_t caller;CPU_ZERO(&caller);
        if(pthread_getaffinity_np(pthread_self(),sizeof caller,&caller))return a;
        const long online=sysconf(_SC_NPROCESSORS_ONLN);
        CPU_ZERO(&a.mask);int count=0,all=0;
        for(long c=0;c<online&&c<CPU_SETSIZE;++c){++all;if(!CPU_ISSET(c,&caller)){CPU_SET(c,&a.mask);++count;}}
        a.valid=count>0&&CPU_COUNT(&caller)<all;
#endif
        return a;
    }
    void apply()const{
#if defined(__linux__)
        if(valid)pthread_setaffinity_np(pthread_self(),sizeof mask,&mask);
#endif
    }
};
} // namespace strata::models::deepseek41
