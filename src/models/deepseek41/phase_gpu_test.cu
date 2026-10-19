#include "phase_cache.hpp"
#include <algorithm>
#include <iostream>
#include <map>
#include <set>

namespace d=strata::models::deepseek41;
namespace {
__global__ void delay(unsigned long long clocks){const auto start=clock64();while(clock64()-start<clocks){} }
__global__ void checksum(const uint8_t* bytes,uint64_t n,unsigned long long* out){unsigned long long sum=0;for(uint64_t i=0;i<n;++i)sum+=bytes[i];*out=sum;}
void require(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
std::vector<uint8_t> payload(d::ExpertKey key,uint64_t bytes){return std::vector<uint8_t>(bytes,1+(key.first*17+key.second)%251);}
void run(bool pageable){
    std::array<uint64_t,40> sizes{};for(int l=0;l<40;++l)sizes[l]=l<14?768:512;
    std::vector<d::ExpertKey> hot;std::set<d::ExpertKey> hot_set;
    for(int e=0;e<36;++e)for(int l=0;l<40;++l){hot.emplace_back(l,e);hot_set.emplace(l,e);}
    auto plan=d::make_phase_plan(sizes,hot,{},32768);
    std::map<d::ExpertKey,std::vector<uint8_t>> ram;
    for(int l=0;l<40;++l)for(int e=0;e<384;++e)if(!hot_set.count({l,e}))ram.emplace(d::ExpertKey{l,e},payload({l,e},sizes[l]));
    d::Stream main,upload,gate;d::ExecutionStats stats;
    d::PhaseCache cache(plan,[&](d::ExpertKey key){const auto it=ram.find(key);return it==ram.end()?nullptr:it->second.data();},upload.value,1536,pageable);
    for(size_t i=0;i<plan.decode.size();++i){auto source=payload(plan.decode[i].key,plan.decode[i].bytes);cache.initialize(i,source.data());d::gpu_check(cudaStreamSynchronize(upload.value));}
    require(cache.pinned_staging_bytes()==(pageable?0:cache.staging_bytes()),"pageable fallback accounting");
    require(!cache.begin_prefill(128,64,true,main.value,stats),"short prompt borrowed");
    require(stats.phase_backup_bytes==0,"short prompt copied backup");
    for(const auto& slice:plan.decode)require(cache.cpu(slice.key)==nullptr,"GPU-only target had a RAM backup before borrowing");
    require(cache.begin_prefill(4096,64,true,main.value,stats),"long CED prompt did not borrow");
    const uint64_t first_backup=stats.phase_backup_bytes;
    uint64_t expected_backup=0;
    for(const auto& slice:plan.decode)if(slice.release_after>=0){
        expected_backup+=slice.bytes;require(cache.cpu(slice.key)!=nullptr,"missing decode RAM backup");
        const auto original=payload(slice.key,slice.bytes);require(std::equal(original.begin(),original.end(),cache.cpu(slice.key)),"D2H backup differs from original bytes");
    }
    require(first_backup==expected_backup,"backup bytes differ from overwritten decode experts");
    // Earlier chunks keep exactly the same temporary addresses and never restore back layers.
    for(const auto& slice:plan.encoder){
        const auto* a=cache.gpu(slice.key,main.value,stats,0);const auto* b=cache.gpu(slice.key,main.value,stats,0);require(a&&a==b,"temporary residency changed between encoder chunks");
        require(ram.count(slice.key),"front temporary expert lost its RAM copy");
    }
    require(stats.phase_restore_h2d_bytes==0,"early encoder chunk restored back experts");
    // Force readers to remain pending while restoration is submitted on another stream.
    // Missing consumption fences would overwrite the mixed-size region before these checksums run.
    // cudaMalloc may synchronize streams, so allocate before deliberately delaying readers.
    strata::core::DeviceArena result_arena(plan.encoder.size()*sizeof(unsigned long long));
    auto* result=static_cast<unsigned long long*>(result_arena.alloc(plan.encoder.size()*sizeof(unsigned long long)));
    d::Event ready;delay<<<1,1,0,gate.value>>>(100000000);d::gpu_check(cudaGetLastError());
    d::gpu_check(cudaEventRecord(ready.value,gate.value));d::gpu_check(cudaStreamWaitEvent(main.value,ready.value,0));
    for(int layer=0;layer<20;++layer){
        for(size_t i=0;i<plan.encoder.size();++i)if(plan.encoder[i].key.first==layer){
            const auto& slice=plan.encoder[i];const auto* pointer=cache.gpu(slice.key,main.value,stats,0);
            require(pointer!=nullptr,"last chunk temporary expert was released before its read");
            checksum<<<1,1,0,main.value>>>(pointer,slice.bytes,result+i);d::gpu_check(cudaGetLastError());
        }
        if(layer==0)require(cudaEventQuery(ready.value)==cudaErrorNotReady,"test did not retain a pending encoder reader before restoration");
        cache.encoder_layer_done(layer,main.value,stats);
        // The back target becomes visible by its exact release deadline, before all front layers finish.
        for(const auto& slice:plan.decode)if(slice.release_after==layer)
            require(cache.gpu(slice.key,main.value,stats,1)!=nullptr,"decode target missed its replay deadline");
    }
    cache.finish_prefill(main.value,stats);
    d::gpu_check(cudaStreamSynchronize(main.value));d::gpu_check(cudaStreamSynchronize(upload.value));
    std::vector<unsigned long long> sums(plan.encoder.size());
    d::gpu_check(cudaMemcpy(sums.data(),result,sums.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<sums.size();++i){
        const auto& slice=plan.encoder[i];require(sums[i]==uint64_t(1+(slice.key.first*17+slice.key.second)%251)*slice.bytes,"restoration overwrote a live encoder reader");
        require(cache.gpu(slice.key,main.value,stats,2)==nullptr,"temporary front expert remained in decode GPU region");
    }
    for(const auto& slice:plan.decode){
        std::vector<uint8_t> data(slice.bytes);const auto* pointer=cache.gpu(slice.key,main.value,stats,1);
        require(pointer!=nullptr,"restored decode slot missing");
        d::gpu_check(cudaMemcpyAsync(data.data(),pointer,slice.bytes,cudaMemcpyDeviceToHost,main.value));d::gpu_check(cudaStreamSynchronize(main.value));
        require(data==payload(slice.key,slice.bytes),"mixed-size restoration payload");
    }
    require(stats.phase_encoder_h2d_bytes==plan.encoder_bytes,"encoder temporary H2D ledger");
    require(stats.phase_restore_h2d_bytes==expected_backup,"restoration H2D was duplicated or incomplete");
    require(stats.phase_backup_bytes==first_backup,"temporary front weights were copied D2H");
    require(stats.phase_replay_reuse_bytes==expected_backup,"replay did not reuse final resident uploads");
    // Vast prefill counts must not drown the decode/replay signal.
    std::vector<int> front_routes(4096*6),decode_routes(16*6);
    for(int t=0;t<4096;++t)for(int k=0;k<6;++k)front_routes[t*6+k]=35-k;
    for(int t=0;t<16;++t)for(int k=0;k<6;++k)decode_routes[t*6+k]=383-k;
    for(int l=20;l<40;++l){cache.observe(l,front_routes.data(),4096,0);cache.observe(l,decode_routes.data(),16,1);cache.observe(l,decode_routes.data(),16,2);}
    cache.adjust_decode(16,main.value,stats);
    d::gpu_check(cudaStreamSynchronize(upload.value));
    require(stats.phase_migrations>0&&stats.phase_dynamic_h2d_bytes+stats.phase_dynamic_d2h_bytes<=1536,"bounded dynamic migration or phase heat separation");
    require(cache.gpu({20,378},main.value,stats,2)!=nullptr,"actual decode hot expert was not promoted with stable ID tie-breaking");
    auto saved=cache.snapshot();cache.observe(20,decode_routes.data(),16,2);cache.restore_snapshot(saved);require(cache.snapshot()==saved,"rejected verification polluted decode heat");
    const auto second_start_backup=stats.phase_backup_bytes;
    require(cache.begin_prefill(4096,64,true,main.value,stats),"second request borrow failed");
    require(stats.phase_backup_bytes==second_start_backup,"persistent backups were copied again");
    // Cancel midway through the final chunk, then re-enter with a short request.
    cache.encoder_layer_done(0,main.value,stats);cache.finish_prefill(main.value,stats);
    require(!cache.begin_prefill(128,64,true,main.value,stats),"short re-entry borrowed");
    for(const auto& slice:plan.decode){
        const auto* pointer=cache.gpu(slice.key,main.value,stats,2);require(pointer!=nullptr,"cancel/re-entry lost a decode target");
        std::vector<uint8_t> data(slice.bytes);d::gpu_check(cudaMemcpyAsync(data.data(),pointer,slice.bytes,cudaMemcpyDeviceToHost,main.value));d::gpu_check(cudaStreamSynchronize(main.value));
        require(data==payload(slice.key,slice.bytes),"cancel/re-entry reused a stale expert");
    }
    std::cout<<"phase GPU readers, deadlines, backup reuse, replay reuse, bounded dynamics, rollback and cancel passed; pageable="<<pageable<<'\n';
}
}
int main(){try{run(false);run(true);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
