#include "strata/models/deepseek41/engram.hpp"
#include "strata/models/deepseek41/prefill_chunk.hpp"
#include "strata/platform/direct_file.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::models::deepseek41 {
namespace {
constexpr uint64_t page_bytes = 4096;
#if defined(__linux__)
struct NativeAio {
    int fd=-1;aio_context_t context=0;
    explicit NativeAio(const std::string& path){
        fd=::open(path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC);
        if(fd<0)throw std::runtime_error("Engram AIO direct open");
        if(::syscall(SYS_io_setup,256,&context)<0){const int code=errno;::close(fd);fd=-1;throw std::runtime_error("Engram io_setup: "+std::string(std::strerror(code)));}
    }
    void abort(){if(context){::syscall(SYS_io_destroy,context);context=0;}}
    ~NativeAio(){abort();if(fd>=0)::close(fd);}
};
#endif
EngramHashSpec read_spec(const Model& m) {
    EngramHashSpec s;
    auto copy = [&](const char* key, auto& out) {
        const auto* v = m.file().get(std::string("deepseek41.") + key);
        if (!v || v->items.size() != out.size()) throw std::runtime_error("Engram metadata array");
        for (size_t i = 0; i < out.size(); ++i) out[i] = v->items[i].u;
    };
    copy(m.ple_engram()?"ple.layer_multipliers":"engram.multipliers", s.multipliers);
    copy(m.ple_engram()?"ple.head_vocab_sizes":"engram.primes", s.primes);
    copy(m.ple_engram()?"ple.head_offsets":"engram.offsets", s.offsets);
    if(m.ple_engram())for(int h=24;h<48;++h)s.offsets[h]-=384006168;
    s.pad = m.token_map().at(m.file().get("deepseek41.engram.pad_token_id")->u);
    return s;
}
struct PageEntry {
    uint64_t key = UINT64_MAX, age = 0;
    std::unique_ptr<std::array<uint8_t, page_bytes>> data;
};
class PageCache {
public:
    explicit PageCache(uint64_t budget) {
        // Include entry storage and conservative allocator overhead in the explicit cap.
        const uint64_t count = budget / (page_bytes + sizeof(PageEntry) + 32) / 8 * 8;
        entries_.resize(static_cast<size_t>(count));
    }
    const uint8_t* get(uint64_t key) {
        if (entries_.empty()) return nullptr;
        const size_t first = bucket(key);
        for (size_t i = first; i < first + 8; ++i) if (entries_[i].key == key) {
            entries_[i].age = ++tick_; return entries_[i].data->data();
        }
        return nullptr;
    }
    void put(uint64_t key, const uint8_t* data) {
        if (entries_.empty()) return;
        const size_t first = bucket(key);
        size_t chosen = first;
        for (size_t i = first; i < first + 8; ++i) {
            if (entries_[i].key == key || !entries_[i].data) { chosen = i; break; }
            if (entries_[i].age < entries_[chosen].age) chosen = i;
        }
        auto& e = entries_[chosen];
        if (!e.data) { e.data = std::make_unique<std::array<uint8_t, page_bytes>>(); ++allocated_; }
        e.key = key; e.age = ++tick_; std::memcpy(e.data->data(), data, page_bytes);
    }
    uint64_t owned() const { return entries_.capacity() * sizeof(PageEntry) + allocated_ * (page_bytes + 32); }
private:
    size_t bucket(uint64_t key) const {
        key ^= key >> 30; key *= 0xbf58476d1ce4e5b9ull; key ^= key >> 27;
        return (key % (entries_.size() / 8)) * 8;
    }
    std::vector<PageEntry> entries_;
    uint64_t tick_ = 0, allocated_ = 0;
};
}
EngramHashSpec engram_hash_spec(const Model& m) { return read_spec(m); }
EngramQuery engram_hash(const EngramHashSpec& s, const std::vector<int32_t>& map,
                       const std::vector<int>& tokens, const std::vector<uint8_t>& image,
                       const EngramHistory& before) {
    if ((!image.empty() && image.size() != tokens.size()) || before.position < 0)
        throw std::invalid_argument("Engram token/image/history mismatch");
    EngramQuery q; q.rows.resize(tokens.size() * 48); q.image = image;
    if (q.image.empty()) q.image.resize(tokens.size());
    q.after = before;
    for (size_t t = 0; t < tokens.size(); ++t) {
        if (tokens[t] < 0 || static_cast<size_t>(tokens[t]) >= map.size()) throw std::out_of_range("Engram token ID");
        const int32_t current = q.image[t] ? -1 : map[tokens[t]];
        for (int layer = 0; layer < 2; ++layer) {
            uint64_t rolling = 0;
            bool blocked = false;
            for (int shift = 0; shift < 4; ++shift) {
                const int32_t source = shift ? q.after.recent[shift - 1] : current;
                blocked |= q.after.position < shift || source == -1;
                const uint64_t value = static_cast<uint64_t>(blocked ? s.pad : source);
                const uint64_t multiplier = s.multipliers[layer * 4 + shift];
                if (!multiplier || !(multiplier & 1) || (value && multiplier > INT64_MAX / value))
                    throw std::invalid_argument("Engram multiplier overflow/geometry");
                rolling ^= value * multiplier;
                if (shift) for (int head = 0; head < 8; ++head) {
                    const size_t col = layer * 24 + (shift - 1) * 8 + head;
                    if (!s.primes[col]) throw std::invalid_argument("Engram zero prime");
                    q.rows[t * 48 + col] = rolling % s.primes[col] + s.offsets[col];
                }
            }
        }
        q.after.recent = {current, q.after.recent[0], q.after.recent[1]}; ++q.after.position;
    }
    return q;
}
struct EngramBatch::State {
    std::mutex mutex;
    std::condition_variable ready;
    bool done = false;
    std::array<bool,2> layer_ready{};
    std::string error;
    std::vector<uint8_t> data;
};
const std::vector<uint8_t>& EngramBatch::wait() {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->ready.wait(lock, [&] { return state->done; });
    if (cancelled.load()) throw std::runtime_error("Engram read cancelled");
    if (!state->error.empty()) throw std::runtime_error(state->error);
    return state->data;
}
const std::vector<uint8_t>& EngramBatch::wait_layer(int table) {
    if (table < 0 || table > 1) throw std::out_of_range("Engram table index");
    std::unique_lock<std::mutex> lock(state->mutex);
    state->ready.wait(lock, [&] { return state->layer_ready[table] || state->done; });
    if (cancelled.load()) throw std::runtime_error("Engram read cancelled");
    if (!state->error.empty()) throw std::runtime_error(state->error);
    if (!state->layer_ready[table]) throw std::runtime_error("Engram incomplete table");
    return state->data;
}
struct EngramReader::Impl {
    struct Work { EngramQuery query; std::shared_ptr<EngramBatch> batch; bool split = false; };
    strata::platform::DirectFile file;
    std::array<uint64_t, 2> offsets, rows;
    uint64_t row_bytes;
    PageCache cache;
    size_t queue_limit;
    bool continuous_reads;
#if defined(__linux__)
    std::unique_ptr<NativeAio> aio;
#endif
    std::mutex mutex;
    std::condition_variable queued, room;
    std::deque<Work> queue;
    bool stop = false;
    std::thread worker;
    mutable std::mutex stats_mutex;
    EngramReadStats counters;
    std::deque<double> latencies;
    Impl(const std::string& path, std::array<uint64_t, 2> o, std::array<uint64_t, 2> r,
         uint64_t budget, size_t limit,bool continuous,bool native_aio,uint64_t packed_row_bytes)
        : offsets(o), rows(r), row_bytes(packed_row_bytes), cache(budget), queue_limit(limit),continuous_reads(continuous) {
        if (!limit) throw std::invalid_argument("Engram zero queue limit");
        if(row_bytes!=272 && row_bytes!=210 && row_bytes!=176)throw std::invalid_argument("Engram packed row bytes");
        std::string error;
        if (!file.open(path, error,native_aio)) throw std::runtime_error("Engram direct read: " + error);
        if(native_aio){
            if(!continuous)throw std::invalid_argument("Engram AIO requires continuous reads");
#if defined(__linux__)
            aio=std::make_unique<NativeAio>(path);
#else
            throw std::runtime_error("Engram native AIO requires Linux");
#endif
        }
        for (int l = 0; l < 2; ++l)
            if (offsets[l] > file.size() || rows[l] > (file.size() - offsets[l]) / row_bytes)
                throw std::out_of_range("Engram table outside GGUF");
        worker = std::thread([this] { run(); });
    }
    ~Impl() {
        { std::lock_guard<std::mutex> lock(mutex); stop = true; for (auto& w : queue) w.batch->cancel(); }
        queued.notify_all(); room.notify_all(); worker.join();
    }
    bool complete_read(uint64_t first,uint32_t bytes,uint8_t* data,const strata::platform::Completion& c) const {
        const uint64_t offset=first*page_bytes;
        if(offset>=file.size())return false;
        const uint64_t expected=std::min<uint64_t>(bytes,file.size()-offset);
        if(!c.ok||c.bytes!=expected)return false;
        // Only pad beyond the original EOF; all requested tensor row bytes
        // were bounds checked. A file truncated after open still fails.
        if(expected<bytes)std::memset(data+expected,0,bytes-expected);
        return true;
    }
    void read_continuous(const std::vector<uint64_t>& missing,const std::vector<uint64_t>& pages,uint8_t* loaded,
                         const EngramBatch& batch,uint64_t& disk,uint64_t& reads) {
        struct Read {uint64_t first,index;uint32_t bytes;};
        std::vector<Read> requests;
        for(size_t i=0;i<missing.size();){
            const uint64_t first=missing[i++];uint32_t count=1;
            while(i<missing.size() && count<16 && missing[i]==first+count){++i;++count;}
            requests.push_back({first,uint64_t(std::lower_bound(pages.begin(),pages.end(),first)-pages.begin()),count*uint32_t(page_bytes)});
        }
        const size_t depth=std::min<size_t>(256,requests.size());
        std::vector<size_t> active(depth,SIZE_MAX);size_t cursor=0,inflight=0;std::string error;
#if defined(__linux__)
        std::vector<iocb> controls(depth);
#endif
        struct Drain {
            Impl& owner;size_t& inflight;int exceptions=std::uncaught_exceptions();
            ~Drain(){if(inflight && std::uncaught_exceptions()>exceptions){
#if defined(__linux__)
                if(owner.aio){owner.aio->abort();return;}
#endif
                owner.file.close();
            }}
        } drain{*this,inflight};
        auto submit=[&](size_t slot){
            if(cursor==requests.size() || batch.cancelled.load() || !error.empty())return;
            const auto& r=requests[cursor];
            bool accepted=false;
#if defined(__linux__)
            if(aio){
                auto& cb=controls[slot];cb={};cb.aio_data=slot;cb.aio_lio_opcode=IOCB_CMD_PREAD;cb.aio_fildes=aio->fd;
                cb.aio_buf=reinterpret_cast<uintptr_t>(loaded+r.index*page_bytes);cb.aio_nbytes=r.bytes;cb.aio_offset=r.first*page_bytes;
                iocb* pointer=&cb;long result;
                do{result=::syscall(SYS_io_submit,aio->context,1,&pointer);}while(result<0 && errno==EINTR);
                accepted=result==1;if(!accepted)error="Engram io_submit failed: "+std::string(std::strerror(errno));
            }else
#endif
            accepted=file.submit(r.first*page_bytes,loaded+r.index*page_bytes,r.bytes,slot,error);
            if(accepted){
                active[slot]=cursor++;++inflight;
            }
        };
        for(size_t slot=0;slot<depth;++slot)submit(slot);
        // Retire and replenish individual slots. Keep storage alive and drain
        // every accepted read even on cancellation or an I/O error.
        while(inflight){
            strata::platform::Completion complete[64];int count=0;
#if defined(__linux__)
            if(aio){
                io_event events[64];long result;
                do{result=::syscall(SYS_io_getevents,aio->context,1,64,events,nullptr);}while(result<0 && errno==EINTR);
                if(result<0)throw std::runtime_error("Engram io_getevents failed");
                count=result;for(int j=0;j<count;++j)complete[j]={events[j].data,events[j].res<0?0u:uint32_t(events[j].res),events[j].res>=0 && events[j].res2==0};
            }else
#endif
            while(!(count=file.wait(complete,64,-1))){}
            for(int j=0;j<count;++j){const auto& c=complete[j];
            if(c.tag>=depth || active[c.tag]==SIZE_MAX)throw std::runtime_error("Engram invalid rolling completion");
            const auto& r=requests[active[c.tag]];active[c.tag]=SIZE_MAX;--inflight;
            if(!complete_read(r.first,r.bytes,loaded+r.index*page_bytes,c))error="Engram short/failed continuous read";
            disk+=c.bytes;++reads;submit(c.tag);
            }
        }
        if(!error.empty())throw std::runtime_error(error);
    }
    void read(Work& work) {
        auto& q = work.query;
        if (q.rows.size() != q.image.size() * 48) throw std::invalid_argument("Engram query shape");
        auto& data = work.batch->state->data;
        data.resize(q.rows.size() * row_bytes, 0);
        // Validate both tables before publishing either, then write disjoint bytes.
        for (size_t i = 0; i < q.rows.size(); ++i)
            if (!q.image[i / 48] && q.rows[i] >= rows[(i % 48) / 24]) throw std::out_of_range("Engram table row");
        for (int pass = 0; pass < (work.split ? 2 : 1); ++pass) {
        const double layer_start=strata::platform::now_us();
        std::vector<uint64_t> pages;
        for (size_t i = 0; i < q.rows.size(); ++i) if (!q.image[i / 48] && (!work.split || int(i % 48) / 24 == pass)) {
            const int l = (i % 48) / 24;
            if (q.rows[i] >= rows[l]) throw std::out_of_range("Engram table row");
            const uint64_t offset = offsets[l] + q.rows[i] * row_bytes;
            pages.push_back(offset / page_bytes);
            if (offset % page_bytes + row_bytes > page_bytes) pages.push_back(offset / page_bytes + 1);
        }
        std::sort(pages.begin(), pages.end()); pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
        // Batch-local pages retain every required row even when the bounded persistent cache evicts it.
        std::unordered_map<uint64_t, std::array<uint8_t, page_bytes>> loaded;
        using Pages=std::unique_ptr<uint8_t,decltype(&strata::platform::DirectFile::free_aligned)>;
        Pages contiguous(nullptr,&strata::platform::DirectFile::free_aligned);
        if(continuous_reads && !pages.empty()){
            // Read directly into the final aligned page slab: no per-page
            // allocation, value-initialization, hash insertion or staging copy.
            contiguous.reset(static_cast<uint8_t*>(strata::platform::DirectFile::alloc_aligned(pages.size()*page_bytes)));
            if(!contiguous)throw std::bad_alloc();
        }
        auto page_data=[&](uint64_t p)->const uint8_t*{
            if(!continuous_reads)return loaded.at(p).data();
            const auto at=std::lower_bound(pages.begin(),pages.end(),p);
            if(at==pages.end() || *at!=p)throw std::logic_error("Engram missing slab page");
            return contiguous.get()+size_t(at-pages.begin())*page_bytes;
        };
        std::vector<uint64_t> missing;
        uint64_t hits = 0, disk = 0, reads = 0;
        for (size_t i=0;i<pages.size();++i) {const uint64_t p=pages[i];
            if (const auto* cached = cache.get(p)) { std::memcpy(continuous_reads?contiguous.get()+i*page_bytes:loaded[p].data(), cached, page_bytes); ++hits; }
            else missing.push_back(p);
        }
        struct Read { uint64_t first; uint32_t bytes; void* buffer; };
        size_t cursor = 0;
        if(continuous_reads)read_continuous(missing,pages,contiguous.get(),*work.batch,disk,reads);
        while (!continuous_reads && cursor < missing.size()) {
            if (work.batch->cancelled.load()) return;
            std::vector<Read> pending;
            std::string error;
            for (int j = 0; j < 64 && cursor < missing.size(); ++j) {
                const uint64_t first = missing[cursor++]; uint32_t count = 1;
                while (cursor < missing.size() && count < 16 && missing[cursor] == first + count) { ++count; ++cursor; }
                const uint32_t bytes = count * page_bytes;
                void* buffer = strata::platform::DirectFile::alloc_aligned(bytes);
                if (!buffer) { error = "Engram aligned buffer allocation"; break; }
                pending.push_back({first, bytes, buffer});
                if (!file.submit(first * page_bytes, buffer, bytes, pending.size() - 1, error)) {
                    pending.pop_back(); strata::platform::DirectFile::free_aligned(buffer); break;
                }
            }
            for (size_t done = 0; done < pending.size(); ++done) {
                strata::platform::Completion c;
                while (!file.wait(&c, 1, -1)) {}
                if (c.tag >= pending.size()) { error = "Engram unexpected I/O completion"; continue; }
                const auto& r = pending[c.tag];
                if (!complete_read(r.first,r.bytes,static_cast<uint8_t*>(r.buffer),c)) error = "Engram short/failed direct read";
                else for (uint32_t b = 0; b < r.bytes; b += page_bytes) {
                    const uint64_t key = r.first + b / page_bytes;
                    const auto* src = static_cast<uint8_t*>(r.buffer) + b;
                    std::memcpy(loaded[key].data(), src, page_bytes); cache.put(key, src);
                }
                disk += c.bytes; ++reads;
            }
            for (const auto& r : pending) strata::platform::DirectFile::free_aligned(r.buffer);
            if (!error.empty()) throw std::runtime_error(error);
        }
        if(work.batch->cancelled.load())return;
        for (size_t i = 0; i < q.rows.size(); ++i) if (!q.image[i / 48] && (!work.split || int(i % 48) / 24 == pass)) {
            const int l = (i % 48) / 24;
            const uint64_t offset = offsets[l] + q.rows[i] * row_bytes, page = offset / page_bytes;
            const uint64_t in_page = offset % page_bytes, first = std::min(row_bytes, page_bytes - in_page);
            std::memcpy(data.data() + i * row_bytes, page_data(page) + in_page, first);
            if (first < row_bytes) std::memcpy(data.data() + i * row_bytes + first, page_data(page+1), row_bytes - first);
        }
        { std::lock_guard<std::mutex> lock(stats_mutex);
          counters.logical_bytes += data.size() / (work.split ? 2 : 1); counters.disk_bytes += disk;
          counters.reads += reads; counters.page_hits += hits; counters.cache_owned_bytes = cache.owned();
          if(work.split){counters.layer_disk_bytes[pass]+=disk;counters.layer_reads[pass]+=reads;counters.layer_page_hits[pass]+=hits;counters.layer_wall_us[pass]+=strata::platform::now_us()-layer_start;} }
        { std::lock_guard<std::mutex> lock(work.batch->state->mutex);
          if (work.split) work.batch->state->layer_ready[pass] = true;
          else work.batch->state->layer_ready = {true,true}; }
        work.batch->state->ready.notify_all();
        if(continuous_reads){
            // Cache maintenance is not needed by this immutable GPU payload.
            const double fill_start=strata::platform::now_us();
            for(uint64_t p:missing)cache.put(p,page_data(p));
            std::lock_guard<std::mutex> lock(stats_mutex);counters.cache_owned_bytes=cache.owned();
            if(work.split)counters.layer_cache_fill_us[pass]+=strata::platform::now_us()-fill_start;
        }
        }
        std::lock_guard<std::mutex> lock(stats_mutex); ++counters.batches;
    }
    void run() {
        for (;;) {
            Work work;
            { std::unique_lock<std::mutex> lock(mutex); queued.wait(lock, [&] { return stop || !queue.empty(); });
              if (queue.empty() && stop) return;
              work = std::move(queue.front()); queue.pop_front(); room.notify_one(); }
            const double start = strata::platform::now_us();
            std::string error;
            try { if (!work.batch->cancelled.load()) read(work); }
            catch (const std::exception& e) { error = e.what(); }
            { std::lock_guard<std::mutex> lock(stats_mutex);
              latencies.push_back(strata::platform::now_us() - start); if (latencies.size() > 1024) latencies.pop_front(); }
            { std::lock_guard<std::mutex> lock(work.batch->state->mutex);
              work.batch->state->error = std::move(error); work.batch->state->done = true; }
            work.batch->state->ready.notify_all();
        }
    }
};
EngramReader::EngramReader(const std::string& path, const std::array<uint64_t, 2>& offsets,
                         const std::array<uint64_t, 2>& rows, uint64_t budget, size_t limit,bool continuous,bool aio,uint64_t packed_row_bytes)
    : impl_(std::make_unique<Impl>(path, offsets, rows, budget, limit,continuous,aio,packed_row_bytes)) {}
EngramReader::EngramReader(const Model& m, uint64_t budget,bool continuous,bool aio)
    : EngramReader(m.tensor(m.ple_engram()?"per_layer_token_embd.weight":"blk.1.engram_embd.weight").file->path(),
        [&]{
            const auto& first=m.tensor(m.ple_engram()?"per_layer_token_embd.weight":"blk.1.engram_embd.weight");
            const auto& second=m.tensor(m.ple_engram()?"per_layer_token_embd.weight":"blk.14.engram_embd.weight");
            if(first.file!=second.file)throw std::runtime_error("Engram tables must share a shard");
            const uint64_t base=first.file->data_start()+first.info->offset;
            return std::array<uint64_t,2>{base,m.ple_engram()?base+384006168*m.engram_row_bytes():second.file->data_start()+second.info->offset};
        }(),
        {384006168,384016682}, budget,4,continuous,aio,m.engram_row_bytes()) {}
EngramReader::~EngramReader() = default;
std::shared_ptr<EngramBatch> EngramReader::submit(EngramQuery query, uint64_t generation, bool split_layers) {
    // Bound each request independently of the queue, including hash/page scratch.
    if (query.image.size() > PREFILL_CHUNK_MAX) throw std::invalid_argument("Engram batch exceeds 32768 tokens");
    auto batch = std::make_shared<EngramBatch>(); batch->generation = generation;
    batch->state = std::make_shared<EngramBatch::State>();
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->room.wait(lock, [&] { return impl_->stop || impl_->queue.size() < impl_->queue_limit; });
    if (impl_->stop) throw std::runtime_error("Engram reader closed");
    impl_->queue.push_back({std::move(query), batch, split_layers}); lock.unlock(); impl_->queued.notify_one(); return batch;
}
EngramReadStats EngramReader::stats() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    auto stats = impl_->counters;
    std::vector<double> values(impl_->latencies.begin(), impl_->latencies.end());
    if (!values.empty()) { std::sort(values.begin(), values.end()); stats.p95_us = values[(values.size() - 1) * 95 / 100]; }
    return stats;
}
void engram_gate_cpu(float* h, const float* kv, const float* qw, const float* kw,
                     const uint8_t* image, int tokens, int width) {
    for (int t = 0; t < tokens; ++t) for (int c = 0; c < 4; ++c) {
        float hh = 0, kk = 0, dot = 0;
        float* x = h + (t * 4 + c) * width;
        const float* key = kv + (t * 5 + c) * width;
        for (int d = 0; d < width; ++d) {
            hh += x[d] * x[d]; kk += key[d] * key[d];
            const float q=x[d]*qw[c*width+d];
            dot += (kw?q*kw[c*width+d]:q)*key[d];
        }
        dot /= std::sqrt((hh / width + 1e-20f) * (kk / width + 1e-20f) * width);
        float input = std::sqrt(std::max(std::abs(dot), 1e-6f)); if (dot < 0) input = -input;
        const float gate = image && image[t] ? 0.f : 1.f / (1.f + std::exp(-input));
        const float* value = kv + (t * 5 + 4) * width;
        for (int d = 0; d < width; ++d) x[d] += gate * value[d];
    }
}
} // namespace strata::models::deepseek41
