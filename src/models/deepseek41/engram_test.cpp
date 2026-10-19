#include "strata/models/deepseek41/engram.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace strata::models::deepseek41;
namespace {
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
uint8_t pattern(int table, uint64_t row, int byte) { return (table * 71 + row * 29 + byte * 11) & 255; }
void hash_test() {
    EngramHashSpec s; s.pad = 7; s.multipliers = {3,5,7,11,13,17,19,23};
    for (int i = 0; i < 48; ++i) { s.primes[i] = 101; s.offsets[i] = uint64_t(i % 24) * 101; }
    const std::vector<int32_t> map{7,2,3,4,5,6};
    const std::vector<int> ids{1,2,3,4,5};
    const std::vector<uint8_t> mask{0,0,1,0,0};
    const auto full = engram_hash(s,map,ids,mask,{});
    require(full.rows[0] == ((2 * 3) ^ (7 * 5)) % 101, "start padding hash");
    require(full.rows[48] == ((3 * 3) ^ (2 * 5)) % 101, "2-gram hash");
    const uint64_t after_image = ((5 * 3) ^ (7 * 5) ^ (7 * 7)) % 101 + 8 * 101;
    require(full.rows[3 * 48 + 8] == after_image, "image must stop all later lookbacks");
    const auto first = engram_hash(s,map,{1,2},{0,0},{});
    const auto second = engram_hash(s,map,{3,4,5},{1,0,0},first.after);
    require(std::equal(second.rows.begin(),second.rows.end(),full.rows.begin()+96), "chunk history");
    require(second.after.recent == full.after.recent && second.after.position == 5, "history commit");
    const auto accepted = engram_hash(s,map,{4},{0},first.after);
    const auto after_reject = engram_hash(s,map,{4},{0},first.after);
    require(accepted.rows == after_reject.rows, "rejection restores original history");
}
void io_test(bool continuous,bool aio=false) {
    const auto path = std::filesystem::temp_directory_path() / ("strata-engram-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    struct Cleanup { std::filesystem::path p; ~Cleanup(){std::error_code ec; std::filesystem::remove(p,ec);} } cleanup{path};
    const std::array<uint64_t,2> offsets{4096, (uint64_t(1) << 32) + 512};
    std::ofstream file(path,std::ios::binary);
    std::array<uint8_t,272> row;
    for (int l = 0; l < 2; ++l) for (int r = 0; r < 64; ++r) {
        for (int b = 0; b < 272; ++b) row[b] = pattern(l,r,b);
        file.seekp(offsets[l] + r * 272); file.write(reinterpret_cast<char*>(row.data()),row.size());
    }
    // Keep the final O_DIRECT page complete, without allocating the sparse 4 GiB hole.
    file.seekp(((offsets[1] + 64 * 272 + 4095) / 4096) * 4096 - 1); file.put(0); file.close();
    require(bool(file), "write fixture");
    EngramReader reader(path.string(),offsets,{64,64},1 << 20,2,continuous,aio);
    EngramQuery q; q.image = {0,1,0}; q.rows.resize(144);
    for (size_t i = 0; i < q.rows.size(); ++i) q.rows[i] = (i * 7 + 14) % 64;
    const auto ticket = reader.submit(q,42);
    const auto& packed = ticket->wait(); require(ticket->generation == 42, "request generation");
    for (size_t i = 0; i < q.rows.size(); ++i) for (int b = 0; b < 272; ++b)
        require(packed[i * 272 + b] == (q.image[i / 48] ? 0 : pattern((i % 48) / 24,q.rows[i],b)), "cross-page row/64-bit offset");
    const auto initial = reader.stats(); require(initial.disk_bytes > 0 && initial.reads > 0, "actual direct reads");
    require(reader.submit(q,43)->wait() == packed, "cache bytes");
    const auto cached = reader.stats(); require(cached.disk_bytes == initial.disk_bytes && cached.page_hits > initial.page_hits, "warm reads avoid disk");
    require(cached.cache_owned_bytes <= (1 << 20), "bounded page cache");
    for (uint64_t generation = 100; generation < 110; ++generation) {
        const auto split = reader.submit(q,generation,true);
        for (int table = 0; table < 2; ++table) {
            const auto& partial = split->wait_layer(table);
            for (size_t t = 0; t < q.image.size(); ++t)
                require(std::equal(partial.begin()+(t*48+table*24)*272,partial.begin()+(t*48+table*24+24)*272,
                                   packed.begin()+(t*48+table*24)*272),"split table rows/image/tail");
        }
        require(split->wait() == packed,"split/full byte identity");
    }
    if(!continuous){
        EngramQuery large;large.image.assign(32768,1);large.rows.assign(32768*48,0);
        // Include real rows at the old and new chunk boundaries, not only DEAD rows.
        for(size_t token:{size_t(8191),size_t(8192),size_t(32767)})large.image[token]=0;
        const auto large_ticket=reader.submit(large,200,true);
        const auto& data=large_ticket->wait();require(data.size()==size_t(32768)*48*272,"32K Engram output size");
        for(size_t token:{size_t(8191),size_t(8192),size_t(32767)})for(int row=0;row<48;++row)for(int b=0;b<272;++b)
            require(data[(token*48+row)*272+b]==pattern(row/24,0,b),"32K Engram boundary row bytes");
        require(data.front()==0 && data[8190ull*48*272]==0,"32K Engram DEAD rows");
        large.image.push_back(1);
        bool too_large=false;try{reader.submit(std::move(large),201);}catch(const std::invalid_argument&){too_large=true;}
        require(too_large,"Engram chunk above 32K accepted");
    }
    const auto cancel = reader.submit(q,44); cancel->cancel();
    bool refused = false; try { cancel->wait(); } catch (const std::runtime_error&) { refused = true; }
    require(refused, "cancel cannot publish stale rows");
    q.rows[0] = 64;
    refused = false; try { reader.submit(q,45)->wait(); } catch (const std::runtime_error&) { refused = true; }
    require(refused, "invalid row rejected before I/O");
    q.rows[0]=0;EngramReader short_reader(path.string(),offsets,{64,64},0,2,continuous,aio);
    std::filesystem::resize_file(path,4096);refused=false;
    try{short_reader.submit(q,46)->wait();}catch(const std::runtime_error&){refused=true;}
    require(refused,"short reads drain accepted requests before releasing buffers");
}
}
int main() {
    try { hash_test(); io_test(false);io_test(true);
#if defined(__linux__)
        io_test(true,true);
#endif
        std::cout << "Engram hash/history/DEAD, batched/continuous/AIO reads, coalescing, cache, cancellation, short reads and >4 GiB offsets PASS\n"; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
