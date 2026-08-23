#include "ventus/outcore/gpu_cache.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint64_t entry_bytes = 24ULL << 20;
    constexpr std::uint64_t budget_bytes = 32ULL << 20;
    std::vector<std::byte> host(entry_bytes, std::byte{0x5a});
    cudaStream_t stream{};
    ventus::outcore::gpu_cache_cuda_check(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags test");

    ventus::outcore::GpuByteLruCache cache(budget_bytes);
    auto first = cache.upload("expert:1", host.data(), host.size(), stream);
    first->wait(stream);
    ventus::outcore::gpu_cache_cuda_check(cudaMemsetAsync(first->data(), 0x11, first->bytes(), stream), "cudaMemsetAsync first");
    first->record_use(stream);

    auto second = cache.upload("expert:2", host.data(), host.size(), stream);
    second->wait(stream);
    assert(!cache.get("expert:1"));
    assert(cache.get("expert:2"));
    assert(first->bytes() == entry_bytes);

    ventus::outcore::gpu_cache_cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize test");
    auto stats = cache.stats();
    assert(stats.entries == 1);
    assert(stats.resident_bytes == entry_bytes);
    assert(stats.evictions == 1);
    assert(stats.hits == 1);
    assert(stats.misses == 3);

    cache.clear();
    assert(first->bytes() == entry_bytes);
    first.reset();
    second.reset();
    ventus::outcore::gpu_cache_cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy test");
    std::cout << "gpu cache test passed\n";
    return 0;
}
