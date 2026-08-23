#include "ventus/outcore/byte_cache.hpp"

#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

namespace {

std::shared_ptr<ventus::outcore::ByteBuffer> bytes(std::size_t size) {
    return std::make_shared<ventus::outcore::ByteBuffer>(size);
}

}  // namespace

int main() {
    ventus::outcore::ByteLruCache cache(100);
    assert(cache.put("a", bytes(40)));
    assert(cache.put("b", bytes(40)));
    assert(cache.get("a"));
    assert(cache.put("c", bytes(40)));
    assert(!cache.get("b"));
    assert(cache.get("a"));
    assert(cache.get("c"));

    auto stats = cache.stats();
    assert(stats.resident_bytes == 80);
    assert(stats.entries == 2);
    assert(stats.evictions == 1);
    assert(stats.hits == 3);
    assert(stats.misses == 1);

    assert(!cache.put("too-large", bytes(101)));
    stats = cache.stats();
    assert(stats.rejected == 1);

    cache.set_budget(40);
    stats = cache.stats();
    assert(stats.resident_bytes == 40);
    assert(stats.entries == 1);

    const auto borrowed = cache.get("c");
    assert(borrowed);
    cache.clear();
    assert(borrowed->size() == 40);
    assert(cache.stats().resident_bytes == 0);

    std::cout << "byte cache test passed\n";
    return 0;
}
