#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ventus::outcore {

using ByteBuffer = std::vector<std::byte>;

struct CacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t insertions{};
    std::uint64_t evictions{};
    std::uint64_t rejected{};
    std::uint64_t resident_bytes{};
    std::uint64_t budget_bytes{};
    std::size_t entries{};
};

class ByteLruCache {
  public:
    explicit ByteLruCache(std::uint64_t budget_bytes);

    [[nodiscard]] std::shared_ptr<const ByteBuffer> get(std::string_view key);
    bool put(std::string key, std::shared_ptr<ByteBuffer> data);
    void set_budget(std::uint64_t budget_bytes);
    void clear();

    [[nodiscard]] CacheStats stats() const;

  private:
    struct Entry {
        std::shared_ptr<ByteBuffer> data;
        std::list<std::string>::iterator recency;
    };

    void evict_to_budget(std::uint64_t incoming_bytes);

    mutable std::mutex mutex_;
    std::uint64_t budget_bytes_{};
    std::uint64_t resident_bytes_{};
    std::uint64_t hits_{};
    std::uint64_t misses_{};
    std::uint64_t insertions_{};
    std::uint64_t evictions_{};
    std::uint64_t rejected_{};
    std::list<std::string> recency_;
    std::unordered_map<std::string, Entry> entries_;
};

}  // namespace ventus::outcore
