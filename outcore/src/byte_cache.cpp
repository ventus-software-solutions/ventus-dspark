#include "ventus/outcore/byte_cache.hpp"

#include <stdexcept>

namespace ventus::outcore {

ByteLruCache::ByteLruCache(std::uint64_t budget_bytes) : budget_bytes_(budget_bytes) {
    if (budget_bytes == 0) {
        throw std::invalid_argument("cache budget must be non-zero");
    }
}

std::shared_ptr<const ByteBuffer> ByteLruCache::get(std::string_view key) {
    std::lock_guard lock(mutex_);
    const auto match = entries_.find(std::string(key));
    if (match == entries_.end()) {
        ++misses_;
        return {};
    }
    recency_.splice(recency_.begin(), recency_, match->second.recency);
    ++hits_;
    return match->second.data;
}

bool ByteLruCache::put(std::string key, std::shared_ptr<ByteBuffer> data) {
    if (!data) {
        throw std::invalid_argument("cache data must not be null");
    }
    std::lock_guard lock(mutex_);
    const auto bytes = static_cast<std::uint64_t>(data->size());
    if (bytes > budget_bytes_) {
        ++rejected_;
        return false;
    }

    const auto existing = entries_.find(key);
    if (existing != entries_.end()) {
        resident_bytes_ -= existing->second.data->size();
        recency_.erase(existing->second.recency);
        entries_.erase(existing);
    }
    evict_to_budget(bytes);
    recency_.push_front(key);
    resident_bytes_ += bytes;
    entries_.emplace(std::move(key), Entry{std::move(data), recency_.begin()});
    ++insertions_;
    return true;
}

void ByteLruCache::set_budget(std::uint64_t budget_bytes) {
    if (budget_bytes == 0) {
        throw std::invalid_argument("cache budget must be non-zero");
    }
    std::lock_guard lock(mutex_);
    budget_bytes_ = budget_bytes;
    evict_to_budget(0);
}

void ByteLruCache::clear() {
    std::lock_guard lock(mutex_);
    entries_.clear();
    recency_.clear();
    resident_bytes_ = 0;
}

CacheStats ByteLruCache::stats() const {
    std::lock_guard lock(mutex_);
    return CacheStats{
        hits_,
        misses_,
        insertions_,
        evictions_,
        rejected_,
        resident_bytes_,
        budget_bytes_,
        entries_.size(),
    };
}

void ByteLruCache::evict_to_budget(std::uint64_t incoming_bytes) {
    while (!recency_.empty() && resident_bytes_ + incoming_bytes > budget_bytes_) {
        const auto& key = recency_.back();
        const auto match = entries_.find(key);
        resident_bytes_ -= match->second.data->size();
        entries_.erase(match);
        recency_.pop_back();
        ++evictions_;
    }
}

}  // namespace ventus::outcore
