#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ventus::outcore {

inline void gpu_cache_cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::uint64_t bytes) : bytes_(bytes) {
        if (bytes == 0) {
            throw std::invalid_argument("device buffer size must be non-zero");
        }
        gpu_cache_cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc device cache entry");
        gpu_cache_cuda_check(cudaEventCreateWithFlags(&last_use_, cudaEventDisableTiming), "cudaEventCreate cache entry");
    }

    ~DeviceBuffer() {
        if (last_use_ != nullptr) {
            static_cast<void>(cudaEventSynchronize(last_use_));
            static_cast<void>(cudaEventDestroy(last_use_));
        }
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] void* data() const { return data_; }
    [[nodiscard]] std::uint64_t bytes() const { return bytes_; }

    void record_use(cudaStream_t stream) {
        gpu_cache_cuda_check(cudaEventRecord(last_use_, stream), "cudaEventRecord cache entry");
    }

    void wait(cudaStream_t stream) const {
        gpu_cache_cuda_check(cudaStreamWaitEvent(stream, last_use_), "cudaStreamWaitEvent cache entry");
    }

  private:
    void* data_{};
    std::uint64_t bytes_{};
    cudaEvent_t last_use_{};
};

struct GpuCacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::uint64_t resident_bytes{};
    std::uint64_t budget_bytes{};
    std::size_t entries{};
};

class GpuByteLruCache {
  public:
    explicit GpuByteLruCache(std::uint64_t budget_bytes) : budget_bytes_(budget_bytes) {
        if (budget_bytes == 0) {
            throw std::invalid_argument("GPU cache budget must be non-zero");
        }
    }

    [[nodiscard]] std::shared_ptr<DeviceBuffer> get(std::string_view key) {
        std::lock_guard lock(mutex_);
        const auto match = entries_.find(std::string(key));
        if (match == entries_.end()) {
            ++misses_;
            return {};
        }
        recency_.splice(recency_.begin(), recency_, match->second.recency);
        ++hits_;
        return match->second.buffer;
    }

    [[nodiscard]] std::shared_ptr<DeviceBuffer> upload(
        std::string key,
        const void* host,
        std::uint64_t bytes,
        cudaStream_t stream) {
        if (auto cached = get(key)) {
            return cached;
        }
        if (bytes > budget_bytes_) {
            throw std::invalid_argument("GPU cache entry exceeds budget");
        }
        auto buffer = std::make_shared<DeviceBuffer>(bytes);
        gpu_cache_cuda_check(cudaMemcpyAsync(buffer->data(), host, bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync GPU cache upload");
        buffer->record_use(stream);

        std::lock_guard lock(mutex_);
        while (!recency_.empty() && resident_bytes_ + bytes > budget_bytes_) {
            const auto& oldest = recency_.back();
            const auto match = entries_.find(oldest);
            resident_bytes_ -= match->second.buffer->bytes();
            entries_.erase(match);
            recency_.pop_back();
            ++evictions_;
        }
        recency_.push_front(key);
        resident_bytes_ += bytes;
        entries_.emplace(std::move(key), Entry{buffer, recency_.begin()});
        return buffer;
    }

    void clear() {
        std::lock_guard lock(mutex_);
        entries_.clear();
        recency_.clear();
        resident_bytes_ = 0;
    }

    [[nodiscard]] GpuCacheStats stats() const {
        std::lock_guard lock(mutex_);
        return GpuCacheStats{hits_, misses_, evictions_, resident_bytes_, budget_bytes_, entries_.size()};
    }

  private:
    struct Entry {
        std::shared_ptr<DeviceBuffer> buffer;
        std::list<std::string>::iterator recency;
    };

    mutable std::mutex mutex_;
    std::uint64_t budget_bytes_{};
    std::uint64_t resident_bytes_{};
    std::uint64_t hits_{};
    std::uint64_t misses_{};
    std::uint64_t evictions_{};
    std::list<std::string> recency_;
    std::unordered_map<std::string, Entry> entries_;
};

}  // namespace ventus::outcore
