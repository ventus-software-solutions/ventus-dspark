#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr std::size_t kAlignment = 4096;
constexpr std::size_t kChunkBytes = 4 * 1024 * 1024;
constexpr std::uint64_t kIterations = 64;

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void usage() {
    std::cerr << "Usage: cuda-pager-bench --file ALIGNED_BUNDLE\n";
}

}  // namespace

int main(int argc, char** argv) try {
    std::string path;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--file" && i + 1 < argc) {
            path = argv[++i];
        } else {
            usage();
            return 2;
        }
    }
    if (path.empty()) {
        usage();
        return 2;
    }

    const int descriptor = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (descriptor < 0) {
        throw std::runtime_error("open O_DIRECT failed: " + std::string(std::strerror(errno)));
    }
    struct stat file_stat {};
    if (fstat(descriptor, &file_stat) != 0) {
        close(descriptor);
        throw std::runtime_error("fstat failed: " + std::string(std::strerror(errno)));
    }
    const auto file_bytes = static_cast<std::uint64_t>(file_stat.st_size);
    if (file_bytes == 0 || file_bytes % kAlignment != 0) {
        close(descriptor);
        throw std::runtime_error("file size must be a non-zero multiple of 4096 bytes");
    }

    std::array<void*, 2> host{};
    std::array<void*, 2> device{};
    std::array<cudaStream_t, 2> streams{};
    std::array<bool, 2> pending{};
    for (std::size_t slot = 0; slot < host.size(); ++slot) {
        cuda_check(cudaHostAlloc(&host[slot], kChunkBytes, cudaHostAllocDefault), "cudaHostAlloc");
        cuda_check(cudaMalloc(&device[slot], kChunkBytes), "cudaMalloc");
        cuda_check(cudaStreamCreateWithFlags(&streams[slot], cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }

    const auto start = std::chrono::steady_clock::now();
    std::uint64_t chunks = 0;
    std::uint64_t checksum = 0;
    for (std::uint64_t iteration = 0; iteration < kIterations; ++iteration) {
        std::uint64_t offset = 0;
        while (offset < file_bytes) {
            const std::size_t slot = chunks % host.size();
            if (pending[slot]) {
                cuda_check(cudaStreamSynchronize(streams[slot]), "cudaStreamSynchronize reuse");
            }

            const auto chunk = static_cast<std::size_t>(std::min<std::uint64_t>(kChunkBytes, file_bytes - offset));
            const auto bytes_read = pread(descriptor, host[slot], chunk, static_cast<off_t>(offset));
            if (bytes_read != static_cast<ssize_t>(chunk)) {
                throw std::runtime_error("short direct read at byte " + std::to_string(offset));
            }
            checksum += static_cast<unsigned char*>(host[slot])[(iteration + chunks) % chunk];
            cuda_check(
                cudaMemcpyAsync(device[slot], host[slot], chunk, cudaMemcpyHostToDevice, streams[slot]),
                "cudaMemcpyAsync H2D");
            pending[slot] = true;
            ++chunks;
            offset += chunk;
        }
    }
    for (std::size_t slot = 0; slot < streams.size(); ++slot) {
        if (pending[slot]) {
            cuda_check(cudaStreamSynchronize(streams[slot]), "cudaStreamSynchronize final");
        }
    }
    const auto stop = std::chrono::steady_clock::now();

    for (std::size_t slot = 0; slot < host.size(); ++slot) {
        cuda_check(cudaStreamDestroy(streams[slot]), "cudaStreamDestroy");
        cuda_check(cudaFree(device[slot]), "cudaFree");
        cuda_check(cudaFreeHost(host[slot]), "cudaFreeHost");
    }
    close(descriptor);

    const double seconds = std::chrono::duration<double>(stop - start).count();
    const auto total_bytes = file_bytes * kIterations;
    const double gib_per_second = (static_cast<double>(total_bytes) / static_cast<double>(1ULL << 30)) / seconds;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "file: " << path << '\n';
    std::cout << "file_bytes: " << file_bytes << '\n';
    std::cout << "iterations: " << kIterations << '\n';
    std::cout << "pipeline_seconds: " << seconds << '\n';
    std::cout << "pipeline_gib_per_s: " << gib_per_second << '\n';
    std::cout << "average_bundle_ms: " << seconds * 1000.0 / kIterations << '\n';
    std::cout << "checksum: " << checksum << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "cuda-pager-bench: " << error.what() << '\n';
    return 1;
}
