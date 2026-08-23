#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

void usage() {
    std::cerr << "Usage: outcore-io-bench --file FILE [--iterations N]\n";
}

}  // namespace

int main(int argc, char** argv) try {
    std::string path;
    std::uint64_t iterations = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--file" && i + 1 < argc) {
            path = argv[++i];
        } else if (argument == "--iterations" && i + 1 < argc) {
            iterations = std::stoull(argv[++i]);
        } else {
            usage();
            return 2;
        }
    }
    if (path.empty() || iterations == 0) {
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
        throw std::runtime_error("direct-I/O benchmark file size must be a non-zero multiple of 4096 bytes");
    }

    void* allocation = nullptr;
    if (posix_memalign(&allocation, kAlignment, kChunkBytes) != 0) {
        close(descriptor);
        throw std::runtime_error("posix_memalign failed");
    }

    const auto start = std::chrono::steady_clock::now();
    std::uint64_t checksum = 0;
    for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
        std::uint64_t offset = 0;
        while (offset < file_bytes) {
            const auto chunk = static_cast<std::size_t>(std::min<std::uint64_t>(kChunkBytes, file_bytes - offset));
            const auto bytes_read = pread(descriptor, allocation, chunk, static_cast<off_t>(offset));
            if (bytes_read != static_cast<ssize_t>(chunk)) {
                free(allocation);
                close(descriptor);
                throw std::runtime_error("short direct read at byte " + std::to_string(offset));
            }
            checksum += static_cast<unsigned char*>(allocation)[iteration % chunk];
            offset += chunk;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    free(allocation);
    close(descriptor);

    const double seconds = std::chrono::duration<double>(stop - start).count();
    const auto total_bytes = file_bytes * iterations;
    const double gib_per_second = (static_cast<double>(total_bytes) / static_cast<double>(1ULL << 30)) / seconds;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "file: " << path << '\n';
    std::cout << "file_bytes: " << file_bytes << '\n';
    std::cout << "iterations: " << iterations << '\n';
    std::cout << "elapsed_seconds: " << seconds << '\n';
    std::cout << "direct_read_gib_per_s: " << gib_per_second << '\n';
    std::cout << "checksum: " << checksum << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "outcore-io-bench: " << error.what() << '\n';
    return 1;
}
