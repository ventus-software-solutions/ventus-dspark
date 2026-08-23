#include "ventus/outcore/byte_cache.hpp"
#include "ventus/outcore/model_store.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage() {
    std::cerr << "Usage: outcore-cache-probe --model ROOT --layer N --expert N [--repeats N]\n";
}

}  // namespace

int main(int argc, char** argv) try {
    std::filesystem::path model_path;
    std::uint32_t layer = 0;
    std::uint32_t expert = 0;
    std::uint32_t repeats = 3;
    bool has_layer = false;
    bool has_expert = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (argument == "--layer" && i + 1 < argc) {
            layer = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            has_layer = true;
        } else if (argument == "--expert" && i + 1 < argc) {
            expert = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            has_expert = true;
        } else if (argument == "--repeats" && i + 1 < argc) {
            repeats = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else {
            usage();
            return 2;
        }
    }
    if (model_path.empty() || !has_layer || !has_expert || repeats == 0) {
        usage();
        return 2;
    }

    ventus::outcore::RepackedModelStore store(model_path);
    const auto location = store.locate_expert(layer, expert);
    ventus::outcore::ByteLruCache cache(64ULL << 20);

    std::cout << "path: " << location.path.string() << '\n';
    std::cout << "offset: " << location.offset << '\n';
    std::cout << "length: " << location.length << '\n';
    std::cout << "tensors: " << location.tensor_count << '\n';
    std::uint64_t checksum = 0;
    for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        const auto data = store.load_expert(layer, expert, cache);
        const auto stop = std::chrono::steady_clock::now();
        checksum += static_cast<std::uint8_t>(data->front());
        const double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "load_" << repeat << "_ms: " << milliseconds << '\n';
    }
    const auto stats = cache.stats();
    std::cout << "hits: " << stats.hits << '\n';
    std::cout << "misses: " << stats.misses << '\n';
    std::cout << "resident_bytes: " << stats.resident_bytes << '\n';
    std::cout << "checksum: " << checksum << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "outcore-cache-probe: " << error.what() << '\n';
    return 1;
}
