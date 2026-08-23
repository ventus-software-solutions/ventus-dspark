#include "ventus/outcore/model_store.hpp"

#include "ventus/outcore/safetensors.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ventus::outcore {

RepackedModelStore::RepackedModelStore(std::filesystem::path root) : root_(std::move(root)) {
    if (!std::filesystem::is_directory(root_)) {
        throw std::invalid_argument("repacked model root is not a directory: " + root_.string());
    }
}

ExpertLocation RepackedModelStore::locate_expert(std::uint32_t layer, std::uint32_t expert) const {
    std::ostringstream filename;
    filename << "layer-" << std::setw(3) << std::setfill('0') << layer << ".experts.safetensors";
    const auto path = root_ / "layers" / filename.str();
    const auto index = SafeTensorIndex::open(path);
    const auto prefix = "layers." + std::to_string(layer) + ".ffn.experts." + std::to_string(expert) + ".";

    std::uint64_t first = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t last = 0;
    std::uint64_t total = 0;
    std::uint32_t count = 0;
    for (const auto& tensor : index.tensors()) {
        if (!tensor.name.starts_with(prefix)) {
            continue;
        }
        const auto begin = index.absolute_offset(tensor);
        const auto end = begin + tensor.storage_bytes();
        first = std::min(first, begin);
        last = std::max(last, end);
        total += tensor.storage_bytes();
        ++count;
    }
    if (count == 0) {
        throw std::runtime_error("expert not found: " + expert_key(layer, expert));
    }
    if (last - first != total) {
        throw std::runtime_error("expert tensors are not contiguous: " + expert_key(layer, expert));
    }
    return ExpertLocation{path, first, total, count};
}

std::shared_ptr<ByteBuffer> RepackedModelStore::read(const ExpertLocation& location) const {
    auto data = std::make_shared<ByteBuffer>(location.length);
    std::ifstream stream(location.path, std::ios::binary);
    stream.seekg(static_cast<std::streamoff>(location.offset));
    stream.read(reinterpret_cast<char*>(data->data()), static_cast<std::streamsize>(data->size()));
    if (!stream) {
        throw std::runtime_error("short expert read from " + location.path.string());
    }
    return data;
}

std::shared_ptr<const ByteBuffer> RepackedModelStore::load_expert(
    std::uint32_t layer,
    std::uint32_t expert,
    ByteLruCache& cache) const {
    const auto key = expert_key(layer, expert);
    if (auto cached = cache.get(key)) {
        return cached;
    }
    auto data = read(locate_expert(layer, expert));
    const std::shared_ptr<const ByteBuffer> result = data;
    static_cast<void>(cache.put(key, std::move(data)));
    return result;
}

std::string RepackedModelStore::expert_key(std::uint32_t layer, std::uint32_t expert) {
    return "layer:" + std::to_string(layer) + ":expert:" + std::to_string(expert);
}

}  // namespace ventus::outcore
