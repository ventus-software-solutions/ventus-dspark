#pragma once

#include "ventus/outcore/byte_cache.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace ventus::outcore {

struct ExpertLocation {
    std::filesystem::path path;
    std::uint64_t offset{};
    std::uint64_t length{};
    std::uint32_t tensor_count{};
};

class RepackedModelStore {
  public:
    explicit RepackedModelStore(std::filesystem::path root);

    [[nodiscard]] ExpertLocation locate_expert(std::uint32_t layer, std::uint32_t expert) const;
    [[nodiscard]] std::shared_ptr<ByteBuffer> read(const ExpertLocation& location) const;
    [[nodiscard]] std::shared_ptr<const ByteBuffer> load_expert(
        std::uint32_t layer,
        std::uint32_t expert,
        ByteLruCache& cache) const;

    [[nodiscard]] static std::string expert_key(std::uint32_t layer, std::uint32_t expert);

  private:
    std::filesystem::path root_;
};

}  // namespace ventus::outcore
