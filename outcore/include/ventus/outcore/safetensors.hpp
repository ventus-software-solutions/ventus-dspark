#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ventus::outcore {

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t data_begin{};
    std::uint64_t data_end{};

    [[nodiscard]] std::uint64_t storage_bytes() const;
};

class SafeTensorIndex {
  public:
    static SafeTensorIndex open(const std::filesystem::path& path);

    [[nodiscard]] const std::filesystem::path& path() const;
    [[nodiscard]] std::uint64_t header_bytes() const;
    [[nodiscard]] std::uint64_t data_offset() const;
    [[nodiscard]] std::uint64_t file_bytes() const;
    [[nodiscard]] const std::vector<TensorInfo>& tensors() const;
    [[nodiscard]] std::optional<TensorInfo> find(const std::string& name) const;
    [[nodiscard]] std::uint64_t absolute_offset(const TensorInfo& tensor) const;

  private:
    std::filesystem::path path_;
    std::uint64_t header_bytes_{};
    std::uint64_t file_bytes_{};
    std::vector<TensorInfo> tensors_;
};

}  // namespace ventus::outcore
