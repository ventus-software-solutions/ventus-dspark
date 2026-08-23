#include "ventus/outcore/safetensors.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void write_u64_le(std::ofstream& stream, std::uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (i * 8)) & 0xff);
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "ventus-outcore-reader-test.safetensors";
    const std::string header =
        R"({"__metadata__":{"format":"pt"},"a":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]},"b":{"dtype":"I8","shape":[3],"data_offsets":[8,11]}}   )";

    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        assert(stream);
        write_u64_le(stream, header.size());
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
        const std::array<unsigned char, 11> payload{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        stream.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    }

    const auto index = ventus::outcore::SafeTensorIndex::open(path);
    assert(index.header_bytes() == header.size());
    assert(index.tensors().size() == 2);
    assert(index.file_bytes() == 8 + header.size() + 11);

    const auto a = index.find("a");
    assert(a.has_value());
    assert(a->dtype == "F16");
    assert(a->shape.size() == 2 && a->shape[0] == 2 && a->shape[1] == 2);
    assert(a->storage_bytes() == 8);
    assert(index.absolute_offset(*a) == 8 + header.size());

    const auto b = index.find("b");
    assert(b.has_value());
    assert(b->storage_bytes() == 3);
    assert(index.absolute_offset(*b) == 8 + header.size() + 8);
    assert(!index.find("missing").has_value());

    std::filesystem::remove(path);
    std::cout << "safetensors reader test passed\n";
    return 0;
}
