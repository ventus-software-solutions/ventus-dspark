#include "ventus/outcore/safetensors.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kAlignment = 4096;
constexpr std::size_t kCopyBufferBytes = 4 * 1024 * 1024;

struct SourceTensor {
    std::filesystem::path shard;
    ventus::outcore::TensorInfo tensor;
    std::uint64_t absolute_offset{};
};

void write_u64_le(std::ostream& stream, std::uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (i * 8)) & 0xff);
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::string build_header(const std::vector<SourceTensor>& tensors) {
    std::ostringstream json;
    json << '{';
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i].tensor;
        if (i != 0) {
            json << ',';
        }
        json << '"' << tensor.name << "\":{\"dtype\":\"" << tensor.dtype << "\",\"shape\":[";
        for (std::size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
            if (dimension != 0) {
                json << ',';
            }
            json << tensor.shape[dimension];
        }
        json << "],\"data_offsets\":[" << offset << ',' << offset + tensor.storage_bytes() << "]}";
        offset += tensor.storage_bytes();
    }
    json << '}';

    std::string header = json.str();
    const auto data_offset = 8 + header.size();
    const auto padding = (kAlignment - (data_offset % kAlignment)) % kAlignment;
    header.append(padding, ' ');
    return header;
}

void copy_range(
    std::istream& source,
    std::ostream& destination,
    std::uint64_t offset,
    std::uint64_t length,
    std::vector<char>& buffer) {
    source.clear();
    source.seekg(static_cast<std::streamoff>(offset));
    if (!source) {
        throw std::runtime_error("cannot seek source tensor");
    }
    while (length != 0) {
        const auto chunk = static_cast<std::streamsize>(std::min<std::uint64_t>(length, buffer.size()));
        source.read(buffer.data(), chunk);
        if (source.gcount() != chunk) {
            throw std::runtime_error("short source tensor read");
        }
        destination.write(buffer.data(), chunk);
        if (!destination) {
            throw std::runtime_error("short bundle write");
        }
        length -= static_cast<std::uint64_t>(chunk);
    }
}

void verify_bundle(const std::filesystem::path& output, const std::vector<SourceTensor>& tensors) {
    const auto packed = ventus::outcore::SafeTensorIndex::open(output);
    if (packed.tensors().size() != tensors.size()) {
        throw std::runtime_error("verification tensor count mismatch");
    }

    std::ifstream packed_stream(output, std::ios::binary);
    std::vector<char> expected(kCopyBufferBytes);
    std::vector<char> actual(kCopyBufferBytes);
    for (const auto& source_tensor : tensors) {
        const auto packed_tensor = packed.find(source_tensor.tensor.name);
        if (!packed_tensor || packed_tensor->dtype != source_tensor.tensor.dtype ||
            packed_tensor->shape != source_tensor.tensor.shape ||
            packed_tensor->storage_bytes() != source_tensor.tensor.storage_bytes()) {
            throw std::runtime_error("verification metadata mismatch for " + source_tensor.tensor.name);
        }

        std::ifstream source_stream(source_tensor.shard, std::ios::binary);
        source_stream.seekg(static_cast<std::streamoff>(source_tensor.absolute_offset));
        packed_stream.seekg(static_cast<std::streamoff>(packed.absolute_offset(*packed_tensor)));
        std::uint64_t remaining = source_tensor.tensor.storage_bytes();
        while (remaining != 0) {
            const auto chunk = static_cast<std::streamsize>(std::min<std::uint64_t>(remaining, expected.size()));
            source_stream.read(expected.data(), chunk);
            packed_stream.read(actual.data(), chunk);
            if (source_stream.gcount() != chunk || packed_stream.gcount() != chunk ||
                !std::equal(expected.begin(), expected.begin() + chunk, actual.begin())) {
                throw std::runtime_error("verification byte mismatch for " + source_tensor.tensor.name);
            }
            remaining -= static_cast<std::uint64_t>(chunk);
        }
    }
}

void usage() {
    std::cerr << "Usage: outcore-extract --model MODEL_DIRECTORY --prefix TENSOR_PREFIX --output FILE\n";
}

}  // namespace

int main(int argc, char** argv) try {
    std::filesystem::path model_path;
    std::filesystem::path output_path;
    std::string prefix;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (argument == "--prefix" && i + 1 < argc) {
            prefix = argv[++i];
        } else if (argument == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            usage();
            return 2;
        }
    }
    if (model_path.empty() || prefix.empty() || output_path.empty()) {
        usage();
        return 2;
    }

    std::vector<std::filesystem::path> shards;
    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            shards.push_back(entry.path());
        }
    }
    std::sort(shards.begin(), shards.end());

    std::vector<SourceTensor> tensors;
    for (const auto& shard : shards) {
        const auto index = ventus::outcore::SafeTensorIndex::open(shard);
        for (const auto& tensor : index.tensors()) {
            if (tensor.name.starts_with(prefix)) {
                tensors.push_back(SourceTensor{shard, tensor, index.absolute_offset(tensor)});
            }
        }
    }
    std::sort(tensors.begin(), tensors.end(), [](const SourceTensor& left, const SourceTensor& right) {
        return left.tensor.name < right.tensor.name;
    });
    if (tensors.empty()) {
        throw std::runtime_error("no tensors match prefix " + prefix);
    }

    std::filesystem::create_directories(output_path.parent_path());
    const auto header = build_header(tensors);
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create " + output_path.string());
    }
    write_u64_le(output, header.size());
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    std::vector<char> buffer(kCopyBufferBytes);
    std::filesystem::path current_shard;
    std::ifstream source;
    std::uint64_t payload_bytes = 0;
    for (const auto& tensor : tensors) {
        if (tensor.shard != current_shard) {
            source.close();
            source.open(tensor.shard, std::ios::binary);
            if (!source) {
                throw std::runtime_error("cannot open " + tensor.shard.string());
            }
            current_shard = tensor.shard;
        }
        copy_range(source, output, tensor.absolute_offset, tensor.tensor.storage_bytes(), buffer);
        payload_bytes += tensor.tensor.storage_bytes();
    }
    output.close();

    verify_bundle(output_path, tensors);
    std::cout << "output: " << output_path.string() << '\n';
    std::cout << "tensors: " << tensors.size() << '\n';
    std::cout << "header_bytes: " << header.size() << '\n';
    std::cout << "payload_bytes: " << payload_bytes << '\n';
    std::cout << "verified: byte_exact\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "outcore-extract: " << error.what() << '\n';
    return 1;
}
