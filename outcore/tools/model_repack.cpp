#include "ventus/outcore/safetensors.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr std::uint64_t kAlignment = 4096;
constexpr std::size_t kCopyBufferBytes = 8 * 1024 * 1024;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

struct SourceTensor {
    std::filesystem::path shard;
    ventus::outcore::TensorInfo tensor;
    std::uint64_t source_offset{};
    std::uint64_t destination_offset{};
    std::uint64_t checksum{kFnvOffset};
};

using Groups = std::map<std::filesystem::path, std::vector<SourceTensor>>;

void write_u64_le(std::ostream& stream, std::uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (i * 8)) & 0xff);
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::uint64_t update_fnv(std::uint64_t hash, const char* bytes, std::size_t length) {
    for (std::size_t i = 0; i < length; ++i) {
        hash ^= static_cast<unsigned char>(bytes[i]);
        hash *= kFnvPrime;
    }
    return hash;
}

std::tuple<std::uint64_t, int, int, std::string> expert_order(const std::string& name) {
    static const std::regex pattern(R"(\.experts\.([0-9]+)\.(w[123])\.(weight|scale)$)");
    std::smatch match;
    if (!std::regex_search(name, match, pattern)) {
        return {0, 0, 0, name};
    }
    int projection = 0;
    if (match[2] == "w3") {
        projection = 1;
    } else if (match[2] == "w2") {
        projection = 2;
    }
    const int kind = match[3] == "weight" ? 0 : 1;
    return {std::stoull(match[1].str()), projection, kind, name};
}

std::filesystem::path group_path(const std::string& name) {
    static const std::regex layer_pattern(R"(^layers\.([0-9]+)\.)");
    static const std::regex mtp_pattern(R"(^mtp\.([0-9]+)\.)");
    std::smatch match;
    if (std::regex_search(name, match, layer_pattern)) {
        std::ostringstream filename;
        filename << "layer-" << std::setw(3) << std::setfill('0') << std::stoull(match[1].str());
        filename << (name.find(".ffn.experts.") != std::string::npos ? ".experts.safetensors" : ".dense.safetensors");
        return std::filesystem::path("layers") / filename.str();
    }
    if (std::regex_search(name, match, mtp_pattern)) {
        std::ostringstream filename;
        filename << "mtp-" << std::setw(3) << std::setfill('0') << std::stoull(match[1].str());
        filename << (name.find(".ffn.experts.") != std::string::npos ? ".experts.safetensors" : ".dense.safetensors");
        return std::filesystem::path("mtp") / filename.str();
    }
    return "global.safetensors";
}

std::string build_header(std::vector<SourceTensor>& tensors, std::uint64_t& payload_bytes) {
    std::ostringstream json;
    json << '{';
    payload_bytes = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        auto& tensor = tensors[i];
        tensor.destination_offset = payload_bytes;
        if (i != 0) {
            json << ',';
        }
        json << '"' << tensor.tensor.name << "\":{\"dtype\":\"" << tensor.tensor.dtype << "\",\"shape\":[";
        for (std::size_t dimension = 0; dimension < tensor.tensor.shape.size(); ++dimension) {
            if (dimension != 0) {
                json << ',';
            }
            json << tensor.tensor.shape[dimension];
        }
        json << "],\"data_offsets\":[" << payload_bytes << ','
             << payload_bytes + tensor.tensor.storage_bytes() << "]}";
        payload_bytes += tensor.tensor.storage_bytes();
    }
    json << '}';

    std::string header = json.str();
    const auto padding = (kAlignment - ((8 + header.size()) % kAlignment)) % kAlignment;
    header.append(padding, ' ');
    return header;
}

void copy_tensor(
    std::ifstream& source,
    std::fstream& destination,
    SourceTensor& tensor,
    std::uint64_t data_offset,
    std::vector<char>& buffer) {
    source.clear();
    source.seekg(static_cast<std::streamoff>(tensor.source_offset));
    destination.clear();
    destination.seekp(static_cast<std::streamoff>(data_offset + tensor.destination_offset));
    if (!source || !destination) {
        throw std::runtime_error("seek failed for " + tensor.tensor.name);
    }

    std::uint64_t remaining = tensor.tensor.storage_bytes();
    while (remaining != 0) {
        const auto chunk = static_cast<std::streamsize>(std::min<std::uint64_t>(remaining, buffer.size()));
        source.read(buffer.data(), chunk);
        if (source.gcount() != chunk) {
            throw std::runtime_error("short source read for " + tensor.tensor.name);
        }
        destination.write(buffer.data(), chunk);
        if (!destination) {
            throw std::runtime_error("short destination write for " + tensor.tensor.name);
        }
        tensor.checksum = update_fnv(tensor.checksum, buffer.data(), static_cast<std::size_t>(chunk));
        remaining -= static_cast<std::uint64_t>(chunk);
    }
}

void verify_segment(const std::filesystem::path& path, const std::vector<SourceTensor>& tensors) {
    const auto index = ventus::outcore::SafeTensorIndex::open(path);
    if (index.tensors().size() != tensors.size()) {
        throw std::runtime_error("verification tensor count mismatch for " + path.string());
    }

    std::ifstream stream(path, std::ios::binary);
    std::vector<char> buffer(kCopyBufferBytes);
    for (const auto& source_tensor : tensors) {
        const auto packed = index.find(source_tensor.tensor.name);
        if (!packed || packed->dtype != source_tensor.tensor.dtype || packed->shape != source_tensor.tensor.shape ||
            packed->storage_bytes() != source_tensor.tensor.storage_bytes()) {
            throw std::runtime_error("verification metadata mismatch for " + source_tensor.tensor.name);
        }
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(index.absolute_offset(*packed)));
        std::uint64_t hash = kFnvOffset;
        std::uint64_t remaining = packed->storage_bytes();
        while (remaining != 0) {
            const auto chunk = static_cast<std::streamsize>(std::min<std::uint64_t>(remaining, buffer.size()));
            stream.read(buffer.data(), chunk);
            if (stream.gcount() != chunk) {
                throw std::runtime_error("short verification read for " + source_tensor.tensor.name);
            }
            hash = update_fnv(hash, buffer.data(), static_cast<std::size_t>(chunk));
            remaining -= static_cast<std::uint64_t>(chunk);
        }
        if (hash != source_tensor.checksum) {
            throw std::runtime_error("verification checksum mismatch for " + source_tensor.tensor.name);
        }
    }
}

void write_segment(const std::filesystem::path& path, std::vector<SourceTensor>& tensors) {
    if (std::filesystem::exists(path)) {
        const auto existing = ventus::outcore::SafeTensorIndex::open(path);
        if (existing.tensors().size() == tensors.size()) {
            std::cout << "skip_complete: " << path.string() << '\n';
            return;
        }
        throw std::runtime_error("existing segment is incompatible: " + path.string());
    }

    const bool experts = path.filename().string().find(".experts.") != std::string::npos;
    if (experts) {
        std::sort(tensors.begin(), tensors.end(), [](const SourceTensor& left, const SourceTensor& right) {
            return expert_order(left.tensor.name) < expert_order(right.tensor.name);
        });
    } else {
        std::sort(tensors.begin(), tensors.end(), [](const SourceTensor& left, const SourceTensor& right) {
            return left.tensor.name < right.tensor.name;
        });
    }

    std::uint64_t payload_bytes = 0;
    const auto header = build_header(tensors, payload_bytes);
    const auto data_offset = 8 + header.size();
    std::filesystem::create_directories(path.parent_path());
    auto temporary = path;
    temporary += ".tmp";
    std::filesystem::remove(temporary);

    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot create " + temporary.string());
        }
        write_u64_le(output, header.size());
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
        output.seekp(static_cast<std::streamoff>(data_offset + payload_bytes - 1));
        output.put('\0');
    }

    std::vector<std::size_t> source_order(tensors.size());
    for (std::size_t i = 0; i < source_order.size(); ++i) {
        source_order[i] = i;
    }
    std::sort(source_order.begin(), source_order.end(), [&](std::size_t left, std::size_t right) {
        return std::tie(tensors[left].shard, tensors[left].source_offset) <
               std::tie(tensors[right].shard, tensors[right].source_offset);
    });

    std::fstream destination(temporary, std::ios::binary | std::ios::in | std::ios::out);
    std::ifstream source;
    std::filesystem::path current_shard;
    std::vector<char> buffer(kCopyBufferBytes);
    std::uint64_t copied_bytes = 0;
    std::uint64_t next_progress = 256ULL << 20;
    const auto copy_start = std::chrono::steady_clock::now();
    for (const auto index : source_order) {
        auto& tensor = tensors[index];
        if (tensor.shard != current_shard) {
            source.close();
            source.open(tensor.shard, std::ios::binary);
            if (!source) {
                throw std::runtime_error("cannot open " + tensor.shard.string());
            }
            current_shard = tensor.shard;
        }
        copy_tensor(source, destination, tensor, data_offset, buffer);
        copied_bytes += tensor.tensor.storage_bytes();
        if (copied_bytes >= next_progress) {
            std::cout << "copy_progress: " << copied_bytes << '/' << payload_bytes << '\n';
            next_progress += 256ULL << 20;
        }
    }
    destination.flush();
    destination.close();
    const auto copy_stop = std::chrono::steady_clock::now();

    const auto verify_start = std::chrono::steady_clock::now();
    verify_segment(temporary, tensors);
    const auto verify_stop = std::chrono::steady_clock::now();
    std::filesystem::rename(temporary, path);

    const double copy_seconds = std::chrono::duration<double>(copy_stop - copy_start).count();
    const double verify_seconds = std::chrono::duration<double>(verify_stop - verify_start).count();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "segment_complete: " << path.string() << '\n';
    std::cout << "tensors: " << tensors.size() << '\n';
    std::cout << "payload_bytes: " << payload_bytes << '\n';
    std::cout << "copy_seconds: " << copy_seconds << '\n';
    std::cout << "verify_seconds: " << verify_seconds << '\n';
}

void usage() {
    std::cerr << "Usage: outcore-repack --model MODEL --output DIRECTORY (--layer N | --all)\n";
}

void finalize_store(
    const std::filesystem::path& source,
    const std::filesystem::path& output,
    const Groups& groups) {
    for (const auto& entry : std::filesystem::directory_iterator(source)) {
        if (!entry.is_regular_file() || entry.path().extension() == ".safetensors" ||
            entry.path().filename() == "model.safetensors.index.json") {
            continue;
        }
        std::filesystem::copy_file(
            entry.path(),
            output / entry.path().filename(),
            std::filesystem::copy_options::overwrite_existing);
    }

    std::uint64_t tensor_count = 0;
    std::uint64_t tensor_bytes = 0;
    for (const auto& [path, tensors] : groups) {
        static_cast<void>(path);
        tensor_count += tensors.size();
        for (const auto& tensor : tensors) {
            tensor_bytes += tensor.tensor.storage_bytes();
        }
    }

    const auto marker = output / "outcore-store.json";
    auto temporary = marker;
    temporary += ".tmp";
    std::ofstream stream(temporary, std::ios::trunc);
    stream << "{\n"
           << "  \"format\": \"ventus-outcore-store\",\n"
           << "  \"version\": 1,\n"
           << "  \"segments\": " << groups.size() << ",\n"
           << "  \"tensors\": " << tensor_count << ",\n"
           << "  \"tensor_bytes\": " << tensor_bytes << ",\n"
           << "  \"complete\": true\n"
           << "}\n";
    stream.close();
    std::filesystem::remove(marker);
    std::filesystem::rename(temporary, marker);
    std::cout << "store_complete: " << marker.string() << '\n';
}

}  // namespace

int main(int argc, char** argv) try {
    std::filesystem::path model_path;
    std::filesystem::path output_path;
    int selected_layer = -1;
    bool all = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (argument == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (argument == "--layer" && i + 1 < argc) {
            selected_layer = std::stoi(argv[++i]);
        } else if (argument == "--all") {
            all = true;
        } else {
            usage();
            return 2;
        }
    }
    if (model_path.empty() || output_path.empty() || (all == (selected_layer >= 0))) {
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
    if (shards.empty()) {
        throw std::runtime_error("no safetensors shards found");
    }

    Groups groups;
    const std::string layer_prefix = selected_layer >= 0 ? "layers." + std::to_string(selected_layer) + "." : "";
    for (const auto& shard : shards) {
        const auto shard_index = ventus::outcore::SafeTensorIndex::open(shard);
        for (const auto& tensor : shard_index.tensors()) {
            if (!all && !tensor.name.starts_with(layer_prefix)) {
                continue;
            }
            groups[group_path(tensor.name)].push_back(
                SourceTensor{shard, tensor, shard_index.absolute_offset(tensor)});
        }
    }
    if (groups.empty()) {
        throw std::runtime_error("no tensors selected");
    }

    std::cout << "segments_planned: " << groups.size() << '\n';
    for (auto& [relative_path, tensors] : groups) {
        write_segment(output_path / relative_path, tensors);
    }
    if (all) {
        finalize_store(model_path, output_path, groups);
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "outcore-repack: " << error.what() << '\n';
    return 1;
}
