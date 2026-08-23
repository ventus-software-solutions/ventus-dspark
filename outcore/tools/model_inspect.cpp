#include "ventus/outcore/safetensors.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ventus::outcore::TensorInfo;

struct ExpertKey {
    std::string block;
    std::uint64_t layer{};
    std::uint64_t expert{};

    bool operator<(const ExpertKey& other) const {
        if (block != other.block) {
            return block < other.block;
        }
        return layer < other.layer || (layer == other.layer && expert < other.expert);
    }
};

std::string category(const std::string& name) {
    if (name.find(".ffn.experts.") != std::string::npos) {
        return "routed_experts";
    }
    if (name.find(".ffn.shared_experts.") != std::string::npos) {
        return "shared_experts";
    }
    if (name.find(".attn.") != std::string::npos) {
        return "attention";
    }
    if (name == "embed.weight" || name == "head.weight") {
        return "embed_and_head";
    }
    if (name.find(".ffn.gate.") != std::string::npos) {
        return "router";
    }
    if (name.find("mtp") != std::string::npos) {
        return "mtp";
    }
    return "other";
}

std::string gib(std::uint64_t bytes) {
    const auto value = static_cast<long double>(bytes) / static_cast<long double>(1ULL << 30);
    std::ostringstream output;
    output << std::fixed << std::setprecision(3) << value;
    return output.str();
}

std::string json_string(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned int>(ch) << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(ch);
                }
        }
    }
    output << '"';
    return output.str();
}

void usage() {
    std::cerr << "Usage: outcore-inspect --model MODEL_DIRECTORY [--manifest OUTPUT.jsonl]\n";
}

}  // namespace

int main(int argc, char** argv) try {
    std::filesystem::path model_path;
    std::filesystem::path manifest_path;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (argument == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else {
            usage();
            return 2;
        }
    }
    if (model_path.empty()) {
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
        throw std::runtime_error("no .safetensors shards found in " + model_path.string());
    }

    std::map<std::string, std::uint64_t> dtype_bytes;
    std::map<std::string, std::uint64_t> category_bytes;
    std::map<ExpertKey, std::uint64_t> expert_bytes;
    std::set<std::string> tensor_names;
    std::uint64_t tensor_count = 0;
    std::uint64_t storage_bytes = 0;
    const std::regex expert_pattern(R"(^(layers|mtp)\.([0-9]+)\.ffn\.experts\.([0-9]+)\.)");
    std::ofstream manifest;
    if (!manifest_path.empty()) {
        manifest.open(manifest_path, std::ios::trunc);
        if (!manifest) {
            throw std::runtime_error("cannot create manifest " + manifest_path.string());
        }
        manifest << "{\"format\":\"ventus-outcore-manifest\",\"version\":1,\"model\":"
                 << json_string(std::filesystem::absolute(model_path).string()) << "}\n";
    }

    for (const auto& shard : shards) {
        const auto index = ventus::outcore::SafeTensorIndex::open(shard);
        for (const TensorInfo& tensor : index.tensors()) {
            if (!tensor_names.insert(tensor.name).second) {
                throw std::runtime_error("duplicate tensor name: " + tensor.name);
            }
            const auto bytes = tensor.storage_bytes();
            ++tensor_count;
            storage_bytes += bytes;
            dtype_bytes[tensor.dtype] += bytes;
            category_bytes[category(tensor.name)] += bytes;

            std::smatch match;
            if (std::regex_search(tensor.name, match, expert_pattern)) {
                const ExpertKey key{match[1].str(), std::stoull(match[2].str()), std::stoull(match[3].str())};
                expert_bytes[key] += bytes;
            }

            if (manifest) {
                manifest << "{\"name\":" << json_string(tensor.name)
                         << ",\"shard\":" << json_string(shard.filename().string())
                         << ",\"offset\":" << index.absolute_offset(tensor)
                         << ",\"length\":" << bytes
                         << ",\"dtype\":" << json_string(tensor.dtype)
                         << ",\"shape\":[";
                for (std::size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
                    if (dimension != 0) {
                        manifest << ',';
                    }
                    manifest << tensor.shape[dimension];
                }
                manifest << "],\"category\":" << json_string(category(tensor.name));
                if (match.size() == 4) {
                    manifest << ",\"block\":" << json_string(match[1].str())
                             << ",\"layer\":" << match[2].str()
                             << ",\"expert\":" << match[3].str();
                }
                manifest << "}\n";
            }
        }
    }

    std::map<std::uint64_t, std::uint64_t> expert_size_counts;
    for (const auto& [key, bytes] : expert_bytes) {
        static_cast<void>(key);
        ++expert_size_counts[bytes];
    }

    std::cout << "model: " << std::filesystem::absolute(model_path).string() << '\n';
    std::cout << "shards: " << shards.size() << '\n';
    std::cout << "tensors: " << tensor_count << '\n';
    std::cout << "tensor_storage_bytes: " << storage_bytes << " (" << gib(storage_bytes) << " GiB)\n";
    std::cout << "routed_expert_bundles: " << expert_bytes.size() << '\n';

    std::cout << "\nbytes_by_category:\n";
    for (const auto& [name, bytes] : category_bytes) {
        std::cout << "  " << name << ": " << bytes << " (" << gib(bytes) << " GiB)\n";
    }
    std::cout << "\nbytes_by_dtype:\n";
    for (const auto& [name, bytes] : dtype_bytes) {
        std::cout << "  " << name << ": " << bytes << " (" << gib(bytes) << " GiB)\n";
    }
    std::cout << "\nexpert_bundle_sizes:\n";
    for (const auto& [bytes, count] : expert_size_counts) {
        std::cout << "  " << bytes << " bytes: " << count << " bundles\n";
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "outcore-inspect: " << error.what() << '\n';
    return 1;
}
