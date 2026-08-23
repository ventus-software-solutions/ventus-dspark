#include "ventus/outcore/safetensors.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ventus::outcore {
namespace {

constexpr std::uint64_t kLengthPrefixBytes = 8;
constexpr std::uint64_t kMaxHeaderBytes = 1ULL << 30;

class HeaderParser {
  public:
    explicit HeaderParser(std::string_view input) : input_(input) {}

    std::vector<TensorInfo> parse() {
        std::vector<TensorInfo> tensors;
        expect('{');
        skip_space();
        if (consume('}')) {
            return tensors;
        }

        while (true) {
            const auto name = parse_string();
            expect(':');
            if (name == "__metadata__") {
                skip_value();
            } else {
                tensors.push_back(parse_tensor(name));
            }

            skip_space();
            if (consume('}')) {
                break;
            }
            expect(',');
        }

        skip_space();
        if (position_ != input_.size()) {
            fail("unexpected bytes after top-level object");
        }
        return tensors;
    }

  private:
    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("safetensors header byte " + std::to_string(position_) + ": " + message);
    }

    void skip_space() {
        while (position_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
    }

    bool consume(char expected) {
        skip_space();
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    std::string parse_string() {
        skip_space();
        if (position_ >= input_.size() || input_[position_] != '"') {
            fail("expected string");
        }
        ++position_;

        std::string result;
        while (position_ < input_.size()) {
            const char ch = input_[position_++];
            if (ch == '"') {
                return result;
            }
            if (ch != '\\') {
                result.push_back(ch);
                continue;
            }
            if (position_ >= input_.size()) {
                fail("unterminated string escape");
            }
            const char escaped = input_[position_++];
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u':
                    // Safetensors tensor names and keys are ASCII. Validate and preserve
                    // uncommon unicode escapes as a placeholder rather than misaligning.
                    for (int i = 0; i < 4; ++i) {
                        if (position_ >= input_.size() || !std::isxdigit(static_cast<unsigned char>(input_[position_]))) {
                            fail("invalid unicode escape");
                        }
                        ++position_;
                    }
                    result.push_back('?');
                    break;
                default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }

    std::uint64_t parse_u64() {
        skip_space();
        if (position_ >= input_.size() || !std::isdigit(static_cast<unsigned char>(input_[position_]))) {
            fail("expected unsigned integer");
        }
        std::uint64_t value = 0;
        while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
            const auto digit = static_cast<std::uint64_t>(input_[position_] - '0');
            if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
                fail("integer overflow");
            }
            value = value * 10 + digit;
            ++position_;
        }
        return value;
    }

    std::vector<std::uint64_t> parse_u64_array() {
        std::vector<std::uint64_t> values;
        expect('[');
        skip_space();
        if (consume(']')) {
            return values;
        }
        while (true) {
            values.push_back(parse_u64());
            if (consume(']')) {
                return values;
            }
            expect(',');
        }
    }

    void skip_literal() {
        skip_space();
        const auto start = position_;
        while (position_ < input_.size()) {
            const char ch = input_[position_];
            if (std::isspace(static_cast<unsigned char>(ch)) || ch == ',' || ch == ']' || ch == '}') {
                break;
            }
            ++position_;
        }
        if (position_ == start) {
            fail("expected value");
        }
    }

    void skip_value() {
        skip_space();
        if (position_ >= input_.size()) {
            fail("expected value");
        }
        if (input_[position_] == '"') {
            static_cast<void>(parse_string());
            return;
        }
        if (consume('{')) {
            if (consume('}')) {
                return;
            }
            while (true) {
                static_cast<void>(parse_string());
                expect(':');
                skip_value();
                if (consume('}')) {
                    return;
                }
                expect(',');
            }
        }
        if (consume('[')) {
            if (consume(']')) {
                return;
            }
            while (true) {
                skip_value();
                if (consume(']')) {
                    return;
                }
                expect(',');
            }
        }
        skip_literal();
    }

    TensorInfo parse_tensor(const std::string& name) {
        TensorInfo tensor;
        tensor.name = name;
        bool has_dtype = false;
        bool has_shape = false;
        bool has_offsets = false;

        expect('{');
        if (consume('}')) {
            fail("empty tensor descriptor");
        }
        while (true) {
            const auto key = parse_string();
            expect(':');
            if (key == "dtype") {
                tensor.dtype = parse_string();
                has_dtype = true;
            } else if (key == "shape") {
                tensor.shape = parse_u64_array();
                has_shape = true;
            } else if (key == "data_offsets") {
                const auto offsets = parse_u64_array();
                if (offsets.size() != 2) {
                    fail("data_offsets must contain two integers");
                }
                tensor.data_begin = offsets[0];
                tensor.data_end = offsets[1];
                has_offsets = true;
            } else {
                skip_value();
            }
            if (consume('}')) {
                break;
            }
            expect(',');
        }

        if (!has_dtype || !has_shape || !has_offsets) {
            fail("incomplete descriptor for tensor " + name);
        }
        if (tensor.data_end < tensor.data_begin) {
            fail("reversed data offsets for tensor " + name);
        }
        return tensor;
    }

    std::string_view input_;
    std::size_t position_{};
};

std::uint64_t decode_le_u64(const std::array<unsigned char, kLengthPrefixBytes>& bytes) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (i * 8);
    }
    return value;
}

}  // namespace

std::uint64_t TensorInfo::storage_bytes() const {
    return data_end - data_begin;
}

SafeTensorIndex SafeTensorIndex::open(const std::filesystem::path& path) {
    SafeTensorIndex index;
    index.path_ = path;
    index.file_bytes_ = std::filesystem::file_size(path);
    if (index.file_bytes_ < kLengthPrefixBytes) {
        throw std::runtime_error(path.string() + ": file is too small for a safetensors header");
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(path.string() + ": cannot open file");
    }

    std::array<unsigned char, kLengthPrefixBytes> prefix{};
    stream.read(reinterpret_cast<char*>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
    if (!stream) {
        throw std::runtime_error(path.string() + ": cannot read header length");
    }
    index.header_bytes_ = decode_le_u64(prefix);
    if (index.header_bytes_ == 0 || index.header_bytes_ > kMaxHeaderBytes) {
        throw std::runtime_error(path.string() + ": invalid header length " + std::to_string(index.header_bytes_));
    }
    if (index.header_bytes_ > index.file_bytes_ - kLengthPrefixBytes) {
        throw std::runtime_error(path.string() + ": header extends beyond file");
    }

    std::string header(static_cast<std::size_t>(index.header_bytes_), '\0');
    stream.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!stream) {
        throw std::runtime_error(path.string() + ": cannot read complete header");
    }
    index.tensors_ = HeaderParser(header).parse();

    const auto payload_bytes = index.file_bytes_ - index.data_offset();
    for (const auto& tensor : index.tensors_) {
        if (tensor.data_end > payload_bytes) {
            throw std::runtime_error(path.string() + ": tensor " + tensor.name + " extends beyond payload");
        }
    }
    return index;
}

const std::filesystem::path& SafeTensorIndex::path() const {
    return path_;
}

std::uint64_t SafeTensorIndex::header_bytes() const {
    return header_bytes_;
}

std::uint64_t SafeTensorIndex::data_offset() const {
    return kLengthPrefixBytes + header_bytes_;
}

std::uint64_t SafeTensorIndex::file_bytes() const {
    return file_bytes_;
}

const std::vector<TensorInfo>& SafeTensorIndex::tensors() const {
    return tensors_;
}

std::optional<TensorInfo> SafeTensorIndex::find(const std::string& name) const {
    const auto match = std::find_if(tensors_.begin(), tensors_.end(), [&](const TensorInfo& tensor) {
        return tensor.name == name;
    });
    if (match == tensors_.end()) {
        return std::nullopt;
    }
    return *match;
}

std::uint64_t SafeTensorIndex::absolute_offset(const TensorInfo& tensor) const {
    return data_offset() + tensor.data_begin;
}

}  // namespace ventus::outcore
