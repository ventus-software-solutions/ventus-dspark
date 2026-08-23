#include "ventus/outcore/quant.hpp"

#include <cuda_fp8.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

int main() {
    std::uint64_t decode_mismatches = 0;
    for (std::uint16_t bits = 0; bits <= 0xff; ++bits) {
        __nv_fp8_e4m3 cuda_value;
        cuda_value.__x = static_cast<__nv_fp8_storage_t>(bits);
        const float cuda_decoded = static_cast<float>(cuda_value);
        const float oracle_decoded = ventus::outcore::decode_fp8_e4m3fn(static_cast<std::uint8_t>(bits));
        std::uint32_t cuda_bits = 0;
        std::uint32_t oracle_bits = 0;
        std::memcpy(&cuda_bits, &cuda_decoded, sizeof(cuda_bits));
        std::memcpy(&oracle_bits, &oracle_decoded, sizeof(oracle_bits));
        if (std::isnan(cuda_decoded) && std::isnan(oracle_decoded)) {
            continue;
        }
        if (cuda_bits != oracle_bits) {
            ++decode_mismatches;
        }
    }

    constexpr std::array<float, 22> values{
        -1000.0F, -448.0F, -400.0F, -10.0F, -6.0F, -1.5F, -1.0F, -0.1F,
        -0.001953125F, -0.0F, 0.0F, 0.001953125F, 0.1F, 0.5F, 1.0F, 1.5F,
        6.0F, 10.0F, 100.0F, 400.0F, 448.0F, 1000.0F,
    };
    std::uint64_t encode_mismatches = 0;
    for (const float value : values) {
        const __nv_fp8_e4m3 cuda_encoded(value);
        const auto oracle_encoded = ventus::outcore::encode_fp8_e4m3fn(value);
        if (cuda_encoded.__x != oracle_encoded) {
            ++encode_mismatches;
            std::cerr << "encode mismatch for " << value << ": cuda="
                      << static_cast<unsigned int>(cuda_encoded.__x) << " oracle="
                      << static_cast<unsigned int>(oracle_encoded) << '\n';
        }
    }

    std::cout << "decode_mismatches: " << decode_mismatches << '\n';
    std::cout << "encode_mismatches: " << encode_mismatches << '\n';
    return decode_mismatches == 0 && encode_mismatches == 0 ? 0 : 1;
}
