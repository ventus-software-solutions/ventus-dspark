#include "ventus/outcore/quant.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ventus::outcore {

float decode_fp4_e2m1(std::uint8_t nibble) {
    constexpr std::array<float, 16> table{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
        0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
    };
    return table[nibble & 0x0f];
}

float decode_e8m0(std::uint8_t bits) {
    if (bits == 0xff) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::ldexp(1.0F, static_cast<int>(bits) - 127);
}

void dequantize_fp4_row(
    const std::uint8_t* packed,
    const std::uint8_t* scales,
    std::size_t logical_columns,
    float* output) {
    if (logical_columns == 0 || logical_columns % 32 != 0) {
        throw std::invalid_argument("FP4 row width must be a non-zero multiple of 32");
    }
    for (std::size_t column = 0; column < logical_columns; ++column) {
        const auto byte = packed[column / 2];
        const auto nibble = static_cast<std::uint8_t>((column % 2 == 0) ? (byte & 0x0f) : (byte >> 4));
        output[column] = decode_fp4_e2m1(nibble) * decode_e8m0(scales[column / 32]);
    }
}

}  // namespace ventus::outcore
