#include "ventus/outcore/quant.hpp"

#include <algorithm>
#include <array>
#include <bit>
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

float decode_fp8_e4m3fn(std::uint8_t bits) {
    const bool negative = (bits & 0x80) != 0;
    const auto magnitude = static_cast<std::uint8_t>(bits & 0x7f);
    const auto exponent = static_cast<std::uint8_t>(magnitude >> 3);
    const auto mantissa = static_cast<std::uint8_t>(magnitude & 0x07);
    if (exponent == 0x0f && mantissa == 0x07) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float value = 0.0F;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -9);
    } else {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, static_cast<int>(exponent) - 7);
    }
    return negative ? -value : value;
}

std::uint8_t encode_fp8_e4m3fn(float value) {
    const bool negative = std::signbit(value);
    const float magnitude = std::abs(value);
    if (std::isnan(magnitude)) {
        return static_cast<std::uint8_t>(negative ? 0xff : 0x7f);
    }
    if (magnitude >= 448.0F) {
        return static_cast<std::uint8_t>((negative ? 0x80 : 0x00) | 0x7e);
    }

    std::uint8_t best = 0;
    float best_distance = std::numeric_limits<float>::infinity();
    for (std::uint8_t candidate = 0; candidate <= 0x7e; ++candidate) {
        const float distance = std::abs(magnitude - decode_fp8_e4m3fn(candidate));
        if (distance < best_distance || (distance == best_distance && (candidate & 1) == 0)) {
            best = candidate;
            best_distance = distance;
        }
    }
    return static_cast<std::uint8_t>((negative ? 0x80 : 0x00) | best);
}

float round_to_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7f800000U) == 0x7f800000U) {
        return value;
    }
    const auto rounding_bias = static_cast<std::uint32_t>(0x7fffU + ((bits >> 16) & 1U));
    bits = (bits + rounding_bias) & 0xffff0000U;
    return std::bit_cast<float>(bits);
}

void quantize_bf16_to_fp8_block(
    const float* input,
    std::size_t count,
    std::uint8_t* output,
    std::uint8_t& scale) {
    if (count == 0) {
        throw std::invalid_argument("FP8 block must not be empty");
    }
    float absolute_maximum = 1.0e-4F;
    for (std::size_t i = 0; i < count; ++i) {
        absolute_maximum = std::max(absolute_maximum, std::abs(round_to_bf16(input[i])));
    }

    const float unrounded_scale = absolute_maximum / 448.0F;
    int exponent = 0;
    const float fraction = std::frexp(unrounded_scale, &exponent);
    const int scale_exponent = fraction == 0.5F ? exponent - 1 : exponent;
    if (scale_exponent < -127 || scale_exponent > 127) {
        throw std::overflow_error("FP8 activation scale is outside E8M0 range");
    }
    scale = static_cast<std::uint8_t>(scale_exponent + 127);
    const float scale_value = decode_e8m0(scale);
    for (std::size_t i = 0; i < count; ++i) {
        const float value = std::clamp(round_to_bf16(input[i]) / scale_value, -448.0F, 448.0F);
        output[i] = encode_fp8_e4m3fn(value);
    }
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
