#pragma once

#include <cstddef>
#include <cstdint>

namespace ventus::outcore {

[[nodiscard]] float decode_fp4_e2m1(std::uint8_t nibble);
[[nodiscard]] float decode_e8m0(std::uint8_t bits);
[[nodiscard]] float decode_fp8_e4m3fn(std::uint8_t bits);
[[nodiscard]] std::uint8_t encode_fp8_e4m3fn(float value);
[[nodiscard]] float round_to_bf16(float value);

void quantize_bf16_to_fp8_block(
    const float* input,
    std::size_t count,
    std::uint8_t* output,
    std::uint8_t& scale);

void dequantize_fp4_row(
    const std::uint8_t* packed,
    const std::uint8_t* scales,
    std::size_t logical_columns,
    float* output);

}  // namespace ventus::outcore
