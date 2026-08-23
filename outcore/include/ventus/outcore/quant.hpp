#pragma once

#include <cstddef>
#include <cstdint>

namespace ventus::outcore {

[[nodiscard]] float decode_fp4_e2m1(std::uint8_t nibble);
[[nodiscard]] float decode_e8m0(std::uint8_t bits);

void dequantize_fp4_row(
    const std::uint8_t* packed,
    const std::uint8_t* scales,
    std::size_t logical_columns,
    float* output);

}  // namespace ventus::outcore
