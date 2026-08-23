#include "ventus/outcore/quant.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

int main() {
    constexpr std::array<float, 16> fp4_values{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
        0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
    };
    for (std::uint8_t nibble = 0; nibble < fp4_values.size(); ++nibble) {
        assert(ventus::outcore::decode_fp4_e2m1(nibble) == fp4_values[nibble]);
    }
    assert(ventus::outcore::decode_e8m0(127) == 1.0F);
    assert(ventus::outcore::decode_e8m0(128) == 2.0F);
    assert(ventus::outcore::decode_e8m0(126) == 0.5F);
    assert(ventus::outcore::decode_e8m0(0) == std::ldexp(1.0F, -127));
    assert(std::isnan(ventus::outcore::decode_e8m0(255)));

    std::array<std::uint8_t, 16> packed{};
    for (std::size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<std::uint8_t>(((15 - i) << 4) | i);
    }
    const std::array<std::uint8_t, 1> scales{128};
    std::array<float, 32> output{};
    ventus::outcore::dequantize_fp4_row(packed.data(), scales.data(), output.size(), output.data());
    for (std::size_t i = 0; i < packed.size(); ++i) {
        assert(output[i * 2] == fp4_values[i] * 2.0F);
        assert(output[i * 2 + 1] == fp4_values[15 - i] * 2.0F);
    }

    std::cout << "quant oracle test passed\n";
    return 0;
}
