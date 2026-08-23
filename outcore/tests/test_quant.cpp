#include "ventus/outcore/quant.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

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

    for (std::uint16_t bits = 0; bits <= 0xff; ++bits) {
        const auto encoded = static_cast<std::uint8_t>(bits);
        const auto decoded = ventus::outcore::decode_fp8_e4m3fn(encoded);
        if ((encoded & 0x7f) == 0x7f) {
            assert(std::isnan(decoded));
        } else {
            assert(ventus::outcore::encode_fp8_e4m3fn(decoded) == encoded);
        }
    }
    assert(ventus::outcore::decode_fp8_e4m3fn(0x01) == std::ldexp(1.0F, -9));
    assert(ventus::outcore::decode_fp8_e4m3fn(0x7e) == 448.0F);
    assert(ventus::outcore::encode_fp8_e4m3fn(1000.0F) == 0x7e);
    assert(ventus::outcore::encode_fp8_e4m3fn(-1000.0F) == 0xfe);
    assert(ventus::outcore::encode_fp8_e4m3fn(std::numeric_limits<float>::quiet_NaN()) == 0x7f);
    assert(ventus::outcore::round_to_bf16(1.00390625F) == 1.0F);

    const std::array<float, 8> activation{-9.0F, -1.0F, -0.125F, 0.0F, 0.25F, 1.0F, 7.0F, 10.0F};
    std::array<std::uint8_t, activation.size()> activation_fp8{};
    std::uint8_t activation_scale = 0;
    ventus::outcore::quantize_bf16_to_fp8_block(
        activation.data(), activation.size(), activation_fp8.data(), activation_scale);
    assert(activation_scale != 0xff);
    const float activation_scale_value = ventus::outcore::decode_e8m0(activation_scale);
    for (std::size_t i = 0; i < activation.size(); ++i) {
        const float reconstructed = ventus::outcore::decode_fp8_e4m3fn(activation_fp8[i]) * activation_scale_value;
        assert(std::isfinite(reconstructed));
    }

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
