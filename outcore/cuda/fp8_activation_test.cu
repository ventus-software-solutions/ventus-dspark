#include "ventus/outcore/quant.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 128;

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void quantize_activation(
    const __nv_bfloat16* input,
    std::uint8_t* output,
    std::uint8_t* scales,
    std::uint64_t element_count) {
    __shared__ float absolute_values[kBlockSize];
    __shared__ float scale_value;
    const auto group = static_cast<std::uint64_t>(blockIdx.x);
    const auto index = group * kBlockSize + threadIdx.x;
    const float value = index < element_count ? __bfloat162float(input[index]) : 0.0F;
    absolute_values[threadIdx.x] = fabsf(value);
    __syncthreads();

    for (int width = kBlockSize / 2; width != 0; width /= 2) {
        if (threadIdx.x < width) {
            absolute_values[threadIdx.x] = fmaxf(absolute_values[threadIdx.x], absolute_values[threadIdx.x + width]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float absolute_maximum = fmaxf(absolute_values[0], 1.0e-4F);
        const float unrounded_scale = absolute_maximum * (1.0F / 448.0F);
        const auto bits = static_cast<std::uint32_t>(__float_as_uint(unrounded_scale));
        const int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127;
        const int rounded_exponent = exponent + ((bits & 0x7fffffU) != 0 ? 1 : 0);
        scales[group] = static_cast<std::uint8_t>(rounded_exponent + 127);
        scale_value = __int_as_float((rounded_exponent + 127) << 23);
    }
    __syncthreads();

    if (index < element_count) {
        const float normalized = fminf(448.0F, fmaxf(-448.0F, value / scale_value));
        const __nv_fp8_e4m3 quantized(normalized);
        output[index] = quantized.__x;
    }
}

}  // namespace

int main() try {
    constexpr std::uint64_t element_count = 4096;
    constexpr std::uint64_t group_count = element_count / kBlockSize;
    std::vector<float> input(element_count);
    std::vector<__nv_bfloat16> input_bf16(element_count);
    for (std::uint64_t i = 0; i < element_count; ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.017F) * (1.0F + static_cast<float>(i % 31));
        if (i % 257 == 0) {
            input[i] = i % 2 == 0 ? 447.0F : -448.0F;
        }
        input_bf16[i] = __float2bfloat16_rn(input[i]);
    }

    std::vector<std::uint8_t> expected(element_count);
    std::vector<std::uint8_t> expected_scales(group_count);
    for (std::uint64_t group = 0; group < group_count; ++group) {
        ventus::outcore::quantize_bf16_to_fp8_block(
            input.data() + group * kBlockSize,
            kBlockSize,
            expected.data() + group * kBlockSize,
            expected_scales[group]);
    }

    __nv_bfloat16* device_input = nullptr;
    std::uint8_t* device_output = nullptr;
    std::uint8_t* device_scales = nullptr;
    cuda_check(cudaMalloc(&device_input, input_bf16.size() * sizeof(__nv_bfloat16)), "cudaMalloc input");
    cuda_check(cudaMalloc(&device_output, expected.size()), "cudaMalloc output");
    cuda_check(cudaMalloc(&device_scales, expected_scales.size()), "cudaMalloc scales");
    cuda_check(
        cudaMemcpy(device_input, input_bf16.data(), input_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
        "cudaMemcpy input");

    quantize_activation<<<group_count, kBlockSize>>>(device_input, device_output, device_scales, element_count);
    cuda_check(cudaGetLastError(), "quantize_activation launch");
    cuda_check(cudaDeviceSynchronize(), "quantize_activation synchronize");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    constexpr int iterations = 1000;
    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iterations; ++i) {
        quantize_activation<<<group_count, kBlockSize>>>(device_input, device_output, device_scales, element_count);
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float elapsed_ms = 0;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    std::vector<std::uint8_t> actual(element_count);
    std::vector<std::uint8_t> actual_scales(group_count);
    cuda_check(cudaMemcpy(actual.data(), device_output, actual.size(), cudaMemcpyDeviceToHost), "cudaMemcpy output");
    cuda_check(cudaMemcpy(actual_scales.data(), device_scales, actual_scales.size(), cudaMemcpyDeviceToHost), "cudaMemcpy scales");

    std::uint64_t value_mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        value_mismatches += actual[i] != expected[i] ? 1 : 0;
    }
    std::uint64_t scale_mismatches = 0;
    for (std::size_t i = 0; i < actual_scales.size(); ++i) {
        scale_mismatches += actual_scales[i] != expected_scales[i] ? 1 : 0;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "elements: " << element_count << '\n';
    std::cout << "groups: " << group_count << '\n';
    std::cout << "value_bit_mismatches: " << value_mismatches << '\n';
    std::cout << "scale_bit_mismatches: " << scale_mismatches << '\n';
    std::cout << "kernel_ms: " << elapsed_ms / iterations << '\n';

    cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    cuda_check(cudaFree(device_input), "cudaFree input");
    cuda_check(cudaFree(device_output), "cudaFree output");
    cuda_check(cudaFree(device_scales), "cudaFree scales");
    return value_mismatches == 0 && scale_mismatches == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "cuda-fp8-activation-test: " << error.what() << '\n';
    return 1;
}
