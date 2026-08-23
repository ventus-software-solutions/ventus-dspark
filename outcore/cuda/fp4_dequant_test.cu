#include "ventus/outcore/quant.hpp"
#include "ventus/outcore/safetensors.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__device__ float decode_fp4(std::uint8_t nibble) {
    constexpr float table[16] = {
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
        0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
    };
    return table[nibble & 0x0f];
}

__device__ float decode_scale(std::uint8_t bits) {
    return bits == 0xff ? __int_as_float(0x7fc00000) : ldexpf(1.0F, static_cast<int>(bits) - 127);
}

__global__ void dequantize_fp4_kernel(
    const std::uint8_t* packed,
    const std::uint8_t* scales,
    __half* output,
    std::uint64_t logical_columns,
    std::uint64_t logical_elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= logical_elements) {
        return;
    }
    const auto column = index % logical_columns;
    const auto byte = packed[index / 2];
    const auto nibble = static_cast<std::uint8_t>((column % 2 == 0) ? (byte & 0x0f) : (byte >> 4));
    const auto scale_index = (index / logical_columns) * (logical_columns / 32) + column / 32;
    output[index] = __float2half_rn(decode_fp4(nibble) * decode_scale(scales[scale_index]));
}

std::vector<std::uint8_t> read_tensor(
    const ventus::outcore::SafeTensorIndex& index,
    const ventus::outcore::TensorInfo& tensor) {
    std::vector<std::uint8_t> bytes(tensor.storage_bytes());
    std::ifstream stream(index.path(), std::ios::binary);
    stream.seekg(static_cast<std::streamoff>(index.absolute_offset(tensor)));
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("cannot read tensor " + tensor.name);
    }
    return bytes;
}

void usage() {
    std::cerr << "Usage: cuda-fp4-test --bundle EXPERT.safetensors\n";
}

}  // namespace

int main(int argc, char** argv) try {
    std::string bundle_path;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--bundle" && i + 1 < argc) {
            bundle_path = argv[++i];
        } else {
            usage();
            return 2;
        }
    }
    if (bundle_path.empty()) {
        usage();
        return 2;
    }

    const auto index = ventus::outcore::SafeTensorIndex::open(bundle_path);
    const ventus::outcore::TensorInfo* weight = nullptr;
    for (const auto& tensor : index.tensors()) {
        if (tensor.name.ends_with(".w1.weight")) {
            weight = &tensor;
            break;
        }
    }
    if (weight == nullptr || weight->shape.size() != 2 || weight->dtype != "I8") {
        throw std::runtime_error("bundle has no packed w1 weight");
    }
    auto scale_name = weight->name;
    scale_name.replace(scale_name.size() - 6, 6, "scale");
    const auto scale = index.find(scale_name);
    if (!scale || scale->shape.size() != 2 || scale->dtype != "F8_E8M0") {
        throw std::runtime_error("bundle has no matching E8M0 scale tensor");
    }

    const std::uint64_t rows = weight->shape[0];
    const std::uint64_t logical_columns = weight->shape[1] * 2;
    const std::uint64_t logical_elements = rows * logical_columns;
    if (scale->shape[0] != rows || scale->shape[1] != logical_columns / 32) {
        throw std::runtime_error("weight and scale shapes disagree");
    }

    const auto packed = read_tensor(index, *weight);
    const auto scales = read_tensor(index, *scale);
    std::vector<float> expected(logical_elements);
    for (std::uint64_t row = 0; row < rows; ++row) {
        ventus::outcore::dequantize_fp4_row(
            packed.data() + row * logical_columns / 2,
            scales.data() + row * logical_columns / 32,
            logical_columns,
            expected.data() + row * logical_columns);
    }

    std::uint8_t* device_packed = nullptr;
    std::uint8_t* device_scales = nullptr;
    __half* device_output = nullptr;
    cuda_check(cudaMalloc(&device_packed, packed.size()), "cudaMalloc packed");
    cuda_check(cudaMalloc(&device_scales, scales.size()), "cudaMalloc scales");
    cuda_check(cudaMalloc(&device_output, logical_elements * sizeof(__half)), "cudaMalloc output");
    cuda_check(cudaMemcpy(device_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice), "cudaMemcpy packed");
    cuda_check(cudaMemcpy(device_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice), "cudaMemcpy scales");

    constexpr int threads = 256;
    const auto blocks = static_cast<unsigned int>((logical_elements + threads - 1) / threads);
    dequantize_fp4_kernel<<<blocks, threads>>>(
        device_packed, device_scales, device_output, logical_columns, logical_elements);
    cuda_check(cudaGetLastError(), "dequantize_fp4_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "dequantize_fp4_kernel synchronize");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    constexpr int iterations = 100;
    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iterations; ++i) {
        dequantize_fp4_kernel<<<blocks, threads>>>(
            device_packed, device_scales, device_output, logical_columns, logical_elements);
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float elapsed_ms = 0;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    std::vector<__half> actual(logical_elements);
    cuda_check(
        cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(__half), cudaMemcpyDeviceToHost),
        "cudaMemcpy output");

    std::uint64_t mismatches = 0;
    float max_abs_error = 0.0F;
    for (std::uint64_t i = 0; i < logical_elements; ++i) {
        const __half expected_half = __float2half_rn(expected[i]);
        std::uint16_t expected_bits = 0;
        std::uint16_t actual_bits = 0;
        std::memcpy(&expected_bits, &expected_half, sizeof(expected_bits));
        std::memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
        if (expected_bits != actual_bits) {
            ++mismatches;
        }
        max_abs_error = std::max(max_abs_error, std::abs(__half2float(actual[i]) - expected[i]));
    }

    const double input_bytes = static_cast<double>(packed.size() + scales.size());
    const double average_ms = elapsed_ms / iterations;
    const double input_gib_per_second = (input_bytes / static_cast<double>(1ULL << 30)) / (average_ms / 1000.0);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "weight: " << weight->name << '\n';
    std::cout << "logical_shape: " << rows << 'x' << logical_columns << '\n';
    std::cout << "packed_bytes: " << packed.size() << '\n';
    std::cout << "scale_bytes: " << scales.size() << '\n';
    std::cout << "kernel_ms: " << average_ms << '\n';
    std::cout << "kernel_input_gib_per_s: " << input_gib_per_second << '\n';
    std::cout << "half_bit_mismatches: " << mismatches << '\n';
    std::cout << "max_abs_error_vs_float_oracle: " << max_abs_error << '\n';

    cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    cuda_check(cudaFree(device_packed), "cudaFree packed");
    cuda_check(cudaFree(device_scales), "cudaFree scales");
    cuda_check(cudaFree(device_output), "cudaFree output");
    return mismatches == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "cuda-fp4-test: " << error.what() << '\n';
    return 1;
}
