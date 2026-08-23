#include "ventus/outcore/quant.hpp"
#include "ventus/outcore/safetensors.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kQuantBlock = 128;
constexpr float kSwiGluLimit = 10.0F;

struct Matrix {
    std::string name;
    std::uint64_t rows{};
    std::uint64_t columns{};
    std::vector<std::uint8_t> packed;
    std::vector<std::uint8_t> scales;
};

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void cublas_check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
    }
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

Matrix load_matrix(const ventus::outcore::SafeTensorIndex& index, const std::string& projection) {
    const ventus::outcore::TensorInfo* weight = nullptr;
    for (const auto& tensor : index.tensors()) {
        if (tensor.name.ends_with("." + projection + ".weight")) {
            weight = &tensor;
            break;
        }
    }
    if (weight == nullptr || weight->dtype != "I8" || weight->shape.size() != 2) {
        throw std::runtime_error("missing packed " + projection + " weight");
    }
    auto scale_name = weight->name;
    scale_name.replace(scale_name.size() - 6, 6, "scale");
    const auto scale = index.find(scale_name);
    if (!scale || scale->dtype != "F8_E8M0" || scale->shape.size() != 2) {
        throw std::runtime_error("missing " + projection + " scale");
    }
    const auto rows = weight->shape[0];
    const auto columns = weight->shape[1] * 2;
    if (scale->shape[0] != rows || scale->shape[1] != columns / 32) {
        throw std::runtime_error(projection + " scale shape mismatch");
    }
    return Matrix{weight->name, rows, columns, read_tensor(index, *weight), read_tensor(index, *scale)};
}

__device__ float fp4_value(std::uint8_t nibble) {
    constexpr float table[16] = {
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
        0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
    };
    return table[nibble & 0x0f];
}

__device__ float e8m0_value(std::uint8_t bits) {
    return ldexpf(1.0F, static_cast<int>(bits) - 127);
}

__global__ void dequantize_weight(
    const std::uint8_t* packed,
    const std::uint8_t* scales,
    __nv_bfloat16* output,
    std::uint64_t columns,
    std::uint64_t elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const auto column = index % columns;
    const auto byte = packed[index / 2];
    const auto nibble = static_cast<std::uint8_t>(column % 2 == 0 ? byte & 0x0f : byte >> 4);
    const auto scale_index = (index / columns) * (columns / 32) + column / 32;
    output[index] = __float2bfloat16_rn(fp4_value(nibble) * e8m0_value(scales[scale_index]));
}

__global__ void quantize_activation(
    const __nv_bfloat16* input,
    std::uint8_t* output,
    std::uint8_t* scales,
    std::uint64_t elements) {
    __shared__ float absolute_values[kQuantBlock];
    __shared__ float scale_value;
    const auto group = static_cast<std::uint64_t>(blockIdx.x);
    const auto index = group * kQuantBlock + threadIdx.x;
    const float value = index < elements ? __bfloat162float(input[index]) : 0.0F;
    absolute_values[threadIdx.x] = fabsf(value);
    __syncthreads();
    for (int width = kQuantBlock / 2; width != 0; width /= 2) {
        if (threadIdx.x < width) {
            absolute_values[threadIdx.x] = fmaxf(absolute_values[threadIdx.x], absolute_values[threadIdx.x + width]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float maximum = fmaxf(absolute_values[0], 1.0e-4F);
        const float raw_scale = maximum * (1.0F / 448.0F);
        const auto bits = static_cast<std::uint32_t>(__float_as_uint(raw_scale));
        const int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127;
        const int rounded = exponent + ((bits & 0x7fffffU) != 0 ? 1 : 0);
        scales[group] = static_cast<std::uint8_t>(rounded + 127);
        scale_value = __int_as_float((rounded + 127) << 23);
    }
    __syncthreads();
    if (index < elements) {
        const float normalized = fminf(448.0F, fmaxf(-448.0F, value / scale_value));
        const __nv_fp8_e4m3 encoded(normalized);
        output[index] = encoded.__x;
    }
}

__global__ void dequantize_activation(
    const std::uint8_t* input,
    const std::uint8_t* scales,
    __nv_bfloat16* output,
    std::uint64_t elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    __nv_fp8_e4m3 value;
    value.__x = input[index];
    output[index] = __float2bfloat16_rn(static_cast<float>(value) * e8m0_value(scales[index / kQuantBlock]));
}

__global__ void swiglu(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* output,
    std::uint64_t elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float limited_gate = fminf(__bfloat162float(gate[index]), kSwiGluLimit);
    const float limited_up = fminf(kSwiGluLimit, fmaxf(-kSwiGluLimit, __bfloat162float(up[index])));
    const float silu = limited_gate / (1.0F + expf(-limited_gate));
    output[index] = __float2bfloat16_rn(silu * limited_up);
}

void gemm_vector(
    cublasHandle_t handle,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int rows,
    int columns) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    cublas_check(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            rows,
            1,
            columns,
            &alpha,
            weight,
            CUDA_R_16BF,
            columns,
            input,
            CUDA_R_16BF,
            columns,
            &beta,
            output,
            CUDA_R_16BF,
            rows,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx");
}

float cpu_weight_value(const Matrix& matrix, std::uint64_t row, std::uint64_t column) {
    const auto byte = matrix.packed[row * (matrix.columns / 2) + column / 2];
    const auto nibble = static_cast<std::uint8_t>(column % 2 == 0 ? byte & 0x0f : byte >> 4);
    const auto scale = matrix.scales[row * (matrix.columns / 32) + column / 32];
    return ventus::outcore::decode_fp4_e2m1(nibble) * ventus::outcore::decode_e8m0(scale);
}

std::vector<float> cpu_gemm(const Matrix& matrix, const std::vector<float>& input) {
    std::vector<float> output(matrix.rows);
    for (std::uint64_t row = 0; row < matrix.rows; ++row) {
        float sum = 0.0F;
        for (std::uint64_t column = 0; column < matrix.columns; ++column) {
            sum += cpu_weight_value(matrix, row, column) * input[column];
        }
        output[row] = ventus::outcore::round_to_bf16(sum);
    }
    return output;
}

void usage() {
    std::cerr << "Usage: cuda-expert-test --bundle EXPERT.safetensors\n";
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
    const auto w1 = load_matrix(index, "w1");
    const auto w3 = load_matrix(index, "w3");
    const auto w2 = load_matrix(index, "w2");
    if (w1.columns != 4096 || w1.rows != 2048 || w3.rows != w1.rows || w2.columns != w1.rows || w2.rows != 4096) {
        throw std::runtime_error("unexpected DeepSeek expert dimensions");
    }

    std::vector<float> input(w1.columns);
    std::vector<__nv_bfloat16> input_bf16(w1.columns);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.013F) * 0.25F;
        input_bf16[i] = __float2bfloat16_rn(input[i]);
        input[i] = ventus::outcore::round_to_bf16(input[i]);
    }

    std::vector<std::uint8_t> input_fp8(input.size());
    std::vector<std::uint8_t> input_scales(input.size() / kQuantBlock);
    std::vector<float> quantized_input(input.size());
    for (std::size_t group = 0; group < input_scales.size(); ++group) {
        ventus::outcore::quantize_bf16_to_fp8_block(
            input.data() + group * kQuantBlock,
            kQuantBlock,
            input_fp8.data() + group * kQuantBlock,
            input_scales[group]);
        for (std::size_t offset = 0; offset < kQuantBlock; ++offset) {
            const auto position = group * kQuantBlock + offset;
            quantized_input[position] = ventus::outcore::decode_fp8_e4m3fn(input_fp8[position]) *
                                        ventus::outcore::decode_e8m0(input_scales[group]);
        }
    }

    const auto cpu_gate = cpu_gemm(w1, quantized_input);
    const auto cpu_up = cpu_gemm(w3, quantized_input);
    std::vector<float> cpu_hidden(cpu_gate.size());
    for (std::size_t i = 0; i < cpu_hidden.size(); ++i) {
        const float gate = std::min(cpu_gate[i], kSwiGluLimit);
        const float up = std::clamp(cpu_up[i], -kSwiGluLimit, kSwiGluLimit);
        cpu_hidden[i] = ventus::outcore::round_to_bf16((gate / (1.0F + std::exp(-gate))) * up);
    }
    const auto cpu_output = cpu_gemm(w2, cpu_hidden);

    auto upload_matrix = [](const Matrix& matrix, std::uint8_t*& packed, std::uint8_t*& scales, __nv_bfloat16*& values) {
        cuda_check(cudaMalloc(&packed, matrix.packed.size()), "cudaMalloc packed weight");
        cuda_check(cudaMalloc(&scales, matrix.scales.size()), "cudaMalloc weight scales");
        cuda_check(cudaMalloc(&values, matrix.rows * matrix.columns * sizeof(__nv_bfloat16)), "cudaMalloc weight values");
        cuda_check(cudaMemcpy(packed, matrix.packed.data(), matrix.packed.size(), cudaMemcpyHostToDevice), "cudaMemcpy packed weight");
        cuda_check(cudaMemcpy(scales, matrix.scales.data(), matrix.scales.size(), cudaMemcpyHostToDevice), "cudaMemcpy weight scales");
        const auto elements = matrix.rows * matrix.columns;
        dequantize_weight<<<static_cast<unsigned int>((elements + 255) / 256), 256>>>(
            packed, scales, values, matrix.columns, elements);
        cuda_check(cudaGetLastError(), "dequantize_weight launch");
    };

    std::uint8_t *w1_packed = nullptr, *w1_scales = nullptr, *w3_packed = nullptr, *w3_scales = nullptr;
    std::uint8_t *w2_packed = nullptr, *w2_scales = nullptr;
    __nv_bfloat16 *w1_values = nullptr, *w3_values = nullptr, *w2_values = nullptr;
    cudaEvent_t dequant_start{}, dequant_stop{};
    cuda_check(cudaEventCreate(&dequant_start), "cudaEventCreate dequant start");
    cuda_check(cudaEventCreate(&dequant_stop), "cudaEventCreate dequant stop");
    cuda_check(cudaEventRecord(dequant_start), "cudaEventRecord dequant start");
    upload_matrix(w1, w1_packed, w1_scales, w1_values);
    upload_matrix(w3, w3_packed, w3_scales, w3_values);
    upload_matrix(w2, w2_packed, w2_scales, w2_values);
    cuda_check(cudaEventRecord(dequant_stop), "cudaEventRecord dequant stop");
    cuda_check(cudaEventSynchronize(dequant_stop), "cudaEventSynchronize dequant");
    float dequant_ms = 0;
    cuda_check(cudaEventElapsedTime(&dequant_ms, dequant_start, dequant_stop), "cudaEventElapsedTime dequant");

    auto dequantize_all = [&] {
        const auto launch = [](const Matrix& matrix, const std::uint8_t* packed, const std::uint8_t* scales, __nv_bfloat16* values) {
            const auto elements = matrix.rows * matrix.columns;
            dequantize_weight<<<static_cast<unsigned int>((elements + 255) / 256), 256>>>(
                packed, scales, values, matrix.columns, elements);
        };
        launch(w1, w1_packed, w1_scales, w1_values);
        launch(w3, w3_packed, w3_scales, w3_values);
        launch(w2, w2_packed, w2_scales, w2_values);
    };
    cudaEvent_t repeat_dequant_start{}, repeat_dequant_stop{};
    cuda_check(cudaEventCreate(&repeat_dequant_start), "cudaEventCreate repeat dequant start");
    cuda_check(cudaEventCreate(&repeat_dequant_stop), "cudaEventCreate repeat dequant stop");
    constexpr int dequant_iterations = 100;
    cuda_check(cudaEventRecord(repeat_dequant_start), "cudaEventRecord repeat dequant start");
    for (int i = 0; i < dequant_iterations; ++i) {
        dequantize_all();
    }
    cuda_check(cudaEventRecord(repeat_dequant_stop), "cudaEventRecord repeat dequant stop");
    cuda_check(cudaEventSynchronize(repeat_dequant_stop), "cudaEventSynchronize repeat dequant");
    float repeat_dequant_ms = 0;
    cuda_check(
        cudaEventElapsedTime(&repeat_dequant_ms, repeat_dequant_start, repeat_dequant_stop),
        "cudaEventElapsedTime repeat dequant");

    __nv_bfloat16 *device_input = nullptr, *device_quantized_input = nullptr;
    __nv_bfloat16 *device_gate = nullptr, *device_up = nullptr, *device_hidden = nullptr, *device_output = nullptr;
    std::uint8_t *device_input_fp8 = nullptr, *device_input_scales = nullptr;
    cuda_check(cudaMalloc(&device_input, input_bf16.size() * sizeof(__nv_bfloat16)), "cudaMalloc input");
    cuda_check(cudaMalloc(&device_quantized_input, input_bf16.size() * sizeof(__nv_bfloat16)), "cudaMalloc quantized input");
    cuda_check(cudaMalloc(&device_input_fp8, input_fp8.size()), "cudaMalloc input fp8");
    cuda_check(cudaMalloc(&device_input_scales, input_scales.size()), "cudaMalloc input scales");
    cuda_check(cudaMalloc(&device_gate, w1.rows * sizeof(__nv_bfloat16)), "cudaMalloc gate");
    cuda_check(cudaMalloc(&device_up, w3.rows * sizeof(__nv_bfloat16)), "cudaMalloc up");
    cuda_check(cudaMalloc(&device_hidden, w1.rows * sizeof(__nv_bfloat16)), "cudaMalloc hidden");
    cuda_check(cudaMalloc(&device_output, w2.rows * sizeof(__nv_bfloat16)), "cudaMalloc output");
    cuda_check(cudaMemcpy(device_input, input_bf16.data(), input_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "cudaMemcpy input");

    cublasHandle_t handle{};
    cublas_check(cublasCreate(&handle), "cublasCreate");
    auto forward = [&] {
        quantize_activation<<<input.size() / kQuantBlock, kQuantBlock>>>(
            device_input, device_input_fp8, device_input_scales, input.size());
        dequantize_activation<<<static_cast<unsigned int>((input.size() + 255) / 256), 256>>>(
            device_input_fp8, device_input_scales, device_quantized_input, input.size());
        gemm_vector(handle, w1_values, device_quantized_input, device_gate, w1.rows, w1.columns);
        gemm_vector(handle, w3_values, device_quantized_input, device_up, w3.rows, w3.columns);
        swiglu<<<static_cast<unsigned int>((w1.rows + 255) / 256), 256>>>(device_gate, device_up, device_hidden, w1.rows);
        gemm_vector(handle, w2_values, device_hidden, device_output, w2.rows, w2.columns);
    };
    forward();
    cuda_check(cudaDeviceSynchronize(), "expert forward synchronize");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    constexpr int iterations = 100;
    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iterations; ++i) {
        forward();
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float elapsed_ms = 0;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    std::vector<__nv_bfloat16> gpu_output(w2.rows);
    cuda_check(cudaMemcpy(gpu_output.data(), device_output, gpu_output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "cudaMemcpy output");
    float max_absolute_error = 0.0F;
    float max_relative_error = 0.0F;
    std::uint64_t outside_tolerance = 0;
    for (std::size_t i = 0; i < gpu_output.size(); ++i) {
        const float actual = __bfloat162float(gpu_output[i]);
        const float expected = cpu_output[i];
        const float absolute = std::abs(actual - expected);
        const float relative = absolute / std::max(1.0e-5F, std::abs(expected));
        max_absolute_error = std::max(max_absolute_error, absolute);
        max_relative_error = std::max(max_relative_error, relative);
        outside_tolerance += absolute > 0.125F + 0.03F * std::abs(expected) ? 1 : 0;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "expert: " << w1.name.substr(0, w1.name.find(".w1.weight")) << '\n';
    std::cout << "packed_weight_bytes: " << w1.packed.size() + w3.packed.size() + w2.packed.size() +
        w1.scales.size() + w3.scales.size() + w2.scales.size() << '\n';
    std::cout << "cold_weight_prepare_ms: " << dequant_ms << '\n';
    std::cout << "resident_weight_dequant_ms: " << repeat_dequant_ms / dequant_iterations << '\n';
    std::cout << "resident_forward_ms: " << elapsed_ms / iterations << '\n';
    std::cout << "max_absolute_error: " << max_absolute_error << '\n';
    std::cout << "max_relative_error: " << max_relative_error << '\n';
    std::cout << "outside_tolerance: " << outside_tolerance << '/' << gpu_output.size() << '\n';

    cublas_check(cublasDestroy(handle), "cublasDestroy");
    cudaFree(w1_packed); cudaFree(w1_scales); cudaFree(w1_values);
    cudaFree(w3_packed); cudaFree(w3_scales); cudaFree(w3_values);
    cudaFree(w2_packed); cudaFree(w2_scales); cudaFree(w2_values);
    cudaFree(device_input); cudaFree(device_quantized_input); cudaFree(device_input_fp8); cudaFree(device_input_scales);
    cudaFree(device_gate); cudaFree(device_up); cudaFree(device_hidden); cudaFree(device_output);
    cudaEventDestroy(dequant_start); cudaEventDestroy(dequant_stop);
    cudaEventDestroy(repeat_dequant_start); cudaEventDestroy(repeat_dequant_stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return outside_tolerance == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "cuda-expert-test: " << error.what() << '\n';
    return 1;
}
