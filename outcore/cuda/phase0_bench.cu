#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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

double gib(std::size_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(1ULL << 30);
}

double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float milliseconds = 0;
    cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    return milliseconds;
}

}  // namespace

int main() try {
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    std::cout << "gpu: " << properties.name << '\n';
    std::cout << "compute_capability: " << properties.major << '.' << properties.minor << '\n';
    std::cout << "vram_total_bytes: " << total_bytes << '\n';
    std::cout << "vram_free_bytes: " << free_bytes << '\n';

    constexpr std::size_t transfer_bytes = 256ULL << 20;
    constexpr int transfer_iterations = 16;
    void* host = nullptr;
    void* device = nullptr;
    cuda_check(cudaHostAlloc(&host, transfer_bytes, cudaHostAllocDefault), "cudaHostAlloc");
    cuda_check(cudaMalloc(&device, transfer_bytes), "cudaMalloc transfer buffer");
    std::fill_n(static_cast<unsigned char*>(host), transfer_bytes, static_cast<unsigned char>(0x5a));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    cuda_check(cudaMemcpy(device, host, transfer_bytes, cudaMemcpyHostToDevice), "cudaMemcpy warmup");
    cuda_check(cudaEventRecord(start), "cudaEventRecord H2D");
    for (int i = 0; i < transfer_iterations; ++i) {
        cuda_check(cudaMemcpyAsync(device, host, transfer_bytes, cudaMemcpyHostToDevice), "cudaMemcpyAsync H2D");
    }
    const double h2d_ms = elapsed_ms(start, stop);

    cuda_check(cudaEventRecord(start), "cudaEventRecord D2H");
    for (int i = 0; i < transfer_iterations; ++i) {
        cuda_check(cudaMemcpyAsync(host, device, transfer_bytes, cudaMemcpyDeviceToHost), "cudaMemcpyAsync D2H");
    }
    const double d2h_ms = elapsed_ms(start, stop);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "h2d_gib_per_s: " << (gib(transfer_bytes) * transfer_iterations) / (h2d_ms / 1000.0) << '\n';
    std::cout << "d2h_gib_per_s: " << (gib(transfer_bytes) * transfer_iterations) / (d2h_ms / 1000.0) << '\n';

    cuda_check(cudaFree(device), "cudaFree transfer buffer");
    cuda_check(cudaFreeHost(host), "cudaFreeHost");

    // One DeepSeek expert projection at a representative prefill width.
    constexpr int m = 2048;
    constexpr int n = 128;
    constexpr int k = 4096;
    constexpr int gemm_iterations = 100;
    __half* weights = nullptr;
    __half* activations = nullptr;
    __half* output = nullptr;
    cuda_check(cudaMalloc(&weights, static_cast<std::size_t>(m) * k * sizeof(__half)), "cudaMalloc weights");
    cuda_check(cudaMalloc(&activations, static_cast<std::size_t>(k) * n * sizeof(__half)), "cudaMalloc activations");
    cuda_check(cudaMalloc(&output, static_cast<std::size_t>(m) * n * sizeof(__half)), "cudaMalloc output");
    cuda_check(cudaMemset(weights, 0, static_cast<std::size_t>(m) * k * sizeof(__half)), "cudaMemset weights");
    cuda_check(cudaMemset(activations, 0, static_cast<std::size_t>(k) * n * sizeof(__half)), "cudaMemset activations");

    cublasHandle_t handle{};
    cublas_check(cublasCreate(&handle), "cublasCreate");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    auto gemm = [&] {
        cublas_check(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                weights,
                CUDA_R_16F,
                m,
                activations,
                CUDA_R_16F,
                k,
                &beta,
                output,
                CUDA_R_16F,
                m,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx");
    };
    for (int i = 0; i < 5; ++i) {
        gemm();
    }
    cuda_check(cudaDeviceSynchronize(), "GEMM warmup synchronize");
    cuda_check(cudaEventRecord(start), "cudaEventRecord GEMM");
    for (int i = 0; i < gemm_iterations; ++i) {
        gemm();
    }
    const double gemm_ms = elapsed_ms(start, stop);
    const double operations = 2.0 * m * n * k * gemm_iterations;
    std::cout << "fp16_gemm_shape: " << m << 'x' << n << 'x' << k << '\n';
    std::cout << "fp16_gemm_ms: " << gemm_ms / gemm_iterations << '\n';
    std::cout << "fp16_gemm_tflops: " << operations / (gemm_ms / 1000.0) / 1.0e12 << '\n';

    cublas_check(cublasDestroy(handle), "cublasDestroy");
    cuda_check(cudaFree(weights), "cudaFree weights");
    cuda_check(cudaFree(activations), "cudaFree activations");
    cuda_check(cudaFree(output), "cudaFree output");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "phase0-cuda-bench: " << error.what() << '\n';
    return 1;
}
