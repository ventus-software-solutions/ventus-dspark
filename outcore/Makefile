CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -Iinclude
BUILD_DIR ?= build

LIB_SOURCE := src/byte_cache.cpp src/model_store.cpp src/quant.cpp src/safetensors.cpp

.PHONY: all test inspect cuda-bench cuda-pager-bench cuda-fp4-test cuda-fp8-test cuda-fp8-activation-test cuda-expert-test cuda-gpu-cache-test clean

all: $(BUILD_DIR)/outcore-inspect $(BUILD_DIR)/outcore-extract $(BUILD_DIR)/outcore-repack $(BUILD_DIR)/outcore-cache-probe $(BUILD_DIR)/outcore-io-bench $(BUILD_DIR)/test-safetensors $(BUILD_DIR)/test-quant $(BUILD_DIR)/test-byte-cache

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/outcore-inspect: $(LIB_SOURCE) tools/model_inspect.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/outcore-extract: $(LIB_SOURCE) tools/extract_bundle.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/outcore-repack: $(LIB_SOURCE) tools/model_repack.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/outcore-cache-probe: $(LIB_SOURCE) tools/cache_probe.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/outcore-io-bench: tools/io_bench.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/test-safetensors: $(LIB_SOURCE) tests/test_safetensors.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/test-quant: $(LIB_SOURCE) tests/test_quant.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/test-byte-cache: $(LIB_SOURCE) tests/test_byte_cache.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

test: $(BUILD_DIR)/test-safetensors $(BUILD_DIR)/test-quant $(BUILD_DIR)/test-byte-cache
	$(BUILD_DIR)/test-safetensors
	$(BUILD_DIR)/test-quant
	$(BUILD_DIR)/test-byte-cache

inspect: $(BUILD_DIR)/outcore-inspect
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model is required" && exit 2)
	$(BUILD_DIR)/outcore-inspect --model "$(MODEL)"

cuda-bench: | $(BUILD_DIR)
	nvcc -std=c++20 -O3 -arch=sm_86 cuda/phase0_bench.cu -lcublas -o $(BUILD_DIR)/phase0-cuda-bench
	$(BUILD_DIR)/phase0-cuda-bench

cuda-pager-bench: | $(BUILD_DIR)
	@test -n "$(FILE)" || (echo "FILE=/path/to/aligned-bundle is required" && exit 2)
	nvcc -std=c++20 -O3 -arch=sm_86 cuda/pager_bench.cu -o $(BUILD_DIR)/cuda-pager-bench
	$(BUILD_DIR)/cuda-pager-bench --file "$(FILE)"

cuda-fp4-test: | $(BUILD_DIR)
	@test -n "$(FILE)" || (echo "FILE=/path/to/expert-bundle is required" && exit 2)
	nvcc -Iinclude -std=c++20 -O3 -arch=sm_86 src/quant.cpp src/safetensors.cpp cuda/fp4_dequant_test.cu -o $(BUILD_DIR)/cuda-fp4-test
	$(BUILD_DIR)/cuda-fp4-test --bundle "$(FILE)"

cuda-fp8-test: | $(BUILD_DIR)
	nvcc -Iinclude -std=c++20 -O3 -arch=sm_86 src/quant.cpp cuda/fp8_format_test.cu -o $(BUILD_DIR)/cuda-fp8-test
	$(BUILD_DIR)/cuda-fp8-test

cuda-fp8-activation-test: | $(BUILD_DIR)
	nvcc -Iinclude -std=c++20 -O3 -arch=sm_86 src/quant.cpp cuda/fp8_activation_test.cu -o $(BUILD_DIR)/cuda-fp8-activation-test
	$(BUILD_DIR)/cuda-fp8-activation-test

cuda-expert-test: | $(BUILD_DIR)
	@test -n "$(FILE)" || (echo "FILE=/path/to/expert-bundle is required" && exit 2)
	nvcc -Iinclude -std=c++20 -O3 -arch=sm_86 src/quant.cpp src/safetensors.cpp cuda/expert_forward_test.cu -lcublas -o $(BUILD_DIR)/cuda-expert-test
	$(BUILD_DIR)/cuda-expert-test --bundle "$(FILE)"

cuda-gpu-cache-test: | $(BUILD_DIR)
	nvcc -Iinclude -std=c++20 -O3 -arch=sm_86 cuda/gpu_cache_test.cu -o $(BUILD_DIR)/cuda-gpu-cache-test
	$(BUILD_DIR)/cuda-gpu-cache-test

clean:
	rm -rf $(BUILD_DIR)
