CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -Iinclude
BUILD_DIR ?= build

LIB_SOURCE := src/quant.cpp src/safetensors.cpp

.PHONY: all test inspect cuda-bench cuda-pager-bench cuda-fp4-test clean

all: $(BUILD_DIR)/outcore-inspect $(BUILD_DIR)/outcore-extract $(BUILD_DIR)/outcore-io-bench $(BUILD_DIR)/test-safetensors $(BUILD_DIR)/test-quant

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/outcore-inspect: $(LIB_SOURCE) tools/model_inspect.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/outcore-extract: $(LIB_SOURCE) tools/extract_bundle.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/outcore-io-bench: tools/io_bench.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/test-safetensors: $(LIB_SOURCE) tests/test_safetensors.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/test-quant: $(LIB_SOURCE) tests/test_quant.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

test: $(BUILD_DIR)/test-safetensors $(BUILD_DIR)/test-quant
	$(BUILD_DIR)/test-safetensors
	$(BUILD_DIR)/test-quant

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

clean:
	rm -rf $(BUILD_DIR)
