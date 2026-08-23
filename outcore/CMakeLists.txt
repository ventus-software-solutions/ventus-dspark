cmake_minimum_required(VERSION 3.24)

project(ventus_outcore VERSION 0.1.0 LANGUAGES CXX)

add_library(ventus_outcore
    src/byte_cache.cpp
    src/model_store.cpp
    src/quant.cpp
    src/safetensors.cpp
)
target_include_directories(ventus_outcore PUBLIC include)
target_compile_features(ventus_outcore PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(ventus_outcore PRIVATE /W4 /permissive-)
else()
    target_compile_options(ventus_outcore PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(outcore-inspect tools/model_inspect.cpp)
target_link_libraries(outcore-inspect PRIVATE ventus_outcore)

add_executable(outcore-extract tools/extract_bundle.cpp)
target_link_libraries(outcore-extract PRIVATE ventus_outcore)

add_executable(outcore-repack tools/model_repack.cpp)
target_link_libraries(outcore-repack PRIVATE ventus_outcore)

add_executable(outcore-cache-probe tools/cache_probe.cpp)
target_link_libraries(outcore-cache-probe PRIVATE ventus_outcore)

if(UNIX)
    add_executable(outcore-io-bench tools/io_bench.cpp)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test-safetensors tests/test_safetensors.cpp)
    target_link_libraries(test-safetensors PRIVATE ventus_outcore)
    add_test(NAME safetensors-reader COMMAND test-safetensors)

    add_executable(test-quant tests/test_quant.cpp)
    target_link_libraries(test-quant PRIVATE ventus_outcore)
    add_test(NAME quant-oracle COMMAND test-quant)

    add_executable(test-byte-cache tests/test_byte_cache.cpp)
    target_link_libraries(test-byte-cache PRIVATE ventus_outcore)
    add_test(NAME byte-cache COMMAND test-byte-cache)
endif()
