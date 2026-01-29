cmake_minimum_required(VERSION 3.15)
project(p3achygo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# Place all executables in the top-level build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_BINARIES "Build binary executables" ON)
option(BOARD_SIZE_9 "Use 9x9 board" OFF)
option(USE_CUDA "Enable CUDA support" ON)
option(USE_TENSORFLOW "Enable TensorFlow support" OFF)
option(USE_TENSORRT "Enable TensorRT support" OFF)
option(USE_ASAN "Enable AddressSanitizer" OFF)
option(USE_LTO "Enable Link Time Optimization" OFF)
option(USE_MIMALLOC "Use mimalloc allocator" OFF)

# Set board size
if(BOARD_SIZE_9)
    add_compile_definitions(BOARD_LEN=9)
else()
    add_compile_definitions(BOARD_LEN=19)
endif()

# AddressSanitizer configuration
if(USE_ASAN)
    message(STATUS "Enabling AddressSanitizer (ASan)")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
endif()

# Link Time Optimization
if(USE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        message(STATUS "Enabling Link Time Optimization (LTO)")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "LTO is not supported: ${ipo_error}")
    endif()
endif()

# Mimalloc allocator
if(USE_MIMALLOC)
    find_library(MIMALLOC_LIB mimalloc PATHS /usr/local/lib REQUIRED)
    message(STATUS "Using mimalloc: ${MIMALLOC_LIB}")
    # Create interface library for mimalloc
    add_library(mimalloc_iface INTERFACE)
    target_link_libraries(mimalloc_iface INTERFACE ${MIMALLOC_LIB})
endif()

# Find packages
find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

# Fetch external dependencies
include(FetchContent)

# Abseil - must match TensorFlow's absl version (20230802)
FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20230802.1
)
set(ABSL_PROPAGATE_CXX_STD ON)
FetchContent_MakeAvailable(absl)

# Protobuf - v21.9 corresponds to protobuf 3.21.9 (TensorFlow 2.16 version)
FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v21.9
    GIT_SUBMODULES ""
)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disable protobuf tests")
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "Build static protobuf")
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Disable zlib in protobuf")
FetchContent_MakeAvailable(protobuf)

# Doctest (for testing)
if(BUILD_TESTS)
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG v2.4.11
    )
    FetchContent_MakeAvailable(doctest)
endif()

# Google Benchmark (for benchmarks)
if(BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests")
    FetchContent_MakeAvailable(benchmark)
endif()

# Indicators (progress bars)
FetchContent_Declare(
    indicators
    GIT_REPOSITORY https://github.com/p-ranav/indicators.git
    GIT_TAG v2.3
)
FetchContent_MakeAvailable(indicators)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})

# Enable testing if requested
if(BUILD_TESTS)
    enable_testing()
endif()

if(USE_CUDA AND (USE_TENSORRT_RTX OR USE_TENSORRT))
    # Full TensorRT implementation (requires CUDA and TensorFlow)
    find_package(CUDAToolkit REQUIRED)

    if(USE_TENSORRT_RTX)
        # RTX variant
        set(TENSORRT_ROOT "/opt/tensorrt-rtx")
        set(_TENSORRT_LIBNAME "tensorrt_rtx")
        set(_NVONNXPARSER_LIBNAME "tensorrt_onnxparser_rtx")
    else()
        # Standard TensorRT
        set(TENSORRT_ROOT "/opt/tensorrt")
        set(_TENSORRT_LIBNAME "nvinfer")
        set(_NVONNXPARSER_LIBNAME "nvonnxparser")
    endif()

    find_library(TENSORRT_LIB
        NAMES ${_TENSORRT_LIBNAME}
        PATHS "${TENSORRT_ROOT}/lib"
        NO_DEFAULT_PATH
        REQUIRED
    )

    find_library(TENSORRT_NVONNXPARSER_LIB
        NAMES ${_NVONNXPARSER_LIBNAME}
        PATHS "${TENSORRT_ROOT}/lib"
        NO_DEFAULT_PATH
        REQUIRED
    )
endif()

add_library(trt_deps INTERFACE)

target_link_libraries(trt_deps INTERFACE
  ${TENSORRT_LIB}
)

# Force DT_NEEDED entry even if the linker thinks it's unused
target_link_options(trt_deps INTERFACE
  "LINKER:--no-as-needed"
)

# Add subdirectories for all modules
# Order matters - dependencies must be added before dependents
add_subdirectory(cc/constants)
add_subdirectory(cc/core)
add_subdirectory(cc/game)
add_subdirectory(cc/sgf)
add_subdirectory(cc/proto)
add_subdirectory(cc/tensorflow)
add_subdirectory(cc/nn)
add_subdirectory(cc/mcts)
add_subdirectory(cc/analysis)
add_subdirectory(cc/recorder)
add_subdirectory(cc/shuffler)
add_subdirectory(cc/data)
add_subdirectory(cc/selfplay)
add_subdirectory(cc/eval)
add_subdirectory(cc/gtp)

# Print configuration summary
message(STATUS "========================================")
message(STATUS "P3achyGo Configuration Summary")
message(STATUS "========================================")
if(BOARD_SIZE_9)
    message(STATUS "Board size: 9x9")
else()
    message(STATUS "Board size: 19x19")
endif()
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "Build binaries: ${BUILD_BINARIES}")
message(STATUS "CUDA support: ${USE_CUDA}")
message(STATUS "TensorFlow support: ${USE_TENSORFLOW}")
message(STATUS "TensorRT support: ${USE_TENSORRT}")
message(STATUS "AddressSanitizer: ${USE_ASAN}")
message(STATUS "Link Time Optimization: ${USE_LTO}")
message(STATUS "Mimalloc allocator: ${USE_MIMALLOC}")
if (USE_TENSORFLOW)
    message(STATUS "========================================")
    message(STATUS "Note: TensorFlow is required and expected in cc/tensorflow/")
    message(STATUS "========================================")
endif()
