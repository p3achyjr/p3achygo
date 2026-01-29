# Engine scripts and utilities

# Compare engines utility
if(BUILD_BINARIES)
    add_executable(compare_engines
        compare_engines.cc
    )
    target_link_libraries(compare_engines PRIVATE
        engine_factory
        benchmark_engine
        go_dataset
        tf_engine
        trt_engine
        absl::flags
        absl::flags_parse
        absl::log
        absl::check
    )
endif()

# Build and run TensorRT engine utility
# Only build if TensorRT is enabled (requires calibrator support)
if(BUILD_BINARIES AND USE_CUDA AND USE_TENSORRT)
    add_executable(build_and_run_trt_engine
        build_and_run_trt_engine.cc
    )
    target_link_libraries(build_and_run_trt_engine PRIVATE
        benchmark_engine
        engine_factory
        go_dataset
        go_features
        trt_engine
        trt
        trt_utils
        trt_deps
        CUDA::cudart
        absl::flags
        absl::flags_parse
        absl::log
        absl::check
    )

    # Add TensorRT include directories
    if (USE_TENSORRT_RTX)
        target_include_directories(build_and_run_trt_engine PRIVATE
            /opt/tensorrt-rtx/include
        )
        target_link_libraries(build_and_run_trt_engine PRIVATE
            ${TENSORRT_NVONNXPARSER_LIB}
        )
    else()
        target_include_directories(build_and_run_trt_engine PRIVATE
            /opt/tensorrt/include
        )
        target_link_libraries(build_and_run_trt_engine PRIVATE
            ${TENSORRT_NVONNXPARSER_LIB}
        )
    endif()
endif()
