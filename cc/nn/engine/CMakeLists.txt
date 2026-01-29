# Neural network engine library

# Go features encoding
add_library(go_features
    go_features.cc
    go_features.h
    buf_utils.h
)
target_link_libraries(go_features PUBLIC
    constants
    game_hdrs
    absl::status
)


# go_dataset (uses standalone protobuf, not TensorFlow)
add_library(go_dataset
    go_dataset.cc
    go_dataset.h
)
target_link_libraries(go_dataset PUBLIC
    go_features
    tfrecord
    tf_example_proto
    constants
    game_hdrs
    absl::check
)

# Engine header-only interface
add_library(engine_hdr INTERFACE)
target_sources(engine_hdr INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/engine.h)
target_link_libraries(engine_hdr INTERFACE go_features)

# TensorFlow engine - only needs TensorFlow when actually building
if(USE_TENSORFLOW)
    add_library(tf_engine
        tf_engine.cc
        tf_engine.h
    )
    target_link_libraries(tf_engine PUBLIC
        engine_hdr
        game_hdrs
        tensorflow
        absl::flat_hash_map
        absl::status
    )
else()
    # Dummy tf_engine for non-TensorFlow builds
    add_library(tf_engine
        tf_engine_dummy.cc
        tf_engine.h
    )
    target_link_libraries(tf_engine PUBLIC
        engine_hdr
        game_hdrs
        absl::log
        absl::status
    )
endif()

# TensorRT engine (with CPU-only dummy variant)
if(USE_CUDA AND (USE_TENSORRT_RTX OR USE_TENSORRT))
    # Full TensorRT implementation (requires CUDA and TensorFlow)
    add_library(trt_utils
        trt_logger.cc
        trt_logger.h
        trt_names.h
    )
    target_link_libraries(trt_utils PUBLIC
        CUDA::cudart
    )

    # Add TensorRT include directories
    if (USE_TENSORRT_RTX)
        target_include_directories(trt_utils PUBLIC
            /opt/tensorrt-rtx/include
        )
    else()
        target_include_directories(trt_utils PUBLIC
            /opt/tensorrt/include
        )
    endif()

    add_library(trt_engine
        trt_engine.cc
        trt_engine.h
    )
    target_link_libraries(trt_engine PUBLIC
        engine_hdr
        trt_utils
        game_hdrs
        trt_deps
        CUDA::cudart
        absl::log
        absl::check
    )

    # TRT calibrator library - only build if not using RTX
    # (RTX version doesn't support calibrator API, but it's only needed for model building, not inference)
    if(USE_TENSORRT)
        add_library(trt
            trt_calibrator.cc
            trt_calibrator.h
        )
        target_link_libraries(trt PUBLIC
            go_dataset
            go_features
            trt_engine
            trt_utils
            constants
            game_hdrs
            CUDA::cudart
            absl::log
            absl::check
            absl::status
        )
    else()
        # Empty trt library for RTX (calibration not supported/needed for inference)
        add_library(trt INTERFACE)
        target_link_libraries(trt INTERFACE trt_engine)
        message(STATUS "Skipping TRT calibrator build (not supported with TensorRT-RTX)")
    endif()
else()
    # CPU-only dummy implementation
    add_library(trt_engine
        trt_engine_dummy.cc
        trt_engine.h
    )
    target_link_libraries(trt_engine PUBLIC
        engine_hdr
        absl::log
        absl::check
    )

    # Empty trt library for compatibility
    add_library(trt INTERFACE)
    target_link_libraries(trt INTERFACE trt_engine)
endif()

# Engine factory
add_library(engine_factory
    engine_factory.cc
    engine_factory.h
)
target_link_libraries(engine_factory PUBLIC
    engine_hdr
    tf_engine
    trt_engine
    absl::log
)

# Benchmark engine
add_library(benchmark_engine
    benchmark_engine.cc
    benchmark_engine.h
)
target_link_libraries(benchmark_engine PUBLIC
    engine_hdr
    go_dataset
    absl::log
)

# Scripts and utilities
add_subdirectory(scripts)
