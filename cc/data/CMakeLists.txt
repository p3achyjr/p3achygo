# Data library - dataset creation from SGF files

# TFRecord reader/writer (minimal port from TensorFlow)
add_subdirectory(tfrecord)

# Filename format - header-only
add_library(data_filename_format INTERFACE)
target_sources(data_filename_format INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/filename_format.h)

# Dataset creator (uses standalone protobuf, not TensorFlow)
add_library(dataset_creator
    coordinator.cc
    coordinator.h
    worker.cc
    worker.h
)
target_link_libraries(dataset_creator PUBLIC
    game
    make_tf_example
    tfrecord
    sgf
    filename_buffer
    absl::check
    absl::str_format
    absl::synchronization
    indicators::indicators
)

# Dataset creation binary
if(BUILD_BINARIES)
    add_executable(data_main main.cc)
    target_link_libraries(data_main PRIVATE
        dataset_creator
        absl::flags
        absl::flags_parse
        absl::log
        absl::log_initialize
    )
    set_target_properties(data_main PROPERTIES OUTPUT_NAME "dataset_creator")
endif()
