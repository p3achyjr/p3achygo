# TFRecord reader/writer library - minimal port from TensorFlow
# Provides TFRecord format I/O without requiring full TensorFlow dependency

find_package(ZLIB REQUIRED)

add_library(tfrecord
    record_reader.cc
    record_writer.cc
)

target_include_directories(tfrecord PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tfrecord PUBLIC
    absl::crc32c
    absl::status
    absl::statusor
    absl::strings
    ZLIB::ZLIB
)
