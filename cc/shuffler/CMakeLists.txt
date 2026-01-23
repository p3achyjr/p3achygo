# Filename buffer
add_library(filename_buffer
    filename_buffer.cc
    filename_buffer.h
)
target_link_libraries(filename_buffer PUBLIC
    constants
    core_impl
    tensorflow
    absl::flat_hash_set
)

# Shuffler main library
add_library(shuffler
    chunk_info.cc
    chunk_info.h
    chunk_manager.cc
    chunk_manager.h
    constants.h
    tf_record_watcher.cc
    tf_record_watcher.h
)
target_link_libraries(shuffler PUBLIC
    filename_buffer
    core_impl
    data_filename_format
    tensorflow
    absl::flat_hash_set
    absl::log
    absl::check
    absl::status
    absl::strings
    absl::time
)

# Shuffler binary
if(BUILD_BINARIES)
    add_executable(shuffler_main main.cc)
    target_link_libraries(shuffler_main PRIVATE
        shuffler
        absl::flags
        absl::flags_parse
        absl::log
        absl::log_initialize
    )
    set_target_properties(shuffler_main PROPERTIES OUTPUT_NAME "shuffler")
endif()
