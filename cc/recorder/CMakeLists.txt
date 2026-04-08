# Recorder library - for recording games to SGF and TFRecords

# Directory utilities - header-only
add_library(recorder_dir INTERFACE)
target_sources(recorder_dir INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/dir.h)

# SGF recorder (no TensorFlow dependency)
add_library(sgf_recorder
    sgf_recorder.cc
    sgf_recorder.h
)
target_link_libraries(sgf_recorder PUBLIC
    constants
    core_filepath
    game
    mcts
    sgf_serializer
    sgf_tree
    absl::check
    absl::strings
)

# TF Example proto (uses standalone protobuf, not TensorFlow)
add_library(make_tf_example INTERFACE)
target_sources(make_tf_example INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/make_tf_example.h)
target_link_libraries(make_tf_example INTERFACE
    game
    tf_example_proto
)

add_library(tf_recorder
    tf_recorder.cc
    tf_recorder.h
)
target_link_libraries(tf_recorder PUBLIC
    make_tf_example
    tfrecord
    constants
    core_filepath
    game
    absl::check
    absl::status
    absl::strings
)

# Game recorder (orchestrates SGF and TF recording)
add_library(game_recorder
    game_recorder.cc
    game_recorder.h
)
target_link_libraries(game_recorder PUBLIC
    recorder_dir
    sgf_recorder
    tf_recorder
    core_filepath
    game
    absl::log
)

# Tests
if(BUILD_TESTS)
    add_executable(tf_recorder_test __tests__/tf_recorder_test.cc)
    target_link_libraries(tf_recorder_test PRIVATE
        tf_recorder
        game
        constants
        mcts_constants
        core_doctest
    )
    add_test(NAME tf_recorder_test COMMAND tf_recorder_test)
endif()

# Recorder umbrella library
add_library(recorder INTERFACE)
target_link_libraries(recorder INTERFACE
    recorder_dir
    game_recorder
    sgf_recorder
    tf_recorder
    make_tf_example
)
