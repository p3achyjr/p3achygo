# Neural network library

# Add the engine subdirectory first
add_subdirectory(engine)

# Create tensor shape utilities - header-only
add_library(create_tensor_shape INTERFACE)
target_sources(create_tensor_shape INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/create_tensor_shape.h)
target_link_libraries(create_tensor_shape INTERFACE tensorflow)

# NN board utilities - only build with TensorFlow as it uses TensorFlow headers
if(USE_TENSORFLOW)
    add_library(nn_board_utils
        nn_board_utils.cc
        nn_board_utils.h
    )
    target_link_libraries(nn_board_utils PUBLIC
        create_tensor_shape
        constants
        game
        tensorflow
        absl::status
    )
else()
    # Dummy nn_board_utils for non-TensorFlow builds
    add_library(nn_board_utils INTERFACE)
    target_sources(nn_board_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/nn_board_utils.h)
    target_link_libraries(nn_board_utils INTERFACE
        create_tensor_shape
        constants
        game
        absl::status
    )
endif()

# Full implementation with TensorFlow
add_library(nn_interface
    nn_interface.cc
    nn_interface.h
)

target_link_libraries(nn_interface
  PUBLIC
    constants
    core_cache
    game
    engine_hdr
    go_features
    absl::log
    absl::status
    absl::inlined_vector
)

# NN umbrella library
add_library(nn INTERFACE)
target_link_libraries(nn INTERFACE nn_interface)

# Tests
if(BUILD_TESTS)
    add_executable(nn_interface_sync_test nn_interface_sync_test.cc)
    target_link_libraries(nn_interface_sync_test PRIVATE
        nn_interface
        game
        core_impl
        core_doctest
    )
    add_test(NAME nn_interface_sync_test COMMAND nn_interface_sync_test)
    set_tests_properties(nn_interface_sync_test PROPERTIES TIMEOUT 265)
endif()

if(BUILD_TESTS AND USE_TENSORFLOW)
    add_executable(nn_board_utils_test nn_board_utils_test.cc)
    target_link_libraries(nn_board_utils_test PRIVATE
        create_tensor_shape
        nn_board_utils
        core_doctest
        tensorflow
    )
    add_test(NAME nn_board_utils_test COMMAND nn_board_utils_test)
endif()
