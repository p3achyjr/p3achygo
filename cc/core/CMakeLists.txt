# Core utilities library

# Header-only libraries
add_library(core_cache INTERFACE)
target_sources(core_cache INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cache.h)
target_link_libraries(core_cache INTERFACE absl::hash)

add_library(core_elo INTERFACE)
target_sources(core_elo INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/elo.h)

add_library(core_file_log_sink INTERFACE)
target_sources(core_file_log_sink INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/file_log_sink.h)
target_link_libraries(core_file_log_sink INTERFACE
    absl::log
    absl::strings
)

add_library(core_util INTERFACE)
target_sources(core_util INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/util.h)
target_link_libraries(core_util INTERFACE
    absl::flat_hash_map
    absl::flat_hash_set
    absl::inlined_vector
)

add_library(core_types INTERFACE)
target_sources(core_types INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/types.h)

add_library(core_filepath INTERFACE)
target_sources(core_filepath INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/filepath.h)

add_library(core_ring_buffer INTERFACE)
target_sources(core_ring_buffer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/ring_buffer.h)
target_link_libraries(core_ring_buffer INTERFACE core_rand)

add_library(core_lru_cache INTERFACE)
target_sources(core_lru_cache INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lru_cache.h)
target_link_libraries(core_lru_cache INTERFACE absl::flat_hash_set)

add_library(core_vmath INTERFACE)
target_sources(core_vmath INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vmath.h)
target_compile_options(core_vmath INTERFACE -mavx -fno-math-errno)

add_library(core_doctest INTERFACE)
target_sources(core_doctest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/doctest_include.h)
target_link_libraries(core_doctest INTERFACE doctest::doctest)

# Compiled libraries
add_library(core_rand
    rand.cc
    rand.h
)
target_link_libraries(core_rand PUBLIC absl::int128)

add_library(core_probability
    probability.cc
    probability.h
)
target_link_libraries(core_probability PUBLIC core_rand)

# Core umbrella library
add_library(core_impl INTERFACE)
target_link_libraries(core_impl INTERFACE
    core_cache
    core_elo
    core_file_log_sink
    core_filepath
    core_lru_cache
    core_probability
    core_rand
    core_ring_buffer
    core_util
    core_vmath
    absl::flat_hash_map
    absl::flat_hash_set
    absl::inlined_vector
    absl::strings
    absl::log
)

add_library(core INTERFACE)
target_link_libraries(core INTERFACE core_impl)

# Tests
if(BUILD_TESTS)
    add_executable(core_rand_test rand_test.cc)
    target_link_libraries(core_rand_test PRIVATE
        core_rand
        core_doctest
    )
    add_test(NAME core_rand_test COMMAND core_rand_test)

    add_executable(core_vmath_test vmath_test.cc)
    target_link_libraries(core_vmath_test PRIVATE
        core_vmath
        core_doctest
    )
    add_test(NAME core_vmath_test COMMAND core_vmath_test)

    add_executable(core_probability_test probability_test.cc)
    target_link_libraries(core_probability_test PRIVATE
        core_probability
        core_doctest
    )
    add_test(NAME core_probability_test COMMAND core_probability_test)
endif()
