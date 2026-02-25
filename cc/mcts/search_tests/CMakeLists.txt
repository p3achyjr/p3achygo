# Tests and tools for parallel search.

set(SEARCH_DEPS
    gumbel
    core_impl
    engine_factory
    absl::flags
    absl::flags_parse
    absl::log
)

# Tests
if(BUILD_TESTS)
    add_executable(search_test search_test.cc)
    target_link_libraries(search_test PRIVATE
        ${SEARCH_DEPS}
        doctest::doctest
    )
    add_test(NAME search_test COMMAND search_test)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_executable(search_benchmark search_benchmark.cc)
    target_link_libraries(search_benchmark PRIVATE ${SEARCH_DEPS})
endif()

# Game-playing binary (always built with binaries)
if(BUILD_BINARIES)
    add_executable(play_game play_game.cc)
    target_link_libraries(play_game PRIVATE ${SEARCH_DEPS} absl::log_initialize m dl)
endif()
