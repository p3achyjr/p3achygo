# Game library

# Header-only game types
add_library(game_hdrs INTERFACE)
target_sources(game_hdrs INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/color.h
    ${CMAKE_CURRENT_SOURCE_DIR}/loc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/move.h
)
target_link_libraries(game_hdrs INTERFACE
    constants
    absl::hash
)

# Zobrist hashing
add_library(zobrist
    zobrist.cc
    zobrist.h
)
target_link_libraries(zobrist PUBLIC
    constants
    core_impl
    absl::int128
)

# Board implementation
add_library(board
    board.cc
    board.h
)
target_link_libraries(board PUBLIC
    game_hdrs
    zobrist
    constants
    core_impl
    absl::flat_hash_map
    absl::flat_hash_set
    absl::inlined_vector
    absl::check
)

# Board DSL (for tests)
add_library(board_dsl
    board_dsl.cc
    board_dsl.h
)
target_link_libraries(board_dsl PUBLIC
    board
    game_hdrs
    constants
)

# Symmetry operations
add_library(symmetry
    symmetry.cc
    symmetry.h
)
target_link_libraries(symmetry PUBLIC
    game_hdrs
    core_rand
)

# Game implementation
add_library(game_impl
    game.cc
    game.h
)
target_link_libraries(game_impl PUBLIC
    board
    game_hdrs
    absl::inlined_vector
)

# Game umbrella library
add_library(game INTERFACE)
target_link_libraries(game INTERFACE
    board
    game_impl
    game_hdrs
    symmetry
    zobrist
    constants
    core_impl
)

# Tests
if(BUILD_TESTS)
    add_executable(game_board_test board_test.cc)
    target_link_libraries(game_board_test PRIVATE
        board
        board_dsl
        core_doctest
        absl::flat_hash_set
    )
    target_link_libraries(game_board_test PRIVATE m)
    add_test(NAME game_board_test COMMAND game_board_test)

    add_executable(game_symmetry_test symmetry_test.cc)
    target_link_libraries(game_symmetry_test PRIVATE
        symmetry
        core_doctest
    )
    add_test(NAME game_symmetry_test COMMAND game_symmetry_test)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_executable(game_board_benchmark board_benchmark.cc)
    target_link_libraries(game_board_benchmark PRIVATE
        board
        benchmark::benchmark_main
    )
endif()
