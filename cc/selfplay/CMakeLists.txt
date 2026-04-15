# Self-play library - generate training games

add_library(self_play
    self_play_thread.cc
    self_play_thread.h
    book.h
)
target_link_libraries(self_play PUBLIC
    trt_deps
    constants
    core_impl
    game
    mcts
    nn
    engine_factory
    recorder
    absl::flat_hash_map
    absl::log
    absl::check
    absl::strings
)

# Tests
if(BUILD_TESTS)
    add_executable(move_sel_manager_test __tests__/move_sel_manager_test.cc)
    target_link_libraries(move_sel_manager_test PRIVATE
        self_play
        core_doctest
    )
    add_test(NAME move_sel_manager_test COMMAND move_sel_manager_test)
endif()

# Self-play binary
option(BUILD_BINARIES "Build binary executables" ON)
if(BUILD_BINARIES)
    add_executable(selfplay_main main.cc)
    target_link_libraries(selfplay_main PRIVATE
        self_play
        nn
        recorder
        absl::flags
        absl::flags_parse
        absl::hash
        absl::log
        absl::check
        absl::log_initialize
        absl::strings
        m
        dl
    )
    set_target_properties(selfplay_main PROPERTIES OUTPUT_NAME "selfplay")
endif()
