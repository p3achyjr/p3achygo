# GTP (Go Text Protocol) library

add_library(gtp
    client.cc
    client.h
    command.cc
    command.h
    parse.cc
    parse.h
    pass_guard.cc
    pass_guard.h
    service.cc
    service.h
    time_control.cc
    time_control.h
)
target_link_libraries(gtp PUBLIC
    analysis
    core_impl
    player_config
    mcts
    nn
    engine_factory
    sgf_recorder
    sgf
    absl::log
    absl::check
    absl::status
)

# Tests
if(BUILD_TESTS)
    add_executable(gtp_time_control_test __tests__/time_control_test.cc)
    target_link_libraries(gtp_time_control_test PRIVATE
        gtp
        core_doctest
    )
    add_test(NAME gtp_time_control_test COMMAND gtp_time_control_test)

    add_executable(gtp_pass_guard_test __tests__/pass_guard_test.cc)
    target_link_libraries(gtp_pass_guard_test PRIVATE
        gtp
        core_doctest
    )
    add_test(NAME gtp_pass_guard_test COMMAND gtp_pass_guard_test)
endif()

# GTP binary
option(BUILD_BINARIES "Build binary executables" ON)
if(BUILD_BINARIES)
    add_executable(gtp_main main.cc)
    target_link_libraries(gtp_main PRIVATE
        gtp
        absl::flags
        absl::flags_parse
        m
        dl
    )
    set_target_properties(gtp_main PROPERTIES OUTPUT_NAME "p3achygo")
endif()
