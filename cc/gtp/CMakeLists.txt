# GTP (Go Text Protocol) library

add_library(gtp
    client.cc
    client.h
    command.cc
    command.h
    parse.cc
    parse.h
    service.cc
    service.h
)
target_link_libraries(gtp PUBLIC
    analysis
    core_impl
    mcts
    nn
    engine_factory
    sgf_recorder
    sgf
    absl::log
    absl::check
    absl::status
)

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
    set_target_properties(gtp_main PROPERTIES OUTPUT_NAME "gtp")
endif()
