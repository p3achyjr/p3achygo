# Eval library - engine evaluation and matches

add_library(eval
    eval.cc
    eval.h
)
target_link_libraries(eval PUBLIC
    trt_deps
    constants
    core_impl
    game
    mcts
    nn
    recorder
    absl::log
    absl::check
    absl::strings
)

# Eval binary
option(BUILD_BINARIES "Build binary executables" ON)
if(BUILD_BINARIES)
    add_executable(eval_main main.cc)
    target_link_libraries(eval_main PRIVATE
        eval
        core_impl
        nn
        engine_factory
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
    set_target_properties(eval_main PROPERTIES OUTPUT_NAME "eval")
endif()
