# Experiments - utilities for running custom experiments

option(BUILD_BINARIES "Build binary executables" ON)
if(BUILD_BINARIES)
    add_executable(experiments_main
        main.cc
        play_games.cc
        bias_callback.cc
    )
    target_link_libraries(experiments_main PRIVATE
        core_impl
        game
        gumbel
        node_table
        nn_interface
        engine_factory
        absl::flags
        absl::flags_parse
        absl::log
        absl::log_initialize
        m
        dl
    )
    set_target_properties(experiments_main PROPERTIES OUTPUT_NAME "experiments")

    add_executable(gumbel_visits_study
        gumbel_visits_study.cc
    )
    target_link_libraries(gumbel_visits_study PRIVATE
        core_impl
        game
        gumbel
        node_table
        nn_interface
        engine_factory
        tfrecord
        tf_example_proto
        absl::flags
        absl::flags_parse
        absl::log
        absl::log_initialize
        m
        dl
    )
endif()
