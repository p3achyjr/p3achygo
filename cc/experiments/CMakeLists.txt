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

    add_executable(reuse_visits_study
        reuse_visits_study.cc
    )
    target_link_libraries(reuse_visits_study PRIVATE
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
        absl::synchronization
        m
        dl
    )

    add_executable(root_net_err
        root_net_err.cc
    )
    target_link_libraries(root_net_err PRIVATE
        core_impl
        game
        nn_interface
        engine_factory
        tfrecord
        tf_example_proto
        absl::flags
        absl::flags_parse
        absl::log
        absl::log_initialize
        absl::synchronization
        m
        dl
    )

    add_executable(value_dist_gallery
        value_dist_gallery.cc
    )
    target_link_libraries(value_dist_gallery PRIVATE
        tfrecord
        tf_example_proto
        absl::flags
        absl::flags_parse
        absl::log
        absl::log_initialize
        absl::flat_hash_set
        absl::strings
        absl::check
        m
    )

    add_executable(value_visit_study
        value_visit_study.cc
    )
    target_link_libraries(value_visit_study PRIVATE
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
    add_executable(opening_sample_study
        opening_sample_study.cc
    )
    target_link_libraries(opening_sample_study PRIVATE
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

    add_executable(upside_var_study
        upside_var_study.cc
    )
    target_link_libraries(upside_var_study PRIVATE
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
    add_executable(scoring_compare scoring_compare.cc)
    target_link_libraries(scoring_compare PRIVATE
        core_impl
        game
        gumbel
        node_table
        nn_interface
        engine_factory
        sgf
        absl::flags
        absl::flags_parse
        absl::log
        absl::log_initialize
        m
        dl
    )
endif()
