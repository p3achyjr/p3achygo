# SGF (Smart Game Format) library

# SGF tree - header-only
add_library(sgf_tree INTERFACE)
target_sources(sgf_tree INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/sgf_tree.h)
target_link_libraries(sgf_tree INTERFACE game)

# SGF visitor - header-only
add_library(sgf_visitor INTERFACE)
target_sources(sgf_visitor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/sgf_visitor.h)

# SGF serializer
add_library(sgf_serializer
    sgf_serializer.cc
    sgf_serializer.h
)
target_link_libraries(sgf_serializer PUBLIC
    sgf_tree
    sgf_visitor
    constants
    game
    absl::strings
)

# SGF parser
add_library(parse_sgf
    parse_sgf.cc
    parse_sgf.h
)
target_link_libraries(parse_sgf PUBLIC
    sgf_tree
    sgf_visitor
    constants
    core_impl
    absl::statusor
    absl::strings
    absl::str_format
)

# SGF test data - header-only
add_library(parse_sgf_test_data INTERFACE)
target_sources(parse_sgf_test_data INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/parse_sgf_test_data.h)

# SGF umbrella library
add_library(sgf INTERFACE)
target_link_libraries(sgf INTERFACE
    parse_sgf
    sgf_serializer
    sgf_tree
    sgf_visitor
)

# Tests
if(BUILD_TESTS)
    add_executable(sgf_serializer_test __tests__/sgf_serializer_test.cc)
    target_link_libraries(sgf_serializer_test PRIVATE
        sgf_serializer
        core_doctest
    )
    target_link_libraries(sgf_serializer_test PRIVATE m)
    add_test(NAME sgf_serializer_test COMMAND sgf_serializer_test)

    add_executable(sgf_parse_test __tests__/parse_sgf_test.cc)
    target_link_libraries(sgf_parse_test PRIVATE
        parse_sgf
        parse_sgf_test_data
        sgf_serializer
        core_doctest
    )
    add_test(NAME sgf_parse_test COMMAND sgf_parse_test)
endif()
