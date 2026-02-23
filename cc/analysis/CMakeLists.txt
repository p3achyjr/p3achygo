# Analysis library

add_library(analysis
    analysis.cc
    analysis.h
)
target_link_libraries(analysis PUBLIC
    constants
    game
    mcts
)
