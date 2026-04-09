"""
Various constants.
"""

BOARD_LEN = 19

SCORE_RANGE_MIDPOINT = 400
SCORE_RANGE = 800

## Game Representation Constants
BOARD_LEN = 19
EMPTY = 0
BLACK = 1
WHITE = -1

NON_MOVE = (-1, -1)
PASS_MOVE = (BOARD_LEN, 0)

PASS_MOVE_ENCODING = 361
NUM_MOVES = BOARD_LEN * BOARD_LEN + 1  # 362
NUM_V_BUCKETS = 51  # must match mcts::kNumVBuckets in C++


def num_input_planes(version=1) -> int:
    """Returns number of input planes for a given model version."""
    return 15


def num_input_features(version=1) -> int:
    """Returns number of input planes for a given model version."""
    return 8
