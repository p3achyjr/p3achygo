'''
Various constants.
'''

BOARD_LEN = 19
NUM_INPUT_PLANES = 7
NUM_INPUT_FEATURES = 1

SCORE_RANGE_MIDPOINT = 400
SCORE_RANGE = 800

## Game Representation Constants
BOARD_LEN = 19
EMPTY = 0
BLACK = 1
WHITE = 2
BLACK_RL = 1
WHITE_RL = -1  # whoops :)

NON_MOVE = (-1, -1)
PASS_MOVE = (BOARD_LEN - 1, BOARD_LEN)
PASS_MOVE_RL = (BOARD_LEN, 0)  # whoops :)
