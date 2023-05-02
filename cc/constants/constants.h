#ifndef __CONSTANTS_CONSTANTS_H_
#define __CONSTANTS_CONSTANTS_H_

/* Length of one side of board */
#ifndef BOARD_LEN
#define BOARD_LEN 19
#endif

/* Number of states per board location */
#ifndef NUM_STATES
#define NUM_STATES 3
#endif

/* Game Related Constants */
#ifndef EMPTY
#define EMPTY 0
#endif

#ifndef BLACK
#define BLACK 1
#endif

#ifndef WHITE
#define WHITE -1
#endif

namespace constants {

/* Maximum number of board locations */
static constexpr auto kMaxNumBoardLocs = BOARD_LEN * BOARD_LEN;

/* Maximum number of moves allowed per board state */
static constexpr auto kMaxNumMoves = BOARD_LEN * BOARD_LEN + 1;

/* Maximum number of moves per game */
static constexpr auto kMaxGameLen = 600;

/* Number of feature planes for neural network */
static constexpr auto kNumInputFeaturePlanes = 7;

/* Number of score logits from neural network */
static constexpr auto kNumValueLogits = 2;

/* Number of score logits from neural network */
static constexpr auto kNumScoreLogits = 800;

/* Score inflection point */
static constexpr auto kScoreInflectionPoint = 400;

/* Number of most recent moves to feed to NN */
static constexpr auto kNumLastMoves = 5;

/* Integer value for pass move encoding, from NN. */
static constexpr auto kPassMoveEncoding = 361;

/* Integer value for an illegal move encoding. */
static constexpr auto kNoopMoveEncoding = -1;

/* Number of Passes before forbidding moves in pass-alive regions. */
static constexpr auto kNumPassesBeforeBensons = 2;

}  // namespace constants

#endif  // __CONSTANTS_CONSTANTS_H_
