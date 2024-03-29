#ifndef CONSTANTS_CONSTANTS_H_
#define CONSTANTS_CONSTANTS_H_

#include <cstddef>

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
static constexpr int kNumBoardLocs = BOARD_LEN * BOARD_LEN;

/* Maximum number of moves allowed per board state */
static constexpr int kMaxMovesPerPosition = BOARD_LEN * BOARD_LEN + 1;

/* Maximum number of moves per game */
static constexpr int kMaxGameLen = 600;

/* Number of feature planes for neural network */
static constexpr int kNumInputFeaturePlanes = 13;

/* Number of feature scalars in game state vector */
static constexpr int kNumInputFeatureScalars = 7;

/* Number of score logits from neural network */
static constexpr int kNumValueLogits = 2;

/* Number of score logits from neural network */
static constexpr int kNumScoreLogits = 800;

/* Score inflection point */
static constexpr int kScoreInflectionPoint = 400;

/* Number of most recent moves to feed to NN */
static constexpr int kNumLastMoves = 5;

/* Integer value for pass move encoding, from NN. */
static constexpr int kPassMoveEncoding = 361;

/* Integer value for an illegal move encoding. */
static constexpr int kNoopMoveEncoding = -1;

/* Number of Passes before forbidding moves in pass-alive regions. */
static constexpr int kNumPassesBeforeBensons = 3;

/* Maximum number of threads. */
static constexpr int kMaxNumThreads = 256;

/* Default Max NN Cache Size */
static constexpr size_t kDefaultNNCacheSize = 1048576;

/* Go-Exploit Buffer Size. https://arxiv.org/pdf/2302.12359.pdf. */
static constexpr int kGoExploitBufferSize = 8192;

}  // namespace constants

#endif  // CONSTANTS_CONSTANTS_H_
