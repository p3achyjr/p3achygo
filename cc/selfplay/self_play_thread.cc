#include "cc/selfplay/self_play_thread.h"

#include <chrono>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/core/file_log_sink.h"
#include "cc/core/probability.h"
#include "cc/core/ring_buffer.h"
#include "cc/game/game.h"
#include "cc/mcts/gumbel.h"
#include "cc/recorder/game_recorder.h"

#define LOG_TO_SINK(severity, sink) LOG(severity).ToSinkOnly(&sink)

namespace selfplay {
namespace {
using namespace ::game;
using namespace ::core;
using namespace ::mcts;
using namespace ::nn;
using namespace ::recorder;

struct GumbelParams {
  int n;
  int k;
};

struct InitState {
  Board board;
  absl::InlinedVector<Move, constants::kMaxGameLen> last_moves;
  Color color_to_move;
};

// Whether the current thread should log.
static constexpr int kShouldLogShard = 8;

// Max size of seen state buffer, as described in
// https://arxiv.org/pdf/2302.12359.pdf.
static constexpr int kGoExploitBufferSize = 64;

// Probability to add any state to the seen buffer.
static constexpr float kAddSeenStateProb = .02f;

// Probability of drawing a state from the visited buffer.
static constexpr float kUseSeenStateProb = .5f;

// Max Number of beginning moves to sample directly.
static constexpr int kMaxNumRawPolicyMoves = 30;

// Thresholds at which to compute pass-alive regions.
static constexpr int kComputePAMoveNums[] = {175, 200, 250, 300, 350, 400};

// Possible values of N, K for gumbel search.
static const GumbelParams kGumbelParamChoices[] = {
    // GumbelParams{8, 2},   GumbelParams{16, 2},  GumbelParams{16, 4},
    GumbelParams{32, 4}, GumbelParams{32, 4}, GumbelParams{32, 4},
    // GumbelParams{48, 8},  GumbelParams{48, 8},  GumbelParams{48, 8},
    // GumbelParams{64, 4},  GumbelParams{64, 16}, GumbelParams{96, 8},
    // GumbelParams{128, 2}, GumbelParams{128, 4},
};

// Size of `kGumbelParamsChoices`.
static const size_t kNumGumbelChoices =
    sizeof(kGumbelParamChoices) / sizeof(GumbelParams);

// Threshold beneath which loss is guaranteed.
static constexpr float kDownBadThreshold = -.95;

// Number of moves at down bad threshold before decreasing visit count.
static constexpr float kNumDownBadMovesThreshold = 5;

// Minimal visit count for guaranteed lost games.
static constexpr GumbelParams kDownBadParams = GumbelParams{4, 2};

// Whether the thread should continue running.
static std::atomic<bool> running = true;

void AddNewInitState(RingBuffer<InitState, kGoExploitBufferSize>& buffer,
                     const Game& game, Color color_to_move) {
  Board board = game.board();
  absl::InlinedVector<Move, constants::kMaxGameLen> last_moves = game.moves();

  CHECK(last_moves.size() == constants::kNumLastMoves);
  CHECK(last_moves[last_moves.size() - 5] ==
        game.moves()[game.moves().size() - 5]);
  CHECK(last_moves[last_moves.size() - 4] ==
        game.moves()[game.moves().size() - 4]);
  CHECK(last_moves[last_moves.size() - 3] ==
        game.moves()[game.moves().size() - 3]);
  CHECK(last_moves[last_moves.size() - 2] ==
        game.moves()[game.moves().size() - 2]);
  CHECK(last_moves[last_moves.size() - 1] ==
        game.moves()[game.moves().size() - 1]);

  buffer.Append(InitState{board, last_moves, color_to_move});
}

InitState GetInitState(Probability& probability,
                       RingBuffer<InitState, kGoExploitBufferSize>& buffer) {
  InitState s_0 = InitState{
      Board(), {kNoopMove, kNoopMove, kNoopMove, kNoopMove, kNoopMove}, BLACK};
  if (probability.Uniform() <= kUseSeenStateProb) {
    std::optional<InitState> seen_state = buffer.Pop();
    if (!seen_state) {
      return s_0;
    }

    return seen_state.value();
  }

  return s_0;
}

}  // namespace

void Run(size_t seed, int thread_id, NNInterface* nn_interface,
         GameRecorder* game_recorder, std::string logfile, int max_moves) {
  FileSink sink(logfile.c_str());
  Probability probability(seed);
  RingBuffer<InitState, kGoExploitBufferSize> go_exploit_buffer;
  auto search_dur_ema = 0;

  // Main loop.
  while (true) {
    // Populate initial state either from seen states or s_0.
    InitState init_state = GetInitState(probability, go_exploit_buffer);

    // Game state.
    Game game(init_state.board, init_state.last_moves);
    Color color_to_move = init_state.color_to_move;

    // Search Tree.
    std::unique_ptr<TreeNode> root_node = std::make_unique<TreeNode>();

    // Completed Q-values for each timestep.
    std::vector<std::array<float, constants::kMaxNumMoves>> mcts_pis;

    // Whether the i'th move is trainable.
    std::vector<uint8_t> move_trainables;

    // Number of consecutive moves at which we are beneath kDownBadThreshold.
    int num_consecutive_down_bad_moves_b = 0;
    int num_consecutive_down_bad_moves_w = 0;

    // Number of moves for which to sample directly from the policy.
    auto num_moves_raw_policy =
        RandRange(probability.prng(), 0, kMaxNumRawPolicyMoves);

    // Gumbel N, K for this game.
    GumbelParams gumbel_params = kGumbelParamChoices[RandRange(
        probability.prng(), 0, kNumGumbelChoices)];

    // Begin Search.
    GumbelEvaluator gumbel_evaluator(nn_interface, thread_id);
    while (IsRunning() && !game.IsGameOver() && game.num_moves() < max_moves) {
      // Choose n, k = 1 if we have not reached `num_moves_raw_policy` number of
      // moves.
      // Choose n, k = kDownBadParams if we are down bad.
      int num_consecutive_down_bad_moves =
          color_to_move == BLACK ? num_consecutive_down_bad_moves_b
                                 : num_consecutive_down_bad_moves_w;
      bool sampling_raw_policy = game.num_moves() <= num_moves_raw_policy;
      bool is_down_bad =
          num_consecutive_down_bad_moves >= kNumDownBadMovesThreshold;
      int gumbel_n = sampling_raw_policy
                         ? 1
                         : (is_down_bad ? kDownBadParams.n : gumbel_params.n);
      int gumbel_k = sampling_raw_policy
                         ? 1
                         : (is_down_bad ? kDownBadParams.k : gumbel_params.k);

      // Run and Profile Search.
      auto begin = std::chrono::high_resolution_clock::now();
      GumbelResult gumbel_res =
          gumbel_evaluator.SearchRoot(probability, game, root_node.get(),
                                      color_to_move, gumbel_n, gumbel_k);
      auto end = std::chrono::high_resolution_clock::now();

      // Gather statistics.
      Loc nn_move = gumbel_res.nn_move;
      Loc move = gumbel_res.mcts_move;
      int nn_move_n = NAction(root_node.get(), nn_move);
      float nn_move_q = QAction(root_node.get(), nn_move);
      int move_n = NAction(root_node.get(), move);
      float move_q = QAction(root_node.get(), move);
      float root_q_outcome = Q(root_node.get());
      bool is_move_trainable = !sampling_raw_policy && !is_down_bad;

      // Update tracking data structures.
      mcts_pis.push_back(gumbel_res.pi_improved);
      move_trainables.push_back(is_move_trainable);
      num_consecutive_down_bad_moves +=
          (root_q_outcome < kDownBadThreshold ? 1 : 0);

      // Play move, and calculate PA regions if we hit a checkpoint.
      game.PlayMove(move, color_to_move);
      if (std::find(std::begin(kComputePAMoveNums),
                    std::end(kComputePAMoveNums), game.num_moves())) {
        game.CalculatePassAliveRegions();
      }
      color_to_move = OppositeColor(color_to_move);

      root_node = std::move(root_node->children[move]);
      if (!root_node) {
        // this is possible if we draw directly from policy, or if pass is the
        // only legal move found in search.
        root_node = std::make_unique<TreeNode>();
      }

      // Add to seen state buffer.
      if (probability.Uniform() <= kAddSeenStateProb) {
        AddNewInitState(go_exploit_buffer, game, color_to_move);
      }

      auto search_dur =
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
      search_dur_ema = search_dur_ema == 0
                           ? search_dur
                           : (search_dur_ema * 0.9 + search_dur * 0.1);

      if (thread_id % kShouldLogShard == 0) {
        LOG_TO_SINK(INFO, sink) << "-------------------";
        LOG_TO_SINK(INFO, sink) << "N: " << gumbel_n << ", K: " << gumbel_k;
        LOG_TO_SINK(INFO, sink) << "Raw NN Move: " << nn_move
                                << ", q: " << nn_move_q << ", n: " << nn_move_n;
        LOG_TO_SINK(INFO, sink) << "Gumbel Move: " << move << ", q: " << move_q
                                << ", n: " << move_n;
        LOG_TO_SINK(INFO, sink) << "Move Num: " << game.num_moves();
        LOG_TO_SINK(INFO, sink)
            << "Last 5 Moves: " << game.move(game.num_moves() - 5) << ", "
            << game.move(game.num_moves() - 4) << ", "
            << game.move(game.num_moves() - 3) << ", "
            << game.move(game.num_moves() - 2) << ", "
            << game.move(game.num_moves() - 1);
        LOG_TO_SINK(INFO, sink)
            << "Tree Visit Count: " << root_node->n
            << " Player to Move: " << root_node->color_to_move
            << " Value: " << root_node->q
            << " Outcome: " << root_node->q_outcome;
        LOG_TO_SINK(INFO, sink) << "Board:\n" << game.board();
        LOG_TO_SINK(INFO, sink)
            << "Search Took " << search_dur
            << "us. Search EMA: " << search_dur_ema << "us.";
      }
    }

    nn_interface->UnregisterThread(thread_id);
    if (!IsRunning()) break;

    game.WriteResult();

    LOG_TO_SINK(INFO, sink) << "Black Score: " << game.result().bscore;
    LOG_TO_SINK(INFO, sink) << "White Score: " << game.result().wscore;

    auto begin = std::chrono::high_resolution_clock::now();
    game_recorder->RecordGame(thread_id, init_state.board, game, mcts_pis,
                              move_trainables);
    auto end = std::chrono::high_resolution_clock::now();

    LOG_TO_SINK(INFO, sink)
        << "Recording Game Took "
        << std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
               .count()
        << "us";

    // Threads start off auto-registered, so doing this at the beginning of the
    // loop is incorrect.
    nn_interface->RegisterThread(thread_id);
  }
}

void SignalStop() { running.store(false, std::memory_order_release); }

bool IsRunning() { return running.load(std::memory_order_acquire); }

}  // namespace selfplay

#undef LOG_TO_SINK
