#include "cc/selfplay/self_play_thread.h"

#include <chrono>
#include <sstream>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/core/file_log_sink.h"
#include "cc/core/probability.h"
#include "cc/core/ring_buffer.h"
#include "cc/game/game.h"
#include "cc/mcts/gumbel.h"
#include "cc/recorder/game_recorder.h"
#include "cc/selfplay/go_exploit_buffer.h"

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

// Whether the current thread should log.
static constexpr int kShouldLogShard = 8;

// Probability to add any state to the seen buffer.
static constexpr float kAddSeenStateProb = .02f;

// Probability of drawing a state from the visited buffer.
static constexpr float kUseSeenStateProb = .5f;

// Max Number of beginning moves to sample directly.
static constexpr int kMaxNumRawPolicyMoves = 30;

// Probability of exploration.
static constexpr float kOpeningExploreProb = .95f;

// Thresholds at which to compute pass-alive regions.
static constexpr int kComputePAMoveNums[] = {175, 200, 250, 300, 350, 400};

// Threshold below which all positions should be trained on.
static constexpr int kTrainableMoveLb = 250;

// Threshold above which all positions are trained on with minimum probability.
static constexpr int kTrainableMoveUb = 400;

// Low train probability.
static constexpr float kLowTrainProb = .25;

// Probability that game is selected for full tree logging.
static constexpr float kLogFullTreeProb = .002;

// Possible values of N, K for gumbel search.
static const GumbelParams kGumbelParamChoices[] = {
    GumbelParams{32, 6},
};

// Size of `kGumbelParamsChoices`.
static const size_t kNumGumbelChoices =
    sizeof(kGumbelParamChoices) / sizeof(GumbelParams);

// Values of N, K for moves to store for training (playout cap randomization).
static constexpr GumbelParams kSelectedForTrainingParams = {256, 8};

// Probability that we pick a move for training.
static constexpr float kMoveSelectedForTrainingProb = .25;

// Threshold beneath which we are down bad (should resign).
static constexpr float kDownBadThreshold = -.90;

// Number of moves at down bad threshold before decreasing visit count.
static constexpr float kNumDownBadMovesThreshold = 5;

// Minimal visit count for almost-guaranteed lost games.
static constexpr GumbelParams kDownBadParams = GumbelParams{32, 6};

// Whether the thread should continue running.
static std::atomic<bool> running = true;

void AddNewInitState(GoExploitBuffer* buffer, const Game& game,
                     Color color_to_move) {
  Board board = game.board();
  absl::InlinedVector<Move, constants::kMaxGameLen> last_moves;
  for (int off = constants::kNumLastMoves; off > 0; --off) {
    Move last_move = game.moves()[game.moves().size() - off];
    last_moves.emplace_back(last_move);
  }

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

  buffer->Add(InitState{board, last_moves, color_to_move, game.num_moves()});
}

InitState GetInitState(Probability& probability, GoExploitBuffer* buffer) {
  InitState s_0 = InitState{
      Board(), {kNoopMove, kNoopMove, kNoopMove, kNoopMove, kNoopMove}, BLACK};
  if (probability.Uniform() <= kUseSeenStateProb) {
    std::optional<InitState> seen_state = buffer->Get();
    if (!seen_state) {
      return s_0;
    }

    return seen_state.value();
  }

  return s_0;
}

}  // namespace

void Run(size_t seed, int thread_id, NNInterface* nn_interface,
         GameRecorder* game_recorder, GoExploitBuffer* go_exploit_buffer,
         std::string logfile, int max_moves) {
  FileSink sink(logfile.c_str());
  Probability probability(seed);
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

    // Root Q values for each timestep (outcome only).
    std::vector<float> root_q_outcomes;

    // Whether the i'th move is trainable.
    std::vector<uint8_t> move_trainables;

    // Number of consecutive moves at which we are beneath kDownBadThreshold.
    int num_consecutive_down_bad_moves_b = 0;
    int num_consecutive_down_bad_moves_w = 0;

    // Whether to log full MCTS search trees.
    bool log_mcts_trees = probability.Uniform() < kLogFullTreeProb;

    // Tree roots throughout game (if logging full trees).
    std::vector<std::unique_ptr<TreeNode>> search_roots;

    // Number of moves for which to sample directly from the policy.
    auto num_moves_raw_policy =
        probability.Uniform() < kOpeningExploreProb
            ? RandRange(probability.prng(), 0, kMaxNumRawPolicyMoves)
            : 0;

    // Gumbel N, K for this game (for non-trainable moves).
    GumbelParams default_gumbel_params = kGumbelParamChoices[RandRange(
        probability.prng(), 0, kNumGumbelChoices)];

    // Begin Search.
    GumbelEvaluator gumbel_evaluator(nn_interface, thread_id);
    while (IsRunning() && !game.IsGameOver() && game.num_moves() < max_moves) {
      // Choose n, k = 1 if we have not reached `num_moves_raw_policy` number of
      // moves.
      // Choose n, k = kDownBadParams if we are down bad.
      bool sampling_raw_policy = game.num_moves() <= num_moves_raw_policy;
      bool is_either_down_bad =
          num_consecutive_down_bad_moves_b >= kNumDownBadMovesThreshold ||
          num_consecutive_down_bad_moves_w >= kNumDownBadMovesThreshold;
      bool is_move_selected_for_training = [sampling_raw_policy, &probability,
                                            &root_node]() {
        if (sampling_raw_policy) {
          return false;
        }

        // Use win percentage, downweight roots with very low winrate.
        if (!root_node) {
          // This root is not expanded yet.
          return probability.Uniform() < kMoveSelectedForTrainingProb;
        }

        float root_q = QOutcome(root_node.get());
        if (root_q > kDownBadThreshold) {
          // Do not anneal probability.
          return probability.Uniform() < kMoveSelectedForTrainingProb;
        }

        // Anneal probability linearly.
        float penalty =
            (kDownBadThreshold - root_q) / (kDownBadThreshold + 1.0f);
        float p = penalty * kMoveSelectedForTrainingProb;

        return probability.Uniform() < p;
      }();

      int gumbel_n, gumbel_k;
      if (sampling_raw_policy) {
        gumbel_n = 1;
        gumbel_k = 1;
      } else if (is_move_selected_for_training) {
        gumbel_n = kSelectedForTrainingParams.n;
        gumbel_k = kSelectedForTrainingParams.k;
      } else if (is_either_down_bad) {
        gumbel_n = kDownBadParams.n;
        gumbel_k = kDownBadParams.k;
      } else {
        gumbel_n = default_gumbel_params.n;
        gumbel_k = default_gumbel_params.k;
      }

      // Pre Search Statistics.
      int n_pre = N(root_node.get());
      float q_pre = Q(root_node.get());
      float qz_pre = QOutcome(root_node.get());

      // Run and Profile Search.
      auto begin = std::chrono::high_resolution_clock::now();
      GumbelResult gumbel_res =
          gumbel_evaluator.SearchRoot(probability, game, root_node.get(),
                                      color_to_move, gumbel_n, gumbel_k);
      auto end = std::chrono::high_resolution_clock::now();

      // Post Search Statstics.
      int n_post = N(root_node.get());
      float q_post = Q(root_node.get());
      float root_q_outcome = QOutcome(root_node.get());

      Loc nn_move = gumbel_res.nn_move;
      Loc move = gumbel_res.mcts_move;

      int nn_move_n = NAction(root_node.get(), nn_move);
      float nn_move_q = QAction(root_node.get(), nn_move);
      float nn_move_qz = QOutcomeAction(root_node.get(), nn_move);

      int move_n = NAction(root_node.get(), move);
      float move_q = QAction(root_node.get(), move);
      float move_qz = QOutcomeAction(root_node.get(), move_qz);

      // Update tracking data structures.
      mcts_pis.push_back(gumbel_res.pi_improved);
      move_trainables.push_back(is_move_selected_for_training);
      root_q_outcomes.push_back(root_q_outcome);
      int& consecutive_db_moves = color_to_move == BLACK
                                      ? num_consecutive_down_bad_moves_b
                                      : num_consecutive_down_bad_moves_w;
      if (root_q_outcome < kDownBadThreshold) {
        ++consecutive_db_moves;
      } else {
        consecutive_db_moves = 0;
      }

      // Need to store this temporarily so we do not accidentally invoke the
      // destructor.
      std::unique_ptr<TreeNode> next_root =
          std::move(root_node->children[move]);
      if (log_mcts_trees) {
        search_roots.emplace_back(std::move(root_node));
      }

      // Play move, and calculate PA regions if we hit a checkpoint.
      game.PlayMove(move, color_to_move);
      if (std::find(std::begin(kComputePAMoveNums),
                    std::end(kComputePAMoveNums), game.num_moves())) {
        game.CalculatePassAliveRegions();
      }
      color_to_move = OppositeColor(color_to_move);

      root_node = std::move(next_root);
      if (!root_node) {
        // this is possible if we draw directly from policy, or if pass is the
        // only legal move found in search.
        root_node = std::make_unique<TreeNode>();
      }

      // Add to seen state buffer.
      auto should_add_init_state = [&probability](float root_q) {
        // Adds states with sharper Q values with lower probability, with a
        // penalty of 1 - .75(max(0, |Q| - .5)^2/.5^2)
        static constexpr float kDecayThreshold = .5;
        static constexpr float kMaxDiff = 1 - kDecayThreshold;
        static constexpr float kMaxPenalty = .75;

        float base_probability = kAddSeenStateProb;

        float root_q_normalized = root_q < 0 ? -root_q : root_q;
        float diff = std::max(0.0f, root_q_normalized - kDecayThreshold);

        // squared scaling.
        float penalty = kMaxPenalty * ((diff * diff) / (kMaxDiff * kMaxDiff));
        float scaling = 1 - penalty;
        float p = base_probability * scaling;

        return probability.Uniform() <= p;
      };

      if (should_add_init_state(root_q_outcome) <= kAddSeenStateProb) {
        AddNewInitState(go_exploit_buffer, game, color_to_move);
      }

      auto search_dur =
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
      search_dur_ema = search_dur_ema == 0
                           ? search_dur
                           : (search_dur_ema * 0.9 + search_dur * 0.1);

      auto log_fn = [&]() {
        std::stringstream s;
        s << "\n----- Move Num: " << game.num_moves() << " -----\n";
        s << "N: " << gumbel_n << ", K: " << gumbel_k
          << ", Trainable: " << is_move_selected_for_training << "\n";
        s << "Pre-Search Stats:"
          << "\n  N: " << n_pre << "\n  Q: " << q_pre << "\n  Q_z: " << qz_pre
          << "\n";
        s << "Post-Search Stats:"
          << "\n  N: " << n_post << "\n  Q: " << q_post
          << "\n  Q_z: " << root_q_outcome << "\n";
        s << "Raw NN Move: " << nn_move << "\n  n: " << nn_move_n
          << "\n  q: " << nn_move_q << "\n  qz: " << nn_move_qz << "\n";
        s << "Gumbel Move: " << move << "\n  n: " << move_n
          << "\n  q: " << move_q << "\n  qz: " << move_qz << "\n";
        s << "Last 5 Moves: " << game.move(game.num_moves() - 5) << ", "
          << game.move(game.num_moves() - 4) << ", "
          << game.move(game.num_moves() - 3) << ", "
          << game.move(game.num_moves() - 2) << ", "
          << game.move(game.num_moves() - 1) << "\n";
        s << "Root Color: " << OppositeColor(color_to_move)
          << " Next Root Visits: " << root_node->n
          << " Player to Move: " << root_node->color_to_move
          << " Value: " << root_node->q << " Outcome: " << root_node->q_outcome
          << "\n";
        s << "Board:\n" << game.board() << "\n";
        s << "Search Took " << search_dur
          << "us. Search EMA: " << search_dur_ema << "us.\n";

        LOG_TO_SINK(INFO, sink) << s.str();
      };

      if (thread_id % kShouldLogShard == 0) {
        log_fn();
      }
    }

    nn_interface->UnregisterThread(thread_id);
    if (!IsRunning()) break;

    game.WriteResult();

    LOG_TO_SINK(INFO, sink) << "Black Score: " << game.result().bscore;
    LOG_TO_SINK(INFO, sink) << "White Score: " << game.result().wscore;

    auto begin = std::chrono::high_resolution_clock::now();
    game_recorder->RecordGame(thread_id, init_state.board, game, mcts_pis,
                              move_trainables, root_q_outcomes,
                              std::move(search_roots));
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
