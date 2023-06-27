#include "cc/selfplay/self_play_thread.h"

#include <chrono>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/core/file_log_sink.h"
#include "cc/core/probability.h"
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

static constexpr int kShouldLogShard = 8;

static std::atomic<bool> running = true;

}  // namespace

void Run(size_t seed, int thread_id, NNInterface* nn_interface,
         GameRecorder* game_recorder, std::string logfile, int gumbel_n,
         int gumbel_k, int max_moves) {
  FileSink sink(logfile.c_str());
  Probability probability(seed + thread_id);
  auto search_dur_ema = 0;

  // Main loop.
  while (true) {
    // Initialize game related objects.
    Game game;
    std::unique_ptr<TreeNode> root_node = std::make_unique<TreeNode>();
    std::vector<std::array<float, constants::kMaxNumMoves>> mcts_pis;

    GumbelEvaluator gumbel_evaluator(nn_interface, thread_id);
    auto color_to_move = BLACK;
    while (IsRunning() && !game.IsGameOver() && game.num_moves() < max_moves) {
      auto begin = std::chrono::high_resolution_clock::now();
      GumbelResult gumbel_res =
          gumbel_evaluator.SearchRoot(probability, game, root_node.get(),
                                      color_to_move, gumbel_n, gumbel_k);
      auto end = std::chrono::high_resolution_clock::now();
      Loc nn_move = gumbel_res.nn_move;
      Loc move = gumbel_res.mcts_move;
      float nn_move_q =
          QAction(root_node.get(), nn_move.as_index(game.board_len()));
      float move_q = QAction(root_node.get(), move.as_index(game.board_len()));
      mcts_pis.push_back(gumbel_res.pi_improved);
      game.PlayMove(move, color_to_move);
      color_to_move = OppositeColor(color_to_move);

      root_node =
          std::move(root_node->children[move.as_index(game.board_len())]);
      if (!root_node) {
        // this is possible if pass is the only legal move found in search.
        LOG(INFO) << "Root node is nullptr. "
                  << "Last 5 Moves: " << game.move(game.num_moves() - 5) << ", "
                  << game.move(game.num_moves() - 4) << ", "
                  << game.move(game.num_moves() - 3) << ", "
                  << game.move(game.num_moves() - 2) << ", "
                  << game.move(game.num_moves() - 1)
                  << ", Move Count: " << game.num_moves();
        LOG(INFO) << "Board:\n" << game.board();
        root_node = std::make_unique<TreeNode>();
      }

      auto search_dur =
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
      search_dur_ema = search_dur_ema == 0
                           ? search_dur
                           : (search_dur_ema * 0.9 + search_dur * 0.1);

      if (thread_id % kShouldLogShard == 0) {
        LOG_TO_SINK(INFO, sink) << "-------------------";
        LOG_TO_SINK(INFO, sink)
            << "Raw NN Move: " << nn_move << ", q: " << nn_move_q;
        LOG_TO_SINK(INFO, sink) << "Gumbel Move: " << move << ", q: " << move_q;
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
            << " Value: " << root_node->q;
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
    game_recorder->RecordGame(thread_id, game, mcts_pis);
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
