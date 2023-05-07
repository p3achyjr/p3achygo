#include "cc/self_play_thread.h"

#include <chrono>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/game.h"
#include "cc/mcts/gumbel.h"
#include "cc/recorder/sgf_recorder.h"

#define LOG_TO_SINK(severity, sink) LOG(severity).ToSinkOnly(&sink)

namespace {

class ThreadSink : public absl::LogSink {
 public:
  ThreadSink(const char* filename)
      : filename_(filename), fp_(fopen(filename, "w")) {
    PCHECK(fp_ != nullptr) << "Failed to open " << filename_;
  }
  ~ThreadSink() {
    fputc('\f', fp_);
    fflush(fp_);
    PCHECK(fclose(fp_) == 0) << "Failed to close " << filename_;
  }

  void Send(const absl::LogEntry& entry) override {
    absl::FPrintF(fp_, "%s\r\n", entry.text_message_with_prefix());
    fflush(fp_);
  }

 private:
  const char* filename_;
  FILE* const fp_;
};

static constexpr auto kGumbelK = 2;
static constexpr auto kGumbelN = 2;

}  // namespace

void ExecuteSelfPlay(int thread_id, nn::NNInterface* nn_interface,
                     recorder::SgfRecorder* sgf_recorder, std::string logfile) {
  ThreadSink sink(logfile.c_str());
  core::Probability probability(static_cast<uint64_t>(std::time(nullptr)) +
                                thread_id);

  // initialize game related objects.
  game::Game game;
  std::unique_ptr<mcts::TreeNode> root_node =
      std::make_unique<mcts::TreeNode>();

  mcts::GumbelEvaluator gumbel_evaluator(nn_interface, thread_id);
  auto color_to_move = BLACK;
  while (!game.IsGameOver()) {
    LOG_TO_SINK(INFO, sink) << "-------------------";
    LOG_TO_SINK(INFO, sink) << "Searching...";
    auto begin = std::chrono::high_resolution_clock::now();
    std::pair<game::Loc, game::Loc> move_pair = gumbel_evaluator.SearchRoot(
        probability, game, root_node.get(), color_to_move, kGumbelN, kGumbelK);
    auto end = std::chrono::high_resolution_clock::now();
    game::Loc nn_move = move_pair.first;
    game::Loc move = move_pair.second;
    game.PlayMove(move, color_to_move);
    color_to_move = game::OppositeColor(color_to_move);
    root_node = std::move(root_node->children[move.as_index(game.board_len())]);

    LOG_TO_SINK(INFO, sink) << "Raw NN Move: " << nn_move;
    LOG_TO_SINK(INFO, sink) << "Gumbel Move: " << move;
    LOG_TO_SINK(INFO, sink) << "Move Num: " << game.move_num();
    LOG_TO_SINK(INFO, sink)
        << "Last 5 Moves: " << game.move(game.move_num() - 5) << ", "
        << game.move(game.move_num() - 4) << ", "
        << game.move(game.move_num() - 3) << ", "
        << game.move(game.move_num() - 2) << ", "
        << game.move(game.move_num() - 1);
    LOG_TO_SINK(INFO, sink) << "Tree Visit Count: " << root_node->n
                            << " Player to Move: " << root_node->color_to_move
                            << " Value: " << root_node->q;
    LOG_TO_SINK(INFO, sink) << "Board:\n" << game.board();
    LOG_TO_SINK(INFO, sink)
        << "Search Took "
        << std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
               .count()
        << "us";
    LOG(INFO) << "Thread " << thread_id << " moved";
  }

  nn_interface->UnregisterThread(thread_id);
  game::Scores scores = game.GetScores();

  LOG_TO_SINK(INFO, sink) << "Black Score: " << scores.black_score;
  LOG_TO_SINK(INFO, sink) << "White Score: " << scores.white_score;

  game.WriteResult();

  auto begin = std::chrono::high_resolution_clock::now();
  sgf_recorder->RecordGame(thread_id, game);
  auto end = std::chrono::high_resolution_clock::now();

  LOG_TO_SINK(INFO, sink)
      << "Recording Game Took "
      << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
             .count()
      << "ns";
}

#undef LOG_TO_SINK
