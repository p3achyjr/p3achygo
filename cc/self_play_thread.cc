#include "cc/self_play_thread.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/mcts/gumbel.h"

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

static constexpr auto kGumbelK = 8;
static constexpr auto kGumbelN = 64;

}  // namespace

void ExecuteSelfPlay(int thread_id, nn::NNInterface* nn_interface,
                     std::string logfile) {
  ThreadSink sink(logfile.c_str());
  core::Probability probability(static_cast<uint64_t>(std::time(nullptr)) +
                                thread_id);

  // initialize game related objects.
  game::Zobrist zobrist_table;
  game::Board board(&zobrist_table);
  std::unique_ptr<mcts::TreeNode> root_node =
      std::make_unique<mcts::TreeNode>();
  std::vector<game::Loc> move_history = {game::kNoopLoc, game::kNoopLoc,
                                         game::kNoopLoc, game::kNoopLoc,
                                         game::kNoopLoc};

  mcts::GumbelEvaluator gumbel_evaluator(nn_interface, thread_id);
  auto color_to_move = BLACK;
  while (!board.IsGameOver()) {
    LOG(INFO).ToSinkOnly(&sink) << "-------------------";
    LOG(INFO).ToSinkOnly(&sink) << "Searching...";
    std::pair<game::Loc, game::Loc> move_pair = gumbel_evaluator.SearchRoot(
        probability, board, root_node.get(), move_history, color_to_move,
        kGumbelN, kGumbelK);
    game::Loc nn_move = move_pair.first;
    game::Loc move = move_pair.second;
    board.Move(move, color_to_move);
    move_history.emplace_back(move);
    color_to_move = game::OppositeColor(color_to_move);
    root_node = std::move(root_node->children[board.LocAsMove(move)]);

    LOG(INFO).ToSinkOnly(&sink) << "Raw NN Move: " << nn_move;
    LOG(INFO).ToSinkOnly(&sink) << "Gumbel Move: " << move;
    LOG(INFO).ToSinkOnly(&sink) << "Move Num: " << move_history.size() - 5;
    LOG(INFO).ToSinkOnly(&sink)
        << "Last 5 Moves: " << move_history[move_history.size() - 5] << ", "
        << move_history[move_history.size() - 4] << ", "
        << move_history[move_history.size() - 3] << ", "
        << move_history[move_history.size() - 2] << ", "
        << move_history[move_history.size() - 1];
    LOG(INFO).ToSinkOnly(&sink)
        << "Tree Visit Count: " << root_node->n
        << " Player to Move: " << root_node->color_to_move
        << " Value: " << root_node->q;
    LOG(INFO).ToSinkOnly(&sink) << "Board:\n" << board;
    LOG(INFO) << "Thread " << thread_id << " moved";
  }

  nn_interface->UnregisterThread(thread_id);
  game::Scores scores = board.GetScores();

  LOG(INFO).ToSinkOnly(&sink) << "Black Score: " << scores.black_score;
  LOG(INFO).ToSinkOnly(&sink) << "White Score: " << scores.white_score;
}
