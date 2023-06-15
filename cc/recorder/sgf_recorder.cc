#include "cc/recorder/sgf_recorder.h"

#include <memory>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/filepath.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/recorder/sgf_serializer.h"
#include "cc/recorder/sgf_tree.h"

namespace recorder {
namespace {

using ::core::FilePath;
using ::game::Game;
using ::game::Move;

static constexpr int kMaxNumProperties = 32;
static constexpr char kP3achyGoName[] = "p3achygo";

class SgfRecorderImpl final : public SgfRecorder {
 public:
  SgfRecorderImpl(std::string path, int num_threads);
  ~SgfRecorderImpl() override = default;

  // Disable Copy and Move.
  SgfRecorderImpl(SgfRecorderImpl const&) = delete;
  SgfRecorderImpl& operator=(SgfRecorderImpl const&) = delete;
  SgfRecorderImpl(SgfRecorderImpl&&) = delete;
  SgfRecorderImpl& operator=(SgfRecorderImpl&&) = delete;

  // Recorder Impl.
  void RecordGame(int thread_id, const Game& game) override;
  void Flush() override;

 private:
  std::unique_ptr<SgfNode> ToSgfNode(const Game& game);

  const std::string path_;
  std::array<std::vector<Game>, constants::kMaxNumThreads> games_;
  const int num_threads_;
  int batch_num_;
};

SgfRecorderImpl::SgfRecorderImpl(std::string path, int num_threads)
    : path_(path), num_threads_(num_threads), batch_num_(0) {}

void SgfRecorderImpl::RecordGame(int thread_id, const Game& game) {
  CHECK(game.has_result());

  std::vector<Game>& thread_games = games_[thread_id];
  thread_games.emplace_back(game);
}

std::unique_ptr<SgfNode> SgfRecorderImpl::ToSgfNode(const Game& game) {
  std::unique_ptr<SgfNode> root_node = std::make_unique<SgfNode>();
  root_node->AddProperty(std::make_unique<SgfKomiProp>(game.komi()));
  root_node->AddProperty(std::make_unique<SgfResultProp>(game.result()));
  root_node->AddProperty(std::make_unique<SgfBPlayerProp>(kP3achyGoName));
  root_node->AddProperty(std::make_unique<SgfWPlayerProp>(kP3achyGoName));

  int num_moves = game.num_moves();
  SgfNode* current_node = root_node.get();
  for (int i = 0; i < num_moves; ++i) {
    const Move& move = game.move(i);
    std::unique_ptr<SgfNode> child = std::make_unique<SgfNode>();
    if (move.color == BLACK) {
      child->AddProperty(std::make_unique<SgfBMoveProp>(move.loc));
    } else if (move.color == WHITE) {
      child->AddProperty(std::make_unique<SgfWMoveProp>(move.loc));
    }

    SgfNode* tmp = child.get();
    current_node->AddChild(std::move(child));
    current_node = tmp;
  }

  return root_node;
}

// Only one thread can call this function. Additionally, no thread can call
// `RecordGame` while this function is running.
void SgfRecorderImpl::Flush() {
  int games_in_batch = 0;
  std::string sgfs = "";
  SgfSerializer serializer;
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    std::vector<Game>& thread_games = games_[thread_id];
    if (thread_games.empty()) {
      continue;
    }

    for (const auto& game : thread_games) {
      sgfs += serializer.Serialize(ToSgfNode(game).get());
      sgfs += "\n";

      ++games_in_batch;
    }
    thread_games.clear();
  }

  if (sgfs == "") {
    return;
  }

  std::string path =
      FilePath(path_) /
      absl::StrFormat("game_b%d_g%d.sgfs", batch_num_, games_in_batch);
  FILE* const sgf_file = fopen(path.c_str(), "w");
  absl::FPrintF(sgf_file, "%s", sgfs);
  fclose(sgf_file);

  ++batch_num_;
}
}  // namespace

/* static */ std::unique_ptr<SgfRecorder> SgfRecorder::Create(std::string path,
                                                              int num_threads) {
  return std::make_unique<SgfRecorderImpl>(path, num_threads);
}
}  // namespace recorder
