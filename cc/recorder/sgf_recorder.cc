#include "cc/recorder/sgf_recorder.h"

#include <memory>

#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/recorder/sgf_serializer.h"
#include "cc/recorder/sgf_tree.h"

namespace recorder {
namespace {

using ::game::Game;
using ::game::Move;

static constexpr int kMaxNumProperties = 32;
static constexpr char kP3achyGoName[] = "p3achygo";

class SgfRecorderImpl final : public SgfRecorder {
 public:
  SgfRecorderImpl(std::string path, int num_threads, int flush_interval);
  ~SgfRecorderImpl() override = default;

  // Disable Copy and Move.
  SgfRecorderImpl(SgfRecorderImpl const&) = delete;
  SgfRecorderImpl& operator=(SgfRecorderImpl const&) = delete;
  SgfRecorderImpl(SgfRecorderImpl&&) = delete;
  SgfRecorderImpl& operator=(SgfRecorderImpl&&) = delete;

  // Recorder Impl.
  void RecordGame(int thread_id, const Game& game) override;

 private:
  std::unique_ptr<SgfNode> ToSgfNode(const Game& game);
  void Flush() ABSL_LOCKS_EXCLUDED(mu_);

  std::string path_;
  int num_threads_;
  std::array<std::vector<std::unique_ptr<SgfNode>>, constants::kMaxNumThreads>
      sgfs_;

  absl::Mutex mu_;
  int games_buffered_ ABSL_GUARDED_BY(mu_);
  int games_written_ ABSL_GUARDED_BY(mu_);
  int flush_interval_;
};

SgfRecorderImpl::SgfRecorderImpl(std::string path, int num_threads,
                                 int flush_interval)
    : path_(path),
      num_threads_(num_threads),
      games_buffered_(0),
      games_written_(0),
      flush_interval_(flush_interval) {}

void SgfRecorderImpl::RecordGame(int thread_id, const Game& game) {
  DCHECK(game.has_result());
  std::vector<std::unique_ptr<SgfNode>>& thread_sgfs = sgfs_[thread_id];
  thread_sgfs.emplace_back(ToSgfNode(game));

  mu_.Lock();
  ++games_buffered_;
  mu_.Unlock();

  if (games_buffered_ >= flush_interval_) {
    Flush();
  }
}

std::unique_ptr<SgfNode> SgfRecorderImpl::ToSgfNode(const Game& game) {
  std::unique_ptr<SgfNode> root_node = std::make_unique<SgfNode>();
  root_node->AddProperty(std::make_unique<SgfKomiProp>(game.komi()));
  root_node->AddProperty(std::make_unique<SgfResultProp>(game.result()));
  root_node->AddProperty(std::make_unique<SgfBPlayerProp>(kP3achyGoName));
  root_node->AddProperty(std::make_unique<SgfWPlayerProp>(kP3achyGoName));

  int num_moves = game.move_num();
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

void SgfRecorderImpl::Flush() {
  absl::MutexLock lock(&mu_);
  for (int i = 0; i < num_threads_; ++i) {
    std::vector<std::unique_ptr<SgfNode>>& thread_sgfs = sgfs_[i];
    if (thread_sgfs.empty()) {
      continue;
    }

    std::string path = path_ + absl::StrFormat("game_%d.sgf", games_written_);
    SgfSerializer serializer;
    for (const auto& sgf : thread_sgfs) {
      FILE* const sgf_file = fopen(path.c_str(), "w");
      absl::FPrintF(sgf_file, "%s", serializer.Serialize(sgf.get()).c_str());
    }
    thread_sgfs.clear();

    ++games_written_;
  }
  games_buffered_ = 0;
}
}  // namespace

/* static */ std::unique_ptr<SgfRecorder> SgfRecorder::Create(
    std::string path, int num_threads, int flush_interval) {
  return std::make_unique<SgfRecorderImpl>(path, num_threads, flush_interval);
}
}  // namespace recorder
