#include "cc/recorder/game_recorder.h"

#include <filesystem>
#include <memory>
#include <string>

#include "cc/game/game.h"
#include "cc/recorder/dir.h"

namespace recorder {
namespace {

namespace fs = std::filesystem;

class GameRecorderImpl final : public GameRecorder {
 public:
  GameRecorderImpl(std::string path, int num_threads, int flush_interval);
  ~GameRecorderImpl() = default;

  // Disable Copy and Move.
  GameRecorderImpl(GameRecorderImpl const&) = delete;
  GameRecorderImpl& operator=(GameRecorderImpl const&) = delete;
  GameRecorderImpl(GameRecorderImpl&&) = delete;
  GameRecorderImpl& operator=(GameRecorderImpl&&) = delete;

  void RecordGame(int thread_id, const game::Game& game) override;

  static std::unique_ptr<GameRecorderImpl> Create(std::string path,
                                                  int num_threads,
                                                  int flush_interval);

 private:
  std::unique_ptr<SgfRecorder> sgf_recorder_;
  std::unique_ptr<TfRecorder> tf_recorder_;
};
}  // namespace

GameRecorderImpl::GameRecorderImpl(std::string path, int num_threads,
                                   int flush_interval)
    : sgf_recorder_(SgfRecorder::Create(fs::path(path) / recorder::kSgfDir,
                                        num_threads, flush_interval)),
      tf_recorder_(TfRecorder::Create(fs::path(path) / recorder::kTfDir,
                                      num_threads, flush_interval)) {}

void GameRecorderImpl::RecordGame(int thread_id, const game::Game& game) {
  sgf_recorder_->RecordGame(thread_id, game);
  tf_recorder_->RecordGame(thread_id, game);
}

/* static */ std::unique_ptr<GameRecorder> GameRecorder::Create(
    std::string path, int num_threads, int flush_interval) {
  return std::make_unique<GameRecorderImpl>(path, num_threads, flush_interval);
}
}  // namespace recorder
