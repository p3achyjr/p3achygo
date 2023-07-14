#include "cc/recorder/tf_recorder.h"

#include <memory>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "cc/constants/constants.h"
#include "cc/core/filepath.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/recorder/make_tf_example.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"

namespace recorder {
namespace {

using namespace ::game;

using ::tensorflow::io::RecordWriter;
using ::tensorflow::io::RecordWriterOptions;

using ::core::FilePath;

// Keep in sync with //cc/shuffler/chunk_info.h
static constexpr char kChunkFormat[] =
    "gen%03d_b%03d_g%03d_n%05d_%s.tfrecord.zz";
static constexpr char kChunkDoneFormat[] = "gen%03d_b%03d_g%03d_n%05d_%s.done";

class TfRecorderImpl final : public TfRecorder {
 public:
  TfRecorderImpl(std::string path, int num_threads, int gen,
                 std::string worker_id);
  ~TfRecorderImpl() = default;

  // Disable Copy and Move.
  TfRecorderImpl(TfRecorderImpl const&) = delete;
  TfRecorderImpl& operator=(TfRecorderImpl const&) = delete;
  TfRecorderImpl(TfRecorderImpl&&) = delete;
  TfRecorderImpl& operator=(TfRecorderImpl&&) = delete;

  void RecordGame(int thread_id, const game::Board& init_board,
                  const Game& game, const ImprovedPolicies& mcts_pis,
                  const std::vector<uint8_t>& move_trainables,
                  const std::vector<float>& root_qs) override;
  void Flush() override;

 private:
  struct Record {
    Board init_board;
    Game game;
    ImprovedPolicies mcts_pis;
    std::vector<uint8_t> move_trainables;
    std::vector<float> root_qs;
  };

  const std::string path_;
  const int num_threads_;
  const int gen_;
  const std::string worker_id_;

  std::array<std::vector<Record>, constants::kMaxNumThreads> thread_records_;
  std::array<int, constants::kMaxNumThreads> thread_game_counts_;
  int batch_num_;
};

TfRecorderImpl::TfRecorderImpl(std::string path, int num_threads, int gen,
                               std::string worker_id)
    : path_(path),
      num_threads_(num_threads),
      gen_(gen),
      worker_id_(worker_id),
      thread_game_counts_{},
      batch_num_(0) {}

void TfRecorderImpl::RecordGame(int thread_id, const Board& init_board,
                                const Game& game,
                                const ImprovedPolicies& mcts_pis,
                                const std::vector<uint8_t>& move_trainables,
                                const std::vector<float>& root_qs) {
  if (path_.empty()) {
    return;
  }

  CHECK(game.has_result());
  CHECK(game.num_moves() == mcts_pis.size());
  thread_records_[thread_id].emplace_back(
      Record{init_board, game, mcts_pis, move_trainables, root_qs});
  ++thread_game_counts_[thread_id];
}

// Only one thread can call this method. Additionally, no thread can call
// `RecordGame` while this method is running.
void TfRecorderImpl::Flush() {
  if (path_.empty()) {
    return;
  }

  int num_games =
      std::accumulate(thread_game_counts_.begin(),
                      thread_game_counts_.begin() + num_threads_, 0);
  int num_records = std::accumulate(
      thread_records_.begin(), thread_records_.begin() + num_threads_, 0,
      [](int n, const std::vector<Record>& records) {
        for (const auto& record : records) {
          n += record.game.num_moves();
        }

        return n;
      });

  if (num_records == 0) {
    return;
  }

  // Create File.
  std::string path =
      FilePath(path_) / absl::StrFormat(kChunkFormat, gen_, batch_num_,
                                        num_games, num_records, worker_id_);
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(path, &file));

  // Create Writer.
  RecordWriterOptions options;
  options.compression_type = RecordWriterOptions::ZLIB_COMPRESSION;
  options.zlib_options.compression_level = 2;
  RecordWriter writer(file.get(), options);

  // Flush each thread.
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    std::vector<Record>& records = thread_records_[thread_id];
    if (records.empty()) {
      continue;
    }

    for (const auto& record : records) {
      // Replay game from beginning. We do not store full board positions in
      // `Game` because MCTS performs many copies of `Game` objects.
      const Game& game = record.game;
      const ImprovedPolicies& mcts_pis = record.mcts_pis;
      const std::vector<uint8_t>& move_trainables = record.move_trainables;
      const std::vector<float>& root_qs = record.root_qs;
      Board board = record.init_board;
      for (int move_num = 0; move_num < game.num_moves(); ++move_num) {
        // Populate last moves as indices.
        std::array<int16_t, constants::kNumLastMoves> last_moves;
        for (int off = 0; off < constants::kNumLastMoves; ++off) {
          Loc last_move = game.moves()[move_num + off].loc;
          last_moves[off] = last_move;
        }

        Move move = game.move(move_num);

        bool is_trainable = move_trainables[move_num];
        if (is_trainable) {
          // Coerce into example and write result.
          Move next_move = move_num < game.num_moves() - 1
                               ? game.move(move_num + 1)
                               : Move{OppositeColor(move.color), kPassLoc};
          const std::array<float, constants::kMaxNumMoves>& pi =
              mcts_pis[move_num];
          Color color = move.color;
          float z = game.result().winner == color ? 1.0 : -1.0;
          float q30 =
              move_num + 6 < game.num_moves() ? root_qs[move_num + 6] : z;
          float q100 =
              move_num + 16 < game.num_moves() ? root_qs[move_num + 16] : z;
          float q200 =
              move_num + 50 < game.num_moves() ? root_qs[move_num + 50] : z;
          tensorflow::Example example = MakeTfExample(
              board.position(), last_moves, board.GetStonesInAtari(),
              board.GetStonesWithLiberties(2), board.GetStonesWithLiberties(3),
              pi, next_move.loc, game.result(), q30, q100, q200, move.color,
              game.komi(), BOARD_LEN);
          std::string data;
          example.SerializeToString(&data);
          TF_CHECK_OK(writer.WriteRecord(data));
        }

        // Play next move.
        board.PlayMove(move.loc, move.color);
      }
    }

    records.clear();
  }

  // Close file.
  TF_CHECK_OK(writer.Close());
  TF_CHECK_OK(file->Close());

  // Write .done file to indicate that we are done writing.
  std::string done_filename =
      FilePath(path_) / absl::StrFormat(kChunkDoneFormat, gen_, batch_num_,
                                        num_games, num_records, worker_id_);
  FILE* const lock_file = fopen(done_filename.c_str(), "w");
  absl::FPrintF(lock_file, "");
  fclose(lock_file);

  // Update metadata fields.
  ++batch_num_;
  std::fill(thread_game_counts_.begin(),
            thread_game_counts_.begin() + num_threads_, 0);
}
}  // namespace

/* static */ std::unique_ptr<TfRecorder> TfRecorder::Create(
    std::string path, int num_threads, int gen, std::string worker_id) {
  return std::make_unique<TfRecorderImpl>(path, num_threads, gen, worker_id);
}

}  // namespace recorder
