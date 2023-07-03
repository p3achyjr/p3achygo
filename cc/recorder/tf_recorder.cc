#include "cc/recorder/tf_recorder.h"

#include <memory>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "cc/constants/constants.h"
#include "cc/core/filepath.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"

namespace recorder {
namespace {

using ::game::Board;
using ::game::Color;
using ::game::Game;
using ::game::Loc;
using ::game::Move;

using ::tensorflow::io::RecordWriter;
using ::tensorflow::io::RecordWriterOptions;

using ::core::FilePath;

// Keep in sync with //cc/shuffler/chunk_info.h
static constexpr char kChunkFormat[] = "gen%d_b%d_g%d_n%d_%s.tfrecord.zz";
static constexpr char kChunkDoneFormat[] = "gen%d_b%d_g%d_n%d_%s.done";

template <typename T, size_t N>
tensorflow::Feature MakeBytesFeature(const std::array<T, N>& data) {
  tensorflow::Feature feature;
  feature.mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(data.data()), sizeof(T) * N);
  return feature;
}

tensorflow::Example MakeTfExample(
    const std::array<Color, BOARD_LEN * BOARD_LEN>& board,
    const std::array<int16_t, constants::kNumLastMoves>& last_moves,
    const std::array<float, constants::kMaxNumMoves>& pi_improved,
    const Game::Result result, Color color, float komi, uint8_t bsize) {
  tensorflow::Example example;
  auto& features = *example.mutable_features()->mutable_feature();

  features["bsize"].mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(&bsize), sizeof(uint8_t));
  features["board"] = MakeBytesFeature(board);
  features["last_moves"] = MakeBytesFeature(last_moves);
  features["color"].mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(&color), sizeof(Color));
  features["komi"].mutable_float_list()->add_value(komi);
  features["own"] = MakeBytesFeature(result.ownership);
  features["pi"] = MakeBytesFeature(pi_improved);

  float margin = color == BLACK ? result.bscore - result.wscore
                                : result.wscore - result.bscore;
  features["result"].mutable_float_list()->add_value(margin);

  return example;
}

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
                  const std::vector<uint8_t>& move_trainables) override;
  void Flush() override;

 private:
  struct Record {
    Board init_board;
    Game game;
    ImprovedPolicies mcts_pis;
    std::vector<uint8_t> move_trainables;
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
                                const std::vector<uint8_t>& move_trainables) {
  CHECK(game.has_result());
  CHECK(game.num_moves() == mcts_pis.size());
  thread_records_[thread_id].emplace_back(
      Record{init_board, game, mcts_pis, move_trainables});
  ++thread_game_counts_[thread_id];
}

// Only one thread can call this method. Additionally, no thread can call
// `RecordGame` while this method is running.
void TfRecorderImpl::Flush() {
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
      Board board = record.init_board;
      for (int move_num = 0; move_num < game.num_moves(); ++move_num) {
        // Populate last moves as indices.
        std::array<int16_t, constants::kNumLastMoves> last_moves;
        for (int off = 0; off < constants::kNumLastMoves; ++off) {
          Loc last_move = game.moves()[move_num + off].loc;
          last_moves[off] = last_move;
        }

        Move move = game.move(move_num);
        const std::array<float, constants::kMaxNumMoves>& pi =
            mcts_pis[move_num];
        bool is_trainable = move_trainables[move_num];

        if (is_trainable) {
          // Coerce into example and write result.
          tensorflow::Example example =
              MakeTfExample(board.position(), last_moves, pi, game.result(),
                            move.color, game.komi(), BOARD_LEN);
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
