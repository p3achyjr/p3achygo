#include "cc/recorder/tf_recorder.h"

#include <memory>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "cc/constants/constants.h"
#include "cc/core/filepath.h"
#include "cc/core/probability.h"
#include "cc/data/filename_format.h"
#include "cc/data/tfrecord/record_writer.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/recorder/make_tf_example.h"
#include "example.pb.h"

namespace recorder {
namespace {

using namespace ::game;

using ::data::RecordWriter;
using ::data::RecordWriterOptions;

using ::core::FilePath;

inline int Timestamp() {
  auto now = std::chrono::steady_clock::now();
  auto duration = now.time_since_epoch();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);

  return seconds.count();
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
                  const std::vector<uint8_t>& move_trainables,
                  const std::vector<float>& root_qs,
                  const std::vector<float>& root_scores,
                  const std::vector<float>& klds) override;
  void Flush() override;

 private:
  struct Record {
    Board init_board;
    Game game;
    ImprovedPolicies mcts_pis;
    std::vector<uint8_t> move_trainables;
    std::vector<float> root_qs;
    std::vector<float> root_scores;
    std::vector<float> klds;
  };

  const std::string path_;
  const int num_threads_;
  const int gen_;
  const std::string worker_id_;

  std::array<std::vector<Record>, constants::kMaxNumThreads> thread_records_;
  std::array<int, constants::kMaxNumThreads> thread_game_counts_;
  int batch_num_;
  core::Probability probability_;
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
                                const std::vector<float>& root_qs,
                                const std::vector<float>& root_scores,
                                const std::vector<float>& klds) {
  CHECK(game.has_result());
  CHECK(game.num_moves() == mcts_pis.size() &&
        game.num_moves() == move_trainables.size() &&
        game.num_moves() == root_qs.size() &&
        game.num_moves() == root_scores.size() &&
        game.num_moves() == klds.size());
  thread_records_[thread_id].emplace_back(Record{
      init_board, game, mcts_pis, move_trainables, root_qs, root_scores, klds});
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
          for (const auto& is_trainable : record.move_trainables) {
            if (is_trainable) ++n;
          }
        }

        return n;
      });

  if (num_records == 0) {
    return;
  }

  const int timestamp = Timestamp();

  // Create File.
  std::string path =
      FilePath(path_) / absl::StrFormat(data::kChunkFormat, gen_, batch_num_,
                                        num_games, num_records, timestamp,
                                        worker_id_);

  // Create Writer with zlib compression.
  RecordWriterOptions options = RecordWriterOptions::Zlib();
  options.zlib_options.compression_level = 2;
  RecordWriter writer(path, options);
  CHECK(writer.Init().ok()) << "Failed to initialize RecordWriter";

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
      const std::vector<float>& root_scores = record.root_scores;
      const size_t num_trainable_moves =
          std::accumulate(move_trainables.begin(), move_trainables.end(), 0,
                          [](size_t x, size_t y) { return x + y; });
      const float avg_kld =
          std::accumulate(record.klds.begin(), record.klds.end(), 0.0F,
                          std::plus<float>()) /
          num_trainable_moves;
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
          const std::array<float, constants::kMaxMovesPerPosition>& pi =
              mcts_pis[move_num];
          Color color = move.color;
          float z = [&]() {
            if (game.result().winner == EMPTY) {
              return 0.0f;
            }
            return game.result().winner == color ? 1.0f : -1.0f;
          }();
          const auto exp_weighted_short_term_value_score =
              [&](const float lambda,
                  const int horizon) -> std::pair<float, float> {
            float N = 0;
            for (int i = 0; i <= horizon; ++i) {
              N += std::pow(lambda, i);
            }

            float q_short_term = 0, score_short_term = 0;
            for (int i = 0; i <= horizon; ++i) {
              float v_mult = (i % 2 == 0) ? 1.0f : -1.0f;  // turn multiplier
              q_short_term +=
                  v_mult * std::pow(lambda, i) * root_qs[move_num + i];
              score_short_term +=
                  v_mult * std::pow(lambda, i) * root_scores[move_num + i];
            }

            return {q_short_term / N, score_short_term / N};
          };
          const auto [q6, q6_score] = exp_weighted_short_term_value_score(
              5.0f / 6.0f, std::min(6, game.num_moves() - move_num - 1));
          const auto [q16, q16_score] = exp_weighted_short_term_value_score(
              15.0f / 16.0f, std::min(16, game.num_moves() - move_num - 1));
          const auto [q50, q50_score] = exp_weighted_short_term_value_score(
              49.0f / 50.0f, std::min(50, game.num_moves() - move_num - 1));
          tensorflow::Example example = MakeTfExample(
              board.position(), last_moves, board.GetStonesInAtari(),
              board.GetStonesWithLiberties(2), board.GetStonesWithLiberties(3),
              board.GetLadderedStones(), pi, next_move.loc, game.result(), q6,
              q16, q50, q6_score, q16_score, q50_score, move.color, game.komi(),
              BOARD_LEN);
          std::string data;
          example.SerializeToString(&data);

          // Policy surprise weighting.
          const float freq_weight =
              0.5F + 0.5F * (record.klds[move_num] / avg_kld);
          for (int i = 0; i < std::floor(freq_weight); ++i) {
            CHECK(writer.WriteRecord(data).ok());
          }

          if (probability_.Uniform() <
              (freq_weight - std::floor(freq_weight))) {
            CHECK(writer.WriteRecord(data).ok());
          }
        }

        // Play next move.
        board.PlayMove(move.loc, move.color);
      }
    }

    records.clear();
  }

  // Close file.
  CHECK(writer.Close().ok());

  // Write .done file to indicate that we are done writing.
  std::string done_filename =
      FilePath(path_) / absl::StrFormat(data::kChunkDoneFormat, gen_,
                                        batch_num_, num_games, num_records,
                                        timestamp, worker_id_);
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
