#include "cc/recorder/tf_recorder.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>

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
#include "cc/recorder/move_search_stats.h"
#include "example.pb.h"

namespace recorder {
namespace {

// Returns percentiles at p01, p05..p95 (5% steps), p99 — 21 values total.
// vals is sorted in place.
std::vector<float> ComputePercentiles(std::vector<float> vals) {
  std::sort(vals.begin(), vals.end());
  const int n = static_cast<int>(vals.size());
  auto at_pct = [&](float pct) -> float {
    if (n == 0) return 0.0f;
    int idx = std::clamp(static_cast<int>(std::round(pct / 100.0f * (n - 1))),
                         0, n - 1);
    return vals[idx];
  };
  std::vector<float> out;
  out.reserve(21);
  out.push_back(at_pct(1.0f));
  for (int i = 5; i <= 95; i += 5) out.push_back(at_pct(static_cast<float>(i)));
  out.push_back(at_pct(99.0f));
  return out;
}

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
                  const Game& game,
                  const std::vector<MoveSearchRecord>& move_infos) override;
  void Flush() override;

 private:
  struct Record {
    Board init_board;
    Game game;
    std::vector<MoveSearchRecord> move_infos;
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

void TfRecorderImpl::RecordGame(
    int thread_id, const Board& init_board, const Game& game,
    const std::vector<MoveSearchRecord>& move_infos) {
  CHECK(game.has_result());
  CHECK(game.num_moves() == move_infos.size());
  thread_records_[thread_id].emplace_back(Record{init_board, game, move_infos});
  ++thread_game_counts_[thread_id];
}

// Only one thread can call this method. Additionally, no thread can call
// `RecordGame` while this method is running.
void TfRecorderImpl::Flush() {
  size_t trainable_visit_count = 0, fast_visit_count = 0,
         total_num_trainable_moves = 0, total_num_fast_moves = 0;
  // Buffer examples first. We need to account for policy surprise weighting.
  std::vector<std::string> tf_examples;
  std::vector<MoveSearchStats> all_move_stats;
  std::vector<float> all_freq_weights;
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    std::vector<Record>& records = thread_records_[thread_id];
    if (records.empty()) {
      continue;
    }

    for (const auto& record : records) {
      // Replay game from beginning. We do not store full board positions in
      // `Game` because MCTS performs many copies of `Game` objects.
      const Game& game = record.game;
      const std::vector<MoveSearchRecord>& move_infos = record.move_infos;
      const size_t num_trainable_moves =
          std::accumulate(move_infos.begin(), move_infos.end(), 0,
                          [](size_t x, const MoveSearchRecord& y) {
                            return x + y.move_trainable;
                          });
      const float kld_sum = [&]() {
        float sum = 0;
        for (const MoveSearchRecord& move_info : move_infos) {
          if (move_info.move_trainable) {
            sum += move_info.kld;
          }
        }
        return sum;
      }();
      for (const MoveSearchRecord& move_info : move_infos) {
        const bool trainable = move_info.move_trainable;
        const size_t vc = static_cast<size_t>(move_info.move_stats.visit_count);
        trainable_visit_count += vc * (trainable ? 1 : 0);
        fast_visit_count += vc * (trainable ? 0 : 1);
        total_num_trainable_moves += (trainable ? 1 : 0);
        total_num_fast_moves += (trainable ? 0 : 1);
      }
      const float avg_kld =
          num_trainable_moves > 0 ? kld_sum / num_trainable_moves : 0.0f;
      Board board = record.init_board;
      for (int move_num = 0; move_num < game.num_moves(); ++move_num) {
        // Populate last moves as indices.
        std::array<int16_t, constants::kNumLastMoves> last_moves;
        for (int off = 0; off < constants::kNumLastMoves; ++off) {
          Loc last_move = game.moves()[move_num + off].loc;
          last_moves[off] = last_move;
        }

        Move move = game.move(move_num);

        const MoveSearchRecord& move_info = move_infos[move_num];
        const bool is_trainable = move_info.move_trainable;
        if (is_trainable) {
          // Coerce into example and write result.
          const Move next_move =
              move_num < game.num_moves() - 1
                  ? game.move(move_num + 1)
                  : Move{OppositeColor(move.color), kPassLoc};
          const PolicyArray pi_aux_dist = move_num < game.num_moves() - 1
                                              ? move_infos[move_num + 1].mcts_pi
                                              : PolicyArray{};
          const PolicyArray& pi = move_info.mcts_pi;
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
              const MoveSearchRecord& td_move_info = move_infos[move_num + i];
              float v_mult = (i % 2 == 0) ? 1.0f : -1.0f;  // turn multiplier
              q_short_term +=
                  v_mult * std::pow(lambda, i) * td_move_info.root_q_outcome;
              score_short_term +=
                  v_mult * std::pow(lambda, i) * td_move_info.root_score;
            }

            return {q_short_term / N, score_short_term / N};
          };
          // Go all the way to the end of the game.
          const auto [q6, q6_score] = exp_weighted_short_term_value_score(
              5.0f / 6.0f, game.num_moves() - move_num - 1);
          const auto [q16, q16_score] = exp_weighted_short_term_value_score(
              15.0f / 16.0f, game.num_moves() - move_num - 1);
          const auto [q50, q50_score] = exp_weighted_short_term_value_score(
              49.0f / 50.0f, game.num_moves() - move_num - 1);
          tensorflow::Example example = MakeTfExample(
              board.position(), last_moves, board.GetStonesInAtari(),
              board.GetStonesWithLiberties(2), board.GetStonesWithLiberties(3),
              board.GetLadderedStones(), pi, next_move.loc, pi_aux_dist,
              game.result(), move_info.mcts_value_dist, q6, q16, q50, q6_score,
              q16_score, q50_score, move.color, game.komi(), BOARD_LEN);
          std::string data;
          example.SerializeToString(&data);

          // Policy surprise weighting.
          const float freq_weight =
              avg_kld == 0.0f ? 1.0 : (0.5F + 0.5F * (move_info.kld / avg_kld));
          for (int i = 0; i < std::floor(freq_weight); ++i) {
            tf_examples.push_back(data);
          }

          if (probability_.Uniform() <
              (freq_weight - std::floor(freq_weight))) {
            tf_examples.push_back(data);
          }
        }

        // Play next move.
        board.PlayMove(move.loc, move.color);
      }

      for (const auto& move_info : move_infos) {
        // Skip policy-sampled moves (no real search). visit_count == 1
        // means the root was evaluated but no Gumbel search was run, so
        // stats are not meaningful.
        const auto& s = move_info.move_stats;
        if (!s.sampled_raw_policy && s.visit_count > 1.0f) {
          all_move_stats.push_back(s);
          const float freq_weight =
              avg_kld == 0.0f ? 1.0f
                              : (0.5f + 0.5f * (move_info.kld / avg_kld));
          all_freq_weights.push_back(freq_weight);
        }
      }
    }

    records.clear();
  }

  if (tf_examples.size() == 0) {
    return;
  }

  const int num_games =
      std::accumulate(thread_game_counts_.begin(),
                      thread_game_counts_.begin() + num_threads_, 0);
  const int num_records = tf_examples.size();
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

  for (const std::string& example : tf_examples) {
    CHECK(writer.WriteRecord(example).ok());
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

  // Dump visit count info
  std::string visit_count_filename =
      FilePath(path_) /
      absl::StrFormat("gen%03d_b%03d_g%03d_n%05d_t%d_%s.visit_count", gen_,
                      batch_num_, num_games, num_records, timestamp,
                      worker_id_);
  FILE* const visit_count_file = fopen(visit_count_filename.c_str(), "w");
  absl::FPrintF(
      visit_count_file,
      "Trainable Visits: %lu\nFast Visits: %lu\nTrainable Moves: %lu\nFast "
      "Moves: %lu\nVisits Per Trainable Move: %lu\nVisits Per Fast Move: %lu\n",
      trainable_visit_count, fast_visit_count, total_num_trainable_moves,
      total_num_fast_moves,
      total_num_trainable_moves > 0
          ? trainable_visit_count / total_num_trainable_moves
          : 0,
      total_num_fast_moves > 0 ? fast_visit_count / total_num_fast_moves : 0);
  fclose(visit_count_file);

  // Write .stats text file: percentile table for each field (p0..p100 at 5%
  // increments), one row per field.
  std::string stats_filename =
      FilePath(path_) / absl::StrFormat(data::kStatsFormat, gen_, batch_num_,
                                        num_games, num_records, timestamp,
                                        worker_id_);
  FILE* const stats_file = fopen(stats_filename.c_str(), "w");

  const int n_stats = static_cast<int>(all_move_stats.size());

  // Step 1: collect — generic over any container.
  // Default filter for move stats: skip sampled, zero, and non-finite.
  auto collect_stats = [&](std::function<float(const MoveSearchStats&)> get) {
    std::vector<float> vals;
    vals.reserve(n_stats);
    for (const auto& s : all_move_stats) {
      if (s.sampled_raw_policy) continue;
      const float v = get(s);
      if (v == 0.0f || !std::isfinite(v)) continue;
      vals.push_back(v);
    }
    return vals;
  };

  auto collect_weights = [&]() {
    std::vector<float> vals;
    vals.reserve(all_freq_weights.size());
    for (float v : all_freq_weights) {
      if (v != 0.0f && std::isfinite(v)) vals.push_back(v);
    }
    return vals;
  };

  // Step 3: write — plain write of pre-computed percentiles.
  auto write_percentiles = [&](const char* name,
                               const std::vector<float>& pcts) {
    absl::FPrintF(stats_file, "%-24s", name);
    for (float v : pcts) absl::FPrintF(stats_file, " %9.6f", v);
    absl::FPrintF(stats_file, "\n");
  };

  // Percentile header: p01, p05, p10, ..., p95, p99 (21 columns).
  absl::FPrintF(stats_file,
                "# percentiles: p01 p05 p10 ... p95 p99 (%d moves)\n", n_stats);
  absl::FPrintF(stats_file, "%-24s", "field");
  absl::FPrintF(stats_file, " %9s", "p01");
  for (int i = 5; i <= 95; i += 5)
    absl::FPrintF(stats_file, " %9s", absl::StrFormat("p%02d", i).c_str());
  absl::FPrintF(stats_file, " %9s\n", "p99");

  write_percentiles("nn_q",
                    ComputePercentiles(collect_stats(
                        [](const MoveSearchStats& s) { return s.nn_q; })));
  write_percentiles("mcts_q",
                    ComputePercentiles(collect_stats(
                        [](const MoveSearchStats& s) { return s.mcts_q; })));
  write_percentiles(
      "nn_mcts_diff",
      ComputePercentiles(collect_stats(
          [](const MoveSearchStats& s) { return s.nn_mcts_diff; })));
  write_percentiles(
      "v_outcome_stddev",
      ComputePercentiles(collect_stats(
          [](const MoveSearchStats& s) { return s.v_outcome_stddev; })));
  write_percentiles(
      "prior_entropy",
      ComputePercentiles(collect_stats(
          [](const MoveSearchStats& s) { return s.prior_entropy; })));
  write_percentiles(
      "nn_uncertainty",
      ComputePercentiles(collect_stats(
          [](const MoveSearchStats& s) { return s.nn_uncertainty; })));
  write_percentiles("kld",
                    ComputePercentiles(collect_stats(
                        [](const MoveSearchStats& s) { return s.kld; })));
  write_percentiles("pre_kld",
                    ComputePercentiles(collect_stats(
                        [](const MoveSearchStats& s) { return s.pre_kld; })));
  write_percentiles(
      "sel_mult_modifier",
      ComputePercentiles(collect_stats(
          [](const MoveSearchStats& s) { return s.sel_mult_modifier; })));
  write_percentiles(
      "visit_count",
      ComputePercentiles(collect_stats(
          [](const MoveSearchStats& s) { return s.visit_count; })));
  write_percentiles("freq_weight", ComputePercentiles(collect_weights()));

  // Compute expected_std by visit_count_pre bin (intervals of 5, capped at
  // 200). Everything at n>=200 is collapsed into a single weighted average.
  constexpr int kExpectedStdCap = 200;
  std::map<int, float> bin_expected_std;
  {
    std::map<int, std::pair<float, int>> bin_sum_count;
    float above_cap_sum = 0.0f;
    int above_cap_cnt = 0;
    for (const auto& s : all_move_stats) {
      if (s.sampled_raw_policy) continue;
      if (s.v_outcome_stddev <= 0 || !std::isfinite(s.v_outcome_stddev))
        continue;
      if (s.visit_count_pre <= 0) continue;
      const int n = static_cast<int>(s.visit_count_pre);
      if (n >= kExpectedStdCap) {
        above_cap_sum += s.v_outcome_stddev;
        ++above_cap_cnt;
      } else {
        const int bin = (n / 5) * 5;
        bin_sum_count[bin].first += s.v_outcome_stddev;
        bin_sum_count[bin].second += 1;
      }
    }
    for (const auto& [bin, sc] : bin_sum_count)
      if (sc.second > 0) bin_expected_std[bin] = sc.first / sc.second;
    if (above_cap_cnt > 0)
      bin_expected_std[kExpectedStdCap] = above_cap_sum / above_cap_cnt;
  }

  // v_outcome_stddev_adj: self-consistent std_adj percentiles (written before
  // expected_std.n* scalars so Python can read them in field order).
  write_percentiles(
      "v_outcome_stddev_adj",
      ComputePercentiles(collect_stats([&](const MoveSearchStats& s) {
        if (s.v_outcome_stddev <= 0 || s.visit_count_pre <= 0) return 0.0f;
        const int n = static_cast<int>(s.visit_count_pre);
        const int bin = n >= kExpectedStdCap ? kExpectedStdCap : (n / 5) * 5;
        auto it = bin_expected_std.find(bin);
        if (it == bin_expected_std.end() || it->second <= 0.0f) return 0.0f;
        return s.v_outcome_stddev / it->second;
      })));

  // Write expected_std.nN= scalar lines (key=value, parsed by C++/Python).
  for (const auto& [bin, expected] : bin_expected_std)
    absl::FPrintF(stats_file, "expected_std.n%d=%f\n", bin, expected);

  // Write scalar metadata: mean sel_mult (used by Python to compute
  // sel_mult_base = 1 / mean for the next generation).
  float sel_mult_sum = 0.0f;
  float sel_mult_count = 0;
  for (const auto& s : all_move_stats) {
    if (s.sampled_raw_policy) continue;
    if (!std::isfinite(s.sel_mult_modifier)) continue;
    sel_mult_sum += s.sel_mult_modifier_weight * s.sel_mult_modifier;
    sel_mult_count += s.sel_mult_modifier_weight;
  }
  const float sel_mult_mean =
      sel_mult_count > 0 ? sel_mult_sum / sel_mult_count : 1.0f;
  absl::FPrintF(stats_file, "sel_mult_mean=%f\n", sel_mult_mean);

  fclose(stats_file);

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
