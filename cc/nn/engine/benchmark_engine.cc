#include "cc/nn/engine/benchmark_engine.h"

#include <cfloat>
#include <sstream>

#include "absl/log/log.h"

namespace nn {
namespace {
template <size_t N>
int Argmax(const std::array<float, N>& arr) {
  size_t arg_max = 0;
  float max_val = -FLT_MAX;
  for (size_t i = 0; i < N; ++i) {
    if (arr[i] > max_val) {
      max_val = arr[i];
      arg_max = i;
    }
  }

  return static_cast<int>(arg_max);
}
}  // namespace

void DefaultStats::Update(const NNInferResult& result,
                          const GoDataset::Row& row, double time_us) {
  static constexpr double kMaxLoss = 16;
  int mv_pred = Argmax(result.move_probs);
  int outcome_pred = Argmax(result.value_probs);
  int score_pred =
      Argmax(result.score_probs) + 0.5 - constants::kScoreInflectionPoint;

  int mv = Argmax(row.labels.policy);
  float score = row.labels.score_margin;
  bool did_win = row.labels.did_win;

  float pi_ce_loss = result.move_probs[mv] != 0.0
                         ? -std::log(result.move_probs[mv])
                         : kMaxLoss;
  float v_ce_loss =
      result.move_probs[mv] != 0.0
          ? -std::log(result.value_probs[static_cast<int>(did_win)])
          : kMaxLoss;

  auto avg = [](double avg, double x, double cnt) {
    return ((cnt - 1) / cnt) * avg + (1 / cnt) * x;
  };

  double cnt = stats_["num_examples"] + 1;
  stats_["num_examples"] = cnt;
  stats_["avg_us"] = avg(stats_["avg_us"], time_us, cnt);
  stats_["policy_loss"] = avg(stats_["policy_loss"], pi_ce_loss, cnt);
  stats_["outcome_loss"] = avg(stats_["outcome_loss"], v_ce_loss, cnt);
  stats_["policy_percent"] =
      avg(stats_["policy_percent"], mv == mv_pred ? 1 : 0, cnt);
  stats_["outcome_percent"] =
      avg(stats_["outcome_percent"],
          static_cast<int>(did_win) == outcome_pred ? 1 : 0, cnt);
  stats_["score_diff"] =
      avg(stats_["score_diff"], std::abs(score - score_pred), cnt);
}

std::string DefaultStats::ToString() {
  std::stringstream ss;

  ss << "\nStats:";
  ss << "\n  Avg Inference Time: " << stats_["avg_us"] << "us";
  ss << "\n  Avg Policy Loss: " << stats_["policy_loss"];
  ss << "\n  Avg Outcome Loss: " << stats_["outcome_loss"];
  ss << "\n  Correct Move Percentage: " << stats_["policy_percent"];
  ss << "\n  Correct Outcome Percentage: " << stats_["outcome_percent"];
  ss << "\n  Mean Score Diff: " << stats_["score_diff"];

  return ss.str();
}

void Benchmark(Engine* const engine, GoDataset* const go_ds, Stats& stats) {
  // Warm up.
  LOG(INFO) << "Warming Up...";
  for (int i = 0; i < 100; ++i) {
    engine->RunInference();
  }

  LOG(INFO) << "Starting Benchmark...";
  NNInferResult result;
  for (const std::vector<GoDataset::Row>& batch : *go_ds) {
    for (int batch_id = 0; batch_id < batch.size(); ++batch_id) {
      engine->LoadBatch(batch_id, batch[batch_id].features);
    }

    auto start = std::chrono::steady_clock::now();
    engine->RunInference();
    auto end = std::chrono::steady_clock::now();
    auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    for (int batch_id = 0; batch_id < batch.size(); ++batch_id) {
      engine->GetBatch(batch_id, result);
      stats.Update(result, batch[batch_id], elapsed_us);
    }
  }
}

}  // namespace nn
