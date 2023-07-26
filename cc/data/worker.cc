#include "cc/data/worker.h"

#include <algorithm>
#include <filesystem>
#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "cc/core/probability.h"
#include "cc/game/board.h"
#include "cc/recorder/make_tf_example.h"
#include "cc/sgf/parse_sgf.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"

namespace data {
namespace {
namespace fs = std::filesystem;

using namespace ::sgf;
using namespace ::game;

using ::tensorflow::io::RecordWriter;
using ::tensorflow::io::RecordWriterOptions;

// Shard Length for single shard.
static constexpr size_t kShardLen = 350000;
static constexpr int kResignScoreEstimateLb = 5;
static constexpr int kResignScoreEstimateUb = 20;

float ScoreMarginEstimateForResign(int move_count,
                                   core::Probability& probability) {
  auto bound = [](int lb, int ub, int x) {
    return std::min(std::max(x, lb), ub);
  };

  int min_move_count = 150;
  int bound_range = 100;
  int bound_scale = 5;

  float bound_adjustment =
      std::floor(bound(0, bound_range, move_count - min_move_count) *
                 bound_scale / bound_range);
  int score_est = core::RandRange(probability.prng(),
                                  kResignScoreEstimateLb - bound_adjustment,
                                  kResignScoreEstimateUb - bound_adjustment);
  return score_est + 0.5;
}

void FlushShard(const int shard_num, std::vector<tensorflow::Example>& examples,
                std::string out_dir, core::Probability& probability,
                const bool is_dry_run) {
  if (examples.empty()) {
    return;
  }

  // Shuffle.
  std::shuffle(examples.begin(), examples.end(), probability.prng());
  if (is_dry_run) {
    for (const auto& example : examples) {
      std::string data;
      example.SerializeToString(&data);
    }

    examples.clear();
    return;
  }

  // Create File.
  std::string filename =
      fs::path(out_dir) / absl::StrFormat("shard%04d.tfrecord.zz", shard_num);
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file));

  // Create Writer.
  RecordWriterOptions options;
  options.compression_type = RecordWriterOptions::ZLIB_COMPRESSION;
  options.zlib_options.compression_level = 2;
  RecordWriter writer(file.get(), options);

  for (const auto& example : examples) {
    std::string data;
    example.SerializeToString(&data);
    TF_CHECK_OK(writer.WriteRecord(data));
  }

  examples.clear();
}

}  // namespace

void Worker(int worker_id, Coordinator* coordinator, const std::string out_dir,
            const bool is_dry_run) {
  core::Probability probability(worker_id);
  std::vector<tensorflow::Example> examples;
  while (true) {
    std::optional<std::string> file = coordinator->GetFile();
    if (!file) break;

    int num_examples = 0;
    absl::StatusOr<std::unique_ptr<sgf::SgfNode>> sgf_tree =
        ParseSgfFile(*file);
    if (!sgf_tree.ok()) {
      coordinator->MarkError();
      continue;
    }

    GameInfo game_info = ExtractGameInfo(sgf_tree->get());
    if (game_info.result.by_resign) {
      // Populate a fake score estimate.
      float score_margin = ScoreMarginEstimateForResign(
          game_info.main_variation.size(), probability);
      if (game_info.result.winner == BLACK) {
        game_info.result.bscore = score_margin;
      } else {
        game_info.result.wscore = score_margin;
      }
    } else {
      // Add two pass moves for scored games.
      Color last_color = game_info.main_variation.back().color;
      game_info.main_variation.emplace_back(
          Move{OppositeColor(last_color), kPassLoc});
      game_info.main_variation.emplace_back(Move{last_color, kPassLoc});
    }

    Board board;
    for (int i = 0; i < game_info.main_variation.size(); ++i) {
      if (i == game_info.main_variation.size() - 1 &&
          game_info.result.by_resign) {
        // We do not want the net to learn that midgame positions have the next
        // policy as a pass, so just skip here.
        break;
      }

      // Populate last moves as indices.
      std::array<int16_t, constants::kNumLastMoves> last_moves;
      for (int off = 0; off < constants::kNumLastMoves; ++off) {
        int k = constants::kNumLastMoves - off;
        if (i - k < 0) {
          last_moves[off] = kNoopLoc;
        } else {
          Loc last_move = game_info.main_variation[i - k].loc;
          last_moves[off] = last_move;
        }
      }

      Move move = game_info.main_variation[i];
      Move next_move = i < game_info.main_variation.size() - 1
                           ? game_info.main_variation[i + 1]
                           : Move{OppositeColor(move.color), kPassLoc};
      std::array<float, constants::kMaxMovesPerPosition> pi{};
      pi[move.loc] = 1.0;
      tensorflow::Example example = recorder::MakeTfExample(
          board.position(), last_moves, board.GetStonesInAtari(),
          board.GetStonesWithLiberties(2), board.GetStonesWithLiberties(3), pi,
          next_move.loc, game_info.result, 0 /* q30 */, 0 /* q100 */,
          0 /* q200 */, move.color, game_info.komi, BOARD_LEN);
      examples.emplace_back(example);
      if (examples.size() == kShardLen) {
        FlushShard(coordinator->GetShardNum(), examples, out_dir, probability,
                   is_dry_run);
      }

      board.PlayMove(move.loc, move.color);
      ++num_examples;
    }

    coordinator->MarkDone(num_examples);
  }

  // Flush stragglers.
  FlushShard(coordinator->GetShardNum(), examples, out_dir, probability,
             is_dry_run);
}

}  // namespace data
