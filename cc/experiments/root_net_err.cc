// Root NN outputs for golden rows, next to the row's stored root value
// histogram (mcts_value_dist).
//
// Reads the first --num_examples records of --chunk_path, rebuilds each
// position (board + last moves; no ko state), runs one NN eval with a random
// symmetry and writes one packed binary record per row to --out_path:
//   int32 row, int8 color, float32 komi, int8 board[361],
//   int16 last_moves[5], uint32 mcts_value_dist[51],
//   float32 value_probs[2] (loss, win; to-move view), float32 err2_outcome
//   (q6_err head; search uses sqrt of it as init_err_est).
// Records are written in completion order; match by row.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/data/tfrecord/record_reader.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"
#include "cc/proto/feature_util.h"
#include "example.pb.h"

ABSL_FLAG(std::string, model_path, "", "TRT engine.");
ABSL_FLAG(std::string, chunk_path, "", "Golden chunk (.tfrecord.zz).");
ABSL_FLAG(std::string, out_path, "", "Output binary file.");
ABSL_FLAG(int, num_examples, 20000, "Number of leading records to use.");
ABSL_FLAG(int, num_threads, 16, "Worker threads (= NN batch).");
ABSL_FLAG(uint64_t, seed, 20261125, "Symmetry RNG seed.");

namespace {

using ::game::AsLoc;
using ::game::Color;
using ::game::Game;
using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;

constexpr int64_t kTimeoutUs = 400;

struct Row {
  int32_t row;
  Color color;
  float komi;
  std::array<Color, constants::kNumBoardLocs> board;
  std::array<int16_t, constants::kNumLastMoves> last_moves;
  std::array<uint32_t, constants::kNumVBuckets> dist;
};

Game MakeGame(const Row& p) {
  game::Board board(p.komi);
  for (Color c : {BLACK, WHITE}) {
    for (int i = 0; i < constants::kNumBoardLocs; ++i) {
      if (p.board[i] == c) {
        CHECK(game::MoveOk(board.PlayMove(AsLoc(i), c)));
      }
    }
  }
  absl::InlinedVector<game::Move, constants::kNumLastMoves> last_moves;
  for (int16_t m : p.last_moves) last_moves.push_back({p.color, AsLoc(m)});
  return Game(board, last_moves, /*init_mv_num=*/0);
}

void Worker(nn::NNInterface* nn, int tid, int num_threads,
            const std::vector<Row>* rows, std::FILE* out, absl::Mutex* mu) {
  nn->RegisterThread(tid);
  core::Probability prob(absl::GetFlag(FLAGS_seed) + tid);
  for (size_t j = tid; j < rows->size(); j += num_threads) {
    const Row& p = (*rows)[j];
    Game game = MakeGame(p);
    nn::NNInferResult r = nn->LoadAndGetInference(tid, game, p.color, prob);
    absl::MutexLock l(mu);
    std::fwrite(&p.row, sizeof(p.row), 1, out);
    std::fwrite(&p.color, sizeof(p.color), 1, out);
    std::fwrite(&p.komi, sizeof(p.komi), 1, out);
    std::fwrite(p.board.data(), sizeof(p.board), 1, out);
    std::fwrite(p.last_moves.data(), sizeof(p.last_moves), 1, out);
    std::fwrite(p.dist.data(), sizeof(p.dist), 1, out);
    std::fwrite(r.value_probs.data(), sizeof(r.value_probs), 1, out);
    std::fwrite(&r.err2_outcome, sizeof(r.err2_outcome), 1, out);
  }
  nn->UnregisterThread(tid);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  static_assert(sizeof(Color) == 1);
  const int num_examples = absl::GetFlag(FLAGS_num_examples);
  const int num_threads = absl::GetFlag(FLAGS_num_threads);

  std::vector<Row> rows;
  data::SequentialRecordReader reader(absl::GetFlag(FLAGS_chunk_path),
                                      data::RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok());
  std::string record;
  for (int idx = 0; static_cast<int>(rows.size()) < num_examples; ++idx) {
    if (!reader.ReadRecord(&record).ok()) break;
    Example ex;
    CHECK(ex.ParseFromString(record));
    Row p;
    p.row = idx;
    const std::string& b = GetFeatureValues<std::string>("board", ex).Get(0);
    const std::string& lm =
        GetFeatureValues<std::string>("last_moves", ex).Get(0);
    const std::string& c = GetFeatureValues<std::string>("color", ex).Get(0);
    const std::string& d =
        GetFeatureValues<std::string>("mcts_value_dist", ex).Get(0);
    CHECK_EQ(b.size(), sizeof(p.board));
    CHECK_EQ(lm.size(), sizeof(p.last_moves));
    CHECK_EQ(c.size(), sizeof(Color));
    CHECK_EQ(d.size(), sizeof(p.dist));
    std::memcpy(p.board.data(), b.data(), b.size());
    std::memcpy(p.last_moves.data(), lm.data(), lm.size());
    std::memcpy(&p.color, c.data(), sizeof(Color));
    std::memcpy(p.dist.data(), d.data(), d.size());
    p.komi = GetFeatureValues<float>("komi", ex).Get(0);
    rows.push_back(p);
  }

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  auto engine =
      nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                       num_threads, nn::GetVersionFromModelPath(model_path));
  nn::NNInterface nn_interface(num_threads, kTimeoutUs, /*cache_size=*/0,
                               std::move(engine));
  std::FILE* out = std::fopen(absl::GetFlag(FLAGS_out_path).c_str(), "wb");
  CHECK(out);
  absl::Mutex mu;
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(Worker, &nn_interface, t, num_threads, &rows, out,
                         &mu);
  }
  for (auto& t : threads) t.join();
  std::fclose(out);
  LOG(INFO) << "done: " << rows.size() << " rows";
  return 0;
}
