// Stress test for the round-free NNInterface (SignalKind::kRoundFree): many
// threads queue rows at any time through LoadAndGetInference,
// LoadEntry / FetchEntry and LoadAndGetOwnership, on an engine whose batch is
// smaller than the thread count. Each thread's position has its own komi, the
// engine echoes the komi of each row, and every thread checks that it gets its
// own value back (no slot mixup, no stale result, no row past the engine
// batch).
//
// Environment variables:
//   TEST_SECONDS=N — seconds per configuration (default 3)

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "cc/core/probability.h"
#include "cc/game/game.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/nn_interface.h"
// After the absl headers, so doctest's CHECK is the one in effect.
#include "cc/core/doctest_include.h"

namespace nn {
namespace {

int TestSeconds() {
  const char* env = std::getenv("TEST_SECONDS");
  if (env) {
    int val = std::atoi(env);
    if (val > 0) return val;
  }
  return 3;
}

// Echoes each row's komi. Rows past the batch size, or a GetBatch of a row
// that the last RunInference did not load, are errors.
class EchoEngine final : public Engine {
 public:
  explicit EchoEngine(int batch_size)
      : batch_size_(batch_size), in_(batch_size), out_(batch_size) {}

  Kind kind() override { return Kind::kUnknown; }
  std::string path() override { return "echo"; }

  void LoadBatch(int i, const GoFeatures& f) override {
    if (i < 0 || i >= batch_size_) {
      bad_row_.store(true);
      return;
    }
    in_[i] = f.komi;
    ++loaded_;
  }
  void RunInference() override {
    max_rows_ = std::max(max_rows_, loaded_);
    if (loaded_ > 1) multi_row_batches_.fetch_add(1);
    batches_.fetch_add(1);
    for (int i = 0; i < batch_size_; ++i) {
      out_[i] = i < loaded_ ? in_[i] : -1.0f;
      std::this_thread::yield();
    }
    loaded_ = 0;
  }
  void GetBatch(int i, NNInferResult& r) override {
    if (i < 0 || i >= batch_size_ || out_[i] < 0) bad_row_.store(true);
    r = NNInferResult{};
    r.move_logits.fill(out_[i]);
    r.move_probs.fill(out_[i]);
    r.opt_move_probs.fill(out_[i]);
    r.value_probs.fill(out_[i]);
  }
  void GetOwnership(int i,
                    std::array<float, constants::kNumBoardLocs>& own) override {
    if (i < 0 || i >= batch_size_ || out_[i] < 0) bad_row_.store(true);
    own.fill(out_[i] + 0.5f);
  }

  bool bad_row() const { return bad_row_.load(); }
  int max_rows() const { return max_rows_; }
  int64_t batches() const { return batches_.load(); }
  int64_t multi_row_batches() const { return multi_row_batches_.load(); }

 private:
  const int batch_size_;
  // Written and read only by the inference thread.
  std::vector<float> in_;
  std::vector<float> out_;
  int loaded_ = 0;
  int max_rows_ = 0;
  std::atomic<int64_t> batches_{0};
  std::atomic<int64_t> multi_row_batches_{0};
  std::atomic<bool> bad_row_{false};
};

void RunRoundFreeStress(int num_threads, int engine_batch, size_t cache_size,
                        NNInterface::WakeStrategy wake) {
  auto* engine = new EchoEngine(engine_batch);
  NNInterface nn_interface(num_threads, cache_size,
                           std::unique_ptr<Engine>(engine), engine_batch, wake);
  CHECK(nn_interface.signal_kind() == NNInterface::SignalKind::kRoundFree);
  std::atomic<bool> stop{false};
  std::atomic<int> wrong{0};
  std::atomic<int64_t> calls{0};

  auto worker = [&](int tid) {
    core::Probability prob(static_cast<uint64_t>(tid) + 1);
    std::mt19937 rng(static_cast<uint32_t>(tid) * 2654435761u);
    game::Game game;
    // Each thread's komi is its id; with the cache on, a hit for this key is
    // this thread's own earlier result.
    game.SetKomi(static_cast<float>(tid));
    const float expected = static_cast<float>(tid);
    while (!stop.load(std::memory_order_relaxed)) {
      const int op = rng() % 3;
      if (op == 0) {
        NNInferResult r =
            nn_interface.LoadAndGetInference(tid, game, BLACK, prob);
        if (r.value_probs[0] != expected || r.move_logits[0] != expected) {
          wrong.fetch_add(1);
        }
      } else if (op == 1) {
        nn_interface.LoadEntry(tid, /*offset=*/0, game, BLACK, prob);
        if (rng() % 4 == 0) std::this_thread::yield();
        NNInferResult r =
            nn_interface.FetchEntry(tid, /*offset=*/0, game, BLACK);
        if (r.value_probs[0] != expected || r.move_logits[0] != expected) {
          wrong.fetch_add(1);
        }
      } else {
        std::array<float, constants::kNumBoardLocs> own =
            nn_interface.LoadAndGetOwnership(tid, game, BLACK);
        if (own[0] != expected + 0.5f) wrong.fetch_add(1);
      }
      calls.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> workers;
  for (int tid = 0; tid < num_threads; ++tid) {
    workers.emplace_back(worker, tid);
  }
  std::this_thread::sleep_for(std::chrono::seconds(TestSeconds()));
  stop.store(true);
  for (auto& w : workers) w.join();

  INFO("threads=" << num_threads << " batch=" << engine_batch
                  << " cache=" << cache_size << " wake=" << int(wake)
                  << " calls=" << calls.load()
                  << " batches=" << engine->batches()
                  << " multi_row_batches=" << engine->multi_row_batches()
                  << " max_rows=" << engine->max_rows());
  CHECK(wrong.load() == 0);
  CHECK(!engine->bad_row());
  CHECK(engine->max_rows() <= engine_batch);
  CHECK(calls.load() > 0);
  if (cache_size == 0 && engine_batch > 1) {
    CHECK(engine->multi_row_batches() > 0);
  }
}

}  // namespace

TEST_CASE("NNInterface round-free: thread / batch / cache / wake sweep") {
  for (auto wake : {NNInterface::WakeStrategy::kGenCounter,
                    NNInterface::WakeStrategy::kMutex}) {
    for (size_t cache_size : {size_t{0}, size_t{4096}}) {
      RunRoundFreeStress(/*num_threads=*/2, /*engine_batch=*/1, cache_size,
                         wake);
      RunRoundFreeStress(/*num_threads=*/32, /*engine_batch=*/16, cache_size,
                         wake);
      RunRoundFreeStress(/*num_threads=*/64, /*engine_batch=*/64, cache_size,
                         wake);
    }
  }
}

}  // namespace nn
