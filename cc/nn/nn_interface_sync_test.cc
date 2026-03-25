// Stress-tests two synchronization invariants in NNInterface:
//
//  1. RunInference() must never overlap with GetBatch() for any thread slot.
//     CountingEngine writes a generation counter to per-slot buffers in
//     RunInference(), yielding between writes to maximize the race window.
//     GetBatch() reads all elements and asserts they are identical.
//
//  2. Each thread must receive a result from an inference cycle that ran
//     AFTER its LoadBatch() call (no stale results). LoadBatch() records the
//     engine generation at load time; GetBatch() asserts result_gen > load_gen.
//
// Adversarial conditions:
//   - 128 threads on a 32-core machine (4:1 ratio, 4 OS scheduling rounds).
//   - Per-iteration random jitter (100–1000 µs) de-syncs threads so they do
//     not all call LoadAndGetInference in lockstep.
//   - Every kSlowStride-th thread sleeps 5–50 ms before loading, forcing the
//     200 µs timeout to fire and create partial inference batches. Slow threads
//     must wait for the next cycle that includes them and must not receive a
//     stale result from the partial batch they missed.
//
// Runs for kTestSeconds (default 120). Passes iff no race or stale result
// is detected and no generation goes backwards.

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "cc/core/doctest_include.h"
#include "cc/core/probability.h"
#include "cc/game/game.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/nn_interface.h"

namespace nn {
namespace {

// Each thread slot holds kSlotElems elements. A race is detectable when
// RunInference() writes some elements in generation G+1 while GetBatch()
// is still reading elements written in generation G.
static constexpr int kSlotElems = 32;

// Test duration. Long enough to exercise thousands of inference cycles.
static constexpr int kTestSeconds = 120;

// Every kSlowStride-th thread sleeps much longer than the inference timeout
// to force partial batch scenarios.
static constexpr int kSlowStride = 8;

class CountingEngine : public Engine {
 public:
  explicit CountingEngine(int num_threads)
      : num_threads_(num_threads),
        generation_(0),
        buffer_(num_threads * kSlotElems),
        load_gen_(num_threads),
        race_detected_(false),
        stale_detected_(false) {}

  Kind kind() override { return Kind::kUnknown; }
  std::string path() override { return ""; }
  void GetOwnership(int,
                    std::array<float, constants::kNumBoardLocs>&) override {}

  // Record the engine generation at the moment this thread's data is loaded.
  // RunInference() will strictly advance generation_, so GetBatch() can verify
  // the result is newer than what was seen here.
  void LoadBatch(int t, const GoFeatures&) override {
    load_gen_[t].store(generation_.load(std::memory_order_acquire),
                       std::memory_order_release);
  }

  // Write generation counter to every element of every thread slot.
  // yield() between slots widens the window for races.
  void RunInference() override {
    int gen = generation_.fetch_add(1, std::memory_order_relaxed) + 1;
    for (int t = 0; t < num_threads_; ++t) {
      for (int i = 0; i < kSlotElems; ++i) {
        buffer_[t * kSlotElems + i].store(gen, std::memory_order_relaxed);
      }
      std::this_thread::yield();
    }
  }

  // Read all elements of this slot. Two checks:
  //   Race:      all elements must be equal (no concurrent RunInference write).
  //   Freshness: result gen must be strictly greater than the gen captured at
  //              LoadBatch() time, i.e. the result is from an inference that
  //              ran after this thread's data was loaded.
  void GetBatch(int batch_id, NNInferResult& result) override {
    int vals[kSlotElems];
    for (int i = 0; i < kSlotElems; ++i) {
      vals[i] =
          buffer_[batch_id * kSlotElems + i].load(std::memory_order_relaxed);
      std::this_thread::yield();  // widen race window
    }

    // Check 1: internal consistency.
    for (int i = 1; i < kSlotElems; ++i) {
      if (vals[i] != vals[0]) {
        race_detected_.store(true, std::memory_order_relaxed);
      }
    }

    // Check 2: freshness. load_gen_ was stored with release in LoadBatch();
    // we pair with acquire here to establish the happens-before.
    int load_gen = load_gen_[batch_id].load(std::memory_order_acquire);
    if (vals[0] <= load_gen) {
      stale_detected_.store(true, std::memory_order_relaxed);
    }

    result.move_logits.fill(static_cast<float>(vals[0]));
    result.move_probs.fill(0.0f);
    result.opt_move_probs.fill(0.0f);
    result.value_probs.fill(0.0f);
    result.score_probs.fill(0.0f);
  }

  bool race_detected() const {
    return race_detected_.load(std::memory_order_relaxed);
  }
  bool stale_detected() const {
    return stale_detected_.load(std::memory_order_relaxed);
  }

 private:
  const int num_threads_;
  std::atomic<int> generation_;
  std::vector<std::atomic<int>> buffer_;    // [thread * kSlotElems + i]
  std::vector<std::atomic<int>> load_gen_;  // generation captured at LoadBatch
  std::atomic<bool> race_detected_;
  std::atomic<bool> stale_detected_;
};

// Run the full sync stress-test with a given WakeStrategy.
void RunSyncTest(NNInterface::WakeStrategy strategy) {
  static constexpr int kNumThreads = 128;

  // cache_size=0: every call goes through the full inference pipeline.
  auto* engine = new CountingEngine(kNumThreads);
  NNInterface nn_interface(kNumThreads,
                           /*timeout_us=*/200,
                           /*cache_size=*/0,
                           std::unique_ptr<Engine>(engine), strategy);
  std::atomic<bool> stop{false};
  std::atomic<bool> error{false};

  auto worker = [&](int tid) {
    game::Game game;
    core::Probability prob(static_cast<uint64_t>(tid));
    int last_gen = 0;

    // Per-thread RNG for jitter. Seed uniquely per thread.
    std::mt19937 rng(static_cast<uint32_t>(tid) * 2654435761u);
    std::uniform_int_distribution<int> jitter_us(100, 1000);
    std::uniform_int_distribution<int> slow_ms(5, 50);

    // Every kSlowStride-th thread deliberately misses several inference
    // batches to exercise the partial-batch wakeup path.
    const bool is_slow = (tid % kSlowStride == 0);

    while (!stop.load(std::memory_order_relaxed) &&
           !error.load(std::memory_order_relaxed)) {
      // Jitter: break lockstep so threads load at different times.
      std::this_thread::sleep_for(std::chrono::microseconds(jitter_us(rng)));

      // Slow threads: sleep well past the 200µs timeout so the infer thread
      // fires without them. They must wait for the next cycle that includes
      // their slot and must receive a fresh (not stale) result.
      if (is_slow) {
        std::this_thread::sleep_for(std::chrono::milliseconds(slow_ms(rng)));
      }

      NNInferResult result =
          nn_interface.LoadAndGetInference(tid, game, BLACK, prob);

      int gen = static_cast<int>(result.move_logits[0]);

      // Generation must be non-decreasing.
      if (gen < last_gen) {
        error.store(true, std::memory_order_relaxed);
        return;
      }
      last_gen = gen;
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(kNumThreads);
  for (int tid = 0; tid < kNumThreads; ++tid) {
    workers.emplace_back(worker, tid);
  }

  std::this_thread::sleep_for(std::chrono::seconds(kTestSeconds));
  stop.store(true, std::memory_order_relaxed);

  for (auto& t : workers) {
    t.join();
  }

  CHECK_MESSAGE(!engine->race_detected(),
                "RunInference() overlapped with GetBatch() — "
                "res_ready synchronization is broken");
  CHECK_MESSAGE(!engine->stale_detected(),
                "GetBatch() returned a result from before LoadBatch() — "
                "result freshness violated");
  CHECK_MESSAGE(!error.load(), "generation went backwards — ordering violated");
}

}  // namespace

TEST_CASE("NNInterface sync: kGenCounter — no race or stale result") {
  RunSyncTest(NNInterface::WakeStrategy::kGenCounter);
}

TEST_CASE("NNInterface sync: kMutex — no race or stale result") {
  RunSyncTest(NNInterface::WakeStrategy::kMutex);
}

}  // namespace nn
