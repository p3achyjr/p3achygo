// Stress-tests three synchronization invariants in NNInterface:
//
//  1. RunInference() must never overlap with GetBatch() for any thread slot.
//     CountingEngine writes a per-slot function value to kSlotElems elements in
//     RunInference(), yielding between writes to maximize the race window.
//     GetBatch() reads all elements and asserts they are identical.
//
//  2. Each thread must receive a result from an inference cycle that ran
//     AFTER its LoadBatch() call (no stale results). LoadBatch() records the
//     engine generation at load time; GetBatch() asserts result_gen > load_gen.
//
//  3. Each thread must receive its OWN slot's result (no slot mixup).
//     RunInference() computes f(tid) = (tid + tid) % kPrime for each slot.
//     The worker thread verifies the result matches f(tid).
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
// Environment variables:
//   TEST_SECONDS=N   — test duration (default 120)
//   TEST_CACHE_SIZE=N — per-interface cache size (default 0)

#include <atomic>
#include <chrono>
#include <cstdlib>
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
// RunInference() writes some elements while GetBatch() is still reading.
static constexpr int kSlotElems = 32;

// Mersenne prime used for the per-slot function: f(tid) = (2 * tid) % kPrime.
// Each slot gets a unique value, so slot mixup is detectable.
static constexpr int kPrime = (1 << 19) - 1;  // 524287

// Every kSlowStride-th thread sleeps much longer than the inference timeout
// to force partial batch scenarios.
static constexpr int kSlowStride = 8;

int TestSeconds() {
  const char* env = std::getenv("TEST_SECONDS");
  if (env) {
    int val = std::atoi(env);
    if (val > 0) return val;
  }
  return 120;
}

size_t TestCacheSize() {
  const char* env = std::getenv("TEST_CACHE_SIZE");
  if (env) {
    int val = std::atoi(env);
    if (val >= 0) return static_cast<size_t>(val);
  }
  return 0;
}

// Per-slot function: deterministic, unique per tid.
inline int SlotFn(int tid) { return (tid + tid) % kPrime; }

class CountingEngine : public Engine {
 public:
  explicit CountingEngine(int num_threads)
      : num_threads_(num_threads),
        generation_(0),
        buffer_(num_threads * kSlotElems),
        result_gen_(num_threads),
        load_gen_(num_threads),
        race_detected_(false),
        stale_detected_(false),
        wrong_slot_detected_(false) {}

  Kind kind() override { return Kind::kUnknown; }
  std::string path() override { return ""; }
  void GetOwnership(int,
                    std::array<float, constants::kNumBoardLocs>&) override {}

  // Record the engine generation at the moment this thread's data is loaded.
  void LoadBatch(int t, const GoFeatures&) override {
    load_gen_[t].store(generation_.load(std::memory_order_acquire),
                       std::memory_order_release);
  }

  // Compute f(tid) for each slot and write to all kSlotElems elements.
  // yield() between slots widens the window for races.
  void RunInference() override {
    int gen = generation_.fetch_add(1, std::memory_order_relaxed) + 1;
    for (int t = 0; t < num_threads_; ++t) {
      int fval = SlotFn(t);
      for (int i = 0; i < kSlotElems; ++i) {
        buffer_[t * kSlotElems + i].store(fval, std::memory_order_relaxed);
      }
      result_gen_[t].store(gen, std::memory_order_relaxed);
      std::this_thread::yield();
    }
  }

  // Read all elements of this slot. Three checks:
  //   Race:      all elements must be equal (no concurrent RunInference write).
  //   Freshness: result gen must be strictly greater than load gen.
  //   Slot:      value must equal f(batch_id) (no slot mixup).
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

    // Check 2: freshness.
    int load_gen = load_gen_[batch_id].load(std::memory_order_acquire);
    int res_gen = result_gen_[batch_id].load(std::memory_order_relaxed);
    if (res_gen <= load_gen) {
      stale_detected_.store(true, std::memory_order_relaxed);
    }

    // Check 3: slot correctness.
    if (vals[0] != SlotFn(batch_id)) {
      wrong_slot_detected_.store(true, std::memory_order_relaxed);
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
  bool wrong_slot_detected() const {
    return wrong_slot_detected_.load(std::memory_order_relaxed);
  }

 private:
  const int num_threads_;
  std::atomic<int> generation_;
  std::vector<std::atomic<int>> buffer_;       // [thread * kSlotElems + i]
  std::vector<std::atomic<int>> result_gen_;   // gen written by RunInference
  std::vector<std::atomic<int>> load_gen_;     // gen captured at LoadBatch
  std::atomic<bool> race_detected_;
  std::atomic<bool> stale_detected_;
  std::atomic<bool> wrong_slot_detected_;
};

// Run the full sync stress-test with a given WakeStrategy.
void RunSyncTest(NNInterface::WakeStrategy strategy) {
  static constexpr int kNumThreads = 128;
  const size_t cache_size = TestCacheSize();

  auto* engine = new CountingEngine(kNumThreads);
  NNInterface nn_interface(kNumThreads,
                           /*timeout_us=*/200,
                           /*cache_size=*/cache_size,
                           std::unique_ptr<Engine>(engine), strategy);
  std::atomic<bool> stop{false};
  std::atomic<bool> error{false};

  auto worker = [&](int tid) {
    game::Game game;
    core::Probability prob(static_cast<uint64_t>(tid));
    const int expected = SlotFn(tid);

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

      int got = static_cast<int>(result.move_logits[0]);
      if (got != expected) {
        error.store(true, std::memory_order_relaxed);
        return;
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(kNumThreads);
  for (int tid = 0; tid < kNumThreads; ++tid) {
    workers.emplace_back(worker, tid);
  }

  std::this_thread::sleep_for(std::chrono::seconds(TestSeconds()));
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
  CHECK_MESSAGE(!engine->wrong_slot_detected(),
                "GetBatch() returned wrong slot's result — "
                "slot identity violated");
  CHECK_MESSAGE(!error.load(),
                "worker got wrong result — slot mixup or cache corruption");
}

// ---------------------------------------------------------------------------
// Eval-like test: two NNInterfaces, each game thread alternates between them.
//
// In eval, there are two NNInterfaces (cur_nn and cand_nn). Each game thread
// uses one for black and the other for white, alternating each ply. The
// assignment flips based on game_id % 2 (cur plays black in even games, white
// in odd games). All game threads share both NNInterfaces concurrently.
//
// This exercises:
//   - Two inference threads running simultaneously with interleaved load
//     patterns (some threads loading on nn_a while others load on nn_b).
//   - Partial batches on both interfaces: because threads alternate, at any
//     given moment only ~half the threads are loading on a given interface,
//     so the timeout fires frequently with partial batches.
//   - Cross-interface ordering: a thread must never receive a stale result
//     from interface A just because interface B ran inference in between.
//   - Thread slot reuse across interfaces: thread_id i maps to the same
//     slot on whichever interface it is currently querying.
// ---------------------------------------------------------------------------
void RunDualInterfaceTest(NNInterface::WakeStrategy strategy) {
  // 64 games, each using 1 thread → 64 threads per NNInterface.
  static constexpr int kNumGames = 64;
  const size_t cache_size = TestCacheSize();

  auto* engine_a = new CountingEngine(kNumGames);
  auto* engine_b = new CountingEngine(kNumGames);
  NNInterface nn_a(kNumGames, /*timeout_us=*/200, /*cache_size=*/cache_size,
                   std::unique_ptr<Engine>(engine_a), strategy);
  NNInterface nn_b(kNumGames, /*timeout_us=*/200, /*cache_size=*/cache_size,
                   std::unique_ptr<Engine>(engine_b), strategy);

  std::atomic<bool> stop{false};
  std::atomic<bool> error{false};

  // Each "game" thread alternates between nn_a and nn_b each ply, just as
  // eval alternates between black_nn and white_nn.
  auto game_worker = [&](int game_id) {
    game::Game game;
    core::Probability prob(static_cast<uint64_t>(game_id));
    const int expected = SlotFn(game_id);

    std::mt19937 rng(static_cast<uint32_t>(game_id) * 2654435761u);
    std::uniform_int_distribution<int> jitter_us(100, 1000);
    std::uniform_int_distribution<int> slow_ms(5, 50);

    const bool is_slow = (game_id % kSlowStride == 0);

    // In eval, cur_is_black = (game_id % 2 == 0), so black_nn/white_nn flip.
    NNInterface* black_nn = (game_id % 2 == 0) ? &nn_a : &nn_b;
    NNInterface* white_nn = (game_id % 2 == 0) ? &nn_b : &nn_a;

    int ply = 0;

    while (!stop.load(std::memory_order_relaxed) &&
           !error.load(std::memory_order_relaxed)) {
      // Jitter to desync threads.
      std::this_thread::sleep_for(std::chrono::microseconds(jitter_us(rng)));
      if (is_slow) {
        std::this_thread::sleep_for(std::chrono::milliseconds(slow_ms(rng)));
      }

      // Alternate which interface we query, just like eval alternates
      // between black_nn and white_nn each ply.
      NNInterface* active_nn = (ply % 2 == 0) ? black_nn : white_nn;

      NNInferResult result =
          active_nn->LoadAndGetInference(game_id, game, BLACK, prob);

      int got = static_cast<int>(result.move_logits[0]);
      if (got != expected) {
        error.store(true, std::memory_order_relaxed);
        return;
      }
      ++ply;
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(kNumGames);
  for (int game_id = 0; game_id < kNumGames; ++game_id) {
    workers.emplace_back(game_worker, game_id);
  }

  std::this_thread::sleep_for(std::chrono::seconds(TestSeconds()));
  stop.store(true, std::memory_order_relaxed);

  for (auto& t : workers) {
    t.join();
  }

  CHECK_MESSAGE(!engine_a->race_detected(),
                "nn_a: RunInference() overlapped with GetBatch()");
  CHECK_MESSAGE(!engine_a->stale_detected(),
                "nn_a: GetBatch() returned stale result");
  CHECK_MESSAGE(!engine_a->wrong_slot_detected(),
                "nn_a: GetBatch() returned wrong slot's result");
  CHECK_MESSAGE(!engine_b->race_detected(),
                "nn_b: RunInference() overlapped with GetBatch()");
  CHECK_MESSAGE(!engine_b->stale_detected(),
                "nn_b: GetBatch() returned stale result");
  CHECK_MESSAGE(!engine_b->wrong_slot_detected(),
                "nn_b: GetBatch() returned wrong slot's result");
  CHECK_MESSAGE(!error.load(),
                "worker got wrong result — slot mixup or cache corruption");
}

}  // namespace

TEST_CASE("NNInterface sync: kGenCounter — no race or stale result") {
  RunSyncTest(NNInterface::WakeStrategy::kGenCounter);
}

TEST_CASE("NNInterface sync: kMutex — no race or stale result") {
  RunSyncTest(NNInterface::WakeStrategy::kMutex);
}

TEST_CASE("NNInterface sync: dual-interface eval — kGenCounter") {
  RunDualInterfaceTest(NNInterface::WakeStrategy::kGenCounter);
}

TEST_CASE("NNInterface sync: dual-interface eval — kMutex") {
  RunDualInterfaceTest(NNInterface::WakeStrategy::kMutex);
}

}  // namespace nn
