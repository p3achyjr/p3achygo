#ifndef NN_PIPELINED_NN_INTERFACE_H_
#define NN_PIPELINED_NN_INTERFACE_H_

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/game/symmetry.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/nn_cache.h"
#include "cc/nn/nn_interface_base.h"

namespace nn {

/*
 * NN interface for pipelined search: one engine (batch num_threads / W) per
 * wave. Thread t lives in wave t / (num_threads / W), row t % (num_threads /
 * W). Each wave has its own signal count and wake-up; one inference thread
 * runs whichever wave is ready, with the engine call outside mu_ (never two
 * waves at once), so one wave's workers can load while the other wave's batch
 * is on the GPU.
 */
class PipelinedNNInterface final : public NNInterfaceBase {
 public:
  // The NN cache (NNCache) has 2^cache_log2_size slots, shared by all threads.
  PipelinedNNInterface(int num_threads, int64_t timeout, int cache_log2_size,
                       std::vector<std::unique_ptr<Engine>> wave_engines,
                       SignalKind signal_kind, int num_shared_search_tasks,
                       WakeStrategy wake_strategy);
  ~PipelinedNNInterface() override;

  PipelinedNNInterface(PipelinedNNInterface const&) = delete;
  PipelinedNNInterface& operator=(PipelinedNNInterface const&) = delete;

  SignalKind signal_kind() const override { return signal_kind_; }

  NNInferResult LoadAndGetInference(int thread_id, const game::Game& game,
                                    game::Color color_to_move,
                                    core::Probability& probability)
      ABSL_LOCKS_EXCLUDED(mu_) override;
  std::array<float, constants::kNumBoardLocs> LoadAndGetOwnership(
      int thread_id, const game::Game& game, game::Color color_to_move)
      ABSL_LOCKS_EXCLUDED(mu_) override;

  void LoadEntry(int thread_id, int offset, const game::Game& game,
                 game::Color color_to_move, core::Probability& probability)
      ABSL_LOCKS_EXCLUDED(mu_) override;
  NNInferResult FetchEntry(int thread_id, int offset, const game::Game& game,
                           game::Color color_to_move)
      ABSL_LOCKS_EXCLUDED(mu_) override;
  void SignalReadyForInference(int wave) ABSL_LOCKS_EXCLUDED(mu_) override {
    absl::MutexLock l(&mu_);
    ++num_signaled_search_tasks_[wave];
  }
  void UnregisterSearchTask(int wave) ABSL_LOCKS_EXCLUDED(mu_) override {
    absl::MutexLock l(&mu_);
    ++num_exited_search_tasks_[wave];
  }
  bool LoadedFromCache(int thread_id, int offset) const override {
    return load_hits_[thread_id + offset] != nullptr;
  }
  CacheStats TakeCacheStats() override {
    const NNCache::Stats s = cache_.TakeStats();
    return {s.lookups, s.hits};
  }
  int num_waves() const override { return num_waves_; }
  int wave_size() const override { return wave_size_; }

 private:
  static constexpr int kMaxWaves = 2;
  // NNCache mutex pool size (as KataGo's default).
  static constexpr int kCacheLog2Mutexes = 16;

  struct ThreadInfo {
    bool loaded_for_inference =
        false;  // If this thread has loaded its data for inference.
    std::atomic<bool> res_ready{false};  // Whether inference result is ready.
    bool res_cached = false;             // Whether the cache key is cached.

    ThreadInfo() = default;
    ThreadInfo(const ThreadInfo&) = delete;
    ThreadInfo& operator=(const ThreadInfo&) = delete;
  };

  int RowOf(int canonical_tid) const { return canonical_tid % wave_size_; }
  // Cache key: color, position hash and simple-ko point (no move history).
  NNKey MakeKey(const game::Game& game, game::Color color_to_move) const {
    return MakeNNKey(game, color_to_move, /*num_cache_last_moves=*/0,
                     /*cache_key_ko=*/true);
  }
  void LoadBatch(int thread_id, const game::Game& game,
                 game::Color color_to_move, game::Symmetry sym) {
    engines_[WaveOf(thread_id)]->LoadBatch(
        RowOf(thread_id), MakeGoFeatures(game, color_to_move, sym));
  }
  NNInferResult GetBatch(int thread_id, game::Symmetry sym)
      ABSL_LOCKS_EXCLUDED(mu_);
  void SignalLoadedAndBlockUntilReady(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);

  // Inference loop / step over the waves.
  static bool AnyWaveReady(PipelinedNNInterface* self);
  bool WaveShouldInfer(int wave) const;
  void InferWavesLoop();
  void InferWaves() ABSL_LOCKS_EXCLUDED(mu_);

  const int num_threads_;
  const int num_waves_;
  const int wave_size_;  // num_threads_ / num_waves_.
  const SignalKind signal_kind_;
  const int64_t timeout_;

  // Synchronization
  absl::Mutex mu_;
  absl::InlinedVector<ThreadInfo, constants::kMaxNumThreads> thread_info_;

  // Inference thread. Runs InferWavesLoop until told to stop.
  std::thread infer_thread_;
  int last_wave_ = -1;  // Last wave run by the inference thread.
  std::atomic<bool> running_;

  // NN cache shared by all threads.
  NNCache cache_;
  // Per thread: the cache entry found by its last LoadEntry, returned by the
  // matching FetchEntry (not looked up again: another thread may overwrite
  // the slot in between).
  std::vector<std::shared_ptr<const NNCache::Entry>> load_hits_;

  // Engines: one per wave.
  std::vector<std::unique_ptr<Engine>> engines_;

  // Per-thread symmetries.
  absl::InlinedVector<game::Symmetry, constants::kMaxNumThreads> symmetries_;

  // Explicit signalling, per wave.
  const int num_shared_search_tasks_;
  std::array<int, kMaxWaves> num_signaled_search_tasks_{};
  std::array<int, kMaxWaves> num_exited_search_tasks_{};

  // Wakeup strategy, one per wave.
  std::vector<std::unique_ptr<Waker>> wakers_;
};

}  // namespace nn

#endif
