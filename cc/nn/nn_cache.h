#ifndef NN_NN_CACHE_H_
#define NN_NN_CACHE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include "cc/nn/engine/engine.h"
#include "cc/nn/nn_interface_base.h"

namespace nn {

/*
 * NN result cache shared by all threads of an interface. KataGo's NNCacheTable
 * design (neuralnet/nneval.cpp): a direct-mapped table of 2^log2_size slots,
 * slot = hash(key) & mask. Insert overwrites the slot (no LRU). A pool of
 * 2^log2_mutexes mutexes guards the slots (mutex = slot & pool mask; the pool
 * is clamped to the table size). Get compares the full key.
 */
class NNCache final {
 public:
  using Entry = std::pair<NNKey, NNInferResult>;

  NNCache(int log2_size, int log2_mutexes);

  NNCache(NNCache const&) = delete;
  NNCache& operator=(NNCache const&) = delete;

  // The entry for `key`, or nullptr. A returned entry stays valid while held,
  // even if its slot is overwritten meanwhile.
  std::shared_ptr<const Entry> Get(const NNKey& key);
  void Insert(const NNKey& key, const NNInferResult& result);

  // Get() calls and hits since the last call, then resets both.
  struct Stats {
    uint64_t lookups = 0;
    uint64_t hits = 0;
  };
  Stats TakeStats();

 private:
  const uint64_t mask_;
  const uint64_t mutex_mask_;
  std::unique_ptr<std::shared_ptr<const Entry>[]> entries_;
  std::unique_ptr<std::mutex[]> mutexes_;
  std::atomic<uint64_t> lookups_{0};
  std::atomic<uint64_t> hits_{0};
};

}  // namespace nn

#endif
