#include "cc/nn/nn_cache.h"

#include <algorithm>

#include "absl/log/check.h"

namespace nn {

PerThreadNNCache::PerThreadNNCache(int num_threads, size_t total_size) {
  CHECK(num_threads > 0) << num_threads;
  caches_.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    caches_.emplace_back(total_size / num_threads);
  }
}

std::optional<NNInferResult> PerThreadNNCache::GetImpl(int thread_id,
                                                       const NNKey& key) {
  return caches_[thread_id].Get(key);
}

void PerThreadNNCache::Insert(int thread_id, const NNKey& key,
                              const NNInferResult& result) {
  caches_[thread_id].Insert(key, result);
}

SharedNNCache::SharedNNCache(size_t num_entries, int log2_mutexes) {
  CHECK(log2_mutexes >= 0 && log2_mutexes <= 31) << log2_mutexes;
  if (num_entries == 0) return;
  int log2_size = 0;
  while ((size_t{2} << log2_size) <= num_entries) ++log2_size;
  log2_mutexes = std::min(log2_mutexes, log2_size);
  entries_ = std::make_unique<Entry[]>(uint64_t{1} << log2_size);
  mask_ = (uint64_t{1} << log2_size) - 1;
  mutexes_ = std::make_unique<std::mutex[]>(uint64_t{1} << log2_mutexes);
  mutex_mask_ = (uint64_t{1} << log2_mutexes) - 1;
}

std::optional<NNInferResult> SharedNNCache::GetImpl(int /*thread_id*/,
                                                    const NNKey& key) {
  if (entries_ == nullptr) return std::nullopt;
  const uint64_t slot = SlotOf(key);
  Entry entry;
  {
    std::lock_guard<std::mutex> l(mutexes_[slot & mutex_mask_]);
    if (entries_[slot] != nullptr && entries_[slot]->first == key) {
      entry = entries_[slot];
    }
  }
  if (entry == nullptr) return std::nullopt;
  return entry->second;
}

void SharedNNCache::Insert(int /*thread_id*/, const NNKey& key,
                           const NNInferResult& result) {
  if (entries_ == nullptr) return;
  Entry entry =
      std::make_shared<const std::pair<NNKey, NNInferResult>>(key, result);
  const uint64_t slot = SlotOf(key);
  {
    std::lock_guard<std::mutex> l(mutexes_[slot & mutex_mask_]);
    entries_[slot].swap(entry);
  }
  // `entry` now holds the replaced entry; it is freed here, unlocked.
}

}  // namespace nn
