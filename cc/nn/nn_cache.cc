#include "cc/nn/nn_cache.h"

#include <algorithm>

#include "absl/hash/hash.h"
#include "absl/log/check.h"

namespace nn {

NNCache::NNCache(int log2_size, int log2_mutexes)
    : mask_((uint64_t{1} << log2_size) - 1),
      mutex_mask_((uint64_t{1} << std::min(log2_mutexes, log2_size)) - 1),
      entries_(std::make_unique<std::shared_ptr<const Entry>[]>(mask_ + 1)),
      mutexes_(std::make_unique<std::mutex[]>(mutex_mask_ + 1)) {
  CHECK(log2_size >= 0 && log2_size <= 40) << log2_size;
  CHECK(log2_mutexes >= 0 && log2_mutexes <= 31) << log2_mutexes;
}

std::shared_ptr<const NNCache::Entry> NNCache::Get(const NNKey& key) {
  lookups_.fetch_add(1, std::memory_order_relaxed);
  const uint64_t idx = absl::HashOf(key) & mask_;
  std::lock_guard<std::mutex> l(mutexes_[idx & mutex_mask_]);
  const std::shared_ptr<const Entry>& entry = entries_[idx];
  if (entry != nullptr && entry->first == key) {
    hits_.fetch_add(1, std::memory_order_relaxed);
    return entry;
  }
  return nullptr;
}

NNCache::Stats NNCache::TakeStats() {
  return Stats{lookups_.exchange(0, std::memory_order_relaxed),
               hits_.exchange(0, std::memory_order_relaxed)};
}

void NNCache::Insert(const NNKey& key, const NNInferResult& result) {
  // Build the entry before locking and swap it in under the lock; the old
  // entry is freed after unlocking (as KataGo's NNCacheTable::set).
  std::shared_ptr<const Entry> entry = std::make_shared<const Entry>(key, result);
  const uint64_t idx = absl::HashOf(key) & mask_;
  {
    std::lock_guard<std::mutex> l(mutexes_[idx & mutex_mask_]);
    entries_[idx].swap(entry);
  }
}

}  // namespace nn
