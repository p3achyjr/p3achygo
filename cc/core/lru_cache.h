#ifndef CORE_LRU_CACHE_H_
#define CORE_LRU_CACHE_H_

#include <list>
#include <optional>

#include "absl/container/flat_hash_map.h"

namespace core {

/*
 * Simple LRU Cache Implementation. Not thread-safe.
 *
 * Type `K` must be hashable via absl::Hash.
 */
template <typename K, typename V>
class LRUCache final {
 public:
  LRUCache() : max_size_(0) {}
  LRUCache(int max_size) : max_size_(max_size) {}
  ~LRUCache() = default;

  bool Contains(const K& key) const { return cache_.contains(key); }

  std::optional<V> Get(const K& key) {
    auto v_it = cache_.find(key);
    if (v_it == cache_.end()) {
      return std::nullopt;
    }

    // Promote to MRU.
    lru_list_.splice(lru_list_.end(), lru_list_, v_it->second.second);
    return v_it->second.first;
  }

  void Insert(const K& key, const V& val) {
    auto v_it = cache_.find(key);
    if (v_it != cache_.end()) {
      // Update value and promote to MRU.
      v_it->second.first = val;
      lru_list_.splice(lru_list_.end(), lru_list_, v_it->second.second);
      return;
    }

    lru_list_.push_back(key);
    auto list_it = std::prev(lru_list_.end());
    cache_.emplace(key, std::make_pair(val, list_it));

    if (cache_.size() > static_cast<size_t>(max_size_)) {
      cache_.erase(lru_list_.front());
      lru_list_.pop_front();
    }
  }

 private:
  int max_size_;
  std::list<K> lru_list_;
  absl::flat_hash_map<K, std::pair<V, typename std::list<K>::iterator>> cache_;
};

}  // namespace core

#endif
