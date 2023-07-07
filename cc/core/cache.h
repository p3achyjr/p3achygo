#ifndef __CC_CORE_CACHE_H_
#define __CC_CORE_CACHE_H_

#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <vector>

#include "absl/hash/hash.h"
#include "cc/constants/constants.h"

namespace core {

/*
 * A simple cache implementation. Only holds one element per cache key.
 *
 * !! All operations are unchecked! Calling into a cache created from the
 * !! default constructor will likely lead to segfaults.
 */
template <typename K, typename V>
class Cache final {
 public:
  Cache() : size_(0) {}
  Cache(int size) : size_(size) { cache_.resize(size_); }

  ~Cache() = default;

  // Disable Copy.
  Cache(Cache const&) = delete;
  Cache& operator=(Cache const&) = delete;
  Cache(Cache&&) = default;
  Cache& operator=(Cache&& other) {
    size_ = other.size_;
    cache_ = other.cache_;

    return *this;
  }

  void Insert(const K& key, const V& val) {
    size_t hash = absl::HashOf(key);
    size_t tbl_index = hash % size_;
    cache_[tbl_index] = Entry{hash, val};
  }

  bool Contains(const K& key) {
    size_t hash = absl::HashOf(key);
    size_t tbl_index = hash % size_;
    const std::optional<Entry>& elem = cache_[tbl_index];
    if (!elem || elem->hash != hash) {
      return false;
    }

    return true;
  }

  std::optional<V> Get(const K& key) {
    size_t hash = absl::HashOf(key);
    size_t tbl_index = hash % size_;
    const std::optional<Entry>& elem = cache_[tbl_index];
    if (!elem || elem->hash != hash) {
      return {};
    }

    return elem->val;
  }

 private:
  struct Entry {
    size_t hash;
    V val;
  };

  int size_;
  std::vector<std::optional<Entry>> cache_;
};

}  // namespace core

#endif
