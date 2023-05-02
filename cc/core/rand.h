#ifndef __CORE_RAND_H_
#define __CORE_RAND_H_

#include <cstdint>

#include "absl/numeric/int128.h"

namespace core {

/*
 * PCG 64-32 implementation
 *
 * Supports up to 4 concurrent generators, so as to retain a period of 2^64
 * in case we need greater than 32-bit random numbers.
 *
 * NOT THREAD SAFE.
 */
class PRng final {
 public:
  // default constructor will create a PRng object with system time as the
  // initial seed.
  PRng();
  // Constructors create and return a new PRng object, supporting 32, 64, and
  // 128-bit numbers respectively.
  PRng(uint64_t seed);
  PRng(uint64_t seed0, uint64_t seed1);
  PRng(uint64_t seed0, uint64_t seed1, uint64_t seed2, uint64_t seed3);
  ~PRng() = default;
  // Disable Copy
  PRng(PRng const&) = delete;
  PRng& operator=(PRng const&) = delete;

  uint32_t next();
  uint64_t next64();
  absl::uint128 next128();

 private:
  uint64_t state_[4];
};

// Returns int in range [lo, hi)
int RandRange(PRng& rng, int lo, int hi);

}  // namespace core

#endif  // __CORE_RAND_H_
