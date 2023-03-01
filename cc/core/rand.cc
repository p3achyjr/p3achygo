#include "cc/core/rand.h"

#include <chrono>
#include <iostream>
#include <random>

#ifndef PCG_MULT
#define PCG_MULT 6364136223846793005u
#endif

#ifndef PCG_ADD
#define PCG_ADD 1442695040888963407u
#endif

namespace core {

using ::absl::uint128;

namespace {

std::uniform_int_distribution<> distrib(1, 6);

struct RandResult {
  uint64_t new_state;
  uint32_t rand;
};

inline uint32_t rotate32(uint32_t x, unsigned r) {
  return x >> r | x << (-r & 31);
}

// use top 5 bits to determine rotation (log2(32) = 5)
inline RandResult pcg32(uint64_t state) {
  uint64_t x = state;
  unsigned count = (unsigned)(x >> 59);  // 59 = 64 - 5

  state = x * PCG_MULT + PCG_ADD;
  x ^= x >> 18;  // 18 = (64 - 27)/2
  return RandResult{
      state, rotate32((uint32_t)(x >> 27), count)  // 27 = 32 - 5
  };
}

}  // namespace

PRng::PRng() {
  uint64_t seed = static_cast<uint64_t>(std::time(nullptr));
  state_[0] = seed + PCG_ADD;
  state_[1] = seed + PCG_ADD + 1;
  state_[2] = seed + PCG_ADD + 2;
  state_[3] = seed + PCG_ADD + 3;
}

PRng::PRng(uint64_t seed) : PRng::PRng(seed, 0, 0, 0) {}

PRng::PRng(uint64_t seed0, uint64_t seed1) : PRng::PRng(seed0, seed1, 0, 0) {}

PRng::PRng(uint64_t seed0, uint64_t seed1, uint64_t seed2, uint64_t seed3) {
  state_[0] = seed0 + PCG_ADD;
  state_[1] = seed1 + PCG_ADD;
  state_[2] = seed2 + PCG_ADD;
  state_[3] = seed3 + PCG_ADD;
}

uint32_t PRng::next() {
  RandResult rand_result = pcg32(state_[0]);
  state_[0] = rand_result.new_state;

  return rand_result.rand;
}

uint64_t PRng::next64() {
  RandResult rand_result0 = pcg32(state_[0]);
  RandResult rand_result1 = pcg32(state_[1]);

  state_[0] = rand_result0.new_state;
  state_[1] = rand_result1.new_state;

  return ((uint64_t)rand_result0.rand << 32) | ((uint64_t)rand_result1.rand);
}

uint128 PRng::next128() {
  RandResult rand_result0 = pcg32(state_[0]);
  RandResult rand_result1 = pcg32(state_[1]);
  RandResult rand_result2 = pcg32(state_[2]);
  RandResult rand_result3 = pcg32(state_[3]);

  state_[0] = rand_result0.new_state;
  state_[1] = rand_result1.new_state;
  state_[2] = rand_result2.new_state;
  state_[3] = rand_result3.new_state;

  return absl::MakeUint128((static_cast<uint64_t>(rand_result0.rand) << 32) |
                               (static_cast<uint64_t>(rand_result1.rand)),
                           (static_cast<uint64_t>(rand_result2.rand) << 32) |
                               (static_cast<uint64_t>(rand_result3.rand)));
}

int RandRange(PRng& prng, int lo, int hi) {
  int width = hi - lo;
  uint32_t bit_mask = 0;
  int shift = 1;
  while (width >> shift) {
    bit_mask = bit_mask << 1 | 0x1;
    ++shift;
  }

  bit_mask = bit_mask << 1 | 0x1;

  uint32_t rand = prng.next();
  while ((rand & bit_mask) >= width) {
    rand = prng.next();
  }

  return (rand & bit_mask) + lo;
}

}  // namespace core