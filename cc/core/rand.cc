#include "cc/core/rand.h"

#include <chrono>
#include <iostream>
#include <random>

static constexpr uint64_t PCG_MULT = 6364136223846793005u;

static constexpr uint64_t PCG_INC[4] = {
    1442695040888963407u,  // must all be odd
    6364136223846793007u,
    1865811235122147685u,
    7664345821815920749u,
};

namespace core {

using ::absl::uint128;

namespace {

struct RandResult {
  uint64_t new_state;
  uint32_t rand;
};

inline uint32_t rotate32(uint32_t x, unsigned r) {
  return x >> r | x << (-r & 31);
}

// use top 5 bits to determine rotation (log2(32) = 5)
inline RandResult pcg32(uint64_t state, uint64_t pcg_add) {
  uint64_t x = state;
  unsigned count = (unsigned)(x >> 59);  // 59 = 64 - 5

  state = x * PCG_MULT + pcg_add;
  x ^= x >> 18;  // 18 = (64 - 27)/2
  return RandResult{
      state, rotate32((uint32_t)(x >> 27), count)  // 27 = 32 - 5
  };
}

}  // namespace

PRng::PRng() {
  uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
  state_[0] = seed + PCG_INC[0];
  state_[1] = seed + PCG_INC[1];
  state_[2] = seed + PCG_INC[2];
  state_[3] = seed + PCG_INC[3];
}

PRng::PRng(uint64_t seed) : PRng::PRng(seed, 0, 0, 0) {}

PRng::PRng(uint64_t seed0, uint64_t seed1) : PRng::PRng(seed0, seed1, 0, 0) {}

PRng::PRng(uint64_t seed0, uint64_t seed1, uint64_t seed2, uint64_t seed3) {
  state_[0] = seed0 + PCG_INC[0];
  state_[1] = seed1 + PCG_INC[1];
  state_[2] = seed2 + PCG_INC[2];
  state_[3] = seed3 + PCG_INC[3];
}

uint32_t PRng::next() {
  RandResult rand_result = pcg32(state_[0], PCG_INC[0]);
  state_[0] = rand_result.new_state;

  return rand_result.rand;
}

uint64_t PRng::next64() {
  RandResult rand_result0 = pcg32(state_[0], PCG_INC[0]);
  RandResult rand_result1 = pcg32(state_[1], PCG_INC[1]);

  state_[0] = rand_result0.new_state;
  state_[1] = rand_result1.new_state;

  return ((uint64_t)rand_result0.rand << 32) | ((uint64_t)rand_result1.rand);
}

uint128 PRng::next128() {
  RandResult rand_result0 = pcg32(state_[0], PCG_INC[0]);
  RandResult rand_result1 = pcg32(state_[1], PCG_INC[1]);
  RandResult rand_result2 = pcg32(state_[2], PCG_INC[2]);
  RandResult rand_result3 = pcg32(state_[3], PCG_INC[3]);

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
  if (lo == hi) {
    return lo;
  }

  uint32_t width = uint32_t(hi) - uint32_t(lo);
  uint32_t bit_mask = 0;
  uint32_t shift = 1;
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
