#include "cc/core/probability.h"

#include <cmath>
#include <limits>

namespace core {

Probability::Probability(uint64_t seed) : prng_(seed) {}

PRng& Probability::prng() { return prng_; }

float Probability::GumbelSample() {
  float cdf = prng_.next() / static_cast<float>(0xffffffff);
  return -logf(-logf(cdf));
}

float Probability::Uniform() {
  // Assumes [s(1) exp(8) man(23)] formatting.
  // Hardcode exp = 127 to normalize exp term to 1.
  const uint32_t rand = prng_.next();

  // upper bits are the most random for PCG.
  const uint32_t man = rand >> 9;
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t x = exp << 23 | man;

  float res;
  std::memcpy(&res, &x, sizeof(x));
  return res - 1.0f;
}

}  // namespace core
