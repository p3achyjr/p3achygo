#include <cmath>
#include <limits>

#include "cc/core/probability.h"

namespace core {

Probability::Probability(uint64_t seed) : prng_(0) {}

PRng& Probability::prng() { return prng_; }

float Probability::GumbelSample() { return 0.0; }

float Probability::Uniform() { return 0.0; }

}  // namespace core
