#ifndef CORE_PROBABILITY_H_
#define CORE_PROBABILITY_H_

#include "cc/core/rand.h"

namespace core {

/*
 * Routines for generating probabilistic numbers from a set prng.
 */
class Probability final {
 public:
  explicit Probability(uint64_t seed);
  Probability() = default;
  ~Probability() = default;

  // Disable Copy
  Probability(Probability const&) = delete;
  Probability& operator=(Probability const&) = delete;

  // returns underlying prng
  PRng& prng();

  // returns a sample x ~ Gumbel(0, 1)
  float GumbelSample();

  // returns a sample x ~ Uniform(0, 1)
  float Uniform();

  // returns a sample x ~ Gaussian(0, 1)
  float Gaussian();

 private:
  PRng prng_;
};

}  // namespace core

#endif  // CORE_PROBABILITY_H_
