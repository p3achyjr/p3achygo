#ifndef __PROBABILITY_H_
#define __PROBABILITY_H_

#include "cc/core/rand.h"

namespace core {

/*
 * Routines for generating probabilistic numbers from a set prng.
 */
class Probability final {
 public:
  // Disable Copy
  Probability(Probability const&) = delete;
  Probability& operator=(Probability const&) = delete;

  // returns a sample x ~ Gumbel(0, 1)
  float GumbelSample();

 private:
  PRng prng_;
};

}  // namespace core

#endif  // __PROBABILITY_H_
