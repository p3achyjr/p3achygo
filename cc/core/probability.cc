#include "cc/core/probability.h"

#include <cmath>
#include <limits>

namespace core {

float Probability::GumbelSample() {
  float cdf = prng_.next() / static_cast<float>(0xffffffff);
  return -logf(-logf(cdf));
}

}  // namespace core
