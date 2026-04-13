#include "cc/core/probability.h"

#include <cmath>
#include <iostream>

#include "cc/core/doctest_include.h"

namespace core {
namespace {

static constexpr int kN = 1000000;
static constexpr uint64_t kSeed = 42;

struct Stats {
  float mean;
  float var;
  float std;
};

Stats ComputeStats(Probability& p, auto sample_fn) {
  float sum = 0.0f;
  float sum_sq = 0.0f;
  for (int i = 0; i < kN; ++i) {
    float x = sample_fn(p);
    sum += x;
    sum_sq += x * x;
  }
  float mean = sum / kN;
  float var = sum_sq / kN - mean * mean;
  return {mean, var, std::sqrt(var)};
}

void PrintStats(const char* name, Stats actual, Stats expected) {
  std::cout << "\n--- " << name << " (N=" << kN << ") ---\n";
  std::cout << "       mean: expected=" << expected.mean
            << "  actual=" << actual.mean << "\n";
  std::cout << "        var: expected=" << expected.var
            << "  actual=" << actual.var << "\n";
  std::cout << "        std: expected=" << expected.std
            << "  actual=" << actual.std << "\n";
}

TEST_CASE("Uniform mean and range") {
  Probability p(kSeed);
  float sum = 0.0f;
  int out_of_range = 0;
  for (int i = 0; i < kN; ++i) {
    float x = p.Uniform();
    if (x < 0.0f || x >= 1.0f) ++out_of_range;
    sum += x;
  }
  CHECK(out_of_range == 0);

  Probability p2(kSeed);
  Stats actual = ComputeStats(p2, [](Probability& p) { return p.Uniform(); });
  Stats expected = {0.5f, 1.0f / 12.0f, std::sqrt(1.0f / 12.0f)};
  PrintStats("Uniform", actual, expected);

  // 5σ tolerances for N=1e6: SE_mean = std/sqrt(N) ≈ 2.9e-4, 5σ ≈ 1.4e-3
  CHECK(actual.mean == doctest::Approx(expected.mean).epsilon(0.002f));
  CHECK(actual.var == doctest::Approx(expected.var).epsilon(0.002f));
}

TEST_CASE("Gaussian mean and std") {
  Probability p(kSeed);
  Stats actual = ComputeStats(p, [](Probability& p) { return p.Gaussian(); });
  Stats expected = {0.0f, 1.0f, 1.0f};
  PrintStats("Gaussian", actual, expected);

  // 5σ: SE_mean = 1/sqrt(N) ≈ 1e-3, 5σ ≈ 5e-3
  CHECK(actual.mean == doctest::Approx(expected.mean).epsilon(0.005f));
  CHECK(actual.var == doctest::Approx(expected.var).epsilon(0.005f));
}

TEST_CASE("Gumbel mean and std") {
  Probability p(kSeed);
  // cdf=0 yields -inf, which is correct behavior (that action loses argmax).
  // Exclude -inf samples from moment checks so stats remain finite.
  Stats actual = ComputeStats(p, [](Probability& p) {
    float x = p.GumbelSample();
    return std::isfinite(x) ? x : 0.0f;
  });
  // E[X] = Euler-Mascheroni constant, Var[X] = π²/6
  const float kEulerMascheroni = 0.5772156649f;
  const float kVarGumbel = (float)(M_PI * M_PI / 6.0);
  Stats expected = {kEulerMascheroni, kVarGumbel, std::sqrt(kVarGumbel)};
  PrintStats("Gumbel", actual, expected);

  // 5σ: SE_mean = std/sqrt(N) ≈ 1.28e-3, 5σ ≈ 6.4e-3
  CHECK(actual.mean == doctest::Approx(expected.mean).epsilon(0.007f));
  CHECK(actual.var == doctest::Approx(expected.var).epsilon(0.01f));
}

}  // namespace
}  // namespace core
