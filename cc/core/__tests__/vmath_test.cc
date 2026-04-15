#include "cc/core/vmath.h"

#include <chrono>

#include "cc/core/doctest_include.h"

#define CHECK_FLOAT_VEC_EQ(x, true)            \
  do {                                         \
    for (size_t i = 0; i < x.size(); ++i) {    \
      CHECK(x[i] == doctest::Approx(true[i])); \
    }                                          \
  } while (0)

namespace core {

__attribute__((noinline)) float ScalarMax(const std::array<float, 362>& a) {
  float m = a[0];
  for (int i = 1; i < 362; ++i) m = std::max(m, a[i]);
  return m;
}

__attribute__((noinline)) float ScalarSum(const std::array<float, 362>& a) {
  float s = 0.0f;
  for (int i = 0; i < 362; ++i) s += a[i];
  return s;
}

__attribute__((noinline)) std::array<float, 362> ScalarSoftmax(
    const std::array<float, 362>& a) {
  std::array<float, 362> out;
  float m = a[0];
  for (int i = 1; i < 362; ++i) m = std::max(m, a[i]);
  float s = 0.0f;
  for (int i = 0; i < 362; ++i) {
    out[i] = expf(a[i] - m);
    s += out[i];
  }
  for (int i = 0; i < 362; ++i) out[i] /= s;
  return out;
}

TEST_CASE("VMathTest") {
  alignas(MM_ALIGN) std::array<float, 3> small = {1.0, 1.5, 2.0};
  alignas(MM_ALIGN) std::array<float, 6> small_uneven = {1.0, 1.5, 2.0,
                                                         2.5, 3.0, 3.5};
  alignas(MM_ALIGN) std::array<float, 8> even = {5.2,  1.3, -4.4, 4.5,
                                                 -3.0, 7.2, 2.1,  -4.0};
  alignas(MM_ALIGN) std::array<float, 10> uneven = {5.2, 1.3, -4.4, 4.5, -3.0,
                                                    7.2, 2.1, -4.0, 9.8, 4.6};
  alignas(MM_ALIGN) std::array<float, 7> all_neg = {
      -149.944, -157.025, -158.732, -158.947, -160.693, -161.818, -161.623};
  // N=4: SIMD path, but loop body never executes and there are no stragglers.
  alignas(MM_ALIGN) std::array<float, 4> n4_exact = {1.0, 2.0, 3.0, 4.0};
  // All equal: softmax must be exactly uniform.
  alignas(MM_ALIGN) std::array<float, 4> all_equal = {2.0, 2.0, 2.0, 2.0};
  // Large spread: one dominant value; tests numerical stability of max subtraction.
  alignas(MM_ALIGN) std::array<float, 4> large_spread = {100.0, 0.0, 0.0, 0.0};

  SUBCASE("Max") {
    CHECK(MaxV(small) == doctest::Approx(2.0));
    CHECK(MaxV(small_uneven) == doctest::Approx(3.5));
    CHECK(MaxV(even) == doctest::Approx(7.2));
    CHECK(MaxV(uneven) == doctest::Approx(9.8));
    CHECK(MaxV(all_neg) == doctest::Approx(-149.944));
    CHECK(MaxV(n4_exact) == doctest::Approx(4.0));
    CHECK(MaxV(all_equal) == doctest::Approx(2.0));
    CHECK(MaxV(large_spread) == doctest::Approx(100.0));
  }

  SUBCASE("Sum") {
    CHECK(SumV(small) == doctest::Approx(4.5));
    CHECK(SumV(small_uneven) == doctest::Approx(13.5));
    CHECK(SumV(even) == doctest::Approx(8.9));
    CHECK(SumV(uneven) == doctest::Approx(23.3));
    CHECK(SumV(all_neg) == doctest::Approx(-1108.782));
    CHECK(SumV(n4_exact) == doctest::Approx(10.0));
    CHECK(SumV(all_equal) == doctest::Approx(8.0));
    CHECK(SumV(large_spread) == doctest::Approx(100.0));
  }

  SUBCASE("Softmax") {
    alignas(MM_ALIGN) std::array<float, 3> small_softmax = {
        0.18632372, 0.30719589, 0.50648039};
    alignas(MM_ALIGN) std::array<float, 8> even_softmax = {
        1.11714669e-01, 2.26131844e-03, 7.56629338e-06, 5.54758628e-02,
        3.06828327e-05, 8.25465956e-01, 5.03265673e-03, 1.12875833e-05};
    alignas(MM_ALIGN) std::array<float, 10> uneven_softmax = {
        9.17561645e-03, 1.85732016e-04, 6.21452908e-07, 4.55647628e-03,
        2.52011581e-06, 6.77991447e-02, 4.13354202e-04, 9.27098797e-07,
        9.12829923e-01, 5.03568507e-03};
    alignas(MM_ALIGN) std::array<float, 7> all_neg_softmax = {
        9.9884784e-01, 8.3996827e-04, 1.5237786e-04, 1.2289763e-04,
        2.1442285e-05, 6.9612906e-06, 8.4600651e-06};
    alignas(MM_ALIGN) std::array<float, 4> n4_exact_softmax = {
        0.03205860, 0.08714432, 0.23688282, 0.64391426};
    alignas(MM_ALIGN) std::array<float, 4> all_equal_softmax = {
        0.25, 0.25, 0.25, 0.25};
    alignas(MM_ALIGN) std::array<float, 4> large_spread_softmax = {
        1.0, 0.0, 0.0, 0.0};
    CHECK_FLOAT_VEC_EQ(SoftmaxV(small), small_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(even), even_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(uneven), uneven_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(all_neg), all_neg_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(n4_exact), n4_exact_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(all_equal), all_equal_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(large_spread), large_spread_softmax);
  }

  SUBCASE("Timing N=362") {
    // 362 = 4*90 + 2: exercises 90 SIMD iterations + 2 stragglers.
    alignas(MM_ALIGN) std::array<float, 362> large;
    for (int i = 0; i < 362; ++i) {
      large[i] = (i % 41) * 0.25f - 5.0f;
    }

    // Scalar reference values to check accuracy.
    float ref_max = large[0];
    for (int i = 1; i < 362; ++i) ref_max = std::max(ref_max, large[i]);

    float ref_sum = 0.0f;
    for (int i = 0; i < 362; ++i) ref_sum += large[i];

    alignas(MM_ALIGN) std::array<float, 362> ref_softmax;
    float ref_exp_sum = 0.0f;
    for (int i = 0; i < 362; ++i) {
      ref_softmax[i] = expf(large[i] - ref_max);
      ref_exp_sum += ref_softmax[i];
    }
    for (int i = 0; i < 362; ++i) ref_softmax[i] /= ref_exp_sum;

    CHECK(MaxV(large) == doctest::Approx(ref_max));
    CHECK(SumV(large) == doctest::Approx(ref_sum));
    CHECK_FLOAT_VEC_EQ(SoftmaxV(large), ref_softmax);

    constexpr int kIters = 100000;
    volatile float sink = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
      sink = MaxV(large);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
      sink = SumV(large);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
      sink = SoftmaxV(large)[0];
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
      sink = ScalarMax(large);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
      sink = ScalarSum(large);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
      sink = ScalarSoftmax(large)[0];
    }
    auto t6 = std::chrono::high_resolution_clock::now();

    auto ns = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };
    MESSAGE("MaxV<362>     (SIMD):   " << ns(t0, t1) / kIters << " ns/call");
    MESSAGE("MaxV<362>     (scalar): " << ns(t3, t4) / kIters << " ns/call");
    MESSAGE("SumV<362>     (SIMD):   " << ns(t1, t2) / kIters << " ns/call");
    MESSAGE("SumV<362>     (scalar): " << ns(t4, t5) / kIters << " ns/call");
    MESSAGE("SoftmaxV<362> (SIMD):   " << ns(t2, t3) / kIters << " ns/call");
    MESSAGE("SoftmaxV<362> (scalar): " << ns(t5, t6) / kIters << " ns/call");
  }
}
}  // namespace core
