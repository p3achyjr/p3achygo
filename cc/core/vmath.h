#ifndef __CC_CORE_VMATH_H_
#define __CC_CORE_VMATH_H_

#include <immintrin.h>

#include <array>
#include <cfloat>
#include <cmath>

#define MM_ALIGN 16

namespace core {

static constexpr int kMmSize = 4;
static constexpr float kMinFloat = -FLT_MAX;

// Vectorized Max.
// Treat array as MM_SIZE-way sliced array. Calculate max of each slice and
// reduce at the end.
// !! Requires arguments as MM_ALIGN-byte aligned.
template <size_t N>
float MaxV(const std::array<float, N> floats) {
  if (N < kMmSize) {
    float max = kMinFloat;
    for (int i = 0; i < N; ++i) {
      if (floats[i] > max) max = floats[i];
    }

    return max;
  }

  const float* data = floats.data();

  // Populate initial batch
  __m128 maxes = _mm_load_ps(&data[0]);
  int mm_size = (N / kMmSize) * kMmSize;
  for (int i = kMmSize; i < mm_size; i += kMmSize) {
    // Element-wise max.
    __m128 batch = _mm_load_ps(&data[i]);
    maxes = _mm_max_ps(batch, maxes);
  }

  // Get max of stragglers.
  float max = kMinFloat;
  for (int i = mm_size; i < N; ++i) {
    if (data[i] > max) max = data[i];
  }

  // Get actual max.
  alignas(MM_ALIGN) float vec_maxes[kMmSize];
  _mm_store_ps(vec_maxes, maxes);
  for (int i = 0; i < kMmSize; ++i) {
    if (vec_maxes[i] > max) max = vec_maxes[i];
  }

  return max;
}

// Vectorized Sum.
// !! Requires arguments as MM_ALIGN-byte aligned.
template <size_t N>
float SumV(const std::array<float, N> floats) {
  if (N < kMmSize) {
    // normal operation.
    float sum = 0;
    for (int i = 0; i < N; ++i) {
      sum += floats[i];
    }

    return sum;
  }

  const float* data = floats.data();

  __m128 sums = _mm_load_ps(&data[0]);
  int mm_size = (N / kMmSize) * kMmSize;
  for (int i = kMmSize; i < mm_size; i += kMmSize) {
    // Element-wise sum.
    __m128 batch = _mm_load_ps(&data[i]);
    sums = _mm_add_ps(batch, sums);
  }

  // Sum together xmm portion.
  alignas(MM_ALIGN) float vec_sums[kMmSize];
  float sum = 0;
  _mm_store_ps(vec_sums, sums);

  for (int i = 0; i < kMmSize; ++i) {
    sum += vec_sums[i];
  }

  // Sum stragglers.
  for (int i = mm_size; i < N; ++i) {
    sum += data[i];
  }

  return sum;
}

// Vectorized Softmax.
// !! Requires arguments as MM_ALIGN-byte aligned.
template <size_t N>
std::array<float, N> SoftmaxV(const std::array<float, N>& logits) {
  float max = MaxV(logits);
  int mm_size = (N / kMmSize) * kMmSize;

  // Calculate normalized logits.
  alignas(MM_ALIGN) float norm_logits[N];
  const float* logits_data = logits.data();
  __m128 max_vec = _mm_set1_ps(max);
  for (int i = 0; i < mm_size; i += kMmSize) {
    // Subtract `max` from each float in batch.
    __m128 batch = _mm_load_ps(&logits_data[i]);
    __m128 normed_batch = _mm_sub_ps(batch, max_vec);
    _mm_store_ps(&norm_logits[i], normed_batch);
  }

  // Finish loop on stragglers.
  for (int i = mm_size; i < N; ++i) {
    norm_logits[i] = logits[i] - max;
  }

  // Calculate e^norm_logits.
  // TODO: maybe implement vectorized exp.
  alignas(MM_ALIGN) std::array<float, N> exps;
  for (int i = 0; i < N; ++i) {
    exps[i] = expf(norm_logits[i]);
  }

  // Calculate total mass.
  float total = SumV(exps);

  // Calculate exps / total
  alignas(MM_ALIGN) std::array<float, N> softmax;
  float* softmax_data = softmax.data();
  float* exps_data = exps.data();
  __m128 total_vec = _mm_set1_ps(total);
  for (int i = 0; i < mm_size; i += kMmSize) {
    // Calculate batch / total.
    __m128 batch = _mm_load_ps(&exps_data[i]);
    __m128 div = _mm_div_ps(batch, total_vec);
    _mm_store_ps(&softmax_data[i], div);
  }

  // Finish on stragglers.
  for (int i = mm_size; i < N; ++i) {
    softmax_data[i] = exps[i] / total;
  }

  return softmax;
}

}  // namespace core

#endif  // __CC_CORE_VMATH_H_
