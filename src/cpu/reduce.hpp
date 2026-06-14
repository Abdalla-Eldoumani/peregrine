#pragma once
#include <cstdint>

namespace pg::cpu {

// Summation reductions over a 2-D row-major array. Plain (non-AVX2)
// translation unit: NumPy's own reductions are single-threaded and
// memory-bound, so there is nothing for SIMD to accelerate past the load
// ports, and keeping it intrinsic-free lets it link into the no-AVX2 fallback
// module. Accumulation happens in the input dtype T (locked decision: match
// NumPy's algorithm class, which sums in the input dtype, not a wider one).

// Sum of every element. Pairwise over the flat m*n buffer, matching NumPy's
// pairwise class along a contiguous axis. Returns the input dtype.
template <typename T>
T sum_all(const T* a, int64_t m, int64_t n);

// Sum along one axis of an m-by-n array into out. axis==1 sums each row
// pairwise into out[m]; axis==0 accumulates row-by-row sequentially into
// out[n] (NumPy's class for the non-contiguous axis: it reduces sequentially
// there, so a sequential single pass matches both its result class and its
// error class). The caller sizes out to n for axis 0 and m for axis 1.
template <typename T>
void sum_axis(const T* a, T* out, int64_t m, int64_t n, int axis);

extern template float sum_all<float>(const float*, int64_t, int64_t);
extern template double sum_all<double>(const double*, int64_t, int64_t);
extern template void sum_axis<float>(const float*, float*, int64_t, int64_t, int);
extern template void sum_axis<double>(const double*, double*, int64_t, int64_t, int);

} // namespace pg::cpu
