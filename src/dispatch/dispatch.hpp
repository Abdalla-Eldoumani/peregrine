#pragma once
#include <cstdint>

namespace fme::dispatch {

// Every op routes through here. v1 has one backend (cpu naive); the packed
// CPU kernels, the CUDA backend, and the calibrated crossover table all land
// behind this same entry point so the binding layer never changes.
template <typename T>
void matmul(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

extern template void matmul<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
extern template void matmul<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

// transpose, sum_all, and sum_axis route through the same entry point. They
// have one CPU implementation each (a plain, non-AVX2 TU that runs on every
// CPU), so the routing is a direct forward today; keeping them behind dispatch
// means the CUDA backend can join later without touching the binding layer.
template <typename T>
void transpose(const T* a, T* out, int64_t m, int64_t n);

template <typename T>
T sum_all(const T* a, int64_t m, int64_t n);

template <typename T>
void sum_axis(const T* a, T* out, int64_t m, int64_t n, int axis);

extern template void transpose<float>(const float*, float*, int64_t, int64_t);
extern template void transpose<double>(const double*, double*, int64_t, int64_t);
extern template float sum_all<float>(const float*, int64_t, int64_t);
extern template double sum_all<double>(const double*, int64_t, int64_t);
extern template void sum_axis<float>(const float*, float*, int64_t, int64_t, int);
extern template void sum_axis<double>(const double*, double*, int64_t, int64_t, int);

} // namespace fme::dispatch
