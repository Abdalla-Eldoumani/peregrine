#pragma once
#include <cstdint>

namespace fme::cpu {

// Reference GEMM: cache-blocked i-k-j with row-local accumulation. Correct for
// every shape including zero-sized dimensions. This kernel is the permanent
// correctness oracle for the packed kernels that replace it on the fast path;
// it never gets SIMD intrinsics so it stays trivially auditable.
template <typename T>
void gemm_naive(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

extern template void gemm_naive<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
extern template void gemm_naive<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

} // namespace fme::cpu
