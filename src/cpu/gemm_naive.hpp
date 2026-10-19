#pragma once
#include <cstdint>

namespace pg::cpu {

// Reference GEMM: cache-blocked i-k-j with row-local accumulation. Correct for
// every shape including zero-sized dimensions. This kernel is the permanent
// correctness oracle for the packed kernels that replace it on the fast path;
// it never gets SIMD intrinsics so it stays trivially auditable. Element (i, p)
// of A is a[i * rs_a + p * cs_a] and (p, j) of B is b[p * rs_b + j * cs_b], any
// stride; C is C-contiguous with ldc = n and is overwritten.
template <typename T>
void gemm_naive(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
                T* c, int64_t m, int64_t k, int64_t n);

extern template void gemm_naive<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t,
                                       float*, int64_t, int64_t, int64_t);
extern template void gemm_naive<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t,
                                        double*, int64_t, int64_t, int64_t);

} // namespace pg::cpu
