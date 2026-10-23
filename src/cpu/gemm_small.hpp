#pragma once
#include <cstdint>

namespace pg::cpu {

// Register-blocked GEMM that reads A and B in place through element strides
// (any sign, any value on a length-1 dimension) and overwrites the C-contiguous
// m x n output c. No heap. From 2^16 vector FMAs a team of two threads (capped
// by the thread setting) splits the rows of C on six-row tile boundaries, or
// its columns on two-vector panel boundaries when an operand is column-major and
// the panels split evenly; gemm_small_ldc is always single-threaded. Each output element is one FMA
// chain in k order starting from +0.0 and stored as acc - (-0.0), so an
// accumulator that underflowed to -0.0 lands as +0.0. The bits therefore depend
// only on the operand values: not on the strides, not on the tile that computed
// the element, and they equal the packed engine's for k up to its KC, whose
// first k panel is the same chain. gemm_small is defined in the AVX2-flagged
// gemm_small_avx2.cpp; gemm_small_naive is its scalar twin (std::fma, bitwise
// equal) in a fallback TU. This header stays intrinsic-free so fallback TUs can
// include it.
template <typename T>
void gemm_small(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
                T* c, int64_t m, int64_t k, int64_t n);

template <typename T>
void gemm_small_naive(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
                      T* c, int64_t m, int64_t k, int64_t n);

// The same computation into rows of c that are ldc elements apart, so the skinny
// driver can hand each thread a column slab of C or a scratch tile.
template <typename T>
void gemm_small_ldc(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
                    T* c, int64_t ldc, int64_t m, int64_t k, int64_t n);

template <typename T>
void gemm_small_ldc_naive(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
                          T* c, int64_t ldc, int64_t m, int64_t k, int64_t n);

extern template void gemm_small<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t);
extern template void gemm_small<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t);
extern template void gemm_small_naive<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t);
extern template void gemm_small_naive<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t);
extern template void gemm_small_ldc<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t, int64_t);
extern template void gemm_small_ldc<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t, int64_t);
extern template void gemm_small_ldc_naive<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t, int64_t);
extern template void gemm_small_ldc_naive<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t, int64_t);

} // namespace pg::cpu
