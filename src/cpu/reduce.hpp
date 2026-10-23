#pragma once
#include <cstdint>

namespace pg::cpu {

// Summation reductions over a 2-D row-major array, accumulated in the input
// dtype (NumPy's algorithm class sums in the input dtype, not a wider one).
//
// Every result is fixed by the shape alone. A contiguous run (a row for axis 1,
// the whole buffer for axis=None) is summed pairwise: runs of at most 128
// elements go through eight lanes (lane l takes elements l, l + 8, ...) combined
// as ((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7)), then the tail below the
// 8-lane stride in order; a longer run splits at (n / 2) rounded down to a
// multiple of 8 and adds left + right. Axis 0 sums blocks of 128 rows
// sequentially from +0.0 and adds the blocks pairwise (block range [b0, b1)
// splits at b0 + (b1 - b0) / 2), so its error grows with the block size plus
// log2 of the block count instead of with m. Threads take whole subtrees of
// these trees, and their partial results are added in the trees' own order, so
// the bits do not depend on the team size. Accumulators start at +0.0, the
// empty-sum identity, so an all -0.0 input sums to +0.0.
//
// The trees, the threading, and the partial combines live in reduce.cpp; the
// leaf kernels come in two sets with identical arithmetic, the AVX2 set in
// reduce_avx2.cpp and its scalar twin in reduce.cpp, chosen by `avx2`. This
// header stays intrinsic-free.
template <typename T>
T sum_all(const T* a, int64_t m, int64_t n, bool avx2);

// axis 1 sums each row into out[m]; axis 0 sums each column into out[n]. With
// mean set each sum is then divided by the reduced count converted to T, in T,
// as NumPy's total / count does; dividing in double and rounding afterwards
// gives different float32 bits once the count exceeds 2**24.
template <typename T>
void sum_axis(const T* a, T* out, int64_t m, int64_t n, int axis, bool avx2, bool mean);

// Leaf kernels, one set per instruction set. sum_rows: out[r] is the pairwise
// sum of the n elements at x + r * n, for r < rows. sum_block: acc[j] is the
// sequential sum from +0.0 of x[i * ld + j] over i < rows, for j < w.
// sum_block_stream is the same sum for arrays read from DRAM (it prefetches
// rows ahead); sum_block_naive is the twin of both.
template <typename T>
void sum_rows_avx2(const T* x, int64_t n, int64_t rows, T* out);
template <typename T>
void sum_block_avx2(const T* x, int64_t ld, int64_t rows, int64_t w, T* acc);
template <typename T>
void sum_block_stream_avx2(const T* x, int64_t ld, int64_t rows, int64_t w, T* acc);
template <typename T>
void sum_rows_naive(const T* x, int64_t n, int64_t rows, T* out);
template <typename T>
void sum_block_naive(const T* x, int64_t ld, int64_t rows, int64_t w, T* acc);

extern template float sum_all<float>(const float*, int64_t, int64_t, bool);
extern template double sum_all<double>(const double*, int64_t, int64_t, bool);
extern template void sum_axis<float>(const float*, float*, int64_t, int64_t, int, bool, bool);
extern template void sum_axis<double>(const double*, double*, int64_t, int64_t, int, bool, bool);
extern template void sum_rows_avx2<float>(const float*, int64_t, int64_t, float*);
extern template void sum_rows_avx2<double>(const double*, int64_t, int64_t, double*);
extern template void sum_block_avx2<float>(const float*, int64_t, int64_t, int64_t, float*);
extern template void sum_block_avx2<double>(const double*, int64_t, int64_t, int64_t, double*);
extern template void sum_block_stream_avx2<float>(const float*, int64_t, int64_t, int64_t, float*);
extern template void sum_block_stream_avx2<double>(const double*, int64_t, int64_t, int64_t, double*);
extern template void sum_rows_naive<float>(const float*, int64_t, int64_t, float*);
extern template void sum_rows_naive<double>(const double*, int64_t, int64_t, double*);
extern template void sum_block_naive<float>(const float*, int64_t, int64_t, int64_t, float*);
extern template void sum_block_naive<double>(const double*, int64_t, int64_t, int64_t, double*);

} // namespace pg::cpu
