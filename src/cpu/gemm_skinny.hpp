#pragma once
#include <cstdint>

namespace pg::cpu {

// The threading both skinny kernels share, so the twin splits the work the same
// way. Threads take whole slabs of SKINNY_SLAB rows or columns of C. The work
// is m * n * (k + 1), the multiply-adds plus the stores of C, which is what a
// rank-1 product spends its time on. Single-call diagnostic with a forced team
// (f64, 1 vs 6 threads): (1, 32, 256) at 8.4K units 4.6 vs 3.8 us, (1, 64, 512)
// at 33K 11.9 vs 5.0, (1, 128, 1024) at 132K 30.8 vs 12.6; f32 gained nothing
// at 8.4K.
inline constexpr int64_t SKINNY_SLAB = 64;
inline constexpr double SKINNY_SERIAL_BELOW = 32768.0;
inline constexpr double SKINNY_PER_THREAD = 8192.0;

// GEMM for shapes with m <= 4, n <= 4, or k <= 4 (row vector times matrix,
// matrix times column vector, rank-k updates), where the cost is moving the
// long operand and C through memory. Same arguments and bits as gemm_small (one
// FMA chain per element in k order from +0.0, stored as acc - (-0.0)); it is
// correct for any shape. Threads split the longer output dimension only, never
// k, so the bits do not depend on the team size. gemm_skinny is defined in the
// AVX2-flagged gemm_skinny_avx2.cpp, gemm_skinny_naive (its bitwise-equal scalar
// twin) in a fallback TU; this header stays intrinsic-free.
template <typename T>
void gemm_skinny(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
                 T* c, int64_t m, int64_t k, int64_t n);

template <typename T>
void gemm_skinny_naive(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
                       T* c, int64_t m, int64_t k, int64_t n);

extern template void gemm_skinny<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t);
extern template void gemm_skinny<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t);
extern template void gemm_skinny_naive<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t);
extern template void gemm_skinny_naive<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t);

} // namespace pg::cpu
