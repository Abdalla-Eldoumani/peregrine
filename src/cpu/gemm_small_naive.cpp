#include "cpu/gemm_small.hpp"

#include <cmath>

namespace pg::cpu {

// The scalar twin of gemm_small_avx2.cpp and the reference its bits are held
// to: per element, one std::fma chain in k order from +0.0 (std::fma rounds
// once, as VFMADD does), then the same - (-0.0). C's row is the accumulator, so
// B is read along its rows.
template <typename T>
void gemm_small_ldc_naive(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c,
                          int64_t ldc, int64_t m, int64_t k, int64_t n) {
    for (int64_t i = 0; i < m; ++i) {
        T* ci = c + i * ldc;
        for (int64_t j = 0; j < n; ++j) {
            ci[j] = T(0);
        }
        for (int64_t p = 0; p < k; ++p) {
            const T aip = a[i * rs_a + p * cs_a];
            const T* bp = b + p * rs_b;
            for (int64_t j = 0; j < n; ++j) {
                ci[j] = std::fma(aip, bp[j * cs_b], ci[j]);
            }
        }
        for (int64_t j = 0; j < n; ++j) {
            ci[j] = ci[j] - T(-0.0);
        }
    }
}

template <typename T>
void gemm_small_naive(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c,
                      int64_t m, int64_t k, int64_t n) {
    gemm_small_ldc_naive<T>(a, rs_a, cs_a, b, rs_b, cs_b, c, n, m, k, n);
}

template void gemm_small_naive<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t);
template void gemm_small_naive<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t);
template void gemm_small_ldc_naive<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t, int64_t);
template void gemm_small_ldc_naive<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t, int64_t);

} // namespace pg::cpu
