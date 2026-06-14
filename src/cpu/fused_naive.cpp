#include "cpu/fused.hpp"

#include <cmath>

namespace pg::cpu {

// Scalar fallback for CPUs without AVX2+FMA. A plain translation unit: it pulls
// in no AVX2 intrinsics header and is deliberately kept OUT of PG_AVX2_SOURCES,
// so the arch flag never reaches it and the no-AVX2 fallback module links it
// without an illegal-instruction crash at import (the legacy import-crash rule,
// the same reason gemm_naive.cpp stays plain). The WSL/GCC build is the
// compile-time leak detector: MSVC would compile a stray intrinsic here
// silently, GCC hard-errors.
//
// The arithmetic mirrors the AVX2 bodies element-for-element so the two paths are
// tolerance-equal (in fact bitwise-equal for these ops): axpby's two roundings,
// fma3's single std::fma rounding, and the NaN-checked relu. All index math is
// int64_t.

namespace {

// Scalar NaN-safe relu: keep a NaN (sx != sx), else clamp at 0. A bare
// std::max(sx, 0) would collapse a NaN to 0 and diverge from np.maximum.
template <typename T>
inline T relu_nan_safe_scalar(T sx) {
    if (sx != sx) {
        return sx;
    }
    return sx > T(0) ? sx : T(0);
}

} // namespace

template <typename T>
void fused_axpby_naive(const T* x, const T* y, T* out, int64_t n, T a, T b) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a * x[i] + b * y[i];
    }
}

template <typename T>
void fused_fma3_naive(const T* x, const T* y, const T* z, T* out, int64_t n) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::fma(x[i], y[i], z[i]);
    }
}

template <typename T>
void fused_scaled_relu_naive(const T* x, T* out, int64_t n, T scale) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        out[i] = relu_nan_safe_scalar<T>(scale * x[i]);
    }
}

template void fused_axpby_naive<float>(const float*, const float*, float*, int64_t, float, float);
template void fused_axpby_naive<double>(const double*, const double*, double*, int64_t, double, double);
template void fused_fma3_naive<float>(const float*, const float*, const float*, float*, int64_t);
template void fused_fma3_naive<double>(const double*, const double*, const double*, double*, int64_t);
template void fused_scaled_relu_naive<float>(const float*, float*, int64_t, float);
template void fused_scaled_relu_naive<double>(const double*, double*, int64_t, double);

} // namespace pg::cpu
