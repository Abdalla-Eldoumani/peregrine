#include "cpu/fused.hpp"
#include "cpu/threads.hpp"

#include <cmath>
#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

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
// bitwise-equal on MSVC: axpby's two roundings, fma3's single std::fma rounding,
// and the NaN-checked relu. GCC may contract the AVX2 axpby into one rounding,
// which this TU cannot (no FMA instruction without the arch flag). Threads split
// the elements exactly as the AVX2 bodies do. All index math is int64_t.

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

// Runs f(lo, hi) on each thread's elements, split at 64-byte lines of out.
template <typename T, typename F>
void elementwise(T* out, int64_t n, const F& f) {
    constexpr int64_t LINE = 64 / static_cast<int64_t>(sizeof(T));
    const auto addr = reinterpret_cast<std::uintptr_t>(out);
    int64_t base = addr % sizeof(T) == 0 ? static_cast<int64_t>((64 - addr % 64) % 64 / sizeof(T)) : 0;
    base = base < n ? base : n;
    const int64_t lines = (n - base) / LINE;
    const int nt = team_size(static_cast<double>(n) * static_cast<double>(sizeof(T)), FUSED_SERIAL_BELOW_BYTES,
                             FUSED_PER_THREAD_BYTES, lines);
    const unsigned csr = mxcsr();
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt) if(nt > 1)
#endif
    {
        const mxcsr_scope fp(csr);
#if defined(_OPENMP)
        const int64_t t = omp_get_thread_num();
        const int64_t team = omp_get_num_threads();
#else
        const int64_t t = 0;
        const int64_t team = 1;
#endif
        f(t == 0 ? 0 : base + LINE * (lines * t / team), t == team - 1 ? n : base + LINE * (lines * (t + 1) / team));
    }
}

} // namespace

template <typename T>
void fused_axpby_naive(const T* x, const T* y, T* out, int64_t n, T a, T b) {
    elementwise(out, n, [=](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) {
            out[i] = a * x[i] + b * y[i];
        }
    });
}

template <typename T>
void fused_fma3_naive(const T* x, const T* y, const T* z, T* out, int64_t n) {
    elementwise(out, n, [=](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) {
            out[i] = std::fma(x[i], y[i], z[i]);
        }
    });
}

template <typename T>
void fused_scaled_relu_naive(const T* x, T* out, int64_t n, T scale) {
    elementwise(out, n, [=](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) {
            out[i] = relu_nan_safe_scalar<T>(scale * x[i]);
        }
    });
}

template void fused_axpby_naive<float>(const float*, const float*, float*, int64_t, float, float);
template void fused_axpby_naive<double>(const double*, const double*, double*, int64_t, double, double);
template void fused_fma3_naive<float>(const float*, const float*, const float*, float*, int64_t);
template void fused_fma3_naive<double>(const double*, const double*, const double*, double*, int64_t);
template void fused_scaled_relu_naive<float>(const float*, float*, int64_t, float);
template void fused_scaled_relu_naive<double>(const double*, double*, int64_t, double);

} // namespace pg::cpu
