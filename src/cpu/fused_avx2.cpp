#include "cpu/fused.hpp"

#include <cmath>
#include <immintrin.h>

namespace pg::cpu {

// AVX2+FMA bodies for the three fused ops. This is the ONLY fused TU that may
// include <immintrin.h>: it joins PG_AVX2_SOURCES in CMakeLists, so the arch
// flag (/arch:AVX2 on MSVC, -mavx2 -mfma on GCC) is applied here and nowhere a
// fallback-linked TU can reach (the legacy import-crash isolation rule).
//
// Each op is one flat pass: an AVX2 block over the largest multiple of the lane
// width (8 for f32, 4 for f64) plus a scalar remainder for the tail, wrapped in
// an OpenMP parallel-for over the element range. Elementwise has no
// cross-element accumulation, so the result is bitwise identical regardless of
// thread count -- the schedule splits the index space, never the arithmetic of
// any single element. All index math is int64_t (a flat element count overflows
// int32 long before the allocation would fail).
//
// The scalar tail and the scalar bodies inside the parallel loop must MIRROR the
// vector arithmetic exactly so the tail elements match the blocked ones: axpby
// uses a*x + b*y (the same two roundings as add(mul(a,x), mul(b,y)) -- NOT a
// fused fmadd of b*y + a*x, which would round once and diverge), fma3 uses
// std::fma (the same single rounding as _mm256_fmadd), and scaled_relu uses the
// NaN-checked scalar form that mirrors the blendv idiom.

namespace {

// NaN-safe relu for one f32 vector of v = scale*x. A bare _mm256_max_ps(v, 0)
// returns 0 in any NaN lane (maxps yields its second operand on an unordered
// compare), which does NOT match np.maximum -- NumPy propagates the NaN. The
// fix: take the max, then blend the original v back wherever v was NaN. The NaN
// mask is cmp(v, v, UNORD), all-ones exactly where v != v. Verified bit-for-bit
// against np.maximum(scale*x, 0).
inline __m256 relu_nan_safe_ps(__m256 v) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 m = _mm256_max_ps(v, zero);
    const __m256 nanmask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
    return _mm256_blendv_ps(m, v, nanmask);
}

// The f64 form over 4 lanes; same idiom with the _pd intrinsics.
inline __m256d relu_nan_safe_pd(__m256d v) {
    const __m256d zero = _mm256_setzero_pd();
    const __m256d m = _mm256_max_pd(v, zero);
    const __m256d nanmask = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
    return _mm256_blendv_pd(m, v, nanmask);
}

// Scalar mirror of the relu idiom: keep a NaN (sx != sx), else clamp at 0. A
// bare std::max(sx, 0) would drop the NaN, reintroducing the trap at the tail.
template <typename T>
inline T relu_nan_safe_scalar(T sx) {
    if (sx != sx) {
        return sx;
    }
    return sx > T(0) ? sx : T(0);
}

} // namespace

template <typename T>
void fused_axpby(const T* x, const T* y, T* out, int64_t n, T a, T b) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a * x[i] + b * y[i];
    }
}

template <>
void fused_axpby<float>(const float* x, const float* y, float* out, int64_t n, float a, float b) {
    const __m256 va = _mm256_set1_ps(a);
    const __m256 vb = _mm256_set1_ps(b);
    const int64_t blocked = n & ~int64_t{7};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < blocked; i += 8) {
        const __m256 vx = _mm256_loadu_ps(x + i);
        const __m256 vy = _mm256_loadu_ps(y + i);
        // a*x and b*y each rounded, then summed (rounded again): two roundings,
        // matching the unfused NumPy expression. NOT contracted -- an fmadd of
        // b*y + (a*x) would fuse b*y into the add and round once, diverging from
        // the scalar tail and the naive kernel bit-for-bit (~16% of f32 cases).
        const __m256 r = _mm256_add_ps(_mm256_mul_ps(va, vx), _mm256_mul_ps(vb, vy));
        _mm256_storeu_ps(out + i, r);
    }
    for (int64_t i = blocked; i < n; ++i) {
        out[i] = a * x[i] + b * y[i];
    }
}

template <>
void fused_axpby<double>(const double* x, const double* y, double* out, int64_t n, double a, double b) {
    const __m256d va = _mm256_set1_pd(a);
    const __m256d vb = _mm256_set1_pd(b);
    const int64_t blocked = n & ~int64_t{3};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < blocked; i += 4) {
        const __m256d vx = _mm256_loadu_pd(x + i);
        const __m256d vy = _mm256_loadu_pd(y + i);
        // Two roundings (a*x and b*y rounded, then summed), not a fused
        // contraction -- same arithmetic as the f32 body and the scalar tail.
        const __m256d r = _mm256_add_pd(_mm256_mul_pd(va, vx), _mm256_mul_pd(vb, vy));
        _mm256_storeu_pd(out + i, r);
    }
    for (int64_t i = blocked; i < n; ++i) {
        out[i] = a * x[i] + b * y[i];
    }
}

template <typename T>
void fused_fma3(const T* x, const T* y, const T* z, T* out, int64_t n) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::fma(x[i], y[i], z[i]);
    }
}

template <>
void fused_fma3<float>(const float* x, const float* y, const float* z, float* out, int64_t n) {
    const int64_t blocked = n & ~int64_t{7};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < blocked; i += 8) {
        const __m256 vx = _mm256_loadu_ps(x + i);
        const __m256 vy = _mm256_loadu_ps(y + i);
        const __m256 vz = _mm256_loadu_ps(z + i);
        // The one op that is a SINGLE fused multiply-add per element: one
        // rounding, the documented feature (more accurate than x*y then +z).
        const __m256 r = _mm256_fmadd_ps(vx, vy, vz);
        _mm256_storeu_ps(out + i, r);
    }
    for (int64_t i = blocked; i < n; ++i) {
        out[i] = std::fma(x[i], y[i], z[i]);
    }
}

template <>
void fused_fma3<double>(const double* x, const double* y, const double* z, double* out, int64_t n) {
    const int64_t blocked = n & ~int64_t{3};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < blocked; i += 4) {
        const __m256d vx = _mm256_loadu_pd(x + i);
        const __m256d vy = _mm256_loadu_pd(y + i);
        const __m256d vz = _mm256_loadu_pd(z + i);
        const __m256d r = _mm256_fmadd_pd(vx, vy, vz);
        _mm256_storeu_pd(out + i, r);
    }
    for (int64_t i = blocked; i < n; ++i) {
        out[i] = std::fma(x[i], y[i], z[i]);
    }
}

template <typename T>
void fused_scaled_relu(const T* x, T* out, int64_t n, T scale) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        out[i] = relu_nan_safe_scalar<T>(scale * x[i]);
    }
}

template <>
void fused_scaled_relu<float>(const float* x, float* out, int64_t n, float scale) {
    const __m256 vscale = _mm256_set1_ps(scale);
    const int64_t blocked = n & ~int64_t{7};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < blocked; i += 8) {
        const __m256 v = _mm256_mul_ps(vscale, _mm256_loadu_ps(x + i));
        _mm256_storeu_ps(out + i, relu_nan_safe_ps(v));
    }
    for (int64_t i = blocked; i < n; ++i) {
        out[i] = relu_nan_safe_scalar<float>(scale * x[i]);
    }
}

template <>
void fused_scaled_relu<double>(const double* x, double* out, int64_t n, double scale) {
    const __m256d vscale = _mm256_set1_pd(scale);
    const int64_t blocked = n & ~int64_t{3};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < blocked; i += 4) {
        const __m256d v = _mm256_mul_pd(vscale, _mm256_loadu_pd(x + i));
        _mm256_storeu_pd(out + i, relu_nan_safe_pd(v));
    }
    for (int64_t i = blocked; i < n; ++i) {
        out[i] = relu_nan_safe_scalar<double>(scale * x[i]);
    }
}

template void fused_axpby<float>(const float*, const float*, float*, int64_t, float, float);
template void fused_axpby<double>(const double*, const double*, double*, int64_t, double, double);
template void fused_fma3<float>(const float*, const float*, const float*, float*, int64_t);
template void fused_fma3<double>(const double*, const double*, const double*, double*, int64_t);
template void fused_scaled_relu<float>(const float*, float*, int64_t, float);
template void fused_scaled_relu<double>(const double*, double*, int64_t, double);

} // namespace pg::cpu
