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

#if defined(FME_HAS_CUDA)
// Device-resident GEMM routing. a, b, c are DEVICE pointers; the binding's
// device matmul(Array, Array) path (04-04) calls this for operands that are
// already on the device, so it never stages host memory. Separate from the host
// matmul<T> above on purpose: matmul<T> is the CPU decision and stays
// byte-identical, while this is the device-resident decision the wrapper reaches
// only when BOTH operands are an fme.Array.
//
// Priority rules (src/dispatch/CLAUDE.md): rule 2 (float64 GEMM never AUTO-routes
// to this GPU -- GA106 FP64 is 1/64 FP32, measured 194 vs 6519 GFLOP/s) is
// enforced UPSTREAM by the wrapper and binding never auto-selecting the device
// path for host float64 arrays; a forced device-resident f64 (both operands
// explicitly to_device'd) is allowed and computed here, slow and docstring-warned.
// Auto-routing a host f64 array to the GPU is the bug rule 2 forbids; a forced
// device f64 is not. Rule 3 (f32 device-resident stays on device) is the f32 arm.
// The decision stays PURE: no warnings, no logging, no fallback -- the wrapper
// owns those.
template <typename T>
void matmul_device(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

extern template void matmul_device<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
extern template void matmul_device<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);
#endif

// Fused elementwise ops over a flat n-element buffer. Unlike transpose/sum_*
// (one plain TU each, direct forward), fused has an AVX2 fast path AND a scalar
// fallback, so these route like matmul: avx2 && fma -> cpu::fused_*<T>, else
// cpu::fused_*_naive<T>. Both dtypes route normally -- there is no
// f64-never-auto-routes exclusion (that is a GEMM-only, GPU-only rule). Scalars
// a/b/scale are in the operand type T.
template <typename T>
void fused_axpby(const T* x, const T* y, T* out, int64_t n, T a, T b);

template <typename T>
void fused_fma3(const T* x, const T* y, const T* z, T* out, int64_t n);

template <typename T>
void fused_scaled_relu(const T* x, T* out, int64_t n, T scale);

extern template void fused_axpby<float>(const float*, const float*, float*, int64_t, float, float);
extern template void fused_axpby<double>(const double*, const double*, double*, int64_t, double, double);
extern template void fused_fma3<float>(const float*, const float*, const float*, float*, int64_t);
extern template void fused_fma3<double>(const double*, const double*, const double*, double*, int64_t);
extern template void fused_scaled_relu<float>(const float*, float*, int64_t, float);
extern template void fused_scaled_relu<double>(const double*, double*, int64_t, double);

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
