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
// device matmul(Array, Array) path calls this for operands that are already on
// the device, so it never stages host memory. Separate from the host matmul<T>
// above on purpose: matmul<T> is the CPU decision and stays byte-identical, while
// this is the device-resident decision the wrapper reaches only when BOTH
// operands are an fme.Array.
//
// Routing rules: float64 GEMM never AUTO-routes to this GPU (GA106 FP64 is 1/64
// FP32, measured 194 vs 6519 GFLOP/s). That exclusion is enforced UPSTREAM by the
// wrapper and binding never auto-selecting the device path for host float64
// arrays; a forced device-resident f64 (both operands explicitly to_device'd) is
// allowed and computed here, slow and docstring-warned. Auto-routing a host f64
// array to the GPU is the bug; a forced device f64 is not. An f32 device-resident
// pair stays on the device (the f32 arm). The decision stays PURE: no warnings, no
// logging, no fallback -- the wrapper owns those.
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

#if defined(FME_HAS_CUDA)
// Device-resident fused routing. x, y, z, out are DEVICE pointers; the binding's
// device fused Array overloads call these for operands already on the device, so
// they never stage host memory. Separate from the host fused_* above on purpose:
// the host entries are the CPU avx2/naive decision and stay byte-identical, while
// these are the device-resident decision the wrapper reaches only when ALL
// operands are an fme.Array. Unlike matmul there is NO f64-never-auto-routes
// exclusion here: the fused device path is reached only by an explicit
// to_device, so both dtypes route through (the wrapper never auto-stages a host
// fused array to the GPU). Pure device-in/out forward to cuda::fused_*<T>.
template <typename T>
void fused_axpby_device(const T* x, const T* y, T* out, int64_t n, T a, T b);

template <typename T>
void fused_fma3_device(const T* x, const T* y, const T* z, T* out, int64_t n);

template <typename T>
void fused_scaled_relu_device(const T* x, T* out, int64_t n, T scale);

extern template void fused_axpby_device<float>(const float*, const float*, float*, int64_t, float, float);
extern template void fused_axpby_device<double>(const double*, const double*, double*, int64_t, double, double);
extern template void fused_fma3_device<float>(const float*, const float*, const float*, float*, int64_t);
extern template void fused_fma3_device<double>(const double*, const double*, const double*, double*, int64_t);
extern template void fused_scaled_relu_device<float>(const float*, float*, int64_t, float);
extern template void fused_scaled_relu_device<double>(const double*, double*, int64_t, double);
#endif

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
