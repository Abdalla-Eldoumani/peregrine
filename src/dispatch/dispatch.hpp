#pragma once
#include <cstdint>

namespace pg::dispatch {

// Every op routes through here, so the binding layer never changes when a
// kernel does.
//
// Host matmul reads A and B through element strides: element (i, p) of A is
// a[i * rs_a + p * cs_a] and (p, j) of B is b[p * rs_b + j * cs_b], any sign or
// zero, so transposed, F-ordered, sliced, and reversed operands need no copy.
// C is C-contiguous with ldc = n and is overwritten. Every path gives the same
// bits for every layout of the same values.
template <typename T>
void matmul(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
            T* c, int64_t m, int64_t k, int64_t n);

extern template void matmul<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t,
                                   float*, int64_t, int64_t, int64_t);
extern template void matmul<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t,
                                    double*, int64_t, int64_t, int64_t);

// C-contiguous operands.
template <typename T>
inline void matmul(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n) {
    matmul<T>(a, k, 1, b, n, 1, c, m, k, n);
}

// The host matmul paths. The choice is a pure function of the shape and the
// CPU features, never of the layout or the thread count, because each path
// rounds differently and a layout- or thread-dependent choice would change the
// bits.
enum class gemm_path { naive, small, skinny, packed };
gemm_path choose_gemm_path(int64_t m, int64_t k, int64_t n, bool avx2_fma) noexcept;

#if defined(PG_HAS_CUDA)
// Device-resident GEMM routing. a, b, c are DEVICE pointers; the binding's
// device matmul(Array, Array) path calls this for operands that are already on
// the device, so it never stages host memory. Separate from the host matmul<T>
// above on purpose: matmul<T> is the CPU decision and stays byte-identical, while
// this is the device-resident decision the wrapper reaches only when BOTH
// operands are a pg.Array.
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

#if defined(PG_HAS_CUDA)
// Device-resident fused routing. x, y, z, out are DEVICE pointers; the binding's
// device fused Array overloads call these for operands already on the device, so
// they never stage host memory. Separate from the host fused_* above on purpose:
// the host entries are the CPU avx2/naive decision and stay byte-identical, while
// these are the device-resident decision the wrapper reaches only when ALL
// operands are a pg.Array. Unlike matmul there is NO f64-never-auto-routes
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

// transpose, sum_all, and sum_axis route through the same entry point, so the
// binding layer never changes when a kernel does. transpose and the sums pick
// their AVX2 bodies on avx2 && fma, else the scalar twins, with the same bits
// either way.
template <typename T>
void transpose(const T* a, T* out, int64_t m, int64_t n);

template <typename T>
T sum_all(const T* a, int64_t m, int64_t n);

template <typename T>
void sum_axis(const T* a, T* out, int64_t m, int64_t n, int axis);

// sum_axis divided by the reduced count converted to T, in T, inside the same
// pass; the bits equal sum_axis followed by that division.
template <typename T>
void mean_axis(const T* a, T* out, int64_t m, int64_t n, int axis);

extern template void transpose<float>(const float*, float*, int64_t, int64_t);
extern template void transpose<double>(const double*, double*, int64_t, int64_t);
extern template float sum_all<float>(const float*, int64_t, int64_t);
extern template double sum_all<double>(const double*, int64_t, int64_t);
extern template void sum_axis<float>(const float*, float*, int64_t, int64_t, int);
extern template void sum_axis<double>(const double*, double*, int64_t, int64_t, int);
extern template void mean_axis<float>(const float*, float*, int64_t, int64_t, int);
extern template void mean_axis<double>(const double*, double*, int64_t, int64_t, int);

} // namespace pg::dispatch
