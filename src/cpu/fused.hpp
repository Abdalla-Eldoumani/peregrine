#pragma once
#include <cstdint>

namespace fme::cpu {

// Fused elementwise kernels over a flat C-contiguous buffer of n elements.
// Three ops, two bodies each: an AVX2+FMA fast path (fused_avx2.cpp, joined to
// FME_AVX2_SOURCES) and a plain scalar fallback (fused_naive.cpp). Dispatch
// picks between them on cpu::detect().avx2 && fma, exactly as it chooses
// gemm_blis vs gemm_naive. This header stays intrinsic-free (it includes no AVX2
// intrinsics header) so the no-AVX2 fallback module links it without dragging
// AVX2 codegen into a TU that runs on a CPU that lacks the instructions (the
// legacy import crash).
// Scalars a/b/scale are passed in the operand type T and broadcast once inside
// the kernel; every index is int64_t (a flat element count exceeds int32 well
// before the allocation would fail).

// out[i] = a*x[i] + b*y[i]. Two roundings (a*x, then +b*y), matching the
// unfused NumPy expression within the elementwise tolerance; deliberately NOT
// contracted into one rounding.
template <typename T>
void fused_axpby(const T* x, const T* y, T* out, int64_t n, T a, T b);

// out[i] = fma(x[i], y[i], z[i]). A true single-rounding FMA: more accurate
// than the unfused x*y then +z (two roundings), which the oracle allows.
// inf*0+z yields NaN like NumPy.
template <typename T>
void fused_fma3(const T* x, const T* y, const T* z, T* out, int64_t n);

// out[i] = maximum(scale*x[i], 0). A NaN input (scale*x NaN) PROPAGATES as NaN
// to match np.maximum; it does NOT collapse to 0 (the SIMD maxps/maxpd trap).
template <typename T>
void fused_scaled_relu(const T* x, T* out, int64_t n, T scale);

// float and double of the three ops above are EXPLICIT SPECIALIZATIONS (the AVX2
// bodies in fused_avx2.cpp), not implicit instantiations of the primary template
// (whose body is the generic scalar fallback that only the AVX2 TU's tail reuses).
// These specialization declarations MUST precede the extern-template instantiation
// declarations below: an explicit specialization has to be declared before the
// first use that would otherwise instantiate the primary template
// ([temp.expl.spec]/6). GCC enforces this and rejects the specialization
// definitions in fused_avx2.cpp as "specialization after instantiation" without
// these; MSVC accepts the out-of-order form silently, so the WSL/GCC build is the
// only place the omission surfaces. The _naive twins are plain templates (separate
// names), so they need no specialization declaration.
template <>
void fused_axpby<float>(const float*, const float*, float*, int64_t, float, float);
template <>
void fused_axpby<double>(const double*, const double*, double*, int64_t, double, double);
template <>
void fused_fma3<float>(const float*, const float*, const float*, float*, int64_t);
template <>
void fused_fma3<double>(const double*, const double*, const double*, double*, int64_t);
template <>
void fused_scaled_relu<float>(const float*, float*, int64_t, float);
template <>
void fused_scaled_relu<double>(const double*, double*, int64_t, double);

// The _naive twins: identical contract, scalar bodies in a plain (non-AVX2) TU.
template <typename T>
void fused_axpby_naive(const T* x, const T* y, T* out, int64_t n, T a, T b);

template <typename T>
void fused_fma3_naive(const T* x, const T* y, const T* z, T* out, int64_t n);

template <typename T>
void fused_scaled_relu_naive(const T* x, T* out, int64_t n, T scale);

extern template void fused_axpby<float>(const float*, const float*, float*, int64_t, float, float);
extern template void fused_axpby<double>(const double*, const double*, double*, int64_t, double, double);
extern template void fused_fma3<float>(const float*, const float*, const float*, float*, int64_t);
extern template void fused_fma3<double>(const double*, const double*, const double*, double*, int64_t);
extern template void fused_scaled_relu<float>(const float*, float*, int64_t, float);
extern template void fused_scaled_relu<double>(const double*, double*, int64_t, double);

extern template void fused_axpby_naive<float>(const float*, const float*, float*, int64_t, float, float);
extern template void fused_axpby_naive<double>(const double*, const double*, double*, int64_t, double, double);
extern template void fused_fma3_naive<float>(const float*, const float*, const float*, float*, int64_t);
extern template void fused_fma3_naive<double>(const double*, const double*, const double*, double*, int64_t);
extern template void fused_scaled_relu_naive<float>(const float*, float*, int64_t, float);
extern template void fused_scaled_relu_naive<double>(const double*, double*, int64_t, double);

} // namespace fme::cpu
