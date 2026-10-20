#pragma once
#include <cstdint>

namespace pg::cpu {

// Fused elementwise kernels over a flat C-contiguous buffer of n elements.
// Three ops, two bodies each: an AVX2+FMA fast path (fused_avx2.cpp, joined to
// PG_AVX2_SOURCES) and a plain scalar fallback (fused_naive.cpp). Dispatch
// picks between them on cpu::detect().avx2 && fma, exactly as it chooses
// gemm_blis vs gemm_naive. This header stays intrinsic-free (it includes no AVX2
// intrinsics header) so the no-AVX2 fallback module links it without dragging
// AVX2 codegen into a TU that runs on a CPU that lacks the instructions (the
// legacy import crash).
// Scalars a/b/scale are passed in the operand type T and broadcast once inside
// the kernel; every index is int64_t (a flat element count exceeds int32 well
// before the allocation would fail).

// Threading both twins share, in bytes of out. Threads take whole 64-byte lines
// of out. Single-call diagnostic with out= and forced teams of 1, 2, 3, 6: at
// 16 KiB one thread ties a team (f32 4096 elements axpby 0.89 us alone, 0.89
// with 3) and below it wins (f32 1024 elements 0.39 alone, 0.68 with 2); from
// 32 KiB a team wins (f32 8192 elements 1.72 alone, 1.08 with 6; f64 4096
// elements fma3 2.08 alone, 1.01 with 6).
inline constexpr double FUSED_SERIAL_BELOW_BYTES = 32768.0;
inline constexpr double FUSED_PER_THREAD_BYTES = 8192.0;
// The AVX2 bodies stream (non-temporal stores) once the buffers an op touches,
// inputs and out, exceed the 12 MiB L3: the result cannot stay cached, and a
// streaming store skips the read for ownership. Single-call diagnostic, 6
// threads, out=: f32 2896^2 axpby 5.03 -> 3.25 ms, f64 5793^2 axpby 50.7 -> 31.4
// ms, f64 1024^2 axpby (24 MiB) 0.81 -> 0.56 ms; at or under 12 MiB streaming
// lost, since the buffers stay in L3 between calls (f32 1024^2 axpby at 12 MiB
// 0.125 -> 0.164 ms, scaled_relu at 8 MiB 0.049 -> 0.107 ms).
inline constexpr double FUSED_STREAM_BYTES = 12.0 * 1024 * 1024;
// Past twice the L3 every byte comes from DRAM on every call, and there the
// team is capped at FUSED_DRAM_STREAMS / arrays members, about 12 concurrent
// streams in all: fma3 3, axpby 4, scaled_relu 6 (no change at 6 threads). DRAM
// reads peak at 4 threads and nontemporal copy at 3 on this machine (32.8 and
// 34.1 GB/s, against 27.1 and 23.6 at 6). Paced diagnostic, out=, median us:
// fma3 6 -> 3 threads f32 2896^2 6848 -> 4661, f64 5793^2 48985 -> 39773, f64
// 1024^2 (32 MiB) 1219 -> 905; axpby 6 -> 4 threads f32 2896^2 3900 -> 3452,
// f64 6781 -> 6639; scaled_relu within noise from 2 to 6 threads. fma3 f32
// 1024^2 (16 MiB, partly in L3) still ran faster with 6 (251 against 296).
inline constexpr double FUSED_DRAM_BYTES = 24.0 * 1024 * 1024;
inline constexpr int FUSED_DRAM_STREAMS = 12;
// Streaming groups prefetch their inputs this far ahead into L1. Paced
// diagnostic, out=, median us, without -> with: fma3 at 3 threads f32 2896^2
// 4079 -> 3789, f64 10154 -> 9243 (512 B and 2 KiB in between or equal, T1
// equal, NTA weaker); axpby at 4 threads f32 2896^2 3452 -> 3514, f64 6639 ->
// 6249 (at 6, 3900 -> 3450 and 6781 -> 6088); scaled_relu at 6 threads f32 1820
// -> 1851, f64 3947 -> 3991, so its one input stream skips it.
inline constexpr int64_t FUSED_PREFETCH_BYTES = 1024;
// How far past an input, modulo 4 KiB, out may sit before an ascending loop
// stalls on 4K aliasing, and how far before one before a descending loop does
// (see descend() in fused_avx2.cpp). fma3 f64 1024 elements, one input, one
// thread, L1-resident, ascending: out 128 to 448 bytes past 202-205 ns, 512 or
// more 171-182; descending: out 64 to 512 bytes before 187-225 ns, else 149-170.
inline constexpr std::uintptr_t FUSED_ALIAS_BYTES = 512;
// Below this much out the direction test costs more than the stall it avoids:
// bench loop, out= and fresh, median ns with -> without it, fma3 f32 8x8 222 ->
// 217, 16x16 alloc 414 -> 404, 24x24 alloc 471 -> 461, f64 12x12 alloc 433 ->
// 427, 24x24 out= 333 -> 327; at 32x32 it pays (f32 out= 314 -> 325, f64 alloc
// 769 -> 791). The stall comes mostly from arrays near a whole number of pages,
// which the Windows heap lays 16 bytes past each other modulo 4 KiB.
inline constexpr double FUSED_ALIAS_MIN_BYTES = 4096.0;

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
// (which has no definition, so no other T can be instantiated by accident).
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

} // namespace pg::cpu
