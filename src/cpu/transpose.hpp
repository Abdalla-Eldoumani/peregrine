#pragma once
#include <cstdint>

namespace pg::cpu {

// Out-of-place 2-D transpose: writes the n-by-m transpose of an m-by-n
// row-major array into out. Two bodies with one contract: transpose_avx2
// (transpose_avx2.cpp, in PG_AVX2_SOURCES: in-register tiles, threaded,
// streaming stores for large outputs) and transpose (transpose.cpp, scalar, the
// no-AVX2 fallback). The result is a pure permutation of the input's bits, so
// the two are bitwise equal everywhere. This header stays intrinsic-free.
template <typename T>
void transpose(const T* a, T* out, int64_t m, int64_t n);

template <typename T>
void transpose_avx2(const T* a, T* out, int64_t m, int64_t n);

// Threading for transpose_avx2, in bytes of out; threads take bands of W rows
// of out. Single-call diagnostic, forced teams of 1, 2, 3, 6: f32 64x64 (16
// KiB) 2.09 us alone, 2.05 with 3; f64 64x64 (32 KiB) 2.42 alone, 1.28 with 6;
// f32 128x128 8.2 alone, 2.8 with 3.
inline constexpr double TRANSPOSE_SERIAL_BELOW_BYTES = 32768.0;
inline constexpr double TRANSPOSE_PER_THREAD_BYTES = 8192.0;
// Streaming stores once out exceeds the 12 MiB L3. Single-call diagnostic, 6
// threads, GB/s of input plus output: f32 2896^2 8.2 -> 9.1, f64 2896^2 8.6 ->
// 9.8, f64 4096^2 8.4 -> 9.3.
inline constexpr double TRANSPOSE_STREAM_BYTES = 12.0 * 1024 * 1024;
// The width of a's row block a band walks down. 256 to 2048 bytes measured
// within noise of each other (f32 4096^2, 6 threads: 15.8 to 17.2 ms).
inline constexpr int64_t TRANSPOSE_COL_BYTES = 512;

extern template void transpose<float>(const float*, float*, int64_t, int64_t);
extern template void transpose<double>(const double*, double*, int64_t, int64_t);
extern template void transpose_avx2<float>(const float*, float*, int64_t, int64_t);
extern template void transpose_avx2<double>(const double*, double*, int64_t, int64_t);

} // namespace pg::cpu
