#include "cpu/transpose.hpp"
#include "cpu/threads.hpp"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pg::cpu {

// The AVX2 transpose. The bits are a pure permutation (loads, shuffles, and
// stores never touch a value), so it equals the scalar twin in transpose.cpp for
// every shape, thread count, and store kind.
//
// A tile is 2W rows by W columns of a (W = 8 for f32, 4 for f64): two in-register
// W x W transposes give each of W rows of out one whole 64-byte segment, stored
// back to back, so a streaming store always completes its line before the next
// one starts. Threads own bands of rows of out (columns of a), so each writes one
// contiguous range of out. Inside a band, tiles run along a row block of
// COL_BYTES of a, where the two tiles sharing each line of a run back to back.

namespace {

template <typename T>
struct simd;

template <>
struct simd<float> {
    using reg = __m256;
    static constexpr int64_t W = 8;
    static reg load(const float* p) { return _mm256_loadu_ps(p); }
    static void storeu(float* p, reg v) { _mm256_storeu_ps(p, v); }
    static void stream(float* p, reg v) { _mm256_stream_ps(p, v); }
    // r[k] holds row k of an 8 x 8 block; on return r[k] holds column k.
    static void transpose(reg* r) {
        const reg t0 = _mm256_unpacklo_ps(r[0], r[1]);
        const reg t1 = _mm256_unpackhi_ps(r[0], r[1]);
        const reg t2 = _mm256_unpacklo_ps(r[2], r[3]);
        const reg t3 = _mm256_unpackhi_ps(r[2], r[3]);
        const reg t4 = _mm256_unpacklo_ps(r[4], r[5]);
        const reg t5 = _mm256_unpackhi_ps(r[4], r[5]);
        const reg t6 = _mm256_unpacklo_ps(r[6], r[7]);
        const reg t7 = _mm256_unpackhi_ps(r[6], r[7]);
        const reg s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
        const reg s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
        const reg s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
        const reg s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
        const reg s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
        const reg s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
        const reg s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
        const reg s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
        r[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
        r[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
        r[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
        r[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
        r[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
        r[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
        r[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
        r[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
    }
};

template <>
struct simd<double> {
    using reg = __m256d;
    static constexpr int64_t W = 4;
    static reg load(const double* p) { return _mm256_loadu_pd(p); }
    static void storeu(double* p, reg v) { _mm256_storeu_pd(p, v); }
    static void stream(double* p, reg v) { _mm256_stream_pd(p, v); }
    static void transpose(reg* r) {
        const reg t0 = _mm256_unpacklo_pd(r[0], r[1]);
        const reg t1 = _mm256_unpackhi_pd(r[0], r[1]);
        const reg t2 = _mm256_unpacklo_pd(r[2], r[3]);
        const reg t3 = _mm256_unpackhi_pd(r[2], r[3]);
        r[0] = _mm256_permute2f128_pd(t0, t2, 0x20);
        r[1] = _mm256_permute2f128_pd(t1, t3, 0x20);
        r[2] = _mm256_permute2f128_pd(t0, t2, 0x31);
        r[3] = _mm256_permute2f128_pd(t1, t3, 0x31);
    }
};

// The tile at rows [i, i + 2W) and columns [j, j + W) of a.
template <bool Stream, typename T>
void tile(const T* a, T* out, int64_t m, int64_t n, int64_t i, int64_t j) {
    using S = simd<T>;
    constexpr int64_t W = S::W;
    typename S::reg lo[W];
    typename S::reg hi[W];
    for (int64_t k = 0; k < W; ++k) {
        lo[k] = S::load(a + (i + k) * n + j);
        hi[k] = S::load(a + (i + W + k) * n + j);
    }
    S::transpose(lo);
    S::transpose(hi);
    for (int64_t k = 0; k < W; ++k) {
        T* o = out + (j + k) * m + i;
        if constexpr (Stream) {
            S::stream(o, lo[k]);
            S::stream(o + W, hi[k]);
        } else {
            S::storeu(o, lo[k]);
            S::storeu(o + W, hi[k]);
        }
    }
}

// Rows [i0, i1) by columns [j0, j1) of a, one element at a time.
template <typename T>
void scalar_block(const T* a, T* out, int64_t m, int64_t n, int64_t i0, int64_t i1, int64_t j0, int64_t j1) {
    for (int64_t i = i0; i < i1; ++i) {
        for (int64_t j = j0; j < j1; ++j) {
            out[j * m + i] = a[i * n + j];
        }
    }
}

// Rows [j0, j1) of out, that is columns [j0, j1) of a, every row of a.
template <bool Stream, typename T>
void band(const T* a, T* out, int64_t m, int64_t n, int64_t j0, int64_t j1) {
    constexpr int64_t W = simd<T>::W;
    constexpr int64_t COLS = TRANSPOSE_COL_BYTES / static_cast<int64_t>(sizeof(T));
    const int64_t m_t = m - m % (2 * W);
    const int64_t n_t = n - n % W;
    const int64_t jt = j1 < n_t ? j1 : n_t;
    for (int64_t jb = j0; jb < jt; jb += COLS) {
        const int64_t je = jb + COLS < jt ? jb + COLS : jt;
        for (int64_t i = 0; i < m_t; i += 2 * W) {
            for (int64_t j = jb; j < je; j += W) {
                tile<Stream>(a, out, m, n, i, j);
            }
        }
    }
    scalar_block(a, out, m, n, m_t, m, j0, jt);
    if (j1 > jt) {
        scalar_block(a, out, m, n, int64_t{0}, m, jt, j1);
    }
}

} // namespace

template <typename T>
void transpose_avx2(const T* a, T* out, int64_t m, int64_t n) {
    if (m <= 0 || n <= 0) {
        return;
    }
    // A single row or column keeps its element order.
    if (m == 1 || n == 1) {
        std::memcpy(out, a, static_cast<std::size_t>(m * n) * sizeof(T));
        return;
    }
    constexpr int64_t W = simd<T>::W;
    const double bytes = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(sizeof(T));
    const int64_t units = (n + W - 1) / W;
    const int nt = team_size(bytes, TRANSPOSE_SERIAL_BELOW_BYTES, TRANSPOSE_PER_THREAD_BYTES, units);
    // Streaming needs every tile row of out on a 64-byte line boundary, which
    // holds when out is 64-byte aligned and a row of out is a whole number of
    // lines. shortcut: other row lengths keep regular stores; peel each row to
    // its first line boundary if they turn out to matter.
    const bool stream = bytes > TRANSPOSE_STREAM_BYTES && reinterpret_cast<std::uintptr_t>(out) % 64 == 0 &&
                        (m * static_cast<int64_t>(sizeof(T))) % 64 == 0;
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
        const int64_t j0 = W * (units * t / team);
        const int64_t j1e = W * (units * (t + 1) / team);
        const int64_t j1 = j1e < n ? j1e : n;
        if (stream) {
            band<true>(a, out, m, n, j0, j1);
            // Streaming stores are weakly ordered; fence them before the
            // region's closing barrier hands the result to the caller.
            _mm_sfence();
        } else {
            band<false>(a, out, m, n, j0, j1);
        }
    }
}

template void transpose_avx2<float>(const float*, float*, int64_t, int64_t);
template void transpose_avx2<double>(const double*, double*, int64_t, int64_t);

} // namespace pg::cpu
