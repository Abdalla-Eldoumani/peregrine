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
//
// Each W x W transpose loads half rows into 128-bit lanes (rows k and k + W/2 of
// the block share a register), so the lane-crossing permute stage disappears:
// one stage on the single shuffle port for f64 (was two) and two for f32 (was
// three). Named locals throughout: MSVC kept the loop-indexed __m256 arrays of
// the previous tile on the stack. Median pg.transpose call, 6 threads, before ->
// after: f32 32x32 480 -> 425 ns, f32 64x64 1417 -> 1106 ns.

namespace {

template <bool Stream>
void put(float* p, __m256 v) {
    if constexpr (Stream) {
        _mm256_stream_ps(p, v);
    } else {
        _mm256_storeu_ps(p, v);
    }
}

template <bool Stream>
void put(double* p, __m256d v) {
    if constexpr (Stream) {
        _mm256_stream_pd(p, v);
    } else {
        _mm256_storeu_pd(p, v);
    }
}

// Four floats from p in the low lane and four from q in the high lane.
__m256 lanes(const float* p, const float* q) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(p)), _mm_loadu_ps(q), 1);
}

__m256d lanes(const double* p, const double* q) {
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(p)), _mm_loadu_pd(q), 1);
}

// f32: columns [0, 4) of the 8 rows at p (row stride n) to rows [0, 4) of out at
// o (row stride m); with Two, also rows [8, 16) of p to the next 8 elements of
// those rows of out, each row of out getting its whole 64 bytes back to back.
template <bool Stream, bool Two>
void quad(const float* p, int64_t n, float* o, int64_t m) {
    __m256 r0 = lanes(p, p + 4 * n);
    __m256 r1 = lanes(p + n, p + 5 * n);
    __m256 r2 = lanes(p + 2 * n, p + 6 * n);
    __m256 r3 = lanes(p + 3 * n, p + 7 * n);
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 c0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 c1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 c2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 c3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    if constexpr (Two) {
        const float* q = p + 8 * n;
        r0 = lanes(q, q + 4 * n);
        r1 = lanes(q + n, q + 5 * n);
        r2 = lanes(q + 2 * n, q + 6 * n);
        r3 = lanes(q + 3 * n, q + 7 * n);
        t0 = _mm256_unpacklo_ps(r0, r1);
        t1 = _mm256_unpackhi_ps(r0, r1);
        t2 = _mm256_unpacklo_ps(r2, r3);
        t3 = _mm256_unpackhi_ps(r2, r3);
        put<Stream>(o, c0);
        put<Stream>(o + 8, _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)));
        put<Stream>(o + m, c1);
        put<Stream>(o + m + 8, _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2)));
        put<Stream>(o + 2 * m, c2);
        put<Stream>(o + 2 * m + 8, _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)));
        put<Stream>(o + 3 * m, c3);
        put<Stream>(o + 3 * m + 8, _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2)));
    } else {
        put<Stream>(o, c0);
        put<Stream>(o + m, c1);
        put<Stream>(o + 2 * m, c2);
        put<Stream>(o + 3 * m, c3);
    }
}

// f64: columns [0, 2) of the 4 rows at p to rows [0, 2) of out; with Two, also
// rows [4, 8) of p to the next 4 elements of those rows.
template <bool Stream, bool Two>
void quad(const double* p, int64_t n, double* o, int64_t m) {
    __m256d r0 = lanes(p, p + 2 * n);
    __m256d r1 = lanes(p + n, p + 3 * n);
    const __m256d c0 = _mm256_unpacklo_pd(r0, r1);
    const __m256d c1 = _mm256_unpackhi_pd(r0, r1);
    if constexpr (Two) {
        const double* q = p + 4 * n;
        r0 = lanes(q, q + 2 * n);
        r1 = lanes(q + n, q + 3 * n);
        put<Stream>(o, c0);
        put<Stream>(o + 4, _mm256_unpacklo_pd(r0, r1));
        put<Stream>(o + m, c1);
        put<Stream>(o + m + 4, _mm256_unpackhi_pd(r0, r1));
    } else {
        put<Stream>(o, c0);
        put<Stream>(o + m, c1);
    }
}

template <typename T>
constexpr int64_t width() {
    return 32 / static_cast<int64_t>(sizeof(T));
}

// The tile at rows [i, i + 2W) (Two) or [i, i + W) and columns [j, j + W) of a.
// The W-row form serves a row remainder of W or more with regular stores, since
// each row of out then gets half a line.
template <bool Stream, bool Two, typename T>
void tile(const T* a, T* out, int64_t m, int64_t n, int64_t i, int64_t j) {
    constexpr int64_t H = width<T>() / 2;
    const T* p = a + i * n + j;
    T* o = out + j * m + i;
    quad<Stream, Two>(p, n, o, m);
    quad<Stream, Two>(p + H, n, o + H * m, m);
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

// Writes a zero to the first element of each 4 KiB page of [lo, hi), so a fresh
// output faults its pages in address order before the tiles write them (see
// TRANSPOSE_PREFAULT_BYTES). Volatile keeps the stores the tiles overwrite.
template <typename T>
void prefault(T* lo, T* hi) {
    if (lo >= hi) {
        return;
    }
    *static_cast<volatile T*>(lo) = T(0);
    const std::uintptr_t first = (reinterpret_cast<std::uintptr_t>(lo) | 4095) + 1;
    for (std::uintptr_t q = first; q < reinterpret_cast<std::uintptr_t>(hi); q += 4096) {
        *reinterpret_cast<volatile T*>(q) = T(0);
    }
}

// Rows [j0, j1) of out, that is columns [j0, j1) of a, every row of a.
template <bool Stream, typename T>
void band(const T* a, T* out, int64_t m, int64_t n, int64_t j0, int64_t j1, bool fresh) {
    constexpr int64_t W = width<T>();
    constexpr int64_t COLS = TRANSPOSE_COL_BYTES / static_cast<int64_t>(sizeof(T));
    const int64_t m_t = m - m % (2 * W);
    const int64_t m_h = m - m_t >= W ? m_t + W : m_t;
    const int64_t n_t = n - n % W;
    const int64_t jt = j1 < n_t ? j1 : n_t;
    for (int64_t jb = j0; jb < jt; jb += COLS) {
        const int64_t je = jb + COLS < jt ? jb + COLS : jt;
        if (fresh) {
            prefault(out + jb * m, out + je * m);
        }
        for (int64_t i = 0; i < m_t; i += 2 * W) {
            for (int64_t j = jb; j < je; j += W) {
                tile<Stream, true>(a, out, m, n, i, j);
            }
        }
        if (m_h > m_t) {
            for (int64_t j = jb; j < je; j += W) {
                tile<false, false>(a, out, m, n, m_t, j);
            }
        }
    }
    scalar_block(a, out, m, n, m_h, m, j0, jt);
    if (j1 > jt) {
        scalar_block(a, out, m, n, int64_t{0}, m, jt, j1);
    }
}

// Streaming stores are weakly ordered; the fence orders them before the
// caller, or the region's closing barrier, hands the result on.
template <typename T>
void run_band(bool stream, bool fresh, const T* a, T* out, int64_t m, int64_t n, int64_t j0, int64_t j1) {
    if (stream) {
        band<true>(a, out, m, n, j0, j1, fresh);
        _mm_sfence();
    } else {
        band<false>(a, out, m, n, j0, j1, fresh);
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
    constexpr int64_t W = width<T>();
    const double bytes = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(sizeof(T));
    const int64_t units = (n + W - 1) / W;
    const int nt = team_size(bytes, TRANSPOSE_SERIAL_BELOW_BYTES, TRANSPOSE_PER_THREAD_BYTES, units);
    // Streaming needs every tile row of out on a 64-byte line boundary, which
    // holds when out is 64-byte aligned and a row of out is a whole number of
    // lines. shortcut: other row lengths keep regular stores; peel each row to
    // its first line boundary if they turn out to matter.
    const bool stream = bytes > TRANSPOSE_STREAM_BYTES && reinterpret_cast<std::uintptr_t>(out) % 64 == 0 &&
                        (m * static_cast<int64_t>(sizeof(T))) % 64 == 0;
    const bool fresh = bytes >= TRANSPOSE_PREFAULT_BYTES;
    // A permutation never consults MXCSR (moves and shuffles ignore FTZ, DAZ,
    // and rounding), so no mxcsr_scope; and a team of one skips the OpenMP
    // runtime, whose if(false) region still costs a call into vcomp. With the
    // 8 x 8 f32 tile below, median pg.transpose 8x8: f32 334 -> 241 ns, f64 320
    // -> 243 ns.
    if (nt == 1) {
        run_band(stream, fresh, a, out, m, n, int64_t{0}, n);
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
    {
        const int64_t t = omp_get_thread_num();
        const int64_t team = omp_get_num_threads();
        const int64_t j0 = W * (units * t / team);
        const int64_t j1e = W * (units * (t + 1) / team);
        run_band(stream, fresh, a, out, m, n, j0, j1e < n ? j1e : n);
    }
#endif
}

template void transpose_avx2<float>(const float*, float*, int64_t, int64_t);
template void transpose_avx2<double>(const double*, double*, int64_t, int64_t);

} // namespace pg::cpu
