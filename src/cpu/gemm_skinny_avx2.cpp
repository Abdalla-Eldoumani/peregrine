#include "cpu/gemm_skinny.hpp"

#include "cpu/gemm_small.hpp"
#include "cpu/threads.hpp"

#include <immintrin.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pg::cpu {

namespace {

// C rows [i0, i1) when n <= 4 < m and A's columns are unit-stride. Each block
// solves C^T = B^T A^T, so the vectors run down A's columns in place, into a
// scratch tile whose rows are C's columns, and copies it back. The tile kernel
// computes the same chain per element either way round, so the bits do not
// change. A row-major A stays on the direct path, which broadcasts its rows:
// gathering its columns cost more (single-call diagnostic, f64 (4096, 4096, 1):
// 22.9 ms transposed against 8.2 ms direct).
template <typename T>
void transposed_rows(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c,
                     int64_t i0, int64_t i1, int64_t k, int64_t n) {
    constexpr int64_t IB = 256;
    alignas(32) T ct[4 * IB];
    for (int64_t i = i0; i < i1; i += IB) {
        const int64_t ib = i1 - i < IB ? i1 - i : IB;
        gemm_small_ldc<T>(b, cs_b, rs_b, a + i * rs_a, cs_a, rs_a, ct, IB, n, k, ib);
        for (int64_t r = 0; r < ib; ++r) {
            for (int64_t j = 0; j < n; ++j) {
                c[(i + r) * n + j] = ct[j * IB + r];
            }
        }
    }
}

// The two streaming kernels below read a long operand that lives in DRAM. The
// hardware prefetchers follow each row of it as its own stream, and many
// streams at once cost DRAM bandwidth: at (4096, 4096, 1) float32, six threads
// each walking 4, 8, or 16 rows at a time reached about 36, 30-33, and 25-28
// GB/s. So each kernel walks few rows, and a software read-ahead pulls the rows
// it needs next into L2 one line at a time in address order, one line for
// every line it consumes, which leaves DRAM a single sequential stream per
// thread (float32 (4096, 4096, 1) 2.0 -> 1.7 ms, float64 (4, 4096, 4096) 6.3
// -> 4.2 ms). Prefetching into L1 instead was no faster and noisier,
// non-temporal was slower (2.7 ms), and reading ahead at twice the rate
// consumed gained nothing.
struct read_ahead {
    const char* row;
    int64_t ld;   // bytes from one row to the next
    int64_t len;  // bytes to fetch from each row
    int64_t rows; // rows still to fetch
    int64_t off;

    void step(int64_t lines) {
        for (; lines > 0 && rows > 0; --lines) {
            _mm_prefetch(row + off, _MM_HINT_T1);
            off += 64;
            if (off >= len) {
                off = 0;
                row += ld;
                --rows;
            }
        }
    }
};

template <typename T>
read_ahead rows_ahead(const T* first, int64_t ld, int64_t len, int64_t rows) {
    return read_ahead{reinterpret_cast<const char*>(first), ld * static_cast<int64_t>(sizeof(T)),
                      len * static_cast<int64_t>(sizeof(T)), rows, 0};
}

// Four rows of A in the lanes of one vector (xmm for float32, ymm for float64),
// so lane r carries row r's chain and each step is one FMA for four elements.
// quad runs k steps p .. p + 3: it loads four consecutive values of each row
// and transposes them in registers into the four steps' columns, applied in k
// order.
template <typename T>
struct four;

template <>
struct four<float> {
    using reg = __m128;
    static reg zero() { return _mm_setzero_ps(); }
    static reg bcast(const float* p) { return _mm_broadcast_ss(p); }
    static reg column(const float* q, int64_t ld) { return _mm_setr_ps(q[0], q[ld], q[2 * ld], q[3 * ld]); }
    static reg fmadd(reg x, reg y, reg acc) { return _mm_fmadd_ps(x, y, acc); }
    static reg quad(reg acc, const float* q, int64_t ld, const float* x, int64_t xs) {
        const reg t0 = _mm_loadu_ps(q);
        const reg t1 = _mm_loadu_ps(q + ld);
        const reg t2 = _mm_loadu_ps(q + 2 * ld);
        const reg t3 = _mm_loadu_ps(q + 3 * ld);
        const reg u0 = _mm_unpacklo_ps(t0, t1);
        const reg u1 = _mm_unpackhi_ps(t0, t1);
        const reg u2 = _mm_unpacklo_ps(t2, t3);
        const reg u3 = _mm_unpackhi_ps(t2, t3);
        acc = _mm_fmadd_ps(_mm_movelh_ps(u0, u2), bcast(x), acc);
        acc = _mm_fmadd_ps(_mm_movehl_ps(u2, u0), bcast(x + xs), acc);
        acc = _mm_fmadd_ps(_mm_movelh_ps(u1, u3), bcast(x + 2 * xs), acc);
        return _mm_fmadd_ps(_mm_movehl_ps(u3, u1), bcast(x + 3 * xs), acc);
    }
    // v + 0.0 written v - (-0.0): MSVC folds an intrinsic add of +0.0 away.
    static void put(float* y, reg v) { _mm_storeu_ps(y, _mm_sub_ps(v, _mm_set1_ps(-0.0f))); }
};

template <>
struct four<double> {
    using reg = __m256d;
    static reg zero() { return _mm256_setzero_pd(); }
    static reg bcast(const double* p) { return _mm256_broadcast_sd(p); }
    static reg column(const double* q, int64_t ld) { return _mm256_setr_pd(q[0], q[ld], q[2 * ld], q[3 * ld]); }
    static reg fmadd(reg x, reg y, reg acc) { return _mm256_fmadd_pd(x, y, acc); }
    // Rows 0 and 2 share a register's halves, as do rows 1 and 3, so one unpack
    // per step yields a column.
    static reg pair(const double* lo, const double* hi) {
        return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(lo)), _mm_loadu_pd(hi), 1);
    }
    static reg quad(reg acc, const double* q, int64_t ld, const double* x, int64_t xs) {
        const reg t0 = pair(q, q + 2 * ld);
        const reg t1 = pair(q + ld, q + 3 * ld);
        const reg t2 = pair(q + 2, q + 2 * ld + 2);
        const reg t3 = pair(q + ld + 2, q + 3 * ld + 2);
        acc = _mm256_fmadd_pd(_mm256_unpacklo_pd(t0, t1), bcast(x), acc);
        acc = _mm256_fmadd_pd(_mm256_unpackhi_pd(t0, t1), bcast(x + xs), acc);
        acc = _mm256_fmadd_pd(_mm256_unpacklo_pd(t2, t3), bcast(x + 2 * xs), acc);
        return _mm256_fmadd_pd(_mm256_unpackhi_pd(t2, t3), bcast(x + 3 * xs), acc);
    }
    static void put(double* y, reg v) { _mm256_storeu_pd(y, _mm256_sub_pd(v, _mm256_set1_pd(-0.0))); }
};

// C = A x for a row-major A (cs_a == 1) and n == 1, rows of C at y[0 .. rows).
// Four rows at a time keeps the streams few, and with one FMA per k step in
// each chain they still run one element per cycle, about twice what a thread's
// share of DRAM delivers. The old six-row broadcast tile used one lane of eight
// and reached 20 GB/s here.
template <typename T>
void matvec(const T* a, int64_t rs_a, const T* x, int64_t xs, T* y, int64_t rows, int64_t k) {
    using F = four<T>;
    constexpr int64_t LINES = 16 * static_cast<int64_t>(sizeof(T)) / 64; // a quad's bytes, in lines
    int64_t i = 0;
    for (; i + 4 <= rows; i += 4) {
        const T* r = a + i * rs_a;
        // the next group's address is formed only when it exists
        read_ahead next = i + 8 <= rows ? rows_ahead(r + 4 * rs_a, rs_a, k, 4) : rows_ahead(r, rs_a, k, 0);
        typename F::reg acc = F::zero();
        int64_t p = 0;
        for (; p + 4 <= k; p += 4) {
            next.step(LINES);
            acc = F::quad(acc, r + p, rs_a, x + p * xs, xs);
        }
        for (; p < k; ++p) {
            acc = F::fmadd(F::column(r + p, rs_a), F::bcast(x + p * xs), acc);
        }
        F::put(y + i, acc);
    }
    if (i < rows) {
        gemm_small_ldc<T>(a + i * rs_a, rs_a, 1, x, xs, 1, y + i, 1, rows - i, k, 1);
    }
}

template <typename T>
struct vec;

template <>
struct vec<double> {
    using reg = __m256d;
    static constexpr int64_t W = 4;
    static reg zero() { return _mm256_setzero_pd(); }
    static reg load(const double* p) { return _mm256_loadu_pd(p); }
    static reg bcast(const double* p) { return _mm256_broadcast_sd(p); }
    static reg fmadd(reg x, reg y, reg acc) { return _mm256_fmadd_pd(x, y, acc); }
    static reg plus_zero(reg v) { return _mm256_sub_pd(v, _mm256_set1_pd(-0.0)); }
    static void store(double* p, reg v) { _mm256_storeu_pd(p, v); }
    // VMASKMOV touches no memory in masked-off lanes, so the edge of a row never faults.
    static __m256i mask(int64_t live) {
        return _mm256_cmpgt_epi64(_mm256_set1_epi64x(live), _mm256_setr_epi64x(0, 1, 2, 3));
    }
    static reg load_part(const double* p, __m256i m) { return _mm256_maskload_pd(p, m); }
    static void store_part(double* p, __m256i m, reg v) { _mm256_maskstore_pd(p, m, v); }
};

template <>
struct vec<float> {
    using reg = __m256;
    static constexpr int64_t W = 8;
    static reg zero() { return _mm256_setzero_ps(); }
    static reg load(const float* p) { return _mm256_loadu_ps(p); }
    static reg bcast(const float* p) { return _mm256_broadcast_ss(p); }
    static reg fmadd(reg x, reg y, reg acc) { return _mm256_fmadd_ps(x, y, acc); }
    static reg plus_zero(reg v) { return _mm256_sub_ps(v, _mm256_set1_ps(-0.0f)); }
    static void store(float* p, reg v) { _mm256_storeu_ps(p, v); }
    static __m256i mask(int64_t live) {
        return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(live)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    }
    static reg load_part(const float* p, __m256i m) { return _mm256_maskload_ps(p, m); }
    static void store_part(float* p, __m256i m, reg v) { _mm256_maskstore_ps(p, m, v); }
};

// The k block of the panel kernel. Each block of B is read twice from L2 (by
// the read-ahead, then by the tiles) while C makes one round trip per block:
// (4, 4096, 4096) at 8, 16, 32 rows took float64 4.3, 4.2-4.3, 4.5-5.1 ms and
// float32 2.0-2.1, 2.2-2.3, 2.4-2.5 ms.
constexpr int64_t PANEL_KB = 8;

// M rows of C by two vectors of columns (one 64-byte line of a B row) over kb
// steps of k, starting from +0.0 on the first block and from C after it.
template <typename T, int M, bool EDGE>
void panel_tile(const T* a, int64_t rs_a, int64_t cs_a, const T* bp, int64_t rs_b, T* c, int64_t ldc, int64_t kb,
                int64_t live, bool first, bool last) {
    using S = vec<T>;
    const __m256i m0 = S::mask(live);
    const __m256i m1 = S::mask(live - S::W);
    typename S::reg c00 = S::zero(), c01 = S::zero(), c10 = S::zero(), c11 = S::zero();
    typename S::reg c20 = S::zero(), c21 = S::zero(), c30 = S::zero(), c31 = S::zero();
#define PG_LOAD(r)                                                                                           \
    if constexpr (r < M) {                                                                                   \
        if (!first) {                                                                                        \
            c##r##0 = EDGE ? S::load_part(c + r * ldc, m0) : S::load(c + r * ldc);                           \
            c##r##1 = EDGE ? S::load_part(c + r * ldc + S::W, m1) : S::load(c + r * ldc + S::W);             \
        }                                                                                                    \
    }
#define PG_ROW(r)                                                                                            \
    if constexpr (r < M) {                                                                                   \
        av = S::bcast(ap + r * rs_a);                                                                        \
        c##r##0 = S::fmadd(av, b0, c##r##0);                                                                 \
        c##r##1 = S::fmadd(av, b1, c##r##1);                                                                 \
    }
#define PG_STORE(r)                                                                                          \
    if constexpr (r < M) {                                                                                   \
        if (last) {                                                                                          \
            c##r##0 = S::plus_zero(c##r##0);                                                                 \
            c##r##1 = S::plus_zero(c##r##1);                                                                 \
        }                                                                                                    \
        if constexpr (EDGE) {                                                                                \
            S::store_part(c + r * ldc, m0, c##r##0);                                                         \
            S::store_part(c + r * ldc + S::W, m1, c##r##1);                                                  \
        } else {                                                                                             \
            S::store(c + r * ldc, c##r##0);                                                                  \
            S::store(c + r * ldc + S::W, c##r##1);                                                           \
        }                                                                                                    \
    }
    PG_LOAD(0) PG_LOAD(1) PG_LOAD(2) PG_LOAD(3)
    typename S::reg b0, b1, av;
    for (int64_t p = 0; p < kb; ++p) {
        const T* bq = bp + p * rs_b;
        b0 = EDGE ? S::load_part(bq, m0) : S::load(bq);
        b1 = EDGE ? S::load_part(bq + S::W, m1) : S::load(bq + S::W);
        const T* ap = a + p * cs_a;
        PG_ROW(0) PG_ROW(1) PG_ROW(2) PG_ROW(3)
    }
    PG_STORE(0) PG_STORE(1) PG_STORE(2) PG_STORE(3)
#undef PG_LOAD
#undef PG_ROW
#undef PG_STORE
}

// C[0:M, 0:w) for m = M <= 4 over a B with unit-stride rows, k blocks
// outermost. While the tiles sweep one block, the read-ahead fetches the next
// block's row segments, a line for each line a tile consumes.
template <typename T, int M>
void panel(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, T* c, int64_t ldc, int64_t k,
           int64_t w) {
    constexpr int64_t NR = 2 * vec<T>::W;
    for (int64_t p0 = 0; p0 < k; p0 += PANEL_KB) {
        const int64_t kb = PANEL_KB < k - p0 ? PANEL_KB : k - p0;
        const int64_t rest = k - p0 - kb;
        const int64_t nk = PANEL_KB < rest ? PANEL_KB : rest;
        const T* ap = a + p0 * cs_a;
        const T* bp = b + p0 * rs_b;
        read_ahead next = rows_ahead(nk > 0 ? bp + kb * rs_b : bp, rs_b, w, nk);
        int64_t j = 0;
        for (; j + NR <= w; j += NR) {
            panel_tile<T, M, false>(ap, rs_a, cs_a, bp + j, rs_b, c + j, ldc, kb, NR, p0 == 0, rest == 0);
            next.step(nk);
        }
        if (j < w) {
            panel_tile<T, M, true>(ap, rs_a, cs_a, bp + j, rs_b, c + j, ldc, kb, w - j, p0 == 0, rest == 0);
        }
    }
}

template <typename T>
void panel_rows(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, T* c, int64_t ldc, int64_t m,
                int64_t k, int64_t w) {
    switch (m) {
    case 1:
        panel<T, 1>(a, rs_a, cs_a, b, rs_b, c, ldc, k, w);
        break;
    case 2:
        panel<T, 2>(a, rs_a, cs_a, b, rs_b, c, ldc, k, w);
        break;
    case 3:
        panel<T, 3>(a, rs_a, cs_a, b, rs_b, c, ldc, k, w);
        break;
    default:
        panel<T, 4>(a, rs_a, cs_a, b, rs_b, c, ldc, k, w);
        break;
    }
}

// The long operand's size from which the streaming kernels run. Below it the
// operand stays in the 12 MiB L3 from call to call, and gemm_small's wider
// tiles and deeper k blocks win. Single-process sweeps, six threads, old ->
// streaming medians: A x at 4 MiB float64 83 -> 98 us, at 8 MiB float64 396 ->
// 319-352 and float32 132-177 -> 112-124, at 16 MiB float32 398 -> 259-295;
// (4, k, n) at 8 MiB float64 106 -> 149, at 12 MiB float64 209-235 -> 214-250
// and float32 220-391 -> 240, at 16 MiB float64 420-431 -> 285-297.
constexpr double STREAM_BYTES = 12.0 * 1024 * 1024;

// Thread t of team computes its share of whole slabs of the long dimension.
// stream picks the streaming kernels for an A x or a short A times a long B.
template <typename T>
void slab(int64_t t, int64_t team, bool transposed, bool by_rows, bool stream, const T* a, int64_t rs_a,
          int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c, int64_t m, int64_t k, int64_t n) {
    const int64_t len = by_rows ? m : n;
    const int64_t slabs = (len + SKINNY_SLAB - 1) / SKINNY_SLAB;
    const int64_t lo = slabs * t / team * SKINNY_SLAB;
    const int64_t end = slabs * (t + 1) / team * SKINNY_SLAB;
    const int64_t hi = end < len ? end : len;
    if (lo >= hi) {
        return;
    }
    if (transposed) {
        transposed_rows<T>(a, rs_a, cs_a, b, rs_b, cs_b, c, lo, hi, k, n);
    } else if (stream && by_rows && n == 1 && cs_a == 1) {
        matvec<T>(a + lo * rs_a, rs_a, b, rs_b, c + lo, hi - lo, k);
    } else if (stream && !by_rows && m <= 4 && cs_b == 1) {
        panel_rows<T>(a, rs_a, cs_a, b + lo, rs_b, c + lo, n, m, k, hi - lo);
    } else if (by_rows) {
        gemm_small_ldc<T>(a + lo * rs_a, rs_a, cs_a, b, rs_b, cs_b, c + lo * n, n, hi - lo, k, n);
    } else {
        gemm_small_ldc<T>(a, rs_a, cs_a, b + lo * cs_b, rs_b, cs_b, c + lo, n, m, k, hi - lo);
    }
}

} // namespace

template <typename T>
void gemm_skinny(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c, int64_t m,
                 int64_t k, int64_t n) {
    if (m <= 0 || n <= 0) {
        return;
    }
    const bool transposed = n <= 4 && m > 4 && rs_a == 1;
    const bool by_rows = m >= n;
    const int64_t len = by_rows ? m : n;
    const double wide = static_cast<double>(by_rows ? m : n) * static_cast<double>(k) * sizeof(T);
    const bool stream = k > 0 && wide >= STREAM_BYTES;
    const int nt = team_size(static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k + 1),
                             SKINNY_SERIAL_BELOW, SKINNY_PER_THREAD, (len + SKINNY_SLAB - 1) / SKINNY_SLAB);
    const unsigned csr = mxcsr();
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt) if(nt > 1)
#endif
    {
        const mxcsr_scope fp(csr);
#if defined(_OPENMP)
        slab<T>(omp_get_thread_num(), omp_get_num_threads(), transposed, by_rows, stream, a, rs_a, cs_a, b, rs_b,
                cs_b, c, m, k, n);
#else
        slab<T>(0, 1, transposed, by_rows, stream, a, rs_a, cs_a, b, rs_b, cs_b, c, m, k, n);
#endif
    }
}

template void gemm_skinny<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t);
template void gemm_skinny<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t);

} // namespace pg::cpu
