#include "cpu/gemm_small.hpp"

#include <cstring>
#include <immintrin.h>

namespace pg::cpu {

namespace {

// The light pack of B, half of the 32 KiB L1d, so a panel and the A rows that
// stream past it stay in L1 together. A panel holds as many k rows as fit, so
// k beyond that runs in blocks whose partial sums go through C exactly.
constexpr int64_t PACK_BYTES = 16 * 1024;

// The k block for at most four rows of C over B read in place. Such a tile walks
// down one panel of B, a new row and often a new page per step, and the next
// panel repeats the walk one panel to the right; past about 32 rows the rows
// stop sharing the prefetcher's streams and L1 sets, and every step misses.
// Single-call diagnostic, f64 (4, 2048, 8192): 19.7 ms at 256, 8.8 at 32, 9.2
// at 16; with more rows of C the tile reuses each panel row from L1 instead,
// and the deep block saves C reloads ((96, 96, 96) 40 us at 256, 49 at 32).
constexpr int64_t SHALLOW_KB = 32;

// The rows of in-place B one column chunk may span, half of the 256 KiB L2.
constexpr int64_t CHUNK_BYTES = 128 * 1024;

inline int64_t lesser(int64_t x, int64_t y) {
    return x < y ? x : y;
}

template <typename T>
struct simd;

template <>
struct simd<double> {
    using reg = __m256d;
    static constexpr int64_t W = 4;
    static reg zero() { return _mm256_setzero_pd(); }
    static reg load(const double* p) { return _mm256_loadu_pd(p); }
    static reg bcast(const double* p) { return _mm256_broadcast_sd(p); }
    static reg fmadd(reg x, reg y, reg acc) { return _mm256_fmadd_pd(x, y, acc); }
    // v + 0.0 is written v - (-0.0), the same IEEE operation under every rounding
    // mode and MXCSR state, because MSVC folds an intrinsic add of +0.0 to v and
    // would leak an underflowed -0.0.
    static reg plus_zero(reg v) { return _mm256_sub_pd(v, _mm256_set1_pd(-0.0)); }
    static void store(double* p, reg v) { _mm256_storeu_pd(p, v); }
    // VMASKMOV touches no memory in masked-off lanes, so a partial vector at the
    // very end of C or of a row never faults.
    static __m256i mask(int64_t live) {
        return _mm256_cmpgt_epi64(_mm256_set1_epi64x(live), _mm256_setr_epi64x(0, 1, 2, 3));
    }
    static reg load_part(const double* p, int64_t live) { return _mm256_maskload_pd(p, mask(live)); }
    static void store_part(double* p, int64_t live, reg v) { _mm256_maskstore_pd(p, mask(live), v); }
};

template <>
struct simd<float> {
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
    static reg load_part(const float* p, int64_t live) { return _mm256_maskload_ps(p, mask(live)); }
    static void store_part(float* p, int64_t live, reg v) { _mm256_maskstore_ps(p, mask(live), v); }
};

// The value a chain starts a k block from: +0.0 on the first block, else the
// partial sum the previous block left in C. The C pointer is formed only for a
// live vector, never past the end of a row.
template <typename T>
inline typename simd<T>::reg start(const T* c, int64_t ldc, int r, int v, int64_t nr, bool first) {
    using S = simd<T>;
    const int64_t live = nr - v * S::W;
    if (first || live <= 0) {
        return S::zero();
    }
    const T* cr = c + r * ldc + v * S::W;
    return live >= S::W ? S::load(cr) : S::load_part(cr, live);
}

template <typename T>
inline void put(T* c, int64_t ldc, int r, int v, int64_t nr, bool last, typename simd<T>::reg x) {
    using S = simd<T>;
    const int64_t live = nr - v * S::W;
    if (live <= 0) {
        return;
    }
    T* cr = c + r * ldc + v * S::W;
    if (last) {
        x = S::plus_zero(x);
    }
    if (live >= S::W) {
        S::store(cr, x);
    } else {
        S::store_part(cr, live, x);
    }
}

// Every (row, vector) accumulator a tile shape uses: one row of 8 vectors, two
// of 4, three of 3, or up to six of 2. The accumulators are named locals and
// each row's broadcast is consumed by its FMAs at once, the microkernel's
// layout: MSVC keeps a loop-indexed accumulator array on the stack (every FMA
// loaded and stored it), and does not inline the lambdas of a compile-time
// unroll.
#define PG_TILE_ACCS(X)                                                                                      \
    X(0, 0) X(0, 1) X(0, 2) X(0, 3) X(0, 4) X(0, 5) X(0, 6) X(0, 7) X(1, 0) X(1, 1) X(1, 2) X(1, 3) X(2, 0) \
    X(2, 1) X(2, 2) X(3, 0) X(3, 1) X(4, 0) X(4, 1) X(5, 0) X(5, 1)
#define PG_DECL(r, v) [[maybe_unused]] typename S::reg c##r##v = S::zero();
#define PG_START(r, v)                                                                                       \
    if constexpr (r < MR && v < NV) {                                                                        \
        c##r##v = start(c, ldc, r, v, nr, first);                                                            \
    }
#define PG_PUT(r, v)                                                                                         \
    if constexpr (r < MR && v < NV) {                                                                        \
        put(c, ldc, r, v, nr, last, c##r##v);                                                                \
    }
#define PG_B(v)                                                                                              \
    if constexpr (v < NV) {                                                                                  \
        b##v = S::load(bq + v * S::W);                                                                       \
    }
#define PG_F(r, v)                                                                                           \
    if constexpr (v < NV) {                                                                                  \
        c##r##v = S::fmadd(av, b##v, c##r##v);                                                               \
    }
#define PG_ROW(r, fmas)                                                                                      \
    if constexpr (r < MR) {                                                                                  \
        av = S::bcast(ap + r * rs_a);                                                                        \
        fmas                                                                                                 \
    }

// One MR x (NV vectors) tile of C over kb steps of k. a is the tile's first A
// row at the block's first k column; row p of the B panel is bp + p * ldb, NV
// full vectors wide. first starts each chain from +0.0 instead of the partial
// sum the previous k block left in C; last applies the final + 0.0. nr is the
// live column count, so the panel's padded columns are never stored.
template <typename T, int MR, int NV>
void tile(const T* a, int64_t rs_a, int64_t cs_a, const T* bp, int64_t ldb, T* c, int64_t ldc, int64_t kb,
          int64_t nr, bool first, bool last) {
    using S = simd<T>;
    static_assert(MR >= 1 && MR <= 6 && NV >= 1 && NV <= (MR == 1 ? 8 : MR == 2 ? 4 : MR == 3 ? 3 : 2));
    PG_TILE_ACCS(PG_DECL)
    PG_TILE_ACCS(PG_START)
    [[maybe_unused]] typename S::reg b0, b1, b2, b3, b4, b5, b6, b7;
    typename S::reg av;
    for (int64_t p = 0; p < kb; ++p) {
        const T* bq = bp + p * ldb;
        const T* ap = a + p * cs_a;
        PG_B(0) PG_B(1) PG_B(2) PG_B(3) PG_B(4) PG_B(5) PG_B(6) PG_B(7)
        PG_ROW(0, PG_F(0, 0) PG_F(0, 1) PG_F(0, 2) PG_F(0, 3) PG_F(0, 4) PG_F(0, 5) PG_F(0, 6) PG_F(0, 7))
        PG_ROW(1, PG_F(1, 0) PG_F(1, 1) PG_F(1, 2) PG_F(1, 3))
        PG_ROW(2, PG_F(2, 0) PG_F(2, 1) PG_F(2, 2))
        PG_ROW(3, PG_F(3, 0) PG_F(3, 1))
        PG_ROW(4, PG_F(4, 0) PG_F(4, 1))
        PG_ROW(5, PG_F(5, 0) PG_F(5, 1))
    }
    PG_TILE_ACCS(PG_PUT)
}

#undef PG_TILE_ACCS
#undef PG_DECL
#undef PG_START
#undef PG_PUT
#undef PG_B
#undef PG_F
#undef PG_ROW

// The row tile for the last mr < MR rows of C.
template <typename T, int MR, int NV>
void rows(int64_t mr, const T* a, int64_t rs_a, int64_t cs_a, const T* bp, int64_t ldb, T* c, int64_t ldc,
          int64_t kb, int64_t nr, bool first, bool last) {
    if constexpr (MR > 1) {
        if (mr < MR) {
            rows<T, MR - 1, NV>(mr, a, rs_a, cs_a, bp, ldb, c, ldc, kb, nr, first, last);
            return;
        }
    }
    tile<T, MR, NV>(a, rs_a, cs_a, bp, ldb, c, ldc, kb, nr, first, last);
}

// Copies kb x nr of B into a panel of width columns, zero-padding the rest, so
// the tile loads full vectors from any layout. The source is read along its
// nearer stride: walking the far one touches a new line per element, and at a
// power-of-two row pitch those lines share one L1 set.
template <typename T>
void light_pack(T* dst, const T* src, int64_t rs, int64_t cs, int64_t kb, int64_t nr, int64_t width) {
    if ((cs < 0 ? -cs : cs) <= (rs < 0 ? -rs : rs)) {
        for (int64_t p = 0; p < kb; ++p) {
            for (int64_t j = 0; j < nr; ++j) {
                dst[p * width + j] = src[p * rs + j * cs];
            }
        }
    } else {
        for (int64_t j = 0; j < nr; ++j) {
            for (int64_t p = 0; p < kb; ++p) {
                dst[p * width + j] = src[p * rs + j * cs];
            }
        }
    }
    for (int64_t p = 0; p < kb; ++p) {
        for (int64_t j = nr; j < width; ++j) {
            dst[p * width + j] = T(0);
        }
    }
}

// k blocks outermost, so a block's rows of B serve every panel and row tile
// before the next block is read. B read in place (unit-stride rows; only a
// partial last panel is packed) is swept in column chunks whose rows of B stay
// in L2, row tiles outside panels, so C is written in row segments: walking
// panels outside row tiles wrote C in 64-byte column strips, and a line that
// straddles two strips was fetched twice (single-call diagnostic, f64
// (2000, 4, 2000), one thread: 1.24x the old packed path's time by strips,
// 0.28x by chunks).
// Strided B is light-packed one panel at a time and reused by every row tile.
template <typename T, int MR, int NV>
void drive(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c, int64_t ldc,
           int64_t m, int64_t k, int64_t n) {
    constexpr int64_t NR = NV * simd<T>::W;
    constexpr int64_t KB = PACK_BYTES / static_cast<int64_t>(sizeof(T)) / NR;
    alignas(32) T buf[KB * NR];
    const int64_t depth = cs_b == 1 && m <= 4 ? lesser(KB, SHALLOW_KB) : KB;
    for (int64_t p0 = 0; p0 < k; p0 += depth) {
        const int64_t kb = lesser(depth, k - p0);
        const bool first = p0 == 0;
        const bool last = p0 + kb == k;
        if (cs_b == 1) {
            const int64_t full = n - n % NR;
            if (full < n) {
                light_pack(buf, b + p0 * rs_b + full, rs_b, cs_b, kb, n - full, NR);
            }
            const int64_t fit = CHUNK_BYTES / (kb * static_cast<int64_t>(sizeof(T))) / NR * NR;
            const int64_t jb = fit > NR ? fit : NR;
            for (int64_t j1 = 0; j1 < n; j1 += jb) {
                const int64_t j2 = lesser(j1 + jb, n);
                for (int64_t i0 = 0; i0 < m; i0 += MR) {
                    const int64_t mr = lesser(MR, m - i0);
                    const T* ai = a + i0 * rs_a + p0 * cs_a;
                    for (int64_t j0 = j1; j0 < j2; j0 += NR) {
                        const bool edge = j0 >= full;
                        rows<T, MR, NV>(mr, ai, rs_a, cs_a, edge ? buf : b + p0 * rs_b + j0, edge ? NR : rs_b,
                                        c + i0 * ldc + j0, ldc, kb, lesser(NR, n - j0), first, last);
                    }
                }
            }
            continue;
        }
        for (int64_t j0 = 0; j0 < n; j0 += NR) {
            const int64_t nr = lesser(NR, n - j0);
            light_pack(buf, b + p0 * rs_b + j0 * cs_b, rs_b, cs_b, kb, nr, NR);
            for (int64_t i0 = 0; i0 < m; i0 += MR) {
                rows<T, MR, NV>(lesser(MR, m - i0), a + i0 * rs_a + p0 * cs_a, rs_a, cs_a, buf, NR,
                                c + i0 * ldc + j0, ldc, kb, nr, first, last);
            }
        }
    }
}

// The narrowest panel that covers n, so a short row of C does not pay for
// columns it does not have.
template <typename T, int MR, int NV>
void widths(int64_t nv, const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c,
            int64_t ldc, int64_t m, int64_t k, int64_t n) {
    if constexpr (NV > 1) {
        if (nv < NV) {
            widths<T, MR, NV - 1>(nv, a, rs_a, cs_a, b, rs_b, cs_b, c, ldc, m, k, n);
            return;
        }
    }
    drive<T, MR, NV>(a, rs_a, cs_a, b, rs_b, cs_b, c, ldc, m, k, n);
}

} // namespace

// Tiles: 6 x 2 vectors is the packed microkernel's 12 accumulators plus two B
// registers and a broadcast, 15 of the 16 ymm. Fewer rows get wider tiles so at
// least 8 independent chains cover the 4-cycle FMA latency on both FMA ports:
// 1 x 8 (B straight from memory), 2 x 4, 3 x 3, and 4 x 2, the widest four-row
// tile that leaves the register file a spare.
template <typename T>
void gemm_small_ldc(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c,
                    int64_t ldc, int64_t m, int64_t k, int64_t n) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (k <= 0) {
        for (int64_t i = 0; i < m; ++i) {
            std::memset(c + i * ldc, 0, static_cast<size_t>(n) * sizeof(T));
        }
        return;
    }
    const int64_t nv = (n + simd<T>::W - 1) / simd<T>::W;
    switch (m) {
    case 1:
        widths<T, 1, 8>(nv, a, rs_a, cs_a, b, rs_b, cs_b, c, ldc, m, k, n);
        break;
    case 2:
        widths<T, 2, 4>(nv, a, rs_a, cs_a, b, rs_b, cs_b, c, ldc, m, k, n);
        break;
    case 3:
        widths<T, 3, 3>(nv, a, rs_a, cs_a, b, rs_b, cs_b, c, ldc, m, k, n);
        break;
    case 4:
        widths<T, 4, 2>(nv, a, rs_a, cs_a, b, rs_b, cs_b, c, ldc, m, k, n);
        break;
    default:
        widths<T, 6, 2>(nv, a, rs_a, cs_a, b, rs_b, cs_b, c, ldc, m, k, n);
        break;
    }
}

template <typename T>
void gemm_small(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c, int64_t m,
                int64_t k, int64_t n) {
    gemm_small_ldc<T>(a, rs_a, cs_a, b, rs_b, cs_b, c, n, m, k, n);
}

template void gemm_small<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t);
template void gemm_small<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t);
template void gemm_small_ldc<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t, int64_t);
template void gemm_small_ldc<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t, int64_t);

} // namespace pg::cpu
