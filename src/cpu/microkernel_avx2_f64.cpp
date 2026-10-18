#include "cpu/microkernel.hpp"

#include <immintrin.h>

namespace pg::cpu {

namespace {

// The deepest k unroll whose inner loop each compiler keeps free of stack
// spills, read from the PG_KERNEL_LISTINGS listings: MSVC 19.4x holds 8, GCC 13
// spills an accumulator at 8 and holds 4.
#if defined(_MSC_VER)
constexpr int64_t K_UNROLL = 8;
#else
constexpr int64_t K_UNROLL = 4;
#endif

// About eleven k steps of A ahead, the distance BLIS's Haswell kernels use. A is
// the streamed operand (a fresh MR x kc panel from L2 on every call); the B
// micro-panel is reused by every ir call of a jr column, so after the first call
// it sits in L1 and a prefetch of it would only spend a load slot.
constexpr std::uintptr_t A_PREFETCH_BYTES = 512;

// The address is formed in integer arithmetic because near the end of the last
// panel it points past the buffer; forming that pointer directly is undefined
// behavior even though the prefetch itself never faults.
inline void prefetch(const void* p, std::uintptr_t ahead) {
    _mm_prefetch(reinterpret_cast<const char*>(reinterpret_cast<std::uintptr_t>(p) + ahead), _MM_HINT_T0);
}

// The value stored for a finished vector v of the tile. On the first k panel it
// is v + 0.0, bit for bit what the old memset-then-add produced (an underflowed
// -0.0 becomes +0.0), without reading C; on later panels it is C + v. v + 0.0
// is written v - (-0.0), the same IEEE operation with the same bits for every v,
// rounding mode, and MXCSR state, because MSVC 19.4x folds an intrinsic add of
// +0.0 down to v and would leak the -0.0.
inline __m256d update(const double* cr, __m256d v, bool first) {
    return first ? _mm256_sub_pd(v, _mm256_set1_pd(-0.0)) : _mm256_add_pd(_mm256_loadu_pd(cr), v);
}

inline __m256d update_masked(const double* cr, __m256i mask, __m256d v, bool first) {
    return first ? _mm256_sub_pd(v, _mm256_set1_pd(-0.0)) : _mm256_add_pd(_mm256_maskload_pd(cr, mask), v);
}

// Adds the finished 6x8 tile t (row-major, 32-byte aligned) into the live mr x nr
// cells of C. The edge path uses VMASKMOV, which writes nothing in masked-off
// lanes and suppresses faults there, so a flush against the very end of C is
// safe. _mm256_cmpgt_epi64 sets a lane when nr exceeds its column index, the
// live-column predicate at every width 0..8. The high half is skipped outright
// when nr <= 4: cr + 4 can then lie more than one past the end of C on the last
// row, and forming that pointer would be undefined behavior.
inline void store_tile(double* c, int64_t ldc, int64_t mr, int64_t nr, bool first, const double* t) {
    if (mr == 6 && nr == 8) {
        for (int64_t r = 0; r < 6; ++r) {
            double* cr = c + r * ldc;
            _mm256_storeu_pd(cr + 0, update(cr + 0, _mm256_load_pd(t + 8 * r + 0), first));
            _mm256_storeu_pd(cr + 4, update(cr + 4, _mm256_load_pd(t + 8 * r + 4), first));
        }
        return;
    }
    const __m256i nrv = _mm256_set1_epi64x(nr);
    const __m256i m0 = _mm256_cmpgt_epi64(nrv, _mm256_setr_epi64x(0, 1, 2, 3));
    const __m256i m1 = _mm256_cmpgt_epi64(nrv, _mm256_setr_epi64x(4, 5, 6, 7));
    for (int64_t r = 0; r < mr; ++r) {
        double* cr = c + r * ldc;
        _mm256_maskstore_pd(cr + 0, m0, update_masked(cr + 0, m0, _mm256_load_pd(t + 8 * r + 0), first));
        if (nr > 4) {
            _mm256_maskstore_pd(cr + 4, m1, update_masked(cr + 4, m1, _mm256_load_pd(t + 8 * r + 4), first));
        }
    }
}

} // namespace

// One k step of the 6x8 tile: two aligned B loads, then for each row one
// broadcast consumed at once by its two FMAs. b0, b1 and av are declared once
// outside the k loop and reassigned here: with per-step const temporaries MSVC
// reorders the unrolled steps, keeps several broadcasts live, and spills
// accumulators (34 to 39 stack references per unrolled iteration). Every
// accumulator takes its FMAs in k order, one chain per C element, so the bits do
// not depend on the unroll depth or on where the tail loop starts.
#define PG_F64_STEP(u)                                                             \
    b0 = _mm256_load_pd(bp + 8 * (u));                                             \
    b1 = _mm256_load_pd(bp + 8 * (u) + 4);                                         \
    av = _mm256_broadcast_sd(ap + 6 * (u) + 0);                                    \
    c00 = _mm256_fmadd_pd(av, b0, c00);                                            \
    c01 = _mm256_fmadd_pd(av, b1, c01);                                            \
    av = _mm256_broadcast_sd(ap + 6 * (u) + 1);                                    \
    c10 = _mm256_fmadd_pd(av, b0, c10);                                            \
    c11 = _mm256_fmadd_pd(av, b1, c11);                                            \
    av = _mm256_broadcast_sd(ap + 6 * (u) + 2);                                    \
    c20 = _mm256_fmadd_pd(av, b0, c20);                                            \
    c21 = _mm256_fmadd_pd(av, b1, c21);                                            \
    av = _mm256_broadcast_sd(ap + 6 * (u) + 3);                                    \
    c30 = _mm256_fmadd_pd(av, b0, c30);                                            \
    c31 = _mm256_fmadd_pd(av, b1, c31);                                            \
    av = _mm256_broadcast_sd(ap + 6 * (u) + 4);                                    \
    c40 = _mm256_fmadd_pd(av, b0, c40);                                            \
    c41 = _mm256_fmadd_pd(av, b1, c41);                                            \
    av = _mm256_broadcast_sd(ap + 6 * (u) + 5);                                    \
    c50 = _mm256_fmadd_pd(av, b0, c50);                                            \
    c51 = _mm256_fmadd_pd(av, b1, c51);

// The f64 register tile: MR=6 rows by NR=8 columns in 12 ymm accumulators, plus
// two B registers and one broadcast register, 15 of the 16. The packed panels are
// 64-byte aligned and zero-padded to MR/NR, so the k loop runs to kc with no
// branch and the padded lanes add zero. _mm256_fmadd_pd is a single-rounded FMA
// on MSVC (/fp:precise) and GCC alike.
void microkernel_f64_6x8(const double* ap, const double* bp, double* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr, bool first) {
    // Only live elements are prefetched (row start and last live column), so no
    // address past C is formed; the two cover a row that straddles two lines.
    for (int64_t r = 0; r < mr; ++r) {
        prefetch(c + r * ldc, 0);
        prefetch(c + r * ldc + (nr - 1), 0);
    }

    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();
    __m256d b0, b1, av;

    // Count-down loops keep the trip count in one register with a fused sub/jne
    // test; a p < kc test costs MSVC a second counter and a cmp.
    for (int64_t q = kc / K_UNROLL; q != 0; --q) {
        // Four steps of A are 192 bytes, three lines.
        prefetch(ap, A_PREFETCH_BYTES);
        prefetch(ap, A_PREFETCH_BYTES + 64);
        prefetch(ap, A_PREFETCH_BYTES + 128);
        PG_F64_STEP(0)
        PG_F64_STEP(1)
        PG_F64_STEP(2)
        PG_F64_STEP(3)
        if constexpr (K_UNROLL == 8) {
            prefetch(ap, A_PREFETCH_BYTES + 192);
            prefetch(ap, A_PREFETCH_BYTES + 256);
            prefetch(ap, A_PREFETCH_BYTES + 320);
            PG_F64_STEP(4)
            PG_F64_STEP(5)
            PG_F64_STEP(6)
            PG_F64_STEP(7)
        }
        ap += 6 * K_UNROLL;
        bp += 8 * K_UNROLL;
    }
    for (int64_t q = kc % K_UNROLL; q != 0; --q) {
        PG_F64_STEP(0)
        ap += 6;
        bp += 8;
    }

    // The accumulators leave through an aligned stack tile even for a full tile:
    // with the masked edge store reading them directly, GCC 13 spills one
    // accumulator inside the unrolled k loop. Twelve stores and loads per call,
    // against kc * 12 FMAs.
    alignas(32) double t[6 * 8];
    _mm256_store_pd(t + 0, c00);
    _mm256_store_pd(t + 4, c01);
    _mm256_store_pd(t + 8, c10);
    _mm256_store_pd(t + 12, c11);
    _mm256_store_pd(t + 16, c20);
    _mm256_store_pd(t + 20, c21);
    _mm256_store_pd(t + 24, c30);
    _mm256_store_pd(t + 28, c31);
    _mm256_store_pd(t + 32, c40);
    _mm256_store_pd(t + 36, c41);
    _mm256_store_pd(t + 40, c50);
    _mm256_store_pd(t + 44, c51);
    store_tile(c, ldc, mr, nr, first, t);
}

#undef PG_F64_STEP

} // namespace pg::cpu
