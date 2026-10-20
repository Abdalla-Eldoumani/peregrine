#include "cpu/microkernel.hpp"

#include <immintrin.h>

namespace pg::cpu {

namespace {

// The deepest k unroll whose inner loop each compiler keeps free of stack
// spills, read from the PG_KERNEL_LISTINGS listings (same split as f64).
#if defined(_MSC_VER)
constexpr int64_t K_UNROLL = 8;
#else
constexpr int64_t K_UNROLL = 4;
#endif

// About eleven k steps of A ahead (24 bytes a step), the distance BLIS's Haswell
// kernels use. The current B panel is not prefetched and the next one goes to
// L2 through b_next, for the reasons given in the f64 kernel. On its own it
// measured within noise for f32; with the barrier-free driver it is part of
// the schedule that measured 1.8 to 3.4 percent faster at n = 2048.
constexpr std::uintptr_t A_PREFETCH_BYTES = 256;

// Integer arithmetic: near the end of the last panel the address lies past the
// buffer, and forming that pointer directly is undefined behavior.
inline void prefetch(const void* p, std::uintptr_t ahead) {
    _mm_prefetch(reinterpret_cast<const char*>(reinterpret_cast<std::uintptr_t>(p) + ahead), _MM_HINT_T0);
}

// The value stored for a finished vector v of the tile: v - (-0.0), which is
// v + 0.0, on the first k panel without reading C, and C + v after that. The f64
// kernel gives the reason for the subtraction form.
inline __m256 update(const float* cr, __m256 v, bool first) {
    return first ? _mm256_sub_ps(v, _mm256_set1_ps(-0.0f)) : _mm256_add_ps(_mm256_loadu_ps(cr), v);
}

inline __m256 update_masked(const float* cr, __m256i mask, __m256 v, bool first) {
    return first ? _mm256_sub_ps(v, _mm256_set1_ps(-0.0f)) : _mm256_add_ps(_mm256_maskload_ps(cr, mask), v);
}

// Adds the finished MR x 16 tile t (row-major, 32-byte aligned) into the live
// mr x nr cells of C. The edge path uses VMASKMOV, which writes nothing in
// masked-off lanes and suppresses faults there, so a flush against the very end
// of C is safe. The lane masks compare each vector's live count against
// setr(0..7), the live-column predicate at every width 0..16. The high half is
// skipped outright when nr <= 8: cr + 8 can then lie more than one past the end
// of C on the last row, and forming that pointer would be undefined behavior.
template <int64_t MR>
void store_tile(float* c, int64_t ldc, int64_t mr, int64_t nr, bool first, const float* t) {
    if (mr == MR && nr == 16) {
        for (int64_t r = 0; r < MR; ++r) {
            float* cr = c + r * ldc;
            _mm256_storeu_ps(cr + 0, update(cr + 0, _mm256_load_ps(t + 16 * r + 0), first));
            _mm256_storeu_ps(cr + 8, update(cr + 8, _mm256_load_ps(t + 16 * r + 8), first));
        }
        return;
    }
    const __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const int lo = nr < 8 ? static_cast<int>(nr) : 8;
    const int hi = nr > 8 ? static_cast<int>(nr) - 8 : 0;
    const __m256i m0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(lo), idx);
    const __m256i m1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(hi), idx);
    for (int64_t r = 0; r < mr; ++r) {
        float* cr = c + r * ldc;
        _mm256_maskstore_ps(cr + 0, m0, update_masked(cr + 0, m0, _mm256_load_ps(t + 16 * r + 0), first));
        if (nr > 8) {
            _mm256_maskstore_ps(cr + 8, m1, update_masked(cr + 8, m1, _mm256_load_ps(t + 16 * r + 8), first));
        }
    }
}

} // namespace

// One k step of the 6x16 tile, in the same spill-free form as the f64 kernel: b0,
// b1 and av are reassigned rather than redeclared per step, and every
// accumulator takes its FMAs in k order, one chain per C element.
#define PG_F32_STEP(u)                                                             \
    b0 = _mm256_load_ps(bp + 16 * (u));                                            \
    b1 = _mm256_load_ps(bp + 16 * (u) + 8);                                        \
    av = _mm256_broadcast_ss(ap + 6 * (u) + 0);                                    \
    c00 = _mm256_fmadd_ps(av, b0, c00);                                            \
    c01 = _mm256_fmadd_ps(av, b1, c01);                                            \
    av = _mm256_broadcast_ss(ap + 6 * (u) + 1);                                    \
    c10 = _mm256_fmadd_ps(av, b0, c10);                                            \
    c11 = _mm256_fmadd_ps(av, b1, c11);                                            \
    av = _mm256_broadcast_ss(ap + 6 * (u) + 2);                                    \
    c20 = _mm256_fmadd_ps(av, b0, c20);                                            \
    c21 = _mm256_fmadd_ps(av, b1, c21);                                            \
    av = _mm256_broadcast_ss(ap + 6 * (u) + 3);                                    \
    c30 = _mm256_fmadd_ps(av, b0, c30);                                            \
    c31 = _mm256_fmadd_ps(av, b1, c31);                                            \
    av = _mm256_broadcast_ss(ap + 6 * (u) + 4);                                    \
    c40 = _mm256_fmadd_ps(av, b0, c40);                                            \
    c41 = _mm256_fmadd_ps(av, b1, c41);                                            \
    av = _mm256_broadcast_ss(ap + 6 * (u) + 5);                                    \
    c50 = _mm256_fmadd_ps(av, b0, c50);                                            \
    c51 = _mm256_fmadd_ps(av, b1, c51);

// The f32 register tile: MR=6 rows by NR=16 columns in 12 ymm accumulators (each
// row two __m256 of 8 floats), plus two B registers and one broadcast register.
// The packed panels are 64-byte aligned and zero-padded to MR/NR, so the k loop
// runs to kc with no branch and the padded lanes add zero. _mm256_fmadd_ps is a
// single-rounded FMA on MSVC (/fp:precise) and GCC alike.
void microkernel_f32_6x16(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr, bool first,
                          const float* b_next) {
    // Only live elements are prefetched, so no address past C is formed.
    for (int64_t r = 0; r < mr; ++r) {
        prefetch(c + r * ldc, 0);
        prefetch(c + r * ldc + (nr - 1), 0);
    }
    // kc * 8 bytes over the loop, as in the f64 kernel.
    const char* pf_b = reinterpret_cast<const char*>(b_next);

    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    __m256 b0, b1, av;

    for (int64_t q = kc / K_UNROLL; q != 0; --q) {
        // Four steps of A are 96 bytes; two lines cover them.
        prefetch(ap, A_PREFETCH_BYTES);
        prefetch(ap, A_PREFETCH_BYTES + 64);
        _mm_prefetch(pf_b, _MM_HINT_T1);
        pf_b += K_UNROLL * 8;
        PG_F32_STEP(0)
        PG_F32_STEP(1)
        PG_F32_STEP(2)
        PG_F32_STEP(3)
        if constexpr (K_UNROLL == 8) {
            prefetch(ap, A_PREFETCH_BYTES + 128);
            PG_F32_STEP(4)
            PG_F32_STEP(5)
            PG_F32_STEP(6)
            PG_F32_STEP(7)
        }
        ap += 6 * K_UNROLL;
        bp += 16 * K_UNROLL;
    }
    for (int64_t q = kc % K_UNROLL; q != 0; --q) {
        PG_F32_STEP(0)
        ap += 6;
        bp += 16;
    }

    // Through an aligned stack tile for the same GCC reason as the f64 kernel.
    alignas(32) float t[6 * 16];
    _mm256_store_ps(t + 0, c00);
    _mm256_store_ps(t + 8, c01);
    _mm256_store_ps(t + 16, c10);
    _mm256_store_ps(t + 24, c11);
    _mm256_store_ps(t + 32, c20);
    _mm256_store_ps(t + 40, c21);
    _mm256_store_ps(t + 48, c30);
    _mm256_store_ps(t + 56, c31);
    _mm256_store_ps(t + 64, c40);
    _mm256_store_ps(t + 72, c41);
    _mm256_store_ps(t + 80, c50);
    _mm256_store_ps(t + 88, c51);
    store_tile<6>(c, ldc, mr, nr, first, t);
}

#undef PG_F32_STEP

} // namespace pg::cpu
