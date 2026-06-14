#include "cpu/microkernel.hpp"

#include <immintrin.h>

namespace pg::cpu {

// Two f32 register-tile shapes, measured at n=512 to pick the winner instead of
// arguing from register counts (port pressure differs between MSVC and GCC
// codegen, so the count is only a guess). The measurement chose 8x16: it beat
// 6x16 by ~8% on the noise-robust best-min floor at n=512 on the reference
// machine, overruling the register-budget prediction that 8x16's full-file
// accumulator set would lose to reload traffic. gemm_blis.cpp's MR_F32 constant
// carries the measured numbers and selects 8x16 by default; 6x16 is kept here,
// reachable via PG_F32_SHAPE_6x16, so the decision stays reproducible without
// resurrecting a deleted kernel.
// Both compute NR=16 columns as two __m256 of 8 floats each; they differ in MR.
//
// microkernel_f32_8x16 (MR=8): 8 rows x 2 column vectors = 16 ymm accumulators,
// which is the entire 16-register file. The 2 B loads and the A broadcast cannot
// all stay live at once, so the compiler must reload B (or A) each k-step or
// spill an accumulator; the higher arithmetic density (16 FMAs per k-step) is
// the bet that hides that traffic. microkernel_f32_6x16 (MR=6): 6 rows x 2
// vectors = 12 accumulators, the same budget the f64 6x8 kernel proved, leaving
// 2 ymm for the B loads and 1 for the broadcast inside the 16-register file with
// no forced spill. Both use _mm256_fmadd_ps (a real single-rounded FMA on MSVC
// /fp:precise and GCC; neither contracts a split mul+add), so the numerics are
// exactly one rounding per accumulation step as the f32 tolerance contract
// assumes. The packed panels are zero-padded to MR/NR by the caller's f32
// packers, so the k loop runs to kc with no branch and the padded lanes add zero.
//
// The C store is the only place tile extent matters, and it is exactly where the
// legacy kernel corrupted memory: a partial tile (mr < MR or nr < 16) writes only
// its live mr x nr cells. The full tile uses plain loads/stores; the edge tile
// uses VMASKMOV with a mask built by an 8-wide integer compare against
// setr(0..7), which is correct at every f32 tail width 0..8 per the verified
// probe, reads zero in masked-off lanes, writes nothing there, and is
// fault-suppressed even across an unmapped guard page, so a flush against the
// very end of C is safe. The caller zeroes C once and the kernel accumulates
// (+=), so the pc (k-block) loop sums correctly with one store variant.

namespace {

// Builds the low/high 8-wide lane masks for an f32 column tail. nr is the live
// column count in 0..16; the low vector covers columns 0..7 and the high vector
// covers 8..15. _mm256_cmpgt_epi32 sets a lane to all-ones when the per-vector
// live count exceeds that lane's index, which is the live-column predicate at
// every tail width including a full-low/empty-high split.
inline void edge_masks_f32(int64_t nr, __m256i& m0, __m256i& m1) {
    const __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const int lo = nr < 8 ? static_cast<int>(nr) : 8;
    const int hi = nr > 8 ? static_cast<int>(nr) - 8 : 0;
    m0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(lo), idx);
    m1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(hi), idx);
}

} // namespace

void microkernel_f32_8x16(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    __m256 c60 = _mm256_setzero_ps(), c61 = _mm256_setzero_ps();
    __m256 c70 = _mm256_setzero_ps(), c71 = _mm256_setzero_ps();

    for (int64_t p = 0; p < kc; ++p) {
        const __m256 b0 = _mm256_loadu_ps(bp + 0);
        const __m256 b1 = _mm256_loadu_ps(bp + 8);
        c00 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b0, c00);
        c01 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b1, c01);
        c10 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b0, c10);
        c11 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b1, c11);
        c20 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b0, c20);
        c21 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b1, c21);
        c30 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b0, c30);
        c31 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b1, c31);
        c40 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b0, c40);
        c41 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b1, c41);
        c50 = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b0, c50);
        c51 = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b1, c51);
        c60 = _mm256_fmadd_ps(_mm256_set1_ps(ap[6]), b0, c60);
        c61 = _mm256_fmadd_ps(_mm256_set1_ps(ap[6]), b1, c61);
        c70 = _mm256_fmadd_ps(_mm256_set1_ps(ap[7]), b0, c70);
        c71 = _mm256_fmadd_ps(_mm256_set1_ps(ap[7]), b1, c71);
        ap += 8;
        bp += 16;
    }

    const __m256 acc0[8] = {c00, c10, c20, c30, c40, c50, c60, c70};
    const __m256 acc1[8] = {c01, c11, c21, c31, c41, c51, c61, c71};

    if (mr == 8 && nr == 16) {
        for (int64_t r = 0; r < 8; ++r) {
            float* cr = c + r * ldc;
            _mm256_storeu_ps(cr + 0, _mm256_add_ps(_mm256_loadu_ps(cr + 0), acc0[r]));
            _mm256_storeu_ps(cr + 8, _mm256_add_ps(_mm256_loadu_ps(cr + 8), acc1[r]));
        }
        return;
    }

    __m256i m0, m1;
    edge_masks_f32(nr, m0, m1);
    // The high vector (columns 8..15) is touched only when nr > 8. When nr <= 8 its
    // mask m1 is all-zero, so the masked store there writes nothing -- but cr + 8
    // would still be FORMED, and on the last row of the rightmost tile (jc+jr)+8 can
    // exceed n, making cr + 8 more than one-past-the-end of the m*n C object. That
    // pointer formation is C++ UB ([expr.add]) even unread, so guard the high half
    // on nr > 8: it is formed only when a high-half column is live and cr + 8 is in
    // bounds. Results are identical (the guarded-out case stored nothing anyway).
    const bool has_high = nr > 8;
    for (int64_t r = 0; r < mr; ++r) {
        float* cr = c + r * ldc;
        const __m256 old0 = _mm256_maskload_ps(cr + 0, m0);
        _mm256_maskstore_ps(cr + 0, m0, _mm256_add_ps(old0, acc0[r]));
        if (has_high) {
            const __m256 old1 = _mm256_maskload_ps(cr + 8, m1);
            _mm256_maskstore_ps(cr + 8, m1, _mm256_add_ps(old1, acc1[r]));
        }
    }
}

void microkernel_f32_6x16(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int64_t p = 0; p < kc; ++p) {
        const __m256 b0 = _mm256_loadu_ps(bp + 0);
        const __m256 b1 = _mm256_loadu_ps(bp + 8);
        const __m256 a0 = _mm256_set1_ps(ap[0]);
        const __m256 a1 = _mm256_set1_ps(ap[1]);
        const __m256 a2 = _mm256_set1_ps(ap[2]);
        const __m256 a3 = _mm256_set1_ps(ap[3]);
        const __m256 a4 = _mm256_set1_ps(ap[4]);
        const __m256 a5 = _mm256_set1_ps(ap[5]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
        ap += 6;
        bp += 16;
    }

    const __m256 acc0[6] = {c00, c10, c20, c30, c40, c50};
    const __m256 acc1[6] = {c01, c11, c21, c31, c41, c51};

    if (mr == 6 && nr == 16) {
        for (int64_t r = 0; r < 6; ++r) {
            float* cr = c + r * ldc;
            _mm256_storeu_ps(cr + 0, _mm256_add_ps(_mm256_loadu_ps(cr + 0), acc0[r]));
            _mm256_storeu_ps(cr + 8, _mm256_add_ps(_mm256_loadu_ps(cr + 8), acc1[r]));
        }
        return;
    }

    __m256i m0, m1;
    edge_masks_f32(nr, m0, m1);
    // Guard the high vector (columns 8..15) on nr > 8 so cr + 8 is never formed
    // past one-past-the-end when its mask is empty -- same UB-avoidance and same
    // identical results as the 8x16 path above.
    const bool has_high = nr > 8;
    for (int64_t r = 0; r < mr; ++r) {
        float* cr = c + r * ldc;
        const __m256 old0 = _mm256_maskload_ps(cr + 0, m0);
        _mm256_maskstore_ps(cr + 0, m0, _mm256_add_ps(old0, acc0[r]));
        if (has_high) {
            const __m256 old1 = _mm256_maskload_ps(cr + 8, m1);
            _mm256_maskstore_ps(cr + 8, m1, _mm256_add_ps(old1, acc1[r]));
        }
    }
}

} // namespace pg::cpu
