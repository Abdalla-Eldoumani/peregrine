#include "cpu/microkernel.hpp"

#include <immintrin.h>

namespace pg::cpu {

// The f64 register tile: MR=6 rows by NR=8 columns, held in 12 ymm accumulators
// (each row is two __m256d of 4 doubles). Per k-step the kernel loads the 8 B
// values for that step as two vectors, broadcasts the 6 A values, and issues 12
// fused multiply-adds. The packed panels are zero-padded to MR/NR by pack_a and
// pack_b, so the k loop runs to kc with no branch and the padded lanes add zero.
// _mm256_fmadd_pd is a real single-rounded FMA on both MSVC (/fp:precise) and
// GCC; neither contracts a split mul+add, so the numerics are exactly one
// rounding per accumulation step, which is what the tolerance contract assumes.
//
// The C store is the only place tile extent matters, and it is exactly where the
// legacy kernel corrupted memory: a partial tile (mr < 6 or nr < 8) must write
// only its live mr x nr cells and leave every byte past the tile untouched. The
// full tile uses plain loads/stores; the edge tile uses VMASKMOV with a mask
// built by integer compare. VMASKMOV reads zero in masked-off lanes and writes
// nothing there, and the masked-off access is fault-suppressed even across an
// unmapped guard page, so a tile flush against the very end of the C allocation
// is safe. The caller zeroes C once and the kernel always accumulates (+=), so
// the pc (k-block) loop sums correctly across blocks with a single store variant.
void microkernel_f64_6x8(const double* ap, const double* bp, double* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr) {
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

    for (int64_t p = 0; p < kc; ++p) {
        const __m256d b0 = _mm256_loadu_pd(bp + 0);
        const __m256d b1 = _mm256_loadu_pd(bp + 4);
        const __m256d a0 = _mm256_set1_pd(ap[0]);
        const __m256d a1 = _mm256_set1_pd(ap[1]);
        const __m256d a2 = _mm256_set1_pd(ap[2]);
        const __m256d a3 = _mm256_set1_pd(ap[3]);
        const __m256d a4 = _mm256_set1_pd(ap[4]);
        const __m256d a5 = _mm256_set1_pd(ap[5]);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c11 = _mm256_fmadd_pd(a1, b1, c11);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c21 = _mm256_fmadd_pd(a2, b1, c21);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c31 = _mm256_fmadd_pd(a3, b1, c31);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c41 = _mm256_fmadd_pd(a4, b1, c41);
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
        ap += 6;
        bp += 8;
    }

    const __m256d acc0[6] = {c00, c10, c20, c30, c40, c50};
    const __m256d acc1[6] = {c01, c11, c21, c31, c41, c51};

    if (mr == 6 && nr == 8) {
        // Full tile: both vectors of every row store unconditionally.
        for (int64_t r = 0; r < 6; ++r) {
            double* cr = c + r * ldc;
            _mm256_storeu_pd(cr + 0, _mm256_add_pd(_mm256_loadu_pd(cr + 0), acc0[r]));
            _mm256_storeu_pd(cr + 4, _mm256_add_pd(_mm256_loadu_pd(cr + 4), acc1[r]));
        }
        return;
    }

    // Edge tile: build a lane mask for the low (cols 0..3) and high (cols 4..7)
    // vectors from the live column count nr in 0..8. _mm256_cmpgt_epi64 sets a
    // lane to all-ones when nr exceeds that lane's column index, which is exactly
    // the live-column predicate; the same construction is correct at every tail
    // width 0..8 including a full-low/empty-high split. Only the live mr rows are
    // visited, so padded rows are never stored.
    const __m256i idx0 = _mm256_setr_epi64x(0, 1, 2, 3);
    const __m256i idx1 = _mm256_setr_epi64x(4, 5, 6, 7);
    const __m256i nrv = _mm256_set1_epi64x(nr);
    const __m256i m0 = _mm256_cmpgt_epi64(nrv, idx0);
    const __m256i m1 = _mm256_cmpgt_epi64(nrv, idx1);
    // The high vector (columns 4..7) is touched only when nr > 4. When nr <= 4 its
    // mask m1 is all-zero, so the masked store there writes nothing -- but the
    // pointer cr + 4 would still be FORMED, and on the last row of the rightmost
    // tile (jc+jr)+4 can exceed n, making cr + 4 more than one-past-the-end of the
    // m*n C object. Forming such a pointer is C++ UB ([expr.add]) even unread, so
    // guard the whole high half on nr > 4: the formation happens only when at least
    // one high-half column is live and cr + 4 is in bounds. Results are identical
    // (the guarded-out case stored nothing anyway).
    const bool has_high = nr > 4;
    for (int64_t r = 0; r < mr; ++r) {
        double* cr = c + r * ldc;
        const __m256d old0 = _mm256_maskload_pd(cr + 0, m0);
        _mm256_maskstore_pd(cr + 0, m0, _mm256_add_pd(old0, acc0[r]));
        if (has_high) {
            const __m256d old1 = _mm256_maskload_pd(cr + 4, m1);
            _mm256_maskstore_pd(cr + 4, m1, _mm256_add_pd(old1, acc1[r]));
        }
    }
}

} // namespace pg::cpu
