#include "cpu/pack.hpp"

namespace fme::cpu {

// MR and NR are the register-tile shape the microkernel computes (6 rows x 8
// columns of f64 = 12 ymm accumulators). They are structural, not tuning
// constants, so they live here and in the microkernel as compile-time values;
// the packed panels are laid out to match exactly what the microkernel reads.
namespace {
constexpr int64_t MR = 6;
constexpr int64_t NR = 8;
} // namespace

// pack_a copies the live mc x kc block of A (row-major, leading dimension k,
// top-left at A[ic, pc]) into Ap as a sequence of MR-row panels. Within a panel
// the kc steps are contiguous and each step holds MR row values, so the
// microkernel reads one k-step's 6 A scalars with stride 1 and broadcasts them.
// Rows beyond mc inside the last panel and any unused tail are written 0.0 so
// the microkernel computes a full MR x kc strip with no branch on the row count;
// the padded rows contribute zero to the accumulators. Every index is int64_t;
// size_t appears only at the byte-level fill.
template <typename T>
void pack_a(T* ap, const T* a, int64_t ic, int64_t pc, int64_t mc, int64_t kc, int64_t k) {
    const T* a_block = a + ic * k + pc;
    int64_t dst = 0;
    for (int64_t i0 = 0; i0 < mc; i0 += MR) {
        const int64_t rows = (mc - i0 < MR) ? (mc - i0) : MR;
        for (int64_t p = 0; p < kc; ++p) {
            for (int64_t r = 0; r < rows; ++r) {
                ap[dst++] = a_block[(i0 + r) * k + p];
            }
            // Zero-pad the rows past the live count so this k-step still fills a
            // full MR-wide slot; the microkernel multiplies these by packed-zero
            // B lanes or simply does not store them, but the layout must stay
            // MR-strided for the next k-step to land where the kernel expects.
            for (int64_t r = rows; r < MR; ++r) {
                ap[dst++] = T{0};
            }
        }
    }
}

// pack_b copies the live kc x nc block of B (row-major, leading dimension n,
// top-left at B[pc, jc]) into Bp as a sequence of NR-column panels. Within a
// panel the kc steps are contiguous and each step holds NR column values, so the
// microkernel loads one k-step's 8 B values as two aligned ymm vectors with
// stride 1. Columns beyond nc inside the last panel are written 0.0 so the
// microkernel computes a full kc x NR slice with no branch on the column count;
// the padded columns contribute zero to the accumulators and are never stored.
template <typename T>
void pack_b(T* bp, const T* b, int64_t pc, int64_t jc, int64_t kc, int64_t nc, int64_t n) {
    const T* b_block = b + pc * n + jc;
    int64_t dst = 0;
    for (int64_t j0 = 0; j0 < nc; j0 += NR) {
        const int64_t cols = (nc - j0 < NR) ? (nc - j0) : NR;
        for (int64_t p = 0; p < kc; ++p) {
            for (int64_t c = 0; c < cols; ++c) {
                bp[dst++] = b_block[p * n + (j0 + c)];
            }
            for (int64_t c = cols; c < NR; ++c) {
                bp[dst++] = T{0};
            }
        }
    }
}

template void pack_a<float>(float*, const float*, int64_t, int64_t, int64_t, int64_t, int64_t);
template void pack_a<double>(double*, const double*, int64_t, int64_t, int64_t, int64_t, int64_t);
template void pack_b<float>(float*, const float*, int64_t, int64_t, int64_t, int64_t, int64_t);
template void pack_b<double>(double*, const double*, int64_t, int64_t, int64_t, int64_t, int64_t);

} // namespace fme::cpu
