#pragma once
#include <cstdint>

namespace pg::cpu {

// One concrete microkernel per dtype and register-tile shape, not a template:
// each definition lives in its own arch-flagged translation unit
// (microkernel_avx2_f64.cpp, microkernel_avx2_f32.cpp), so plain extern
// declarations cross the TU boundary and no extern-template machinery is needed.
// Each computes a full MR x NR accumulator tile from packed, zero-padded panels;
// the live extents mr and nr drive the masked C store so a partial edge tile
// writes only its real columns and rows and never touches memory past the tile.
//
// first marks the first k panel of a C tile: the kernel then stores acc + 0.0
// without reading C, so a fresh output is never zeroed or loaded. The + 0.0 is
// what memset-then-add did, and it keeps the bits: an accumulator that
// underflowed to -0.0 still lands as +0.0. Later panels store C + acc.
//
// b_next is where the call prefetches B for a later call: kc * 8 bytes from
// b_next go into L2 over the k loop (one line per unrolled step group), an
// eighth of a packed B micro-panel, whose kc rows are 64 bytes for both dtypes.
// The address only steers a prefetch, so it never changes a result; it must
// point into the packed B buffer with an eighth of a panel after it.
//
// f32 runs 6x16 (the f64 6x8 register budget, 12 accumulators): measured
// single-thread at 0.97-0.98 of OpenBLAS at n = 1024 and 2048, where 8x16 (all
// 16 ymm as accumulators, so the k loop spills) reached 0.67-0.70, so 8x16 was
// retired. No intrinsics appear here: the AVX2 SIMD
// header is included only in the flagged .cpp files, or AVX2 codegen would leak
// into the fallback path and recreate the legacy import crash on CPUs without
// AVX2.
void microkernel_f64_6x8(const double* ap, const double* bp, double* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr, bool first,
                         const double* b_next);

void microkernel_f32_6x16(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr, bool first,
                          const float* b_next);

} // namespace pg::cpu
