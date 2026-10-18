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
// f32 runs 6x16 (the f64 6x8 register budget, 12 accumulators). 8x16 stays
// compiled so the two shapes can be measured against each other; it needs all
// 16 ymm for accumulators and spills. No intrinsics appear here: the AVX2 SIMD
// header is included only in the flagged .cpp files, or AVX2 codegen would leak
// into the fallback path and recreate the legacy import crash on CPUs without
// AVX2.
void microkernel_f64_6x8(const double* ap, const double* bp, double* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr, bool first);

void microkernel_f32_6x16(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr, bool first);

void microkernel_f32_8x16(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr, bool first);

} // namespace pg::cpu
