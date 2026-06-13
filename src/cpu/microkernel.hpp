#pragma once
#include <cstdint>

namespace fme::cpu {

// One concrete microkernel per dtype and register-tile shape, not a template:
// each definition lives in its own arch-flagged translation unit
// (microkernel_avx2_f64.cpp, microkernel_avx2_f32.cpp), so plain extern
// declarations cross the TU boundary and no extern-template machinery is needed.
// Each computes a full MR x NR accumulator tile from packed, zero-padded panels;
// the live extents mr and nr drive the masked C store so a partial edge tile
// writes only its real columns and rows and never touches memory past the tile.
// The f32 shapes 8x16 and 6x16 are both declared because CPU-03 picks the winner
// by measurement at n=512; the loser is removed once the bench decides. No
// intrinsics appear here: <immintrin.h> belongs only in the flagged .cpp files,
// or AVX2 codegen would leak into the fallback path and recreate the legacy
// import crash on CPUs without AVX2.
void microkernel_f64_6x8(const double* ap, const double* bp, double* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr);

void microkernel_f32_8x16(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr);

void microkernel_f32_6x16(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr);

} // namespace fme::cpu
