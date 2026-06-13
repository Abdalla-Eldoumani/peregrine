#include "cpu/microkernel.hpp"

#include <immintrin.h>

namespace fme::cpu {

// Wave 1 skeleton: both f32 candidate shapes are empty stubs so the source list
// links before CPU-03 fills and benchmarks them at n=512 to pick a winner. The
// intrinsic header is exercised by one discarded f32 FMA below so this
// translation unit genuinely needs the per-file AVX2 flag; the GCC build
// hard-errors here if the flag is dropped.
namespace {
[[maybe_unused]] void touch_avx2() {
    volatile __m256 z = _mm256_fmadd_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
    (void)z;
}
} // namespace

void microkernel_f32_8x16(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr) {
    (void)ap; (void)bp; (void)c; (void)kc; (void)ldc; (void)mr; (void)nr;
    // Wave 2 fills the 8x16 tile; CPU-03 decides 8x16 vs 6x16 by measurement.
}

void microkernel_f32_6x16(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr) {
    (void)ap; (void)bp; (void)c; (void)kc; (void)ldc; (void)mr; (void)nr;
    // Wave 2 fills the 6x16 tile; CPU-03 decides 8x16 vs 6x16 by measurement.
}

} // namespace fme::cpu
