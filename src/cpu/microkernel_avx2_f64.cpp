#include "cpu/microkernel.hpp"

#include <immintrin.h>

namespace fme::cpu {

// Wave 1 skeleton: empty body so the source list links before the Wave 2 6x8
// AVX2+FMA kernel lands. The intrinsic header is exercised by one discarded FMA
// below so this translation unit genuinely needs the per-file AVX2 flag; the
// GCC build hard-errors here if the flag is dropped, which is the leak detector
// MSVC cannot provide.
void microkernel_f64_6x8(const double* ap, const double* bp, double* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr) {
    (void)ap; (void)bp; (void)c; (void)kc; (void)ldc; (void)mr; (void)nr;
    volatile __m256d z = _mm256_fmadd_pd(_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd());
    (void)z;
    // Wave 2 fills the 12-accumulator register tile and the masked C-edge store.
}

} // namespace fme::cpu
