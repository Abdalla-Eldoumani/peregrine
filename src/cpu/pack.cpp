#include "cpu/pack.hpp"

#include <immintrin.h>

namespace fme::cpu {

// Wave 1 skeleton: the bodies are empty so the source list links before the
// Wave 2 packing code exists; dispatch does not route to the blis path yet, so
// nothing calls these. The intrinsic header is included and exercised by one
// discarded vector op below so this translation unit genuinely depends on the
// per-file AVX2 flag: if that flag is ever dropped, the GCC build fails here
// instead of silently producing a binary that faults on a non-AVX2 CPU.
namespace {
[[maybe_unused]] void touch_avx2() {
    volatile __m256d z = _mm256_setzero_pd();
    (void)z;
}
} // namespace

template <typename T>
void pack_a(T* ap, const T* a, int64_t ic, int64_t pc, int64_t mc, int64_t kc, int64_t k) {
    (void)ap; (void)a; (void)ic; (void)pc; (void)mc; (void)kc; (void)k;
    // Wave 2 fills the MC x KC zero-padded copy.
}

template <typename T>
void pack_b(T* bp, const T* b, int64_t pc, int64_t jc, int64_t kc, int64_t nc, int64_t n) {
    (void)bp; (void)b; (void)pc; (void)jc; (void)kc; (void)nc; (void)n;
    // Wave 2 fills the KC x NC zero-padded copy.
}

template void pack_a<float>(float*, const float*, int64_t, int64_t, int64_t, int64_t, int64_t);
template void pack_a<double>(double*, const double*, int64_t, int64_t, int64_t, int64_t, int64_t);
template void pack_b<float>(float*, const float*, int64_t, int64_t, int64_t, int64_t, int64_t);
template void pack_b<double>(double*, const double*, int64_t, int64_t, int64_t, int64_t, int64_t);

} // namespace fme::cpu
