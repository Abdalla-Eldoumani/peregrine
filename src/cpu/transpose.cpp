#include "cpu/transpose.hpp"

#include <algorithm>

namespace fme::cpu {

template <typename T>
void transpose(const T* a, T* out, int64_t m, int64_t n) {
    if (m == 0 || n == 0) {
        return; // either extent empty: out is the correctly-shaped empty array
    }

    // Cache-blocked so each BS-by-BS tile of out stays warm while its source
    // column is gathered. 32 elements is 256 B at f64, four cache lines per
    // tile side, which keeps both the source stripe and the destination stripe
    // inside L1 on the Comet Lake this targets. Plain scalar copy: a transpose
    // is pure data movement, memory-bandwidth bound, so the gather/scatter
    // pattern dominates and SIMD shuffles win nothing here.
    constexpr int64_t BS = 32;

    for (int64_t i0 = 0; i0 < m; i0 += BS) {
        const int64_t i1 = std::min(i0 + BS, m);
        for (int64_t j0 = 0; j0 < n; j0 += BS) {
            const int64_t j1 = std::min(j0 + BS, n);
            for (int64_t i = i0; i < i1; ++i) {
                const T* ai = a + i * n;
                for (int64_t j = j0; j < j1; ++j) {
                    out[j * m + i] = ai[j];
                }
            }
        }
    }
}

template void transpose<float>(const float*, float*, int64_t, int64_t);
template void transpose<double>(const double*, double*, int64_t, int64_t);

} // namespace fme::cpu
