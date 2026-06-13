#include "cpu/gemm_blis.hpp"

#include <cstring>

#include <immintrin.h>

namespace fme::cpu {

blocking& current_blocking() {
    // Starting grid from the cache budget: KC=256 puts a 256x8 f64 Bp panel at
    // 16KB (half of L1d); MC=72 puts a 72x256 f64 Ap block at 144KB (just over
    // half of 256K L2); NC=4080 puts a 256x4080 f64 Bp block at 8.2MB of the
    // 12M shared L3. Runtime-settable so the sweep walks
    // MC{48,72,96,144} x KC{192,256,320} x NC{2048,4080} without rebuilding.
    static blocking b{72, 256, 4080};
    return b;
}

template <typename T>
void gemm_blis(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n) {
    (void)a; (void)b; (void)k;
    // Wave 1 placeholder: zero C and return. Dispatch does not route here until
    // Wave 2 (03-03), so existing matmul still runs the naive kernel and the
    // suite stays green; forwarding to gemm_naive is deliberately avoided
    // because pulling its header into this AVX2-flagged TU would defeat the
    // isolation this plan establishes. One discarded vector op keeps the AVX2
    // dependency real so the GCC build catches a dropped flag.
    if (m == 0 || n == 0) {
        return;
    }
    volatile __m256d z = _mm256_setzero_pd();
    (void)z;
    std::memset(c, 0, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T));
    // Wave 2 fills the five-loop body (jc/pc/ic + pack + microkernel).
}

template void gemm_blis<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
template void gemm_blis<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

} // namespace fme::cpu
