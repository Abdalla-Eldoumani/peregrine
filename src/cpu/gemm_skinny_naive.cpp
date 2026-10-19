#include "cpu/gemm_skinny.hpp"

#include "cpu/gemm_small.hpp"
#include "cpu/threads.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pg::cpu {

// The scalar twin of gemm_skinny_avx2.cpp: the same team over the same slabs
// of the longer output dimension, each slab through gemm_small_ldc_naive. A
// scalar chain needs no transposition for n <= 4, and the chain per element is
// the same whichever way the work is cut.
template <typename T>
void gemm_skinny_naive(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c,
                       int64_t m, int64_t k, int64_t n) {
    if (m <= 0 || n <= 0) {
        return;
    }
    const bool by_rows = m >= n;
    const int64_t len = by_rows ? m : n;
    const int64_t slabs = (len + SKINNY_SLAB - 1) / SKINNY_SLAB;
    const int nt = team_size(static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k + 1),
                             SKINNY_SERIAL_BELOW, SKINNY_PER_THREAD, slabs);
    const unsigned csr = mxcsr();
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt) if(nt > 1)
#endif
    {
        const mxcsr_scope fp(csr);
        int64_t t = 0;
        int64_t team = 1;
#if defined(_OPENMP)
        t = omp_get_thread_num();
        team = omp_get_num_threads();
#endif
        const int64_t lo = slabs * t / team * SKINNY_SLAB;
        const int64_t end = slabs * (t + 1) / team * SKINNY_SLAB;
        const int64_t hi = end < len ? end : len;
        if (lo < hi && by_rows) {
            gemm_small_ldc_naive<T>(a + lo * rs_a, rs_a, cs_a, b, rs_b, cs_b, c + lo * n, n, hi - lo, k, n);
        } else if (lo < hi) {
            gemm_small_ldc_naive<T>(a, rs_a, cs_a, b + lo * cs_b, rs_b, cs_b, c + lo, n, m, k, hi - lo);
        }
    }
}

template void gemm_skinny_naive<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t);
template void gemm_skinny_naive<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t);

} // namespace pg::cpu
