#include "cpu/gemm_skinny.hpp"

#include "cpu/gemm_small.hpp"
#include "cpu/threads.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pg::cpu {

namespace {

// C rows [i0, i1) when n <= 4 < m and A's columns are unit-stride. Each block
// solves C^T = B^T A^T, so the vectors run down A's columns in place, into a
// scratch tile whose rows are C's columns, and copies it back. The tile kernel
// computes the same chain per element either way round, so the bits do not
// change. A row-major A stays on the direct path, which broadcasts its rows:
// gathering its columns cost more (single-call diagnostic, f64 (4096, 4096, 1):
// 22.9 ms transposed against 8.2 ms direct).
template <typename T>
void transposed_rows(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c,
                     int64_t i0, int64_t i1, int64_t k, int64_t n) {
    constexpr int64_t IB = 256;
    alignas(32) T ct[4 * IB];
    for (int64_t i = i0; i < i1; i += IB) {
        const int64_t ib = i1 - i < IB ? i1 - i : IB;
        gemm_small_ldc<T>(b, cs_b, rs_b, a + i * rs_a, cs_a, rs_a, ct, IB, n, k, ib);
        for (int64_t r = 0; r < ib; ++r) {
            for (int64_t j = 0; j < n; ++j) {
                c[(i + r) * n + j] = ct[j * IB + r];
            }
        }
    }
}

// Thread t of team computes its share of whole slabs of the long dimension.
template <typename T>
void slab(int64_t t, int64_t team, bool transposed, bool by_rows, const T* a, int64_t rs_a, int64_t cs_a,
          const T* b, int64_t rs_b, int64_t cs_b, T* c, int64_t m, int64_t k, int64_t n) {
    const int64_t len = by_rows ? m : n;
    const int64_t slabs = (len + SKINNY_SLAB - 1) / SKINNY_SLAB;
    const int64_t lo = slabs * t / team * SKINNY_SLAB;
    const int64_t end = slabs * (t + 1) / team * SKINNY_SLAB;
    const int64_t hi = end < len ? end : len;
    if (lo >= hi) {
        return;
    }
    if (transposed) {
        transposed_rows<T>(a, rs_a, cs_a, b, rs_b, cs_b, c, lo, hi, k, n);
    } else if (by_rows) {
        gemm_small_ldc<T>(a + lo * rs_a, rs_a, cs_a, b, rs_b, cs_b, c + lo * n, n, hi - lo, k, n);
    } else {
        gemm_small_ldc<T>(a, rs_a, cs_a, b + lo * cs_b, rs_b, cs_b, c + lo, n, m, k, hi - lo);
    }
}

} // namespace

template <typename T>
void gemm_skinny(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b, T* c, int64_t m,
                 int64_t k, int64_t n) {
    if (m <= 0 || n <= 0) {
        return;
    }
    const bool transposed = n <= 4 && m > 4 && rs_a == 1;
    const bool by_rows = m >= n;
    const int64_t len = by_rows ? m : n;
    const int nt = team_size(static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k + 1),
                             SKINNY_SERIAL_BELOW, SKINNY_PER_THREAD, (len + SKINNY_SLAB - 1) / SKINNY_SLAB);
    const unsigned csr = mxcsr();
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt) if(nt > 1)
#endif
    {
        const mxcsr_scope fp(csr);
#if defined(_OPENMP)
        slab<T>(omp_get_thread_num(), omp_get_num_threads(), transposed, by_rows, a, rs_a, cs_a, b, rs_b, cs_b, c,
                m, k, n);
#else
        slab<T>(0, 1, transposed, by_rows, a, rs_a, cs_a, b, rs_b, cs_b, c, m, k, n);
#endif
    }
}

template void gemm_skinny<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t, float*, int64_t, int64_t, int64_t);
template void gemm_skinny<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t, double*, int64_t, int64_t, int64_t);

} // namespace pg::cpu
