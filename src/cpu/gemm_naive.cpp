#include "cpu/gemm_naive.hpp"

#include "cpu/threads.hpp"

#include <algorithm>
#include <cstring>

namespace pg::cpu {

template <typename T>
void gemm_naive(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
                T* c, int64_t m, int64_t k, int64_t n) {
    if (m == 0 || n == 0) {
        return;
    }
    std::memset(c, 0, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T));
    if (k == 0) {
        return; // NumPy semantics: (m, 0) @ (0, n) is an m x n zero matrix
    }

    // 64 rows of B at f64 is 64 * n * 8 bytes; for n <= 4096 the panel stays
    // inside L2 on Comet Lake, which is the machine this blocks for.
    constexpr int64_t KB = 64;

    const int nt = call_threads();
    const unsigned csr = mxcsr();
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt) if(nt > 1)
#endif
    {
        const mxcsr_scope fp(csr);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int64_t i = 0; i < m; ++i) {
            T* ci = c + i * n;
            for (int64_t k0 = 0; k0 < k; k0 += KB) {
                const int64_t k1 = std::min(k0 + KB, k);
                for (int64_t p = k0; p < k1; ++p) {
                    const T aval = a[i * rs_a + p * cs_a];
                    const T* bp = b + p * rs_b;
                    // The same arithmetic either way; the unit-stride form is
                    // the one the compiler vectorizes.
                    if (cs_b == 1) {
                        for (int64_t j = 0; j < n; ++j) {
                            ci[j] += aval * bp[j];
                        }
                    } else {
                        for (int64_t j = 0; j < n; ++j) {
                            ci[j] += aval * bp[j * cs_b];
                        }
                    }
                }
            }
        }
    }
}

template void gemm_naive<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t,
                                float*, int64_t, int64_t, int64_t);
template void gemm_naive<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t,
                                 double*, int64_t, int64_t, int64_t);

} // namespace pg::cpu
