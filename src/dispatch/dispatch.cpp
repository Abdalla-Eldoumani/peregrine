#include "dispatch/dispatch.hpp"
#include "cpu/feature_detect.hpp"
#include "cpu/gemm_blis.hpp"
#include "cpu/gemm_naive.hpp"

namespace fme::dispatch {

template <typename T>
void matmul(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n) {
    // The features are memoized at import and already fold in FME_DISABLE_AVX2,
    // so a forced fallback routes here exactly as a CPU without AVX2 would. The
    // packed kernel owns its own small-matrix branch, so this stays a pure
    // features-in/path-out decision with no side effects. gemm_blis and
    // gemm_naive share the signature, so the call crosses the TU boundary with
    // no adapter and no AVX2 codegen leaking into this fallback-linked unit.
    const auto& f = cpu::detect();
    if (f.avx2 && f.fma) {
        cpu::gemm_blis<T>(a, b, c, m, k, n);
        return;
    }
    cpu::gemm_naive<T>(a, b, c, m, k, n);
}

template void matmul<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
template void matmul<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

} // namespace fme::dispatch
