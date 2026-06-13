#include "dispatch/dispatch.hpp"
#include "cpu/feature_detect.hpp"
#include "cpu/gemm_blis.hpp"
#include "cpu/gemm_naive.hpp"
#include "cpu/reduce.hpp"
#include "cpu/transpose.hpp"

#if defined(FME_HAS_CUDA)
namespace fme::cuda {
// Forward-declared, not #included: gemm_cublas.cuh is itself CUDA-free, but the
// forward-declare matches the binding discipline (04-03/04-04) and keeps this TU
// from depending on any src/cuda header path. The explicit float/double
// instantiations in gemm_cublas.cu satisfy these. gemm runs the PURE device GEMM
// on the compute stream (no host staging, no sync).
template <typename T>
void gemm(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);
} // namespace fme::cuda
#endif

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

#if defined(FME_HAS_CUDA)
template <typename T>
void matmul_device(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n) {
    // Device-in/device-out: a, b, c are device pointers, so this is a pure
    // forward to the cuBLAS GEMM on the compute stream -- no host staging, no
    // sync (the binding owns the result's lifetime; from_device syncs when it
    // crosses back to host). f32 and f64 both forward to gemm<T>: f64 is the
    // FORCED device-resident path (both operands were explicitly to_device'd),
    // which is allowed and correct, just slow. The f64-never-AUTO-routes
    // exclusion (CLAUDE.md rule 2) is NOT enforced by rejecting f64 here -- a
    // forced f64 must compute -- but UPSTREAM, by the wrapper/binding never
    // selecting this device path for a host f64 array. Auto-routing a host f64
    // to the GPU would be the rule-2 bug; routing an already-device-resident f64
    // is the user's explicit choice. Pure: no warnings/logging/fallback.
    cuda::gemm<T>(a, b, c, m, k, n);
}

template void matmul_device<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
template void matmul_device<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);
#endif

// transpose and the two sum entries have one CPU implementation each, valid on
// every CPU, so they forward directly. No avx2/naive branch: unlike matmul
// there is no packed variant to choose, and the kernels are plain TUs the
// fallback path links unconditionally. The decision stays pure: no features
// are read, no state mutated, the call simply crosses the TU boundary.
template <typename T>
void transpose(const T* a, T* out, int64_t m, int64_t n) {
    cpu::transpose<T>(a, out, m, n);
}

template <typename T>
T sum_all(const T* a, int64_t m, int64_t n) {
    return cpu::sum_all<T>(a, m, n);
}

template <typename T>
void sum_axis(const T* a, T* out, int64_t m, int64_t n, int axis) {
    cpu::sum_axis<T>(a, out, m, n, axis);
}

template void transpose<float>(const float*, float*, int64_t, int64_t);
template void transpose<double>(const double*, double*, int64_t, int64_t);
template float sum_all<float>(const float*, int64_t, int64_t);
template double sum_all<double>(const double*, int64_t, int64_t);
template void sum_axis<float>(const float*, float*, int64_t, int64_t, int);
template void sum_axis<double>(const double*, double*, int64_t, int64_t, int);

} // namespace fme::dispatch
