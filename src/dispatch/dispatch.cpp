#include "dispatch/dispatch.hpp"
#include "cpu/feature_detect.hpp"
#include "cpu/fused.hpp"
#include "cpu/gemm_blis.hpp"
#include "cpu/gemm_naive.hpp"
#include "cpu/gemm_skinny.hpp"
#include "cpu/gemm_small.hpp"
#include "cpu/reduce.hpp"
#include "cpu/transpose.hpp"

#include <algorithm>

#if defined(PG_HAS_CUDA)
namespace pg::cuda {
// Forward-declared, not #included: gemm_cublas.cuh is itself CUDA-free, but the
// forward-declare matches the binding discipline and keeps this TU from depending
// on any src/cuda header path. The explicit float/double instantiations in
// gemm_cublas.cu satisfy these. gemm runs the PURE device GEMM on the compute
// stream (no host staging, no sync).
template <typename T>
void gemm(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

// The device-resident fused kernels, forward-declared CUDA-free the same way:
// fused.cu's explicit float/double instantiations satisfy these, and this TU
// pulls no src/cuda header so the deletable-src/cuda invariant holds. Pure
// device-pointer compute on the compute stream.
template <typename T>
void fused_axpby(const T* x, const T* y, T* out, int64_t n, T a, T b);
template <typename T>
void fused_fma3(const T* x, const T* y, const T* z, T* out, int64_t n);
template <typename T>
void fused_scaled_relu(const T* x, T* out, int64_t n, T scale);
} // namespace pg::cuda
#endif

namespace pg::dispatch {

// Below this max dimension packing and a team cost more than they save, so the
// register-tiled direct kernel wins (1.3-5x the packed path's time at n = 16 to
// 96, single-call diagnostics); it still wins at 128-160 on one thread, but the
// threaded packed path overtakes it near 160, so 96 holds until measured.
constexpr int64_t SMALL_MAX_DIM = 96;
// A product with an output dimension or k this thin is bound by reading the
// long operand, so the skinny kernels stream it instead of packing it.
constexpr int64_t SKINNY_MAX_DIM = 4;

gemm_path choose_gemm_path(int64_t m, int64_t k, int64_t n, bool avx2_fma) noexcept {
    if (!avx2_fma) {
        return gemm_path::naive;
    }
    if (std::max(m, std::max(k, n)) <= SMALL_MAX_DIM) {
        return gemm_path::small;
    }
    if (std::min(m, n) <= SKINNY_MAX_DIM || k <= SKINNY_MAX_DIM) {
        return gemm_path::skinny;
    }
    return gemm_path::packed;
}

template <typename T>
void matmul(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
            T* c, int64_t m, int64_t k, int64_t n) {
    // NumPy gives a length-1 dim any stride (it never moves the pointer, and the
    // contiguity and alignment flags ignore it). Unit stride reads the same
    // elements and lets the kernels take a contiguous reading.
    if (m == 1) {
        rs_a = 1;
    }
    if (k == 1) {
        cs_a = 1;
        rs_b = 1;
    }
    if (n == 1) {
        cs_b = 1;
    }
    // The features are memoized at import and already fold in
    // PEREGRINE_DISABLE_AVX2, so a forced fallback routes here exactly as a CPU
    // without AVX2 would. The kernels share the signature, so the call crosses
    // the TU boundary with no adapter and no AVX2 codegen leaking into this
    // fallback-linked unit.
    const auto& f = cpu::detect();
    switch (choose_gemm_path(m, k, n, f.avx2 && f.fma)) {
    case gemm_path::packed:
        cpu::gemm_blis<T>(a, rs_a, cs_a, b, rs_b, cs_b, c, m, k, n);
        return;
    case gemm_path::small:
        cpu::gemm_small<T>(a, rs_a, cs_a, b, rs_b, cs_b, c, m, k, n);
        return;
    case gemm_path::skinny:
        cpu::gemm_skinny<T>(a, rs_a, cs_a, b, rs_b, cs_b, c, m, k, n);
        return;
    case gemm_path::naive:
        break;
    }
    cpu::gemm_naive<T>(a, rs_a, cs_a, b, rs_b, cs_b, c, m, k, n);
}

template void matmul<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t,
                            float*, int64_t, int64_t, int64_t);
template void matmul<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t,
                             double*, int64_t, int64_t, int64_t);

#if defined(PG_HAS_CUDA)
template <typename T>
void matmul_device(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n) {
    // Device-in/device-out: a, b, c are device pointers, so this is a pure
    // forward to the cuBLAS GEMM on the compute stream -- no host staging, no
    // sync (the binding owns the result's lifetime; from_device syncs when it
    // crosses back to host). f32 and f64 both forward to gemm<T>: f64 is the
    // FORCED device-resident path (both operands were explicitly to_device'd),
    // which is allowed and correct, just slow. The f64-never-AUTO-routes exclusion
    // (float64 GEMM never auto-routes to this GPU because GA106 FP64 is 1/64 FP32)
    // is NOT enforced by rejecting f64 here -- a forced f64 must compute -- but
    // UPSTREAM, by the wrapper/binding never selecting this device path for a host
    // f64 array. Auto-routing a host f64 to the GPU would be the bug; routing an
    // already-device-resident f64 is the user's explicit choice. Pure: no
    // warnings/logging/fallback.
    cuda::gemm<T>(a, b, c, m, k, n);
}

template void matmul_device<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
template void matmul_device<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);
#endif

// Fused elementwise host entries. Same branch as matmul (the features are
// memoized at import and fold in PEREGRINE_DISABLE_AVX2, so a forced fallback routes
// here exactly as a CPU without AVX2 would): the AVX2 body when the CPU has
// avx2+fma, the scalar fallback otherwise. The AVX2 and naive bodies share each
// op's signature, so the call crosses the TU boundary with no adapter and no
// AVX2 codegen leaking into this fallback-linked unit. Pure features-in/path-out,
// no side effects. There is no f64 exclusion -- fused never auto-routes to the
// GPU, so both dtypes take the CPU branch here unconditionally.
template <typename T>
void fused_axpby(const T* x, const T* y, T* out, int64_t n, T a, T b) {
    const auto& f = cpu::detect();
    if (f.avx2 && f.fma) {
        cpu::fused_axpby<T>(x, y, out, n, a, b);
        return;
    }
    cpu::fused_axpby_naive<T>(x, y, out, n, a, b);
}

template <typename T>
void fused_fma3(const T* x, const T* y, const T* z, T* out, int64_t n) {
    const auto& f = cpu::detect();
    if (f.avx2 && f.fma) {
        cpu::fused_fma3<T>(x, y, z, out, n);
        return;
    }
    cpu::fused_fma3_naive<T>(x, y, z, out, n);
}

template <typename T>
void fused_scaled_relu(const T* x, T* out, int64_t n, T scale) {
    const auto& f = cpu::detect();
    if (f.avx2 && f.fma) {
        cpu::fused_scaled_relu<T>(x, out, n, scale);
        return;
    }
    cpu::fused_scaled_relu_naive<T>(x, out, n, scale);
}

template void fused_axpby<float>(const float*, const float*, float*, int64_t, float, float);
template void fused_axpby<double>(const double*, const double*, double*, int64_t, double, double);
template void fused_fma3<float>(const float*, const float*, const float*, float*, int64_t);
template void fused_fma3<double>(const double*, const double*, const double*, double*, int64_t);
template void fused_scaled_relu<float>(const float*, float*, int64_t, float);
template void fused_scaled_relu<double>(const double*, double*, int64_t, double);

#if defined(PG_HAS_CUDA)
// Device-resident fused entries: x, y, z, out are device pointers, so each is a
// pure forward to the hand-written grid-stride kernel on the compute stream -- no
// host staging, no sync (the binding owns the output buffer's lifetime; the
// output alloc and the kernel are both on the compute stream, so the single-stream
// ordering needs no cross-stream fence, unlike matmul_device's transfer/compute
// split). Both f32 and f64 forward through: the fused device path is reached only
// when operands are already a pg.Array (the user's explicit to_device), so there
// is no f64-never-AUTO-routes exclusion to enforce here -- that is a GEMM-only rule
// and fused never auto-stages a host array. Pure: no warnings/logging/fallback.
template <typename T>
void fused_axpby_device(const T* x, const T* y, T* out, int64_t n, T a, T b) {
    cuda::fused_axpby<T>(x, y, out, n, a, b);
}

template <typename T>
void fused_fma3_device(const T* x, const T* y, const T* z, T* out, int64_t n) {
    cuda::fused_fma3<T>(x, y, z, out, n);
}

template <typename T>
void fused_scaled_relu_device(const T* x, T* out, int64_t n, T scale) {
    cuda::fused_scaled_relu<T>(x, out, n, scale);
}

template void fused_axpby_device<float>(const float*, const float*, float*, int64_t, float, float);
template void fused_axpby_device<double>(const double*, const double*, double*, int64_t, double, double);
template void fused_fma3_device<float>(const float*, const float*, const float*, float*, int64_t);
template void fused_fma3_device<double>(const double*, const double*, const double*, double*, int64_t);
template void fused_scaled_relu_device<float>(const float*, float*, int64_t, float);
template void fused_scaled_relu_device<double>(const double*, double*, int64_t, double);
#endif

// transpose takes the AVX2 body on avx2 && fma, the scalar twin otherwise; the
// result is a permutation of the input bits, so both give the same array.
template <typename T>
void transpose(const T* a, T* out, int64_t m, int64_t n) {
    const auto& f = cpu::detect();
    if (f.avx2 && f.fma) {
        cpu::transpose_avx2<T>(a, out, m, n);
        return;
    }
    cpu::transpose<T>(a, out, m, n);
}

// The sums take the AVX2 leaf kernels on avx2 && fma, like the fused ops, and
// the scalar twins otherwise; both give the same bits.
template <typename T>
T sum_all(const T* a, int64_t m, int64_t n) {
    const auto& f = cpu::detect();
    return cpu::sum_all<T>(a, m, n, f.avx2 && f.fma);
}

template <typename T>
void sum_axis(const T* a, T* out, int64_t m, int64_t n, int axis) {
    const auto& f = cpu::detect();
    cpu::sum_axis<T>(a, out, m, n, axis, f.avx2 && f.fma, false);
}

template <typename T>
void mean_axis(const T* a, T* out, int64_t m, int64_t n, int axis) {
    const auto& f = cpu::detect();
    cpu::sum_axis<T>(a, out, m, n, axis, f.avx2 && f.fma, true);
}

template void transpose<float>(const float*, float*, int64_t, int64_t);
template void transpose<double>(const double*, double*, int64_t, int64_t);
template float sum_all<float>(const float*, int64_t, int64_t);
template double sum_all<double>(const double*, int64_t, int64_t);
template void sum_axis<float>(const float*, float*, int64_t, int64_t, int);
template void sum_axis<double>(const double*, double*, int64_t, int64_t, int);
template void mean_axis<float>(const float*, float*, int64_t, int64_t, int);
template void mean_axis<double>(const double*, double*, int64_t, int64_t, int);

} // namespace pg::dispatch
