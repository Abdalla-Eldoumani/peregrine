#include "cuda/gemm_cublas.cuh"

#include "cuda/check.cuh"
#include "cuda/context.cuh"

#include <cstdint>

namespace fme::cuda {
namespace {

// cuBLAS dispatched by operand type. The two overloads let the templated gemm
// below stay dtype-generic while each calls the correct precision entry; alpha
// and beta are passed in the operand type (a float* for Sgemm, double* for
// Dgemm), as the v2 API requires.
cublasStatus_t cublas_gemm_nn(cublasHandle_t h, int m, int n, int k,
                              const float* alpha, const float* A, int lda,
                              const float* B, int ldb, const float* beta,
                              float* C, int ldc) {
    return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B,
                       ldb, beta, C, ldc);
}

cublasStatus_t cublas_gemm_nn(cublasHandle_t h, int m, int n, int k,
                              const double* alpha, const double* A, int lda,
                              const double* B, int ldb, const double* beta,
                              double* C, int ldc) {
    return cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B,
                       ldb, beta, C, ldc);
}

} // namespace

template <typename T>
void gemm(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n) {
    // Zero-dim guards FIRST, before any allocation or launch, matching
    // gemm_naive and NumPy: an empty GEMM must never reach cuBLAS (cuBLAS with a
    // zero m/n/k is undefined territory we have no reason to enter). (m,0)@(0,n)
    // and the m==0/n==0 cases produce no output elements to compute; (0,k)@(k,n)
    // has zero rows. The k==0 case is an m x n matrix of exact zeros (NumPy
    // semantics): the caller's C buffer is device memory, so zero it on the
    // compute stream rather than memset on host.
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        FME_CUDA_CHECK(cudaMemsetAsync(
            c, 0, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T),
            context().compute));
        return;
    }

    // Borrow the singleton's handle and stream (04-02): no per-call
    // cublasCreate. The handle is already bound to the compute stream and
    // already carries the math mode decided once at init from FME_ALLOW_TF32
    // (DEFAULT_MATH off by default). We do NOT call cublasSetMathMode per GEMM:
    // the mode is fixed for the session so it cannot drift into a result that
    // violates the tolerance contract. TF32's 10-bit mantissa measured ~830x
    // worse abs error than DEFAULT_MATH at n=512, which is exactly why it is
    // opt-in only and never compared against the standard contract.
    Context& ctx = context();

    // cuBLAS v2 takes int for m/n/k and the leading dimensions; we carry int64
    // internally per the src int64-index rule. This phase's sizes are at most a
    // few thousand per dimension (GPU-08 is n=2048), far inside INT_MAX, and
    // huge-GEMM k-panel tiling is a later (Phase 5+) calibration concern, so the
    // narrowing cast here is safe. The guard makes the assumption explicit: if a
    // dimension ever exceeds int range it is a tiling bug, not a silent
    // out-of-bounds device access.
    if (m > INT32_MAX || k > INT32_MAX || n > INT32_MAX) {
        throw ::fme::cuda_error(
            "cuda gemm: a dimension exceeds the int32 cuBLAS limit; "
            "huge-GEMM tiling is not implemented at this phase");
    }
    const int mi = static_cast<int>(m);
    const int ki = static_cast<int>(k);
    const int ni = static_cast<int>(n);

    // The column-major no-transpose trick, VERIFIED bit-correct on this machine
    // (max abs diff 0.0 tiny / 1.14e-5 at n=512 vs the f64 CPU reference, inside
    // the f32 tolerance contract). NumPy A,B,C are row-major; cuBLAS is
    // column-major. Computing column-major C^T = B^T @ A^T writes exactly the
    // bytes of row-major C = A @ B, so swapping the operands (B first, then A)
    // and passing the swapped dims (N, M, K) yields the row-major result with no
    // CUBLAS_OP_T and no transpose kernel. Leading dimensions are the row-major
    // row strides: B is (k x n) so ldb_for_swapped = N, A is (m x k) so its
    // stride = K, and C is (m x n) so ldc = N.
    const T alpha = static_cast<T>(1);
    const T beta = static_cast<T>(0);
    FME_CUBLAS_CHECK(cublas_gemm_nn(ctx.cublas, ni, mi, ki, &alpha,
                                    /*A=*/b, /*lda=*/ni,
                                    /*B=*/a, /*ldb=*/ki, &beta,
                                    /*C=*/c, /*ldc=*/ni));
}

template void gemm<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
template void gemm<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

template <typename T>
void gemm_host(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n) {
    // Empty output: nothing to stage or compute, matching gemm above and NumPy.
    // (m,0)@(0,n) with m,n>0 falls through to the k==0 zero-fill in gemm after
    // the buffers are allocated; the m==0/n==0 cases produce no elements.
    if (m == 0 || n == 0) {
        return;
    }

    Context& ctx = context();
    const cudaStream_t stream = ctx.compute;

    const size_t a_bytes = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(T);
    const size_t b_bytes = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(T);
    const size_t c_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T);

    // Stream-ordered allocation from the context mempool (release threshold
    // UINT64_MAX, set in 04-02), so repeated calls reuse buffers instead of
    // re-allocating from the OS. All three buffers, the copies, the compute, and
    // the free are ordered on the one compute stream, so no cross-stream event
    // is needed. A zero-k input still allocates dA/dB (size 0 is a valid
    // cudaMallocAsync) and dC, then gemm zero-fills dC.
    T* da = nullptr;
    T* db = nullptr;
    T* dc = nullptr;
    FME_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&da), a_bytes, stream));
    FME_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&db), b_bytes, stream));
    FME_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&dc), c_bytes, stream));

    if (a_bytes > 0) {
        FME_CUDA_CHECK(cudaMemcpyAsync(da, a, a_bytes, cudaMemcpyHostToDevice, stream));
    }
    if (b_bytes > 0) {
        FME_CUDA_CHECK(cudaMemcpyAsync(db, b, b_bytes, cudaMemcpyHostToDevice, stream));
    }

    gemm<T>(da, db, dc, m, k, n);

    FME_CUDA_CHECK(cudaMemcpyAsync(c, dc, c_bytes, cudaMemcpyDeviceToHost, stream));

    FME_CUDA_CHECK(cudaFreeAsync(da, stream));
    FME_CUDA_CHECK(cudaFreeAsync(db, stream));
    FME_CUDA_CHECK(cudaFreeAsync(dc, stream));

    // The D2H copy feeds the caller's host buffer, so sync before returning: the
    // "every D2H feeding a host-visible return syncs first" rule. cudaFreeAsync
    // is stream-ordered behind the copy, so the result is safely in c once the
    // stream drains.
    FME_CUDA_CHECK(cudaStreamSynchronize(stream));
}

template void gemm_host<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
template void gemm_host<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

template <typename T>
float time_matmul(const T* a, const T* b, int64_t m, int64_t k, int64_t n,
                  int reps, int warmups) {
    // The operands are already device-resident, so the timed region is the GEMM
    // and nothing else (no H2D/D2H). reps must be positive: the caller divides by
    // it, and a zero-rep timing is meaningless. An empty output (m==0 || n==0)
    // has no work to time and would make the per-rep cost ill-defined; reject it
    // rather than return a fake zero.
    if (reps <= 0) {
        throw ::fme::cuda_error("cuda time_matmul: reps must be positive");
    }
    if (m == 0 || n == 0) {
        throw ::fme::cuda_error(
            "cuda time_matmul: empty output has no work to time");
    }

    Context& ctx = context();
    const cudaStream_t stream = ctx.compute;
    const size_t c_bytes =
        static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T);

    // One output buffer from the context mempool on the compute stream, reused
    // across every warmup and timed rep (the inputs do not change, so the GEMM
    // overwrites C each iteration -- beta is 0 in gemm). Allocated once so no
    // allocation lands inside the timed region.
    T* dc = nullptr;
    FME_CUDA_CHECK(
        cudaMallocAsync(reinterpret_cast<void**>(&dc), c_bytes, stream));

    cudaEvent_t e0 = nullptr;
    cudaEvent_t e1 = nullptr;
    FME_CUDA_CHECK(cudaEventCreate(&e0));
    FME_CUDA_CHECK(cudaEventCreate(&e1));

    // Warm the clocks: the first GEMM on a cold GPU pays clock ramp and any lazy
    // cuBLAS init, so timing it would overstate the cost (RESEARCH/skill: warm
    // before timing, report warm). These run on the compute stream and are not
    // bracketed by the events.
    for (int i = 0; i < warmups; ++i) {
        gemm<T>(a, b, dc, m, k, n);
    }

    // Sync before the start event so the warmups are fully drained and e0 marks a
    // quiet stream; then the timed region is exactly `reps` GEMMs between the two
    // events on the same stream. cudaEventElapsedTime measures device time between
    // them, the only correct way to time an async launch.
    FME_CUDA_CHECK(cudaStreamSynchronize(stream));
    FME_CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < reps; ++i) {
        gemm<T>(a, b, dc, m, k, n);
    }
    FME_CUDA_CHECK(cudaEventRecord(e1, stream));
    FME_CUDA_CHECK(cudaEventSynchronize(e1));

    float ms = 0.0f;
    FME_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

    // Destroy the events and return the buffer to the pool. These are live-path
    // calls (not teardown), so CHECK-wrapping is correct here.
    FME_CUDA_CHECK(cudaEventDestroy(e0));
    FME_CUDA_CHECK(cudaEventDestroy(e1));
    FME_CUDA_CHECK(cudaFreeAsync(dc, stream));

    return ms / static_cast<float>(reps);
}

template float time_matmul<float>(const float*, const float*, int64_t, int64_t, int64_t, int, int);
template float time_matmul<double>(const double*, const double*, int64_t, int64_t, int64_t, int, int);

} // namespace fme::cuda
