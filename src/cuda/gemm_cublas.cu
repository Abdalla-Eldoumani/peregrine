#include "cuda/gemm_cublas.cuh"

#include "cuda/check.cuh"
#include "cuda/context.cuh"

#include <cstdint>

namespace pg::cuda {
namespace {

// RAII owners for the device scratch and cudaEvents these timing/staging
// entries allocate. The bodies are wrapped in PG_*_CHECK calls that throw on
// failure (an OOM mid-allocation, an async launch error surfaced by a sync), and
// without an owner a throw unwinds past the trailing cudaFreeAsync/cudaEventDestroy
// and leaks the buffer into the context mempool (release threshold UINT64_MAX, so
// it is never returned to the OS and compounds the very OOM that triggered it).
// The CPU side solved exactly this with aligned_buf; these mirror it for the
// device. The destructors run on the unwind path, so they must NOT use the
// throwing CHECK -- they inspect nothing and discard the return code the way
// free_device does on its live path, then clear the sticky error so it cannot
// poison the next CUDA call. Freeing on the same stream the buffer was allocated
// on keeps the cleanup stream-ordered behind any work already queued.
struct device_buf {
    void* ptr = nullptr;
    cudaStream_t stream = nullptr;
    device_buf() = default;
    device_buf(void* p, cudaStream_t s) : ptr(p), stream(s) {}
    ~device_buf() {
        if (ptr != nullptr) {
            cudaFreeAsync(ptr, stream);
            cudaGetLastError();
        }
    }
    device_buf(const device_buf&) = delete;
    device_buf& operator=(const device_buf&) = delete;
};

struct cuda_event {
    cudaEvent_t ev = nullptr;
    ~cuda_event() {
        if (ev != nullptr) {
            cudaEventDestroy(ev);
            cudaGetLastError();
        }
    }
    cuda_event() = default;
    cuda_event(const cuda_event&) = delete;
    cuda_event& operator=(const cuda_event&) = delete;
};

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
        PG_CUDA_CHECK(cudaMemsetAsync(
            c, 0, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T),
            context().compute));
        return;
    }

    // Borrow the singleton's handle and stream: no per-call cublasCreate. The
    // handle is already bound to the compute stream and
    // already carries the math mode decided once at init from PEREGRINE_ALLOW_TF32
    // (DEFAULT_MATH off by default). We do NOT call cublasSetMathMode per GEMM:
    // the mode is fixed for the session so it cannot drift into a result that
    // violates the tolerance contract. TF32's 10-bit mantissa measured ~830x
    // worse abs error than DEFAULT_MATH at n=512, which is exactly why it is
    // opt-in only and never compared against the standard contract.
    Context& ctx = context();

    // cuBLAS v2 takes int for m/n/k and the leading dimensions; we carry int64
    // internally per the src int64-index rule. The bench sizes are at most a few
    // thousand per dimension (up to n=2048), far inside INT_MAX, and huge-GEMM
    // k-panel tiling is a later calibration concern, so the narrowing cast here is
    // safe. The guard makes the assumption explicit: if a
    // dimension ever exceeds int range it is a tiling bug, not a silent
    // out-of-bounds device access.
    if (m > INT32_MAX || k > INT32_MAX || n > INT32_MAX) {
        throw ::pg::cuda_error(
            "cuda gemm: a dimension exceeds the int32 cuBLAS limit; "
            "huge-GEMM tiling is not implemented");
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
    PG_CUBLAS_CHECK(cublas_gemm_nn(ctx.cublas, ni, mi, ki, &alpha,
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
    // UINT64_MAX), so repeated calls reuse buffers instead of re-allocating from
    // the OS. All three buffers, the copies, the compute, and the free are
    // ordered on the one compute stream, so no cross-stream event is needed. A
    // zero-k input still allocates dA/dB (size 0 is a valid cudaMallocAsync) and
    // dC, then gemm zero-fills dC. Each buffer is owned by device_buf so a throw
    // from any CHECK below (a copy or the inner gemm surfacing an async/cuBLAS
    // error) frees what was already allocated instead of leaking it into the
    // mempool.
    void* da_raw = nullptr;
    void* db_raw = nullptr;
    void* dc_raw = nullptr;
    PG_CUDA_CHECK(cudaMallocAsync(&da_raw, a_bytes, stream));
    device_buf da{da_raw, stream};
    PG_CUDA_CHECK(cudaMallocAsync(&db_raw, b_bytes, stream));
    device_buf db{db_raw, stream};
    PG_CUDA_CHECK(cudaMallocAsync(&dc_raw, c_bytes, stream));
    device_buf dc{dc_raw, stream};
    T* da_p = static_cast<T*>(da.ptr);
    T* db_p = static_cast<T*>(db.ptr);
    T* dc_p = static_cast<T*>(dc.ptr);

    if (a_bytes > 0) {
        PG_CUDA_CHECK(cudaMemcpyAsync(da_p, a, a_bytes, cudaMemcpyHostToDevice, stream));
    }
    if (b_bytes > 0) {
        PG_CUDA_CHECK(cudaMemcpyAsync(db_p, b, b_bytes, cudaMemcpyHostToDevice, stream));
    }

    gemm<T>(da_p, db_p, dc_p, m, k, n);

    // The destination `c` is the caller's pageable new T[] buffer (module.cpp
    // gemm_host_typed), so this cudaMemcpyAsync D2H is NOT truly asynchronous:
    // into pageable host memory the driver inserts a synchronous staging copy
    // through its own bounce buffer. It is correct, just not overlapped; a pinned
    // host buffer would be needed to make D2H genuinely async, which this
    // correctness entry does not bother with -- the simple pageable copy is the
    // right tradeoff here.
    PG_CUDA_CHECK(cudaMemcpyAsync(c, dc_p, c_bytes, cudaMemcpyDeviceToHost, stream));

    // The D2H copy feeds the caller's host buffer, so sync before returning: the
    // "every D2H feeding a host-visible return syncs first" rule. The device_buf
    // owners free da/db/dc on scope exit (after this sync); cudaFreeAsync is
    // stream-ordered behind the copy, so the result is safely in c once the
    // stream drains, and the frees ride the same stream ordering.
    //
    // ORDERING CONTRACT: the copy, the three frees, and this sync are correct ONLY
    // because everything in gemm_host runs on the ONE compute stream -- the
    // free-after-copy and copy-before-return orderings are pure stream ordering,
    // with no event/fence. If this function is ever migrated to stage on
    // ctx.transfer, the cross-stream ordering between the transfer-stream copy and
    // the compute-stream GEMM must be fenced exactly as matmul_device does
    // (sync_transfer before the GEMM, sync_compute after) -- otherwise it silently
    // becomes a use-before-ready race. Do not move the staging to another stream
    // without adding those fences.
    PG_CUDA_CHECK(cudaStreamSynchronize(stream));
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
        throw ::pg::cuda_error("cuda time_matmul: reps must be positive");
    }
    if (m == 0 || n == 0) {
        throw ::pg::cuda_error(
            "cuda time_matmul: empty output has no work to time");
    }

    Context& ctx = context();
    const cudaStream_t stream = ctx.compute;
    const size_t c_bytes =
        static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T);

    // One output buffer from the context mempool on the compute stream, reused
    // across every warmup and timed rep (the inputs do not change, so the GEMM
    // overwrites C each iteration -- beta is 0 in gemm). Allocated once so no
    // allocation lands inside the timed region. The device_buf and cuda_event
    // owners free the buffer and destroy the events on scope exit, including the
    // unwind path: a throw from the GEMM, a record, a sync, or the elapsed-time
    // query would otherwise leak the buffer into the mempool and leak the events.
    void* dc_raw = nullptr;
    PG_CUDA_CHECK(cudaMallocAsync(&dc_raw, c_bytes, stream));
    device_buf dc{dc_raw, stream};
    T* dc_p = static_cast<T*>(dc.ptr);

    cuda_event e0;
    cuda_event e1;
    PG_CUDA_CHECK(cudaEventCreate(&e0.ev));
    PG_CUDA_CHECK(cudaEventCreate(&e1.ev));

    // Warm the clocks: the first GEMM on a cold GPU pays clock ramp and any lazy
    // cuBLAS init, so timing it would overstate the cost (warm before timing,
    // report warm). These run on the compute stream and are not bracketed by the
    // events.
    for (int i = 0; i < warmups; ++i) {
        gemm<T>(a, b, dc_p, m, k, n);
    }

    // Sync before the start event so the warmups are fully drained and e0 marks a
    // quiet stream; then the timed region is exactly `reps` GEMMs between the two
    // events on the same stream. cudaEventElapsedTime measures device time between
    // them, the only correct way to time an async launch.
    PG_CUDA_CHECK(cudaStreamSynchronize(stream));
    PG_CUDA_CHECK(cudaEventRecord(e0.ev, stream));
    for (int i = 0; i < reps; ++i) {
        gemm<T>(a, b, dc_p, m, k, n);
    }
    PG_CUDA_CHECK(cudaEventRecord(e1.ev, stream));
    PG_CUDA_CHECK(cudaEventSynchronize(e1.ev));

    float ms = 0.0f;
    PG_CUDA_CHECK(cudaEventElapsedTime(&ms, e0.ev, e1.ev));

    // dc, e0, e1 are released by their owners on return. ms is read into the
    // return value before any destructor runs.
    return ms / static_cast<float>(reps);
}

template float time_matmul<float>(const float*, const float*, int64_t, int64_t, int64_t, int, int);
template float time_matmul<double>(const double*, const double*, int64_t, int64_t, int64_t, int, int);

} // namespace pg::cuda
