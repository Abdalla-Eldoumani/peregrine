#include "cuda/gemm_cublas.cuh"

#include "cuda/check.cuh"
#include "cuda/context.cuh"
#include "cuda/transfer.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

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

// Owners for the cuBLASLt descriptors the overlapped staged product builds per
// call, released on the unwind path as device_buf releases its buffer.
template <typename H, cublasStatus_t (*Destroy)(H)>
struct lt_owned {
    H h = nullptr;
    lt_owned() = default;
    ~lt_owned() {
        if (h != nullptr) {
            Destroy(h);
        }
    }
    lt_owned(const lt_owned&) = delete;
    lt_owned& operator=(const lt_owned&) = delete;
};
using lt_desc = lt_owned<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy>;
using lt_layout = lt_owned<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy>;
using lt_pref = lt_owned<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy>;

// A large staged float32 product overlaps its transfers with its compute: A and
// C are split into row blocks, and while block i computes, block i+1 uploads and
// block i-1 downloads. This GPU has one copy engine (asyncEngineCount 1), so
// uploads and downloads still take turns; the blocks hide the GEMM under them.
// Four blocks weighted 1:2:2:1 measured best of the partitions tried (median ms,
// serial / two halves / four equal / 1:2:2:1 / six blocks): 2048: 7.06 / 5.38 /
// 5.28 / 4.84 / 4.89, 3000: 17.0 / 14.2 / 12.8 / 12.4 / 12.4, 4096: 38.1 / 32.6
// / 30.5 / 28.3 / 27.1. The small first block starts the compute soon after B
// has arrived, and the small last one shortens the download left once it ends.
constexpr int kBlocks = 4;

// Block edges fall on multiples of 64 rows, so every block's A and C pointers
// keep the 256-byte alignment of the whole buffers, which cuBLASLt's heuristic
// assumes by default for the algorithm it picks for the whole product.
constexpr int64_t kRowAlign = 64;

// Overlap from 768 rows and 2*512^3 flops (median ms, overlapped against
// serial). Rows: square 768 measured 0.76 against 0.89, square 512 lost, 0.47
// against 0.43, to the extra launches and events; 768 rows also keeps every
// block at 128 rows or more. Work, at 768 rows or more: 768x512x512 0.525 against
// 0.543 and 2048x256x256 0.47 against 0.56, while under 2*512^3 flops it bought
// nothing (768x300x300 0.324 against 0.327, 1024x256x256 0.34 against 0.34) or
// lost (768x64x64 0.17 against 0.13).
constexpr int64_t kOverlapMinRows = 768;
constexpr double kOverlapMinFlops = 2.0 * 512 * 512 * 512;

// Every block runs the one algorithm cuBLASLt's heuristic picks for the whole
// product, so its elements are summed exactly as the whole product sums them.
// Left to choose per shape, cuBLAS gives a block a different kernel (split-K or
// not), which changed 90 percent or more of the elements at every size tried.
// With a 4 MiB workspace limit the heuristic's first choice reproduced
// cublasSgemm's bits at every one of 136 shapes tested (squares 16 to 4096,
// rectangular and skinny ones), and so did every 64-row-aligned blocking of
// them; with no workspace or 1 MiB it picked other algorithms at 1024 to 4096.
constexpr size_t kLtWorkspaceBytes = size_t{4} << 20;

bool overlaps(int64_t m, int64_t k, int64_t n) {
    return m >= kOverlapMinRows &&
           2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n) >=
               kOverlapMinFlops;
}

// On a throw, waits for the side streams before the caller frees the device
// buffers on the context stream and hands the host result buffer back, since
// copies still queued there read and write both. The normal path has already
// ordered them before the context stream's fence and sets joined.
struct side_join {
    cudaStream_t up;
    cudaStream_t down;
    bool joined = false;
    ~side_join() {
        if (!joined) {
            cudaStreamSynchronize(up);
            cudaStreamSynchronize(down);
            cudaGetLastError();
        }
    }
};

// Queues the blocked product on the context stream and the two side streams and
// joins the side streams back into the context stream; the caller fences. Returns
// false, having queued nothing, when cuBLASLt offers no algorithm for the shape.
bool gemm_host_overlapped(Context& ctx, const float* a, const float* b, float* c, float* da,
                          float* db, float* dc, int64_t m, int64_t k, int64_t n) {
    const cudaStream_t stream = ctx.stream;
    const cudaStream_t up = ctx.upload_stream;
    const cudaStream_t down = ctx.download_stream;

    // Row-major C = A @ B as column-major C^T = B^T @ A^T, the operand swap gemm
    // uses; a row block of A and C is a column block of A^T and C^T.
    lt_desc desc;
    PG_CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc.h, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    lt_layout lb;
    lt_layout la;
    lt_layout lc;
    PG_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lb.h, CUDA_R_32F, n, k, n));
    PG_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&la.h, CUDA_R_32F, k, m, k));
    PG_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lc.h, CUDA_R_32F, n, m, n));
    lt_pref pref;
    PG_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref.h));
    size_t ws_limit = kLtWorkspaceBytes;
    PG_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref.h, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_limit, sizeof(ws_limit)));
    cublasLtMatmulHeuristicResult_t heur{};
    int found = 0;
    const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
        ctx.cublaslt, desc.h, lb.h, la.h, lc.h, lc.h, pref.h, 1, &heur, &found);
    if (hs == CUBLAS_STATUS_NOT_SUPPORTED || (hs == CUBLAS_STATUS_SUCCESS && found == 0)) {
        return false;
    }
    PG_CUBLAS_CHECK(hs);

    const int64_t edge[kBlocks + 1] = {0, m / 6 / kRowAlign * kRowAlign,
                                       m / 2 / kRowAlign * kRowAlign,
                                       5 * m / 6 / kRowAlign * kRowAlign, m};

    void* ws_raw = nullptr;
    PG_CUDA_CHECK(cudaMallocAsync(&ws_raw, kLtWorkspaceBytes, stream));
    device_buf ws{ws_raw, stream};
    cuda_event ready;
    cuda_event done;
    cuda_event uploaded[kBlocks];
    cuda_event computed[kBlocks];
    for (cuda_event* e : {&ready, &done}) {
        PG_CUDA_CHECK(cudaEventCreateWithFlags(&e->ev, cudaEventDisableTiming));
    }
    for (int i = 0; i < kBlocks; ++i) {
        PG_CUDA_CHECK(cudaEventCreateWithFlags(&uploaded[i].ev, cudaEventDisableTiming));
        PG_CUDA_CHECK(cudaEventCreateWithFlags(&computed[i].ev, cudaEventDisableTiming));
    }
    side_join join{up, down};

    // The buffers were allocated on the context stream: the side streams start
    // behind that.
    PG_CUDA_CHECK(cudaEventRecord(ready.ev, stream));
    PG_CUDA_CHECK(cudaStreamWaitEvent(up, ready.ev, 0));
    PG_CUDA_CHECK(cudaStreamWaitEvent(down, ready.ev, 0));
    PG_CUDA_CHECK(cudaMemcpyAsync(db, b, static_cast<size_t>(k) * n * sizeof(float),
                                  cudaMemcpyHostToDevice, up));

    const auto download = [&](int i) {
        PG_CUDA_CHECK(cudaStreamWaitEvent(down, computed[i].ev, 0));
        PG_CUDA_CHECK(cudaMemcpyAsync(c + edge[i] * n, dc + edge[i] * n,
                                      static_cast<size_t>(edge[i + 1] - edge[i]) * n * sizeof(float),
                                      cudaMemcpyDeviceToHost, down));
    };
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int i = 0; i < kBlocks; ++i) {
        const int64_t r0 = edge[i];
        const uint64_t rows = static_cast<uint64_t>(edge[i + 1] - r0);
        // A pageable source returns once staged, so the host queues the next block
        // only then; the GEMMs already queued run meanwhile.
        PG_CUDA_CHECK(cudaMemcpyAsync(da + r0 * k, a + r0 * k, rows * k * sizeof(float),
                                      cudaMemcpyHostToDevice, up));
        PG_CUDA_CHECK(cudaEventRecord(uploaded[i].ev, up));
        PG_CUDA_CHECK(cudaStreamWaitEvent(stream, uploaded[i].ev, 0));
        PG_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(la.h, CUBLASLT_MATRIX_LAYOUT_COLS, &rows,
                                                         sizeof(rows)));
        PG_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(lc.h, CUBLASLT_MATRIX_LAYOUT_COLS, &rows,
                                                         sizeof(rows)));
        PG_CUBLAS_CHECK(cublasLtMatmul(ctx.cublaslt, desc.h, &alpha, db, lb.h, da + r0 * k, la.h,
                                       &beta, dc + r0 * n, lc.h, dc + r0 * n, lc.h, &heur.algo,
                                       ws.ptr, kLtWorkspaceBytes, stream));
        PG_CUDA_CHECK(cudaEventRecord(computed[i].ev, stream));
        // The copy engine takes copies in the order they are queued, so the
        // previous block's download goes behind this block's upload, which is
        // ready at once, rather than ahead of it while waiting for its GEMM.
        if (i > 0) {
            download(i - 1);
        }
    }
    download(kBlocks - 1);

    // The last download waited for the last GEMM, which waited for the last
    // upload: once the context stream passes `done`, both side streams are idle.
    PG_CUDA_CHECK(cudaEventRecord(done.ev, down));
    PG_CUDA_CHECK(cudaStreamWaitEvent(stream, done.ev, 0));
    join.joined = true;
    return true;
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
    // context stream rather than memset on host.
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        PG_CUDA_CHECK(cudaMemsetAsync(
            c, 0, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T),
            context().stream));
        return;
    }

    // Borrow the singleton's handle and stream: no per-call cublasCreate. The
    // handle is already bound to the context stream and
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
    const cudaStream_t stream = ctx.stream;

    const size_t a_bytes = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(T);
    const size_t b_bytes = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(T);
    const size_t c_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T);

    // TF32 changes the compute type the overlapped path would have to match, so
    // it stays on the plain path.
    const bool overlap = std::is_same_v<T, float> && !ctx.tf32_enabled && overlaps(m, k, n);

    // One pre-flight for the buffers together: they are live at once.
    preflight(static_cast<int64_t>(a_bytes + b_bytes + c_bytes +
                                   (overlap ? kLtWorkspaceBytes : 0)));

    // Stream-ordered allocation from the context mempool (release threshold
    // UINT64_MAX), so repeated calls reuse buffers instead of re-allocating from
    // the OS. The buffers are allocated and freed on the context stream; the
    // overlapped path orders its side streams against it with events. A
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

    bool queued = false;
    if constexpr (std::is_same_v<T, float>) {
        if (overlap) {
            queued = gemm_host_overlapped(ctx, a, b, c, da_p, db_p, dc_p, m, k, n);
        }
    }
    if (!queued) {
        if (a_bytes > 0) {
            PG_CUDA_CHECK(cudaMemcpyAsync(da_p, a, a_bytes, cudaMemcpyHostToDevice, stream));
        }
        if (b_bytes > 0) {
            PG_CUDA_CHECK(cudaMemcpyAsync(db_p, b, b_bytes, cudaMemcpyHostToDevice, stream));
        }

        gemm<T>(da_p, db_p, dc_p, m, k, n);

        // `c` is a page-locked pool buffer (the full-rate DMA) or the caller's out
        // array (pageable, staged by the driver); either way the copy is queued.
        PG_CUDA_CHECK(cudaMemcpyAsync(c, dc_p, c_bytes, cudaMemcpyDeviceToHost, stream));
    }

    // The host fence: c is complete, and any asynchronous error from the work
    // above surfaces here, while the caller still holds the host operands. The
    // device_buf frees queue behind the copy on scope exit.
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
    const cudaStream_t stream = ctx.stream;
    const size_t c_bytes =
        static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T);

    // One output buffer from the context mempool on the context stream, reused
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
    // report warm). These run on the context stream and are not bracketed by the
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
