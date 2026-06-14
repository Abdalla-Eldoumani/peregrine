#include "cuda/transfer.cuh"

#include "cuda/check.cuh"
#include "cuda/context.cuh"

#include <cstddef>
#include <cstdint>

namespace fme::cuda {
namespace {

// OOM pre-flight headroom. cudaMemGetInfo reports free VRAM that fluctuates with
// the display compositor and other processes, and cuBLAS keeps an internal
// workspace, so we refuse an allocation that would leave less than this margin
// rather than allocating right up to the reported free and tripping a mid-flight
// failure (or wedging the display). 64MB is comfortably above the cuBLAS v2
// workspace for this phase's sizes and small against the ~4.7GB free measured on
// the dev box.
constexpr int64_t kOomMarginBytes = 64 * 1024 * 1024;

} // namespace

void* alloc_device(int64_t bytes) {
    // Reject a negative byte count explicitly (WR-03): the pre-flight below is
    // gated on `bytes > 0`, so a negative value would SKIP it and then reach
    // cudaMallocAsync via static_cast<size_t>(bytes), which wraps to a near-
    // SIZE_MAX request (~16 EiB). checked_bytes upstream makes this unreachable
    // from the binding, but alloc_device is a public entry in transfer.cuh and
    // must defend its own contract rather than trust the `> 0` skip.
    if (bytes < 0) {
        throw ::fme::cuda_error("alloc_device: negative byte count");
    }
    // A zero-byte buffer is legal (an empty operand): cudaMallocAsync accepts 0
    // and returns a pointer that is valid to free. Skip the pre-flight for it --
    // there is nothing to exhaust -- but still allocate so the handle is uniform.
    Context& ctx = context();
    const cudaStream_t stream = ctx.transfer;

    if (bytes > 0) {
        // Pre-flight against CURRENT free VRAM, never a hardcoded 6144: free
        // fluctuates with the display (measured 4709-5130 MiB on this box at
        // different moments), so a fixed target either fails to guard or wedges
        // the desktop. If the request plus the margin exceeds free, throw
        // fme::cuda_error carrying the cudaErrorMemoryAllocation NAME -- the same
        // name a real allocation failure would carry -- so the wrapper (04-05)
        // composes one byte-math message off the name for both the pre-flight
        // refusal and a mid-flight OOM. (Pitfall 5 / EDGE_CASES VRAM exhaustion.)
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        FME_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const int64_t need = bytes + kOomMarginBytes;
        if (need > static_cast<int64_t>(free_bytes)) {
            throw ::fme::cuda_error(
                std::string(cudaGetErrorName(cudaErrorMemoryAllocation)) +
                ": device allocation of " + std::to_string(bytes) +
                " bytes exceeds free VRAM (" + std::to_string(free_bytes) +
                " bytes free, " + std::to_string(kOomMarginBytes) +
                " bytes margin)");
        }
    }

    void* dptr = nullptr;
    FME_CUDA_CHECK(
        cudaMallocAsync(&dptr, static_cast<std::size_t>(bytes), stream));
    return dptr;
}

void free_device(void* dptr) {
    if (dptr == nullptr) {
        return;
    }
    // ~Array calls this during Python GC, which includes interpreter
    // finalization AFTER the atexit teardown() has destroyed ctx.transfer. Two
    // guards make that path safe (CR-01):
    //
    //   1. If teardown has run, ctx.transfer is a dangling handle the static
    //      Context still names. Skip the free entirely: the mempool buffer is
    //      reclaimed by the CUDA driver at process exit anyway, so leaking it
    //      here is correct, while cudaFreeAsync on the destroyed stream is a
    //      use-after-teardown.
    //   2. On the live path, do NOT use FME_CUDA_CHECK -- it throws, and a throw
    //      escaping ~Array during GC terminates the interpreter (the same hazard
    //      teardown itself avoids, check.cuh). Inspect the return code by hand
    //      and swallow the shutdown/invalid-resource codes that mean the stream
    //      or runtime is already gone; never propagate out of the destructor.
    if (g_torn_down.load()) {
        return;
    }
    Context& ctx = context();
    // cudaFreeAsync on the transfer stream returns the buffer to the mempool
    // (release threshold UINT64_MAX, 04-02) rather than the OS, so the next
    // same-size allocation reuses it. Stream-ordered: it completes behind any
    // copy still queued on the transfer stream that touched this buffer.
    const cudaError_t e = cudaFreeAsync(dptr, ctx.transfer);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading &&
        e != cudaErrorInvalidResourceHandle && e != cudaErrorInvalidValue) {
        // A genuine live-path failure (not a shutdown race). ~Array must never
        // throw, so surface it without propagating: clear the sticky error so it
        // does not poison the next CUDA call, then drop it. A leaked buffer at
        // GC is far better than terminating the interpreter from a destructor.
        cudaGetLastError();
    }
}

void copy_h2d(void* dst_dev, const void* src_host, int64_t bytes) {
    if (bytes <= 0) {
        return;
    }
    // On the TRANSFER stream, not compute: an H2D for one operation can overlap
    // a GEMM still running on the compute stream. Async (queued) -- the caller
    // orders any dependent compute on the same stream or syncs before a
    // host-visible read.
    Context& ctx = context();
    FME_CUDA_CHECK(cudaMemcpyAsync(dst_dev, src_host,
                                   static_cast<std::size_t>(bytes),
                                   cudaMemcpyHostToDevice, ctx.transfer));
}

void copy_d2h(void* dst_host, const void* src_dev, int64_t bytes) {
    if (bytes <= 0) {
        return;
    }
    // D2H on the transfer stream. The HARD rule (check.cuh / EDGE_CASES async
    // correctness): a D2H feeding a returned ndarray is async here but MUST be
    // followed by sync_transfer() before the host buffer is read on the Python
    // side. This function only queues the copy; from_device owns the sync.
    Context& ctx = context();
    FME_CUDA_CHECK(cudaMemcpyAsync(dst_host, src_dev,
                                   static_cast<std::size_t>(bytes),
                                   cudaMemcpyDeviceToHost, ctx.transfer));
}

void sync_transfer() {
    // Block until every queued copy on the transfer stream has finished. This is
    // the API-boundary fence: from_device calls it after copy_d2h so the host
    // buffer is fully written before the ndarray is handed to Python, and
    // to_device calls it after copy_h2d so the operand is fully resident before
    // the unordered compute stream's GEMM reads it (the cross-stream
    // use-before-ready race this fence closes). CHECK-wrapped (a sync can surface
    // an asynchronous launch/copy error) and never used at teardown.
    Context& ctx = context();
    FME_CUDA_CHECK(cudaStreamSynchronize(ctx.transfer));
}

void sync_compute() {
    // Block until every launch on the compute stream (the cuBLAS GEMM) has
    // finished. The device-resident matmul entry calls it after the GEMM so the
    // output buffer is fully written before the result Array can be from_device'd
    // back to host -- the compute-side twin of sync_transfer's D2H fence, the
    // same "async wins inside a call, never across the API boundary" rule applied
    // to the compute stream. CHECK-wrapped and never used at teardown.
    Context& ctx = context();
    FME_CUDA_CHECK(cudaStreamSynchronize(ctx.compute));
}

} // namespace fme::cuda
