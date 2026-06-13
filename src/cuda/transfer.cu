#include "cuda/transfer.cuh"

#include "cuda/check.cuh"
#include "cuda/context.cuh"

#include <cstdint>
#include <mutex>
#include <vector>

namespace fme::cuda {
namespace {

// The pinned-staging cap (CONTEXT line 43, the cuda-sm86 skill). 256MB total
// across every cached pinned buffer. Unbounded cudaHostAlloc fragments Windows
// and degrades the whole OS, so this is a hard ceiling, not a hint.
constexpr int64_t kPinnedCapBytes = 256 * 1024 * 1024;

// OOM pre-flight headroom. cudaMemGetInfo reports free VRAM that fluctuates with
// the display compositor and other processes, and cuBLAS keeps an internal
// workspace, so we refuse an allocation that would leave less than this margin
// rather than allocating right up to the reported free and tripping a mid-flight
// failure (or wedging the display). 64MB is comfortably above the cuBLAS v2
// workspace for this phase's sizes and small against the ~4.7GB free measured on
// the dev box.
constexpr int64_t kOomMarginBytes = 64 * 1024 * 1024;

// A cached pinned buffer: the host pointer, its capacity, and whether it is
// currently lent out. Reuse is by capacity (a request reuses the smallest free
// buffer that fits); eviction frees the largest free buffers first when a new
// allocation would breach the cap, since a few large buffers dominate the total.
struct PinnedSlot {
    void* ptr;
    int64_t capacity;
    bool in_use;
};

// The cache plus its running total, guarded by a mutex: acquire/release run
// after the GIL is dropped (the binding releases it around the CUDA work), so
// two host threads transferring at once would otherwise race the vector and the
// byte counter. A plain std::mutex is ample -- transfers are not a hot inner
// loop, the GEMM is.
std::mutex g_pinned_mtx;
std::vector<PinnedSlot> g_pinned_slots;
int64_t g_pinned_total = 0;

// Free a pinned slot's memory and drop it from the running total. Uses
// FME_CUDA_CHECK because this runs on the live allocation path (acquire's
// eviction), never at teardown -- the cache is process-lifetime and is not torn
// down by atexit (the buffers are returned to the OS at process exit anyway, and
// touching CUDA during static deinit is the destruction-order trap check.cuh
// warns about).
void evict_slot_locked(std::size_t idx) {
    FME_CUDA_CHECK(cudaFreeHost(g_pinned_slots[idx].ptr));
    g_pinned_total -= g_pinned_slots[idx].capacity;
    g_pinned_slots.erase(g_pinned_slots.begin() + static_cast<std::ptrdiff_t>(idx));
}

} // namespace

void* alloc_device(int64_t bytes) {
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

void* pinned_acquire(int64_t bytes) {
    if (bytes <= 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_pinned_mtx);

    // Reuse the smallest free cached buffer that fits the request: a tight fit
    // wastes the least pinned memory and keeps a single large buffer available
    // for a later large transfer. Mark it lent and return it.
    std::size_t best = g_pinned_slots.size();
    for (std::size_t i = 0; i < g_pinned_slots.size(); ++i) {
        if (!g_pinned_slots[i].in_use && g_pinned_slots[i].capacity >= bytes) {
            if (best == g_pinned_slots.size() ||
                g_pinned_slots[i].capacity < g_pinned_slots[best].capacity) {
                best = i;
            }
        }
    }
    if (best != g_pinned_slots.size()) {
        g_pinned_slots[best].in_use = true;
        return g_pinned_slots[best].ptr;
    }

    // No reusable buffer. A request larger than the whole cap can never be
    // satisfied from pinned staging: report nullptr so the caller takes the
    // unpinned fallback rather than over-committing pinned memory.
    if (bytes > kPinnedCapBytes) {
        return nullptr;
    }

    // Make room under the cap by freeing the LARGEST free buffers first (a few
    // big buffers dominate the 256MB total, so freeing them frees the most room
    // per eviction). If even after evicting every free buffer the new allocation
    // would breach the cap, the cache is saturated by in-use buffers: report
    // nullptr and let the caller fall back, never exceed the cap.
    while (g_pinned_total + bytes > kPinnedCapBytes) {
        std::size_t largest = g_pinned_slots.size();
        for (std::size_t i = 0; i < g_pinned_slots.size(); ++i) {
            if (!g_pinned_slots[i].in_use &&
                (largest == g_pinned_slots.size() ||
                 g_pinned_slots[i].capacity > g_pinned_slots[largest].capacity)) {
                largest = i;
            }
        }
        if (largest == g_pinned_slots.size()) {
            return nullptr;  // nothing free to evict; cap would be breached
        }
        evict_slot_locked(largest);
    }

    void* host_ptr = nullptr;
    FME_CUDA_CHECK(
        cudaHostAlloc(&host_ptr, static_cast<std::size_t>(bytes),
                      cudaHostAllocDefault));
    g_pinned_slots.push_back(PinnedSlot{host_ptr, bytes, true});
    g_pinned_total += bytes;
    return host_ptr;
}

void pinned_release(void* host_ptr) {
    if (host_ptr == nullptr) {
        return;
    }
    // Return the buffer to the cache (mark it free for reuse); do NOT free it.
    // The 256MB cap and the eviction policy in acquire bound the total, so a
    // released buffer staying resident is the point -- the next transfer reuses
    // it instead of paying another cudaHostAlloc.
    std::lock_guard<std::mutex> lock(g_pinned_mtx);
    for (PinnedSlot& slot : g_pinned_slots) {
        if (slot.ptr == host_ptr) {
            slot.in_use = false;
            return;
        }
    }
}

} // namespace fme::cuda
