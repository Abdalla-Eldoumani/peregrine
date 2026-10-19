#include "cuda/transfer.cuh"

#include "cuda/check.cuh"
#include "cuda/context.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace pg::cuda {
namespace {

// OOM pre-flight headroom. cudaMemGetInfo reports free VRAM that fluctuates with
// the display compositor and other processes, and cuBLAS keeps an internal
// workspace, so we refuse an allocation that would leave less than this margin
// rather than allocating right up to the reported free and tripping a mid-flight
// failure (or wedging the display). 64MB is comfortably above the cuBLAS v2
// workspace for the supported sizes and small against the ~4.7GB free measured on
// the reference machine.
constexpr int64_t kOomMarginBytes = 64 * 1024 * 1024;

// cudaMemGetInfo measured 2.2-2.5 us per call on this WDDM device, a large share
// of a small device op, so smaller requests are left to the cudaMallocAsync error.
constexpr int64_t kPreflightBytes = int64_t{256} << 20;

// The page-locked pool: power-of-two buffer sizes from one page to the cap, so a
// repeated result shape always finds its size again and the number of distinct
// buffers stays small. Page-locked memory is taken from the OS's nonpaged pool,
// and holding more of it degrades the whole machine, hence the cap.
constexpr int64_t kPinnedCap = int64_t{256} << 20;
constexpr int kMinLog2 = 12;
constexpr int kMaxLog2 = 28;
constexpr int kClasses = kMaxLog2 - kMinLog2 + 1;

int64_t class_bytes(int cls) {
    return int64_t{1} << (cls + kMinLog2);
}

// -1 when the request is larger than the whole pool.
int size_class(int64_t bytes) {
    int cls = 0;
    while (cls < kClasses && class_bytes(cls) < bytes) {
        ++cls;
    }
    return cls < kClasses ? cls : -1;
}

struct host_loan {
    void* data;
    int cls;  // -1: pageable
};

struct pinned_pool {
    std::mutex mu;
    std::vector<void*> free_list[kClasses];
    int64_t reserved = 0;
    int64_t loaned = 0;
};

pinned_pool& pool() {
    // Never destroyed: a NumPy array can release its buffer during interpreter
    // finalization, after this library's static destructors would have run.
    static pinned_pool* p = new pinned_pool;
    return *p;
}

// Under p.mu. Returns a page-locked buffer of class cls, or nullptr when the
// cap leaves no room even after giving back the cached buffers of other sizes.
void* take_pinned(pinned_pool& p, int cls) {
    std::vector<void*>& mine = p.free_list[cls];
    if (!mine.empty()) {
        void* data = mine.back();
        mine.pop_back();
        return data;
    }
    const int64_t size = class_bytes(cls);
    for (int c = kClasses - 1; c >= 0 && p.reserved + size > kPinnedCap; --c) {
        while (!p.free_list[c].empty() && p.reserved + size > kPinnedCap) {
            cudaFreeHost(p.free_list[c].back());
            p.free_list[c].pop_back();
            p.reserved -= class_bytes(c);
        }
    }
    void* data = nullptr;
    if (p.reserved + size > kPinnedCap ||
        cudaHostAlloc(&data, static_cast<std::size_t>(size), cudaHostAllocDefault) !=
            cudaSuccess) {
        data = nullptr;
    }
    // A failed free or an exhausted nonpaged pool is not an error for the
    // caller (the result goes to pageable memory), and it must not linger as
    // the last error for the next kernel launch check to report.
    cudaGetLastError();
    if (data == nullptr) {
        return nullptr;
    }
    p.reserved += size;
    return data;
}

} // namespace

void preflight(int64_t bytes) {
    if (bytes < kPreflightBytes) {
        return;
    }
    Context& ctx = context();
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    PG_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // The mempool keeps freed blocks reserved (release threshold UINT64_MAX), and
    // cudaMemGetInfo counts them as used although the pool can serve them again.
    std::uint64_t reserved = 0;
    std::uint64_t used = 0;
    PG_CUDA_CHECK(cudaMemPoolGetAttribute(ctx.pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
    PG_CUDA_CHECK(cudaMemPoolGetAttribute(ctx.pool, cudaMemPoolAttrUsedMemCurrent, &used));
    const int64_t available = std::min(
        static_cast<int64_t>(free_bytes + (reserved - used)), ctx.free_cap);
    if (bytes + kOomMarginBytes > available) {
        throw ::pg::cuda_error(
            std::string(cudaGetErrorName(cudaErrorMemoryAllocation)) + " at " + __FILE__ +
            ":" + std::to_string(__LINE__) + ": device allocation of " +
            std::to_string(bytes) + " bytes exceeds the " + std::to_string(available) +
            " bytes free less a " + std::to_string(kOomMarginBytes) + " byte margin");
    }
}

void* alloc_device(int64_t bytes) {
    // Reject a negative byte count explicitly: static_cast<size_t>(bytes) would
    // wrap to a near-SIZE_MAX request. checked_bytes upstream makes this
    // unreachable from the binding, but alloc_device must defend its own contract.
    if (bytes < 0) {
        throw ::pg::cuda_error("alloc_device: negative byte count");
    }
    // A zero-byte buffer is legal (an empty operand): cudaMallocAsync accepts 0
    // and returns a pointer that is valid to free, so the handle stays uniform.
    preflight(bytes);
    void* dptr = nullptr;
    PG_CUDA_CHECK(
        cudaMallocAsync(&dptr, static_cast<std::size_t>(bytes), context().stream));
    return dptr;
}

void free_device(void* dptr) {
    if (dptr == nullptr) {
        return;
    }
    // ~Array calls this during Python GC, which includes interpreter
    // finalization AFTER the atexit teardown() has destroyed ctx.stream. Two
    // guards make that path safe:
    //
    //   1. If teardown has run, ctx.stream is a dangling handle the static
    //      Context still names. Skip the free entirely: the mempool buffer is
    //      reclaimed by the CUDA driver at process exit anyway, so leaking it
    //      here is correct, while cudaFreeAsync on the destroyed stream is a
    //      use-after-teardown.
    //   2. On the live path, do NOT use PG_CUDA_CHECK -- it throws, and a throw
    //      escaping ~Array during GC terminates the interpreter (the same hazard
    //      teardown itself avoids, check.cuh). Inspect the return code by hand
    //      and swallow the shutdown/invalid-resource codes that mean the stream
    //      or runtime is already gone; never propagate out of the destructor.
    if (g_torn_down.load()) {
        return;
    }
    // Stream-ordered behind every queued use of the buffer, so no host wait is
    // needed before the free; the block returns to the mempool, not the OS.
    const cudaError_t e = cudaFreeAsync(dptr, context().stream);
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
    const cudaStream_t stream = context().stream;
    PG_CUDA_CHECK(cudaMemcpyAsync(dst_dev, src_host, static_cast<std::size_t>(bytes),
                                  cudaMemcpyHostToDevice, stream));
    // A page-locked source (a from_device result fed back, say) is read by DMA
    // after the call returns, so the caller could change or release it under the
    // copy; a pageable one has already been staged into driver memory.
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, src_host) != cudaSuccess) {
        cudaGetLastError();
    } else if (attr.type == cudaMemoryTypeHost) {
        PG_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

void copy_d2h(void* dst_host, const void* src_dev, int64_t bytes) {
    if (bytes <= 0) {
        return;
    }
    PG_CUDA_CHECK(cudaMemcpyAsync(dst_host, src_dev, static_cast<std::size_t>(bytes),
                                  cudaMemcpyDeviceToHost, context().stream));
}

void sync() {
    PG_CUDA_CHECK(cudaStreamSynchronize(context().stream));
}

bool stream_idle() {
    const cudaError_t e = cudaStreamQuery(context().stream);
    if (e == cudaErrorNotReady) {
        cudaGetLastError();
        return false;
    }
    PG_CUDA_CHECK(e);
    return true;
}

host_buffer acquire_host(int64_t bytes) {
    host_loan* loan = new host_loan{nullptr, size_class(bytes)};
    if (loan->cls >= 0) {
        pinned_pool& p = pool();
        std::lock_guard<std::mutex> lock(p.mu);
        loan->data = take_pinned(p, loan->cls);
        if (loan->data != nullptr) {
            p.loaned += class_bytes(loan->cls);
            return {loan->data, loan};
        }
    }
    loan->cls = -1;
    try {
        loan->data = ::operator new(static_cast<std::size_t>(std::max<int64_t>(bytes, 1)));
    } catch (...) {
        delete loan;
        throw;
    }
    return {loan->data, loan};
}

void release_host(void* raw) noexcept {
    host_loan* loan = static_cast<host_loan*>(raw);
    if (loan->cls < 0) {
        ::operator delete(loan->data);
    } else if (!g_torn_down.load()) {
        try {
            pinned_pool& p = pool();
            std::lock_guard<std::mutex> lock(p.mu);
            p.free_list[loan->cls].push_back(loan->data);
            p.loaned -= class_bytes(loan->cls);
        } catch (...) {
            // Out of memory for the free list: the buffer stays reserved, unused.
        }
    }
    delete loan;
}

pinned_pool_stats pinned_stats() {
    pinned_pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mu);
    return {p.reserved, p.loaned, kPinnedCap};
}

} // namespace pg::cuda
