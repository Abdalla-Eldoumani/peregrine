#pragma once

#include "core/common.hpp"

#include <cstdint>

namespace fme::cuda {

// The dtype a device buffer holds. The Array binding (module.cpp) maps this to
// and from the NumPy dtype; the transfer/GEMM code only ever needs the element
// size, which dtype_size() below gives. Kept here (not a CUDA type) so the
// device-buffer handle is self-describing without the binding having to carry a
// parallel dtype field.
enum class DType : int {
    f32 = 0,
    f64 = 1,
};

inline int64_t dtype_size(DType dt) {
    return dt == DType::f64 ? 8 : 4;
}

// The device-buffer handle fme.Array wraps. It owns a device pointer allocated
// from the context mempool plus the 2-D shape and the element dtype, so it is
// fully self-describing: from_device and the dlpack export read rows/cols/dtype
// off the handle with no parallel state on the binding side (RESEARCH leaves the
// shape/dtype ownership to Claude; carrying it here keeps the binding free of a
// second source of truth that could drift from the buffer it describes).
//
// ptr is a raw device pointer; the byte size is rows*cols*dtype_size. Ownership
// is single-owner: exactly one Array (one capsule deleter) frees a given ptr via
// free_device. Copying the struct does NOT copy the buffer, so the binding must
// never let two live Arrays hold the same ptr (it moves the handle into the
// class instance and the capsule, never duplicates a live one).
struct DeviceBuffer {
    void* ptr;
    int64_t rows;
    int64_t cols;
    DType dtype;

    int64_t bytes() const { return rows * cols * dtype_size(dtype); }
};

// Allocate a device buffer of `bytes` from the context mempool (cudaMallocAsync
// on a context stream, reused via the UINT64_MAX release threshold 04-02 set).
// Pre-flights cudaMemGetInfo against the CURRENT free VRAM and throws
// fme::cuda_error carrying the cudaErrorMemoryAllocation NAME when the request
// would exceed free (with a small headroom margin) -- never discovering OOM by
// failing the allocation, which on a display GPU can wedge the desktop. The
// returned pointer is owned by the caller and freed with free_device.
void* alloc_device(int64_t bytes);

// Free a device buffer previously returned by alloc_device (cudaFreeAsync back
// to the mempool). Returns the memory to the pool, not the OS, so the next
// allocation of the same size reuses it. CHECK-wrapped: never call from teardown.
void free_device(void* dptr);

// Host->device and device->host copies on the context TRANSFER stream (separate
// from the compute stream so a copy can overlap an unrelated GEMM). Both are
// cudaMemcpyAsync: they are queued, not synchronous. A D2H whose result is about
// to cross back into Python MUST be followed by sync_transfer() before the host
// buffer is read -- see from_device in the binding. bytes is int64 per the src
// index rule; the cast to size_t at the cudaMemcpyAsync boundary is guarded by
// the caller's allocation having succeeded for that many bytes.
void copy_h2d(void* dst_dev, const void* src_host, int64_t bytes);
void copy_d2h(void* dst_host, const void* src_dev, int64_t bytes);

// Synchronize the transfer stream: blocks until every queued copy on it has
// completed. This is the "async wins live inside a call, never across the API
// boundary" rule made concrete -- from_device calls it after copy_d2h and before
// returning the host ndarray, so a caller can never read a half-filled buffer.
void sync_transfer();

// Pinned host-staging cache (cudaHostAlloc), total capped at 256MB.
//
// Why pinned at all: a pageable host buffer forces cudaMemcpyAsync to fall back
// to a synchronous staging copy through a driver-internal pinned bounce buffer,
// so H2D/D2H on the transfer stream cannot truly overlap or run async. A pinned
// buffer lets the copy be genuinely asynchronous and hits the full PCIe rate.
//
// Why a CAP and a cache (not allocate-per-call): cudaHostAlloc is expensive and
// pinned pages are a scarce OS resource -- unbounded pinned memory fragments and
// degrades all of Windows (the skill's hard rule). So buffers are pooled and
// reused, and the running total is capped at 256MB.
//
// acquire returns a pinned buffer of at least `bytes`, reusing a cached one that
// fits when possible; release returns it to the cache. The eviction policy is
// documented in transfer.cu. acquire returning nullptr means "cap reached, no
// reusable buffer" and the caller must fall back to an unpinned path; this plan's
// transfers are small enough that the cap is never the binding constraint, but
// the contract is explicit so a future large-transfer path cannot silently
// over-commit pinned memory.
void* pinned_acquire(int64_t bytes);
void pinned_release(void* host_ptr);

} // namespace fme::cuda
