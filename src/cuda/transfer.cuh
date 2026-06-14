#pragma once

#include "core/common.hpp"

#include <cstdint>

namespace pg::cuda {

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

// The device-buffer handle pg.Array wraps. It owns a device pointer allocated
// from the context mempool plus the 2-D shape and the element dtype, so it is
// fully self-describing: from_device and the dlpack export read rows/cols/dtype
// off the handle with no parallel state on the binding side. Carrying the shape
// and dtype on the handle keeps the binding free of a second source of truth that
// could drift from the buffer it describes.
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
// on a context stream, reused via the UINT64_MAX release threshold the context
// sets on the pool).
// Pre-flights cudaMemGetInfo against the CURRENT free VRAM and throws
// pg::cuda_error carrying the cudaErrorMemoryAllocation NAME when the request
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
// to_device also calls it after copy_h2d so operands are fully resident before a
// later compute-stream GEMM reads them: alloc_device/copy_h2d run on the transfer
// stream, the GEMM on the compute stream, and the two streams are unordered, so
// without this fence cuBLAS can read an operand whose H2D copy has not landed (a
// cross-stream use-before-ready race; compute-sanitizer memcheck
// --track-stream-ordered-races all reports it as use-before-alloc).
void sync_transfer();

// Synchronize the compute stream: blocks until every launch on it (the cuBLAS
// GEMM) has completed. The device-resident matmul entry calls it after the GEMM
// so the output buffer is fully written before the result Array crosses back to
// Python -- the compute-side twin of sync_transfer's D2H fence. Without it a
// caller could from_device an output whose GEMM has not finished, the same
// across-the-API-boundary race the D2H sync closes for transfers. CHECK-wrapped
// (a sync can surface an asynchronous launch error) and never used at teardown.
void sync_compute();

// Pinned host-staging is a future optimization and is not implemented here:
// current transfers use pageable host memory (gemm_host stages through a pageable
// new T[]; to_device/from_device copy pageable host buffers). A pageable copy
// forces cudaMemcpyAsync through a driver-internal bounce buffer and cannot fully
// overlap, so a pinned-buffer cache would be the way to make H2D/D2H genuinely
// async at the full PCIe rate when a large-transfer path needs it.

} // namespace pg::cuda
