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

// Refuses a request the device cannot hold, before allocating, by throwing
// pg::cuda_error named cudaErrorMemoryAllocation (the name a failed allocation
// carries, so the wrapper treats both alike). Only requests of at least 256 MiB
// are checked: on WDDM an oversized allocation can spill into system memory
// through the driver's sysmem fallback instead of failing, and that is where it
// matters; smaller requests rely on the cudaMallocAsync error. Free is what
// cudaMemGetInfo reports plus what the mempool holds unused, capped by
// Context::free_cap.
void preflight(int64_t bytes);

// Allocate a device buffer of `bytes` from the context mempool (pre-flight, then
// cudaMallocAsync on the context stream; reused via the pool's UINT64_MAX
// release threshold). The caller owns the pointer and frees it with free_device.
void* alloc_device(int64_t bytes);

// Free a device buffer previously returned by alloc_device (cudaFreeAsync back
// to the mempool, ordered behind every queued use). Never throws, and after
// teardown it leaks instead of touching the destroyed stream.
void free_device(void* dptr);

// Copies queued on the context stream. copy_h2d returns once the source may be
// changed: a pageable source has been staged by then, and a page-locked one is
// waited for. copy_d2h only queues; the host reads the destination after sync().
void copy_h2d(void* dst_dev, const void* src_host, int64_t bytes);
void copy_d2h(void* dst_host, const void* src_dev, int64_t bytes);

// The host fence: blocks until everything queued on the context stream is done,
// and throws pg::cuda_error carrying the name of any asynchronous device error
// queued work raised. Used wherever the host or another library is about to read
// device results.
void sync();

// Whether the context stream has no unfinished work (cudaStreamQuery), for tests.
bool stream_idle();

// Host buffers for device-to-host results, lent to NumPy. A buffer is
// page-locked, from a pool capped at 256 MB of page-locked memory counting the
// buffers on loan, or ordinary pageable memory once the pool cannot fit it.
// release_host is the capsule deleter that ends the loan: it makes no CUDA call,
// and after teardown it leaks the buffer.
struct host_buffer {
    void* data;
    void* loan;  // the capsule pointer for release_host
};
host_buffer acquire_host(int64_t bytes);
void release_host(void* loan) noexcept;

struct pinned_pool_stats {
    int64_t reserved;  // page-locked bytes the pool holds, free or on loan
    int64_t loaned;
    int64_t cap;
};
pinned_pool_stats pinned_stats();

} // namespace pg::cuda
