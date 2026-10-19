#pragma once

#include "core/common.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <atomic>

namespace pg::cuda {

// True once the atexit teardown() has begun destroying the context stream and
// handles (defined in context.cu). free_device and the pinned-pool deleter
// (transfer.cu) read it so an Array or a host result still alive at interpreter
// finalization leaks rather than touching CUDA through a destroyed stream (the
// driver reclaims both at exit). Atomic because GC at shutdown and teardown are
// not otherwise ordered.
extern std::atomic<bool> g_torn_down;

// The process-lifetime CUDA context. One instance exists for the whole process,
// built once on first use (see context() below) and torn down via atexit.
//
// Why a singleton: cublasCreate and cudaStreamCreate cost milliseconds each, so
// creating a handle or stream per matmul would dominate the small-GEMM budget
// and defeat the point of the GPU path (per-call handle creation is the trap this
// avoids). Holding device props, the streams, the mempool, and the
// cublas/cublasLt handles for the process lifetime makes every later operation a
// borrow, not a build.
//
// The struct holds CUDA types and so lives in a .cuh that only src/cuda TUs
// include. Nothing outside src/cuda includes this header: the binding/dispatch
// reach the device through the CUDA-free entry points declared below (which
// return the CUDA-free cuda_device_info from core/common.hpp), so src/cuda stays
// deletable and no CUDA header leaks into the OFF build.
struct Context {
    int device_id;            // the bound device ordinal (0 on this single-GPU box)
    cudaDeviceProp props;     // queried once at init; free/total memory is read at use, not stored
    // Every allocation, copy, kernel, and free runs here, so stream order alone
    // orders a buffer's allocation, its uses, and its free; the host waits only
    // at the fences where it or another library reads device results.
    cudaStream_t stream;
    cudaMemPool_t pool;       // the device default mempool, release threshold raised for reuse
    cublasHandle_t cublas;    // one v2 handle, bound to the stream
    cublasLtHandle_t cublaslt; // held for future use (batched/epilogue work); GEMM uses cublasSgemm/Dgemm
    bool tf32_enabled;        // PEREGRINE_ALLOW_TF32 folded ONCE here so every GEMM sees one math-mode answer
    // The most free VRAM the pre-flight may believe in, from the private
    // PEREGRINE_PREFLIGHT_FREE_CAP (bytes), read once here; INT64_MAX when unset.
    // It lets a test make the pre-flight refuse deterministically.
    int64_t free_cap;
};

// The memoized accessor: builds the Context on first call and returns the same
// instance every time after (mirrors src/cpu feature_detect's const features&
// detect()). Non-const because callers use the handles and streams mutably. The
// build registers the atexit teardown; constructing it has the side effect of
// initializing CUDA, so it is never called at module import (the wrapper's
// no-CUDA-init-at-import rule) -- only on the first device operation.
Context& context();

// Cheap device-presence probe that does NOT build the full Context. has_cuda and
// the requires_cuda test gate must answer "is a usable device here" without paying
// stream/handle/pool creation, so this is separate from context() on purpose: it
// runs cudaGetDeviceCount and reads compute capability, plus the device name for
// a usable device. The verdict is computed on the first call and returned as-is
// after that, so callers may probe on every operation. Returns the CUDA-free
// verdict POD (present + cc + name + reason). present is true only when the
// driver loads, a device exists, and its compute capability is >= 7.0; reason
// names the failure otherwise.
cuda_device_info device_probe();

// Forces context() construction and reports the bound device. This is the entry
// the binding calls to drive context init from Python (and to read the device
// props) without including this header: it returns the CUDA-free
// cuda_device_info, so module.cpp only forward-declares it. If no usable device
// is present it returns a not-present POD (with the probe's reason) and does NOT
// build the context, so calling it on a driverless machine cannot throw.
cuda_device_info context_device_info();

// The atexit teardown, idempotent. Exposed only so a test can run it early in a
// child process and check what still-alive buffers do afterwards; no device call
// may follow it.
void teardown();

// Launches a kernel that traps, without waiting for it, so a test child can see
// an asynchronous device fault surface at the next host fence. The fault is
// sticky: the context is unusable afterwards.
void inject_fault();

} // namespace pg::cuda
