#pragma once

#include "core/common.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <atomic>

namespace pg::cuda {

// True once the atexit teardown() has begun destroying the context streams and
// handles (defined in context.cu). free_device (transfer.cu) reads it so an
// Array still alive at interpreter finalization frees nothing rather than calling
// cudaFreeAsync on the by-then-destroyed transfer stream -- a post-teardown free
// is a no-op (the driver reclaims the mempool at exit), never a dead-stream
// access. Atomic because GC at shutdown and teardown are not otherwise ordered.
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
    cudaStream_t compute;     // cublas is bound to this stream; GEMM launches here
    cudaStream_t transfer;    // H2D/D2H copies run here so they overlap compute
    cudaMemPool_t pool;       // the device default mempool, release threshold raised for reuse
    cublasHandle_t cublas;    // one v2 handle, bound to the compute stream
    cublasLtHandle_t cublaslt; // held for future use (batched/epilogue work); GEMM uses cublasSgemm/Dgemm
    bool tf32_enabled;        // PEREGRINE_ALLOW_TF32 folded ONCE here so every GEMM sees one math-mode answer
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
// runs cudaGetDeviceCount and reads compute capability, nothing
// more. Returns the CUDA-free verdict POD (present + cc + reason). present is
// true only when the driver loads, a device exists, and its compute capability
// is >= 7.0; reason names the failure otherwise.
cuda_device_info device_probe();

// Forces context() construction and reports the bound device. This is the entry
// the binding calls to drive context init from Python (and to read the device
// props) without including this header: it returns the CUDA-free
// cuda_device_info, so module.cpp only forward-declares it. If no usable device
// is present it returns a not-present POD (with the probe's reason) and does NOT
// build the context, so calling it on a driverless machine cannot throw.
cuda_device_info context_device_info();

} // namespace pg::cuda
