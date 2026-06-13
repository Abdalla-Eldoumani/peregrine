#pragma once

#include "core/common.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <string>

// Every CUDA and cuBLAS call in src/cuda goes through one of these macros: a
// naked return code is a silent-corruption or silent-crash bug. On failure they
// throw fme::cuda_error (defined in core/common.hpp, CUDA-include-free) carrying
// the error NAME plus file and line. Only the name travels up; the distinct
// user-facing wording (cudaErrorMemoryAllocation, cudaErrorInsufficientDriver,
// launch timeout) is composed in the Python wrapper, which owns user semantics.
//
// HARD RULE -- teardown must NOT use these macros. They throw, and a throw
// during atexit/static-deinit after the CUDA driver has unloaded crashes the
// process (the destruction-order trap). Teardown code checks return codes by
// hand and swallows cudaErrorCudartUnloading / already-freed states instead of
// throwing. This is enforced where the context singleton's teardown lands.

namespace fme::cuda {

// cuBLAS error-name resolution. 12.8 ships cublasGetStatusName, so prefer it for
// a human-readable name. The fallback switch exists only so check.cuh still
// compiles if that symbol is ever absent (it covers the statuses a GEMM/handle
// path can actually return); it is never the primary path on this toolkit.
inline const char* cublas_status_name(cublasStatus_t s) {
#if defined(CUBLAS_VER_MAJOR)
    return cublasGetStatusName(s);
#else
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
#endif
}

} // namespace fme::cuda

#define FME_CUDA_CHECK(x)                                                       \
    do {                                                                        \
        cudaError_t e_ = (x);                                                   \
        if (e_ != cudaSuccess) {                                                \
            throw ::fme::cuda_error(                                            \
                std::string(cudaGetErrorName(e_)) + " at " + __FILE__ + ":" +   \
                std::to_string(__LINE__));                                      \
        }                                                                       \
    } while (0)

#define FME_CUBLAS_CHECK(x)                                                     \
    do {                                                                        \
        cublasStatus_t s_ = (x);                                                \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                      \
            throw ::fme::cuda_error(                                            \
                std::string(::fme::cuda::cublas_status_name(s_)) + " at " +     \
                __FILE__ + ":" + std::to_string(__LINE__));                     \
        }                                                                       \
    } while (0)
