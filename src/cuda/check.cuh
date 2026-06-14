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

// cuBLAS error-name resolution. cublasGetStatusName has shipped since well before
// 12.x and 12.8 is the only supported toolkit, so call it unconditionally.
//
// A previous version gated a hand-written fallback switch behind
// `#if defined(CUBLAS_VER_MAJOR)`, but CUBLAS_VER_MAJOR is defined by
// cublas_v2.h on every versioned header, so the `#if` arm was ALWAYS taken and
// the `#else` switch was dead, unverified code. Worse, the guard tested
// "is this a versioned cuBLAS header" (always yes), not "does
// cublasGetStatusName exist" (the actual question), so it would not even have
// covered the case it claimed to. There is no clean preprocessor test for a
// function's existence in cuBLAS, so the honest form is the direct call.
inline const char* cublas_status_name(cublasStatus_t s) {
    return cublasGetStatusName(s);
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
