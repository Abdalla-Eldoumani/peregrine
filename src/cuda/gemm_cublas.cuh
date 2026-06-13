#pragma once

#include "core/common.hpp"

#include <cstdint>

namespace fme::cuda {

// Device float32/float64 GEMM through cuBLAS: C(m,n) = A(m,k) @ B(k,n), all
// three operands ROW-MAJOR (C-contiguous, NumPy layout) and DEVICE-resident.
// cuBLAS is column-major, so the body computes the same bytes via the
// no-transpose operand-swap trick (see gemm_cublas.cu) -- no explicit transpose
// kernel, zero layout cost. Mirrors fme::cpu::gemm_naive's signature so dispatch
// (04-05) calls either side across the TU boundary with no adapter.
//
// Pointer contract: a, b, c are DEVICE pointers. Host<->device staging is
// transfer.cu (04-04); this function is pure compute and never touches host
// memory or syncs. The math mode is the one decided at context init from
// FME_ALLOW_TF32 (DEFAULT_MATH off by default), read here from context() so a
// GEMM cannot toggle it mid-session into a tolerance-contract violation. Zero
// dimensions are handled per NumPy without launching cuBLAS.
template <typename T>
void gemm(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

extern template void gemm<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
extern template void gemm<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

// Host-pointer convenience: allocates dA/dB/dC from the context mempool on the
// compute stream, copies the host inputs up, runs the device gemm above, copies
// the result back, and synchronizes before returning. This exists so the
// correctness suite has a callable host->device->host path in THIS plan, before
// the fme.Array residency machinery and the device-resident matmul binding land
// in 04-04. Once those exist, the device-resident path is gemm<T> directly on
// fme.Array buffers; this convenience stays as the simplest correctness entry.
// a, b, c are HOST pointers (c is the caller-owned m*n output buffer).
template <typename T>
void gemm_host(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

extern template void gemm_host<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
extern template void gemm_host<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

} // namespace fme::cuda
