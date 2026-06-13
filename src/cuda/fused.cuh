#pragma once

#include "core/common.hpp"

#include <cstdint>

namespace fme::cuda {

// Device-resident fused elementwise kernels: axpby (a*x + b*y), fma3 (x*y + z,
// one rounding), scaled_relu (maximum(scale*x, 0), NaN-propagating). x, y, z and
// out are DEVICE pointers, C-contiguous (NumPy layout), over a flat n-element
// range. Pure compute on the context compute stream; the binding owns buffer
// lifetime (alloc + free), exactly as gemm<T> leaves staging to its caller.
//
// v1 is DEVICE-RESIDENT ONLY: there is no fused_host / staging twin (unlike
// gemm + gemm_host). A host ndarray fused call computes on the CPU; the GPU path
// is reached only for already-resident fme.Array operands, so the kernel and its
// output buffer live on the single compute stream and there is no transfer ->
// compute cross-stream race surface to fence (the 04-07 lesson is sidestepped by
// construction). The kernels match the SAME unfused-NumPy oracle + elementwise
// tolerance as the CPU path; scaled_relu propagates NaN like np.maximum (the
// device max does NOT, so the kernel restores it), and fma3 is a true single
// rounding via fmaf/fma.
//
// Header is CUDA-free (only core/common.hpp + <cstdint>), mirroring
// gemm_cublas.cuh, so dispatch/bindings forward-declare these without pulling a
// CUDA header and src/cuda stays deletable.
template <typename T>
void fused_axpby(const T* x, const T* y, T* out, int64_t n, T a, T b);

template <typename T>
void fused_fma3(const T* x, const T* y, const T* z, T* out, int64_t n);

template <typename T>
void fused_scaled_relu(const T* x, T* out, int64_t n, T scale);

extern template void fused_axpby<float>(const float*, const float*, float*, int64_t, float, float);
extern template void fused_axpby<double>(const double*, const double*, double*, int64_t, double, double);
extern template void fused_fma3<float>(const float*, const float*, const float*, float*, int64_t);
extern template void fused_fma3<double>(const double*, const double*, const double*, double*, int64_t);
extern template void fused_scaled_relu<float>(const float*, float*, int64_t, float);
extern template void fused_scaled_relu<double>(const double*, double*, int64_t, double);

// cudaEvent-timed warm fused CHAIN, the FUSE-05 GPU measurement primitive plan
// 05 consumes (the time_matmul twin). It takes THREE device array operands x, y,
// z and times the 3-op composition scaled_relu(fma3(axpby(x, y, a, b), z)) --
// three kernel launches per iteration -- NOT a single op, because a single-op
// timer cannot measure the chained work the bench composes. The timed region is
// the chain only: it allocates its own intermediate + output device scratch on
// the compute stream once (so no allocation lands inside the timed window and the
// chain runs transfer-free), warms the clocks, records a start event, runs `reps`
// chain iterations, records a stop event, syncs, and returns cudaEventElapsedTime
// / reps in milliseconds. Device-side timing is immune to the CPU/AV-barrier
// noise the bench-protocol skill documents. a, b, scale are fixed in-body
// constants so the timed chain matches what plan 05 benches; correctness is the
// @gpu oracle test's job, not the timer's.
template <typename T>
float time_fused_chain(const T* x, const T* y, const T* z, int64_t n, int reps,
                       int warmups);

extern template float time_fused_chain<float>(const float*, const float*, const float*, int64_t, int, int);
extern template float time_fused_chain<double>(const double*, const double*, const double*, int64_t, int, int);

} // namespace fme::cuda
