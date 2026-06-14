#pragma once
#include <cstdint>

namespace pg::cpu {

// Packed-panel BLIS-style GEMM: the AVX2+FMA fast path that dispatch routes to
// when the CPU has the features, with the small-matrix branch living inside the
// entry. The signature mirrors gemm_naive exactly so dispatch can call either
// across the TU boundary with no adapter. Defined in the AVX2-flagged
// gemm_blis.cpp; this header stays intrinsic-free so the fallback path links no
// AVX2 codegen.
template <typename T>
void gemm_blis(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

extern template void gemm_blis<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
extern template void gemm_blis<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

// MC, KC, NC are loop bounds, not unroll factors, so making them runtime-settable
// costs nothing at execution time and lets the sweep walk the
// MC{48,72,96,144} x KC{192,256,320} x NC{2048,4080} grid in one process instead
// of forcing 24 rebuilds. MR and NR stay compile-time because they are the
// structural register-tile shape. KC changes the pc-chunking of each element's
// accumulation, so results differ bitwise across KC values within tolerance;
// thread-count invariance at a fixed blocking is unaffected.
struct blocking {
    int64_t mc, kc, nc;
};

// The blocking triple is published and consumed as one atomic unit. The sweep
// hook (_set_gemm_blocking) holds the GIL but the kernel reads blocking after
// releasing it, so a field-by-field write would let a GIL-free reader observe a
// torn (mc-new, nc-old) triple and size a buffer for one MC while looping for
// another. store_blocking publishes the whole struct in one atomic store;
// load_blocking takes a coherent snapshot. gemm_blis calls load_blocking once at
// entry into locals, so a concurrent store cannot change the loop bounds mid-run.
void store_blocking(blocking b) noexcept;
blocking load_blocking() noexcept;

} // namespace pg::cpu
