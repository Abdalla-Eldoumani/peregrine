#pragma once
#include <cstdint>

namespace fme::cpu {

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

blocking& current_blocking();

} // namespace fme::cpu
