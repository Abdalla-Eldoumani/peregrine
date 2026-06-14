#pragma once
#include <cstdint>

namespace pg::cpu {

// Packing copies the live MC x KC block of A and the KC x NC block of B into
// contiguous 64-byte-aligned panels so the microkernel reads both with stride 1
// regardless of the caller's row-major layout: the inner loop then never pays a
// TLB miss or a strided load. Edge panels are zero-padded to the register tile
// extent during the copy, so the microkernel computes a full tile with no branch
// on K and the zeros contribute nothing. Definitions live in the AVX2-flagged
// pack.cpp; this header carries no intrinsics so the translation units the
// no-AVX2 fallback links never see AVX2 codegen.
template <typename T>
void pack_a(T* ap, const T* a, int64_t ic, int64_t pc, int64_t mc, int64_t kc, int64_t k);

template <typename T>
void pack_b(T* bp, const T* b, int64_t pc, int64_t jc, int64_t kc, int64_t nc, int64_t n);

extern template void pack_a<float>(float*, const float*, int64_t, int64_t, int64_t, int64_t, int64_t);
extern template void pack_a<double>(double*, const double*, int64_t, int64_t, int64_t, int64_t, int64_t);
extern template void pack_b<float>(float*, const float*, int64_t, int64_t, int64_t, int64_t, int64_t);
extern template void pack_b<double>(double*, const double*, int64_t, int64_t, int64_t, int64_t, int64_t);

} // namespace pg::cpu
