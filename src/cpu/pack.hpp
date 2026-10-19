#pragma once
#include <cstdint>

namespace pg::cpu {

// Packing copies a block of A or B into the contiguous 64-byte-aligned panels
// the microkernels read with stride 1. It reads the operand through arbitrary
// element strides (element (i, j) at x[i * rs + j * cs], negative and zero
// strides included), so a.T, F-ordered, sliced, and reversed operands are never
// copied first, and every layout yields the same panels, hence the same bits.
// Rows or columns past the block in the last panel are written zero on every
// pack, because the buffers are reused, so the microkernel never branches on an
// edge. Definitions live in the AVX2-flagged pack.cpp; this header carries no
// intrinsics so the TUs the no-AVX2 fallback links never see AVX2 codegen.

// The mc x kc block at a into ceil(mc / MR) panels; panel q holds rows
// q*MR..q*MR+MR-1 as kc steps of MR values each, at ap + q * MR * kc.
template <int MR, typename T>
void pack_a(T* ap, const T* a, int64_t rs, int64_t cs, int64_t mc, int64_t kc);

// Panels p0..p1-1 of the kc x nc block at b; panel q holds columns
// q*NR..q*NR+NR-1 as kc steps of NR values each, at bp + q * kc * NR, so a team
// can split one block's panels between its members.
template <int NR, typename T>
void pack_b_panels(T* bp, const T* b, int64_t rs, int64_t cs, int64_t kc, int64_t nc, int64_t p0, int64_t p1);

extern template void pack_a<6, double>(double*, const double*, int64_t, int64_t, int64_t, int64_t);
extern template void pack_a<6, float>(float*, const float*, int64_t, int64_t, int64_t, int64_t);
extern template void pack_b_panels<8, double>(double*, const double*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
extern template void pack_b_panels<16, float>(float*, const float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

} // namespace pg::cpu
