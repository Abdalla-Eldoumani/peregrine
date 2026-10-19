#pragma once
#include <cstdint>

namespace pg::cpu {

// The AVX2+FMA matmul paths dispatch chooses between, defined in the
// AVX2-flagged gemm_blis.cpp; this header stays intrinsic-free so the fallback
// path links no AVX2 codegen. Both take gemm_naive's strided signature: element
// (i, p) of A at a[i * rs_a + p * cs_a], element (p, j) of B at b[p * rs_b + j *
// cs_b], any stride (negative or zero included), and C C-contiguous with ldc =
// n, overwritten.
//
// gemm_blis is the packed five-loop engine, threaded; gemm_blis_small is the
// direct single-threaded pass for products too small to repay packing. Each
// computes any shape correctly; which one runs is dispatch's call, made on the
// shape alone.
template <typename T>
void gemm_blis(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
               T* c, int64_t m, int64_t k, int64_t n);

template <typename T>
void gemm_blis_small(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
                     T* c, int64_t m, int64_t k, int64_t n);

extern template void gemm_blis<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t,
                                      float*, int64_t, int64_t, int64_t);
extern template void gemm_blis<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t,
                                       double*, int64_t, int64_t, int64_t);
extern template void gemm_blis_small<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t,
                                            float*, int64_t, int64_t, int64_t);
extern template void gemm_blis_small<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t,
                                             double*, int64_t, int64_t, int64_t);

// MC, KC, NC are loop bounds, not unroll factors, so making them runtime-settable
// costs nothing at execution time and lets a sweep walk its grid in one process
// instead of rebuilding per point. Each dtype has its own triple because the
// register tiles differ (6x8 f64, 6x16 f32). MR and NR stay compile-time because
// they are the structural register-tile shape. KC changes the pc-chunking of
// each element's accumulation, so results differ bitwise across KC values within
// tolerance; thread-count invariance at a fixed blocking is unaffected.
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
// f32 selects the float32 triple, otherwise the float64 one.
void store_blocking(blocking b, bool f32) noexcept;
blocking load_blocking(bool f32) noexcept;

} // namespace pg::cpu
