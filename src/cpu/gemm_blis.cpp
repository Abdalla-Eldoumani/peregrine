#include "cpu/gemm_blis.hpp"

#include "cpu/aligned_alloc.hpp"
#include "cpu/microkernel.hpp"
#include "cpu/pack.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <type_traits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pg::cpu {

namespace {

// Sweep winner on the reference machine (i7-10750H, 6 threads pinned):
// MC=96 KC=320 NC=2048. Ap = 96x320x8 = 240KB fits the 256K L2; Bp panel =
// 320x8x8 = 20KB sits in the 32K L1d; NC=2048 is a single jc block at every
// bench size. Chosen on the noise-robust median-of-min GFLOP/s across a
// randomized-order, cooldown-separated f64 n=1024 run (background CPU contention
// on the machine inflated the coefficient of variation, so the spike-resistant
// floor decided, not the raw median). Measured f64 n=1024 floors, GF/s:
//   mc96 kc320 nc2048 = 55.4 (median floor) / 83.9 (best min) -- winner
//   mc72 kc256 nc2048 = 52.3 / 80.0
//   mc48 kc256 nc2048 = 47.9 / 92.5   mc48 kc192 nc2048 = 41.2 / 78.0
//   mc72 kc256 nc4080 = 39.2 (the prior {72,256,4080} default) / 64.6
// Runtime-settable so a sweep walks the MC{48,72,96,144} x KC{192,256,320} x
// NC{2048,4080} grid in one process; re-sweep when the machine or toolchain moves.
//
// std::atomic so the sweep hook can publish a new triple while a kernel reads
// it: blocking is trivially copyable (three int64_t), so the whole struct moves
// in one atomic store/load and no reader ever sees a torn (mc-new, nc-old) pair.
// On a 24-byte struct this is not lock-free, but it is correct and off any hot
// path (the kernel loads it once per call, never per tile).
std::atomic<blocking> g_blocking{blocking{96, 320, 2048}};

} // namespace

void store_blocking(blocking b) noexcept {
    g_blocking.store(b, std::memory_order_relaxed);
}

blocking load_blocking() noexcept {
    return g_blocking.load(std::memory_order_relaxed);
}

namespace {

// RAII owner for an aligned panel buffer so a throw between allocation and the
// trailing free does not leak it, and so the throwing aligned_new never sits on
// the OpenMP region's unwinding path. An exception that tries to leave an omp
// structured block is undefined behavior (std::terminate on both the MSVC
// /openmp:llvm and GCC runtimes); every Ap is therefore allocated here, in
// serial code, before the region, where std::bad_alloc unwinds normally to the
// binding and surfaces as MemoryError.
template <typename T>
struct aligned_buf {
    T* p;
    explicit aligned_buf(int64_t n) : p(aligned_new<T>(n)) {}
    ~aligned_buf() { aligned_delete(p); }
    aligned_buf(const aligned_buf&) = delete;
    aligned_buf& operator=(const aligned_buf&) = delete;
    aligned_buf(aligned_buf&& o) noexcept : p(o.p) { o.p = nullptr; }
    aligned_buf& operator=(aligned_buf&&) = delete;
};

// The register-tile shape the f64 microkernel computes. Structural, shared with
// microkernel_avx2_f64.cpp and pack.cpp; not a tuning knob, so it is a local
// constant rather than a member of the runtime blocking struct.
constexpr int64_t MR = 6;
constexpr int64_t NR = 8;

// The f32 register-tile shape, chosen by measurement at n=512: 8x16 beat 6x16,
// so the f32 path runs microkernel_f32_8x16 with MR_F32=8, NR_F32=16. Both shapes
// were built and benched single-threaded at n=512 on the reference machine
// (i7-10750H, OMP_WAIT_POLICY=ACTIVE). Background CPU contention held the
// coefficient of variation above the 5% gate, so the spike-resistant best-min
// floor over repeated cooldown-separated runs decides, not the raw median (the
// same noise-handling used for the f64 sweep). Measured f32 n=512 best-min floor
// / top-3-mean floor, GF/s, two independent rounds:
//   8x16 = 57.5 / 57.0 (round 2), 55.1 (round 1 best-min) -- winner
//   6x16 = 53.2 / 52.9 (round 2), 53.6 (round 1 best-min) -- loser
// The register-budget argument predicted the opposite (8x16 needs 16 ymm
// accumulators, the whole file, so B is reloaded each k-step instead of held in
// a register the way 6x16's 12-accumulator budget allows). Measurement overruled
// it: 8x16's higher arithmetic density (16 FMAs per k-step) amortizes the B
// reloads and wins by ~8% on this core's port layout -- decide f32 shape by
// measuring, not by counting registers. To reproduce the loser, define
// PG_F32_SHAPE_6x16 and rebuild. Re-measure on a quiet machine before treating
// either number as absolute throughput.
#if defined(PG_F32_SHAPE_6x16)
constexpr int64_t MR_F32 = 6;
#else
constexpr int64_t MR_F32 = 8;
#endif
constexpr int64_t NR_F32 = 16;

// f32 panel packers, local to this TU because pack.cpp's pack_a/pack_b are
// compiled with the f64 MR=6/NR=8 layout and the f32 tile needs MR_F32/NR_F32.
// Identical structure to pack.cpp (strided copy, zero-pad the edge rows/cols to
// the tile extent so the k loop never branches), specialized to the f32 shape.
// No intrinsics: a plain copy, kept here only so the f32 shape decision lives
// entirely in this file and microkernel_avx2_f32.cpp.
void pack_a_f32(float* ap, const float* a, int64_t ic, int64_t pc, int64_t mc, int64_t kc, int64_t k) {
    const float* a_block = a + ic * k + pc;
    int64_t dst = 0;
    for (int64_t i0 = 0; i0 < mc; i0 += MR_F32) {
        const int64_t rows = (mc - i0 < MR_F32) ? (mc - i0) : MR_F32;
        for (int64_t p = 0; p < kc; ++p) {
            for (int64_t r = 0; r < rows; ++r) {
                ap[dst++] = a_block[(i0 + r) * k + p];
            }
            for (int64_t r = rows; r < MR_F32; ++r) {
                ap[dst++] = 0.0f;
            }
        }
    }
}

void pack_b_f32(float* bp, const float* b, int64_t pc, int64_t jc, int64_t kc, int64_t nc, int64_t n) {
    const float* b_block = b + pc * n + jc;
    int64_t dst = 0;
    for (int64_t j0 = 0; j0 < nc; j0 += NR_F32) {
        const int64_t cols = (nc - j0 < NR_F32) ? (nc - j0) : NR_F32;
        for (int64_t p = 0; p < kc; ++p) {
            for (int64_t c = 0; c < cols; ++c) {
                bp[dst++] = b_block[p * n + (j0 + c)];
            }
            for (int64_t c = cols; c < NR_F32; ++c) {
                bp[dst++] = 0.0f;
            }
        }
    }
}

inline void microkernel_f32(const float* ap, const float* bp, float* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr) {
#if defined(PG_F32_SHAPE_6x16)
    microkernel_f32_6x16(ap, bp, c, kc, ldc, mr, nr);
#else
    microkernel_f32_8x16(ap, bp, c, kc, ldc, mr, nr);
#endif
}

// Below this max dimension the pack buffers and the thread spawn cost more than
// they save, so a direct register-blocked pass with no heap and no threading
// wins. 96 is a measurement-validated default; it is a default, not a calibrated
// threshold. The branch lives inside the blis entry by design so dispatch stays a
// pure features-in/path-out decision.
constexpr int64_t SMALL_MAX_DIM = 96;

// Both dtypes have a packed microkernel, so the five-loop path is instantiated
// for double and float; the small direct path carries either below
// SMALL_MAX_DIM. Both paths accumulate over k
// strictly sequentially per output element, which keeps each element's rounding
// order independent of how the ic loop is later split across threads: that is
// what preserves the bitwise thread-count invariance the suite asserts.
template <typename T>
void gemm_small(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n) {
    for (int64_t i = 0; i < m; ++i) {
        T* ci = c + i * n;
        for (int64_t p = 0; p < k; ++p) {
            const T aval = a[i * k + p];
            const T* bp = b + p * n;
            for (int64_t j = 0; j < n; ++j) {
                ci[j] += aval * bp[j];
            }
        }
    }
}

} // namespace

template <typename T>
void gemm_blis(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n) {
    if (m == 0 || n == 0) {
        return;
    }
    std::memset(c, 0, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T));
    if (k == 0) {
        return; // NumPy semantics: (m, 0) @ (0, n) is an m x n zero matrix
    }

    if (std::max(m, std::max(k, n)) <= SMALL_MAX_DIM) {
        gemm_small<T>(a, b, c, m, k, n);
        return;
    }

    // Both dtypes now have a packed five-loop path: f64 runs the 6x8 microkernel,
    // f32 runs the measured winning shape at n=512 (MR_F32 x 16). The two paths share
    // the loop structure, the runtime blocking, and the OpenMP region; only the
    // pack layout and the microkernel callee differ by dtype, dispatched here at
    // compile time so neither path carries a runtime dtype branch in its hot loop.
    if constexpr (std::is_same_v<T, float>) {
        // Snapshot the blocking once, before the GIL-free loops: a concurrent
        // _set_gemm_blocking publishes a whole new triple atomically, so MC/KC/NC
        // below are mutually consistent and fixed for the duration of this call.
        const blocking blk = load_blocking();
        const int64_t MC = blk.mc;
        const int64_t KC = blk.kc;
        const int64_t NC = blk.nc;

        const int64_t ap_cap = ((MC + MR_F32 - 1) / MR_F32) * MR_F32 * KC;
        const int64_t bp_cap = ((NC + NR_F32 - 1) / NR_F32) * NR_F32 * KC;
        aligned_buf<float> bp_owner(bp_cap);
        float* Bp = bp_owner.p;

        // One Ap per thread, all allocated here in serial code: aligned_new can
        // throw std::bad_alloc, and a throw must never cross the omp region below
        // (UB -> std::terminate). Inside the region each thread reads its own
        // buffer by omp_get_thread_num(), so no allocation happens on the
        // unwinding path. omp_get_max_threads() is the upper bound the region can
        // use; the non-OpenMP build needs exactly one.
#if defined(_OPENMP)
        const int num_threads = omp_get_max_threads();
#else
        const int num_threads = 1;
#endif
        std::vector<aligned_buf<float>> ap_pool;
        ap_pool.reserve(static_cast<size_t>(num_threads));
        for (int t = 0; t < num_threads; ++t) {
            ap_pool.emplace_back(ap_cap);
        }

        for (int64_t jc = 0; jc < n; jc += NC) {
            const int64_t nc = std::min(NC, n - jc);
            for (int64_t pc = 0; pc < k; pc += KC) {
                const int64_t kc = std::min(KC, k - pc);
                pack_b_f32(Bp, b, pc, jc, kc, nc, n);

                const int64_t num_ic = (m + MC - 1) / MC;
#if defined(_OPENMP)
#pragma omp parallel
#endif
                {
#if defined(_OPENMP)
                    float* Ap = ap_pool[static_cast<size_t>(omp_get_thread_num())].p;
#else
                    float* Ap = ap_pool[0].p;
#endif
#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
                    for (int64_t icb = 0; icb < num_ic; ++icb) {
                        const int64_t ic = icb * MC;
                        const int64_t mc = std::min(MC, m - ic);
                        pack_a_f32(Ap, a, ic, pc, mc, kc, k);
                        for (int64_t jr = 0; jr < nc; jr += NR_F32) {
                            const int64_t nr = std::min(NR_F32, nc - jr);
                            const float* bpanel = Bp + (jr / NR_F32) * kc * NR_F32;
                            for (int64_t ir = 0; ir < mc; ir += MR_F32) {
                                const int64_t mr = std::min(MR_F32, mc - ir);
                                const float* apanel = Ap + (ir / MR_F32) * MR_F32 * kc;
                                float* ctile = c + (ic + ir) * n + (jc + jr);
                                microkernel_f32(apanel, bpanel, ctile, kc, n, mr, nr);
                            }
                        }
                    }
                }
            }
        }

        return;
    } else {
        // Snapshot the blocking once, before the GIL-free loops (see the f32 arm).
        const blocking blk = load_blocking();
        const int64_t MC = blk.mc;
        const int64_t KC = blk.kc;
        const int64_t NC = blk.nc;

        // Ap holds one MC x KC block as ceil(MC/MR) row panels of MR x KC; Bp holds
        // one KC x NC block as ceil(NC/NR) column panels of KC x NR. Both are sized
        // for the maximum (full-block) extents and reused across edge blocks; the
        // packers zero-pad the live sub-block into the same layout. Bp is shared
        // and read-only inside the parallel region (packed once per (jc, pc) by the
        // serial thread that owns the loop). Ap is written by every thread that
        // packs its own A panel, so each thread needs a private buffer; they are
        // all allocated below in serial code (one per thread), never inside the
        // region, because aligned_new can throw and a throw must not cross the omp
        // boundary (UB -> std::terminate).
        const int64_t ap_cap = ((MC + MR - 1) / MR) * MR * KC;
        const int64_t bp_cap = ((NC + NR - 1) / NR) * NR * KC;
        aligned_buf<double> bp_owner(bp_cap);
        double* Bp = bp_owner.p;

        // Per-thread Ap pool, allocated here so std::bad_alloc unwinds to the
        // binding (MemoryError) instead of terminating from inside the region.
#if defined(_OPENMP)
        const int num_threads = omp_get_max_threads();
#else
        const int num_threads = 1;
#endif
        std::vector<aligned_buf<double>> ap_pool;
        ap_pool.reserve(static_cast<size_t>(num_threads));
        for (int t = 0; t < num_threads; ++t) {
            ap_pool.emplace_back(ap_cap);
        }

        // jc has a single block at n <= NC (every bench size), so all of the
        // thread scaling has to come from the ic loop: a jc-only parallel region would
        // measure 1.0x because there is one jc iteration. The pc loop is the serial
        // k-block reduction and is NEVER parallelized, so each output element
        // accumulates its k-blocks in a fixed order independent of the thread that
        // owns its ic block. That is what keeps the result bitwise identical across
        // thread counts (the 1-vs-12 identity tests/test_threads.py guards).
        for (int64_t jc = 0; jc < n; jc += NC) {
            const int64_t nc = std::min(NC, n - jc);
            for (int64_t pc = 0; pc < k; pc += KC) {
                const int64_t kc = std::min(KC, k - pc);
                pack_b<double>(Bp, b, pc, jc, kc, nc, n);

                const int64_t num_ic = (m + MC - 1) / MC;
                // The parallel region wraps the ic loop. Each thread owns a disjoint
                // set of ic blocks (disjoint C row ranges, so no two threads write
                // the same C element), packs them into its own private Ap from the
                // pool, and runs the macro kernel. schedule(dynamic) balances the
                // ragged final ic block across threads. The pragmas are guarded so
                // the kernel still builds and runs single-threaded without an
                // OpenMP runtime.
#if defined(_OPENMP)
#pragma omp parallel
#endif
                {
#if defined(_OPENMP)
                    double* Ap = ap_pool[static_cast<size_t>(omp_get_thread_num())].p;
#else
                    double* Ap = ap_pool[0].p;
#endif
#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
                    for (int64_t icb = 0; icb < num_ic; ++icb) {
                        const int64_t ic = icb * MC;
                        const int64_t mc = std::min(MC, m - ic);
                        pack_a<double>(Ap, a, ic, pc, mc, kc, k);
                        for (int64_t jr = 0; jr < nc; jr += NR) {
                            const int64_t nr = std::min(NR, nc - jr);
                            const double* bpanel = Bp + (jr / NR) * kc * NR;
                            for (int64_t ir = 0; ir < mc; ir += MR) {
                                const int64_t mr = std::min(MR, mc - ir);
                                const double* apanel = Ap + (ir / MR) * MR * kc;
                                double* ctile = c + (ic + ir) * n + (jc + jr);
                                microkernel_f64_6x8(apanel, bpanel, ctile, kc, n, mr, nr);
                            }
                        }
                    }
                }
            }
        }
    }
}

template void gemm_blis<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
template void gemm_blis<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

} // namespace pg::cpu
