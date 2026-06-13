#include "cpu/gemm_blis.hpp"

#include "cpu/aligned_alloc.hpp"
#include "cpu/microkernel.hpp"
#include "cpu/pack.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace fme::cpu {

blocking& current_blocking() {
    // Sweep winner on zpicy (i7-10750H, 6 threads pinned, Ultimate Performance):
    // MC=96 KC=320 NC=2048. Ap = 96x320x8 = 240KB fits the 256K L2; Bp panel =
    // 320x8x8 = 20KB sits in the 32K L1d; NC=2048 is a single jc block at every
    // bench size. Chosen on the noise-robust median-of-min GFLOP/s across a
    // randomized-order, cooldown-separated f64 n=1024 run (a background antivirus
    // on this machine inflated CV above the protocol gate, so the spike-resistant
    // floor decided, not the raw median). Measured f64 n=1024 floors, GF/s:
    //   mc96 kc320 nc2048 = 55.4 (median floor) / 83.9 (best min) -- winner
    //   mc72 kc256 nc2048 = 52.3 / 80.0
    //   mc48 kc256 nc2048 = 47.9 / 92.5   mc48 kc192 nc2048 = 41.2 / 78.0
    //   mc72 kc256 nc4080 = 39.2 (the prior {72,256,4080} default) / 64.6
    // Full 24-point MC{48,72,96,144} x KC{192,256,320} x NC{2048,4080} grid is in
    // benchmarks/results/tuning/sweep_zpicy.json. Runtime-settable so the sweep
    // walks the grid in one process; re-sweep when the machine or toolchain moves.
    static blocking b{96, 320, 2048};
    return b;
}

namespace {

// The register-tile shape the f64 microkernel computes. Structural, shared with
// microkernel_avx2_f64.cpp and pack.cpp; not a tuning knob, so it is a local
// constant rather than a member of the runtime blocking struct.
constexpr int64_t MR = 6;
constexpr int64_t NR = 8;

// Below this max dimension the pack buffers and (in 03-04) the thread spawn cost
// more than they save, so a direct register-blocked pass with no heap and no
// threading wins. 96 is a measurement-validated default (the small-path bench in
// 03-05 may move it); it is a default, not a calibrated threshold, which is a
// Phase 5 concern. The branch lives inside the blis entry by design so dispatch
// stays a pure features-in/path-out decision.
constexpr int64_t SMALL_MAX_DIM = 96;

// f64 only has a packed microkernel this plan (the f32 shape is decided by
// measurement in 03-05), so the five-loop path is instantiated for double and
// the small direct path carries float until then. Both paths accumulate over k
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

    // The packed five-loop path is f64-only this plan; the f32 microkernel shape
    // is chosen by measurement in 03-05. Until then any f32 input above the small
    // threshold takes the same direct pass (correct, just unblocked) so the
    // routed kernel is correct for every dtype and size today.
    if constexpr (!std::is_same_v<T, double>) {
        gemm_small<T>(a, b, c, m, k, n);
        return;
    } else {
        const blocking& blk = current_blocking();
        const int64_t MC = blk.mc;
        const int64_t KC = blk.kc;
        const int64_t NC = blk.nc;

        // Ap holds one MC x KC block as ceil(MC/MR) row panels of MR x KC; Bp holds
        // one KC x NC block as ceil(NC/NR) column panels of KC x NR. Both are sized
        // for the maximum (full-block) extents and reused across edge blocks; the
        // packers zero-pad the live sub-block into the same layout. Bp is shared
        // and read-only inside the parallel region (packed once per (jc, pc) by the
        // serial thread that owns the loop), so it is allocated here; Ap is written
        // by every thread that packs its own A panel, so each thread allocates a
        // private Ap inside the region (below).
        const int64_t ap_cap = ((MC + MR - 1) / MR) * MR * KC;
        const int64_t bp_cap = ((NC + NR - 1) / NR) * NR * KC;
        double* Bp = aligned_new<double>(bp_cap);

        // jc has a single block at n <= NC (every bench size), so all of CPU-05's
        // scaling has to come from the ic loop: a jc-only parallel region would
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
                // the same C element), packs them into its own private Ap, and runs
                // the macro kernel. schedule(dynamic) balances the ragged final ic
                // block across threads. The pragmas are guarded so the kernel still
                // builds and runs single-threaded without an OpenMP runtime.
#if defined(_OPENMP)
#pragma omp parallel
#endif
                {
                    double* Ap = aligned_new<double>(ap_cap);
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
                    aligned_delete(Ap);
                }
            }
        }

        aligned_delete(Bp);
    }
}

template void gemm_blis<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
template void gemm_blis<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

} // namespace fme::cpu
