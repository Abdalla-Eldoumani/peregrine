#include "cpu/gemm_blis.hpp"

#include "cpu/aligned_alloc.hpp"
#include "cpu/microkernel.hpp"
#include "cpu/pack.hpp"
#include "cpu/threads.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <type_traits>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pg::cpu {

namespace {

// Per-dtype blocking from a 6-thread sweep on an i7-10750H at n = 512, 1024,
// and 2048 (benchmarks/results/tuning/sweep_refbox_t6_*.json: MC {48, 72, 96,
// 144, 168} x KC {192, 256, 320, 384} x NC {2048, 4080}, mirrored walks, median
// ranking). f64 {48, 256, 2048} is within the winner's IQR at every size
// (geometric mean 0.980 of the per-size best; the BLIS Haswell start {72, 256,
// 4080} was 0.960 and 8 percent behind at 1024); the one config ranked above it
// needs MC 168, whose A block (344 KiB) overflows the 256 KiB L2. f32 {48, 256,
// 4080} is within the IQR everywhere (0.997; the start {168, 256, 4080} was
// 0.948). Cache budget: the A and B micro-panels, MR x KC plus KC x NR, are
// 12 + 16 = 28 KiB for f64 and 6 + 16 = 22 KiB for f32, inside the 32 KiB L1d;
// the MC x KC A block is 96 KiB for f64 and 48 KiB for f32, inside L2; KC x NC
// of B is 4 MiB either way, inside the 12 MiB L3.
//
// std::atomic so the sweep hook can publish a new triple while a kernel reads
// it: blocking is trivially copyable (three int64_t), so the whole struct moves
// in one atomic store/load and no reader ever sees a torn (mc-new, nc-old) pair.
// On a 24-byte struct this is not lock-free, but it is correct and off any hot
// path (the kernel loads it once per call, never per tile).
std::atomic<blocking> g_blocking_f64{blocking{48, 256, 2048}};
std::atomic<blocking> g_blocking_f32{blocking{48, 256, 4080}};

} // namespace

void store_blocking(blocking b, bool f32) noexcept {
    (f32 ? g_blocking_f32 : g_blocking_f64).store(b, std::memory_order_relaxed);
}

blocking load_blocking(bool f32) noexcept {
    return (f32 ? g_blocking_f32 : g_blocking_f64).load(std::memory_order_relaxed);
}

namespace {

// The register tile: f64 6x8 and f32 6x16, twelve accumulators either way.
constexpr int64_t MR = 6;
template <typename T>
constexpr int64_t NR = std::is_same_v<T, double> ? 8 : 16;

// Threading thresholds in flops (2mkn), from best-of single-call timings on the
// i7-10750H (vcomp, one logical CPU per core, warm team): extra threads stop
// paying below about 1.5e5 to 3e5 flops (97x8x97 and 100x2x100 gain at most a
// microsecond; 200x4x200 runs 18 us on one thread, 5 us on six), and a member
// with under about 2e5 flops adds fork and barrier time without saving any.
// Neither value changes a result bit, only how many threads share the work.
constexpr double SERIAL_FLOPS = 2.5e5;
constexpr double PER_THREAD_FLOPS = 2.0e5;

template <typename T>
void microkernel(const T* ap, const T* bp, T* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr, bool first) {
    if constexpr (std::is_same_v<T, double>) {
        microkernel_f64_6x8(ap, bp, c, kc, ldc, mr, nr, first);
    } else {
        microkernel_f32_6x16(ap, bp, c, kc, ldc, mr, nr, first);
    }
}

// The packing buffers, owned by the OS thread that calls in: each Python
// thread is the root of its own OpenMP team, so a pool shared by thread number
// would hand two concurrent calls the same buffer. It grows (never shrinks) in
// serial code before the parallel region, so std::bad_alloc unwinds normally to
// the binding (MemoryError); a throw out of an OpenMP region is undefined
// behavior. Workers only use the pointers they are handed.
template <typename T>
class workspace {
public:
    workspace() = default;
    ~workspace() { aligned_delete(p_); }
    workspace(const workspace&) = delete;
    workspace& operator=(const workspace&) = delete;

    T* reserve(int64_t count) {
        if (count > cap_) {
            T* p = aligned_new<T>(count);
            aligned_delete(p_);
            p_ = p;
            cap_ = count;
        }
        return p_;
    }

private:
    T* p_ = nullptr;
    int64_t cap_ = 0;
};

template <typename T>
workspace<T>& caller_workspace() {
    thread_local workspace<T> ws;
    return ws;
}

constexpr int64_t ceil_div(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

// Elements per 64 bytes, so every Ap slot starts on a cache line.
template <typename T>
constexpr int64_t round_line(int64_t count) {
    constexpr int64_t line = 64 / static_cast<int64_t>(sizeof(T));
    return ceil_div(count, line) * line;
}

// Splits the team into tm row groups by tn column groups (tm * tn <= team)
// over mp MR-row panels and np NR-column panels, minimizing the tiles on the
// busiest thread; ties go to more row groups, which pack less A twice.
struct grid {
    int64_t tm, tn;
};

grid make_grid(int64_t team, int64_t mp, int64_t np) {
    grid best{1, 1};
    int64_t best_tiles = -1;
    for (int64_t tm = 1; tm <= team && tm <= mp; ++tm) {
        const int64_t tn = std::min(team / tm, np);
        const int64_t tiles = ceil_div(mp, tm) * ceil_div(np, tn);
        if (best_tiles < 0 || tiles <= best_tiles) {
            best = {tm, tn};
            best_tiles = tiles;
        }
    }
    return best;
}

} // namespace

// The five loops (jc over NC columns, pc over KC steps, ic over MC rows, jr over
// NR columns, ir over MR rows) inside one parallel region per call. Every C
// element is computed as one FMA chain from zero per KC panel in k order, the
// first panel stored and each later one added, all by whichever thread owns the
// element's tile: the result depends on the shape and KC alone, never on the
// team size, the partition, MC, NC, or the operands' layout (packing reads any
// strides into the same panels).
template <typename T>
void gemm_blis(const T* a, int64_t rs_a, int64_t cs_a, const T* b, int64_t rs_b, int64_t cs_b,
               T* c, int64_t m, int64_t k, int64_t n) {
    if (m == 0 || n == 0) {
        return;
    }
    // The first k panel stores each tile outright, so C is never zeroed; only a
    // k == 0 product (NumPy: an m x n zero matrix) has no panel to store.
    if (k == 0) {
        std::memset(c, 0, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(T));
        return;
    }

    constexpr int64_t nr_ = NR<T>;
    // One snapshot per call: a concurrent _set_gemm_blocking publishes a whole
    // new triple, and this call keeps the one it started with.
    const blocking blk = load_blocking(std::is_same_v<T, float>);
    const int64_t KC = blk.kc;
    const int64_t NC = blk.nc;
    // Row blocks stay whole MR panels, so a tile never straddles two threads.
    const int64_t MC = std::max(MR, blk.mc / MR * MR);

    const int64_t mp = ceil_div(m, MR);
    const int64_t kc_max = std::min(KC, k);
    const int64_t nc_max = std::min(NC, n);
    const int nt = team_size(2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n),
                             SERIAL_FLOPS, PER_THREAD_FLOPS, mp * ceil_div(nc_max, nr_));

    const int64_t bp_count = kc_max * ceil_div(nc_max, nr_) * nr_;
    const int64_t ap_count = round_line<T>(std::min(MC, mp * MR) * kc_max);
    T* const Bp = caller_workspace<T>().reserve(bp_count + nt * ap_count);
    T* const ap_base = Bp + bp_count;
    const unsigned csr = mxcsr();

#if defined(_OPENMP)
#pragma omp parallel num_threads(nt) if(nt > 1)
#endif
    {
        const mxcsr_scope fp(csr);
#if defined(_OPENMP)
        // The team actually granted, which nesting or a dynamic runtime can make
        // smaller than nt: partitioning by nt would leave tiles unowned.
        const int64_t team = omp_get_num_threads();
        const int64_t t = omp_get_thread_num();
#else
        const int64_t team = 1;
        const int64_t t = 0;
#endif
        T* const Ap = ap_base + t * ap_count;

        for (int64_t jc = 0; jc < n; jc += NC) {
            const int64_t nc = std::min(NC, n - jc);
            const int64_t np = ceil_div(nc, nr_);
            const grid g = make_grid(team, mp, np);
            const bool computes = t < g.tm * g.tn;
            const int64_t ti = t / g.tn;
            const int64_t tj = t % g.tn;
            const int64_t row0 = ti * mp / g.tm * MR;
            const int64_t row1 = std::min((ti + 1) * mp / g.tm * MR, m);
            const int64_t col0 = tj * np / g.tn * nr_;
            const int64_t col1 = std::min((tj + 1) * np / g.tn * nr_, nc);

            for (int64_t pc = 0; pc < k; pc += KC) {
                const int64_t kc = std::min(KC, k - pc);
                // Every member packs its share of this block's B panels; the
                // barrier publishes the whole block before anyone reads it.
                pack_b_panels<nr_>(Bp, b + pc * rs_b + jc * cs_b, rs_b, cs_b, kc, nc,
                                   t * np / team, (t + 1) * np / team);
#if defined(_OPENMP)
#pragma omp barrier
#endif
                if (computes) {
                    for (int64_t ic = row0; ic < row1; ic += MC) {
                        const int64_t mc = std::min(MC, row1 - ic);
                        pack_a<MR>(Ap, a + ic * rs_a + pc * cs_a, rs_a, cs_a, mc, kc);
                        for (int64_t jr = col0; jr < col1; jr += nr_) {
                            const int64_t nr = std::min(nr_, nc - jr);
                            const T* bpanel = Bp + (jr / nr_) * kc * nr_;
                            for (int64_t ir = 0; ir < mc; ir += MR) {
                                microkernel<T>(Ap + ir * kc, bpanel, c + (ic + ir) * n + (jc + jr), kc, n,
                                               std::min(MR, mc - ir), nr, pc == 0);
                            }
                        }
                    }
                }
                // Nobody repacks Bp while another member still reads it. The
                // region's closing barrier covers the last block.
                if (pc + KC < k || jc + NC < n) {
#if defined(_OPENMP)
#pragma omp barrier
#endif
                }
            }
        }
    }
}

template void gemm_blis<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t,
                               float*, int64_t, int64_t, int64_t);
template void gemm_blis<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t,
                                double*, int64_t, int64_t, int64_t);

} // namespace pg::cpu
