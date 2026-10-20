#include "cpu/gemm_blis.hpp"

#include "cpu/aligned_alloc.hpp"
#include "cpu/microkernel.hpp"
#include "cpu/pack.hpp"
#include "cpu/threads.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
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
void microkernel(const T* ap, const T* bp, T* c, int64_t kc, int64_t ldc, int64_t mr, int64_t nr, bool first,
                 const T* b_next) {
    if constexpr (std::is_same_v<T, double>) {
        microkernel_f64_6x8(ap, bp, c, kc, ldc, mr, nr, first, b_next);
    } else {
        microkernel_f32_6x16(ap, bp, c, kc, ldc, mr, nr, first, b_next);
    }
}

// Each kernel call prefetches an eighth of a B micro-panel (microkernel.hpp), and
// MC / MR = 8 calls run per jr column at the default MC of 48.
constexpr int64_t B_SLICES = 8;

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

// One (jc, pc) step's progress, shared by the team: B panels claimed and packed,
// and members done reading the step's B buffer.
struct step_sync {
    std::atomic<int64_t> claimed;
    std::atomic<int64_t> packed;
    std::atomic<int64_t> done;
};

// Per calling thread for the reason the workspace is, grown and zeroed in serial
// code before the region.
class sync_steps {
public:
    step_sync* reset(int64_t count) {
        if (count > cap_) {
            p_.reset(new step_sync[static_cast<size_t>(count)]);
            cap_ = count;
        }
        for (int64_t s = 0; s < count; ++s) {
            p_[s].claimed.store(0, std::memory_order_relaxed);
            p_[s].packed.store(0, std::memory_order_relaxed);
            p_[s].done.store(0, std::memory_order_relaxed);
        }
        return p_.get();
    }

private:
    std::unique_ptr<step_sync[]> p_;
    int64_t cap_ = 0;
};

sync_steps& caller_sync() {
    thread_local sync_steps ss;
    return ss;
}

// A member waits for another's progress, which lags by microseconds up to one
// step's compute (milliseconds), so it spins on pause first; after about 4096
// pauses (a few hundred microseconds) it yields each round, so a core shared by
// several teams (concurrent callers) goes to the member being waited for.
void wait_until(const std::atomic<int64_t>& x, int64_t target) {
    for (int spins = 0; x.load(std::memory_order_acquire) < target;) {
        if (spins < 4096) {
            ++spins;
            _mm_pause();
        } else {
            std::this_thread::yield();
        }
    }
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

    const int64_t kb = ceil_div(k, KC);
    const int64_t steps = ceil_div(n, NC) * kb;
    const int64_t bp_count = kc_max * ceil_div(nc_max, nr_) * nr_;
    const int64_t ap_count = round_line<T>(std::min(MC, mp * MR) * kc_max);
    // Two B buffers let a member start packing the next step while the rest still
    // read the current one; a single step or a single thread needs one.
    const int64_t nbuf = nt > 1 && steps > 1 ? 2 : 1;
    T* const Bp = caller_workspace<T>().reserve(nbuf * bp_count + nt * ap_count);
    T* const ap_base = Bp + nbuf * bp_count;
    step_sync* const progress = caller_sync().reset(steps);
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

        // Steps run jc-major, pc-minor, every member through every step in order,
        // and a C tile belongs to one member for all of its pc steps, so its k
        // panels still accumulate serially. No team barrier: a member waits only
        // for the B panels it reads and for the buffer it overwrites, so one that
        // finishes its tiles early runs up to a step ahead and packs the panels a
        // slower member has not claimed. Measured at n = 2048, six threads, against
        // the two-barrier schedule with the same B prefetch: the mean wait per
        // member fell from 5.1 to 1.3 ms a call (f64) and from 3.4 to 0.8 ms (f32);
        // f32 ran 1.8 to 3.4 percent faster in three runs, f64 from 3.0 percent
        // faster to 0.9 percent slower, and the 1024 cells 0.3 to 5.0 percent faster.
        for (int64_t s = 0; s < steps; ++s) {
            const int64_t jc = s / kb * NC;
            const int64_t pc = s % kb * KC;
            const int64_t nc = std::min(NC, n - jc);
            const int64_t kc = std::min(KC, k - pc);
            const int64_t np = ceil_div(nc, nr_);
            T* const Bs = Bp + (nbuf == 2 ? s & 1 : 0) * bp_count;
            // The buffer last held step s - 2 (nbuf == 2); every member must be
            // done reading it before anyone repacks it.
            if (s >= 2) {
                wait_until(progress[s - 2].done, team);
            }
            for (int64_t q = progress[s].claimed.fetch_add(1, std::memory_order_relaxed); q < np;
                 q = progress[s].claimed.fetch_add(1, std::memory_order_relaxed)) {
                pack_b_panels<nr_>(Bs, b + pc * rs_b + jc * cs_b, rs_b, cs_b, kc, nc, q, q + 1);
                progress[s].packed.fetch_add(1, std::memory_order_release);
            }
            wait_until(progress[s].packed, np);

            const grid g = make_grid(team, mp, np);
            if (t < g.tm * g.tn) {
                const int64_t ti = t / g.tn;
                const int64_t tj = t % g.tn;
                const int64_t row0 = ti * mp / g.tm * MR;
                const int64_t row1 = std::min((ti + 1) * mp / g.tm * MR, m);
                const int64_t col0 = tj * np / g.tn * nr_;
                const int64_t col1 = std::min((tj + 1) * np / g.tn * nr_, nc);
                for (int64_t ic = row0; ic < row1; ic += MC) {
                    const int64_t mc = std::min(MC, row1 - ic);
                    pack_a<MR>(Ap, a + ic * rs_a + pc * cs_a, rs_a, cs_a, mc, kc);
                    for (int64_t jr = col0; jr < col1; jr += nr_) {
                        const int64_t nr = std::min(nr_, nc - jr);
                        const T* bpanel = Bs + (jr / nr_) * kc * nr_;
                        // The panel the next column reads; after the last column,
                        // the first, where the next ic block starts. The calls of
                        // this column fetch it slice by slice (a column of more than
                        // B_SLICES calls starts over, and a short edge block leaves
                        // the rest to the hardware).
                        const T* bnext = jr + nr_ < col1 ? bpanel + kc * nr_ : Bs + (col0 / nr_) * kc * nr_;
                        for (int64_t ir = 0; ir < mc; ir += MR) {
                            const T* slice = bnext + (ir / MR) % B_SLICES * (kc * nr_ / B_SLICES);
                            microkernel<T>(Ap + ir * kc, bpanel, c + (ic + ir) * n + (jc + jr), kc, n,
                                           std::min(MR, mc - ir), nr, pc == 0, slice);
                        }
                    }
                }
            }
            progress[s].done.fetch_add(1, std::memory_order_release);
        }
    }
}

template void gemm_blis<float>(const float*, int64_t, int64_t, const float*, int64_t, int64_t,
                               float*, int64_t, int64_t, int64_t);
template void gemm_blis<double>(const double*, int64_t, int64_t, const double*, int64_t, int64_t,
                                double*, int64_t, int64_t, int64_t);

} // namespace pg::cpu
