#include "cpu/reduce.hpp"

#include "cpu/threads.hpp"

#include <memory>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pg::cpu {

namespace {

// Pairwise summation, NumPy's algorithm class: error grows as O(log n) * eps
// instead of the O(n) * eps of a running sum. The eight lanes keep the add
// pipeline full instead of waiting on one accumulator's latency chain. This is
// the v3.0.0 algorithm, unchanged, and the order the AVX2 kernels reproduce.
template <typename T>
T pairwise_sum(const T* x, int64_t n) {
    if (n <= 128) {
        T acc[8] = {T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0)};
        int64_t i = 0;
        const int64_t blocked = n & ~int64_t{7};
        for (; i < blocked; i += 8) {
            acc[0] += x[i + 0];
            acc[1] += x[i + 1];
            acc[2] += x[i + 2];
            acc[3] += x[i + 3];
            acc[4] += x[i + 4];
            acc[5] += x[i + 5];
            acc[6] += x[i + 6];
            acc[7] += x[i + 7];
        }
        T s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; i < n; ++i) {
            s += x[i];
        }
        return s;
    }
    const int64_t n2 = (n / 2) & ~int64_t{7};
    return pairwise_sum(x, n2) + pairwise_sum(x + n2, n - n2);
}

// Rows per axis-0 block: up to this many rows the axis-0 sum is the sequential
// order v3.0.0 and NumPy use, bit for bit.
constexpr int64_t ROW_BLOCK = 128;

// A pairwise sum is cut into 2^d subtrees of at least this many elements, at
// most 64 of them, so a static split over six threads stays within about one
// subtree of even. Rows are cut too while there are fewer than ROW_UNITS.
constexpr int64_t CHUNK_MIN = 8192;
constexpr int CHUNK_DEPTH_MAX = 6;
constexpr int64_t ROW_UNITS = 64;

// Axis 0 reads each row of a block in panels of PANEL_MIN to PANEL_MAX
// elements: a longer contiguous run per row keeps the hardware prefetchers
// streaming, and in single-call diagnostics at (4096, 4096) float64 full-row
// panels ran 4.5-4.7 ms against 5.6 ms for 4 KiB ones (noisy). One panel per
// level of the block tree sits in a stack buffer of LEVEL_ELEMS (64 KiB for
// float64). COL_UNITS is the number of (row chunk, panel) pieces to aim for.
constexpr int64_t LEVEL_ELEMS = 8192;
constexpr int64_t PANEL_MIN = 512;
constexpr int64_t PANEL_MAX = 4096;
constexpr int64_t COL_UNITS = 48;

// Team sizing in bytes read. Single-call diagnostics, sum over every element:
// float64 128 KiB took 3.1 us on one thread and 2.2 on two, 256 KiB 6.3 on one
// and 2.2 on four or six; float32 64 KiB lost to one thread. Below 128 KiB the
// fork and join cost more than the split saves.
constexpr double SERIAL_BELOW = 128.0 * 1024;
constexpr double PER_THREAD = 64.0 * 1024;

template <typename T>
void rows_kernel(bool avx2, const T* x, int64_t n, int64_t rows, T* out) {
    if (avx2) {
        sum_rows_avx2<T>(x, n, rows, out);
    } else {
        sum_rows_naive<T>(x, n, rows, out);
    }
}

template <typename T>
void block_kernel(bool avx2, const T* x, int64_t ld, int64_t rows, int64_t w, T* acc) {
    if (avx2) {
        sum_block_avx2<T>(x, ld, rows, w, acc);
    } else {
        sum_block_naive<T>(x, ld, rows, w, acc);
    }
}

// How many levels of the pairwise tree over n elements to cut into chunks.
// Every node above the cut is far longer than a 128-element leaf, so each one
// really splits and the chunks are subtrees of the serial tree.
int chunk_depth(int64_t n) {
    int d = 0;
    while (d < CHUNK_DEPTH_MAX && (n >> (d + 1)) >= CHUNK_MIN) {
        ++d;
    }
    return d;
}

// Start offsets of the 2^d subtrees at depth d of the pairwise tree over
// [base, base + n), left to right.
void chunk_offsets(int64_t base, int64_t n, int d, int64_t*& off) {
    if (d == 0) {
        *off++ = base;
        return;
    }
    const int64_t n2 = (n / 2) & ~int64_t{7};
    chunk_offsets(base, n2, d - 1, off);
    chunk_offsets(base + n2, n - n2, d - 1, off);
}

// Adds the partial sums of subtrees [lo, hi) the way the levels above them add:
// the left half's sum plus the right half's.
template <typename T>
T fold(const T* part, int64_t lo, int64_t hi) {
    if (hi - lo == 1) {
        return part[lo];
    }
    const int64_t mid = lo + (hi - lo) / 2;
    const T left = fold(part, lo, mid);
    return left + fold(part, mid, hi);
}

// Runs body(lo, hi) on each team member's share of [0, units), with the
// caller's MXCSR, so FTZ/DAZ give the same bits at every team size.
template <typename Body>
void parallel_ranges(int nt, int64_t units, const Body& body) {
    // A region with a false if clause still calls into the OpenMP runtime;
    // small calls skip it.
    if (nt <= 1) {
        body(0, units);
        return;
    }
    const unsigned csr = mxcsr();
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt) if(nt > 1)
#endif
    {
        const mxcsr_scope fp(csr);
        int64_t t = 0;
        int64_t team = 1;
#if defined(_OPENMP)
        t = omp_get_thread_num();
        team = omp_get_num_threads();
#endif
        const int64_t lo = units * t / team;
        const int64_t hi = units * (t + 1) / team;
        if (lo < hi) {
            body(lo, hi);
        }
    }
}

// out[r] = pairwise sum of row r, for m rows of n elements, divided by (T)n
// in T when mean is set. With fewer rows than ROW_UNITS each row is also cut
// into subtree chunks so a few long rows still feed the whole team.
template <typename T>
void row_sums(const T* a, int64_t m, int64_t n, T* out, bool avx2, bool mean) {
    if (m <= 0) {
        return;
    }
    const T count = static_cast<T>(n);
    int d = 0;
    if (m < ROW_UNITS) {
        d = chunk_depth(n);
        while ((m << d) > ROW_UNITS) {
            --d;
        }
    }
    const double bytes = static_cast<double>(m) * static_cast<double>(n) * sizeof(T);
    if (d == 0) {
        const int nt = team_size(bytes, SERIAL_BELOW, PER_THREAD, m);
        parallel_ranges(nt, m, [&](int64_t lo, int64_t hi) {
            rows_kernel<T>(avx2, a + lo * n, n, hi - lo, out + lo);
            for (int64_t r = lo; mean && r < hi; ++r) {
                out[r] = out[r] / count;
            }
        });
        return;
    }
    const int64_t c = int64_t{1} << d;
    int64_t off[(int64_t{1} << CHUNK_DEPTH_MAX) + 1];
    int64_t* end = off;
    chunk_offsets(0, n, d, end);
    *end = n;
    T part[ROW_UNITS];
    const int nt = team_size(bytes, SERIAL_BELOW, PER_THREAD, m * c);
    parallel_ranges(nt, m * c, [&](int64_t lo, int64_t hi) {
        for (int64_t u = lo; u < hi; ++u) {
            const int64_t k = u % c;
            rows_kernel<T>(avx2, a + (u / c) * n + off[k], off[k + 1] - off[k], 1, part + u);
        }
    });
    for (int64_t r = 0; r < m; ++r) {
        const T s = fold(part + r * c, 0, c);
        out[r] = mean ? s / count : s;
    }
}

// Levels of a block tree over s blocks: ceil(log2(s)).
int tree_levels(int64_t s) {
    int l = 0;
    while ((int64_t{1} << l) < s) {
        ++l;
    }
    return l;
}

// Columns [j0, j0 + w) of blocks [b0, b1) into dst, through the block tree.
// The left subtree writes dst and the right one the level buffer at lvl, each
// using the deeper buffers, so a tree of L levels needs L panels.
template <typename T>
void block_tree(const T* a, int64_t m, int64_t n, int64_t b0, int64_t b1, int64_t j0, int64_t w, T* dst,
                T* lvl, bool avx2) {
    if (b1 - b0 == 1) {
        const int64_t r0 = b0 * ROW_BLOCK;
        const int64_t rows = m - r0 < ROW_BLOCK ? m - r0 : ROW_BLOCK;
        block_kernel<T>(avx2, a + r0 * n + j0, n, rows, w, dst);
        return;
    }
    const int64_t mid = b0 + (b1 - b0) / 2;
    block_tree(a, m, n, b0, mid, j0, w, dst, lvl, avx2);
    block_tree(a, m, n, mid, b1, j0, w, lvl, lvl + w, avx2);
    for (int64_t j = 0; j < w; ++j) {
        dst[j] = dst[j] + lvl[j];
    }
}

// The first block of chunk k at depth d of the block tree over [b0, b1).
int64_t chunk_block(int64_t b0, int64_t b1, int d, int64_t k) {
    for (int l = d - 1; l >= 0; --l) {
        const int64_t mid = b0 + (b1 - b0) / 2;
        if ((k >> l) & 1) {
            b0 = mid;
        } else {
            b1 = mid;
        }
    }
    return b0;
}

// out[j] = sum over rows of column j, divided by (T)m in T when mean is set.
// Pieces are (row chunk, column panel) pairs: panels for wide arrays, chunks of
// the block tree as well when there are too few panels, as in a tall narrow
// array.
template <typename T>
void col_sums(const T* a, int64_t m, int64_t n, T* out, bool avx2, bool mean) {
    if (n <= 0) {
        return;
    }
    const T count = static_cast<T>(m);
    if (m <= 0) {
        for (int64_t j = 0; j < n; ++j) {
            out[j] = mean ? T(0) / count : T(0);
        }
        return;
    }
    const int64_t nb = (m + ROW_BLOCK - 1) / ROW_BLOCK;
    const double bytes = static_cast<double>(m) * static_cast<double>(n) * sizeof(T);
    // Panels first (they cost nothing extra), then chunks of the block tree when
    // the columns alone give too few pieces. A chunk's tree has fewer levels,
    // which allows wider panels in the same level buffer.
    int d = 0;
    int64_t w = 0;
    int64_t panels = 0;
    for (;;) {
        const int levels = tree_levels((nb + (int64_t{1} << d) - 1) >> d);
        const int64_t cap = levels == 0 || LEVEL_ELEMS / levels > PANEL_MAX
                                ? PANEL_MAX
                                : (LEVEL_ELEMS / levels) & ~int64_t{15};
        w = (n / COL_UNITS) & ~int64_t{15};
        w = w < PANEL_MIN ? PANEL_MIN : w;
        w = w > cap ? cap : w;
        panels = (n + w - 1) / w;
        if ((panels << d) >= COL_UNITS || d == CHUNK_DEPTH_MAX || (int64_t{2} << d) > nb || bytes < SERIAL_BELOW) {
            break;
        }
        ++d;
    }
    const int64_t c = int64_t{1} << d;
    const int64_t units = panels * c;
    // Chunk 0 accumulates into out, the others into rows of the scratch.
    std::unique_ptr<T[]> scratch(c > 1 ? new T[static_cast<size_t>((c - 1) * n)] : nullptr);
    T* const extra = scratch.get();
    const auto dst = [&](int64_t k) { return k == 0 ? out : extra + (k - 1) * n; };
    const auto run = [&](int64_t lo, int64_t hi) {
        alignas(64) T lvl[LEVEL_ELEMS];
        for (int64_t u = lo; u < hi; ++u) {
            const int64_t k = u / panels;
            const int64_t j0 = (u % panels) * w;
            const int64_t b0 = chunk_block(0, nb, d, k);
            const int64_t b1 = k + 1 < c ? chunk_block(0, nb, d, k + 1) : nb;
            block_tree(a, m, n, b0, b1, j0, n - j0 < w ? n - j0 : w, dst(k) + j0, lvl, avx2);
        }
    };
    // The chunks' partial rows add in the block tree's order, column by column.
    const auto finish = [&](int64_t j0, int64_t j1) {
        for (int64_t span = 1; span < c; span <<= 1) {
            for (int64_t k = 0; k < c; k += span << 1) {
                T* const left = dst(k);
                const T* const right = dst(k + span);
                for (int64_t j = j0; j < j1; ++j) {
                    left[j] = left[j] + right[j];
                }
            }
        }
        for (int64_t j = j0; mean && j < j1; ++j) {
            out[j] = out[j] / count;
        }
    };
    const int nt = team_size(bytes, SERIAL_BELOW, PER_THREAD, units);
    if (nt <= 1) {
        run(0, units);
        finish(0, n);
        return;
    }
    const unsigned csr = mxcsr();
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt) if(nt > 1)
#endif
    {
        const mxcsr_scope fp(csr);
        int64_t t = 0;
        int64_t team = 1;
#if defined(_OPENMP)
        t = omp_get_thread_num();
        team = omp_get_num_threads();
#endif
        const int64_t lo = units * t / team;
        const int64_t hi = units * (t + 1) / team;
        run(lo, hi);
        if (c > 1) {
#if defined(_OPENMP)
#pragma omp barrier
#endif
            finish(n * t / team, n * (t + 1) / team);
        } else {
            // With one chunk the pieces are whole panels, so this member's
            // columns are final already.
            finish(lo * w, hi * w < n ? hi * w : n);
        }
    }
}

} // namespace

template <typename T>
void sum_rows_naive(const T* x, int64_t n, int64_t rows, T* out) {
    for (int64_t r = 0; r < rows; ++r) {
        out[r] = pairwise_sum<T>(x + r * n, n);
    }
}

template <typename T>
void sum_block_naive(const T* x, int64_t ld, int64_t rows, int64_t w, T* acc) {
    for (int64_t j = 0; j < w; ++j) {
        acc[j] = T(0);
    }
    for (int64_t i = 0; i < rows; ++i) {
        const T* xi = x + i * ld;
        for (int64_t j = 0; j < w; ++j) {
            acc[j] += xi[j];
        }
    }
}

template <typename T>
T sum_all(const T* a, int64_t m, int64_t n, bool avx2) {
    // The wrapper guarantees a C-contiguous buffer: one run of m * n elements.
    T s = T(0);
    row_sums<T>(a, 1, m * n, &s, avx2, false);
    return s;
}

template <typename T>
void sum_axis(const T* a, T* out, int64_t m, int64_t n, int axis, bool avx2, bool mean) {
    if (axis == 1) {
        row_sums<T>(a, m, n, out, avx2, mean);
    } else {
        col_sums<T>(a, m, n, out, avx2, mean);
    }
}

template float sum_all<float>(const float*, int64_t, int64_t, bool);
template double sum_all<double>(const double*, int64_t, int64_t, bool);
template void sum_axis<float>(const float*, float*, int64_t, int64_t, int, bool, bool);
template void sum_axis<double>(const double*, double*, int64_t, int64_t, int, bool, bool);
template void sum_rows_naive<float>(const float*, int64_t, int64_t, float*);
template void sum_rows_naive<double>(const double*, int64_t, int64_t, double*);
template void sum_block_naive<float>(const float*, int64_t, int64_t, int64_t, float*);
template void sum_block_naive<double>(const double*, int64_t, int64_t, int64_t, double*);

} // namespace pg::cpu
