#include "cpu/reduce.hpp"

#include <immintrin.h>

// The AVX2 leaf kernels of the reductions. Every vector add is the scalar
// twin's add on the same operands in the same order, so the bits match
// reduce.cpp's kernels exactly. Library inline functions stay out of this TU:
// one compiled here with AVX2 could be the copy the linker keeps for the
// fallback TUs too.

namespace pg::cpu {

namespace {

constexpr int64_t LEAF = 128;
// A node this long or shorter has at most 16 leaves (every leaf below a split
// holds at least 64 elements); its leaves are summed side by side, which hides
// the add latency that one leaf's single accumulator chain cannot.
constexpr int64_t SPAN = 1024;
constexpr int MAX_LEAVES = 16;

int64_t half(int64_t n) {
    return (n / 2) & ~int64_t{7};
}

int64_t min64(int64_t a, int64_t b) {
    return a < b ? a : b;
}

// ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)) of one accumulator.
float combine(__m256 a) {
    const __m128 p = _mm_hadd_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    const __m128 q = _mm_hadd_ps(p, p);
    return _mm_cvtss_f32(_mm_add_ss(q, _mm_movehdup_ps(q)));
}

double combine(__m256d lo, __m256d hi) {
    // (a0 + a1, a4 + a5, a2 + a3, a6 + a7)
    const __m256d p = _mm256_hadd_pd(lo, hi);
    const __m128d q = _mm_add_pd(_mm256_castpd256_pd128(p), _mm256_extractf128_pd(p, 1));
    return _mm_cvtsd_f64(_mm_add_sd(q, _mm_unpackhi_pd(q, q)));
}

float leaf(const float* x, int64_t n) {
    const int64_t blocked = n & ~int64_t{7};
    __m256 acc = _mm256_setzero_ps();
    for (int64_t i = 0; i < blocked; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(x + i));
    }
    float s = combine(acc);
    for (int64_t i = blocked; i < n; ++i) {
        s += x[i];
    }
    return s;
}

double leaf(const double* x, int64_t n) {
    const int64_t blocked = n & ~int64_t{7};
    __m256d lo = _mm256_setzero_pd();
    __m256d hi = _mm256_setzero_pd();
    for (int64_t i = 0; i < blocked; i += 8) {
        lo = _mm256_add_pd(lo, _mm256_loadu_pd(x + i));
        hi = _mm256_add_pd(hi, _mm256_loadu_pd(x + i + 4));
    }
    double s = combine(lo, hi);
    for (int64_t i = blocked; i < n; ++i) {
        s += x[i];
    }
    return s;
}

// The scalar tail of each leaf in a group, after its lanes are combined.
template <typename T>
void tails(const T* x, const int64_t* off, const int64_t* len, int g, T* out) {
    for (int j = 0; j < g; ++j) {
        T s = out[j];
        const T* p = x + off[j];
        for (int64_t i = len[j] & ~int64_t{7}; i < len[j]; ++i) {
            s += p[i];
        }
        out[j] = s;
    }
}

// Eight float leaves at x + off[j] of len[j] elements, one accumulator each.
// The shortest leaf's blocked length runs in registers; any leaf longer than
// that finishes its lanes in memory, then the 8 x 8 lanes combine with hadds
// into one vector of eight leaf sums, each in the scalar order.
void group(const float* x, const int64_t* off, const int64_t* len, float* out) {
    int64_t common = len[0] & ~int64_t{7};
    for (int j = 1; j < 8; ++j) {
        common = min64(common, len[j] & ~int64_t{7});
    }
    const float* p0 = x + off[0];
    const float* p1 = x + off[1];
    const float* p2 = x + off[2];
    const float* p3 = x + off[3];
    const float* p4 = x + off[4];
    const float* p5 = x + off[5];
    const float* p6 = x + off[6];
    const float* p7 = x + off[7];
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps();
    __m256 a5 = _mm256_setzero_ps();
    __m256 a6 = _mm256_setzero_ps();
    __m256 a7 = _mm256_setzero_ps();
    for (int64_t i = 0; i < common; i += 8) {
        a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p0 + i));
        a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p1 + i));
        a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p2 + i));
        a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p3 + i));
        a4 = _mm256_add_ps(a4, _mm256_loadu_ps(p4 + i));
        a5 = _mm256_add_ps(a5, _mm256_loadu_ps(p5 + i));
        a6 = _mm256_add_ps(a6, _mm256_loadu_ps(p6 + i));
        a7 = _mm256_add_ps(a7, _mm256_loadu_ps(p7 + i));
    }
    alignas(32) float lanes[64];
    _mm256_store_ps(lanes + 0, a0);
    _mm256_store_ps(lanes + 8, a1);
    _mm256_store_ps(lanes + 16, a2);
    _mm256_store_ps(lanes + 24, a3);
    _mm256_store_ps(lanes + 32, a4);
    _mm256_store_ps(lanes + 40, a5);
    _mm256_store_ps(lanes + 48, a6);
    _mm256_store_ps(lanes + 56, a7);
    for (int j = 0; j < 8; ++j) {
        const int64_t blocked = len[j] & ~int64_t{7};
        if (blocked > common) {
            const float* p = x + off[j];
            __m256 v = _mm256_load_ps(lanes + 8 * j);
            for (int64_t i = common; i < blocked; i += 8) {
                v = _mm256_add_ps(v, _mm256_loadu_ps(p + i));
            }
            _mm256_store_ps(lanes + 8 * j, v);
        }
    }
    // hadd(A, B) gives (A0+A1, A2+A3, B0+B1, B2+B3 | A4+A5, A6+A7, B4+B5, B6+B7);
    // two rounds leave (q_j low half sums | q_j high half sums) for four leaves.
    const __m256 h01 = _mm256_hadd_ps(_mm256_load_ps(lanes + 0), _mm256_load_ps(lanes + 8));
    const __m256 h23 = _mm256_hadd_ps(_mm256_load_ps(lanes + 16), _mm256_load_ps(lanes + 24));
    const __m256 h45 = _mm256_hadd_ps(_mm256_load_ps(lanes + 32), _mm256_load_ps(lanes + 40));
    const __m256 h67 = _mm256_hadd_ps(_mm256_load_ps(lanes + 48), _mm256_load_ps(lanes + 56));
    const __m256 q0 = _mm256_hadd_ps(h01, h23);
    const __m256 q1 = _mm256_hadd_ps(h45, h67);
    const __m256 lo = _mm256_permute2f128_ps(q0, q1, 0x20);
    const __m256 hi = _mm256_permute2f128_ps(q0, q1, 0x31);
    _mm256_storeu_ps(out, _mm256_add_ps(lo, hi));
    tails(x, off, len, 8, out);
}

// Four double leaves, two accumulators each (lanes 0-3 and 4-7).
void group(const double* x, const int64_t* off, const int64_t* len, double* out) {
    int64_t common = len[0] & ~int64_t{7};
    for (int j = 1; j < 4; ++j) {
        common = min64(common, len[j] & ~int64_t{7});
    }
    const double* p0 = x + off[0];
    const double* p1 = x + off[1];
    const double* p2 = x + off[2];
    const double* p3 = x + off[3];
    __m256d l0 = _mm256_setzero_pd();
    __m256d h0 = _mm256_setzero_pd();
    __m256d l1 = _mm256_setzero_pd();
    __m256d h1 = _mm256_setzero_pd();
    __m256d l2 = _mm256_setzero_pd();
    __m256d h2 = _mm256_setzero_pd();
    __m256d l3 = _mm256_setzero_pd();
    __m256d h3 = _mm256_setzero_pd();
    for (int64_t i = 0; i < common; i += 8) {
        l0 = _mm256_add_pd(l0, _mm256_loadu_pd(p0 + i));
        h0 = _mm256_add_pd(h0, _mm256_loadu_pd(p0 + i + 4));
        l1 = _mm256_add_pd(l1, _mm256_loadu_pd(p1 + i));
        h1 = _mm256_add_pd(h1, _mm256_loadu_pd(p1 + i + 4));
        l2 = _mm256_add_pd(l2, _mm256_loadu_pd(p2 + i));
        h2 = _mm256_add_pd(h2, _mm256_loadu_pd(p2 + i + 4));
        l3 = _mm256_add_pd(l3, _mm256_loadu_pd(p3 + i));
        h3 = _mm256_add_pd(h3, _mm256_loadu_pd(p3 + i + 4));
    }
    alignas(32) double lanes[32];
    _mm256_store_pd(lanes + 0, l0);
    _mm256_store_pd(lanes + 4, h0);
    _mm256_store_pd(lanes + 8, l1);
    _mm256_store_pd(lanes + 12, h1);
    _mm256_store_pd(lanes + 16, l2);
    _mm256_store_pd(lanes + 20, h2);
    _mm256_store_pd(lanes + 24, l3);
    _mm256_store_pd(lanes + 28, h3);
    for (int j = 0; j < 4; ++j) {
        const int64_t blocked = len[j] & ~int64_t{7};
        if (blocked > common) {
            const double* p = x + off[j];
            __m256d vl = _mm256_load_pd(lanes + 8 * j);
            __m256d vh = _mm256_load_pd(lanes + 8 * j + 4);
            for (int64_t i = common; i < blocked; i += 8) {
                vl = _mm256_add_pd(vl, _mm256_loadu_pd(p + i));
                vh = _mm256_add_pd(vh, _mm256_loadu_pd(p + i + 4));
            }
            _mm256_store_pd(lanes + 8 * j, vl);
            _mm256_store_pd(lanes + 8 * j + 4, vh);
        }
    }
    // Per leaf, hadd(lo, hi) = (a0+a1, a4+a5, a2+a3, a6+a7); adding its halves
    // gives (q_low, q_high), and the final hadd adds each leaf's pair.
    const __m256d p0s = _mm256_hadd_pd(_mm256_load_pd(lanes + 0), _mm256_load_pd(lanes + 4));
    const __m256d p1s = _mm256_hadd_pd(_mm256_load_pd(lanes + 8), _mm256_load_pd(lanes + 12));
    const __m256d p2s = _mm256_hadd_pd(_mm256_load_pd(lanes + 16), _mm256_load_pd(lanes + 20));
    const __m256d p3s = _mm256_hadd_pd(_mm256_load_pd(lanes + 24), _mm256_load_pd(lanes + 28));
    const __m256d w01 = _mm256_add_pd(_mm256_permute2f128_pd(p0s, p1s, 0x20), _mm256_permute2f128_pd(p0s, p1s, 0x31));
    const __m256d w23 = _mm256_add_pd(_mm256_permute2f128_pd(p2s, p3s, 0x20), _mm256_permute2f128_pd(p2s, p3s, 0x31));
    // (s0, s2, s1, s3)
    const __m256d s = _mm256_hadd_pd(w01, w23);
    _mm256_storeu_pd(out, _mm256_permute4x64_pd(s, 0xD8));
    tails(x, off, len, 4, out);
}

constexpr int group_size(const float*) {
    return 8;
}

constexpr int group_size(const double*) {
    return 4;
}

// Leaves under a node of n elements at x + base, left to right.
void collect(int64_t base, int64_t n, int64_t* off, int64_t* len, int& count) {
    if (n <= LEAF) {
        off[count] = base;
        len[count] = n;
        ++count;
        return;
    }
    const int64_t h = half(n);
    collect(base, h, off, len, count);
    collect(base + h, n - h, off, len, count);
}

// The node's sum from its leaf sums, adding as the recursion does.
template <typename T>
T evaluate(const T* sums, int64_t n, int& k) {
    if (n <= LEAF) {
        return sums[k++];
    }
    const int64_t h = half(n);
    const T left = evaluate(sums, h, k);
    return left + evaluate(sums, n - h, k);
}

template <typename T>
T node(const T* x, int64_t n) {
    if (n <= LEAF) {
        return leaf(x, n);
    }
    if (n > SPAN) {
        const int64_t h = half(n);
        const T left = node(x, h);
        return left + node(x + h, n - h);
    }
    int64_t off[MAX_LEAVES];
    int64_t len[MAX_LEAVES];
    T sums[MAX_LEAVES];
    int count = 0;
    collect(0, n, off, len, count);
    constexpr int g = group_size(static_cast<const T*>(nullptr));
    int j = 0;
    for (; j + g <= count; j += g) {
        group(x, off + j, len + j, sums + j);
    }
    for (; j < count; ++j) {
        sums[j] = leaf(x + off[j], len[j]);
    }
    int k = 0;
    return evaluate(sums, n, k);
}

// Sequential sums of four rows per pass over the panel, so acc is read and
// written once per four rows; each element still adds row by row.
void block(const float* x, int64_t ld, int64_t rows, int64_t w, float* acc) {
    const int64_t wv = w & ~int64_t{7};
    for (int64_t j = 0; j < wv; j += 8) {
        _mm256_storeu_ps(acc + j, _mm256_setzero_ps());
    }
    for (int64_t j = wv; j < w; ++j) {
        acc[j] = 0.0f;
    }
    int64_t i = 0;
    for (; i + 4 <= rows; i += 4) {
        const float* r0 = x + i * ld;
        const float* r1 = r0 + ld;
        const float* r2 = r1 + ld;
        const float* r3 = r2 + ld;
        for (int64_t j = 0; j < wv; j += 8) {
            __m256 v = _mm256_loadu_ps(acc + j);
            v = _mm256_add_ps(v, _mm256_loadu_ps(r0 + j));
            v = _mm256_add_ps(v, _mm256_loadu_ps(r1 + j));
            v = _mm256_add_ps(v, _mm256_loadu_ps(r2 + j));
            v = _mm256_add_ps(v, _mm256_loadu_ps(r3 + j));
            _mm256_storeu_ps(acc + j, v);
        }
        for (int64_t j = wv; j < w; ++j) {
            acc[j] = (((acc[j] + r0[j]) + r1[j]) + r2[j]) + r3[j];
        }
    }
    for (; i < rows; ++i) {
        const float* r0 = x + i * ld;
        for (int64_t j = 0; j < wv; j += 8) {
            _mm256_storeu_ps(acc + j, _mm256_add_ps(_mm256_loadu_ps(acc + j), _mm256_loadu_ps(r0 + j)));
        }
        for (int64_t j = wv; j < w; ++j) {
            acc[j] += r0[j];
        }
    }
}

void block(const double* x, int64_t ld, int64_t rows, int64_t w, double* acc) {
    const int64_t wv = w & ~int64_t{3};
    for (int64_t j = 0; j < wv; j += 4) {
        _mm256_storeu_pd(acc + j, _mm256_setzero_pd());
    }
    for (int64_t j = wv; j < w; ++j) {
        acc[j] = 0.0;
    }
    int64_t i = 0;
    for (; i + 4 <= rows; i += 4) {
        const double* r0 = x + i * ld;
        const double* r1 = r0 + ld;
        const double* r2 = r1 + ld;
        const double* r3 = r2 + ld;
        for (int64_t j = 0; j < wv; j += 4) {
            __m256d v = _mm256_loadu_pd(acc + j);
            v = _mm256_add_pd(v, _mm256_loadu_pd(r0 + j));
            v = _mm256_add_pd(v, _mm256_loadu_pd(r1 + j));
            v = _mm256_add_pd(v, _mm256_loadu_pd(r2 + j));
            v = _mm256_add_pd(v, _mm256_loadu_pd(r3 + j));
            _mm256_storeu_pd(acc + j, v);
        }
        for (int64_t j = wv; j < w; ++j) {
            acc[j] = (((acc[j] + r0[j]) + r1[j]) + r2[j]) + r3[j];
        }
    }
    for (; i < rows; ++i) {
        const double* r0 = x + i * ld;
        for (int64_t j = 0; j < wv; j += 4) {
            _mm256_storeu_pd(acc + j, _mm256_add_pd(_mm256_loadu_pd(acc + j), _mm256_loadu_pd(r0 + j)));
        }
        for (int64_t j = wv; j < w; ++j) {
            acc[j] += r0[j];
        }
    }
}

// Sequential sums of two rows per pass over the panel, each element still adding
// row by row, with the next pass's two rows prefetched a line at a time. A panel
// row is a short run one array row away from the next, and the hardware
// prefetchers do not follow that stride: at (4096, 4096) the read ran at about
// 16 GB/s while the stalled cores clocked down to 40-50 percent, and at 30-32
// GB/s with the prefetch (6 threads, sustained 100 ms batches). Two rows per pass
// against four, same prefetch: float64 (4096, 4096) 4.46 vs 4.77 ms, float32
// (16, 1000000) 2.60 vs 2.77 ms, the other axis-0 cells within 5 percent.
void stream_block(const float* x, int64_t ld, int64_t rows, int64_t w, float* acc) {
    const int64_t wv = w & ~int64_t{7};
    for (int64_t j = 0; j < wv; j += 8) {
        _mm256_storeu_ps(acc + j, _mm256_setzero_ps());
    }
    for (int64_t j = wv; j < w; ++j) {
        acc[j] = 0.0f;
    }
    int64_t i = 0;
    for (; i + 2 <= rows; i += 2) {
        const float* r0 = x + i * ld;
        const float* r1 = r0 + ld;
        const bool ahead = i + 4 <= rows;
        const float* p0 = ahead ? r1 + ld : r0;
        for (int64_t j = 0; j < wv; j += 8) {
            if (ahead && (j & 15) == 0) {
                _mm_prefetch(reinterpret_cast<const char*>(p0 + j), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(p0 + ld + j), _MM_HINT_T0);
            }
            __m256 v = _mm256_add_ps(_mm256_loadu_ps(acc + j), _mm256_loadu_ps(r0 + j));
            _mm256_storeu_ps(acc + j, _mm256_add_ps(v, _mm256_loadu_ps(r1 + j)));
        }
        for (int64_t j = wv; j < w; ++j) {
            acc[j] = (acc[j] + r0[j]) + r1[j];
        }
    }
    if (i < rows) {
        const float* r0 = x + i * ld;
        for (int64_t j = 0; j < wv; j += 8) {
            _mm256_storeu_ps(acc + j, _mm256_add_ps(_mm256_loadu_ps(acc + j), _mm256_loadu_ps(r0 + j)));
        }
        for (int64_t j = wv; j < w; ++j) {
            acc[j] += r0[j];
        }
    }
}

void stream_block(const double* x, int64_t ld, int64_t rows, int64_t w, double* acc) {
    const int64_t wv = w & ~int64_t{3};
    for (int64_t j = 0; j < wv; j += 4) {
        _mm256_storeu_pd(acc + j, _mm256_setzero_pd());
    }
    for (int64_t j = wv; j < w; ++j) {
        acc[j] = 0.0;
    }
    int64_t i = 0;
    for (; i + 2 <= rows; i += 2) {
        const double* r0 = x + i * ld;
        const double* r1 = r0 + ld;
        const bool ahead = i + 4 <= rows;
        const double* p0 = ahead ? r1 + ld : r0;
        for (int64_t j = 0; j < wv; j += 4) {
            if (ahead && (j & 7) == 0) {
                _mm_prefetch(reinterpret_cast<const char*>(p0 + j), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(p0 + ld + j), _MM_HINT_T0);
            }
            __m256d v = _mm256_add_pd(_mm256_loadu_pd(acc + j), _mm256_loadu_pd(r0 + j));
            _mm256_storeu_pd(acc + j, _mm256_add_pd(v, _mm256_loadu_pd(r1 + j)));
        }
        for (int64_t j = wv; j < w; ++j) {
            acc[j] = (acc[j] + r0[j]) + r1[j];
        }
    }
    if (i < rows) {
        const double* r0 = x + i * ld;
        for (int64_t j = 0; j < wv; j += 4) {
            _mm256_storeu_pd(acc + j, _mm256_add_pd(_mm256_loadu_pd(acc + j), _mm256_loadu_pd(r0 + j)));
        }
        for (int64_t j = wv; j < w; ++j) {
            acc[j] += r0[j];
        }
    }
}

} // namespace

template <typename T>
void sum_rows_avx2(const T* x, int64_t n, int64_t rows, T* out) {
    if (n > LEAF) {
        for (int64_t r = 0; r < rows; ++r) {
            out[r] = node(x + r * n, n);
        }
        return;
    }
    // Rows this short are single leaves: a group sums g of them side by side.
    constexpr int g = group_size(static_cast<const T*>(nullptr));
    int64_t off[8];
    int64_t len[8];
    for (int j = 0; j < g; ++j) {
        off[j] = j * n;
        len[j] = n;
    }
    int64_t r = 0;
    for (; r + g <= rows; r += g) {
        group(x + r * n, off, len, out + r);
    }
    for (; r < rows; ++r) {
        out[r] = leaf(x + r * n, n);
    }
}

template <typename T>
void sum_block_avx2(const T* x, int64_t ld, int64_t rows, int64_t w, T* acc) {
    block(x, ld, rows, w, acc);
}

template <typename T>
void sum_block_stream_avx2(const T* x, int64_t ld, int64_t rows, int64_t w, T* acc) {
    stream_block(x, ld, rows, w, acc);
}

template void sum_rows_avx2<float>(const float*, int64_t, int64_t, float*);
template void sum_rows_avx2<double>(const double*, int64_t, int64_t, double*);
template void sum_block_avx2<float>(const float*, int64_t, int64_t, int64_t, float*);
template void sum_block_avx2<double>(const double*, int64_t, int64_t, int64_t, double*);
template void sum_block_stream_avx2<float>(const float*, int64_t, int64_t, int64_t, float*);
template void sum_block_stream_avx2<double>(const double*, int64_t, int64_t, int64_t, double*);

} // namespace pg::cpu
