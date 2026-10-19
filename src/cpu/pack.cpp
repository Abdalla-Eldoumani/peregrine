#include "cpu/pack.hpp"

#include <immintrin.h>

#include <cstring>

namespace pg::cpu {

namespace {

// Pure data movement: loads, shuffles, and stores never round, flush, or
// quiet anything, so NaN payloads, signed zeros, and subnormals reach the
// panels bit for bit whichever path packed them.

// Rows r0..r3 of four k steps in, the four k steps of those rows out.
inline void transpose4(__m256d& v0, __m256d& v1, __m256d& v2, __m256d& v3) {
    const __m256d t0 = _mm256_unpacklo_pd(v0, v1);
    const __m256d t1 = _mm256_unpackhi_pd(v0, v1);
    const __m256d t2 = _mm256_unpacklo_pd(v2, v3);
    const __m256d t3 = _mm256_unpackhi_pd(v2, v3);
    v0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    v1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    v2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    v3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

inline void transpose8(__m256 v[8]) {
    const __m256 t0 = _mm256_unpacklo_ps(v[0], v[1]);
    const __m256 t1 = _mm256_unpackhi_ps(v[0], v[1]);
    const __m256 t2 = _mm256_unpacklo_ps(v[2], v[3]);
    const __m256 t3 = _mm256_unpackhi_ps(v[2], v[3]);
    const __m256 t4 = _mm256_unpacklo_ps(v[4], v[5]);
    const __m256 t5 = _mm256_unpackhi_ps(v[4], v[5]);
    const __m256 t6 = _mm256_unpacklo_ps(v[6], v[7]);
    const __m256 t7 = _mm256_unpackhi_ps(v[6], v[7]);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    v[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    v[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    v[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    v[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    v[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    v[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    v[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    v[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// Any stride, any live count: the edge panels and the layouts without a fast
// path. The zero fill is its own loop, so no element pays a branch.
template <int W, typename T>
void gather(T* dst, const T* src, int64_t step, int64_t lane, int64_t live, int64_t kc) {
    for (int64_t p = 0; p < kc; ++p) {
        const T* s = src + p * step;
        for (int64_t r = 0; r < live; ++r) {
            dst[r] = s[r * lane];
        }
        for (int64_t r = live; r < W; ++r) {
            dst[r] = T{0};
        }
        dst += W;
    }
}

// A full MR = 6 panel of a row-major block (cs == 1): each row is contiguous
// in k, so blocks of k steps are transposed in registers.
void a_rows_panel(double* dst, const double* a, int64_t rs, int64_t kc) {
    const double* r0 = a;
    const double* r1 = a + rs;
    const double* r2 = a + 2 * rs;
    const double* r3 = a + 3 * rs;
    const double* r4 = a + 4 * rs;
    const double* r5 = a + 5 * rs;
    int64_t p = 0;
    for (; p + 4 <= kc; p += 4) {
        __m256d v0 = _mm256_loadu_pd(r0 + p);
        __m256d v1 = _mm256_loadu_pd(r1 + p);
        __m256d v2 = _mm256_loadu_pd(r2 + p);
        __m256d v3 = _mm256_loadu_pd(r3 + p);
        transpose4(v0, v1, v2, v3);
        const __m256d v4 = _mm256_loadu_pd(r4 + p);
        const __m256d v5 = _mm256_loadu_pd(r5 + p);
        const __m256d lo = _mm256_unpacklo_pd(v4, v5);
        const __m256d hi = _mm256_unpackhi_pd(v4, v5);
        _mm256_storeu_pd(dst + 0, v0);
        _mm_storeu_pd(dst + 4, _mm256_castpd256_pd128(lo));
        _mm256_storeu_pd(dst + 6, v1);
        _mm_storeu_pd(dst + 10, _mm256_castpd256_pd128(hi));
        _mm256_storeu_pd(dst + 12, v2);
        _mm_storeu_pd(dst + 16, _mm256_extractf128_pd(lo, 1));
        _mm256_storeu_pd(dst + 18, v3);
        _mm_storeu_pd(dst + 22, _mm256_extractf128_pd(hi, 1));
        dst += 24;
    }
    gather<6>(dst, a + p, 1, rs, 6, kc - p);
}

void a_rows_panel(float* dst, const float* a, int64_t rs, int64_t kc) {
    int64_t p = 0;
    for (; p + 8 <= kc; p += 8) {
        __m256 v[8];
        for (int r = 0; r < 6; ++r) {
            v[r] = _mm256_loadu_ps(a + r * rs + p);
        }
        v[6] = _mm256_setzero_ps();
        v[7] = _mm256_setzero_ps();
        transpose8(v);
        for (int q = 0; q < 8; ++q) {
            _mm_storeu_ps(dst, _mm256_castps256_ps128(v[q]));
            _mm_storel_pi(reinterpret_cast<__m64*>(dst + 4), _mm256_extractf128_ps(v[q], 1));
            dst += 6;
        }
    }
    gather<6>(dst, a + p, 1, rs, 6, kc - p);
}

// A full panel of a column-major block (rs == 1): each k step's MR rows are
// contiguous already.
template <int MR, typename T>
void a_cols_panel(T* dst, const T* a, int64_t cs, int64_t kc) {
    for (int64_t p = 0; p < kc; ++p) {
        std::memcpy(dst, a + p * cs, MR * sizeof(T));
        dst += MR;
    }
}

// A full NR panel of a row-major block (cs == 1): each k step is one 64-byte
// row, copied as raw bytes into the 64-byte-aligned panel.
template <int NR, typename T>
void b_rows_panel(T* dst, const T* b, int64_t rs, int64_t kc) {
    static_assert(NR * sizeof(T) == 64, "a B panel row is one cache line");
    for (int64_t p = 0; p < kc; ++p) {
        const T* s = b + p * rs;
        const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
        const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s) + 1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst), lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst) + 1, hi);
        dst += NR;
    }
}

// A full NR panel of a column-major block (rs == 1): each column is
// contiguous in k, so blocks of k steps are transposed in registers.
void b_cols_panel(double* dst, const double* b, int64_t cs, int64_t kc) {
    int64_t p = 0;
    for (; p + 4 <= kc; p += 4) {
        __m256d v0 = _mm256_loadu_pd(b + p);
        __m256d v1 = _mm256_loadu_pd(b + cs + p);
        __m256d v2 = _mm256_loadu_pd(b + 2 * cs + p);
        __m256d v3 = _mm256_loadu_pd(b + 3 * cs + p);
        transpose4(v0, v1, v2, v3);
        __m256d v4 = _mm256_loadu_pd(b + 4 * cs + p);
        __m256d v5 = _mm256_loadu_pd(b + 5 * cs + p);
        __m256d v6 = _mm256_loadu_pd(b + 6 * cs + p);
        __m256d v7 = _mm256_loadu_pd(b + 7 * cs + p);
        transpose4(v4, v5, v6, v7);
        _mm256_store_pd(dst + 0, v0);
        _mm256_store_pd(dst + 4, v4);
        _mm256_store_pd(dst + 8, v1);
        _mm256_store_pd(dst + 12, v5);
        _mm256_store_pd(dst + 16, v2);
        _mm256_store_pd(dst + 20, v6);
        _mm256_store_pd(dst + 24, v3);
        _mm256_store_pd(dst + 28, v7);
        dst += 32;
    }
    gather<8>(dst, b + p, 1, cs, 8, kc - p);
}

void b_cols_panel(float* dst, const float* b, int64_t cs, int64_t kc) {
    int64_t p = 0;
    for (; p + 8 <= kc; p += 8) {
        for (int half = 0; half < 2; ++half) {
            __m256 v[8];
            for (int c = 0; c < 8; ++c) {
                v[c] = _mm256_loadu_ps(b + (8 * half + c) * cs + p);
            }
            transpose8(v);
            for (int q = 0; q < 8; ++q) {
                _mm256_store_ps(dst + 16 * q + 8 * half, v[q]);
            }
        }
        dst += 128;
    }
    gather<16>(dst, b + p, 1, cs, 16, kc - p);
}

} // namespace

// cs == 1 is tested first, so an operand that is both (a length-1 dim, whose
// stride dispatch sets to 1) takes the row reading.
template <int MR, typename T>
void pack_a(T* ap, const T* a, int64_t rs, int64_t cs, int64_t mc, int64_t kc) {
    static_assert(MR == 6, "the in-register transposes are written for six rows");
    const int64_t full = mc / MR;
    for (int64_t q = 0; q < full; ++q) {
        T* dst = ap + q * MR * kc;
        const T* src = a + q * MR * rs;
        if (cs == 1) {
            a_rows_panel(dst, src, rs, kc);
        } else if (rs == 1) {
            a_cols_panel<MR>(dst, src, cs, kc);
        } else {
            gather<MR>(dst, src, cs, rs, MR, kc);
        }
    }
    if (full * MR < mc) {
        gather<MR>(ap + full * MR * kc, a + full * MR * rs, cs, rs, mc - full * MR, kc);
    }
}

template <int NR, typename T>
void pack_b_panels(T* bp, const T* b, int64_t rs, int64_t cs, int64_t kc, int64_t nc, int64_t p0, int64_t p1) {
    const int64_t full = nc / NR;
    for (int64_t q = p0; q < p1; ++q) {
        T* dst = bp + q * kc * NR;
        const T* src = b + q * NR * cs;
        if (q == full) {
            gather<NR>(dst, src, rs, cs, nc - full * NR, kc);
        } else if (cs == 1) {
            b_rows_panel<NR>(dst, src, rs, kc);
        } else if (rs == 1) {
            b_cols_panel(dst, src, cs, kc);
        } else {
            gather<NR>(dst, src, rs, cs, NR, kc);
        }
    }
}

template void pack_a<6, double>(double*, const double*, int64_t, int64_t, int64_t, int64_t);
template void pack_a<6, float>(float*, const float*, int64_t, int64_t, int64_t, int64_t);
template void pack_b_panels<8, double>(double*, const double*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
template void pack_b_panels<16, float>(float*, const float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

} // namespace pg::cpu
