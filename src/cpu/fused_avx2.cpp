#include "cpu/fused.hpp"
#include "cpu/threads.hpp"

#include <cstdint>
#include <immintrin.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pg::cpu {

// AVX2+FMA bodies for the three fused ops. This is the ONLY fused TU that may
// include <immintrin.h>: it joins PG_AVX2_SOURCES in CMakeLists, so the arch
// flag (/arch:AVX2 on MSVC, -mavx2 -mfma on GCC) is applied here and nowhere a
// fallback-linked TU can reach (the legacy import-crash isolation rule). It uses
// no inline library function either (std::fma among them): an inline function
// compiled here and in a fallback TU is one COMDAT, and the linker may keep this
// AVX2 copy for both.
//
// Every element, the partial vectors at the ends included, goes through the same
// vector instructions, so an element's bits never depend on where it falls
// relative to the alignment peel or a thread boundary. That keeps the result
// bitwise identical for every thread count on GCC too, whose default FP model
// may contract a scalar a*x + b*y differently from the vector body.

namespace {

template <typename T>
struct simd;

template <>
struct simd<float> {
    using reg = __m256;
    static constexpr int64_t W = 8;
    static reg set1(float s) { return _mm256_set1_ps(s); }
    static reg load(const float* p) { return _mm256_loadu_ps(p); }
    // VMASKMOV touches no memory in masked-off lanes (they load as zero), so a
    // partial vector at either end of the buffer never faults.
    static __m256i mask(int64_t live) {
        return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(live)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    }
    static reg load(const float* p, __m256i m) { return _mm256_maskload_ps(p, m); }
    static void store(float* p, __m256i m, reg v) { _mm256_maskstore_ps(p, m, v); }
    static void storeu(float* p, reg v) { _mm256_storeu_ps(p, v); }
    static void store_aligned(float* p, reg v) { _mm256_store_ps(p, v); }
    static void stream(float* p, reg v) { _mm256_stream_ps(p, v); }
    static reg mul(reg x, reg y) { return _mm256_mul_ps(x, y); }
    static reg add(reg x, reg y) { return _mm256_add_ps(x, y); }
    static reg fmadd(reg x, reg y, reg z) { return _mm256_fmadd_ps(x, y, z); }
    // maxps returns its second operand on an unordered compare, so a bare
    // max(v, 0) turns a NaN into 0 where np.maximum keeps it; blend the NaN
    // lanes (v != v) back in.
    static reg relu(reg v) {
        const reg m = _mm256_max_ps(v, _mm256_setzero_ps());
        return _mm256_blendv_ps(m, v, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
    }
};

template <>
struct simd<double> {
    using reg = __m256d;
    static constexpr int64_t W = 4;
    static reg set1(double s) { return _mm256_set1_pd(s); }
    static reg load(const double* p) { return _mm256_loadu_pd(p); }
    static __m256i mask(int64_t live) {
        return _mm256_cmpgt_epi64(_mm256_set1_epi64x(live), _mm256_setr_epi64x(0, 1, 2, 3));
    }
    static reg load(const double* p, __m256i m) { return _mm256_maskload_pd(p, m); }
    static void store(double* p, __m256i m, reg v) { _mm256_maskstore_pd(p, m, v); }
    static void storeu(double* p, reg v) { _mm256_storeu_pd(p, v); }
    static void store_aligned(double* p, reg v) { _mm256_store_pd(p, v); }
    static void stream(double* p, reg v) { _mm256_stream_pd(p, v); }
    static reg mul(reg x, reg y) { return _mm256_mul_pd(x, y); }
    static reg add(reg x, reg y) { return _mm256_add_pd(x, y); }
    static reg fmadd(reg x, reg y, reg z) { return _mm256_fmadd_pd(x, y, z); }
    static reg relu(reg v) {
        const reg m = _mm256_max_pd(v, _mm256_setzero_pd());
        return _mm256_blendv_pd(m, v, _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
    }
};

// Each op computes the vector at element i from operands read through ld, a
// full or a masked load. Named products in a fixed order keep axpby's two
// multiplies and one add; MSVC /fp:precise never contracts them, so it matches
// the unfused NumPy expression and the scalar twin bit for bit.
template <typename T>
struct axpby_op {
    using S = simd<T>;
    const T* x;
    const T* y;
    typename S::reg a;
    typename S::reg b;
    template <typename Load>
    typename S::reg operator()(int64_t i, const Load& ld) const {
        const typename S::reg ax = S::mul(a, ld(x + i));
        const typename S::reg by = S::mul(b, ld(y + i));
        return S::add(ax, by);
    }
};

template <typename T>
struct fma3_op {
    using S = simd<T>;
    const T* x;
    const T* y;
    const T* z;
    template <typename Load>
    typename S::reg operator()(int64_t i, const Load& ld) const {
        return S::fmadd(ld(x + i), ld(y + i), ld(z + i));
    }
};

template <typename T>
struct scaled_relu_op {
    using S = simd<T>;
    const T* x;
    typename S::reg scale;
    template <typename Load>
    typename S::reg operator()(int64_t i, const Load& ld) const {
        return S::relu(S::mul(scale, ld(x + i)));
    }
};

enum class store_kind { unaligned, aligned, stream };

// The live lanes [i, i + live) through masked loads and a masked store.
template <typename T, typename Op>
void part(const Op& op, T* out, int64_t i, int64_t live) {
    using S = simd<T>;
    const __m256i m = S::mask(live);
    S::store(out + i, m, op(i, [m](const T* p) { return S::load(p, m); }));
}

// One thread's elements [lo, hi). In place (out is exactly one of the inputs) is
// safe with every store kind: each element is loaded in the step that stores it,
// before the store, and no thread or lane ever reads an element another one
// stores, so a weakly ordered streaming store has no reader to race.
template <store_kind K, typename T, typename Op>
void range(const Op& op, T* out, int64_t lo, int64_t hi) {
    using S = simd<T>;
    int64_t i = lo;
    if constexpr (K != store_kind::unaligned) {
        const auto addr = reinterpret_cast<std::uintptr_t>(out + i);
        const auto head = static_cast<int64_t>((32 - addr % 32) % 32 / sizeof(T));
        const int64_t live = head < hi - i ? head : hi - i;
        if (live > 0) {
            part(op, out, i, live);
            i += live;
        }
    }
    const auto full = [](const T* p) { return S::load(p); };
    for (; i + S::W <= hi; i += S::W) {
        const typename S::reg r = op(i, full);
        if constexpr (K == store_kind::stream) {
            S::stream(out + i, r);
        } else if constexpr (K == store_kind::aligned) {
            S::store_aligned(out + i, r);
        } else {
            S::storeu(out + i, r);
        }
    }
    if (i < hi) {
        part(op, out, i, hi - i);
    }
}

// make() builds the op inside the region, so each thread broadcasts the scalars
// into its own registers instead of reloading them from the shared frame.
// arrays counts the buffers the op touches, out included.
template <typename T, typename Make>
void elementwise(T* out, int64_t n, int arrays, const Make& make) {
    constexpr int64_t LINE = 64 / static_cast<int64_t>(sizeof(T));
    const auto addr = reinterpret_cast<std::uintptr_t>(out);
    // An out that is not itemsize-aligned never reaches a vector boundary; it
    // keeps unaligned stores and element-count chunks.
    const bool aligned = addr % sizeof(T) == 0;
    int64_t base = aligned ? static_cast<int64_t>((64 - addr % 64) % 64 / sizeof(T)) : 0;
    base = base < n ? base : n;
    // Threads split at 64-byte lines of out counted from base, so no two
    // threads store into one line.
    const int64_t lines = (n - base) / LINE;
    const double bytes = static_cast<double>(n) * static_cast<double>(sizeof(T));
    const int nt = team_size(bytes, FUSED_SERIAL_BELOW_BYTES, FUSED_PER_THREAD_BYTES, lines);
    // An in-place call counts out twice, so it may stream a little early.
    const bool stream = bytes * static_cast<double>(arrays) > FUSED_STREAM_BYTES;
    // Below the serial size the buffers sit in L1 or L2 and the peel's masked
    // head and tail cost more than the line-split vectors they save. fma3 out=
    // float32, one thread: 8x8 72 -> 65 ns, 32x32 254 -> 193-212, 64x64 963 ->
    // 797-840 without the peel.
    const bool peel = aligned && bytes >= FUSED_SERIAL_BELOW_BYTES;
    const store_kind kind = !peel ? store_kind::unaligned : stream ? store_kind::stream : store_kind::aligned;
    const auto body = [&](int64_t lo, int64_t hi) {
        const auto op = make();
        if (kind == store_kind::stream) {
            range<store_kind::stream>(op, out, lo, hi);
            // Streaming stores are weakly ordered; fence them before the
            // region's closing barrier hands the result to the caller.
            _mm_sfence();
        } else if (kind == store_kind::aligned) {
            range<store_kind::aligned>(op, out, lo, hi);
        } else {
            range<store_kind::unaligned>(op, out, lo, hi);
        }
    };
    // One thread runs the body directly: a region with a false if clause still
    // enters the OpenMP runtime (fma3 out= float32 8x8 took 0.30-0.32 us through
    // it, 0.19-0.20 without), and the caller's own thread holds its MXCSR.
    if (nt <= 1) {
        body(0, n);
        return;
    }
    const unsigned csr = mxcsr();
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
        const mxcsr_scope fp(csr);
#if defined(_OPENMP)
        const int64_t t = omp_get_thread_num();
        const int64_t team = omp_get_num_threads();
#else
        const int64_t t = 0;
        const int64_t team = 1;
#endif
        const int64_t lo = t == 0 ? 0 : base + LINE * (lines * t / team);
        const int64_t hi = t == team - 1 ? n : base + LINE * (lines * (t + 1) / team);
        body(lo, hi);
    }
}

template <typename T>
void axpby_impl(const T* x, const T* y, T* out, int64_t n, T a, T b) {
    elementwise(out, n, 3, [=] { return axpby_op<T>{x, y, simd<T>::set1(a), simd<T>::set1(b)}; });
}

template <typename T>
void fma3_impl(const T* x, const T* y, const T* z, T* out, int64_t n) {
    elementwise(out, n, 4, [=] { return fma3_op<T>{x, y, z}; });
}

template <typename T>
void scaled_relu_impl(const T* x, T* out, int64_t n, T scale) {
    elementwise(out, n, 2, [=] { return scaled_relu_op<T>{x, simd<T>::set1(scale)}; });
}

} // namespace

template <>
void fused_axpby<float>(const float* x, const float* y, float* out, int64_t n, float a, float b) {
    axpby_impl(x, y, out, n, a, b);
}

template <>
void fused_axpby<double>(const double* x, const double* y, double* out, int64_t n, double a, double b) {
    axpby_impl(x, y, out, n, a, b);
}

template <>
void fused_fma3<float>(const float* x, const float* y, const float* z, float* out, int64_t n) {
    fma3_impl(x, y, z, out, n);
}

template <>
void fused_fma3<double>(const double* x, const double* y, const double* z, double* out, int64_t n) {
    fma3_impl(x, y, z, out, n);
}

template <>
void fused_scaled_relu<float>(const float* x, float* out, int64_t n, float scale) {
    scaled_relu_impl(x, out, n, scale);
}

template <>
void fused_scaled_relu<double>(const double* x, double* out, int64_t n, double scale) {
    scaled_relu_impl(x, out, n, scale);
}

template void fused_axpby<float>(const float*, const float*, float*, int64_t, float, float);
template void fused_axpby<double>(const double*, const double*, double*, int64_t, double, double);
template void fused_fma3<float>(const float*, const float*, const float*, float*, int64_t);
template void fused_fma3<double>(const double*, const double*, const double*, double*, int64_t);
template void fused_scaled_relu<float>(const float*, float*, int64_t, float);
template void fused_scaled_relu<double>(const double*, double*, int64_t, double);

} // namespace pg::cpu
