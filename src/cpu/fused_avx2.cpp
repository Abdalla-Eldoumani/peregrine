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

// MSVC kept four() below as a call per group, with the op passed through memory.
#if defined(_MSC_VER)
#define PG_FUSED_INLINE __forceinline
#else
#define PG_FUSED_INLINE inline __attribute__((always_inline))
#endif

// The two lines a four-vector group reads from one input, FUSED_PREFETCH_BYTES
// ahead of the group.
template <typename T>
void ahead(const T* p) {
    const char* q = reinterpret_cast<const char*>(p) + FUSED_PREFETCH_BYTES;
    _mm_prefetch(q, _MM_HINT_T0);
    _mm_prefetch(q + 64, _MM_HINT_T0);
}

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
    static reg nan(reg v) { return _mm256_cmp_ps(v, v, _CMP_UNORD_Q); }
    static reg either(reg m, reg k) { return _mm256_or_ps(m, k); }
    static bool any(reg m) { return _mm256_movemask_ps(m) != 0; }
    // Lanes where v is a NaN get v with the quiet bit set (what arithmetic makes
    // of it), the others keep r. The bit is set only where v is already a NaN:
    // on an infinity it would make one.
    static reg pickq(reg r, reg v) {
        const reg q = _mm256_or_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x00400000)));
        return _mm256_blendv_ps(r, q, nan(v));
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
    static reg nan(reg v) { return _mm256_cmp_pd(v, v, _CMP_UNORD_Q); }
    static reg either(reg m, reg k) { return _mm256_or_pd(m, k); }
    static bool any(reg m) { return _mm256_movemask_pd(m) != 0; }
    static reg pickq(reg r, reg v) {
        const reg q = _mm256_or_pd(v, _mm256_castsi256_pd(_mm256_set1_epi64x(0x0008000000000000)));
        return _mm256_blendv_pd(r, q, nan(v));
    }
};

// Each op computes the vector at element i from operands read through ld, a
// full or a masked load. Named products in a fixed order keep axpby's two
// multiplies and one add; MSVC /fp:precise never contracts them, so it matches
// the unfused NumPy expression and the scalar twin bit for bit.
//
// When two NaNs meet in one instruction the result carries the payload of the
// operand in the first source slot, and the compiler may commute a multiply or
// an add, or pick a different FMA form, in each instantiation: the peel, body,
// and tail would then disagree for the same element, so a result's bits would
// depend on alignment. A vector with any NaN lane is therefore redone with the
// payload chosen explicitly: the first NaN operand in written order (a, x, b, y;
// x, y, z; scale, x), quieted. The scalar twin applies the same rule.
//
// plain(i) is the result without that rule; fix(r, i) returns r when it has no
// NaN lane and otherwise redoes the vector at i under the rule (a NaN-free
// result means no operand lane was a NaN, so the rule would leave it as it is).
// The whole-vector loop tests several plain results at once and fixes only
// when some lane is a NaN; operator() is the two together, for single vectors.
// touch(i) prefetches the inputs ahead of the group at i when streaming.
template <typename T>
struct axpby_op {
    using S = simd<T>;
    const T* x;
    const T* y;
    typename S::reg a;
    typename S::reg b;
    void touch(int64_t i) const {
        ahead(x + i);
        ahead(y + i);
    }
    template <typename Load>
    typename S::reg plain(int64_t i, const Load& ld) const {
        return S::add(S::mul(a, ld(x + i)), S::mul(b, ld(y + i)));
    }
    template <typename Load>
    typename S::reg fix(typename S::reg r, int64_t i, const Load& ld) const {
        if (!S::any(S::nan(r))) {
            return r;
        }
        const typename S::reg ax = S::pickq(S::mul(a, ld(x + i)), a);
        const typename S::reg by = S::pickq(S::mul(b, ld(y + i)), b);
        return S::pickq(S::add(ax, by), ax);
    }
    template <typename Load>
    typename S::reg operator()(int64_t i, const Load& ld) const {
        return fix(plain(i, ld), i, ld);
    }
};

template <typename T>
struct fma3_op {
    using S = simd<T>;
    const T* x;
    const T* y;
    const T* z;
    void touch(int64_t i) const {
        ahead(x + i);
        ahead(y + i);
        ahead(z + i);
    }
    template <typename Load>
    typename S::reg plain(int64_t i, const Load& ld) const {
        return S::fmadd(ld(x + i), ld(y + i), ld(z + i));
    }
    template <typename Load>
    typename S::reg fix(typename S::reg r, int64_t i, const Load& ld) const {
        if (!S::any(S::nan(r))) {
            return r;
        }
        return S::pickq(S::pickq(S::pickq(r, ld(z + i)), ld(y + i)), ld(x + i));
    }
    template <typename Load>
    typename S::reg operator()(int64_t i, const Load& ld) const {
        return fix(plain(i, ld), i, ld);
    }
};

// relu keeps a NaN lane, so plain's NaN lanes are exactly those of scale * x.
template <typename T>
struct scaled_relu_op {
    using S = simd<T>;
    const T* x;
    typename S::reg scale;
    // One input stream measured no gain from it (see FUSED_PREFETCH_BYTES).
    void touch(int64_t) const {}
    template <typename Load>
    typename S::reg plain(int64_t i, const Load& ld) const {
        return S::relu(S::mul(scale, ld(x + i)));
    }
    template <typename Load>
    typename S::reg fix(typename S::reg r, int64_t i, const Load& ld) const {
        if (!S::any(S::nan(r))) {
            return r;
        }
        return S::relu(S::pickq(S::mul(scale, ld(x + i)), scale));
    }
    template <typename Load>
    typename S::reg operator()(int64_t i, const Load& ld) const {
        return fix(plain(i, ld), i, ld);
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

template <store_kind K, typename T>
void put(T* p, typename simd<T>::reg v) {
    using S = simd<T>;
    if constexpr (K == store_kind::stream) {
        S::stream(p, v);
    } else if constexpr (K == store_kind::aligned) {
        S::store_aligned(p, v);
    } else {
        S::storeu(p, v);
    }
}

// The whole vector at element i.
template <store_kind K, typename T, typename Op>
void one(const Op& op, T* out, int64_t i) {
    using S = simd<T>;
    put<K>(out + i, op(i, [](const T* p) { return S::load(p); }));
}

// The four whole vectors from element i: their plain results under one NaN
// test, then the stores, which follow every load of the four (so in place stays
// safe). One vector a step spent about 16 fused uops on it, 4.4 cycles a vector
// where the two load ports allow 1.5. Kernel alone, one thread, in L1, one
// vector -> four a step: fma3 f64 1024 elements 255 -> 149-180 ns, f32 136 ->
// 88-100.
template <store_kind K, typename T, typename Op>
PG_FUSED_INLINE void four(const Op& op, T* out, int64_t i) {
    using S = simd<T>;
    constexpr int64_t W = S::W;
    const auto full = [](const T* p) { return S::load(p); };
    if constexpr (K == store_kind::stream) {
        op.touch(i);
    }
    typename S::reg r0 = op.plain(i, full);
    typename S::reg r1 = op.plain(i + W, full);
    typename S::reg r2 = op.plain(i + 2 * W, full);
    typename S::reg r3 = op.plain(i + 3 * W, full);
    if (S::any(S::either(S::either(S::nan(r0), S::nan(r1)), S::either(S::nan(r2), S::nan(r3))))) {
        r0 = op.fix(r0, i, full);
        r1 = op.fix(r1, i + W, full);
        r2 = op.fix(r2, i + 2 * W, full);
        r3 = op.fix(r3, i + 3 * W, full);
    }
    put<K>(out + i, r0);
    put<K>(out + i + W, r1);
    put<K>(out + i + 2 * W, r2);
    put<K>(out + i + 3 * W, r3);
}

// One thread's elements [lo, hi): the head up to out's first 32-byte boundary
// (aligned kinds only), whole vectors four at a time and then singly, and the
// tail, in ascending order or with Down in descending order. The pieces and
// their instructions are the same either way, so the order never changes a
// bit. In place (out is exactly one of the inputs) is safe with every store kind
// and order: each element is loaded in the step that stores it, before the
// store, and no thread or lane ever reads an element another one stores, so a
// weakly ordered streaming store has no reader to race.
template <store_kind K, bool Down, typename T, typename Op>
void range(const Op& shared, T* out, int64_t lo, int64_t hi) {
    using S = simd<T>;
    constexpr int64_t W = S::W;
    // A local copy: through the reference MSVC reloaded the operand pointers
    // after every store, which might have written them.
    const Op op = shared;
    int64_t head = 0;
    if constexpr (K != store_kind::unaligned) {
        const auto addr = reinterpret_cast<std::uintptr_t>(out + lo);
        head = static_cast<int64_t>((32 - addr % 32) % 32 / sizeof(T));
        head = head < hi - lo ? head : hi - lo;
    }
    const int64_t b0 = lo + head;
    const int64_t vectors = (hi - b0) / W;
    const int64_t b4 = b0 + vectors / 4 * 4 * W;
    const int64_t b1 = b0 + vectors * W;
    if constexpr (Down) {
        if (b1 < hi) {
            part(op, out, b1, hi - b1);
        }
        for (int64_t i = b1 - W; i >= b4; i -= W) {
            one<K>(op, out, i);
        }
        for (int64_t i = b4 - 4 * W; i >= b0; i -= 4 * W) {
            four<K>(op, out, i);
        }
        if (head > 0) {
            part(op, out, lo, head);
        }
    } else {
        if (head > 0) {
            part(op, out, lo, head);
        }
        for (int64_t i = b0; i < b4; i += 4 * W) {
            four<K>(op, out, i);
        }
        for (int64_t i = b4; i < b1; i += W) {
            one<K>(op, out, i);
        }
        if (b1 < hi) {
            part(op, out, b1, hi - b1);
        }
    }
}

// 4K aliasing: a load waits for an earlier store still in flight whose address
// matches it in bits 11:0. With out a little past an input modulo 4 KiB, an
// ascending loop's loads of the next vectors match the stores it has just made;
// a descending loop made those loads first. NumPy arrays allocated one after
// another land like that here (8 KiB blocks 8208 bytes apart), and the bench's
// fresh fma3 float64 32x32 outputs sit 32 to 208 bytes past every input. So the
// loop runs down when out is within FUSED_ALIAS_BYTES past some input and not
// within it before any other (which would stall the descending loop instead).
template <typename T>
bool descend(const T* out, const T* const (&in)[3]) {
    bool past = false;
    bool before = false;
    for (const T* p : in) {
        const std::uintptr_t d =
            (reinterpret_cast<std::uintptr_t>(out) - reinterpret_cast<std::uintptr_t>(p)) % 4096;
        past = past || (d != 0 && d < FUSED_ALIAS_BYTES);
        before = before || d > 4096 - FUSED_ALIAS_BYTES;
    }
    return past && !before;
}

// make() builds the op inside the region, so each thread broadcasts the scalars
// into its own registers instead of reloading them from the shared frame.
// arrays counts the buffers the op touches, out included; in holds the inputs,
// repeated to fill three.
template <typename T, typename Make>
void elementwise(T* out, int64_t n, int arrays, const T* const (&in)[3], const Make& make) {
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
    // An in-place call counts out twice, so it may stream and cap a little early.
    const double touched = bytes * static_cast<double>(arrays);
    int nt = team_size(bytes, FUSED_SERIAL_BELOW_BYTES, FUSED_PER_THREAD_BYTES, lines);
    if (touched > FUSED_DRAM_BYTES && nt > FUSED_DRAM_STREAMS / arrays) {
        nt = FUSED_DRAM_STREAMS / arrays;
    }
    const bool stream = touched > FUSED_STREAM_BYTES;
    // Below the serial size the buffers sit in L1 or L2 and the peel's masked
    // head and tail cost more than the line-split vectors they save. fma3 out=
    // float32, one thread: 8x8 72 -> 65 ns, 32x32 254 -> 193-212, 64x64 963 ->
    // 797-840 without the peel.
    const bool peel = aligned && bytes >= FUSED_SERIAL_BELOW_BYTES;
    const store_kind kind = !peel ? store_kind::unaligned : stream ? store_kind::stream : store_kind::aligned;
    // Streaming calls stay ascending: their DRAM numbers (the team cap and the
    // prefetch distance) were measured ascending only.
    const bool down = kind != store_kind::stream && bytes >= FUSED_ALIAS_MIN_BYTES && descend(out, in);
    const auto body = [&](int64_t lo, int64_t hi) {
        const auto op = make();
        if (kind == store_kind::stream) {
            range<store_kind::stream, false>(op, out, lo, hi);
            // Streaming stores are weakly ordered; fence them before the
            // region's closing barrier hands the result to the caller.
            _mm_sfence();
        } else if (kind == store_kind::aligned) {
            if (down) {
                range<store_kind::aligned, true>(op, out, lo, hi);
            } else {
                range<store_kind::aligned, false>(op, out, lo, hi);
            }
        } else if (down) {
            range<store_kind::unaligned, true>(op, out, lo, hi);
        } else {
            range<store_kind::unaligned, false>(op, out, lo, hi);
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
    elementwise(out, n, 3, {x, y, y}, [=] { return axpby_op<T>{x, y, simd<T>::set1(a), simd<T>::set1(b)}; });
}

template <typename T>
void fma3_impl(const T* x, const T* y, const T* z, T* out, int64_t n) {
    elementwise(out, n, 4, {x, y, z}, [=] { return fma3_op<T>{x, y, z}; });
}

template <typename T>
void scaled_relu_impl(const T* x, T* out, int64_t n, T scale) {
    elementwise(out, n, 2, {x, x, x}, [=] { return scaled_relu_op<T>{x, simd<T>::set1(scale)}; });
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
