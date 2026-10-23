// std::getenv is the standard portable read and the value is only ever compared,
// never used to build a path or a command, so MSVC's /W4 C4996 deprecation
// (which steers toward getenv_s) is noise here; silence it TU-locally.
#define _CRT_SECURE_NO_WARNINGS

#include "cpu/feature_detect.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <immintrin.h> // _xgetbv only; this fallback TU must never use an AVX intrinsic
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace pg::cpu {
namespace {

void cpuid_count(int leaf, int subleaf, int out[4]) {
#if defined(_MSC_VER)
    __cpuidex(out, leaf, subleaf);
#else
    unsigned a, b, c, d;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    out[0] = static_cast<int>(a);
    out[1] = static_cast<int>(b);
    out[2] = static_cast<int>(c);
    out[3] = static_cast<int>(d);
#endif
}

// XGETBV raises #UD unless CPUID.1:ECX.OSXSAVE is set, so the caller tests that
// bit first. Inline asm on GCC and Clang because their _xgetbv needs the xsave
// target, which this fallback TU must not enable.
std::uint64_t read_xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    std::uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

features probe() {
    int r[4] = {0, 0, 0, 0};
    cpuid_count(0, 0, r);
    const int max_leaf = r[0];

    // The CPUID feature bits say what the core implements, not whether the OS
    // saves the wider registers across a context switch. Without OS support
    // (OSXSAVE set and XCR0 enabling the state) the first AVX2 instruction
    // faults, so each feature also requires its register state to be enabled.
    features f{false, false, false};
    bool ymm_os = false;
    bool zmm_os = false;
    if (max_leaf >= 1) {
        cpuid_count(1, 0, r);
        f.fma = (r[2] & (1 << 12)) != 0;
        const bool osxsave = (r[2] & (1 << 27)) != 0;
        const bool avx = (r[2] & (1 << 28)) != 0;
        if (osxsave && avx) {
            const std::uint64_t xcr0 = read_xcr0();
            ymm_os = (xcr0 & 0x6) == 0x6;             // XMM (bit 1) and YMM upper (bit 2) state
            zmm_os = ymm_os && (xcr0 & 0xE0) == 0xE0; // opmask, ZMM_Hi256, Hi16_ZMM state
        }
    }
    if (max_leaf >= 7) {
        cpuid_count(7, 0, r);
        f.avx2 = (r[1] & (1 << 5)) != 0;
        f.avx512f = (r[1] & (1 << 16)) != 0;
    }
    f.avx2 = f.avx2 && ymm_os;
    f.fma = f.fma && ymm_os;
    f.avx512f = f.avx512f && zmm_os;

    // PEREGRINE_DISABLE_AVX2 forces the naive fallback so this AVX2 machine can prove
    // the fallback path is reachable and correct (test_fallback.py). It folds in
    // here, before detect() memoizes the result, so every downstream consumer
    // (dispatch routing, the cpu_features binding, the tests) sees one consistent
    // answer for the process lifetime. Any value except a null or literal "0"
    // counts as set, matching the usual environment-flag convention. avx512f is
    // left as probed: nothing routes on it yet, and the override targets the
    // AVX2+FMA fast path specifically.
    const char* disable = std::getenv("PEREGRINE_DISABLE_AVX2");
    if (disable != nullptr && std::strcmp(disable, "0") != 0) {
        f.avx2 = false;
        f.fma = false;
    }
    return f;
}

} // namespace

const features& detect() {
    static const features f = probe();
    return f;
}

} // namespace pg::cpu
