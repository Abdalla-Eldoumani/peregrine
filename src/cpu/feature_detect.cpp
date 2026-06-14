// std::getenv is the standard portable read and the value is only ever compared,
// never used to build a path or a command, so MSVC's /W4 C4996 deprecation
// (which steers toward getenv_s) is noise here; silence it TU-locally.
#define _CRT_SECURE_NO_WARNINGS

#include "cpu/feature_detect.hpp"

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
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

features probe() {
    int r[4] = {0, 0, 0, 0};
    cpuid_count(0, 0, r);
    const int max_leaf = r[0];

    features f{false, false, false};
    if (max_leaf >= 1) {
        cpuid_count(1, 0, r);
        f.fma = (r[2] & (1 << 12)) != 0;
    }
    if (max_leaf >= 7) {
        cpuid_count(7, 0, r);
        f.avx2 = (r[1] & (1 << 5)) != 0;
        f.avx512f = (r[1] & (1 << 16)) != 0;
    }

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
