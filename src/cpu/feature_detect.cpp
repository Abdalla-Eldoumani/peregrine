#include "cpu/feature_detect.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace fme::cpu {
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
    return f;
}

} // namespace

const features& detect() {
    static const features f = probe();
    return f;
}

} // namespace fme::cpu
