#pragma once

namespace pg::cpu {

struct features {
    bool avx2;
    bool fma;
    bool avx512f;
};

// Queried once at module import. SIMD kernels gate on this at runtime so a
// machine without AVX2 gets the scalar path instead of an illegal-instruction
// crash, which is how the legacy .pyd failed.
const features& detect();

} // namespace pg::cpu
