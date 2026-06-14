#pragma once
#include <cstdint>

namespace pg::cpu {

// Out-of-place 2-D transpose: writes the n-by-m transpose of an m-by-n
// row-major array into out. A plain (non-AVX2) translation unit on purpose:
// the copy is memory-bound, so SIMD buys nothing the load/store ports do not
// already saturate, and keeping it intrinsic-free lets it link into the
// no-AVX2 fallback module that the legacy import crash taught us to protect.
template <typename T>
void transpose(const T* a, T* out, int64_t m, int64_t n);

extern template void transpose<float>(const float*, float*, int64_t, int64_t);
extern template void transpose<double>(const double*, double*, int64_t, int64_t);

} // namespace pg::cpu
