#pragma once
#include <cstdint>

namespace fme::dispatch {

// Every op routes through here. v1 has one backend (cpu naive); the packed
// CPU kernels, the CUDA backend, and the calibrated crossover table all land
// behind this same entry point so the binding layer never changes.
template <typename T>
void matmul(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

extern template void matmul<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
extern template void matmul<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

} // namespace fme::dispatch
