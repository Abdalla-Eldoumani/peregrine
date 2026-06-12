#include "dispatch/dispatch.hpp"
#include "cpu/gemm_naive.hpp"

namespace fme::dispatch {

template <typename T>
void matmul(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n) {
    cpu::gemm_naive<T>(a, b, c, m, k, n);
}

template void matmul<float>(const float*, const float*, float*, int64_t, int64_t, int64_t);
template void matmul<double>(const double*, const double*, double*, int64_t, int64_t, int64_t);

} // namespace fme::dispatch
