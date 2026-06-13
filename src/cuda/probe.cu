#include "cuda/check.cuh"

#include <cstdint>

namespace fme::cuda {

// Minimal build probe. Its only job is to prove the CUDA toolchain is wired:
// nvcc compiles a .cu for sm_86 with the same C++20 standard as the .cpp half,
// MSVC accepts the host pass, the CUDA::cudart/cublas/cublasLt link line
// resolves, and check.cuh (the CHECK macros) compiles inside a translation
// unit nvcc owns. The real device probe behind has_cuda (driver load, device
// count, compute capability) lands with the context singleton in a later plan;
// returning true here only attests that this TU was compiled into _core.
//
// int64_t for the touched-element count even though it is one: index and size
// math in this directory is int64_t throughout (an int in a flattened index is
// a bug here even when it fits), so the probe sets the convention from line one.
bool probe_cuda_built() {
    const std::int64_t compiled_units = 1;
    return compiled_units > 0;
}

} // namespace fme::cuda
