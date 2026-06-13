#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

namespace fme {

// One exception type at the boundary. The binding layer translates it to
// ValueError so Python callers see NumPy-style errors, not RuntimeError.
struct shape_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// Lives in core, not src/cuda, so the one binding translator can catch it
// without including a CUDA header, and so the CUDA TUs that throw it share the
// same type. It carries NO CUDA type: the error name is a std::string the CHECK
// macro builds from cudaGetErrorName/cublasGetStatusName. That keeps this header
// CUDA-include-free in every build configuration (the deletable-src/cuda
// invariant) while still giving both sides one exception type. Derived from
// runtime_error because a CUDA/cuBLAS failure is a runtime fault, not a bad
// argument the way shape_error is.
struct cuda_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// CUDA-free description of the device the context singleton bound, plus the
// runtime presence verdict. Lives here, not in context.cuh, for the same reason
// cuda_error does: the binding boundary (module.cpp) reads it to answer a
// has_cuda-style introspection call WITHOUT including a CUDA header, so src/cuda
// stays deletable. It carries no CUDA type -- the .cu fills it from
// cudaDeviceProp and a cudaGetDeviceCount/compute-capability probe and hands
// back this plain struct. present is the device-usable verdict (driver loads,
// device count > 0, compute capability >= 7.0); reason names the failure when
// present is false (mapping to the EDGE_CASES device chain: "no device",
// "driver too old", "compute capability too low"). cc_major/minor and name are
// only meaningful when present.
struct cuda_device_info {
    bool present;
    int device_id;
    int cc_major;
    int cc_minor;
    std::string name;
    std::string reason;
};

struct gemm_dims {
    int64_t m;
    int64_t k;
    int64_t n;
};

inline gemm_dims check_matmul_dims(int64_t am, int64_t ak, int64_t bk, int64_t bn) {
    if (ak != bk) {
        throw shape_error(
            "matmul: inner dimensions do not match: (" + std::to_string(am) + ", " +
            std::to_string(ak) + ") @ (" + std::to_string(bk) + ", " + std::to_string(bn) + ")");
    }
    return gemm_dims{am, ak, bn};
}

} // namespace fme
