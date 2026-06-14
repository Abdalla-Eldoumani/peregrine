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
// present is false ("no device", "driver too old", "compute capability too
// low"). cc_major/minor and name are only meaningful when present.
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

// Element count -> byte count with an overflow and negative-extent check, used
// on every binding allocation (to_device, from_device, the device matmul output)
// before a size_t cast feeds new T[] or alloc_device. Without it, an int64
// rows*cols*elem product that overflows wraps to a small or negative value while
// the copy is told the full size -- a truncated buffer with a full-size copy is a
// heap overflow, and a negative product cast to size_t becomes ~SIZE_MAX.
// CUDA-free (only int64 + shape_error) so it lives in core and both the
// binding and any future caller share one checked path. MSVC has no
// __builtin_mul_overflow, so the check is division-based: a*b overflows iff
// a != 0 and the product divided back by a does not equal b.
inline int64_t checked_bytes(int64_t rows, int64_t cols, int64_t elem) {
    if (rows < 0 || cols < 0 || elem < 0) {
        throw shape_error("array size: negative extent");
    }
    int64_t elems = rows * cols;
    if (rows != 0 && elems / rows != cols) {
        throw shape_error("array size: element count overflows int64");
    }
    int64_t bytes = elems * elem;
    if (elem != 0 && bytes / elem != elems) {
        throw shape_error("array size: byte count overflows int64");
    }
    return bytes;
}

inline gemm_dims check_matmul_dims(int64_t am, int64_t ak, int64_t bk, int64_t bn) {
    if (ak != bk) {
        throw shape_error(
            "matmul: inner dimensions do not match: (" + std::to_string(am) + ", " +
            std::to_string(ak) + ") @ (" + std::to_string(bk) + ", " + std::to_string(bn) + ")");
    }
    return gemm_dims{am, ak, bn};
}

} // namespace fme
