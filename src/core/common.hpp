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
