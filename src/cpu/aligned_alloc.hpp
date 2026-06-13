#pragma once
#include <cstddef>
#include <cstdint>
#include <new>

namespace fme::cpu {

// First aligned buffer in the project: the packed A and B panels must start on
// a 64-byte boundary so the microkernel's aligned vector loads never straddle a
// cache line. The aligned operator-new form is chosen so no platform branch is
// needed; _aligned_malloc (MSVC-only) and std::aligned_alloc (absent on MSVC)
// would each force an #if, while sized operator new/delete with align_val_t
// compiles unchanged on MSVC and GCC.
template <typename T>
T* aligned_new(int64_t count) {
    return static_cast<T*>(
        ::operator new(static_cast<std::size_t>(count) * sizeof(T), std::align_val_t{64}));
}

inline void aligned_delete(void* p) noexcept {
    ::operator delete(p, std::align_val_t{64});
}

} // namespace fme::cpu
