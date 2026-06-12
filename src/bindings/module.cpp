#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "core/common.hpp"
#include "cpu/feature_detect.hpp"
#include "dispatch/dispatch.hpp"

namespace nb = nanobind;

namespace {

// Zero-copy in, owned allocation out. The capsule frees the result buffer when
// the returned ndarray is collected on the Python side.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> matmul_typed(
    const nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>& a,
    const nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>& b) {

    const fme::gemm_dims d = fme::check_matmul_dims(
        static_cast<int64_t>(a.shape(0)), static_cast<int64_t>(a.shape(1)),
        static_cast<int64_t>(b.shape(0)), static_cast<int64_t>(b.shape(1)));

    T* out = new T[static_cast<size_t>(d.m) * static_cast<size_t>(d.n)];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<T*>(p); });

    {
        // The kernel never touches Python objects, so the GIL drops for the
        // whole compute. This is what lets callers thread around the library.
        nb::gil_scoped_release release;
        fme::dispatch::matmul<T>(a.data(), b.data(), out, d.m, d.k, d.n);
    }

    const size_t shape[2] = {static_cast<size_t>(d.m), static_cast<size_t>(d.n)};
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(out, 2, shape, owner);
}

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "fastmathext native core";

    m.def("matmul", &matmul_typed<double>, nb::arg("a"), nb::arg("b"));
    m.def("matmul", &matmul_typed<float>, nb::arg("a"), nb::arg("b"));

    m.def("cpu_features", [] {
        const auto& f = fme::cpu::detect();
        nb::dict d;
        d["avx2"] = f.avx2;
        d["fma"] = f.fma;
        d["avx512f"] = f.avx512f;
        return d;
    });

    m.def("has_cuda_build", [] {
#if defined(FME_HAS_CUDA)
        return true;
#else
        return false;
#endif
    });

    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const fme::shape_error& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });
}
