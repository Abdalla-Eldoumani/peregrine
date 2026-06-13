#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <atomic>
#include <new>

#include "core/common.hpp"
#include "cpu/feature_detect.hpp"
#include "cpu/gemm_blis.hpp"
#include "dispatch/dispatch.hpp"

namespace nb = nanobind;

#if defined(FME_HAS_CUDA)
namespace fme::cuda {
// Forward declaration of the CUDA-free entry point defined in src/cuda/context.cu.
// Declared here rather than via #include "cuda/context.cuh" on purpose: that
// header pulls in cuda_runtime.h/cublas, and a CUDA header in this TU would break
// the deletable-src/cuda invariant and the OFF build. The return type
// fme::cuda_device_info is CUDA-free (core/common.hpp), so the forward
// declaration alone is enough to call across the TU boundary.
cuda_device_info context_device_info();
} // namespace fme::cuda
#endif

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

// Zero-copy in, owned (n, m) copy out: the same capsule pattern as matmul with
// the output extents swapped. transpose deliberately returns an owned buffer,
// not a view, so a caller can never alias the input through the result.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> transpose_typed(
    const nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>& a) {

    const int64_t m = static_cast<int64_t>(a.shape(0));
    const int64_t n = static_cast<int64_t>(a.shape(1));

    T* out = new T[static_cast<size_t>(m) * static_cast<size_t>(n)];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<T*>(p); });

    {
        nb::gil_scoped_release release;
        fme::dispatch::transpose<T>(a.data(), out, m, n);
    }

    const size_t shape[2] = {static_cast<size_t>(n), static_cast<size_t>(m)};
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(out, 2, shape, owner);
}

// Full reduction: returns the scalar in the input dtype. nanobind marshals it
// to a Python float; the wrapper casts it back to the result dtype (f32 round
// trips losslessly through a double-precision Python float). No allocation, so
// no capsule, but the GIL still drops around the kernel for consistency and to
// keep a large reduction from blocking other host threads.
template <typename T>
T sum_all_typed(
    const nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>& a) {

    const int64_t m = static_cast<int64_t>(a.shape(0));
    const int64_t n = static_cast<int64_t>(a.shape(1));

    T result;
    {
        nb::gil_scoped_release release;
        result = fme::dispatch::sum_all<T>(a.data(), m, n);
    }
    return result;
}

// Axis reduction into an owned 1-D buffer: length n for axis 0 (reduce over
// rows), length m for axis 1 (reduce over columns). The wrapper validates axis
// is 0 or 1 before calling, so the binding trusts it and the int crosses
// plainly; None-handling and the axis error wording live in the wrapper.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> sum_axis_typed(
    const nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>& a,
    int axis) {

    const int64_t m = static_cast<int64_t>(a.shape(0));
    const int64_t n = static_cast<int64_t>(a.shape(1));
    const int64_t len = (axis == 0) ? n : m;

    T* out = new T[static_cast<size_t>(len)];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<T*>(p); });

    {
        nb::gil_scoped_release release;
        fme::dispatch::sum_axis<T>(a.data(), out, m, n, axis);
    }

    const size_t shape[1] = {static_cast<size_t>(len)};
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(out, 1, shape, owner);
}

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "fastmathext native core";

    m.def("matmul", &matmul_typed<double>, nb::arg("a"), nb::arg("b"));
    m.def("matmul", &matmul_typed<float>, nb::arg("a"), nb::arg("b"));

    // double before float in every set: the first matching overload wins, and a
    // float64 array must never bind the float32 overload (a silent narrowing).
    m.def("transpose", &transpose_typed<double>, nb::arg("a"));
    m.def("transpose", &transpose_typed<float>, nb::arg("a"));

    m.def("sum_all", &sum_all_typed<double>, nb::arg("a"));
    m.def("sum_all", &sum_all_typed<float>, nb::arg("a"));

    m.def("sum_axis", &sum_axis_typed<double>, nb::arg("a"), nb::arg("axis"));
    m.def("sum_axis", &sum_axis_typed<float>, nb::arg("a"), nb::arg("axis"));

    m.def("cpu_features", [] {
        const auto& f = fme::cpu::detect();
        nb::dict d;
        d["avx2"] = f.avx2;
        d["fma"] = f.fma;
        d["avx512f"] = f.avx512f;
        return d;
    });

    // Private blocking hooks for the MC/KC/NC sweep. MC, KC, NC are loop bounds,
    // not unroll factors, so mutating them at runtime lets benchmarks/sweep_blocking.py
    // walk the whole grid in one process instead of forcing a rebuild per point.
    // Underscore-private: these are a measurement tool, not part of the public API,
    // and changing blocking shifts results bitwise within tolerance (KC repartitions
    // each element's k accumulation), so they must never be reached on a result path.
    // The triple is published in one atomic store (store_blocking) rather than
    // field-by-field: a kernel reads blocking after dropping the GIL, so a torn
    // write would let it size a buffer for one MC and loop for another.
    m.def("_set_gemm_blocking", [](int64_t mc, int64_t kc, int64_t nc) {
        fme::cpu::store_blocking(fme::cpu::blocking{mc, kc, nc});
    });

    m.def("_get_gemm_blocking", [] {
        const fme::cpu::blocking b = fme::cpu::load_blocking();
        nb::dict d;
        d["mc"] = b.mc;
        d["kc"] = b.kc;
        d["nc"] = b.nc;
        return d;
    });

    m.def("has_cuda_build", [] {
#if defined(FME_HAS_CUDA)
        return true;
#else
        return false;
#endif
    });

#if defined(FME_HAS_CUDA)
    // Underscore-private introspection that drives the context singleton to build
    // (GPU-01) and reports the bound device. The wrapper's public has_cuda()
    // (04-05) and the to_device path (04-04) will reach the context through their
    // own entry points; this one exists so the context-init and clean-shutdown
    // tests have a Python-visible trigger before those land. Defined only on the
    // CUDA build: on the OFF build there is no context to introspect and the
    // symbol is absent, matching has_cuda_build() == False. The GIL is held: this
    // queries device props, it launches no kernel, so there is nothing to overlap.
    m.def("_cuda_device_info", [] {
        const fme::cuda_device_info i = fme::cuda::context_device_info();
        nb::dict d;
        d["present"] = i.present;
        d["device_id"] = i.device_id;
        d["cc_major"] = i.cc_major;
        d["cc_minor"] = i.cc_minor;
        d["name"] = i.name;
        d["reason"] = i.reason;
        return d;
    });
#endif

    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const fme::shape_error& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::bad_alloc& e) {
            // A GEMM whose pack buffers (MC*KC*NC, reachable via a pathological
            // _set_gemm_blocking) exceed memory throws bad_alloc out of the
            // kernel; surface it as MemoryError instead of nanobind's default
            // terminate path. The kernel now keeps the throwing allocation off
            // the OpenMP unwinding path so this catch is actually reached.
            PyErr_SetString(PyExc_MemoryError, e.what());
        }
    });
}
