#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <atomic>
#include <new>
#include <type_traits>

#include "core/common.hpp"
#include "cpu/feature_detect.hpp"
#include "cpu/gemm_blis.hpp"
#include "dispatch/dispatch.hpp"

namespace nb = nanobind;

#if defined(FME_HAS_CUDA)
// transfer.cuh is included (not forward-declared) because, unlike context.cuh
// and gemm_cublas.cuh, it pulls NO CUDA header -- it declares only the CUDA-free
// DeviceBuffer/DType handle and the transfer functions over void*/int64. fme.Array
// wraps DeviceBuffer, so the binding needs the struct's fields; including the
// header keeps one source of truth for the handle rather than duplicating it. No
// cuda_runtime.h/cublas reaches this TU, so the deletable-src/cuda invariant and
// the byte-identical OFF build both hold (verified: the OFF build links no CUDA).
#include "cuda/transfer.cuh"

namespace fme::cuda {
// Forward declarations of the CUDA-free entry points defined in src/cuda.
// Declared here rather than via #include "cuda/context.cuh"/"cuda/gemm_cublas.cuh"
// on purpose: those headers pull in cuda_runtime.h/cublas, and a CUDA header in
// this TU would break the deletable-src/cuda invariant and the OFF build. The
// signatures use only CUDA-free types (fme::cuda_device_info from
// core/common.hpp, plain pointers and int64), so the forward declarations alone
// are enough to call across the TU boundary.
cuda_device_info context_device_info();

// gemm_host: host-pointer GEMM convenience (allocates device buffers, copies up,
// runs the device gemm, copies the result back, syncs). This entry exists so the
// GPU-02 correctness suite has a callable host->device->host path. Templated in
// src/cuda; the explicit float/double instantiations there satisfy these.
template <typename T>
void gemm_host(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

// gemm: the PURE device-pointer GEMM (04-03). a, b, c are DEVICE pointers; it
// runs cuBLAS on the compute stream and never touches host memory or syncs. The
// device-resident matmul(Array, Array) entry below reaches it through
// fme::dispatch::matmul_device (the routing tier), never by calling it here, so
// the f64-never-AUTO-routes exclusion lives in dispatch where it is unit-testable.
template <typename T>
void gemm(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

// device_probe: the CUDA-free runtime presence verdict (present + cc + name +
// reason), the cheap chain has_cuda() composes WITHOUT building the context (no
// CUDA init, so it respects the <50ms import budget). Forward-declared rather
// than #include "cuda/context.cuh" because that header pulls cuda_runtime.h;
// cuda_device_info is CUDA-free (core/common.hpp), so the declaration alone calls
// across the TU boundary. The wrapper turns present/reason into has_cuda() and
// the "cuda backend requested but <reason>" token.
cuda_device_info device_probe();

// time_matmul: the cudaEvent-timed warm GEMM (the GPU-08 measurement primitive
// 04-06 consumes). a, b are DEVICE pointers; the timed region is the GEMM only
// (no transfer). Forward-declared CUDA-free (plain pointers + int64 + float), so
// the event/stream body stays in gemm_cublas.cu and no CUDA header reaches this
// TU. Returns warm elapsed ms per rep.
template <typename T>
float time_matmul(const T* a, const T* b, int64_t m, int64_t k, int64_t n,
                  int reps, int warmups);
} // namespace fme::cuda

namespace fme::dispatch {
// The device-resident routing entry (04-05): a, b, c are DEVICE pointers; it
// forwards f32/f64 to the cuBLAS GEMM behind the same FME_HAS_CUDA guard. The
// binding's matmul(Array, Array) path calls THIS rather than cuda::gemm directly,
// so the routing decision (and the f64-never-AUTO-routes rule it documents) stays
// in the dispatch tier. CUDA-free signature (pointers + int64), forward-declared.
template <typename T>
void matmul_device(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);
} // namespace fme::dispatch
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

#if defined(FME_HAS_CUDA)
// Host-pointer device GEMM for the GPU-02 correctness suite: same zero-copy-in,
// capsule-owned-out shape as matmul_typed, but the kernel is the cuBLAS device
// GEMM which stages the host operands to the device, computes, and copies back.
// Underscore-private and CUDA-build-only: the public device path is matmul on
// fme.Array (04-04); this exists so f32/f64 GPU correctness can be proven now,
// against the same assert_matmul_close tolerance contract the CPU path uses.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> gemm_host_typed(
    const nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>& a,
    const nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>& b) {

    const fme::gemm_dims d = fme::check_matmul_dims(
        static_cast<int64_t>(a.shape(0)), static_cast<int64_t>(a.shape(1)),
        static_cast<int64_t>(b.shape(0)), static_cast<int64_t>(b.shape(1)));

    T* out = new T[static_cast<size_t>(d.m) * static_cast<size_t>(d.n)];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<T*>(p); });

    {
        // Drop the GIL around the staging + launch + sync, as for the CPU
        // kernel: gemm_host touches no Python object and syncs the stream before
        // it returns, so the result is fully in out by the time we reacquire.
        nb::gil_scoped_release release;
        fme::cuda::gemm_host<T>(a.data(), b.data(), out, d.m, d.k, d.n);
    }

    const size_t shape[2] = {static_cast<size_t>(d.m), static_cast<size_t>(d.n)};
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(out, 2, shape, owner);
}

// fme.Array: the device-resident array. It owns a DeviceBuffer (device pointer +
// shape + dtype, allocated from the context mempool) and frees it in its
// destructor when the Python object is garbage-collected. Single owner: nanobind
// manages this C++ instance's lifetime, so the device buffer is freed exactly
// once, by ~Array, never by a stray capsule deleter -- the use-after-free / leak
// mitigation (a wrong deleter on a device pointer is a DoS, RESEARCH Security
// Domain). The buffer is moved into the instance at construction (to_device /
// the device matmul) and never shared between two live Arrays.
struct Array {
    fme::cuda::DeviceBuffer buf;

    explicit Array(fme::cuda::DeviceBuffer b) : buf(b) {}

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

    ~Array() {
        // free_device tolerates nullptr; a default/empty buffer is a no-op. This
        // is the ONLY place a device buffer owned by an Array is freed.
        if (buf.ptr != nullptr) {
            // ~Array runs during Python GC, INCLUDING interpreter finalization
            // after the atexit teardown destroyed the context streams. An Array
            // held in a module global (or an inline temporary the GC keeps alive
            // past module finalization) reaches here on a dead transfer stream.
            // free_device is teardown-tolerant for exactly that case (CR-01): it
            // skips the free once teardown has run and otherwise inspects the
            // return code by hand, never throwing. So this stays a plain call --
            // a throw escaping a destructor during GC would terminate the
            // interpreter, and the no-throw guarantee now lives in free_device,
            // not in an assumption that GC-time frees never fail.
            fme::cuda::free_device(buf.ptr);
            buf.ptr = nullptr;
        }
    }
};

// Map the runtime DType to the matching nb::dtype for the dlpack/property export.
nb::dlpack::dtype array_dtype(fme::cuda::DType dt) {
    return dt == fme::cuda::DType::f64 ? nb::dtype<double>() : nb::dtype<float>();
}

// Build the dlpack-exporting ndarray view over an Array's device buffer. The
// returned nb::ndarray<nb::array_api, ...> carries the CUDA device type and id,
// and its owner is the Array Python object itself (passed as `self`): nanobind
// holds a reference to it for as long as the exported dltensor/capsule lives, so
// the device buffer cannot be freed under a consumer (cupy/torch) that imported
// it -- ~Array runs only after both the Array and every dlpack view are gone.
// The deleter therefore does NOT free the buffer (that is ~Array's sole job);
// the owner reference is purely keep-alive. numpy cannot consume a CUDA dltensor
// (it has no device memory), so this export is for device-aware consumers; the
// host path is from_device's explicit synced D2H copy.
nb::ndarray<nb::array_api> array_dlpack_view(nb::handle self, const Array& arr) {
    const size_t shape[2] = {static_cast<size_t>(arr.buf.rows),
                             static_cast<size_t>(arr.buf.cols)};
    return nb::ndarray<nb::array_api>(
        arr.buf.ptr, 2, shape, /*owner=*/self, /*strides=*/nullptr,
        array_dtype(arr.buf.dtype), nb::device::cuda::value, /*device_id=*/0);
}

// to_device: copy a host ndarray up to a fresh device buffer and wrap it in an
// fme.Array. Accepts both f32 and f64 (Open Q3: to_device is just memory; the
// f64-never-auto-routes rule is a dispatch concern in 04-05, not a to_device
// rejection). alloc_device pre-flights cudaMemGetInfo; copy_h2d runs on the
// transfer stream. The H2D is async on the transfer stream, but cuBLAS runs on
// the SEPARATE compute stream and the two streams are unordered -- so the buffer
// must be fully resident before it is handed back, or a later GEMM on compute
// can read an operand whose H2D copy has not landed (a cross-stream
// use-before-ready race: compute-sanitizer --track-stream-ordered-races all
// reports it as use-before-alloc and the GEMM returns garbage under
// instrumentation). sync_transfer() after the copy makes the operand ready
// before any compute-stream op can touch it, symmetric to from_device's
// D2H-before-return sync. A pure H2D (no compute consumer) would not strictly
// need it, but to_device's whole purpose is to feed the device GEMM, so the
// fence belongs here at the residency boundary.
template <typename T>
Array* to_device_typed(
    const nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>& a) {

    const int64_t rows = static_cast<int64_t>(a.shape(0));
    const int64_t cols = static_cast<int64_t>(a.shape(1));
    const fme::cuda::DType dt =
        std::is_same_v<T, double> ? fme::cuda::DType::f64 : fme::cuda::DType::f32;
    const int64_t bytes = rows * cols * static_cast<int64_t>(sizeof(T));

    fme::cuda::DeviceBuffer buf{nullptr, rows, cols, dt};
    {
        // Drop the GIL around the allocation + copy + sync: alloc_device,
        // copy_h2d, and sync_transfer touch no Python object. a.data() stays
        // valid -- nanobind holds the ndarray alive across the call -- and the
        // copy reads from it on the transfer stream.
        nb::gil_scoped_release release;
        buf.ptr = fme::cuda::alloc_device(bytes);
        fme::cuda::copy_h2d(buf.ptr, a.data(), bytes);
        // Fence the transfer stream so the operand is fully resident before the
        // returned Array can be read by a GEMM on the unordered compute stream.
        fme::cuda::sync_transfer();
    }
    return new Array(buf);
}

// from_device: copy an fme.Array's device buffer back to a FRESH host ndarray.
// numpy has no device memory and cannot pull a CUDA dltensor through dlpack, so
// this does an explicit D2H copy on the transfer stream and -- the hard contract
// -- SYNCS the transfer stream before the host buffer is returned to Python
// (async wins live inside a call, never across the API boundary; a D2H result
// returned before its stream syncs is a data race the caller never sees). The
// returned ndarray owns a fresh host allocation via a capsule whose deleter
// frees exactly that allocation.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> from_device_typed(const Array& arr) {
    const int64_t rows = arr.buf.rows;
    const int64_t cols = arr.buf.cols;
    const int64_t bytes = rows * cols * static_cast<int64_t>(sizeof(T));

    T* out = new T[static_cast<size_t>(rows) * static_cast<size_t>(cols)];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<T*>(p); });

    {
        nb::gil_scoped_release release;
        fme::cuda::copy_d2h(out, arr.buf.ptr, bytes);
        // Sync BEFORE the host buffer crosses back into Python: out is fully
        // written only after the transfer stream drains.
        fme::cuda::sync_transfer();
    }

    const size_t shape[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(out, 2, shape, owner);
}

// The REQUIRED device-resident matmul: (Array, Array) -> Array, device-in/
// device-out. It validates the operands are the same dtype and shape-conformable,
// allocates the output device buffer via alloc_device, and runs the PURE device
// GEMM (fme::cuda::gemm<T>) directly on the operands' device pointers -- NO host
// staging, no D2H. The result is a fresh fme.Array wrapping the output buffer.
//
// Deliberately minimal here: this is the device-in/device-out primitive 04-05's
// wrapper consumes for the f32 device-resident GPU path (GPU-05 routing). The
// full residency dispatch -- the mixed-residency TypeError, the residency-out
// return-type selection, the f64-never-auto-routes exclusion, and the warn-once
// CPU fallback -- is layered on top in 04-05; this entry must not silently
// transfer either (it never touches host memory). A dtype/shape mismatch raises
// (shape_error -> ValueError via the translator) rather than computing garbage.
Array* matmul_device(const Array& a, const Array& b) {
    if (a.buf.dtype != b.buf.dtype) {
        throw fme::shape_error(
            "matmul: device operands must have the same dtype");
    }
    const fme::gemm_dims d = fme::check_matmul_dims(a.buf.rows, a.buf.cols,
                                                    b.buf.rows, b.buf.cols);

    const fme::cuda::DType dt = a.buf.dtype;
    const int64_t elem = fme::cuda::dtype_size(dt);
    const int64_t out_bytes = d.m * d.n * elem;

    fme::cuda::DeviceBuffer out{nullptr, d.m, d.n, dt};
    {
        // Drop the GIL around the alloc + device GEMM + syncs: none touch a
        // Python object. The ordering is the correctness contract here, NOT an
        // accident of which stream wins a race:
        //   - The operands were made resident by to_device's sync_transfer, so
        //     cuBLAS on the compute stream can read them.
        //   - The output buffer is allocated on the TRANSFER stream
        //     (alloc_device), but the GEMM writes it on the COMPUTE stream, and
        //     the two streams are unordered -- so sync_transfer() fences the
        //     allocation complete before the GEMM can write the buffer (without
        //     it, compute-sanitizer --track-stream-ordered-races all flags the
        //     GEMM's writes as use-before-alloc).
        //   - sync_compute() after the GEMM fences the result fully written
        //     before the Array crosses back to Python, the compute-side twin of
        //     from_device's D2H sync.
        nb::gil_scoped_release release;
        out.ptr = fme::cuda::alloc_device(out_bytes);
        // Fence the output allocation (transfer stream) before the GEMM writes it
        // on the unordered compute stream.
        fme::cuda::sync_transfer();
        // Route through the dispatch tier, not cuda::gemm directly: the
        // device-resident routing decision (and the f64-never-AUTO-routes rule it
        // documents) belongs in src/dispatch. Both operands are already an
        // fme.Array here, so this is the forced device-resident path -- f64 is
        // allowed and computed, the auto-route exclusion is enforced upstream by
        // the wrapper never sending a host f64 array down here.
        if (dt == fme::cuda::DType::f64) {
            fme::dispatch::matmul_device<double>(
                static_cast<const double*>(a.buf.ptr),
                static_cast<const double*>(b.buf.ptr),
                static_cast<double*>(out.ptr), d.m, d.k, d.n);
        } else {
            fme::dispatch::matmul_device<float>(
                static_cast<const float*>(a.buf.ptr),
                static_cast<const float*>(b.buf.ptr),
                static_cast<float*>(out.ptr), d.m, d.k, d.n);
        }
        // Fence the GEMM complete: the output buffer is fully written before the
        // result Array is usable (a from_device on it must see the finished
        // product, not a half-written buffer).
        fme::cuda::sync_compute();
    }
    return new Array(out);
}

// The cudaEvent-timed device matmul: (Array x, Array y, reps, warmups) -> warm
// ms per rep. Both operands are device-resident, so the timed region is the GEMM
// only -- no H2D/D2H. This is the firm GPU-08 timing primitive 04-06's bench
// reads instead of a wall-clock approximation around an async launch. Same
// dtype + conformable-shape validation as matmul_device (a mismatch is a
// shape_error -> ValueError, never a fake measurement). The GIL drops around the
// whole warmup + timed run; time_matmul itself allocates the single output
// buffer and creates/destroys the events.
float time_matmul_entry(const Array& x, const Array& y, int reps, int warmups) {
    if (x.buf.dtype != y.buf.dtype) {
        throw fme::shape_error(
            "matmul: device operands must have the same dtype");
    }
    const fme::gemm_dims d = fme::check_matmul_dims(x.buf.rows, x.buf.cols,
                                                    y.buf.rows, y.buf.cols);
    const fme::cuda::DType dt = x.buf.dtype;

    float ms = 0.0f;
    {
        nb::gil_scoped_release release;
        if (dt == fme::cuda::DType::f64) {
            ms = fme::cuda::time_matmul<double>(
                static_cast<const double*>(x.buf.ptr),
                static_cast<const double*>(y.buf.ptr), d.m, d.k, d.n, reps,
                warmups);
        } else {
            ms = fme::cuda::time_matmul<float>(
                static_cast<const float*>(x.buf.ptr),
                static_cast<const float*>(y.buf.ptr), d.m, d.k, d.n, reps,
                warmups);
        }
    }
    return ms;
}
#endif

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

    // The PRIVATE compile-time build signal. Underscore-private: the public
    // predicate is the wrapper's has_cuda() (04-05), which composes this build
    // flag AND a usable-device probe (_cuda_device_probe below) into the
    // build-AND-usable-device meaning DESIGN_SYSTEM.md names. Renamed from the old
    // public has_cuda_build: a build flag alone is not the useful predicate
    // (dispatch must not route to a GPU that is absent), so the build-only answer
    // stays private and the runtime chain is the public surface.
    m.def("_has_cuda_build", [] {
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
    // symbol is absent, matching _has_cuda_build() == False. The GIL is held: this
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

    // The CHEAP, never-building device presence probe the wrapper's has_cuda()
    // composes its runtime chain from. Unlike _cuda_device_info, this calls
    // device_probe (cudaGetDeviceCount + a props read), which does NOT build the
    // context -- so has_cuda() pays no stream/handle/pool creation and no CUDA
    // init, honoring the <50ms import budget when called lazily. Returns present
    // plus the cc and the reason string ("no device" / "driver too old" /
    // "compute capability too low") the wrapper turns into the "cuda backend
    // requested but <reason>" RuntimeError token. Never throws on a driverless box.
    m.def("_cuda_device_probe", [] {
        const fme::cuda_device_info i = fme::cuda::device_probe();
        nb::dict d;
        d["present"] = i.present;
        d["device_id"] = i.device_id;
        d["cc_major"] = i.cc_major;
        d["cc_minor"] = i.cc_minor;
        d["name"] = i.name;
        d["reason"] = i.reason;
        return d;
    });

    // double before float, like every other overload set: a float64 array must
    // never bind the float32 overload (a silent narrowing). Private and
    // CUDA-only; the public device matmul is exposed via the Array overload of
    // matmul below (04-04), wired into fme.matmul by 04-05.
    m.def("_gemm_host", &gemm_host_typed<double>, nb::arg("a"), nb::arg("b"));
    m.def("_gemm_host", &gemm_host_typed<float>, nb::arg("a"), nb::arg("b"));

    // fme.Array: the device-resident array handle. Read-only shape (a tuple) and
    // dtype (the numpy dtype), and __dlpack__/__dlpack_device__ reporting a CUDA
    // device. The class owns its device buffer and frees it on GC (~Array); the
    // dlpack export keeps the Array alive via the owner reference rather than
    // freeing through a capsule deleter, so a device-aware consumer can hold the
    // exported tensor without a use-after-free.
    nb::class_<Array>(m, "Array")
        .def_prop_ro(
            "shape",
            [](const Array& self) {
                return nb::make_tuple(self.buf.rows, self.buf.cols);
            })
        .def_prop_ro(
            "dtype",
            [](const Array& self) {
                // Hand back the numpy dtype object so x.dtype == np.float32 holds
                // on the Python side, matching how an ndarray reports its dtype.
                return nb::module_::import_("numpy").attr(
                    self.buf.dtype == fme::cuda::DType::f64 ? "float64"
                                                            : "float32");
            })
        .def(
            "__dlpack__",
            [](nb::handle self) {
                const Array& arr = nb::cast<const Array&>(self);
                return array_dlpack_view(self, arr);
            })
        .def("__dlpack_device__", [](const Array&) {
            // (device_type, device_id): kDLCUDA is 2 in the DLPack device enum.
            // Reported directly so a consumer's __dlpack_device__ negotiation
            // sees CUDA without constructing the full tensor.
            return nb::make_tuple(static_cast<int>(nb::device::cuda::value), 0);
        });

    // to_device(ndarray) -> fme.Array. double before float (a float64 array must
    // never bind the float32 overload). Accepts both dtypes -- it is just memory.
    m.def("to_device", &to_device_typed<double>, nb::arg("a"));
    m.def("to_device", &to_device_typed<float>, nb::arg("a"));

    // from_device(fme.Array) -> ndarray. One entry, dtype-dispatched off the
    // Array (no host operand to overload on): the D2H copy + transfer-stream sync
    // happen in from_device_typed before the host ndarray is returned.
    m.def(
        "from_device",
        [](const Array& arr) -> nb::object {
            return arr.buf.dtype == fme::cuda::DType::f64
                       ? nb::cast(from_device_typed<double>(arr))
                       : nb::cast(from_device_typed<float>(arr));
        },
        nb::arg("x"));

    // The REQUIRED device-resident matmul: matmul(Array, Array) -> Array,
    // device-in/device-out, dtype-dispatched. Registered as a matmul overload
    // AFTER the host ndarray overloads, so a host pair still binds the CPU path
    // and a device pair binds this one. 04-05's wrapper consumes this for the
    // GPU-05 device-resident routing; Task 3 round-trips it against the CPU
    // result through assert_matmul_close.
    m.def("matmul", &matmul_device, nb::arg("a"), nb::arg("b"));

    // The cudaEvent-timed device-matmul timer (GPU-08 primitive 04-06 consumes).
    // Both operands are device-resident fme.Array, so the timed region is the
    // GEMM only -- no transfer. Returns warm elapsed ms per rep from a cudaEvent
    // pair recorded after a sync. Underscore-private: it is a measurement tool the
    // bench reaches through _core, not part of the public surface. Defined only on
    // the CUDA build (it times a device GEMM), absent on the OFF build.
    m.def("_cuda_time_matmul", &time_matmul_entry, nb::arg("x"), nb::arg("y"),
          nb::arg("reps"), nb::arg("warmups"));
#endif

    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const fme::shape_error& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const fme::cuda_error& e) {
            // The ONE new arm for the CUDA path (04-04). Every FME_CUDA_CHECK /
            // FME_CUBLAS_CHECK failure and the alloc_device OOM pre-flight throw
            // fme::cuda_error carrying the cuda error NAME (e.g.
            // cudaErrorMemoryAllocation) plus file:line. Mapped to RuntimeError
            // here; the distinct user-facing wording (the byte-math OOM message,
            // the driver-too-old message) is composed in the wrapper (04-05),
            // keyed off the name in e.what(). cuda_error lives in core/common.hpp
            // and carries no CUDA type, so this arm compiles in every build; on
            // the OFF build nothing throws it, so it is simply never reached. One
            // translator, one new arm -- never a second register call
            // (src/bindings/CLAUDE.md).
            PyErr_SetString(PyExc_RuntimeError, e.what());
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
