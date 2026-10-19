#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <atomic>
#include <memory>
#include <new>
#include <type_traits>

#include "bindings/fast_entry.hpp"
#include "core/common.hpp"
#include "cpu/feature_detect.hpp"
#include "cpu/gemm_blis.hpp"
#include "cpu/gemm_skinny.hpp"
#include "cpu/gemm_small.hpp"
#include "cpu/threads.hpp"
#include "dispatch/dispatch.hpp"

namespace nb = nanobind;

#if defined(PG_HAS_CUDA)
// transfer.cuh is included (not forward-declared) because, unlike context.cuh
// and gemm_cublas.cuh, it pulls NO CUDA header -- it declares only the CUDA-free
// DeviceBuffer/DType handle and the transfer functions over void*/int64. pg.Array
// wraps DeviceBuffer, so the binding needs the struct's fields; including the
// header keeps one source of truth for the handle rather than duplicating it. No
// cuda_runtime.h/cublas reaches this TU, so the deletable-src/cuda invariant and
// the byte-identical OFF build both hold (verified: the OFF build links no CUDA).
#include "cuda/transfer.cuh"

namespace pg::cuda {
// Forward declarations of the CUDA-free entry points defined in src/cuda.
// Declared here rather than via #include "cuda/context.cuh"/"cuda/gemm_cublas.cuh"
// on purpose: those headers pull in cuda_runtime.h/cublas, and a CUDA header in
// this TU would break the deletable-src/cuda invariant and the OFF build. The
// signatures use only CUDA-free types (pg::cuda_device_info from
// core/common.hpp, plain pointers and int64), so the forward declarations alone
// are enough to call across the TU boundary.
cuda_device_info context_device_info();
void teardown();
void inject_fault();

// gemm_host: the staged GEMM for host operands (pre-flight, device buffers,
// copies up, the device gemm, the copy back, and the host fence, on one stream).
// Templated in src/cuda; the explicit float/double instantiations there satisfy
// these.
template <typename T>
void gemm_host(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

// gemm: the PURE device-pointer GEMM. a, b, c are DEVICE pointers; it runs cuBLAS
// on the context stream and never touches host memory or syncs. The
// device-resident matmul(Array, Array) entry below reaches it through
// pg::dispatch::matmul_device (the routing tier), never by calling it here, so
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

// time_matmul: the cudaEvent-timed warm GEMM (the measurement primitive the bench
// consumes). a, b are DEVICE pointers; the timed region is the GEMM only (no
// transfer). Forward-declared CUDA-free (plain pointers + int64 + float), so the
// event/stream body stays in gemm_cublas.cu and no CUDA header reaches this TU.
// Returns warm elapsed ms per rep.
template <typename T>
float time_matmul(const T* a, const T* b, int64_t m, int64_t k, int64_t n,
                  int reps, int warmups);

// time_fused_chain: the cudaEvent-timed warm fused CHAIN (the measurement
// primitive the bench consumes). x, y, z are DEVICE pointers; the timed region
// runs the 3-op chain axpby->fma3->scaled_relu (three launches per iteration) on
// compute-stream scratch -- no transfer. Forward-declared CUDA-free so the
// event/stream body stays in fused.cu and no CUDA header reaches this TU. Returns
// warm elapsed ms per rep for the whole chain.
template <typename T>
float time_fused_chain(const T* x, const T* y, const T* z, int64_t n, int reps,
                       int warmups);
} // namespace pg::cuda

namespace pg::dispatch {
// The device-resident routing entry: a, b, c are DEVICE pointers; it forwards
// f32/f64 to the cuBLAS GEMM behind the same PG_HAS_CUDA guard. The binding's
// matmul(Array, Array) path calls THIS rather than cuda::gemm directly, so the
// routing decision (and the f64-never-AUTO-routes rule it documents) stays in the
// dispatch tier. CUDA-free signature (pointers + int64), forward-declared.
template <typename T>
void matmul_device(const T* a, const T* b, T* c, int64_t m, int64_t k, int64_t n);

// The device-resident fused routing entries: x, y, z, out are DEVICE pointers.
// The binding's device fused Array overloads call THESE rather than cuda::fused_*
// directly, so the routing stays in the dispatch tier (symmetric to
// matmul_device). CUDA-free signatures, forward-declared; fused.cu's explicit
// instantiations satisfy them across the TU boundary.
template <typename T>
void fused_axpby_device(const T* x, const T* y, T* out, int64_t n, T a, T b);
template <typename T>
void fused_fma3_device(const T* x, const T* y, const T* z, T* out, int64_t n);
template <typename T>
void fused_scaled_relu_device(const T* x, T* out, int64_t n, T scale);
} // namespace pg::dispatch
#endif

namespace {

// Host operands arrive as zero-copy views and every host result is written into
// a caller-supplied ndarray the wrapper allocated with np.empty (or the user
// passed as out=), so NumPy owns all host result memory and no capsule is
// involved. The wrapper validates first; the extents are re-checked here so a
// direct _core call can never write past the caller's buffer. Every array
// argument is registered noconvert: nanobind's second overload pass would
// otherwise astype() a mismatched array into a temporary, silently computing in
// another dtype or into a copy the caller never sees.
template <typename T>
using host_in = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <typename T>
using host_out = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <typename T>
using host_out1 = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

std::string shape_str(int64_t r, int64_t c) {
    return "(" + std::to_string(r) + ", " + std::to_string(c) + ")";
}

template <typename A>
std::string shape_str(const A& x) {
    return shape_str(static_cast<int64_t>(x.shape(0)),
                     static_cast<int64_t>(x.shape(1)));
}

template <typename T>
void check_out(const char* op, const host_out<T>& out, int64_t m, int64_t n) {
    if (static_cast<int64_t>(out.shape(0)) != m ||
        static_cast<int64_t>(out.shape(1)) != n) {
        throw pg::shape_error(std::string(op) +
                              ": out must be a writeable C-contiguous array of shape " +
                              shape_str(m, n) + ", got shape " + shape_str(out));
    }
}

template <typename T>
void check_same_shape(const char* op, const host_in<T>& x, const host_in<T>& y) {
    if (x.shape(0) != y.shape(0) || x.shape(1) != y.shape(1)) {
        throw pg::shape_error(std::string(op) +
                              ": operands must have the same shape, got " +
                              shape_str(x) + " and " + shape_str(y));
    }
}

// matmul operands may carry any element strides: the kernels read them in
// place, so a transposed, F-ordered, sliced, or reversed operand is never
// copied. nanobind hands over strides in elements and refuses a byte stride
// that is not a whole number of elements; the wrapper copies those, and
// misaligned or byte-swapped operands, before calling.
template <typename T>
using host_strided = nb::ndarray<const T, nb::ndim<2>, nb::device::cpu>;

template <typename T>
void matmul_into(const host_strided<T>& a, const host_strided<T>& b, const host_out<T>& out) {
    const pg::gemm_dims d = pg::check_matmul_dims(
        static_cast<int64_t>(a.shape(0)), static_cast<int64_t>(a.shape(1)),
        static_cast<int64_t>(b.shape(0)), static_cast<int64_t>(b.shape(1)));
    check_out("matmul", out, d.m, d.n);
    // The kernel never touches Python objects, so the GIL drops for the whole
    // compute. This is what lets callers thread around the library.
    nb::gil_scoped_release release;
    pg::dispatch::matmul<T>(a.data(), a.stride(0), a.stride(1), b.data(), b.stride(0), b.stride(1),
                            out.data(), d.m, d.k, d.n);
}

template <typename T>
void transpose_into(const host_in<T>& a, const host_out<T>& out) {
    const int64_t m = static_cast<int64_t>(a.shape(0));
    const int64_t n = static_cast<int64_t>(a.shape(1));
    check_out("transpose", out, n, m);
    nb::gil_scoped_release release;
    pg::dispatch::transpose<T>(a.data(), out.data(), m, n);
}

// Full reduction: returns the scalar in the input dtype. nanobind marshals it
// to a Python float; the wrapper casts it back to the result dtype (f32 round
// trips losslessly through a double-precision Python float). The GIL still
// drops so a large reduction does not block other host threads.
template <typename T>
T sum_all_typed(const host_in<T>& a) {
    const int64_t m = static_cast<int64_t>(a.shape(0));
    const int64_t n = static_cast<int64_t>(a.shape(1));
    nb::gil_scoped_release release;
    return pg::dispatch::sum_all<T>(a.data(), m, n);
}

// axis 0 reduces over rows into a length-n vector, axis 1 over columns into a
// length-m vector.
template <typename T>
void sum_axis_into(const host_in<T>& a, int axis, const host_out1<T>& out) {
    const int64_t m = static_cast<int64_t>(a.shape(0));
    const int64_t n = static_cast<int64_t>(a.shape(1));
    if (axis != 0 && axis != 1) {
        throw pg::shape_error("sum: axis " + std::to_string(axis) +
                              " is out of bounds for 2-dimensional input");
    }
    const int64_t len = (axis == 0) ? n : m;
    if (static_cast<int64_t>(out.shape(0)) != len) {
        throw pg::shape_error("sum: out must be a writeable C-contiguous array of shape (" +
                              std::to_string(len) + ",), got shape (" +
                              std::to_string(out.shape(0)) + ",)");
    }
    nb::gil_scoped_release release;
    pg::dispatch::sum_axis<T>(a.data(), out.data(), m, n, axis);
}

// The fused kernels compute each out[i] from the inputs at the same index i
// only, reading before writing within every vector block and scalar step, so
// out may be exactly an input (same pointer, same extent) for an in-place
// update. Any other overlap would read elements another block already wrote;
// the wrapper rejects it. Scalars cross as double and narrow to T here: a
// float scalar argument would make nanobind's first pass reject any double not
// exactly representable in float32, and its second pass would then convert the
// arrays to float64, returning a float64 result for float32 operands.
template <typename T>
void axpby_into(const host_in<T>& x, const host_in<T>& y, double a, double b,
                const host_out<T>& out) {
    check_same_shape("axpby", x, y);
    const int64_t m = static_cast<int64_t>(x.shape(0));
    const int64_t n = static_cast<int64_t>(x.shape(1));
    check_out("axpby", out, m, n);
    nb::gil_scoped_release release;
    pg::dispatch::fused_axpby<T>(x.data(), y.data(), out.data(), m * n,
                                 static_cast<T>(a), static_cast<T>(b));
}

template <typename T>
void fma3_into(const host_in<T>& x, const host_in<T>& y, const host_in<T>& z,
               const host_out<T>& out) {
    check_same_shape("fma3", x, y);
    check_same_shape("fma3", x, z);
    const int64_t m = static_cast<int64_t>(x.shape(0));
    const int64_t n = static_cast<int64_t>(x.shape(1));
    check_out("fma3", out, m, n);
    nb::gil_scoped_release release;
    pg::dispatch::fused_fma3<T>(x.data(), y.data(), z.data(), out.data(), m * n);
}

template <typename T>
void scaled_relu_into(const host_in<T>& x, double scale, const host_out<T>& out) {
    const int64_t m = static_cast<int64_t>(x.shape(0));
    const int64_t n = static_cast<int64_t>(x.shape(1));
    check_out("scaled_relu", out, m, n);
    nb::gil_scoped_release release;
    pg::dispatch::fused_scaled_relu<T>(x.data(), out.data(), m * n,
                                       static_cast<T>(scale));
}

// Test and measurement hooks for the small and skinny GEMM kernels, which read
// operands in place through any element strides. Without the twin flag the
// AVX2 kernel runs when the CPU has it and the scalar twin otherwise, as
// dispatch chooses, so a forced fallback never reaches AVX2 code.
template <typename T>
using strided_in = nb::ndarray<const T, nb::ndim<2>, nb::device::cpu>;

template <typename T, bool Skinny, bool Twin>
void gemm_hook(const strided_in<T>& a, const strided_in<T>& b, const host_out<T>& out) {
    const pg::gemm_dims d = pg::check_matmul_dims(
        static_cast<int64_t>(a.shape(0)), static_cast<int64_t>(a.shape(1)),
        static_cast<int64_t>(b.shape(0)), static_cast<int64_t>(b.shape(1)));
    check_out("matmul", out, d.m, d.n);
    const auto& f = pg::cpu::detect();
    const bool avx2 = !Twin && f.avx2 && f.fma;
    const auto kernel = Skinny ? (avx2 ? &pg::cpu::gemm_skinny<T> : &pg::cpu::gemm_skinny_naive<T>)
                               : (avx2 ? &pg::cpu::gemm_small<T> : &pg::cpu::gemm_small_naive<T>);
    nb::gil_scoped_release release;
    kernel(a.data(), a.stride(0), a.stride(1), b.data(), b.stride(0), b.stride(1), out.data(), d.m, d.k, d.n);
}

#if defined(PG_HAS_CUDA)
// A device result copied to the host, as a NumPy array over a buffer lent by
// the page-locked pool (or its pageable fallback). The capsule ends the loan when
// NumPy drops the last reference, so NumPy never frees memory it did not
// allocate. fill writes all rows*cols elements and fences before returning.
template <typename T, typename Fill>
nb::ndarray<nb::numpy, T, nb::ndim<2>> host_result(int64_t rows, int64_t cols, Fill fill) {
    const int64_t bytes = pg::checked_bytes(rows, cols, static_cast<int64_t>(sizeof(T)));
    pg::cuda::host_buffer hb{};
    {
        // A first use of a buffer size page-locks it, which takes milliseconds.
        nb::gil_scoped_release release;
        hb = pg::cuda::acquire_host(bytes);
    }
    nb::capsule owner(hb.loan, &pg::cuda::release_host);
    {
        nb::gil_scoped_release release;
        fill(static_cast<T*>(hb.data));
    }
    const size_t shape[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(hb.data, 2, shape, owner);
}

// The staged GEMM for a host pair the wrapper routes to the GPU: one native
// crossing for the upload, the product, the download, and the fence.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> gemm_host_typed(const host_in<T>& a, const host_in<T>& b) {
    const pg::gemm_dims d = pg::check_matmul_dims(
        static_cast<int64_t>(a.shape(0)), static_cast<int64_t>(a.shape(1)),
        static_cast<int64_t>(b.shape(0)), static_cast<int64_t>(b.shape(1)));
    return host_result<T>(d.m, d.n, [&](T* c) {
        pg::cuda::gemm_host<T>(a.data(), b.data(), c, d.m, d.k, d.n);
    });
}

// The same into the caller's out (matmul out=): the download lands in out
// directly, saving a host copy.
template <typename T>
void gemm_host_into(const host_in<T>& a, const host_in<T>& b, const host_out<T>& out) {
    const pg::gemm_dims d = pg::check_matmul_dims(
        static_cast<int64_t>(a.shape(0)), static_cast<int64_t>(a.shape(1)),
        static_cast<int64_t>(b.shape(0)), static_cast<int64_t>(b.shape(1)));
    check_out("matmul", out, d.m, d.n);
    nb::gil_scoped_release release;
    pg::cuda::gemm_host<T>(a.data(), b.data(), out.data(), d.m, d.k, d.n);
}

// pg.Array: the device-resident array. It owns a DeviceBuffer (device pointer +
// shape + dtype, allocated from the context mempool) and frees it in its
// destructor when the Python object is garbage-collected. Single owner: nanobind
// manages this C++ instance's lifetime, so the device buffer is freed exactly
// once, by ~Array, never by a stray capsule deleter -- the use-after-free / leak
// mitigation (a wrong deleter on a device pointer corrupts memory). The buffer is
// moved into the instance at construction (to_device / the device matmul) and
// never shared between two live Arrays.
struct Array {
    pg::cuda::DeviceBuffer buf;

    explicit Array(pg::cuda::DeviceBuffer b) : buf(b) {}

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

    ~Array() {
        // free_device tolerates nullptr; a default/empty buffer is a no-op. This
        // is the ONLY place a device buffer owned by an Array is freed.
        if (buf.ptr != nullptr) {
            // ~Array runs during Python GC, INCLUDING interpreter finalization
            // after the atexit teardown destroyed the context stream. An Array
            // held in a module global (or an inline temporary the GC keeps alive
            // past module finalization) reaches here on a dead stream.
            // free_device is teardown-tolerant for exactly that case: it
            // skips the free once teardown has run and otherwise inspects the
            // return code by hand, never throwing. So this stays a plain call --
            // a throw escaping a destructor during GC would terminate the
            // interpreter, and the no-throw guarantee now lives in free_device,
            // not in an assumption that GC-time frees never fail.
            pg::cuda::free_device(buf.ptr);
            buf.ptr = nullptr;
        }
    }
};

// Map the runtime DType to the matching nb::dtype for the dlpack/property export.
nb::dlpack::dtype array_dtype(pg::cuda::DType dt) {
    return dt == pg::cuda::DType::f64 ? nb::dtype<double>() : nb::dtype<float>();
}

[[noreturn]] void throw_py(PyObject* type, const char* msg) {
    PyErr_SetString(type, msg);
    throw nb::python_error();
}

// Array.__dlpack__ per the array API: a bare "dltensor" capsule over the device
// buffer, whose owner is the Array itself, so the buffer outlives the Array for
// as long as a consumer holds the capsule. Unless the consumer passes stream=-1
// (it orders the work itself), the host waits for the context stream first, so
// whatever stream the consumer reads on sees the finished buffer; waiting covers
// every consumer stream, so no handle is ever interpreted. From there the
// consumer owns the ordering. max_version is accepted and the legacy capsule
// returned, which the protocol allows. numpy cannot consume a CUDA dltensor; the
// host path is from_device.
nb::ndarray<> array_dlpack(nb::handle self, nb::handle stream, nb::handle max_version,
                           nb::handle dl_device, nb::handle copy) {
    (void)max_version;
    if (!copy.is_none() && nb::cast<bool>(copy)) {
        throw_py(PyExc_BufferError, "__dlpack__: copy=True is not supported, pg.Array exports "
                                     "its device buffer without copying");
    }
    if (!dl_device.is_none() &&
        !dl_device.equal(nb::make_tuple(static_cast<int>(nb::device::cuda::value), 0))) {
        throw_py(PyExc_BufferError, "__dlpack__: pg.Array lives on cuda device 0 and cannot be "
                                     "exported to another device");
    }
    bool wait = true;
    if (!stream.is_none()) {
        if (!PyLong_Check(stream.ptr()) || PyBool_Check(stream.ptr())) {
            throw_py(PyExc_TypeError, "__dlpack__: stream must be None or an int");
        }
        const long long s = nb::cast<long long>(stream);
        if (s == 0 || s < -1) {
            // 0 is ambiguous for CUDA (legacy or per-thread default) and is
            // disallowed by the protocol; other negatives mean nothing there.
            throw_py(PyExc_ValueError, "__dlpack__: stream must be -1, 1, 2, or a stream handle");
        }
        wait = s != -1;
    }
    const Array& arr = nb::cast<const Array&>(self);
    if (wait) {
        nb::gil_scoped_release release;
        pg::cuda::sync();
    }
    const size_t shape[2] = {static_cast<size_t>(arr.buf.rows),
                             static_cast<size_t>(arr.buf.cols)};
    return nb::ndarray<>(arr.buf.ptr, 2, shape, /*owner=*/self, /*strides=*/nullptr,
                         array_dtype(arr.buf.dtype), nb::device::cuda::value,
                         /*device_id=*/0);
}

// to_device: copy a host ndarray up to a fresh device buffer and wrap it in an
// pg.Array. Accepts both f32 and f64 (to_device is just memory; the
// f64-never-auto-routes rule is a dispatch concern, not a to_device rejection).
// No host wait: every later use of the buffer is queued behind the copy on the
// same stream, and copy_h2d returns only once the source may change.
template <typename T>
Array* to_device_typed(
    const nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>& a) {

    const int64_t rows = static_cast<int64_t>(a.shape(0));
    const int64_t cols = static_cast<int64_t>(a.shape(1));
    const pg::cuda::DType dt =
        std::is_same_v<T, double> ? pg::cuda::DType::f64 : pg::cuda::DType::f32;
    // Overflow-checked: an unbounded rows*cols*sizeof(T) product cast to
    // size_t for the device alloc could wrap, leaving alloc_device sized for a
    // truncated buffer while copy_h2d copies the full extent.
    const int64_t bytes =
        pg::checked_bytes(rows, cols, static_cast<int64_t>(sizeof(T)));

    pg::cuda::DeviceBuffer buf{nullptr, rows, cols, dt};
    std::unique_ptr<Array> result;
    {
        // a.data() stays valid: nanobind holds the ndarray alive across the call.
        nb::gil_scoped_release release;
        buf.ptr = pg::cuda::alloc_device(bytes);
        // Owned at once, so a copy or launch that throws frees the buffer.
        result = std::make_unique<Array>(buf);
        pg::cuda::copy_h2d(buf.ptr, a.data(), bytes);
    }
    return result.release();
}

// from_device: copy a pg.Array's device buffer back to the host. The copy is
// queued behind every op that produced the buffer, and the host waits for it
// before the array exists, so the caller never sees a half-written result and an
// asynchronous device error raises here (pg::cuda_error -> RuntimeError).
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> from_device_typed(const Array& arr) {
    // rows/cols come straight off the Array's buffer; host_result's checked_bytes
    // throws shape_error (-> ValueError) before an overflowing size can reach an
    // allocation.
    return host_result<T>(arr.buf.rows, arr.buf.cols, [&](T* dst) {
        pg::cuda::copy_d2h(dst, arr.buf.ptr, arr.buf.bytes());
        pg::cuda::sync();
    });
}

// The REQUIRED device-resident matmul: (Array, Array) -> Array, device-in/
// device-out. It validates the operands are the same dtype and shape-conformable,
// allocates the output device buffer via alloc_device, and runs the PURE device
// GEMM (pg::cuda::gemm<T>) directly on the operands' device pointers -- NO host
// staging, no D2H. The result is a fresh pg.Array wrapping the output buffer.
//
// Deliberately minimal here: this is the device-in/device-out primitive the
// wrapper consumes for the device-resident GPU path. The residency rules (the
// mixed-residency TypeError, the residency-out return type, the
// f64-never-auto-routes exclusion) live in the wrapper; this entry must not
// silently transfer either (it never touches host memory). A dtype/shape
// mismatch raises (shape_error -> ValueError via the translator) rather than
// computing garbage. The product is queued, not waited for: the next fence
// (from_device, __dlpack__) waits for it and reports any error it raised.
Array* matmul_device(const Array& a, const Array& b) {
    if (a.buf.dtype != b.buf.dtype) {
        throw pg::shape_error(
            "matmul: device operands must have the same dtype");
    }
    const pg::gemm_dims d = pg::check_matmul_dims(a.buf.rows, a.buf.cols,
                                                    b.buf.rows, b.buf.cols);

    const pg::cuda::DType dt = a.buf.dtype;
    const int64_t elem = pg::cuda::dtype_size(dt);
    // Overflow-checked: the int32 cuBLAS dimension guard fires only
    // INSIDE cuda::gemm, after alloc_device has already been asked for out_bytes,
    // so an overflowing d.m*d.n*elem would reach the device allocator first.
    const int64_t out_bytes = pg::checked_bytes(d.m, d.n, elem);

    pg::cuda::DeviceBuffer out{nullptr, d.m, d.n, dt};
    std::unique_ptr<Array> result;
    {
        // The allocation, the operands' uploads, and the GEMM are all on the
        // context stream, so stream order is the whole ordering contract.
        nb::gil_scoped_release release;
        out.ptr = pg::cuda::alloc_device(out_bytes);
        // Owned at once, so a copy or launch that throws frees the buffer.
        result = std::make_unique<Array>(out);
        // Route through the dispatch tier, not cuda::gemm directly: the
        // device-resident routing decision (and the f64-never-AUTO-routes rule it
        // documents) belongs in src/dispatch. Both operands are already an
        // pg.Array here, so this is the forced device-resident path -- f64 is
        // allowed and computed, the auto-route exclusion is enforced upstream by
        // the wrapper never sending a host f64 array down here.
        if (dt == pg::cuda::DType::f64) {
            pg::dispatch::matmul_device<double>(
                static_cast<const double*>(a.buf.ptr),
                static_cast<const double*>(b.buf.ptr),
                static_cast<double*>(out.ptr), d.m, d.k, d.n);
        } else {
            pg::dispatch::matmul_device<float>(
                static_cast<const float*>(a.buf.ptr),
                static_cast<const float*>(b.buf.ptr),
                static_cast<float*>(out.ptr), d.m, d.k, d.n);
        }
    }
    return result.release();
}

// The cudaEvent-timed device matmul: (Array x, Array y, reps, warmups) -> warm
// ms per rep. Both operands are device-resident, so the timed region is the GEMM
// only -- no H2D/D2H. This is the timing primitive the bench reads instead of a
// wall-clock approximation around an async launch. Same
// dtype + conformable-shape validation as matmul_device (a mismatch is a
// shape_error -> ValueError, never a fake measurement). The GIL drops around the
// whole warmup + timed run; time_matmul itself allocates the single output
// buffer and creates/destroys the events.
float time_matmul_entry(const Array& x, const Array& y, int reps, int warmups) {
    if (x.buf.dtype != y.buf.dtype) {
        throw pg::shape_error(
            "matmul: device operands must have the same dtype");
    }
    const pg::gemm_dims d = pg::check_matmul_dims(x.buf.rows, x.buf.cols,
                                                    y.buf.rows, y.buf.cols);
    const pg::cuda::DType dt = x.buf.dtype;

    float ms = 0.0f;
    {
        nb::gil_scoped_release release;
        if (dt == pg::cuda::DType::f64) {
            ms = pg::cuda::time_matmul<double>(
                static_cast<const double*>(x.buf.ptr),
                static_cast<const double*>(y.buf.ptr), d.m, d.k, d.n, reps,
                warmups);
        } else {
            ms = pg::cuda::time_matmul<float>(
                static_cast<const float*>(x.buf.ptr),
                static_cast<const float*>(y.buf.ptr), d.m, d.k, d.n, reps,
                warmups);
        }
    }
    return ms;
}

// The device-resident fused ops: device-in/device-out, pg.Array -> pg.Array.
// Each mirrors matmul_device's Array overload (validate, checked_bytes,
// alloc_device, GIL release, route via the dispatch tier, no host wait). The
// wrapper sends only all-pg.Array same-residency operands here; a dtype/shape
// mismatch raises shape_error -> ValueError rather than computing garbage. The
// same-shape rule (no broadcast) is the wrapper's guarantee, so these validate
// only dtype + exact rows/cols equality across operands.
Array* fused_axpby_device_entry(const Array& x, const Array& y, double a,
                                double b) {
    if (x.buf.dtype != y.buf.dtype) {
        throw pg::shape_error("axpby: device operands must have the same dtype");
    }
    if (x.buf.rows != y.buf.rows || x.buf.cols != y.buf.cols) {
        throw pg::shape_error(
            "axpby: device operands must have the same shape");
    }
    const pg::cuda::DType dt = x.buf.dtype;
    const int64_t rows = x.buf.rows;
    const int64_t cols = x.buf.cols;
    const int64_t n = rows * cols;
    const int64_t out_bytes =
        pg::checked_bytes(rows, cols, pg::cuda::dtype_size(dt));

    pg::cuda::DeviceBuffer out{nullptr, rows, cols, dt};
    std::unique_ptr<Array> result;
    {
        nb::gil_scoped_release release;
        out.ptr = pg::cuda::alloc_device(out_bytes);
        // Owned at once, so a copy or launch that throws frees the buffer.
        result = std::make_unique<Array>(out);
        if (dt == pg::cuda::DType::f64) {
            pg::dispatch::fused_axpby_device<double>(
                static_cast<const double*>(x.buf.ptr),
                static_cast<const double*>(y.buf.ptr),
                static_cast<double*>(out.ptr), n, a, b);
        } else {
            pg::dispatch::fused_axpby_device<float>(
                static_cast<const float*>(x.buf.ptr),
                static_cast<const float*>(y.buf.ptr),
                static_cast<float*>(out.ptr), n, static_cast<float>(a),
                static_cast<float>(b));
        }
    }
    return result.release();
}

Array* fused_fma3_device_entry(const Array& x, const Array& y, const Array& z) {
    if (x.buf.dtype != y.buf.dtype || x.buf.dtype != z.buf.dtype) {
        throw pg::shape_error("fma3: device operands must have the same dtype");
    }
    if (x.buf.rows != y.buf.rows || x.buf.cols != y.buf.cols ||
        x.buf.rows != z.buf.rows || x.buf.cols != z.buf.cols) {
        throw pg::shape_error("fma3: device operands must have the same shape");
    }
    const pg::cuda::DType dt = x.buf.dtype;
    const int64_t rows = x.buf.rows;
    const int64_t cols = x.buf.cols;
    const int64_t n = rows * cols;
    const int64_t out_bytes =
        pg::checked_bytes(rows, cols, pg::cuda::dtype_size(dt));

    pg::cuda::DeviceBuffer out{nullptr, rows, cols, dt};
    std::unique_ptr<Array> result;
    {
        nb::gil_scoped_release release;
        out.ptr = pg::cuda::alloc_device(out_bytes);
        // Owned at once, so a copy or launch that throws frees the buffer.
        result = std::make_unique<Array>(out);
        if (dt == pg::cuda::DType::f64) {
            pg::dispatch::fused_fma3_device<double>(
                static_cast<const double*>(x.buf.ptr),
                static_cast<const double*>(y.buf.ptr),
                static_cast<const double*>(z.buf.ptr),
                static_cast<double*>(out.ptr), n);
        } else {
            pg::dispatch::fused_fma3_device<float>(
                static_cast<const float*>(x.buf.ptr),
                static_cast<const float*>(y.buf.ptr),
                static_cast<const float*>(z.buf.ptr),
                static_cast<float*>(out.ptr), n);
        }
    }
    return result.release();
}

Array* fused_scaled_relu_device_entry(const Array& x, double scale) {
    const pg::cuda::DType dt = x.buf.dtype;
    const int64_t rows = x.buf.rows;
    const int64_t cols = x.buf.cols;
    const int64_t n = rows * cols;
    const int64_t out_bytes =
        pg::checked_bytes(rows, cols, pg::cuda::dtype_size(dt));

    pg::cuda::DeviceBuffer out{nullptr, rows, cols, dt};
    std::unique_ptr<Array> result;
    {
        nb::gil_scoped_release release;
        out.ptr = pg::cuda::alloc_device(out_bytes);
        // Owned at once, so a copy or launch that throws frees the buffer.
        result = std::make_unique<Array>(out);
        if (dt == pg::cuda::DType::f64) {
            pg::dispatch::fused_scaled_relu_device<double>(
                static_cast<const double*>(x.buf.ptr),
                static_cast<double*>(out.ptr), n, scale);
        } else {
            pg::dispatch::fused_scaled_relu_device<float>(
                static_cast<const float*>(x.buf.ptr),
                static_cast<float*>(out.ptr), n, static_cast<float>(scale));
        }
    }
    return result.release();
}

// The cudaEvent-timed device fused CHAIN timer (the primitive the bench consumes).
// Three device-resident operands x, y, z; the timed region is the 3-op chain
// axpby->fma3->scaled_relu (three launches/iter) on compute-stream scratch -- no
// transfer. Same dtype + same-shape validation as the fused device entries (a
// mismatch is a shape_error -> ValueError, never a fake measurement). The GIL
// drops around the whole warmup + timed run; time_fused_chain itself allocates the
// scratch buffers and creates/destroys the events. This is the exact 3-operand
// signature the bench's _cuda_time_fused(...) call binds to.
float time_fused_chain_entry(const Array& x, const Array& y, const Array& z,
                             int reps, int warmups) {
    if (x.buf.dtype != y.buf.dtype || x.buf.dtype != z.buf.dtype) {
        throw pg::shape_error(
            "time_fused: device operands must have the same dtype");
    }
    if (x.buf.rows != y.buf.rows || x.buf.cols != y.buf.cols ||
        x.buf.rows != z.buf.rows || x.buf.cols != z.buf.cols) {
        throw pg::shape_error(
            "time_fused: device operands must have the same shape");
    }
    const pg::cuda::DType dt = x.buf.dtype;
    const int64_t n = x.buf.rows * x.buf.cols;

    float ms = 0.0f;
    {
        nb::gil_scoped_release release;
        if (dt == pg::cuda::DType::f64) {
            ms = pg::cuda::time_fused_chain<double>(
                static_cast<const double*>(x.buf.ptr),
                static_cast<const double*>(y.buf.ptr),
                static_cast<const double*>(z.buf.ptr), n, reps, warmups);
        } else {
            ms = pg::cuda::time_fused_chain<float>(
                static_cast<const float*>(x.buf.ptr),
                static_cast<const float*>(y.buf.ptr),
                static_cast<const float*>(z.buf.ptr), n, reps, warmups);
        }
    }
    return ms;
}
#endif

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "peregrine native core";
    if (pg::bindings::add_fast_entries(m.ptr()) < 0) {
        throw nb::python_error();
    }

    // Host entries, double before float in every set: the first matching
    // overload wins, and a float64 array must never bind the float32 overload
    // (a silent narrowing). noconvert on every array argument keeps nanobind
    // from converting a mismatched array instead of rejecting it.
    const auto arr = [](const char* name) { return nb::arg(name).noconvert(); };

    m.def("matmul_into", &matmul_into<double>, arr("a"), arr("b"), arr("out"));
    m.def("matmul_into", &matmul_into<float>, arr("a"), arr("b"), arr("out"));

    m.def("transpose_into", &transpose_into<double>, arr("a"), arr("out"));
    m.def("transpose_into", &transpose_into<float>, arr("a"), arr("out"));

    m.def("sum_all", &sum_all_typed<double>, arr("a"));
    m.def("sum_all", &sum_all_typed<float>, arr("a"));

    m.def("sum_axis_into", &sum_axis_into<double>, arr("a"), nb::arg("axis"),
          arr("out"));
    m.def("sum_axis_into", &sum_axis_into<float>, arr("a"), nb::arg("axis"),
          arr("out"));

    m.def("axpby_into", &axpby_into<double>, arr("x"), arr("y"), nb::arg("a"),
          nb::arg("b"), arr("out"));
    m.def("axpby_into", &axpby_into<float>, arr("x"), arr("y"), nb::arg("a"),
          nb::arg("b"), arr("out"));

    m.def("fma3_into", &fma3_into<double>, arr("x"), arr("y"), arr("z"),
          arr("out"));
    m.def("fma3_into", &fma3_into<float>, arr("x"), arr("y"), arr("z"),
          arr("out"));

    m.def("scaled_relu_into", &scaled_relu_into<double>, arr("x"),
          nb::arg("scale"), arr("out"));
    m.def("scaled_relu_into", &scaled_relu_into<float>, arr("x"),
          nb::arg("scale"), arr("out"));

    m.def("cpu_features", [] {
        const auto& f = pg::cpu::detect();
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
    // write would let it size a buffer for one MC and loop for another. Each dtype
    // has its own triple; f32=True addresses the float32 one.
    m.def("_set_gemm_blocking", [](int64_t mc, int64_t kc, int64_t nc, bool f32) {
        pg::cpu::store_blocking(pg::cpu::blocking{mc, kc, nc}, f32);
    }, nb::arg("mc"), nb::arg("kc"), nb::arg("nc"), nb::arg("f32") = false);

    m.def("_get_gemm_blocking", [](bool f32) {
        const pg::cpu::blocking b = pg::cpu::load_blocking(f32);
        nb::dict d;
        d["mc"] = b.mc;
        d["kc"] = b.kc;
        d["nc"] = b.nc;
        return d;
    }, nb::arg("f32") = false);

    // The thread setting behind pg.set_num_threads and pg.get_num_threads, which
    // own validation. Read once here so the setting reflects the environment at
    // import, as OpenMP's own reading of OMP_NUM_THREADS does.
    pg::cpu::get_num_threads();
    m.def("_set_num_threads", [](int n) { pg::cpu::set_num_threads(n); }, nb::arg("n"));
    m.def("_get_num_threads", [] { return pg::cpu::get_num_threads(); });

    // Test hooks: team_size's arithmetic on the calling thread, and the calling
    // thread's MXCSR (returns the previous value) so a test can run a call
    // under FTZ/DAZ and restore the register afterwards.
    m.def("_team_size",
          [](double work, double serial_below, double per_thread, int64_t max_units) {
              return pg::cpu::team_size(work, serial_below, per_thread, max_units);
          },
          nb::arg("work"), nb::arg("serial_below"), nb::arg("per_thread"), nb::arg("max_units"));
    m.def("_set_mxcsr", [](unsigned csr) {
        const unsigned previous = pg::cpu::mxcsr();
        pg::cpu::set_mxcsr(csr);
        return previous;
    }, nb::arg("csr"));

    m.def("_gemm_small", &gemm_hook<double, false, false>, arr("a"), arr("b"), arr("out"));
    m.def("_gemm_small", &gemm_hook<float, false, false>, arr("a"), arr("b"), arr("out"));
    m.def("_gemm_small_naive", &gemm_hook<double, false, true>, arr("a"), arr("b"), arr("out"));
    m.def("_gemm_small_naive", &gemm_hook<float, false, true>, arr("a"), arr("b"), arr("out"));
    m.def("_gemm_skinny", &gemm_hook<double, true, false>, arr("a"), arr("b"), arr("out"));
    m.def("_gemm_skinny", &gemm_hook<float, true, false>, arr("a"), arr("b"), arr("out"));
    m.def("_gemm_skinny_naive", &gemm_hook<double, true, true>, arr("a"), arr("b"), arr("out"));
    m.def("_gemm_skinny_naive", &gemm_hook<float, true, true>, arr("a"), arr("b"), arr("out"));

    // The PRIVATE compile-time build signal. Underscore-private: the public
    // predicate is the wrapper's has_cuda(), which composes this build flag AND a
    // usable-device probe (_cuda_device_probe below) into a build-AND-usable-device
    // answer. Renamed from the old public has_cuda_build: a build flag alone is not
    // the useful predicate (dispatch must not route to a GPU that is absent), so
    // the build-only answer stays private and the runtime chain is the public
    // surface.
    m.def("_has_cuda_build", [] {
#if defined(PG_HAS_CUDA)
        return true;
#else
        return false;
#endif
    });

#if defined(PG_HAS_CUDA)
    // Underscore-private introspection that drives the context singleton to build
    // and reports the bound device. The wrapper's public has_cuda() and the
    // to_device path reach the context through their own entry points; this one
    // exists so the context-init and clean-shutdown tests have a Python-visible
    // trigger. Defined only on the
    // CUDA build: on the OFF build there is no context to introspect and the
    // symbol is absent, matching _has_cuda_build() == False. The GIL is held: this
    // queries device props, it launches no kernel, so there is nothing to overlap.
    m.def("_cuda_device_info", [] {
        const pg::cuda_device_info i = pg::cuda::context_device_info();
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
        const pg::cuda_device_info i = pg::cuda::device_probe();
        nb::dict d;
        d["present"] = i.present;
        d["device_id"] = i.device_id;
        d["cc_major"] = i.cc_major;
        d["cc_minor"] = i.cc_minor;
        d["name"] = i.name;
        d["reason"] = i.reason;
        return d;
    });

    // The staged GEMM behind pg.matmul's GPU route for host operands, with and
    // without out. double before float, like every other overload set: a float64
    // array must never bind the float32 overload (a silent narrowing).
    m.def("_gemm_host", &gemm_host_typed<double>, arr("a"), arr("b"));
    m.def("_gemm_host", &gemm_host_typed<float>, arr("a"), arr("b"));
    m.def("_gemm_host", &gemm_host_into<double>, arr("a"), arr("b"), arr("out"));
    m.def("_gemm_host", &gemm_host_into<float>, arr("a"), arr("b"), arr("out"));

    // Test and measurement hooks. _cuda_sync is the host fence (it raises the
    // name of a pending asynchronous error); _cuda_stream_idle reports whether
    // queued work remains; _pinned_pool_stats reports the page-locked pool;
    // _cuda_inject_fault queues a trapping kernel and _cuda_teardown runs the exit
    // teardown early, both for child processes only (the context is unusable after
    // either).
    m.def("_cuda_sync", [] {
        nb::gil_scoped_release release;
        pg::cuda::sync();
    });
    m.def("_cuda_stream_idle", [] { return pg::cuda::stream_idle(); });
    m.def("_pinned_pool_stats", [] {
        const pg::cuda::pinned_pool_stats s = pg::cuda::pinned_stats();
        nb::dict d;
        d["reserved"] = s.reserved;
        d["loaned"] = s.loaned;
        d["cap"] = s.cap;
        return d;
    });
    m.def("_cuda_inject_fault", [] { pg::cuda::inject_fault(); });
    m.def("_cuda_teardown", [] { pg::cuda::teardown(); });

    // pg.Array: the device-resident array handle. Read-only shape (a tuple) and
    // dtype (the numpy dtype), and __dlpack__/__dlpack_device__ reporting a CUDA
    // device. The class owns its device buffer and frees it on GC (~Array); the
    // dlpack capsule keeps the Array alive via the owner reference rather than
    // freeing through its deleter, so a device-aware consumer can hold the
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
                    self.buf.dtype == pg::cuda::DType::f64 ? "float64"
                                                            : "float32");
            })
        .def("__dlpack__", &array_dlpack, nb::kw_only(), nb::arg("stream") = nb::none(),
             nb::arg("max_version") = nb::none(), nb::arg("dl_device") = nb::none(),
             nb::arg("copy") = nb::none(),
             "Export the device buffer as a DLPack capsule (the array API protocol).\n\n"
             "Waits for queued device work first unless stream is -1. stream 0 raises\n"
             "ValueError; copy=True or a dl_device other than (2, 0) raises BufferError.")
        .def("__dlpack_device__", [](const Array&) {
            // (device_type, device_id): kDLCUDA is 2 in the DLPack device enum.
            // Reported directly so a consumer's __dlpack_device__ negotiation
            // sees CUDA without constructing the full tensor.
            return nb::make_tuple(static_cast<int>(nb::device::cuda::value), 0);
        });

    // to_device(ndarray) -> pg.Array. double before float (a float64 array must
    // never bind the float32 overload). Accepts both dtypes -- it is just memory.
    m.def("to_device", &to_device_typed<double>, nb::arg("a"));
    m.def("to_device", &to_device_typed<float>, nb::arg("a"));

    // from_device(pg.Array) -> ndarray. One entry, dtype-dispatched off the
    // Array (no host operand to overload on): the D2H copy and the host fence
    // happen in from_device_typed before the host ndarray is returned.
    m.def(
        "from_device",
        [](const Array& arr) -> nb::object {
            return arr.buf.dtype == pg::cuda::DType::f64
                       ? nb::cast(from_device_typed<double>(arr))
                       : nb::cast(from_device_typed<float>(arr));
        },
        nb::arg("x"));

    // The device-resident matmul: (Array, Array) -> Array, dtype-dispatched.
    // The device entries carry private names of their own, never a name a host
    // overload shares: nanobind tries overloads in order, and a host overload
    // tried first would import each Array through __dlpack__, a host wait per
    // operand.
    m.def("_matmul_device", &matmul_device, nb::arg("a"), nb::arg("b"));

    // The cudaEvent-timed device-matmul timer (the primitive the bench consumes).
    // Both operands are device-resident pg.Array, so the timed region is the
    // GEMM only -- no transfer. Returns warm elapsed ms per rep from a cudaEvent
    // pair recorded after a sync. Underscore-private: it is a measurement tool the
    // bench reaches through _core, not part of the public surface. Defined only on
    // the CUDA build (it times a device GEMM), absent on the OFF build.
    m.def("_cuda_time_matmul", &time_matmul_entry, nb::arg("x"), nb::arg("y"),
          nb::arg("reps"), nb::arg("warmups"));

    // The device-resident fused ops; host operands go through the *_into entries
    // above. device-in/device-out -> pg.Array. The wrapper routes here only when
    // every operand is a pg.Array. a, b, scale arrive as Python floats (double);
    // the f32 path narrows them, like the CPU axpby/scaled_relu.
    m.def("_axpby_device", &fused_axpby_device_entry, nb::arg("x"), nb::arg("y"),
          nb::arg("a"), nb::arg("b"));
    m.def("_fma3_device", &fused_fma3_device_entry, nb::arg("x"), nb::arg("y"),
          nb::arg("z"));
    m.def("_scaled_relu_device", &fused_scaled_relu_device_entry, nb::arg("x"),
          nb::arg("scale"));

    // The cudaEvent-timed device fused CHAIN timer (the primitive the bench
    // consumes). Three device-resident pg.Array operands, so the timed region is
    // the 3-op chain only -- no transfer. Returns warm elapsed ms per rep for the
    // whole axpby->fma3->scaled_relu chain. Underscore-private: a measurement tool
    // the bench reaches through _core, not public surface. Defined only on the CUDA
    // build, absent on OFF. This 3-operand (x, y, z, reps, warmups) signature is
    // exactly what the bench's _cuda_time_fused(...) call binds to.
    m.def("_cuda_time_fused", &time_fused_chain_entry, nb::arg("x"), nb::arg("y"),
          nb::arg("z"), nb::arg("reps"), nb::arg("warmups"));
#endif

    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const pg::shape_error& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const pg::cuda_error& e) {
            // The arm for the CUDA path. Every PG_CUDA_CHECK / PG_CUBLAS_CHECK
            // failure and the alloc_device OOM pre-flight throw pg::cuda_error
            // carrying the cuda error NAME (e.g. cudaErrorMemoryAllocation) plus
            // file:line. Mapped to RuntimeError here; the distinct user-facing
            // wording (the byte-math OOM message, the driver-too-old message) is
            // composed in the wrapper, keyed off the name in e.what(). cuda_error
            // lives in core/common.hpp and carries no CUDA type, so this arm
            // compiles in every build; on the OFF build nothing throws it, so it is
            // simply never reached. One translator, one arm -- never a second
            // register call.
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
