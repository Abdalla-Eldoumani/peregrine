// The fast entries: METH_FASTCALL functions that take the common host case of
// each public op (exact ndarrays, 2-D, float32 or float64, native, aligned,
// C-contiguous, or F-contiguous too for matmul, non-empty) straight to
// pg::dispatch, and return NotImplemented
// for anything else so the Python wrapper, which owns every error, promotion,
// and warning, handles it. They never raise for an input the wrapper would
// handle: any failure here declines instead.
//
// This is the only translation unit that uses the NumPy C API. Without
// PY_ARRAY_UNIQUE_SYMBOL each TU that includes the NumPy headers gets its own
// API table, and only the TU that runs _import_array() has a filled one. No
// nanobind headers here, so the nanobind boundary stays in module.cpp.

#include "bindings/fast_entry.hpp"

#include <numpy/arrayobject.h>

#include <cstdint>
#include <cstdio>

#include "dispatch/dispatch.hpp"
#include "pg_build_definitions.h"

namespace {

// Releasing and reacquiring the GIL measured about 0.1 us per call, so it is
// dropped only once the kernel takes about 2 us: a 16-cube matmul (4096
// multiply-adds) or a 64x64 transpose or sum (4096 elements) on the kernels of
// 2026-10-18. Faster kernels move these sizes up.
constexpr double kMatmulGilWork = 4096.0;
constexpr int64_t kElementGilWork = 4096;

// Releases the GIL for its scope. Unwinding restores the thread state before
// any catch handler runs, so a handler may touch Python objects again.
class nogil {
public:
    explicit nogil(bool release) : state_(release ? PyEval_SaveThread() : nullptr) {}
    ~nogil() {
        if (state_ != nullptr) {
            PyEval_RestoreThread(state_);
        }
    }
    nogil(const nogil&) = delete;
    nogil& operator=(const nogil&) = delete;

private:
    PyThreadState* state_;
};

// Routing facts for host matmul, pushed by the Python wrapper. Only GIL holders
// read or write them, and every read happens before the GIL is released.
enum forced_backend : int { kAuto = 0, kCpu = 1, kCuda = 2 };
int g_forced = kAuto;
bool g_auto_ready = false;
bool g_device_usable = false;
double g_f32_cpu_flops_bound = 0.0;

PyArray_Descr* g_descr_f32 = nullptr;
PyArray_Descr* g_descr_f64 = nullptr;

template <typename T>
constexpr int npy_type() {
    return sizeof(T) == sizeof(float) ? NPY_FLOAT : NPY_DOUBLE;
}

PyArrayObject* as_array(PyObject* o) {
    return reinterpret_cast<PyArrayObject*>(o);
}

template <typename T>
T* data_of(PyObject* o) {
    return static_cast<T*>(PyArray_DATA(as_array(o)));
}

// The operand predicate. PyArray_CheckExact rather than PyArray_Check, because
// the wrapper converts subclasses (np.matrix, masked arrays) with asarray first.
// A '>f8' array has type NPY_DOUBLE too, so the byte order is checked apart.
// Read-only inputs pass: the wrapper reads them as they are.
bool is_operand(PyObject* o, int type) {
    if (!PyArray_CheckExact(o)) {
        return false;
    }
    PyArrayObject* a = as_array(o);
    return PyArray_NDIM(a) == 2 && PyArray_TYPE(a) == type && PyArray_ISNOTSWAPPED(a) &&
           PyArray_CHKFLAGS(a, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS) &&
           PyArray_DIM(a, 0) > 0 && PyArray_DIM(a, 1) > 0;
}

// matmul also takes F-contiguous operands, transposed views among them: its
// kernels read strides and sum each element in the same order for every
// layout. The reductions and elementwise ops do not, since the Python path
// computes them on a C-ordered copy and another memory order would change the
// bits.
bool is_matmul_operand(PyObject* o, int type) {
    if (!PyArray_CheckExact(o)) {
        return false;
    }
    PyArrayObject* a = as_array(o);
    return PyArray_NDIM(a) == 2 && PyArray_TYPE(a) == type && PyArray_ISNOTSWAPPED(a) &&
           PyArray_CHKFLAGS(a, NPY_ARRAY_ALIGNED) &&
           (PyArray_CHKFLAGS(a, NPY_ARRAY_C_CONTIGUOUS) || PyArray_CHKFLAGS(a, NPY_ARRAY_F_CONTIGUOUS)) &&
           PyArray_DIM(a, 0) > 0 && PyArray_DIM(a, 1) > 0;
}

// In elements. A length-1 dim may carry any byte stride, even one that is not
// a whole number of elements; dispatch replaces it, since it never moves the
// pointer.
int64_t elem_stride(PyObject* o, int dim) {
    PyArrayObject* a = as_array(o);
    return static_cast<int64_t>(PyArray_STRIDE(a, dim) / PyArray_ITEMSIZE(a));
}

bool same_dims(PyObject* x, PyObject* y) {
    return PyArray_DIM(as_array(x), 0) == PyArray_DIM(as_array(y), 0) &&
           PyArray_DIM(as_array(x), 1) == PyArray_DIM(as_array(y), 1);
}

// The out= predicate: the result's exact shape and dtype, writeable, and the
// same layout rules as an operand.
bool is_out(PyObject* o, int type, npy_intp d0, npy_intp d1) {
    if (!PyArray_CheckExact(o)) {
        return false;
    }
    PyArrayObject* a = as_array(o);
    return PyArray_NDIM(a) == 2 && PyArray_TYPE(a) == type && PyArray_ISNOTSWAPPED(a) &&
           PyArray_CHKFLAGS(a, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE) &&
           PyArray_DIM(a, 0) == d0 && PyArray_DIM(a, 1) == d1;
}

// Address-range overlap of two C- or F-contiguous 2-D arrays. Such an array
// occupies exactly d0*d1*itemsize bytes from its data pointer whatever strides
// its length-1 dims carry, so the range is exact, matching np.may_share_memory.
bool overlaps(PyObject* x, PyObject* y) {
    auto range = [](PyObject* o, std::uintptr_t& lo, std::uintptr_t& hi) {
        PyArrayObject* a = as_array(o);
        lo = reinterpret_cast<std::uintptr_t>(PyArray_DATA(a));
        hi = lo + static_cast<std::uintptr_t>(PyArray_DIM(a, 0)) *
                      static_cast<std::uintptr_t>(PyArray_DIM(a, 1)) *
                      static_cast<std::uintptr_t>(PyArray_ITEMSIZE(a));
    };
    std::uintptr_t xlo, xhi, ylo, yhi;
    range(x, xlo, xhi);
    range(y, ylo, yhi);
    return xlo < yhi && ylo < xhi;
}

// For the elementwise ops an input may be out itself: every array here already
// has the same shape, dtype, and C layout, so the same data pointer is the
// exact alias, which an in-place update needs. Any other overlap declines.
bool alias_ok(PyObject* in, PyObject* out) {
    return PyArray_DATA(as_array(in)) == PyArray_DATA(as_array(out)) || !overlaps(in, out);
}

// Exact float, or exact int (bool excluded) that fits a double. The caller
// narrows with (T), the same double rounding the wrapper's float() gives.
bool scalar_arg(PyObject* o, double& v) {
    if (PyFloat_CheckExact(o)) {
        v = PyFloat_AsDouble(o);
        return true;
    }
    if (PyLong_CheckExact(o)) {
        v = PyLong_AsDouble(o);
        if (v == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        return true;
    }
    return false;
}

// None is -1; otherwise exactly the int 0 or 1.
bool axis_arg(PyObject* o, int& axis) {
    if (o == Py_None) {
        axis = -1;
        return true;
    }
    if (!PyLong_CheckExact(o)) {
        return false;
    }
    int overflow = 0;
    const long v = PyLong_AsLongAndOverflow(o, &overflow);
    if (v == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return false;
    }
    if (overflow != 0 || (v != 0 && v != 1)) {
        return false;
    }
    axis = static_cast<int>(v);
    return true;
}

// A new reference to the array the result is written into: out itself, or a
// fresh array NumPy owns. nullptr (with no error set) declines.
PyObject* result_array(PyObject* out, int type, int nd, npy_intp* dims) {
    if (out != Py_None) {
        Py_INCREF(out);
        return out;
    }
    PyObject* r = PyArray_SimpleNew(nd, dims, type);
    if (r == nullptr) {
        PyErr_Clear();
    }
    return r;
}

PyObject* not_implemented() {
    Py_RETURN_NOTIMPLEMENTED;
}

// Runs a kernel on a result array, dropping the GIL above the work threshold.
// Any C++ exception declines; the result reference is released either way.
template <typename F>
PyObject* run(PyObject* r, bool release, F&& kernel) {
    try {
        nogil g(release);
        kernel();
    } catch (...) {
        Py_DECREF(r);
        PyErr_Clear();
        return not_implemented();
    }
    return r;
}

// The host matmul route. Python decides anything not settled here, which is
// why every unclear case declines. float64 never goes to the GPU.
bool route_to_cpu(int type, npy_intp m, npy_intp k, npy_intp n) {
    if (g_forced == kCpu) {
        return true;
    }
    if (g_forced == kCuda || !g_auto_ready) {
        return false;
    }
    if (type == NPY_DOUBLE || !g_device_usable) {
        return true;
    }
    // In double: 2*m*k*n overflows int64 for large virtual shapes.
    return 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n) <=
           g_f32_cpu_flops_bound;
}

template <typename T>
PyObject* matmul_typed(PyObject* const* args) {
    constexpr int type = npy_type<T>();
    PyObject* a = args[0];
    PyObject* b = args[1];
    PyObject* out = args[2];
    if (!is_matmul_operand(a, type) || !is_matmul_operand(b, type)) {
        return not_implemented();
    }
    const npy_intp m = PyArray_DIM(as_array(a), 0);
    const npy_intp k = PyArray_DIM(as_array(a), 1);
    const npy_intp n = PyArray_DIM(as_array(b), 1);
    if (PyArray_DIM(as_array(b), 0) != k || !route_to_cpu(type, m, k, n)) {
        return not_implemented();
    }
    // A product cannot be computed in place, so out must be disjoint from both.
    if (out != Py_None && (!is_out(out, type, m, n) || overlaps(out, a) || overlaps(out, b))) {
        return not_implemented();
    }
    npy_intp dims[2] = {m, n};
    PyObject* r = result_array(out, type, 2, dims);
    if (r == nullptr) {
        return not_implemented();
    }
    const double work = static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);
    return run(r, work >= kMatmulGilWork, [&] {
        pg::dispatch::matmul<T>(data_of<const T>(a), elem_stride(a, 0), elem_stride(a, 1), data_of<const T>(b),
                                elem_stride(b, 0), elem_stride(b, 1), data_of<T>(r), m, k, n);
    });
}

template <typename T>
PyObject* transpose_typed(PyObject* a) {
    if (!is_operand(a, npy_type<T>())) {
        return not_implemented();
    }
    const npy_intp m = PyArray_DIM(as_array(a), 0);
    const npy_intp n = PyArray_DIM(as_array(a), 1);
    npy_intp dims[2] = {n, m};
    PyObject* r = result_array(Py_None, npy_type<T>(), 2, dims);
    if (r == nullptr) {
        return not_implemented();
    }
    return run(r, m * n >= kElementGilWork,
               [&] { pg::dispatch::transpose<T>(data_of<const T>(a), data_of<T>(r), m, n); });
}

// sum and mean share one body. mean divides in T by (T)count, exactly what the
// wrapper's total / count does under NumPy's scalar rules: dividing in double
// and rounding afterwards gives different float32 bits once count exceeds 2**24.
template <typename T>
PyObject* reduce_typed(PyObject* a, int axis, bool mean) {
    constexpr int type = npy_type<T>();
    if (!is_operand(a, type)) {
        return not_implemented();
    }
    const npy_intp m = PyArray_DIM(as_array(a), 0);
    const npy_intp n = PyArray_DIM(as_array(a), 1);
    const bool release = m * n >= kElementGilWork;
    if (axis < 0) {
        T s = 0;
        try {
            nogil g(release);
            s = pg::dispatch::sum_all<T>(data_of<const T>(a), m, n);
        } catch (...) {
            return not_implemented();
        }
        if (mean) {
            s = s / static_cast<T>(m * n);
        }
        PyObject* r = PyArray_Scalar(&s, type == NPY_FLOAT ? g_descr_f32 : g_descr_f64, nullptr);
        if (r == nullptr) {
            PyErr_Clear();
            return not_implemented();
        }
        return r;
    }
    npy_intp len = axis == 0 ? n : m;
    PyObject* r = result_array(Py_None, type, 1, &len);
    if (r == nullptr) {
        return not_implemented();
    }
    return run(r, release, [&] {
        if (mean) {
            pg::dispatch::mean_axis<T>(data_of<const T>(a), data_of<T>(r), m, n, axis);
        } else {
            pg::dispatch::sum_axis<T>(data_of<const T>(a), data_of<T>(r), m, n, axis);
        }
    });
}

template <typename T>
PyObject* axpby_typed(PyObject* const* args) {
    constexpr int type = npy_type<T>();
    PyObject* x = args[0];
    PyObject* y = args[1];
    PyObject* out = args[4];
    double a = 0.0;
    double b = 0.0;
    if (!is_operand(x, type) || !is_operand(y, type) || !same_dims(x, y) ||
        !scalar_arg(args[2], a) || !scalar_arg(args[3], b)) {
        return not_implemented();
    }
    npy_intp dims[2] = {PyArray_DIM(as_array(x), 0), PyArray_DIM(as_array(x), 1)};
    if (out != Py_None &&
        (!is_out(out, type, dims[0], dims[1]) || !alias_ok(x, out) || !alias_ok(y, out))) {
        return not_implemented();
    }
    PyObject* r = result_array(out, type, 2, dims);
    if (r == nullptr) {
        return not_implemented();
    }
    const int64_t count = dims[0] * dims[1];
    return run(r, count >= kElementGilWork, [&] {
        pg::dispatch::fused_axpby<T>(data_of<const T>(x), data_of<const T>(y), data_of<T>(r), count,
                                     static_cast<T>(a), static_cast<T>(b));
    });
}

template <typename T>
PyObject* fma3_typed(PyObject* const* args) {
    constexpr int type = npy_type<T>();
    PyObject* x = args[0];
    PyObject* y = args[1];
    PyObject* z = args[2];
    PyObject* out = args[3];
    if (!is_operand(x, type) || !is_operand(y, type) || !is_operand(z, type) ||
        !same_dims(x, y) || !same_dims(x, z)) {
        return not_implemented();
    }
    npy_intp dims[2] = {PyArray_DIM(as_array(x), 0), PyArray_DIM(as_array(x), 1)};
    if (out != Py_None && (!is_out(out, type, dims[0], dims[1]) || !alias_ok(x, out) ||
                           !alias_ok(y, out) || !alias_ok(z, out))) {
        return not_implemented();
    }
    PyObject* r = result_array(out, type, 2, dims);
    if (r == nullptr) {
        return not_implemented();
    }
    const int64_t count = dims[0] * dims[1];
    return run(r, count >= kElementGilWork, [&] {
        pg::dispatch::fused_fma3<T>(data_of<const T>(x), data_of<const T>(y), data_of<const T>(z),
                                    data_of<T>(r), count);
    });
}

template <typename T>
PyObject* scaled_relu_typed(PyObject* const* args) {
    constexpr int type = npy_type<T>();
    PyObject* x = args[0];
    PyObject* out = args[2];
    double scale = 0.0;
    if (!is_operand(x, type) || !scalar_arg(args[1], scale)) {
        return not_implemented();
    }
    npy_intp dims[2] = {PyArray_DIM(as_array(x), 0), PyArray_DIM(as_array(x), 1)};
    if (out != Py_None && (!is_out(out, type, dims[0], dims[1]) || !alias_ok(x, out))) {
        return not_implemented();
    }
    PyObject* r = result_array(out, type, 2, dims);
    if (r == nullptr) {
        return not_implemented();
    }
    const int64_t count = dims[0] * dims[1];
    return run(r, count >= kElementGilWork, [&] {
        pg::dispatch::fused_scaled_relu<T>(data_of<const T>(x), data_of<T>(r), count,
                                           static_cast<T>(scale));
    });
}

// The dtype of the first array operand picks the typed body; anything that is
// not an exact float32 or float64 ndarray declines before any other check.
int first_type(PyObject* o) {
    if (!PyArray_CheckExact(o)) {
        return -1;
    }
    return PyArray_TYPE(as_array(o));
}

PyObject* fast_matmul(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        return not_implemented();
    }
    switch (first_type(args[0])) {
    case NPY_FLOAT:
        return matmul_typed<float>(args);
    case NPY_DOUBLE:
        return matmul_typed<double>(args);
    default:
        return not_implemented();
    }
}

PyObject* fast_transpose(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        return not_implemented();
    }
    switch (first_type(args[0])) {
    case NPY_FLOAT:
        return transpose_typed<float>(args[0]);
    case NPY_DOUBLE:
        return transpose_typed<double>(args[0]);
    default:
        return not_implemented();
    }
}

PyObject* reduce_entry(PyObject* const* args, Py_ssize_t nargs, bool mean) {
    int axis = 0;
    if (nargs != 2 || !axis_arg(args[1], axis)) {
        return not_implemented();
    }
    switch (first_type(args[0])) {
    case NPY_FLOAT:
        return reduce_typed<float>(args[0], axis, mean);
    case NPY_DOUBLE:
        return reduce_typed<double>(args[0], axis, mean);
    default:
        return not_implemented();
    }
}

PyObject* fast_sum(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    return reduce_entry(args, nargs, false);
}

PyObject* fast_mean(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    return reduce_entry(args, nargs, true);
}

PyObject* fast_axpby(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        return not_implemented();
    }
    switch (first_type(args[0])) {
    case NPY_FLOAT:
        return axpby_typed<float>(args);
    case NPY_DOUBLE:
        return axpby_typed<double>(args);
    default:
        return not_implemented();
    }
}

PyObject* fast_fma3(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 4) {
        return not_implemented();
    }
    switch (first_type(args[0])) {
    case NPY_FLOAT:
        return fma3_typed<float>(args);
    case NPY_DOUBLE:
        return fma3_typed<double>(args);
    default:
        return not_implemented();
    }
}

PyObject* fast_scaled_relu(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        return not_implemented();
    }
    switch (first_type(args[0])) {
    case NPY_FLOAT:
        return scaled_relu_typed<float>(args);
    case NPY_DOUBLE:
        return scaled_relu_typed<double>(args);
    default:
        return not_implemented();
    }
}

// _set_routing_facts(forced, auto_ready, device_usable, f32_cpu_flops_bound).
// Private and called only by the wrapper, so bad arguments raise.
PyObject* set_routing_facts(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "_set_routing_facts takes 4 positional arguments");
        return nullptr;
    }
    int forced = -1;
    if (PyUnicode_Check(args[0])) {
        if (PyUnicode_CompareWithASCIIString(args[0], "auto") == 0) {
            forced = kAuto;
        } else if (PyUnicode_CompareWithASCIIString(args[0], "cpu") == 0) {
            forced = kCpu;
        } else if (PyUnicode_CompareWithASCIIString(args[0], "cuda") == 0) {
            forced = kCuda;
        }
    }
    if (forced < 0) {
        PyErr_SetString(PyExc_ValueError, "_set_routing_facts: forced must be 'auto', 'cpu', or 'cuda'");
        return nullptr;
    }
    const int ready = PyObject_IsTrue(args[1]);
    const int usable = PyObject_IsTrue(args[2]);
    const double bound = PyFloat_AsDouble(args[3]);
    if (ready < 0 || usable < 0 || (bound == -1.0 && PyErr_Occurred())) {
        return nullptr;
    }
    g_forced = forced;
    g_auto_ready = ready != 0;
    g_device_usable = usable != 0;
    g_f32_cpu_flops_bound = bound;
    Py_RETURN_NONE;
}

PyObject* clear_routing_facts(PyObject*, PyObject*) {
    g_forced = kAuto;
    g_auto_ready = false;
    g_device_usable = false;
    g_f32_cpu_flops_bound = 0.0;
    Py_RETURN_NONE;
}

#ifndef PG_BUILD_COMPILER
#define PG_BUILD_COMPILER "unknown"
#endif
#ifndef PG_BUILD_CXX_FLAGS
#define PG_BUILD_CXX_FLAGS "unknown"
#endif

// Adds key -> value to d and drops the value reference; false on any failure.
bool put(PyObject* d, const char* key, PyObject* value) {
    if (value == nullptr) {
        return false;
    }
    const int rc = PyDict_SetItemString(d, key, value);
    Py_DECREF(value);
    return rc == 0;
}

PyObject* build_info(PyObject*, PyObject*) {
    PyObject* d = PyDict_New();
    if (d == nullptr) {
        return nullptr;
    }
#if defined(PG_HAS_CUDA)
    const long cuda = 1;
#else
    const long cuda = 0;
#endif
#if defined(Py_LIMITED_API)
    char limited[16];
    std::snprintf(limited, sizeof(limited), "0x%08lX", static_cast<unsigned long>(Py_LIMITED_API));
    PyObject* limited_api = PyUnicode_FromString(limited);
#else
    PyObject* limited_api = Py_NewRef(Py_None);
#endif
#if defined(_OPENMP)
    PyObject* openmp = PyLong_FromLong(static_cast<long>(_OPENMP));
#else
    PyObject* openmp = Py_NewRef(Py_None);
#endif
    const bool ok = put(d, "compiler", PyUnicode_FromString(PG_BUILD_COMPILER)) &&
                    put(d, "flags", PyUnicode_FromString(PG_BUILD_CXX_FLAGS)) &&
                    put(d, "definitions", PyUnicode_FromString(PG_BUILD_DEFINITIONS)) &&
                    put(d, "cuda", PyBool_FromLong(cuda)) && put(d, "limited_api", limited_api) &&
                    put(d, "openmp", openmp);
    if (!ok) {
        Py_DECREF(d);
        return nullptr;
    }
    return d;
}

// Function pointers go through void(*)(void) because PyCFunction does not
// match the METH_FASTCALL signature.
#define PG_FASTCALL(fn) reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(fn))

PyMethodDef g_methods[] = {
    {"_fast_matmul", PG_FASTCALL(fast_matmul), METH_FASTCALL, nullptr},
    {"_fast_transpose", PG_FASTCALL(fast_transpose), METH_FASTCALL, nullptr},
    {"_fast_sum", PG_FASTCALL(fast_sum), METH_FASTCALL, nullptr},
    {"_fast_mean", PG_FASTCALL(fast_mean), METH_FASTCALL, nullptr},
    {"_fast_axpby", PG_FASTCALL(fast_axpby), METH_FASTCALL, nullptr},
    {"_fast_fma3", PG_FASTCALL(fast_fma3), METH_FASTCALL, nullptr},
    {"_fast_scaled_relu", PG_FASTCALL(fast_scaled_relu), METH_FASTCALL, nullptr},
    {"_set_routing_facts", PG_FASTCALL(set_routing_facts), METH_FASTCALL, nullptr},
    {"_clear_routing_facts", clear_routing_facts, METH_NOARGS, nullptr},
    {"_build_info", build_info, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

namespace pg::bindings {

int add_fast_entries(PyObject* module) {
    // _import_array, never the import_array() macro: the macro prints the
    // error and returns from the caller, and module init must raise instead.
    if (_import_array() < 0) {
        return -1;
    }
    g_descr_f32 = PyArray_DescrFromType(NPY_FLOAT);
    g_descr_f64 = PyArray_DescrFromType(NPY_DOUBLE);
    if (g_descr_f32 == nullptr || g_descr_f64 == nullptr) {
        return -1;
    }
    return PyModule_AddFunctions(module, g_methods);
}

} // namespace pg::bindings
