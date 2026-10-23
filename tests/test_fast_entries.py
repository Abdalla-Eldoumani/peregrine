"""The native fast entries against the Python path they sit in front of.

Every public op first offers its arguments to a native fast entry, which takes
the common host case and returns NotImplemented for anything else. The Python
path owns every error, promotion, and warning, so the two must be
indistinguishable from outside. The differential property builds the same
random inputs twice, runs one copy through the native entry and one with the
entry stubbed out, and demands bitwise-identical results, identical result
types (a NumPy scalar never turns into a Python float), identical exception
types and messages, identical warnings (category, message, file, line),
identical out= aliasing, and identical input bytes afterwards. A third copy
pins which inputs the native entry takes at all, against a Python restatement
of its acceptance rule.
"""

import enum
import functools
import importlib
import inspect
import itertools
import math
import os
import struct
import subprocess
import sys
import warnings
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import peregrine as pg
from peregrine import policy
from test_calibration import (
    _smoothed_real_calibration,
    _staged_seconds,
    _stub_gpu_machine,
    _write_cache_file,
    real_calibration,
    synthetic_calibration,
)

calibrate = importlib.import_module("peregrine.calibrate")

# name -> (array operands, scalar keywords, takes axis)
OPS = {
    "matmul": (2, (), False),
    "transpose": (1, (), False),
    "sum": (1, (), True),
    "mean": (1, (), True),
    "axpby": (2, ("a", "b"), False),
    "fma3": (3, (), False),
    "scaled_relu": (1, ("scale",), False),
}
OUT_OPS = ("matmul", "axpby", "fma3", "scaled_relu")
NATIVE = {op: getattr(pg._core, "_fast_" + op) for op in OPS}

F32 = np.dtype(np.float32).num
F64 = np.dtype(np.float64).num


class _Sub(np.ndarray):
    pass


class _Axis(enum.IntEnum):
    ONE = 1


NAN_PAYLOAD = struct.unpack("<d", struct.pack("<Q", 0x7FF8DEADBEEF0001))[0]
SPECIALS = [np.nan, NAN_PAYLOAD, np.inf, -np.inf, 0.0, -0.0, 5e-324, 1e-40, 1e30]
OTHER_DTYPES = [
    ">f4", ">f8", "int8", "int32", "int64", "uint8", "bool", "float16", "complex64",
    "object", "datetime64[s]",
]
LAYOUTS = ["C", "C", "C", "F", "T", "rows", "cols", "rev", "unaligned", "readonly",
           "broadcast", "odd1"]
CONTAINERS = ["nd"] * 6 + ["sub", "matrix", "masked", "list"]
SCALARS = [
    1.0, 2.0, -0.5, 0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1e300, 1e-320,
    0.1, 2**53 + 1, 2**63, 2**1100, -3, 0, True, np.float64(0.1), np.float32(0.1),
    np.int64(3), "2", None, 1 + 2j, Fraction(1, 3),
]
AXES = [None, 0, 1, -1, 2, True, np.bool_(True), np.int64(1), 1.0, "0", _Axis.ONE]
OUT_KINDS = ["none"] * 4 + [
    "fresh", "fresh", "wrong_dtype", "wrong_shape", "F", "readonly", "swapped",
    "subclass", "list", "input0", "view0", "shifted", "input1", "twice",
]
FAST_SCALARS = [1.0, 2.0, -0.5, -0.0, float("nan"), float("inf"), 1e300, 0.1, 2**63, -3]
DIM = st.one_of(st.integers(1, 6), st.sampled_from([0, 1, 3, 7, 16, 33]))


# --- building inputs ------------------------------------------------------------


def _values(shape, seed, specials):
    v = np.asarray(np.random.default_rng(seed).standard_normal(shape) * 4.0)
    flat = v.reshape(-1)
    for j, idx in enumerate(specials):
        if flat.size:
            flat[(seed + 7 * j) % flat.size] = SPECIALS[idx]
    return v


def _cast(v, dtype):
    dt = np.dtype(dtype)
    if dt.kind == "M":
        return np.arange(v.size).reshape(v.shape).astype(dt)
    if dt.kind == "b":
        return v > 0
    if dt.kind in "iu":
        lo = 0.0 if dt.kind == "u" else -100.0
        clean = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return np.rint(np.clip(clean, lo, 100.0)).astype(dt)
    return v.astype(dt)


def _layout(arr, layout):
    if arr.ndim != 2:
        return arr
    d0, d1 = arr.shape
    if layout == "F":
        return np.asfortranarray(arr)
    if layout == "T":
        return np.ascontiguousarray(arr.T).T
    if layout == "rows":
        big = np.empty((2 * d0, d1), arr.dtype)
        big[::2] = arr
        big[1::2] = arr
        return big[::2]
    if layout == "cols":
        big = np.empty((d0, 2 * d1), arr.dtype)
        big[:, ::2] = arr
        big[:, 1::2] = arr
        return big[:, ::2]
    if layout == "rev":
        return np.ascontiguousarray(arr[::-1])[::-1]
    if layout == "unaligned" and not arr.dtype.hasobject:
        buf = np.zeros(arr.nbytes + 1, np.uint8)
        u = np.frombuffer(buf, arr.dtype, count=arr.size, offset=1).reshape(arr.shape)
        u[...] = arr
        return u
    if layout == "readonly":
        c = arr.copy()
        c.flags.writeable = False
        return c
    if layout == "broadcast" and d0:
        return np.broadcast_to(arr[:1], arr.shape)
    if layout == "odd1" and not arr.dtype.hasobject:
        c = np.ascontiguousarray(arr)
        strides = [1001 * c.itemsize if c.shape[i] == 1 else c.strides[i] for i in (0, 1)]
        return np.lib.stride_tricks.as_strided(c, c.shape, strides)
    return np.ascontiguousarray(arr)


def _container(arr, container):
    if container == "sub":
        return arr.view(_Sub)
    if container == "matrix" and arr.ndim == 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return arr.view(np.matrix)
    if container == "masked":
        return np.ma.masked_array(arr)
    if container == "list":
        return arr.tolist()
    return arr


def _operand(o):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = _cast(_values(o["shape"], o["seed"], o["specials"]), o["dtype"])
        return _container(_layout(arr, o["layout"]), o["container"])


def _result_dtype(arrays):
    try:
        r = np.result_type(*(np.asarray(x).dtype for x in arrays))
    except TypeError:
        return np.dtype(np.float64)
    if r.kind == "f" and r.itemsize in (4, 8):
        return r.newbyteorder("=")
    return np.dtype(np.float64)


def _result_shape(op, arrays):
    s = [np.shape(x) for x in arrays]
    if not all(len(t) == 2 for t in s):
        return (1, 1)
    return (s[0][0], s[1][1]) if op == "matmul" else s[0]


def _out(spec, arrays):
    kind = spec["out"]
    if kind in ("absent", "none"):
        return None
    shape = _result_shape(spec["op"], arrays)
    dt = _result_dtype(arrays)
    # A filled out, never np.empty: its bytes are compared after the call.
    if kind == "fresh":
        return np.full(shape, -7.0, dt)
    if kind == "wrong_dtype":
        return np.full(shape, -7.0, np.float32 if dt == np.float64 else np.float64)
    if kind == "wrong_shape":
        return np.full((shape[0] + 1, shape[1]), -7.0, dt)
    if kind == "F":
        return np.asfortranarray(np.full(shape, -7.0, dt))
    if kind == "readonly":
        o = np.full(shape, -7.0, dt)
        o.flags.writeable = False
        return o
    if kind == "swapped":
        return np.full(shape, -7.0, dt.newbyteorder(">"))
    if kind == "subclass":
        return np.full(shape, -7.0, dt).view(_Sub)
    if kind == "list":
        return np.full(shape, -7.0, dt).tolist()
    if kind == "input0":
        return arrays[0]
    if kind == "view0":
        return arrays[0][...] if isinstance(arrays[0], np.ndarray) else arrays[0]
    if kind == "input1":
        return arrays[min(1, len(arrays) - 1)]
    if kind == "twice":
        if len(arrays) > 1:
            arrays[1] = arrays[0]
        return arrays[0]
    # shifted: out starts one element into the buffer that holds input 0
    a0 = arrays[0]
    if type(a0) is not np.ndarray or a0.ndim != 2 or a0.dtype.hasobject:
        return np.full(shape, -7.0, dt)
    size = shape[0] * shape[1]
    buf = np.zeros(max(a0.nbytes, size * dt.itemsize) + dt.itemsize, np.uint8)
    moved = np.frombuffer(buf, a0.dtype, count=a0.size).reshape(a0.shape)
    moved[...] = a0
    arrays[0] = moved
    return np.frombuffer(buf, dt, count=size, offset=dt.itemsize).reshape(shape)


def _build(spec):
    """Fresh, independent inputs for one run: (arrays, kwargs, out)."""
    arrays = [_operand(o) for o in spec["operands"]]
    return arrays, dict(spec["kwargs"]), _out(spec, arrays)


@st.composite
def _operand_spec(draw, shape, main, clean):
    return dict(
        shape=shape,
        dtype=main if clean else draw(st.sampled_from([main] * 8 + OTHER_DTYPES)),
        layout="C" if clean else draw(st.sampled_from(LAYOUTS)),
        container="nd" if clean else draw(st.sampled_from(CONTAINERS)),
        seed=draw(st.integers(0, 2**32 - 1)),
        specials=draw(st.lists(st.integers(0, len(SPECIALS) - 1), max_size=3)),
    )


@st.composite
def op_spec(draw, op):
    n_arrays, scalar_names, takes_axis = OPS[op]
    main = draw(st.sampled_from(["<f4", "<f8"]))
    # Half the specs are the common case the native entry exists for.
    clean = draw(st.booleans())
    d0, d1 = draw(DIM), draw(DIM)
    if op == "matmul":
        k2 = d1 if draw(st.integers(0, 9)) else d1 + 1
        shapes = [(d0, d1), (k2, draw(DIM))]
    else:
        shapes = [(d0, d1)] * n_arrays
        if n_arrays > 1 and not draw(st.integers(0, 9)):
            shapes[draw(st.integers(1, n_arrays - 1))] = (d0 + 1, d1)
    if not draw(st.integers(0, 14)):
        shapes[draw(st.integers(0, n_arrays - 1))] = draw(
            st.sampled_from([(), (d0,), (d0, d1, 2)])
        )
    kwargs = {}
    for name in scalar_names:
        if draw(st.booleans()):
            kwargs[name] = draw(st.sampled_from(FAST_SCALARS if clean else SCALARS))
    if takes_axis and draw(st.integers(0, 4)):
        kwargs["axis"] = draw(st.sampled_from(AXES))
    return dict(
        op=op,
        operands=[draw(_operand_spec(s, main, clean)) for s in shapes],
        kwargs=kwargs,
        out=draw(st.sampled_from(OUT_KINDS)) if op in OUT_OPS else "absent",
    )


# --- running both paths ---------------------------------------------------------


def _call(op, arrays, kwargs, out, pass_out, entry):
    # One call site for both paths, so a warning attributed to the caller
    # carries the same file and line whichever path raised it.
    if pass_out:
        kwargs = dict(kwargs, out=out)
    name = "_fast_" + op
    saved = getattr(pg, name)
    setattr(pg, name, entry)
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            try:
                result, error = getattr(pg, op)(*arrays, **kwargs), None
            except Exception as exc:  # the comparison is the point
                result, error = None, exc
    finally:
        setattr(pg, name, saved)
    return result, error, [(w.category, str(w.message), w.filename, w.lineno) for w in record]


def _python_path(op):
    def run(*args, **kwargs):
        saved = getattr(pg, "_fast_" + op)
        setattr(pg, "_fast_" + op, pg._no_fast_entry)
        try:
            return getattr(pg, op)(*args, **kwargs)
        finally:
            setattr(pg, "_fast_" + op, saved)

    return run


@functools.cache
def _supports_out(op):
    return "out" in inspect.signature(getattr(pg, op)).parameters


def _snapshot(x):
    if isinstance(x, np.ndarray):
        a = np.asarray(x)
        if a.dtype.hasobject:
            return repr(a.tolist())
        return a.dtype.str, a.shape, a.tobytes()
    return repr(x)


def _assert_same(fast, slow, fast_inputs, slow_inputs):
    (rf, ef, wf), out_f = fast
    (rs, es, ws), out_s = slow
    assert (ef is None) == (es is None), (ef, es)
    if ef is not None:
        assert type(ef) is type(es)
        assert str(ef) == str(es)
    else:
        assert type(rf) is type(rs)
        if isinstance(rf, np.ndarray):
            assert rf.dtype.str == rs.dtype.str
            assert rf.shape == rs.shape and rf.strides == rs.strides
            for flag in ("c_contiguous", "f_contiguous", "writeable", "aligned"):
                assert getattr(rf.flags, flag) == getattr(rs.flags, flag), flag
            assert rf.flags.owndata == rs.flags.owndata
            assert (rf.base is None) == (rs.base is None)
        if isinstance(rf, (np.ndarray, np.generic)):
            assert rf.tobytes() == rs.tobytes()
        assert (rf is out_f) == (rs is out_s)
    assert wf == ws
    assert [_snapshot(x) for x in fast_inputs] == [_snapshot(x) for x in slow_inputs]


def _native_args(op, arrays, kwargs, out):
    if op == "matmul":
        return (arrays[0], arrays[1], out)
    if op == "transpose":
        return (arrays[0],)
    if op in ("sum", "mean"):
        return (arrays[0], kwargs.get("axis"))
    if op == "axpby":
        return (arrays[0], arrays[1], kwargs.get("a", 1.0), kwargs.get("b", 1.0), out)
    if op == "fma3":
        return (*arrays, out)
    return (arrays[0], kwargs.get("scale", 1.0), out)


# --- the acceptance rule, restated ----------------------------------------------


def _is_operand(x, num):
    return (
        type(x) is np.ndarray
        and x.ndim == 2
        and x.dtype.num == num
        and x.dtype.isnative
        and x.flags.aligned
        and x.flags.c_contiguous
        and x.shape[0] > 0
        and x.shape[1] > 0
    )


def _is_matmul_operand(x, num):
    # matmul alone also takes F order (transposed views among it): its kernels
    # read strides and give every layout the same bits
    return (
        type(x) is np.ndarray
        and x.ndim == 2
        and x.dtype.num == num
        and x.dtype.isnative
        and x.flags.aligned
        and (x.flags.c_contiguous or x.flags.f_contiguous)
        and x.shape[0] > 0
        and x.shape[1] > 0
    )


def _is_out(o, num, shape):
    return (
        type(o) is np.ndarray
        and o.ndim == 2
        and o.dtype.num == num
        and o.dtype.isnative
        and o.flags.aligned
        and o.flags.c_contiguous
        and o.flags.writeable
        and o.shape == shape
    )


def _addr(x):
    return x.__array_interface__["data"][0]


def _overlap(x, y):
    return _addr(x) < _addr(y) + y.nbytes and _addr(y) < _addr(x) + x.nbytes


def _is_scalar(s):
    if type(s) is float:
        return True
    if type(s) is int:
        try:
            float(s)
        except OverflowError:
            return False
        return True
    return False


def accepts(op, args):
    """Whether the native entry takes these arguments (routing aside)."""
    x = args[0]
    if type(x) is not np.ndarray or x.dtype.num not in (F32, F64):
        return False
    num = x.dtype.num
    if op == "matmul":
        a, b, out = args
        if not (_is_matmul_operand(a, num) and _is_matmul_operand(b, num) and b.shape[0] == a.shape[1]):
            return False
        return out is None or (
            _is_out(out, num, (a.shape[0], b.shape[1]))
            and not _overlap(out, a)
            and not _overlap(out, b)
        )
    if op == "transpose":
        return _is_operand(x, num)
    if op in ("sum", "mean"):
        axis = args[1]
        return _is_operand(x, num) and (axis is None or (type(axis) is int and axis in (0, 1)))
    n_arrays = OPS[op][0]
    arrays, scalars, out = args[:n_arrays], args[n_arrays:-1], args[-1]
    if not all(_is_operand(y, num) and y.shape == x.shape for y in arrays):
        return False
    if not all(_is_scalar(s) for s in scalars):
        return False
    return out is None or (
        _is_out(out, num, x.shape)
        and all(_addr(y) == _addr(out) or not _overlap(y, out) for y in arrays)
    )


# --- the differential property --------------------------------------------------


def _check(spec, backend):
    op = spec["op"]
    pass_out = spec["out"] != "absent" and _supports_out(op)
    runs = []
    for entry in (NATIVE[op], pg._no_fast_entry):
        arrays, kwargs, out = _build(spec)
        res = _call(op, arrays, kwargs, out, pass_out, entry)
        runs.append(((res, out), arrays + [out]))
    (fast, fast_inputs), (slow, slow_inputs) = runs
    _assert_same(fast, slow, fast_inputs, slow_inputs)

    # The forced cpu backend takes routing out of the matmul acceptance rule.
    if op != "matmul" or backend == "cpu":
        arrays, kwargs, out = _build(spec)
        args = _native_args(op, arrays, kwargs, out)
        assert (NATIVE[op](*args) is not NotImplemented) == accepts(op, args)


@pytest.mark.parametrize("backend", ["cpu", "auto"])
@pytest.mark.parametrize("op", list(OPS))
@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(data=st.data())
def test_fast_entry_matches_python_path(op, backend, data):
    spec = data.draw(op_spec(op))
    pg.set_backend(backend)
    try:
        _check(spec, backend)
    finally:
        pg.set_backend("auto")


# --- deterministic inputs the property might rarely draw ------------------------


def _spec(op, shape, **operand):
    n_arrays = OPS[op][0]
    shapes = [shape] * n_arrays
    if op == "matmul":
        shapes = [shape, (shape[1], 3)]
    base = dict(dtype="<f8", layout="C", container="nd", seed=5, specials=[])
    operands = [dict(base, shape=s, **operand) for s in shapes]
    return dict(op=op, operands=operands, kwargs={}, out="absent")


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize(
    "kind, operand, accepted",
    [
        ("swapped", {"dtype": ">f8"}, False),
        ("readonly", {"layout": "readonly"}, True),
        ("unaligned", {"layout": "unaligned"}, False),
        ("subclass", {"container": "sub"}, False),
        ("odd-stride length-1 dim", {"layout": "odd1"}, True),
    ],
)
def test_input_kind(op, kind, operand, accepted):
    spec = _spec(op, (4, 1), **operand)
    arrays, kwargs, out = _build(spec)
    if kind == "odd-stride length-1 dim":
        assert arrays[0].strides[1] == 1001 * 8 and arrays[0].flags.c_contiguous
    pg.set_backend("cpu")
    try:
        assert (NATIVE[op](*_native_args(op, arrays, kwargs, out)) is not NotImplemented) == accepted
        _check(spec, "cpu")
    finally:
        pg.set_backend("auto")


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("layout", ["F", "T"])
def test_f_order_is_taken_by_matmul_alone(op, layout):
    spec = _spec(op, (4, 3), layout=layout)
    arrays, kwargs, out = _build(spec)
    assert arrays[0].flags.f_contiguous and not arrays[0].flags.c_contiguous
    pg.set_backend("cpu")
    try:
        taken = NATIVE[op](*_native_args(op, arrays, kwargs, out)) is not NotImplemented
        assert taken == (op == "matmul")
        _check(spec, "cpu")
    finally:
        pg.set_backend("auto")


def test_mean_divides_in_float32():
    # Past 2**24 elements float32(count) rounds; NumPy's float32 total / count
    # divides by that rounded count in float32, and so must the fast entry.
    a = np.ones((1, 2**24 + 1), np.float32)
    fast = NATIVE["mean"](a, None)
    slow = _python_path("mean")(a)
    assert type(fast) is type(slow) is np.float32
    assert fast.tobytes() == slow.tobytes()
    total = pg.sum(a)
    assert fast == total / np.float32(2**24 + 1)
    assert fast != np.float32(float(total) / (2**24 + 1))


# --- routing facts --------------------------------------------------------------


def test_routing_decision_table():
    a32 = np.ones((16, 16), np.float32)
    a64 = a32.astype(np.float64)
    work = 2.0 * 16**3

    def computes(facts, a):
        pg._core._set_routing_facts(*facts)
        return NATIVE["matmul"](a, a, None) is not NotImplemented

    assert computes(("cpu", False, False, 0.0), a32)
    assert not computes(("cuda", True, True, math.inf), a32)
    assert not computes(("cuda", True, False, math.inf), a64)
    assert not computes(("auto", False, False, math.inf), a64)
    assert computes(("auto", True, False, 0.0), a32)
    # float64 never goes to the GPU, so no bound applies to it
    assert computes(("auto", True, True, 0.0), a64)
    assert computes(("auto", True, True, work), a32)
    assert not computes(("auto", True, True, work - 1.0), a32)
    pg._core._clear_routing_facts()
    assert NATIVE["matmul"](a64, a64, None) is NotImplemented
    with pytest.raises(ValueError):
        pg._core._set_routing_facts("gpu", True, True, 0.0)


def _recording(monkeypatch):
    pushed = []
    real = pg._set_routing_facts
    monkeypatch.setattr(pg, "_set_routing_facts", lambda *f: pushed.append(f) or real(*f))
    return pushed


def test_pushes_follow_the_router(tmp_path, monkeypatch):
    monkeypatch.setenv("PEREGRINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(pg, "_cuda_fallback_warned", False)
    _stub_gpu_machine(monkeypatch)
    signature = calibrate._machine_signature()
    payload = synthetic_calibration()
    payload["signature"] = signature
    _write_cache_file(policy._cache_path(signature), payload)
    pushed = _recording(monkeypatch)
    bound = policy.f32_cpu_flops_bound(payload)
    a = np.ones((8, 8), np.float32)

    # the first auto dispatch reads the calibration and hands its bound down
    pg.matmul(a, a)
    assert pushed[-1] == ("auto", True, True, bound)

    # set_backend pushes the forced backend and leaves auto unsettled
    pg.set_backend("cpu")
    assert pushed[-1] == ("cpu", False, False, math.inf)
    pg.set_backend("auto")
    assert pushed[-1] == ("auto", False, False, math.inf)

    # calibrate() installs what it measured
    monkeypatch.setattr(
        calibrate, "_measure_grid", lambda budget, start, cuda: {16: (1e-6, 2e-6, 1e-5, 1e-4)}
    )
    monkeypatch.setattr(
        calibrate,
        "_measure_transfer",
        lambda sizes: {"h2d_gbps": 10.0, "d2h_gbps": 10.0, "fixed_overhead_s": 1e-5},
    )
    result = calibrate.calibrate(force=True, budget_seconds=5.0)
    assert pushed[-1] == ("auto", True, True, policy.f32_cpu_flops_bound(result))

    # after the warn-once fallback the device takes no more auto work
    def _failing_staged(*args):
        raise RuntimeError("cudaErrorLaunchFailure at fake.cu:1")

    monkeypatch.setattr(pg, "_gemm_host_native", _failing_staged, raising=False)
    with pytest.warns(RuntimeWarning, match="falling back to cpu"):
        pg._matmul_gpu_staged(a, a)
    assert pushed[-1] == ("auto", True, False, math.inf)

    # and the reset hook clears the native facts
    pg._core._set_routing_facts("cpu", False, False, 0.0)
    pg._reset_routing_state()
    assert NATIVE["matmul"](a, a, None) is NotImplemented


def _calibrations():
    # with no fixed staging cost the GPU wins from the smallest product on
    early = synthetic_calibration()
    early["staged"]["float32"] = [
        [n, _staged_seconds(n, 0.0)] for n, _ in early["staged"]["float32"]
    ]
    # a flat 10 s GPU wins only far past the largest measured size, where the
    # CPU's extrapolated time passes 10 s
    late = synthetic_calibration()
    late["gpu"]["float32"] = [[n, 10.0] for n, _ in late["gpu"]["float32"]]
    # a GPU ten times slower than the CPU at every size never wins
    never = synthetic_calibration()
    never["gpu"]["float32"] = [[n, 10.0 * t] for n, t in never["cpu"]["float32"]]
    return [
        synthetic_calibration(), early, late, never,
        real_calibration(), _smoothed_real_calibration(),
    ]


def _host(cal, m, k, n):
    return policy.choose_backend("float32", m, k, n, calibration=cal, backend="auto")


@given(data=st.data())
def test_bound_admits_no_gpu_shape(data):
    # The native entry computes a host float32 product on the CPU without asking
    # the policy whenever 2*m*k*n <= the bound, so the policy must agree for
    # every shape under it, skinny or fat, with the long side in any position.
    cal = data.draw(st.sampled_from(_calibrations()))
    bound = policy.f32_cpu_flops_bound(cal)
    cap = min(bound, 2.0 * 8192**3)
    n = data.draw(st.integers(1, max(1, min(8192, int(cap // 2)))))
    k = data.draw(st.integers(1, max(1, min(8192, int(cap // (2 * n))))))
    m = data.draw(st.integers(1, max(1, min(8192, int(cap // (2 * n * k))))))
    assume(2.0 * m * k * n <= bound)
    m, k, n = data.draw(st.permutations([m, k, n]))
    assert _host(cal, m, k, n) == "cpu"


def test_bound_admits_no_gpu_shape_at_the_edge():
    # The shapes with the most work the bound admits: the largest cube, the
    # long vector products, and slabs in between, in every orientation.
    for cal in _calibrations():
        bound = policy.f32_cpu_flops_bound(cal)
        if math.isinf(bound) or bound < 2.0:
            continue
        half = int(bound // 2)
        side = round(half ** (1.0 / 3.0)) + 1
        while 2.0 * side**3 > bound:
            side -= 1
        shapes = [(side, side, side), (1, 1, half)]
        shapes += [(a, a, half // (a * a)) for a in (2, 8, 32) if half >= a * a]
        for shape in shapes:
            for m, k, n in set(itertools.permutations(shape)):
                assert 2.0 * m * k * n <= bound
                assert _host(cal, m, k, n) == "cpu", (m, k, n, cal["signature"])


def test_bound_is_tight():
    # A cube attains the AM-GM byte floor the bound is built on whenever the two
    # bandwidths are equal, so there a cube two walk steps past the bound goes
    # to the GPU: the bound gives up no more than the walk's resolution. With
    # unequal bandwidths a cube moves more than the floor and the bound sits
    # lower; test_measured_bound pins it on the measured calibration.
    for cal in _calibrations():
        bound = policy.f32_cpu_flops_bound(cal)
        equal = cal["transfer"]["h2d_gbps"] == cal["transfer"]["d2h_gbps"]
        if math.isinf(bound) or bound == 0.0 or not equal:
            continue
        n = math.ceil((bound * policy._BOUND_STEP**2 / 2.0) ** (1.0 / 3.0))
        assert _host(cal, n, n, n) == "cuda", (n, cal["signature"])
    early, late, never = _calibrations()[1:4]
    assert policy.f32_cpu_flops_bound(early) == 0.0
    assert _host(early, 1, 1, 1) == "cuda"
    assert 2.0 * 4096**3 < policy.f32_cpu_flops_bound(late) < math.inf
    assert math.isinf(policy.f32_cpu_flops_bound(never))
    assert math.isinf(policy.f32_cpu_flops_bound(None))
    no_gpu = synthetic_calibration()
    del no_gpu["gpu"]
    assert math.isinf(policy.f32_cpu_flops_bound(no_gpu))


def test_measured_bound():
    # On the measured calibration every host product up to about a 59-cube
    # stays on the CPU and the first square sent to the GPU is n=60, so a small
    # calibrated product never leaves the native entry.
    bound = policy.f32_cpu_flops_bound(real_calibration())
    assert 2.0 * 59**3 <= bound < 2.0 * 60**3
    assert _host(real_calibration(), 59, 59, 59) == "cpu"
    assert _host(real_calibration(), 60, 60, 60) == "cuda"


def test_calibrated_cuda_build_keeps_small_products_native(monkeypatch):
    # With a device and the measured calibration installed, the native entry
    # takes an 8x8 auto product itself and hands a product past the bound to
    # the policy.
    monkeypatch.setattr(pg, "_cuda_fallback_warned", False)
    _stub_gpu_machine(monkeypatch)
    pg._install_routing_calibration(real_calibration())
    small = np.ones((8, 8), np.float32)
    large = np.ones((64, 64), np.float32)
    assert NATIVE["matmul"](small, small, None) is not NotImplemented
    assert NATIVE["matmul"](large, large, None) is NotImplemented


# --- the switch, the device path, and the build record --------------------------

_SWITCH_SCRIPT = """\
import peregrine as pg
names = ["matmul", "transpose", "sum", "mean", "axpby", "fma3", "scaled_relu"]
bound = [getattr(pg, "_fast_" + n) is getattr(pg._core, "_fast_" + n) for n in names]
stub = [getattr(pg, "_fast_" + n) is pg._no_fast_entry for n in names]
print("NATIVE" if all(bound) else "STUB" if all(stub) else "MIXED")
"""


@pytest.mark.parametrize("value, expected", [("", "NATIVE"), ("1", "STUB"), ("0", "STUB")])
def test_disable_switch(value, expected):
    env = {k: v for k, v in os.environ.items() if k != "PEREGRINE_DISABLE_FAST_ENTRIES"}
    if value:
        env["PEREGRINE_DISABLE_FAST_ENTRIES"] = value
    p = subprocess.run(
        [sys.executable, "-c", _SWITCH_SCRIPT], env=env, capture_output=True, text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == expected


def test_device_operands_take_the_device_path():
    if not pg.has_cuda():
        pytest.skip("no usable cuda device")
    a = np.ones((8, 8), np.float32)
    xa = pg.to_device(a)
    assert NATIVE["matmul"](xa, xa, None) is NotImplemented
    assert NATIVE["axpby"](xa, xa, 1.0, 1.0, None) is NotImplemented
    assert isinstance(pg.matmul(xa, xa), pg.Array)
    assert isinstance(pg.axpby(xa, xa), pg.Array)


def test_build_info():
    info = pg._core._build_info()
    assert set(info) == {"compiler", "flags", "definitions", "cuda", "limited_api", "openmp"}
    assert info["cuda"] is pg._has_cuda_build()
    assert info["compiler"] != "unknown" and info["flags"] != "unknown"
    assert info["limited_api"] == "0x030C0000"
    assert "Py_LIMITED_API=0x030C0000" in info["definitions"]
    assert isinstance(info["openmp"], int) and info["openmp"] >= 200203
