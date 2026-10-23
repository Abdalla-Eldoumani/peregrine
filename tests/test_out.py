"""out= on matmul, axpby, fma3, and scaled_relu, plus host result provenance and
Python scalar narrowing on the fused ops.

The kernel writes straight into out, so out must already be exactly the array a
fresh result would be: an exact numpy.ndarray of the result dtype (byte order
included), the result shape, C-contiguous, aligned, and writeable. A matmul out
may not overlap either operand; a fused out may be exactly one of the inputs
(an in-place update) but may not overlap them any other way. Every message is
pinned verbatim, and every out= result is pinned bitwise against the fresh
result of the same call.
"""

import fractions

import numpy as np
import pytest
from numpy.lib.stride_tricks import as_strided

import peregrine as pg
from conftest import assert_fused_close, assert_matmul_close, requires_cuda

LEGACY_KILLER_SIZES = [1, 3, 7, 50, 250, 333, 750]
DTYPES = [np.float32, np.float64]
DTYPE_IDS = ["f32", "f64"]
OPS = ["matmul", "axpby", "fma3", "scaled_relu"]
FUSED = ["axpby", "fma3", "scaled_relu"]
N_INPUTS = {"matmul": 2, "axpby": 2, "fma3": 3, "scaled_relu": 1}


def _rand(shape, dtype, seed):
    # strictly positive so the fused oracles never cancel (see test_fused.py)
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) ** 2 + 0.5).astype(dtype)


def _inputs(op, shape, dtype):
    # every op yields a result of `shape`; matmul gets a square left operand
    if op == "matmul":
        m, n = shape
        return [_rand((m, m), dtype, 0), _rand((m, n), dtype, 1)]
    return [_rand(shape, dtype, i) for i in range(N_INPUTS[op])]


def _call(op, arrs, **kw):
    if op == "matmul":
        return pg.matmul(arrs[0], arrs[1], **kw)
    if op == "axpby":
        return pg.axpby(arrs[0], arrs[1], a=2.0, b=-0.5, **kw)
    if op == "fma3":
        return pg.fma3(arrs[0], arrs[1], arrs[2], **kw)
    return pg.scaled_relu(arrs[0], scale=1.5, **kw)


def _raises(exc, msg, op, arrs, **kw):
    with pytest.raises(exc) as info:
        _call(op, arrs, **kw)
    assert str(info.value) == msg


def _unaligned(shape, dtype):
    itemsize = np.dtype(dtype).itemsize
    count = int(np.prod(shape))
    buf = bytearray(count * itemsize + 1)
    return np.frombuffer(buf, dtype=dtype, offset=1, count=count).reshape(shape)


class _Sub(np.ndarray):
    pass


# --- the result lands in out, and out is the return value -------------------


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("size", LEGACY_KILLER_SIZES + [64])
def test_out_is_returned_and_bitwise_equal_to_fresh(op, dtype, size):
    arrs = _inputs(op, (size, size + 1), dtype)
    fresh = _call(op, arrs)
    # NaN-poisoned so an element the kernel skipped cannot pass by accident
    out = np.full(fresh.shape, np.nan, dtype)
    assert _call(op, arrs, out=out) is out
    assert out.tobytes() == fresh.tobytes()


@pytest.mark.parametrize("op", OPS)
def test_out_none_matches_omission(op):
    arrs = _inputs(op, (5, 7), np.float64)
    assert _call(op, arrs, out=None).tobytes() == _call(op, arrs).tobytes()


@pytest.mark.parametrize("op", OPS)
def test_out_receives_promoted_dtype(op):
    # int operands promote to float64, so a float64 out is the right target
    arrs = [a.astype(np.int64) for a in _inputs(op, (3, 4), np.float64)]
    out = np.empty((3, 4))
    assert _call(op, arrs, out=out) is out
    assert out.tobytes() == _call(op, arrs).tobytes()


@pytest.mark.parametrize("op", OPS)
def test_out_zero_sized(op):
    arrs = _inputs(op, (0, 4), np.float64)
    out = np.empty((0, 4))
    assert _call(op, arrs, out=out) is out


# --- TypeError: out is not an ndarray of the result dtype --------------------


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(
    "make,got",
    [
        (lambda: [[0.0] * 4] * 3, "list"),
        (lambda: np.zeros((3, 4)).view(_Sub), "_Sub"),
        (lambda: 0.0, "float"),
    ],
    ids=["list", "subclass", "float"],
)
def test_out_wrong_type(op, make, got):
    msg = f"{op}: out must be a numpy.ndarray of dtype float64, got {got}"
    _raises(TypeError, msg, op, _inputs(op, (3, 4), np.float64), out=make())


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(
    "dtype,out_dtype,got",
    [
        (np.float32, np.float64, "float64"),
        (np.float64, np.float32, "float32"),
        (np.float64, np.int64, "int64"),
        (np.float64, ">f8", ">f8"),
        (np.float32, ">f4", ">f4"),
    ],
    ids=["f32-got-f64", "f64-got-f32", "f64-got-i64", "f64-got-be", "f32-got-be"],
)
def test_out_wrong_dtype(op, dtype, out_dtype, got):
    # byte order is part of the dtype: a '>f8' out is not a float64 out
    want = np.dtype(dtype).name
    msg = f"{op}: out must be a numpy.ndarray of dtype {want}, got {got}"
    out = np.zeros((3, 4), out_dtype)
    _raises(TypeError, msg, op, _inputs(op, (3, 4), dtype), out=out)


# --- ValueError: wrong shape or layout ----------------------------------------


def _layout_msg(op, detail):
    return f"{op}: out must be a writeable C-contiguous array of shape (3, 4), got {detail}"


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("shape", [(4, 3), (3, 5), (12,), (3, 4, 1)])
def test_out_wrong_shape(op, shape):
    out = np.zeros(shape)
    _raises(ValueError, _layout_msg(op, f"shape {shape}"), op,
            _inputs(op, (3, 4), np.float64), out=out)


def _readonly():
    out = np.zeros((3, 4))
    out.flags.writeable = False
    return out


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(
    "make,detail",
    [
        (lambda: np.zeros((3, 4), order="F"), "a non-C-contiguous array"),
        (lambda: np.zeros((3, 8))[:, ::2], "a non-C-contiguous array"),
        (lambda: np.zeros((6, 4))[::2], "a non-C-contiguous array"),
        (lambda: _unaligned((3, 4), np.float64), "an unaligned array"),
        (_readonly, "a read-only array"),
    ],
    ids=["F-order", "col-stride", "row-stride", "unaligned", "read-only"],
)
def test_out_wrong_layout(op, make, detail):
    _raises(ValueError, _layout_msg(op, detail), op,
            _inputs(op, (3, 4), np.float64), out=make())


@pytest.mark.parametrize("op", OPS)
def test_out_check_order(op):
    arrs = _inputs(op, (3, 4), np.float64)
    # type before dtype before shape before layout
    _raises(TypeError, f"{op}: out must be a numpy.ndarray of dtype float64, got float32",
            op, arrs, out=np.zeros((4, 3), np.float32, order="F"))
    _raises(ValueError, _layout_msg(op, "shape (4, 3)"), op, arrs,
            out=np.zeros((4, 3), order="F"))
    ro = np.zeros((3, 4), order="F")
    ro.flags.writeable = False
    _raises(ValueError, _layout_msg(op, "a non-C-contiguous array"), op, arrs, out=ro)


def test_operand_errors_come_before_out_errors():
    bad = [[0.0]]
    with pytest.raises(ValueError, match="inner dimensions do not match"):
        pg.matmul(np.zeros((2, 3)), np.zeros((4, 5)), out=bad)
    with pytest.raises(ValueError, match="2-dimensional"):
        pg.matmul(np.zeros(3), np.zeros((3, 2)), out=bad)
    with pytest.raises(TypeError, match="unsupported dtype float16"):
        pg.matmul(np.zeros((2, 2), np.float16), np.zeros((2, 2)), out=bad)
    with pytest.raises(ValueError, match="operands must have the same shape"):
        pg.axpby(np.zeros((2, 2)), np.zeros((2, 3)), out=bad)
    with pytest.raises(TypeError, match="unsupported dtype bool"):
        pg.scaled_relu(np.zeros((2, 2), bool), out=bad)


# --- matmul: out must not overlap either operand ------------------------------

_MATMUL_OVERLAP = "matmul: out must not overlap an input"


@pytest.mark.parametrize("which", [0, 1])
def test_matmul_out_is_an_operand(which):
    arrs = [_rand((4, 4), np.float64, 0), _rand((4, 4), np.float64, 1)]
    _raises(ValueError, _MATMUL_OVERLAP, "matmul", arrs, out=arrs[which])


@pytest.mark.parametrize("which", [0, 1])
def test_matmul_out_partially_overlaps_an_operand(which):
    buf = _rand((8, 4), np.float64, 2)
    arrs = [_rand((4, 4), np.float64, 0), _rand((4, 4), np.float64, 1)]
    arrs[which] = buf[:4]
    _raises(ValueError, _MATMUL_OVERLAP, "matmul", arrs, out=buf[3:7])


def test_matmul_out_overlaps_a_strided_operand():
    # the operand is copied before the kernel reads it, but the rule is about
    # the arrays as passed, whatever layout they arrive in
    buf = _rand((4, 8), np.float64, 2)
    a = buf[:, ::2]
    # rows 2 and 3 of buf, reinterpreted as a C-contiguous (4, 4) block
    out = buf.reshape(8, 4)[4:]
    _raises(ValueError, _MATMUL_OVERLAP, "matmul", [a, np.eye(4)], out=out)


def test_matmul_out_disjoint_view_of_operand_buffer():
    buf = _rand((8, 4), np.float64, 2)
    a, out = buf[:4], buf[4:]
    b = _rand((4, 4), np.float64, 1)
    expect = pg.matmul(a, b)
    assert pg.matmul(a, b, out=out) is out
    assert out.tobytes() == expect.tobytes()


# --- fused: out may be exactly an input, never a partial overlap ---------------


def _fused_overlap_msg(op):
    return f"{op}: out must be one of the inputs exactly or not overlap them"


@pytest.mark.parametrize("op", FUSED)
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("size", LEGACY_KILLER_SIZES)
def test_fused_in_place_on_each_input(op, dtype, size):
    # The kernels read every element before writing it and touch no other
    # index, so exact aliasing is safe in the vector blocks and the scalar tail
    # alike; the legacy sizes put elements in both.
    for i in range(N_INPUTS[op]):
        arrs = _inputs(op, (size, size + 1), dtype)
        expect = _call(op, [a.copy() for a in arrs])
        target = arrs[i]
        assert _call(op, arrs, out=target) is target
        assert target.tobytes() == expect.tobytes()


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("size", [7, 333])
def test_fused_same_array_twice_with_out_it(dtype, size):
    x = _rand((size, size), dtype, 0)
    z = _rand((size, size), dtype, 1)
    cases = [
        ("axpby", lambda x, z, **kw: pg.axpby(x, x, a=2.0, b=3.0, **kw)),
        ("fma3 x x z", lambda x, z, **kw: pg.fma3(x, x, z, **kw)),
        ("fma3 x x x", lambda x, z, **kw: pg.fma3(x, x, x, **kw)),
    ]
    for _, call in cases:
        xi = x.copy()
        expect = call(x.copy(), z)
        assert call(xi, z, out=xi) is xi
        assert xi.tobytes() == expect.tobytes()


@pytest.mark.parametrize("op", FUSED)
def test_fused_partial_overlap(op):
    for i in range(N_INPUTS[op]):
        buf = _rand((8, 5), np.float64, 7)
        arrs = _inputs(op, (4, 5), np.float64)
        arrs[i] = buf[:4]
        _raises(ValueError, _fused_overlap_msg(op), op, arrs, out=buf[1:5])


@pytest.mark.parametrize("op", FUSED)
def test_fused_same_pointer_but_not_exact(op):
    # same data pointer and shape, but a transposed view is not C-contiguous,
    # and a reinterpreting view has another dtype: neither is the same array
    out = _rand((4, 4), np.float64, 7)
    arrs = _inputs(op, (4, 4), np.float64)
    arrs[0] = out.T
    _raises(ValueError, _fused_overlap_msg(op), op, arrs, out=out)
    arrs[0] = out.view(np.int64)
    _raises(ValueError, _fused_overlap_msg(op), op, arrs, out=out)


@pytest.mark.parametrize("op", FUSED)
def test_fused_exact_alias_ignores_length_one_strides(op):
    out = _rand((1, 6), np.float64, 7)
    arrs = _inputs(op, (1, 6), np.float64)
    arrs[0] = as_strided(out, shape=(1, 6), strides=(999 * 8, 8))
    assert arrs[0].flags.c_contiguous
    expect = _call(op, [a.copy() for a in arrs])
    assert _call(op, arrs, out=out) is out
    assert out.tobytes() == expect.tobytes()


@pytest.mark.parametrize("op", FUSED)
def test_fused_disjoint_view_of_input_buffer(op):
    buf = _rand((8, 5), np.float64, 7)
    arrs = _inputs(op, (4, 5), np.float64)
    arrs[0] = buf[:4]
    expect = _call(op, [a.copy() for a in arrs])
    out = buf[4:]
    assert _call(op, arrs, out=out) is out
    assert out.tobytes() == expect.tobytes()


# --- device operands ------------------------------------------------------------

_cuda_ok, _cuda_reason = requires_cuda()
gpu = pytest.mark.skipif(not _cuda_ok, reason=_cuda_reason)


@gpu
@pytest.mark.parametrize("op", OPS)
def test_device_operands_with_out_are_mixed_residency(op):
    arrs = [pg.to_device(a) for a in _inputs(op, (4, 4), np.float32)]
    if op == "matmul":
        msg = pg._MIXED_RESIDENCY_MSG
    else:
        msg = f"{op}: inputs must all be on the same device, call to_device or from_device first"
    _raises(TypeError, msg, op, arrs, out=np.empty((4, 4), np.float32))


@gpu
def test_matmul_out_on_a_gpu_route(monkeypatch):
    # a forced cuda backend stages a host float32 pair to the device; the staged
    # product lands in out, bit for bit the product staged without out
    monkeypatch.setattr(pg, "_backend", "cuda")
    a = _rand((64, 48), np.float32, 0)
    b = _rand((48, 80), np.float32, 1)
    out = np.full((64, 80), np.nan, np.float32)
    assert pg.matmul(a, b, out=out) is out
    assert_matmul_close(out, a @ b, a, b)
    assert out.tobytes() == pg.matmul(a, b).tobytes()


# --- every host result is NumPy-allocated ---------------------------------------


def _owned(r):
    return type(r) is np.ndarray and r.flags.owndata and r.base is None


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_host_results_are_numpy_owned(monkeypatch, dtype):
    monkeypatch.setattr(pg, "_backend", "cpu")
    a = _rand((5, 7), dtype, 0)
    b = _rand((7, 3), dtype, 1)
    results = [
        pg.matmul(a, b),
        pg.matmul(np.asfortranarray(a), b),
        pg.matmul(a.astype(np.int64), b.astype(np.int64)),
        pg.transpose(a),
        pg.sum(a, axis=0),
        pg.sum(a, axis=1),
        pg.mean(a, axis=0),
        pg.mean(a, axis=1),
        pg.axpby(a, a, a=2.0),
        pg.fma3(a, a, a),
        pg.scaled_relu(a, scale=0.5),
    ]
    for r in results:
        assert _owned(r), (type(r), r.flags.owndata, type(r.base))
    # axis=None stays an exact NumPy scalar of the result dtype, not a float
    assert type(pg.sum(a)) is np.dtype(dtype).type
    assert type(pg.mean(a)) is np.dtype(dtype).type


# --- Python scalars convert to the array dtype (NumPy's weak-scalar rule) -------

_SCALARS = [0.1, 1e300, -0.0, float("nan"), 2**63]
_SCALAR_IDS = ["0.1", "1e300", "-0.0", "nan", "2**63"]


def _as_f32(s):
    # NumPy warns on the out-of-range cast; peregrine narrows silently
    with np.errstate(over="ignore"):
        return np.float32(s)


@pytest.mark.parametrize("s", _SCALARS, ids=_SCALAR_IDS)
def test_axpby_scalars_narrow_to_float32(s):
    x = _rand((5, 37), np.float32, 0)
    y = _rand((5, 37), np.float32, 1)
    got = pg.axpby(x, y, a=s, b=s)
    assert got.dtype == np.float32
    s32 = _as_f32(s)
    # the same kernel fed the pre-narrowed coefficient: bitwise equal, so the
    # computation used float32(s), never s in double precision
    assert got.tobytes() == pg.axpby(x, y, a=float(s32), b=float(s32)).tobytes()
    with np.errstate(over="ignore", invalid="ignore"):
        ref = s32 * x + s32 * y
    assert_fused_close(got, ref)


@pytest.mark.parametrize("s", _SCALARS, ids=_SCALAR_IDS)
def test_scaled_relu_scale_narrows_to_float32(s):
    x = np.random.default_rng(0).standard_normal((5, 37)).astype(np.float32)
    got = pg.scaled_relu(x, scale=s)
    assert got.dtype == np.float32
    s32 = _as_f32(s)
    assert got.tobytes() == pg.scaled_relu(x, scale=float(s32)).tobytes()
    with np.errstate(over="ignore", invalid="ignore"):
        ref = np.maximum(s32 * x, 0)
    assert_fused_close(got, ref)


def test_scalar_too_large_for_float_raises():
    x = _rand((2, 2), np.float32, 0)
    with pytest.raises(OverflowError, match="int too large to convert to float"):
        pg.axpby(x, x, a=2**1100)
    with pytest.raises(OverflowError, match="int too large to convert to float"):
        pg.scaled_relu(x, scale=2**1100)


@pytest.mark.parametrize("s", ["0.5", fractions.Fraction(1, 2), True])
def test_scalar_accepts_what_float_accepts(s):
    x = _rand((3, 3), np.float32, 0)
    want = float(s)
    assert pg.axpby(x, x, a=s).tobytes() == pg.axpby(x, x, a=want).tobytes()
    assert pg.scaled_relu(x, scale=s).tobytes() == pg.scaled_relu(x, scale=want).tobytes()


# --- the native entries re-validate, never crash ----------------------------------


def test_native_into_entries_validate():
    core = pg._core
    a = np.ones((2, 3))
    with pytest.raises(ValueError, match=r"matmul: inner dimensions do not match: \(2, 3\) @ \(2, 3\)"):
        core.matmul_into(a, a, np.empty((2, 3)))
    with pytest.raises(ValueError, match=r"got shape \(3, 3\)"):
        core.matmul_into(a, a.T.copy(), np.empty((3, 3)))
    with pytest.raises(ValueError, match=r"transpose: out must .* shape \(3, 2\), got shape \(2, 3\)"):
        core.transpose_into(a, np.empty((2, 3)))
    with pytest.raises(ValueError, match="axis 2 is out of bounds"):
        core.sum_axis_into(a, 2, np.empty(3))
    with pytest.raises(ValueError, match=r"shape \(3,\), got shape \(2,\)"):
        core.sum_axis_into(a, 0, np.empty(2))
    with pytest.raises(ValueError, match="axpby: operands must have the same shape"):
        core.axpby_into(a, np.ones((3, 2)), 1.0, 1.0, np.empty((2, 3)))
    with pytest.raises(ValueError, match="fma3: operands must have the same shape"):
        core.fma3_into(a, a, np.ones((3, 2)), np.empty((2, 3)))
    with pytest.raises(ValueError, match=r"scaled_relu: out must .* got shape \(3, 2\)"):
        core.scaled_relu_into(a, 1.0, np.empty((3, 2)))
    # dtype, layout, and writeability mismatches fail overload resolution
    # (noconvert arrays are never converted into a temporary)
    readonly = np.empty((2, 3))
    readonly.flags.writeable = False
    for out in (np.empty((2, 3), np.float32), np.empty((2, 3), order="F"), readonly):
        with pytest.raises(TypeError):
            core.scaled_relu_into(a, 1.0, out)
    with pytest.raises(TypeError):
        core.scaled_relu_into(a.astype(np.float32), 1.0, np.empty((2, 3)))
