"""The matmul fast paths, required for the small-matrix speed target.

At n=8 the wrapper helpers alone (2x np.asarray, 2x np.ascontiguousarray,
np.result_type) cost ~1.8 us against a 1.75 us 2x-NumPy budget, so the small
win is unreachable unless matmul skips normalization when both inputs already
qualify. A qualifying pair goes to the native fast entry, which computes it
with none of the wrapper's Python machinery; with fast entries disabled it
takes the Python pre-branch, which still skips the helpers. These tests pin
that both paths are actually taken for qualifying inputs, that everything else
falls through to the full policy chain, and that taking them never weakens the
contract or copies an input.
"""

import tracemalloc

import numpy as np
import pytest

import peregrine as pg
from conftest import assert_matmul_close


@pytest.fixture(params=["native", "python"])
def entry(request, monkeypatch):
    # "native" forces the cpu backend, so routing cannot hand the pair back to
    # Python; "python" stubs the native entry out, leaving the pre-branch.
    if request.param == "native":
        monkeypatch.setattr(pg, "_fast_matmul", pg._core._fast_matmul)
        pg.set_backend("cpu")
        yield request.param
        pg.set_backend("auto")
    else:
        monkeypatch.setattr(pg, "_fast_matmul", pg._no_fast_entry)
        yield request.param


def _qualifying(dtype):
    a = np.ascontiguousarray(np.random.default_rng(0).standard_normal((8, 8)), dtype=dtype)
    b = np.ascontiguousarray(np.random.default_rng(1).standard_normal((8, 8)), dtype=dtype)
    return a, b


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fast_path_skips_helper_chain(monkeypatch, entry, dtype):
    # Make every normalization helper explode. A qualifying call must still
    # succeed, which is only possible if the fast path skipped all three. The
    # native entry also skips the router and the nanobind binding.
    def _boom(*args, **kwargs):
        raise AssertionError("fast path must skip the helper chain for qualifying inputs")

    monkeypatch.setattr(pg, "_prepare", _boom)
    monkeypatch.setattr(pg, "_normalize", _boom)
    monkeypatch.setattr(np, "result_type", _boom)
    if entry == "native":
        monkeypatch.setattr(pg, "_host_backend_for", _boom)
        monkeypatch.setattr(pg, "_matmul_into", _boom)

    a, b = _qualifying(dtype)
    got = pg.matmul(a, b)
    assert got.dtype == dtype
    assert got.shape == (8, 8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fast_path_result_matches_slow_path(entry, dtype):
    # The fast path must return exactly what the full chain returns. The kernel
    # is deterministic and the slow path also lands on a no-copy normalize for
    # these inputs, so bitwise equality is the correct bar.
    a, b = _qualifying(dtype)
    fast = pg.matmul(a, b)
    ref = a @ b
    assert_matmul_close(fast, ref, a, b)
    full = pg._matmul_cpu(pg._normalize(a, a.dtype), pg._normalize(b, b.dtype))
    assert fast.dtype == full.dtype
    assert fast.tobytes() == full.tobytes()


def test_non_contiguous_falls_through_to_slow_path():
    # An F-order operand does not qualify; the slow path's _normalize copy must
    # still run and the result must still be correct.
    rng = np.random.default_rng(0)
    a = np.asfortranarray(rng.standard_normal((16, 8)))
    b = rng.standard_normal((8, 4))
    assert_matmul_close(pg.matmul(a, b), a @ b, a, b)


def test_mixed_dtype_falls_through_to_slow_path():
    # Same-shape, both C-contiguous, but different dtypes: must NOT take the
    # fast path (it would skip the promotion the result dtype depends on).
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 8)).astype(np.float32)
    b = rng.standard_normal((8, 8)).astype(np.float64)
    got = pg.matmul(a, b)
    assert got.dtype == np.float64
    assert_matmul_close(got, a @ b, a, b)


def test_integer_input_falls_through_and_promotes():
    # int inputs are C-contiguous and 2-D but not an accepted float dtype; the
    # fast path must decline so the int->float64 promotion still happens.
    a = np.arange(6, dtype=np.int64).reshape(2, 3)
    b = np.arange(12, dtype=np.int64).reshape(3, 4)
    got = pg.matmul(a, b)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, (a @ b).astype(np.float64))


def test_rejected_dtype_still_raises_on_fast_path_candidate():
    # float16 is C-contiguous, 2-D, same dtype on both operands: a shape match
    # for the fast path, but the dtype is rejected. The TypeError must still
    # fire (the fast path is a pre-branch, not a contract replacement).
    a = np.zeros((8, 8), dtype=np.float16)
    with pytest.raises(TypeError, match="unsupported dtype float16"):
        pg.matmul(a, a)


def test_list_input_falls_through_to_slow_path():
    # Python lists are not ndarrays; the fast path's isinstance gate must
    # decline and _prepare must run.
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[1.0, 0.0], [0.0, 1.0]]
    got = pg.matmul(a, b)
    np.testing.assert_array_equal(got, np.array(a) @ np.array(b))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fast_path_is_zero_copy(entry, dtype):
    # The fast path must not copy a C-contiguous matching-dtype input; the
    # result owns its own buffer but the inputs must reach the kernel without
    # a defensive copy (the zero-copy contract the wrapper promises). NumPy
    # reports its buffer allocations to tracemalloc, so with all three arrays
    # the same size, a copy of either input would at least double the traced
    # peak over the result alone.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((256, 256)).astype(dtype)
    b = rng.standard_normal((256, 256)).astype(dtype)
    pg.matmul(a, b)

    def traced_peak(call):
        tracemalloc.start()
        try:
            call()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    assert traced_peak(lambda: pg.matmul(a, b)) < 1.5 * a.nbytes
    # the instrument sees a copy when there is one
    assert traced_peak(a.copy) >= a.nbytes
