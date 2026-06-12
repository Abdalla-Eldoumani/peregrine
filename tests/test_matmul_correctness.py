"""Correctness oracle: every result is checked against NumPy.

The size list deliberately includes 1, 3, 7, 50, 250, 333, and 750. The legacy
kernel crashed or returned silently wrong results at exactly these sizes
because its vector tail handling assumed remainders divisible by four.
"""

import numpy as np
import pytest

import fastmathext as fme
from conftest import assert_matmul_close

LEGACY_KILLER_SIZES = [1, 3, 7, 50, 250, 333, 750]
ROUND_SIZES = [4, 8, 16, 64, 128, 256]


def _check(m, k, n, dtype, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    got = fme.matmul(a, b)
    ref = a @ b
    assert_matmul_close(got, ref, a, b)


@pytest.mark.parametrize("n", LEGACY_KILLER_SIZES + ROUND_SIZES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_square(n, dtype):
    _check(n, n, n, dtype)


@pytest.mark.parametrize(
    "m,k,n",
    [(1, 1, 1), (1, 100, 1), (100, 1, 100), (13, 77, 205), (130, 77, 5), (5, 1000, 3)],
)
def test_rectangular(m, k, n):
    _check(m, k, n, np.float64)


@pytest.mark.parametrize("shape_a,shape_b", [((0, 5), (5, 3)), ((5, 0), (0, 3)), ((0, 0), (0, 0))])
def test_zero_sized_matches_numpy(shape_a, shape_b):
    a = np.zeros(shape_a)
    b = np.zeros(shape_b)
    got = fme.matmul(a, b)
    ref = a @ b
    assert got.shape == ref.shape
    np.testing.assert_array_equal(got, ref)


def test_transposed_view_input():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((64, 32))
    b = rng.standard_normal((50, 32)).T  # F-order view exercises the copy path
    assert_matmul_close(fme.matmul(a, b), a @ b, a, b)


def test_integer_inputs_promote():
    a = np.arange(6, dtype=np.int64).reshape(2, 3)
    b = np.arange(12, dtype=np.int64).reshape(3, 4)
    got = fme.matmul(a, b)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, (a @ b).astype(np.float64))


def test_nan_and_inf_propagate():
    a = np.array([[1.0, np.nan], [np.inf, 2.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    got = fme.matmul(a, b)
    ref = a @ b
    # the helper's special-value path checks NaN positions, Inf positions,
    # and each Inf's sign, then bounds the finite remainder
    assert_matmul_close(got, ref, a, b)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_copy_pointer_identity(dtype):
    # C-contiguous matching-dtype input must reach the kernel without a copy;
    # _normalize returning the same object is that contract made testable
    arr = np.zeros((8, 8), dtype=dtype)
    normalized = fme._normalize(arr, arr.dtype)
    assert normalized is arr
    assert normalized.ctypes.data == arr.ctypes.data


def test_zero_copy_strided_inputs_take_copy_path():
    rng = np.random.default_rng(0)
    f_order = np.asfortranarray(rng.standard_normal((16, 8)))
    strided = rng.standard_normal((16, 8))[::2]
    for view in (f_order, strided):
        normalized = fme._normalize(view, view.dtype)
        assert normalized.ctypes.data != view.ctypes.data
    b = rng.standard_normal((8, 4))
    for view in (f_order, strided):
        assert_matmul_close(fme.matmul(view, b), view @ b, view, b)


def test_zero_sized_promotes_dtype():
    a = np.zeros((0, 4), dtype=np.int64)
    b = np.zeros((4, 3), dtype=np.int64)
    got = fme.matmul(a, b)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, (a @ b).astype(np.float64))

    a = np.zeros((5, 0), dtype=np.int32)
    b = np.zeros((0, 3), dtype=np.float32)
    got = fme.matmul(a, b)
    # np.result_type(int32, float32) is float64, even on empty shapes
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, np.zeros((5, 3), dtype=np.float64))

    a = np.zeros((0, 0), dtype=np.float32)
    b = np.zeros((0, 0), dtype=np.float64)
    got = fme.matmul(a, b)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, np.zeros((0, 0), dtype=np.float64))
