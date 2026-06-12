"""Correctness oracle: every result is checked against NumPy.

The size list deliberately includes 1, 3, 7, 50, 250, 333, and 750. The legacy
kernel crashed or returned silently wrong results at exactly these sizes
because its vector tail handling assumed remainders divisible by four.
"""

import numpy as np
import pytest

import fastmathext as fme

LEGACY_KILLER_SIZES = [1, 3, 7, 50, 250, 333, 750]
ROUND_SIZES = [4, 8, 16, 64, 128, 256]


def _check(m, k, n, dtype, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    got = fme.matmul(a, b)
    ref = a @ b
    assert got.dtype == ref.dtype
    assert got.shape == ref.shape
    if dtype == np.float64:
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
        return
    # float32 summation order differs from BLAS, so a fixed rtol fails once K
    # grows (K=750 broke rtol=1e-5). Compare both results against the float64
    # ground truth and require ours to stay within a small multiple of
    # NumPy's own rounding error, with a K*eps floor for tiny K.
    truth = a.astype(np.float64) @ b.astype(np.float64)
    err_got = np.abs(got.astype(np.float64) - truth).max()
    err_ref = np.abs(ref.astype(np.float64) - truth).max()
    bound = 4.0 * max(err_ref, k * float(np.finfo(np.float32).eps))
    assert err_got <= bound, (
        f"f32 max abs error {err_got:.3e} exceeds bound {bound:.3e} "
        f"(numpy's own error vs float64 truth: {err_ref:.3e})"
    )


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
    np.testing.assert_allclose(fme.matmul(a, b), a @ b, rtol=1e-12)


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
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_array_equal(np.isinf(got), np.isinf(ref))
