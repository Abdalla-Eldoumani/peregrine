"""Correctness oracle for transpose, sum, and mean: every result is checked
against NumPy, reductions through the single toleranced reduction path.

The size list carries the legacy-killer sizes (1, 3, 7, 50, 250, 333, 750) for
the same reason the matmul suite does: the archived kernel corrupted results at
sizes whose tail was not divisible by four, and the reduction tail (the scalar
remainder below the 8-lane pairwise stride) is the analogous danger here.

transpose is exact (assert_array_equal): it is a pure data copy with no
arithmetic, so any divergence from a.T is a bug, not rounding. sum and mean
route every comparison through assert_reduce_close, the only toleranced
reduction path in the suite; an inline rtol/atol here would be a defect. The
mean-of-empty warning matrix is pinned from the NumPy 2.4.6 probe: axis None and
axis 0 emit two RuntimeWarnings, axis 1 on an empty-output shape emits none.
"""

import warnings

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import peregrine as pg
from conftest import assert_reduce_close

LEGACY_KILLER_SIZES = [1, 3, 7, 50, 250, 333, 750]
ROUND_SIZES = [4, 8, 16, 64, 128, 256]
SIZES = LEGACY_KILLER_SIZES + ROUND_SIZES
DTYPES = [np.float32, np.float64]
AXES = [None, 0, 1]


class TestTranspose:
    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_square(self, n, dtype):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((n, n)).astype(dtype)
        np.testing.assert_array_equal(pg.transpose(a), a.T)

    @pytest.mark.parametrize(
        "m,n",
        [(1, 7), (7, 1), (13, 205), (205, 13), (3, 750), (750, 3), (50, 333)],
    )
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_rectangular(self, m, n, dtype):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((m, n)).astype(dtype)
        np.testing.assert_array_equal(pg.transpose(a), a.T)

    @pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
    def test_zero_sized_matches_numpy(self, shape):
        a = np.zeros(shape)
        got = pg.transpose(a)
        assert got.shape == a.T.shape
        np.testing.assert_array_equal(got, a.T)

    def test_result_is_owned_copy(self):
        # transpose diverges from NumPy's a.T view on purpose: the result is a
        # fresh buffer, so mutating it must not reach back into the input
        a = np.arange(12.0).reshape(3, 4)
        a_before = a.copy()
        t = pg.transpose(a)
        assert t.ctypes.data != a.ctypes.data
        t[0, 0] = 999.0
        np.testing.assert_array_equal(a, a_before)

    def test_integer_promotes(self):
        a = np.arange(6, dtype=np.int64).reshape(2, 3)
        got = pg.transpose(a)
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, a.T.astype(np.float64))


class TestReductions:
    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("axis", AXES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("kind", ["sum", "mean"])
    def test_oracle_square(self, n, axis, dtype, kind):
        rng = np.random.default_rng(n + (axis or 0))
        a = rng.standard_normal((n, n)).astype(dtype)
        op = pg.sum if kind == "sum" else pg.mean
        ref_op = np.sum if kind == "sum" else np.mean
        got = op(a, axis=axis)
        ref = ref_op(a, axis=axis)
        # np scalar for axis=None carries .shape == () and .dtype, so the helper
        # compares it the same way it compares the 1-D arrays for axis 0/1
        assert_reduce_close(np.asarray(got), np.asarray(ref), a, axis, kind=kind)

    @pytest.mark.parametrize("m,n", [(13, 205), (205, 13), (1, 333), (333, 1), (7, 50)])
    @pytest.mark.parametrize("axis", AXES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("kind", ["sum", "mean"])
    def test_oracle_rectangular(self, m, n, axis, dtype, kind):
        rng = np.random.default_rng(m * 31 + n)
        a = rng.standard_normal((m, n)).astype(dtype)
        op = pg.sum if kind == "sum" else pg.mean
        ref_op = np.sum if kind == "sum" else np.mean
        got = op(a, axis=axis)
        ref = ref_op(a, axis=axis)
        assert_reduce_close(np.asarray(got), np.asarray(ref), a, axis, kind=kind)

    @pytest.mark.parametrize("axis", AXES)
    @pytest.mark.parametrize("kind", ["sum", "mean"])
    def test_integer_promotes_to_float64(self, axis, kind):
        # int x int diverges from NumPy by design (NumPy keeps int64 for sum);
        # we are a float library, so the result is float64 and the value matches
        # the float64 reduction of the same data
        a = np.arange(24, dtype=np.int64).reshape(4, 6)
        op = pg.sum if kind == "sum" else pg.mean
        ref_op = np.sum if kind == "sum" else np.mean
        got = op(a, axis=axis)
        ref = ref_op(a.astype(np.float64), axis=axis)
        assert np.asarray(got).dtype == np.float64
        assert_reduce_close(
            np.asarray(got), np.asarray(ref), a.astype(np.float64), axis, kind=kind
        )

    @pytest.mark.parametrize("axis", AXES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_nan_and_inf_propagate(self, axis, dtype):
        # the helper's special-value path checks NaN positions, Inf positions,
        # and each Inf's sign positionally, then bounds the finite remainder
        a = np.array(
            [[1.0, np.nan, 3.0], [np.inf, 5.0, -np.inf], [7.0, 8.0, 9.0]], dtype=dtype
        )
        got = pg.sum(a, axis=axis)
        ref = np.sum(a, axis=axis)
        assert_reduce_close(np.asarray(got), np.asarray(ref), a, axis, kind="sum")

    def test_signed_zero_sums_to_positive_zero(self):
        # NumPy's empty/all--0.0 identity is +0.0; the pairwise accumulators
        # start at +0.0 to match it
        for dtype in DTYPES:
            a = np.full((4, 4), -0.0, dtype=dtype)
            s = pg.sum(a)
            assert float(s) == 0.0
            assert not np.signbit(s)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_axis_none_returns_scalar_of_result_dtype(self, dtype):
        a = np.ones((3, 5), dtype=dtype)
        s = pg.sum(a)
        m = pg.mean(a)
        assert s.dtype == np.dtype(dtype) and s.shape == ()
        assert m.dtype == np.dtype(dtype) and m.shape == ()

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_axis_reduces_to_correct_length(self, dtype):
        a = np.ones((4, 6), dtype=dtype)
        # axis 0 reduces over rows -> length n; axis 1 over columns -> length m
        assert pg.sum(a, axis=0).shape == (6,)
        assert pg.sum(a, axis=1).shape == (4,)


class TestEmptyAndErrors:
    def test_sum_of_empty_axis_none_is_exactly_zero(self):
        for dtype in DTYPES:
            s = pg.sum(np.zeros((0, 5), dtype=dtype))
            assert float(s) == 0.0
            assert s.dtype == np.dtype(dtype)

    def test_sum_of_empty_axis_shapes_match_numpy(self):
        a = np.zeros((0, 5))
        np.testing.assert_array_equal(pg.sum(a, axis=0), np.sum(a, axis=0))
        np.testing.assert_array_equal(pg.sum(a, axis=1), np.sum(a, axis=1))

    def test_mean_empty_axis_none_warns_twice(self):
        # Probe 6: nan scalar + "Mean of empty slice" then an invalid-divide
        # warning, matching NumPy exactly
        a = np.zeros((0, 5))
        with pytest.warns(RuntimeWarning) as record:
            r = pg.mean(a)
        assert np.isnan(r)
        messages = [str(w.message) for w in record]
        assert any("Mean of empty slice" in m for m in messages)
        assert any("invalid value encountered" in m for m in messages)
        assert len(record) == 2

    def test_mean_empty_axis_zero_warns_twice(self):
        a = np.zeros((0, 5))
        with pytest.warns(RuntimeWarning) as record:
            r = pg.mean(a, axis=0)
        assert r.shape == (5,) and np.all(np.isnan(r))
        messages = [str(w.message) for w in record]
        assert any("Mean of empty slice" in m for m in messages)
        assert any("invalid value encountered" in m for m in messages)
        assert len(record) == 2

    def test_mean_empty_axis_one_emits_no_warning(self):
        # axis 1 on a (0, 5) input reduces over the 5 columns into a length-0
        # output: the count is 5 (non-zero) and the output is empty, so neither
        # the empty-slice warning nor the divide warning fires
        a = np.zeros((0, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = pg.mean(a, axis=1)
        assert r.shape == (0,)

    @pytest.mark.parametrize("op", [pg.sum, pg.mean, pg.transpose])
    @pytest.mark.parametrize(
        "name", ["bool", "float16", "complex64", "complex128", "object"]
    )
    def test_rejected_dtype_is_type_error(self, op, name):
        a = np.zeros((2, 2), dtype=name)
        with pytest.raises(TypeError, match=f"unsupported dtype {name}"):
            op(a)

    @pytest.mark.parametrize("op", [pg.sum, pg.mean])
    def test_axis_out_of_range_is_value_error(self, op):
        a = np.zeros((2, 2))
        with pytest.raises(ValueError, match="axis 2 is out of bounds"):
            op(a, axis=2)
        with pytest.raises(ValueError, match="out of bounds"):
            op(a, axis=-1)

    @pytest.mark.parametrize("op", [pg.sum, pg.mean])
    def test_non_int_axis_is_type_error(self, op):
        a = np.zeros((2, 2))
        with pytest.raises(TypeError, match="axis must be None, 0, or 1"):
            op(a, axis=1.0)
        with pytest.raises(TypeError, match="axis must be None, 0, or 1"):
            op(a, axis="0")

    @pytest.mark.parametrize("op", [pg.sum, pg.mean])
    def test_axis_is_keyword_only(self, op):
        a = np.zeros((2, 2))
        with pytest.raises(TypeError):
            op(a, 0)

    @pytest.mark.parametrize("op", [pg.sum, pg.mean, pg.transpose])
    def test_one_dimensional_input_rejected(self, op):
        with pytest.raises(ValueError, match="2-dimensional"):
            op(np.zeros(3))


# overflow-safe magnitude bounds, mirroring the matmul property file: finite
# inputs stay within 1e100 (f64) / ~1e15 (f32) so a reduction over <= 256 rows
# cannot organically overflow, and any non-finite output would trace to a draw
# (none are injected here). 1e15 is not exactly representable in float32, so the
# cap routes through np.float32 to the nearest representable value.
F32_MAX = float(np.float32(1e15))
_F64_ELEMENTS = st.floats(
    min_value=-1e100, max_value=1e100, allow_nan=False, allow_infinity=False,
    allow_subnormal=True,
)
_F32_ELEMENTS = st.floats(
    min_value=-F32_MAX, max_value=F32_MAX, allow_nan=False, allow_infinity=False,
    allow_subnormal=True, width=32,
)
_DIM = st.integers(0, 256)
_AXIS = st.sampled_from((None, 0, 1))


@st.composite
def _drawn_array(draw, dtype, elements):
    m = draw(_DIM)
    n = draw(_DIM)
    return draw(hnp.arrays(dtype, (m, n), elements=elements))


@given(a=_drawn_array(np.float64, _F64_ELEMENTS), axis=_AXIS)
def test_sum_property_float64(a, axis):
    got = pg.sum(a, axis=axis)
    ref = np.sum(a, axis=axis)
    assert_reduce_close(np.asarray(got), np.asarray(ref), a, axis, kind="sum")


@given(a=_drawn_array(np.float32, _F32_ELEMENTS), axis=_AXIS)
def test_sum_property_float32(a, axis):
    got = pg.sum(a, axis=axis)
    ref = np.sum(a, axis=axis)
    assert_reduce_close(np.asarray(got), np.asarray(ref), a, axis, kind="sum")


# the legacy-killer sizes are pinned as square @example draws, generated from
# the list in a loop so a list edit cannot silently leave a pin hole; each pin
# binds every @given parameter, so the property takes plain m, n, seed, dtype
@given(
    m=_DIM,
    n=_DIM,
    seed=st.integers(0, 2**32 - 1),
    dtype=st.sampled_from((np.float32, np.float64)),
)
def test_sum_shapes(m, n, seed, dtype):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, n)).astype(dtype)
    for axis in (None, 0, 1):
        got = pg.sum(a, axis=axis)
        ref = np.sum(a, axis=axis)
        assert_reduce_close(np.asarray(got), np.asarray(ref), a, axis, kind="sum")


for _size in LEGACY_KILLER_SIZES:
    test_sum_shapes = example(m=_size, n=_size, seed=0, dtype=np.float64)(
        test_sum_shapes
    )
