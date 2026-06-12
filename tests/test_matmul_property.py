"""Property-based proof that matmul matches NumPy across the input space.

The unit suite samples points; these properties sample the space. Each
property class exists for a distinct failure mode:

- value properties (elementwise-drawn arrays, both dtypes): denormals, signed
  zeros, and boundary magnitudes that seeded normal data never produces.
  Shapes stay at or below 16x16 because elementwise drawing is expensive;
  shape coverage lives elsewhere.
- shape property (rng content, dims in [0, 256] including empty operands):
  dimension and tail handling. The legacy kernel crashed or silently
  corrupted results at sizes not divisible by four, so every legacy-killer
  size (1, 3, 7, 50, 250, 333, 750) is pinned as an explicit square example
  that runs before any random draw, on every session.
- slow-marked spike (dims in [257, 1024]): blocking and threading boundaries
  at sizes the quick loop never reaches.
- special values: NaN and signed Inf injected at drawn positions must land
  exactly where NumPy lands them, signs included.
- layouts: C-contiguous, Fortran, strided, and transposed views must produce
  the same answer as NumPy does for the identical view.
- promotion: mixed float32 x float64 operands must land on np.result_type.

Magnitude bounds on drawn elements exist for intermediate-overflow safety:
finite inputs stay within 1e100 (float64) or about 1e15 (float32), so the
worst intermediate is k * amax^2, at most 1024 * 1e200 for float64 (far below
the 1.8e308 ceiling) and 1024 * 1e30 for float32 (below 3.4e38). An Inf in a
result can therefore only come from an injected special, never organically,
which is what lets the tolerance contract treat non-finite values as exact.
"""

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import fastmathext as fme
from conftest import assert_matmul_close

# the regression fence: the @example pins on test_shapes mirror this list
# entry for entry; update both together or not at all
LEGACY_KILLER_SIZES = [1, 3, 7, 50, 250, 333, 750]

# st.floats(width=32) rejects bounds that are not exactly representable in
# float32, and 1e15 is not; routing the cap through np.float32 yields the
# nearest representable value (999999986991104.0)
F32_MAX = float(np.float32(1e15))

_F64_ELEMENTS = st.floats(
    min_value=-1e100,
    max_value=1e100,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=True,
)
_F32_ELEMENTS = st.floats(
    min_value=-F32_MAX,
    max_value=F32_MAX,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=True,
    width=32,
)

_DIM = st.integers(0, 256)
_SEED = st.integers(0, 2**32 - 1)
_DTYPE = st.sampled_from((np.float32, np.float64))
_LAYOUT = st.sampled_from(("c", "f", "strided", "transposed"))


@st.composite
def _drawn_operands(draw, dtype, elements):
    # elementwise drawing is what surfaces denormals, signed zeros, and
    # boundary magnitudes, but it costs about 2 s per example at 256x256,
    # so dims stay at or below 16 here and test_shapes owns large shapes
    m = draw(st.integers(0, 16))
    k = draw(st.integers(0, 16))
    n = draw(st.integers(0, 16))
    a = draw(hnp.arrays(dtype, (m, k), elements=elements))
    b = draw(hnp.arrays(dtype, (k, n), elements=elements))
    return a, b


def _rng_operands(m, k, n, seed, dtype):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    return a, b


@given(operands=_drawn_operands(np.float64, _F64_ELEMENTS))
def test_values_float64(operands):
    a, b = operands
    got = fme.matmul(a, b)
    ref = a @ b
    assert_matmul_close(got, ref, a, b)


@given(operands=_drawn_operands(np.float32, _F32_ELEMENTS))
def test_values_float32(operands):
    a, b = operands
    got = fme.matmul(a, b)
    ref = a @ b
    assert_matmul_close(got, ref, a, b)


# explicit examples run first and do not count toward max_examples; each
# @example must bind every @given parameter, which is why this property
# takes exactly these five plain parameters and draws nothing interactively
@given(m=_DIM, k=_DIM, n=_DIM, seed=_SEED, dtype=_DTYPE)
@example(m=1, k=1, n=1, seed=0, dtype=np.float64)
@example(m=3, k=3, n=3, seed=0, dtype=np.float64)
@example(m=7, k=7, n=7, seed=0, dtype=np.float64)
@example(m=50, k=50, n=50, seed=0, dtype=np.float64)
@example(m=250, k=250, n=250, seed=0, dtype=np.float64)
@example(m=333, k=333, n=333, seed=0, dtype=np.float64)
@example(m=750, k=750, n=750, seed=0, dtype=np.float64)
def test_shapes(m, k, n, seed, dtype):
    a, b = _rng_operands(m, k, n, seed, dtype)
    got = fme.matmul(a, b)
    ref = a @ b
    assert_matmul_close(got, ref, a, b)


# all three dims draw large on purpose: joint-large coverage costs about
# 11 s per 200 examples on the dev machine, affordable once slow-marked
@pytest.mark.slow
@given(
    m=st.integers(257, 1024),
    k=st.integers(257, 1024),
    n=st.integers(257, 1024),
    seed=_SEED,
    dtype=_DTYPE,
)
def test_shapes_spike(m, k, n, seed, dtype):
    a, b = _rng_operands(m, k, n, seed, dtype)
    got = fme.matmul(a, b)
    ref = a @ b
    assert_matmul_close(got, ref, a, b)


@st.composite
def _special_value_operands(draw):
    # finite base content plus NaN and signed Inf injected at drawn
    # positions in either operand; the bounded base magnitudes mean every
    # non-finite output traces back to an injection
    m = draw(st.integers(1, 32))
    k = draw(st.integers(1, 32))
    n = draw(st.integers(1, 32))
    rng = np.random.default_rng(draw(_SEED))
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    for _ in range(draw(st.integers(1, 4))):
        special = draw(st.sampled_from((np.nan, np.inf, -np.inf)))
        if draw(st.booleans()):
            a[draw(st.integers(0, m - 1)), draw(st.integers(0, k - 1))] = special
        else:
            b[draw(st.integers(0, k - 1)), draw(st.integers(0, n - 1))] = special
    return a, b


@given(operands=_special_value_operands())
def test_special_values_propagate(operands):
    a, b = operands
    got = fme.matmul(a, b)
    ref = a @ b
    # the helper's special path asserts NaN positions, Inf positions, and
    # each Inf's sign match NumPy, then bounds the mutually finite remainder
    assert_matmul_close(got, ref, a, b)


def _with_layout(rng, rows, cols, layout):
    # each branch returns a (rows, cols) array; ref is computed from the
    # identical view, so only stride handling is under test, never values
    if layout == "c":
        return rng.standard_normal((rows, cols))
    if layout == "f":
        return np.asfortranarray(rng.standard_normal((rows, cols)))
    if layout == "strided":
        return rng.standard_normal((2 * rows, cols))[::2]
    return rng.standard_normal((cols, rows)).T


@given(
    m=st.integers(1, 64),
    k=st.integers(1, 64),
    n=st.integers(1, 64),
    layout_a=_LAYOUT,
    layout_b=_LAYOUT,
    seed=_SEED,
)
def test_layouts(m, k, n, layout_a, layout_b, seed):
    rng = np.random.default_rng(seed)
    a = _with_layout(rng, m, k, layout_a)
    b = _with_layout(rng, k, n, layout_b)
    got = fme.matmul(a, b)
    ref = a @ b
    assert_matmul_close(got, ref, a, b)


@given(
    m=st.integers(1, 8),
    k=st.integers(1, 8),
    n=st.integers(1, 8),
    dtype_a=_DTYPE,
    dtype_b=_DTYPE,
    seed=_SEED,
)
def test_promotion(m, k, n, dtype_a, dtype_b, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k)).astype(dtype_a)
    b = rng.standard_normal((k, n)).astype(dtype_b)
    got = fme.matmul(a, b)
    ref = a @ b
    # NumPy already lands float pairs on result_type, so ref needs no cast;
    # integer promotion diverges by design (int becomes float64 here, stays
    # int64 in NumPy) and is pinned in the unit suite, not as a property
    assert got.dtype == np.result_type(dtype_a, dtype_b)
    assert_matmul_close(got, ref, a, b)
