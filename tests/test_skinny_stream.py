"""The streaming skinny kernels, through the private hooks and pg.matmul.

Once the operand a skinny product streams holds 12 MiB or more, gemm_skinny
leaves the small kernel for two DRAM kernels: A x (n == 1, A's rows
unit-stride), four rows of A at a time in vector lanes, and a short A times a
long B (m <= 4 < n, B's rows unit-stride), in k blocks of 8 rows over tiles one
64-byte line of a B row wide. Both compute every element as the same single
FMA chain in k order from +0.0, stored as acc - (-0.0), so their bits must
equal the scalar twin's and the small kernel's at every remainder they add:
rows past a multiple of 4, k past a multiple of 4 (the quad) and of 8 (the
block), columns past a tile, the last k block, both sides of the 12 MiB
threshold, 1 and 6 threads, and the strided layouts they read in place.
Normal operands make the bitwise comparisons sensitive to the summation order;
small-integer operands, whose sums are exact, pin the values against NumPy.
"""

import numpy as np
import pytest

import peregrine as pg
from conftest import assert_matmul_close

core = pg._core
DTYPES = [np.float64, np.float32]
# Without AVX2 and FMA, matmul takes the naive kernel, whose rounding differs.
AVX2 = pg.cpu_features()["avx2"] and pg.cpu_features()["fma"]
STREAM_BYTES = 12 * 1024 * 1024


@pytest.fixture(autouse=True)
def _restore_thread_setting():
    previous = pg.get_num_threads()
    yield
    pg.set_num_threads(previous)


def run(f, a, b):
    # NaN-filled, so an element the kernel never wrote cannot pass as a result
    out = np.full((a.shape[0], b.shape[1]), np.nan, dtype=a.dtype)
    assert f(a, b, out) is None
    return out


def same_bits(x, y):
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def span(dtype, k):
    """The long dimension at which a k-deep operand reaches the threshold."""
    return -(-STREAM_BYTES // (k * np.dtype(dtype).itemsize))


def threads_available():
    return core._team_size(1e300, 0.0, 1.0, 2**62) > 1


def check(a, b, threads=(1, 6)):
    """The streaming kernel at each team size, the twin, and the small kernel agree bitwise."""
    pg.set_num_threads(threads[0])
    got = run(core._gemm_skinny, a, b)
    for t in threads[1:] if threads_available() else ():
        pg.set_num_threads(t)
        assert same_bits(run(core._gemm_skinny, a, b), got), f"{t} threads differ on {a.shape} @ {b.shape}"
    assert same_bits(run(core._gemm_skinny_naive, a, b), got), f"twin differs on {a.shape} @ {b.shape}"
    assert same_bits(run(core._gemm_small, a, b), got), f"small kernel differs on {a.shape} @ {b.shape}"
    return got


# k = 3 runs only the k tail, 6 one quad and a tail, 1027 and 4096 full rows;
# the four row counts from the threshold up cover every remainder of 4.
@pytest.mark.parametrize("k", [3, 6, 1027, 4096])
@pytest.mark.parametrize("dtype", DTYPES)
def test_matvec_every_row_and_k_remainder(dtype, k):
    rows = span(dtype, k) + 3
    rng = np.random.default_rng([rows, k])
    a = rng.standard_normal((rows, k)).astype(dtype)
    x = rng.standard_normal((k, 1)).astype(dtype)
    for m in range(rows - 3, rows + 1):
        got = check(a[:m], x)
        assert_matmul_close(got, a[:m] @ x, a[:m], x)
    a = rng.integers(-8, 9, (rows, k)).astype(dtype)
    x = rng.integers(-8, 9, (k, 1)).astype(dtype)
    assert np.array_equal(run(core._gemm_skinny, a, x), a @ x)


@pytest.mark.parametrize("dtype", DTYPES)
def test_matvec_threshold(dtype):
    k = 4096
    rows = span(dtype, k)
    assert rows * k * np.dtype(dtype).itemsize == STREAM_BYTES
    rng = np.random.default_rng(7)
    a = rng.standard_normal((rows, k)).astype(dtype)
    x = rng.standard_normal((k, 1)).astype(dtype)
    below = check(a[: rows - 1], x)
    at = check(a, x)
    assert same_bits(below, at[: rows - 1])


@pytest.mark.parametrize("dtype", DTYPES)
def test_matvec_layouts(dtype):
    k = 1027
    rows = span(dtype, k) + 2
    rng = np.random.default_rng(11)
    a = rng.standard_normal((rows, k)).astype(dtype)
    x = rng.standard_normal((k, 1)).astype(dtype)
    ref = check(a, x)
    padded = np.zeros((rows, k + 5), dtype)
    padded[:, :k] = a
    strided_x = np.zeros((3 * k, 1), dtype)
    strided_x[::3] = x
    layouts_a = {"C": a, "padded": padded[:, :k], "neg_rows": np.ascontiguousarray(a[::-1])[::-1]}
    layouts_x = {"C": x, "strided": strided_x[::3], "negative": np.ascontiguousarray(x[::-1])[::-1]}
    for name_a, va in layouts_a.items():
        for name_x, vx in layouts_x.items():
            assert np.array_equal(va, a) and np.array_equal(vx, x)
            for f in (core._gemm_skinny, core._gemm_skinny_naive):
                assert same_bits(run(f, va, vx), ref), (f.__name__, name_a, name_x)
    if AVX2:
        assert same_bits(pg.matmul(a, x), ref)


# k = 5 is one short block with nothing to read ahead, 8 exactly one block,
# 9 a block and a one-row block, 27 three blocks and a remainder. The column
# counts past a multiple of 64 leave the last tile 0, 1, 9, or 15 columns
# short of the 16 float32 or 8 float64 columns of a full one.
@pytest.mark.parametrize("k", [5, 8, 9, 27])
@pytest.mark.parametrize("dtype", DTYPES)
def test_panel_every_m_k_and_column_remainder(dtype, k):
    n0 = -(-span(dtype, k) // 64) * 64
    rng = np.random.default_rng([n0, k])
    a = rng.standard_normal((4, k)).astype(dtype)
    b = rng.standard_normal((k, n0 + 15)).astype(dtype)
    for m in (1, 2, 3, 4):
        for extra in (0, 1, 9, 15):
            vb = b if extra == 15 else b[:, : n0 + extra]
            got = check(a[:m], vb)
            if extra == 9:
                assert_matmul_close(got, a[:m] @ vb, a[:m], vb)
    a = rng.integers(-8, 9, (4, k)).astype(dtype)
    b = rng.integers(-8, 9, (k, n0 + 15)).astype(dtype)
    assert np.array_equal(run(core._gemm_skinny, a, b), a @ b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_panel_threshold(dtype):
    k = 8
    n = span(dtype, k)
    assert n * k * np.dtype(dtype).itemsize == STREAM_BYTES
    rng = np.random.default_rng(13)
    a = rng.standard_normal((3, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    below = check(a, np.ascontiguousarray(b[:, : n - 1]))
    at = check(a, b)
    assert same_bits(below, at[:, : n - 1])


@pytest.mark.parametrize("dtype", DTYPES)
def test_panel_layouts(dtype):
    k = 27
    n = span(dtype, k) + 9
    rng = np.random.default_rng(17)
    a = rng.standard_normal((4, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    ref = check(a, b)
    big_a = np.zeros((9, 3 * k), dtype)
    big_a[1::2, ::3] = a
    layouts_a = {"C": a, "F": np.asfortranarray(a), "strided": big_a[1::2, ::3]}
    layouts_b = {"C": b, "neg_rows": np.ascontiguousarray(b[::-1])[::-1]}
    for name_a, va in layouts_a.items():
        for name_b, vb in layouts_b.items():
            assert np.array_equal(va, a) and np.array_equal(vb, b)
            for f in (core._gemm_skinny, core._gemm_skinny_naive):
                assert same_bits(run(f, va, vb), ref), (f.__name__, name_a, name_b)
    if AVX2:
        assert same_bits(pg.matmul(a, b), ref)


def _streaming_shapes(dtype):
    return [(span(dtype, 1027) + 3, 1027, 1), (4, 27, span(dtype, 27) + 9)]


@pytest.mark.parametrize("dtype", DTYPES)
def test_nan_and_inf_propagate_like_numpy(dtype):
    for m, k, n in _streaming_shapes(dtype):
        rng = np.random.default_rng([m, k, n])
        a = rng.standard_normal((m, k)).astype(dtype)
        b = rng.standard_normal((k, n)).astype(dtype)
        for x in (a, b):
            flat = x.reshape(-1)
            flat[rng.choice(flat.size, size=3, replace=False)] = [np.nan, np.inf, -np.inf]
        with np.errstate(invalid="ignore"):  # NumPy and the f32 truth also meet inf - inf
            got = run(core._gemm_skinny, a, b)
            assert_matmul_close(got, a @ b, a, b)
        other = run(core._gemm_skinny_naive, a, b)
        nan = np.isnan(got)
        # a NaN's payload may come from either FMA operand, so only its place is held equal
        assert np.array_equal(nan, np.isnan(other))
        assert got[~nan].tobytes() == other[~nan].tobytes()


@pytest.mark.parametrize("dtype", DTYPES)
def test_underflowed_negative_zero_lands_positive(dtype):
    # Every product is negative and far below the smallest subnormal, so each
    # chain stays at -0.0; the final - (-0.0) must turn it into +0.0.
    tiny = 1e-200 if dtype == np.float64 else 1e-30
    for m, k, n in _streaming_shapes(dtype):
        rng = np.random.default_rng([m, k, n])
        a = (-tiny * rng.uniform(1.0, 2.0, (m, k))).astype(dtype)
        b = (tiny * rng.uniform(1.0, 2.0, (k, n))).astype(dtype)
        assert np.signbit(a[0, 0] * b[0, 0])
        got = run(core._gemm_skinny, a, b)
        assert not got.view(np.uint64 if dtype == np.float64 else np.uint32).any()
