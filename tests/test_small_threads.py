"""The small kernel's two-thread team and its packed-operand paths.

From 2^16 vector FMAs (m * k * n / W) the small kernel splits the rows of C
between two threads on six-row tile boundaries. Every output element stays one
FMA chain in k order, so the bits must not depend on the thread setting, on
which side of the team threshold a shape falls, or on the caller's MXCSR
reaching the worker. The same holds for the paths that repack an operand
before the tiles read it: a column-major B goes through in-register transposes,
and a column-major A whose column pitch crowds a few L1 sets is copied into a
contiguous six-row panel. Each is held bitwise to the scalar twin and to the
C-ordered operands, and exactly to NumPy on small-integer operands.
"""

import numpy as np
import pytest

import peregrine as pg
from conftest import assert_matmul_close

core = pg._core
DTYPES = [np.float64, np.float32]
# Without AVX2, dispatch takes the naive oracle instead of the small kernel.
SMALL_DISPATCH = pg.cpu_features()["avx2"] and pg.cpu_features()["fma"]
_MXCSR_DEFAULT = 0x1F80  # every exception masked, round to nearest
_FTZ_DAZ = 0x8040  # flush-to-zero (bit 15) and denormals-are-zero (bit 6)


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


def ints(rng, shape, dtype):
    return rng.integers(-8, 9, size=shape).astype(dtype)


def normal(rng, shape, dtype):
    return rng.standard_normal(shape).astype(dtype)


def _team_can_grow():
    return core._team_size(1e300, 0.0, 1.0, 2**62) > 1


def vector_fmas(m, k, n, dtype):
    return m * k * n // (4 if dtype == np.float64 else 8)


# Both sides of the 2^16 vector-FMA threshold, row counts off the six-row tile
# grid, the largest cubes the small path takes, and a two-tile-high C.
TEAM_SHAPES = {
    np.float64: [(63, 64, 64), (64, 64, 64), (65, 64, 64), (95, 96, 93), (96, 96, 96), (12, 2000, 96)],
    np.float32: [(80, 80, 80), (81, 81, 81), (95, 96, 93), (96, 96, 96), (12, 3000, 96)],
}


def test_threshold_shapes_straddle_the_team_threshold():
    assert vector_fmas(63, 64, 64, np.float64) < 2**16 <= vector_fmas(64, 64, 64, np.float64)
    assert vector_fmas(80, 80, 80, np.float32) < 2**16 <= vector_fmas(81, 81, 81, np.float32)


@pytest.mark.parametrize("dtype", DTYPES)
def test_thread_setting_does_not_change_small_bits(dtype):
    if not _team_can_grow():
        pytest.skip("OpenMP allows one thread here")
    for m, k, n in TEAM_SHAPES[dtype]:
        rng = np.random.default_rng([m, k, n, 7])
        a, b = normal(rng, (m, k), dtype), normal(rng, (k, n), dtype)
        twin = run(core._gemm_small_naive, a, b)
        for va in (a, np.asfortranarray(a)):
            for vb in (b, np.asfortranarray(b)):
                for t in (1, 2, 3, 6, 1):
                    pg.set_num_threads(t)
                    assert same_bits(run(core._gemm_small, va, vb), twin), (m, k, n, t)
                    if SMALL_DISPATCH and max(m, k, n) <= 96:  # dispatch takes the small path
                        assert same_bits(pg.matmul(va, vb), twin), (m, k, n, t)
        assert_matmul_close(twin, a @ b, a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_team_results_are_exact_on_integer_operands(dtype):
    if not _team_can_grow():
        pytest.skip("OpenMP allows one thread here")
    pg.set_num_threads(6)
    for m, k, n in TEAM_SHAPES[dtype]:
        rng = np.random.default_rng([m, k, n, 8])
        a, b = ints(rng, (m, k), dtype), ints(rng, (k, n), dtype)
        assert np.array_equal(run(core._gemm_small, a, b), a @ b), (m, k, n)


@pytest.mark.parametrize("dtype", DTYPES)
def test_caller_ftz_daz_reaches_the_team_worker(dtype):
    if not _team_can_grow():
        pytest.skip("OpenMP allows one thread here")
    # Products of two values near 2**e are subnormal; every other row of a is
    # lifted so its products stay normal, and both threads' rows mix the kinds.
    m = 64 if dtype == np.float64 else 96
    rng = np.random.default_rng(11)
    e, lift = (-520, 2.0**300) if dtype == np.float64 else (-70, 2.0**40)
    a = rng.uniform(1.0, 2.0, (m, m)) * 2.0**e
    a[::2] *= lift
    a = a.astype(dtype)
    b = (rng.uniform(1.0, 2.0, (m, m)) * 2.0**e).astype(dtype)
    tiny = np.finfo(dtype).tiny
    pg.set_num_threads(2)
    plain = run(core._gemm_small, a, b)
    previous = core._set_mxcsr(_MXCSR_DEFAULT | _FTZ_DAZ)
    try:
        pg.set_num_threads(1)
        one = run(core._gemm_small, a, b)
        pg.set_num_threads(2)
        two = run(core._gemm_small, a, b)
    finally:
        core._set_mxcsr(previous)
    assert np.any((plain[m // 2:] != 0) & (np.abs(plain[m // 2:]) < tiny))
    for r in (one, two):
        assert not np.any((r != 0) & (np.abs(r) < tiny))
    assert same_bits(one, two)


def _column_major_views(x):
    """x column-major, plus column-major with its columns walked backwards."""
    f = np.asfortranarray(x)
    rev = np.asfortranarray(x[:, ::-1])[:, ::-1]
    assert np.array_equal(rev, x) and rev.strides[0] == x.itemsize
    assert x.shape[1] == 1 or rev.strides[1] < 0
    return f, rev


# Column-major A at pitches that crowd L1 (float64 64 and 96 rows, float32 128
# rows), on both sides of the line-count bound (2 * k against 8 ways of the
# sets the pitch reaches) and of the bound on room left in the B buffer behind
# the edge panel, plus 100 rows for a short last tile and pitches that do not
# crowd (float64 32 rows, 65 rows off the 64-byte grid).
PACKED_A_SHAPES = {
    np.float64: [(64, 32, 24), (64, 33, 24), (96, 64, 20), (96, 65, 20), (64, 146, 24), (64, 147, 24),
                 (100, 40, 17), (128, 20, 9), (32, 40, 24), (65, 40, 24), (64, 64, 64)],
    np.float32: [(128, 32, 40), (128, 33, 40), (128, 186, 40), (128, 187, 40), (130, 40, 33), (64, 64, 64)],
}


@pytest.mark.parametrize("dtype", DTYPES)
def test_column_major_a_packs_to_the_same_bits(dtype):
    for m, k, n in PACKED_A_SHAPES[dtype]:
        rng = np.random.default_rng([m, k, n, 9])
        a, b = normal(rng, (m, k), dtype), normal(rng, (k, n), dtype)
        ref = run(core._gemm_small_naive, a, b)
        assert same_bits(run(core._gemm_small, a, b), ref), (m, k, n)
        for va in _column_major_views(a):
            for vb in (b, np.asfortranarray(b)):
                assert same_bits(run(core._gemm_small, va, vb), ref), (m, k, n, va.strides, vb.strides)
        ia, ib = ints(rng, (m, k), dtype), ints(rng, (k, n), dtype)
        assert np.array_equal(run(core._gemm_small, np.asfortranarray(ia), ib), ia @ ib), (m, k, n)


# Column-major B on every k remainder of the 4 x 4 (float64) and 8 x 8
# (float32) transposes, full and partial panels, and every row count that takes
# the two-vector panel, from four-row tiles up.
B_M = [4, 5, 6, 7, 13]
B_K = {np.float64: [1, 3, 4, 5, 7, 8, 9, 31, 33], np.float32: [1, 7, 8, 9, 15, 16, 17, 33]}
B_N = {np.float64: [5, 8, 9, 15, 16, 17, 33], np.float32: [9, 16, 17, 31, 32, 33]}


@pytest.mark.parametrize("dtype", DTYPES)
def test_column_major_b_packs_to_the_same_bits(dtype):
    rng = np.random.default_rng(12)
    for m in B_M:
        for k in B_K[dtype]:
            for n in B_N[dtype]:
                a, b = normal(rng, (m, k), dtype), normal(rng, (k, n), dtype)
                ref = run(core._gemm_small_naive, a, b)
                for vb in _column_major_views(b):
                    assert same_bits(run(core._gemm_small, a, vb), ref), (m, k, n, vb.strides)
                ia, ib = ints(rng, (m, k), dtype), ints(rng, (k, n), dtype)
                assert np.array_equal(run(core._gemm_small, ia, np.asfortranarray(ib)), ia @ ib), (m, k, n)


# Several k blocks, partial sums carried through C between them. A column-major
# A at a crowding pitch (512 bytes: float64 64 rows, float32 128 rows) has no
# room for a packed tile behind its 256-step blocks but has it in the shorter
# last block, so one product mixes in-place and packed A tiles; the team splits
# the float64 rows too. Four-row C takes the shallow 32-step blocks.
MULTI_BLOCK = {
    np.float64: [(64, 600, 40), (64, 513, 24), (4, 300, 64), (12, 600, 40), (6, 257, 16)],
    np.float32: [(128, 600, 40), (128, 513, 24), (4, 300, 64), (13, 520, 49), (6, 257, 16)],
}


@pytest.mark.parametrize("dtype", DTYPES)
def test_k_blocks_mix_in_place_and_packed_a_tiles(dtype):
    pg.set_num_threads(6)
    for m, k, n in MULTI_BLOCK[dtype]:
        rng = np.random.default_rng([m, k, n, 13])
        a, b = normal(rng, (m, k), dtype), normal(rng, (k, n), dtype)
        ref = run(core._gemm_small_naive, a, b)
        for va in (a, *_column_major_views(a)):
            for vb in (b, np.asfortranarray(b)):
                assert same_bits(run(core._gemm_small, va, vb), ref), (m, k, n, va.strides, vb.strides)
        assert_matmul_close(ref, a @ b, a, b)
        ia, ib = ints(rng, (m, k), dtype), ints(rng, (k, n), dtype)
        assert np.array_equal(run(core._gemm_small, np.asfortranarray(ia), ib), ia @ ib), (m, k, n)
