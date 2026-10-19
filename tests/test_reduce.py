"""sum and mean: the summation order is fixed by the shape, so the bits never
depend on the thread count or the instruction set.

Three orders are pinned exactly, through Python models of them:
- axis=None and axis=1 sum each contiguous run pairwise (the v3.0.0 order:
  runs of up to 128 elements through eight lanes combined as ((l0 + l1) +
  (l2 + l3)) + ((l4 + l5) + (l6 + l7)) plus the tail in order, longer runs
  split at (n // 2) rounded down to a multiple of 8);
- axis=0 sums blocks of 128 rows sequentially from +0.0 and adds the blocks
  pairwise (block range [b0, b1) splits at b0 + (b1 - b0) // 2). Up to 128
  rows that is the sequential order of v3.0.0.
The AVX2 kernels and their scalar twin (a PEREGRINE_DISABLE_AVX2 child) must
agree bit for bit, every thread count must give the same bits, and the results
stay within the reduction tolerance contract of NumPy. mean is the sum divided
by the reduced count converted to the dtype, in the dtype.
"""

import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import peregrine as pg
from conftest import assert_reduce_close

DTYPES = ["float32", "float64"]
AXES = [None, 0, 1]
THREADS = [1, 2, 3, 6]

# Every shape class, at sizes past the threading threshold where it has one:
# the 16-row and 7-row arrays cut each row into subtrees, the tall ones cut the
# axis-0 block tree into chunks, and the odd sizes leave a remainder at every
# level (lane tails, short last blocks, uneven chunks).
SHAPES = {
    "tiny": [(1, 1), (3, 7), (7, 3)],
    "square": [(64, 64), (1000, 1000)],
    "short-wide": [(16, 1_000_000)],
    "tall-narrow": [(1_000_000, 16)],
    "legacy-odd": [(333, 333), (750, 750), (7, 333_333), (100_003, 13)],
}
CASES = [(cls, shape) for cls, shapes in SHAPES.items() for shape in shapes]


def _build(m, n, dtype, seed, special=False):
    a = np.random.default_rng(seed).standard_normal((m, n)).astype(dtype)
    if special:
        a.flat[::1009] = np.inf
        a.flat[503::2003] = -np.inf
        a.flat[17::4001] = np.nan
    return a


def _pairwise(x):
    """v3.0.0's pairwise sum of a 1-D array, in its dtype."""
    n = x.size
    if n <= 128:
        blocked = n & ~7
        lanes = np.zeros(8, x.dtype)
        if blocked:
            # accumulate runs each lane sequentially, from the +0.0 row
            lanes = np.add.accumulate(np.vstack([lanes, x[:blocked].reshape(-1, 8)]), axis=0)[-1]
        s = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        for v in x[blocked:]:
            s = s + v
        return s
    h = (n // 2) & ~7
    left = _pairwise(x[:h])
    return left + _pairwise(x[h:])


def _axis0(a):
    """Blocks of 128 rows summed sequentially from +0.0, added pairwise."""
    m, n = a.shape
    if m == 0:
        return np.zeros(n, a.dtype)

    def node(b0, b1):
        if b1 - b0 == 1:
            rows = a[b0 * 128 : min(m, b1 * 128)]
            return np.add.accumulate(np.vstack([np.zeros((1, n), a.dtype), rows]), axis=0)[-1]
        mid = b0 + (b1 - b0) // 2
        left = node(b0, mid)
        return left + node(mid, b1)

    return node(0, -(-m // 128))


def _model(a, axis):
    if axis is None:
        return _pairwise(a.ravel())
    if axis == 1:
        return np.array([_pairwise(row) for row in a], a.dtype).reshape(a.shape[0])
    return _axis0(a)


def _bits(r):
    r = np.asarray(r)
    return r.dtype.str, r.shape, r.tobytes()


@pytest.fixture(autouse=True)
def _restore_thread_setting():
    previous = pg.get_num_threads()
    yield
    pg.set_num_threads(previous)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cls, shape", CASES, ids=[f"{c}-{s[0]}x{s[1]}" for c, s in CASES])
def test_thread_count_never_changes_bits(cls, shape, dtype):
    a = _build(*shape, dtype, seed=shape[0] * 7 + shape[1])
    for op in (pg.sum, pg.mean):
        for axis in AXES:
            pg.set_num_threads(THREADS[0])
            ref = _bits(op(a, axis=axis))
            for t in THREADS[1:]:
                pg.set_num_threads(t)
                assert _bits(op(a, axis=axis)) == ref, (op.__name__, axis, t)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cls, shape", CASES, ids=[f"{c}-{s[0]}x{s[1]}" for c, s in CASES])
def test_close_to_numpy(cls, shape, dtype):
    a = _build(*shape, dtype, seed=shape[0] + 3 * shape[1])
    for kind, op, ref_op in (("sum", pg.sum, np.sum), ("mean", pg.mean, np.mean)):
        for axis in AXES:
            got = np.asarray(op(a, axis=axis))
            assert_reduce_close(got, np.asarray(ref_op(a, axis=axis)), a, axis, kind=kind)


# The pairwise model is slow in Python, so these sizes stay moderate while still
# reaching every path: one leaf, a split, leaf groups with remainders, and runs
# cut into 2, 16, and 64 subtrees (1_000_003 elements); rows from 1 to 129 and
# long rows cut into subtrees.
PAIRWISE_SHAPES = [
    (1, 1), (1, 7), (1, 8), (1, 127), (1, 128), (1, 129), (1, 1000), (3, 4097),
    (1, 16_387), (1, 131_077), (1, 1_000_003), (37, 17), (5, 129), (2, 3001),
    (9, 50_000), (1000, 3),
]
# Above 12 MiB the axis-0 kernel reads the rows of a block in pairs and
# prefetches the next pair (below it, in fours): the last four shapes are over
# that size in both dtypes, with blocks of 2, 3, and 5 rows (the row left after
# the pairs, and both sides of the prefetch) and a short last column panel.
AXIS0_SHAPES = [
    (1, 9), (128, 37), (129, 37), (1000, 3), (5000, 70), (20_000, 5), (256, 1100),
    (2, 1_600_003), (5, 700_001), (131, 25_003), (133, 25_003),
]


@pytest.mark.parametrize("threads", [1, 6])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", PAIRWISE_SHAPES)
def test_pairwise_bits_are_v300s(shape, dtype, threads):
    pg.set_num_threads(threads)
    a = _build(*shape, dtype, seed=11)
    for axis in (None, 1):
        assert _bits(pg.sum(a, axis=axis)) == _bits(np.asarray(_model(a, axis), dtype)), axis


@pytest.mark.parametrize("threads", [1, 6])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", AXIS0_SHAPES)
def test_axis0_bits_are_the_blocked_order(shape, dtype, threads):
    pg.set_num_threads(threads)
    a = _build(*shape, dtype, seed=13)
    assert _bits(pg.sum(a, axis=0)) == _bits(_model(a, 0))


@pytest.mark.parametrize("dtype", DTYPES)
def test_axis0_up_to_128_rows_is_the_sequential_order(dtype):
    # v3.0.0's (and NumPy's) column order: a running sum down each column
    a = _build(128, 1000, dtype, seed=17)
    seq = np.add.accumulate(np.vstack([np.zeros((1, 1000), dtype), a]), axis=0)[-1]
    assert _bits(pg.sum(a, axis=0)) == _bits(seq)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cls, shape", CASES, ids=[f"{c}-{s[0]}x{s[1]}" for c, s in CASES])
def test_mean_divides_the_sum_in_dtype(cls, shape, dtype):
    a = _build(*shape, dtype, seed=19)
    for axis in AXES:
        count = a.size if axis is None else a.shape[axis]
        expect = np.asarray(pg.sum(a, axis=axis)) / np.asarray(count, dtype)
        assert _bits(pg.mean(a, axis=axis)) == _bits(expect), axis


@pytest.mark.parametrize("axis", [0, 1])
def test_mean_over_axis_divides_by_the_rounded_float32_count(axis):
    # 2**24 + 1 rounds in float32, so a division in double would differ
    shape = (2**24 + 1, 1) if axis == 0 else (1, 2**24 + 1)
    a = np.ones(shape, np.float32)
    total = pg.sum(a, axis=axis)
    got = pg.mean(a, axis=axis)
    assert _bits(got) == _bits(total / np.float32(2**24 + 1))
    assert got[0] != np.float32(float(total[0]) / (2**24 + 1))


TWIN_SCRIPT = (
    inspect.getsource(_build)
    + r"""
import json
import sys

import numpy as np

import peregrine as pg

assert pg.cpu_features()["avx2"] is False, pg.cpu_features()
out = {}
for key, m, n, dtype, seed, special in json.loads(sys.argv[2]):
    a = _build(m, n, dtype, seed, special)
    for op in ("sum", "mean"):
        for axis in (None, 0, 1):
            out[f"{key}-{op}-{axis}"] = np.asarray(getattr(pg, op)(a, axis=axis))
np.savez(sys.argv[1], **out)
"""
)

# Sizes that still reach every path (row subtrees, axis-0 chunks, leaf groups
# with remainders) without regenerating the largest arrays in the child.
TWIN_CASES = [
    ("tiny", 3, 7), ("square", 1000, 1000), ("short-wide", 16, 200_000), ("tall-narrow", 200_000, 16),
    ("odd-rows", 7, 333_333), ("odd-tall", 100_003, 13), ("legacy", 333, 333), ("blocks", 5000, 70),
    ("leaf", 1, 129),
]


def test_avx2_and_scalar_twin_agree_bitwise(tmp_path):
    if not pg.cpu_features()["avx2"]:
        pytest.skip("this process runs the scalar twin too (no AVX2, or PEREGRINE_DISABLE_AVX2)")
    specs = [
        [f"{name}-{dt}-{special}", m, n, dt, m + n, special]
        for name, m, n in TWIN_CASES
        for dt in DTYPES
        for special in (False, True)
    ]
    path = tmp_path / "twin.npz"
    env = dict(os.environ, PEREGRINE_DISABLE_AVX2="1")
    env.pop("PEREGRINE_NUM_THREADS", None)
    p = subprocess.run(
        [sys.executable, "-c", TWIN_SCRIPT, str(path), json.dumps(specs)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert p.returncode == 0, p.stderr
    twin = np.load(path)
    for key, m, n, dt, seed, special in specs:
        a = _build(m, n, dt, seed, special)
        for op in ("sum", "mean"):
            for axis in AXES:
                got = np.asarray(getattr(pg, op)(a, axis=axis))
                ref = twin[f"{key}-{op}-{axis}"]
                if special:
                    # which of two NaNs survives an add is not specified
                    np.testing.assert_array_equal(got, ref, err_msg=f"{key} {op} {axis}")
                else:
                    assert _bits(got) == _bits(ref), (key, op, axis)


@pytest.mark.parametrize("threads", [1, 6])
@pytest.mark.parametrize("dtype", DTYPES)
def test_nan_and_inf_follow_numpy_across_subtrees(dtype, threads):
    # +inf and -inf land in different subtrees and blocks, so they meet only
    # where partial sums combine: inf + -inf is nan there, as in NumPy
    pg.set_num_threads(threads)
    a = _build(1000, 1000, dtype, seed=23)
    a[0, 0] = np.inf
    a[999, 999] = -np.inf
    a[999, 0] = -np.inf
    a[0, 999] = np.inf
    a[500, 3] = np.nan
    a[3, 700] = np.inf
    for kind, op, ref_op in (("sum", pg.sum, np.sum), ("mean", pg.mean, np.mean)):
        for axis in AXES:
            got = np.asarray(op(a, axis=axis))
            with np.errstate(invalid="ignore"):
                ref = np.asarray(ref_op(a, axis=axis))
            assert_reduce_close(got, ref, a, axis, kind=kind)
    assert np.isnan(pg.sum(a))
    assert np.isnan(pg.sum(a, axis=0)[[0, 3, 999]]).all()
    assert np.isinf(pg.sum(a, axis=0)[700])


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_sums_match_numpy_exactly(shape, dtype):
    a = np.zeros(shape, dtype)
    for axis in AXES:
        got = np.asarray(pg.sum(a, axis=axis))
        ref = np.asarray(np.sum(a, axis=axis))
        assert got.dtype == ref.dtype and got.shape == ref.shape
        np.testing.assert_array_equal(got, ref)
        assert not np.signbit(got).any()


_MXCSR_DEFAULT = 0x1F80  # every exception masked, round to nearest
_FTZ_DAZ = 0x8040  # flush-to-zero (bit 15) and denormals-are-zero (bit 6)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(2000, 3000), (16, 200_000), (200_000, 16)])
def test_caller_ftz_daz_reaches_every_worker(shape, dtype, axis):
    # Positive subnormals, with every other row lifted to normal values just
    # above the subnormal range, close enough that the subnormals still move
    # every sum (a larger lift would absorb them and prove nothing). DAZ reads
    # the subnormals as zero and the sums of the normal values stay normal, so
    # under the caller's FTZ/DAZ the result is the sum of the array with them
    # zeroed, at any team size; a worker keeping its own MXCSR would add them in.
    # The exponents keep a worker's partial sum of subnormals normal, so the
    # caller's DAZ cannot zero it again when the partials combine.
    rng = np.random.default_rng(29)
    e, lift = (-140, 2.0**20) if dtype == "float32" else (-1030, 2.0**40)
    a = rng.uniform(1.0, 2.0, shape) * 2.0**e
    a[::2] *= lift
    a = a.astype(dtype)
    tiny = np.finfo(dtype).tiny
    zeroed = np.where(a < tiny, 0, a).astype(dtype)
    pg.set_num_threads(6)
    plain = np.asarray(pg.sum(a, axis=axis))
    expect = np.asarray(pg.sum(zeroed, axis=axis))
    assert _bits(plain) != _bits(expect)  # the subnormals count without DAZ
    previous = pg._core._set_mxcsr(_MXCSR_DEFAULT | _FTZ_DAZ)
    try:
        pg.set_num_threads(1)
        one = np.asarray(pg.sum(a, axis=axis))
        pg.set_num_threads(6)
        six = np.asarray(pg.sum(a, axis=axis))
    finally:
        pg._core._set_mxcsr(previous)
    assert _bits(one) == _bits(expect)
    assert _bits(six) == _bits(expect)
