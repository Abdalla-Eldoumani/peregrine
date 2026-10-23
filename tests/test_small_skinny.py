"""The small and skinny GEMM kernels, through their private hooks.

Both kernels compute every output element as one FMA chain in k order starting
from +0.0 and store acc - (-0.0). The result therefore depends only on the
operand values, and the suite holds it bitwise equal across everything that
must not change it: the AVX2 kernel and its scalar twin, the two kernels, every
operand layout (C, F, transposed, strided, negative strides, odd strides on a
length-1 dimension), every team size, and the packed engine wherever k fits in
its first k panel. Against NumPy, small-integer operands give exact sums, so the
values must match exactly; random normal operands go through
assert_matmul_close.

_gemm_small and _gemm_skinny run the AVX2 kernel when the CPU has it and the
twin otherwise, so under PEREGRINE_DISABLE_AVX2=1 this whole file runs the
twins, and the subprocess test below checks that they reproduce the AVX2 bits.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import peregrine as pg
from conftest import assert_matmul_close

core = pg._core
DTYPES = [np.float64, np.float32]
AVX2 = pg.cpu_features()["avx2"] and pg.cpu_features()["fma"]
KERNELS = {
    "small": (core._gemm_small, core._gemm_small_naive),
    "skinny": (core._gemm_skinny, core._gemm_skinny_naive),
}
CUBE = list(range(1, 25)) + [31, 32, 33, 63, 64, 65, 95, 96, 97]
EDGES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 33, 63, 64, 65, 96, 97]
LEGACY_KILLER_SIZES = [1, 3, 7, 50, 250, 333, 750]
SKINNY_SHAPES = [
    (1, 1, 100000), (1, 5, 100000), (1, 100000, 1), (1, 100000, 9), (1, 300, 1000), (1, 257, 33),
    (100000, 1, 1), (100000, 5, 1), (9, 100000, 1), (1000, 300, 1), (33, 257, 1),
    (4, 3, 100000), (4, 300, 5000), (4, 100000, 6), (4, 33, 97),
    (100000, 3, 4), (5000, 300, 4), (6, 100000, 4), (97, 33, 4),
    (1000, 1, 1000), (1000, 2, 999), (999, 3, 1001), (1000, 4, 1000), (7, 3, 100000), (100000, 4, 7),
    (2, 3, 5), (3, 2, 1), (4, 4, 4),
]
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


def check_all(a, b, ref=None):
    """Every kernel and twin on (a, b) gives the same bits; returns them."""
    got = run(core._gemm_small, a, b)
    for fast, twin in KERNELS.values():
        for f in (fast, twin):
            r = got if f is core._gemm_small else run(f, a, b)
            assert same_bits(r, got), f"{f.__name__} differs on {a.shape} @ {b.shape}"
    if ref is not None:
        assert_matmul_close(got, ref, a, b)
    return got


@pytest.mark.parametrize("m", CUBE)
@pytest.mark.parametrize("dtype", DTYPES)
def test_small_cube_exact_operands(dtype, m):
    rng = np.random.default_rng([m, 1])
    base_a = ints(rng, (m, 97), dtype)
    base_b = ints(rng, (97, 97), dtype)
    for k in CUBE:
        a = np.ascontiguousarray(base_a[:, :k])
        for n in CUBE:
            b = np.ascontiguousarray(base_b[:k, :n])
            fast = run(core._gemm_small, a, b)
            assert same_bits(fast, run(core._gemm_small_naive, a, b)), (m, k, n)
            assert same_bits(fast, run(core._gemm_skinny, a, b)), (m, k, n)
            assert np.array_equal(fast, a @ b), (m, k, n)


@pytest.mark.parametrize("m", EDGES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_small_edges_random_operands(dtype, m):
    rng = np.random.default_rng([m, 2])
    for k in EDGES:
        for n in EDGES:
            a, b = normal(rng, (m, k), dtype), normal(rng, (k, n), dtype)
            fast = run(core._gemm_small, a, b)
            assert same_bits(fast, run(core._gemm_small_naive, a, b)), (m, k, n)
            assert_matmul_close(fast, a @ b, a, b)


@pytest.mark.parametrize("shape", SKINNY_SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_skinny_shapes(dtype, shape):
    m, k, n = shape
    rng = np.random.default_rng(list(shape))
    a, b = normal(rng, (m, k), dtype), normal(rng, (k, n), dtype)
    check_all(a, b, a @ b)
    a, b = ints(rng, (m, k), dtype), ints(rng, (k, n), dtype)
    if k <= 4096:  # exact in float32 below 2**24
        assert np.array_equal(run(core._gemm_skinny, a, b), a @ b)


@pytest.mark.parametrize("n", LEGACY_KILLER_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_legacy_sizes(dtype, n):
    shapes = [(1, n, n), (n, n, 1), (3, n, 7), (n, 7, 3), (n, 3, n), (4, n, n), (n, n, 4)]
    if n <= 333:
        shapes.append((n, n, n))
    for m, k, nn in shapes:
        rng = np.random.default_rng([m, k, nn])
        a, b = normal(rng, (m, k), dtype), normal(rng, (k, nn), dtype)
        check_all(a, b, a @ b)


def _layouts(x):
    """The same logical matrix in every layout the kernels must read in place."""
    c = np.ascontiguousarray(x)
    big = np.zeros((2 * x.shape[0] + 1, 3 * x.shape[1] + 2), dtype=x.dtype)
    big[1::2, 2::3] = x
    odd = [s if d != 1 else (7777 if i == 0 else -12345) * c.itemsize
           for i, (d, s) in enumerate(zip(c.shape, c.strides))]
    out = {
        "C": c,
        "F": np.asfortranarray(x),
        "T": np.ascontiguousarray(x.T).T,
        "strided": big[1::2, 2::3],
        "negative": np.ascontiguousarray(x[::-1, ::-1])[::-1, ::-1],
        "neg_rows": np.ascontiguousarray(x[::-1])[::-1],
        "len1_odd": np.lib.stride_tricks.as_strided(c, c.shape, odd),
    }
    for v in out.values():
        assert np.array_equal(v, x)
    return out


LAYOUT_SHAPES = [
    (1, 1, 1), (1, 7, 1), (5, 1, 9), (1, 33, 17), (17, 33, 1), (13, 29, 31), (97, 65, 33),
    (4, 300, 1000), (1000, 300, 3), (1000, 300, 1), (300, 2, 400), (3, 1, 5000), (2, 70, 4),
]


@pytest.mark.parametrize("shape", LAYOUT_SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_every_layout_gives_the_same_bits(dtype, shape):
    m, k, n = shape
    rng = np.random.default_rng(list(shape))
    a, b = normal(rng, (m, k), dtype), normal(rng, (k, n), dtype)
    ref = check_all(a, b, a @ b)
    la, lb = _layouts(a), _layouts(b)
    for (fast, twin) in KERNELS.values():
        for name_a, va in la.items():
            for name_b, vb in lb.items():
                for f in (fast, twin):
                    assert same_bits(run(f, va, vb), ref), (f.__name__, name_a, name_b)


def _team_can_grow():
    return core._team_size(1e300, 0.0, 1.0, 2**62) > 1


# Each one is far above the serial threshold and has enough slabs for six threads.
THREADED_SHAPES = [(1, 2000, 3000), (4, 64, 100000), (100000, 64, 3), (2000, 3, 2000), (3000, 500, 1)]


@pytest.mark.parametrize("shape", THREADED_SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_thread_count_does_not_change_bits(dtype, shape):
    if not _team_can_grow():
        pytest.skip("OpenMP allows one thread here")
    m, k, n = shape
    rng = np.random.default_rng(list(shape))
    a, b = normal(rng, (m, k), dtype), normal(rng, (k, n), dtype)
    for va in (a, np.asfortranarray(a)):  # F order takes the transposed path when n <= 4
        results = []
        for t in (1, 6, 1, 3):
            pg.set_num_threads(t)
            results.append(run(core._gemm_skinny, va, b))
            results.append(run(core._gemm_skinny_naive, va, b))
        assert all(same_bits(r, results[0]) for r in results)
    assert_matmul_close(results[0], a @ b, a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_caller_ftz_daz_reaches_every_worker(dtype):
    if not _team_can_grow():
        pytest.skip("OpenMP allows one thread here")
    # Products of two values near 2**e are subnormal; every other row of a is
    # lifted so its products stay normal, and the result mixes both kinds.
    rng = np.random.default_rng(9)
    e, lift = (-520, 2.0**300) if dtype == np.float64 else (-70, 2.0**40)
    a = rng.uniform(1.0, 2.0, (4, 257)) * 2.0**e
    a[::2] *= lift
    a = a.astype(dtype)
    b = (rng.uniform(1.0, 2.0, (257, 8000)) * 2.0**e).astype(dtype)
    tiny = np.finfo(dtype).tiny
    pg.set_num_threads(6)
    plain = run(core._gemm_skinny, a, b)
    previous = core._set_mxcsr(_MXCSR_DEFAULT | _FTZ_DAZ)
    try:
        pg.set_num_threads(1)
        one = run(core._gemm_skinny, a, b)
        pg.set_num_threads(6)
        six = run(core._gemm_skinny, a, b)
    finally:
        core._set_mxcsr(previous)
    assert np.any((plain != 0) & (np.abs(plain) < tiny))
    for r in (one, six):
        assert not np.any((r != 0) & (np.abs(r) < tiny))
    assert same_bits(one, six)


SPECIAL_SHAPES = [(5, 7, 9), (24, 33, 17), (96, 96, 96), (1, 300, 1000), (1000, 300, 1), (4, 1000, 6), (500, 3, 400)]


@pytest.mark.parametrize("shape", SPECIAL_SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_nan_and_inf_propagate_like_numpy(dtype, shape):
    m, k, n = shape
    rng = np.random.default_rng(list(shape))
    a, b = normal(rng, (m, k), dtype), normal(rng, (k, n), dtype)
    for x in (a, b):
        flat = x.reshape(-1)
        idx = rng.choice(flat.size, size=min(flat.size, 3), replace=False)
        flat[idx] = [np.nan, np.inf, -np.inf][: idx.size]
    with np.errstate(invalid="ignore"):  # NumPy and the f32 truth also meet inf - inf
        ref = a @ b
        for fast, _ in KERNELS.values():
            assert_matmul_close(run(fast, a, b), ref, a, b)
    for fast, twin in KERNELS.values():
        got = run(fast, a, b)
        other = run(twin, a, b)
        nan = np.isnan(got)
        # a NaN's payload may come from either FMA operand, so only its place is held equal
        assert np.array_equal(nan, np.isnan(other))
        assert got[~nan].tobytes() == other[~nan].tobytes()


@pytest.mark.parametrize("shape", [(5, 7, 9), (6, 96, 16), (1, 300, 1000), (1000, 300, 1), (4, 1000, 4),
                                   (2, 1, 3000), (3000, 2, 5), (97, 97, 97)], ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_underflowed_negative_zero_lands_positive(dtype, shape):
    # Every product is negative and far below the smallest subnormal, so each
    # chain stays at -0.0, which the exact sum also rounds to; the final
    # - (-0.0) must turn it into +0.0, as the packed engine and NumPy do.
    m, k, n = shape
    tiny = 1e-200 if dtype == np.float64 else 1e-30
    rng = np.random.default_rng(list(shape))
    a = (-tiny * rng.uniform(1.0, 2.0, (m, k))).astype(dtype)
    b = (tiny * rng.uniform(1.0, 2.0, (k, n))).astype(dtype)
    assert np.signbit(a[0, 0] * b[0, 0])
    for fast, twin in KERNELS.values():
        for f in (fast, twin):
            got = run(f, a, b)
            assert not got.view(np.uint64 if dtype == np.float64 else np.uint32).any(), f.__name__


@pytest.mark.parametrize("shape", [(3, 0, 4), (0, 5, 3), (3, 5, 0), (0, 0, 0), (1, 0, 100000)], ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_dimensions(dtype, shape):
    m, k, n = shape
    a, b = np.ones((m, k), dtype), np.ones((k, n), dtype)
    for fast, twin in KERNELS.values():
        for f in (fast, twin):
            got = run(f, a, b)
            assert got.shape == (m, n)
            assert not got.view(np.uint64 if dtype == np.float64 else np.uint32).any(), f.__name__


def test_mismatched_inner_dimension_raises():
    a, b = np.ones((3, 4)), np.ones((5, 2))
    for f in (core._gemm_small, core._gemm_skinny_naive):
        with pytest.raises(ValueError):
            f(a, b, np.empty((3, 2)))
    with pytest.raises(ValueError):
        core._gemm_small(np.ones((3, 4)), np.ones((4, 2)), np.empty((2, 3)))


@pytest.mark.skipif(not AVX2, reason="the packed engine runs only on an AVX2 CPU")
@pytest.mark.parametrize("dtype", DTYPES)
def test_packed_engine_agrees_within_its_first_k_panel(dtype):
    # max(m, k, n) > 96 keeps matmul on the packed path; k <= KC makes its
    # first panel the whole chain, the same one these kernels compute.
    kc = core._get_gemm_blocking(f32=dtype == np.float32)["kc"]
    shapes = [(97, 100, 130), (1, 200, 5000), (5000, kc, 3), (200, 1, 300), (128, kc, 128), (300, 4, 97)]
    for m, k, n in shapes:
        assert k <= kc
        rng = np.random.default_rng([m, k, n])
        a, b = normal(rng, (m, k), dtype), normal(rng, (k, n), dtype)
        packed = np.empty((m, n), dtype)
        core.matmul_into(a, b, packed)
        assert same_bits(check_all(a, b), packed), (m, k, n)


TWIN_SCRIPT = """\
import json
import sys

import numpy as np

import peregrine as pg

assert pg.cpu_features()["avx2"] is False, pg.cpu_features()
out = {}
for i, (dtype, m, k, n) in enumerate(json.load(open(sys.argv[1]))):
    rng = np.random.default_rng([m, k, n])
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    for name in ("_gemm_small", "_gemm_skinny"):
        c = np.empty((m, n), dtype)
        getattr(pg._core, name)(a, b, c)
        out[f"{name}{i}"] = c
np.savez(sys.argv[2], **out)
"""


@pytest.mark.skipif(not AVX2, reason="the AVX2 kernels need an AVX2 CPU")
def test_forced_fallback_runs_twins_with_the_same_bits(tmp_path):
    # The hooks pick the kernel from the features detected at import, so the
    # forced fallback is only reachable in a fresh process.
    cases = [(np.dtype(d).name, m, k, n) for d in DTYPES
             for m, k, n in [(5, 7, 9), (96, 96, 96), (1, 3000, 4000), (4000, 300, 3), (2000, 3, 2000)]]
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    env = dict(os.environ, PEREGRINE_DISABLE_AVX2="1")
    env.pop("PEREGRINE_NUM_THREADS", None)
    p = subprocess.run(
        [sys.executable, "-c", TWIN_SCRIPT, str(tmp_path / "cases.json"), str(tmp_path / "twin.npz")],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert p.returncode == 0, p.stderr
    with np.load(tmp_path / "twin.npz") as z:
        for i, (dtype, m, k, n) in enumerate(cases):
            rng = np.random.default_rng([m, k, n])
            a = rng.standard_normal((m, k)).astype(dtype)
            b = rng.standard_normal((k, n)).astype(dtype)
            for name in ("_gemm_small", "_gemm_skinny"):
                assert same_bits(z[f"{name}{i}"], run(getattr(core, name), a, b)), (name, dtype, m, k, n)
