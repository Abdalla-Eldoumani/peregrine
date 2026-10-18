"""The packed microkernels: the store-only first k panel and every edge tile.

Every product here has a dimension above 96, the largest that gemm_blis still
sends to its small direct path, so each one runs the packed f64 6x8 or f32 6x16
microkernel.

The first k panel of each C tile stores acc + 0.0 without reading C. The + 0.0
matters: an accumulator whose FMA chain underflows from a negative product is
-0.0, and the old memset-then-add (and OpenBLAS's C = 0 then C += acc) turned it
into +0.0. A plain store would leak the -0.0.

The edge grid walks every row remainder mod 6 and column remainder mod NR, with
k inside one KC panel and across two (so the masked path both stores and
accumulates). Small-integer operands make every partial sum exact, so any
summation order gives the same bits and the packed result must equal the naive
kernel's bitwise; random normal operands, where the orders differ, go through
assert_matmul_close. The output is a NaN-filled view with NaN guard rows on both
sides: a first panel that read C would leak the NaN, and a store past the tile
would hit a guard.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import peregrine as pg
from conftest import assert_matmul_close

DTYPES = [np.float64, np.float32]
NR = {np.float64: 8, np.float32: 16}
MR = 6


def _kc(dtype):
    return pg._core._get_gemm_blocking(f32=dtype == np.float32)["kc"]


@pytest.fixture(autouse=True)
def _cpu_backend():
    # A CUDA build may auto-route a float32 product to the device; these tests
    # are about the CPU kernel.
    prev = pg._backend
    pg.set_backend("cpu")
    yield
    pg.set_backend(prev)


def _signed_zero_shapes(dtype):
    nr = NR[dtype]
    full_m, full_n = 17 * MR, 7 * nr  # whole tiles only
    edge_m, edge_n = 17 * MR - 5, 7 * nr - 1
    multi_k = 2 * _kc(dtype) + 3  # three k panels, the last a short one
    return [
        (full_m, 100, full_n),
        (edge_m, 100, edge_n),
        (full_m, multi_k, full_n),
        (edge_m, multi_k, edge_n),
    ]


@pytest.mark.parametrize("dtype", DTYPES)
def test_first_panel_turns_underflowed_negative_zero_positive(dtype):
    # Every product is negative and far below the smallest subnormal, so each
    # FMA rounds to -0.0 and every accumulator ends at -0.0; the exact result
    # also rounds to -0.0, while NumPy, the naive kernel, and the old memset
    # path all return +0.0.
    tiny = 1e-200 if dtype == np.float64 else 1e-30
    rng = np.random.default_rng(7)
    for m, k, n in _signed_zero_shapes(dtype):
        a = (-tiny * rng.uniform(1.0, 2.0, size=(m, k))).astype(dtype)
        b = (tiny * rng.uniform(1.0, 2.0, size=(k, n))).astype(dtype)
        assert np.all(np.signbit(a[:, :1] * b[:1, :1])), "the products must underflow to -0.0"
        got = pg.matmul(a, b)
        bits = got.view(np.uint64 if dtype == np.float64 else np.uint32)
        assert not bits.any(), f"{(m, k, n)}: expected +0.0 everywhere, found -0.0 or a nonzero"


def _edge_cases():
    cases = []
    for dtype in DTYPES:
        kc = _kc(dtype)
        nr = NR[dtype]
        for k in (97, kc + 5):
            for rm in range(MR):
                for rn in range(nr):
                    cases.append((np.dtype(dtype).name, 17 * MR + rm, k, 7 * nr + rn))
    return cases


# The naive child regenerates the exact operands with the same recipe as
# _exact_operands below; any drift between the two changes the inputs and fails
# the bitwise comparison, so it can never pass silently.
NAIVE_SCRIPT = """\
import json
import sys

import numpy as np

import peregrine as pg

assert pg.cpu_features()["avx2"] is False, pg.cpu_features()
out = {}
for i, (dtype, m, k, n) in enumerate(json.load(open(sys.argv[1]))):
    rng = np.random.default_rng([m, k, n])
    a = rng.integers(-8, 9, size=(m, k)).astype(dtype)
    b = rng.integers(-8, 9, size=(k, n)).astype(dtype)
    out[f"c{i}"] = pg.matmul(a, b)
np.savez(sys.argv[2], **out)
"""


def _exact_operands(dtype, m, k, n):
    rng = np.random.default_rng([m, k, n])
    a = rng.integers(-8, 9, size=(m, k)).astype(dtype)
    b = rng.integers(-8, 9, size=(k, n)).astype(dtype)
    return a, b


@pytest.fixture(scope="module")
def naive_results(tmp_path_factory):
    # One child for the whole grid: PEREGRINE_DISABLE_AVX2 is read once at
    # feature detection, so the naive kernel is only reachable in a fresh process.
    tmp = tmp_path_factory.mktemp("naive")
    cases = _edge_cases()
    (tmp / "cases.json").write_text(json.dumps(cases))
    env = dict(os.environ, PEREGRINE_DISABLE_AVX2="1")
    p = subprocess.run(
        [sys.executable, "-c", NAIVE_SCRIPT, str(tmp / "cases.json"), str(tmp / "naive.npz")],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert p.returncode == 0, p.stderr
    with np.load(tmp / "naive.npz") as z:
        results = [z[f"c{i}"] for i in range(len(cases))]
    return dict(zip(map(tuple, cases), results))


def _into_guarded(a, b):
    m, n = a.shape[0], b.shape[1]
    big = np.full((m + 2, n), np.nan, dtype=a.dtype)
    out = big[1:-1]
    assert out.flags.c_contiguous and np.isnan(out).all()
    got = pg.matmul(a, b, out=out)
    assert got is out
    assert np.isnan(big[0]).all() and np.isnan(big[-1]).all(), "a store landed outside the C tile"
    return got


@pytest.mark.parametrize("case", _edge_cases(), ids=lambda c: f"{c[0]}-{c[1]}x{c[2]}x{c[3]}")
def test_edge_tiles_exact_operands_match_naive_bitwise(case, naive_results):
    dtype, m, k, n = case
    a, b = _exact_operands(dtype, m, k, n)
    got = _into_guarded(a, b)
    ref = naive_results[tuple(case)]
    assert got.tobytes() == ref.tobytes()


@pytest.mark.parametrize("case", _edge_cases(), ids=lambda c: f"{c[0]}-{c[1]}x{c[2]}x{c[3]}")
def test_edge_tiles_random_operands_within_tolerance(case):
    dtype, m, k, n = case
    rng = np.random.default_rng([m, k, n, 1])
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    assert_matmul_close(_into_guarded(a, b), a @ b, a, b)
