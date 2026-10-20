"""The packed engine's schedule never reaches a result bit.

The driver runs its (jc, pc) steps with two B buffers and no team barrier:
members claim B panels as they arrive, wait only for the panels they read and
for the buffer they are about to overwrite, and may run a step ahead of one
another. Each C tile still belongs to one member for all of its pc steps, so its
k panels accumulate in order and the bits depend on the shape and KC alone.
These tests walk the schedule's edge paths and require every combination to
give the bits of one thread at the default blocking:

- one step (k <= KC and n <= NC, one B buffer), one jc block over three pc
  steps, and several jc blocks times pc steps (NC lowered through the private
  blocking hook), so a buffer is reused from the third step on;
- MC of one MR panel, the default, and twice the default, so a B column runs one
  kernel call, eight, or sixteen (the next-panel prefetch slices wrap at eight);
- 1, 2, 3, and 6 threads, and four concurrent callers, each its own team.

KC stays at its default throughout: it sets the k panels each element sums in,
so it is the one blocking value allowed to change bits.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import peregrine as pg
from conftest import assert_matmul_close

DTYPES = [np.float64, np.float32]
NR = {np.float64: 8, np.float32: 16}
MR = 6


@pytest.fixture(autouse=True)
def _cpu_and_defaults():
    # A CUDA build may auto-route a float32 product to the device; the schedule
    # under test is the CPU engine's. The blocking and the thread setting are
    # process-wide, so both are put back whatever the test does.
    prev_backend = pg._backend
    prev_threads = pg.get_num_threads()
    saved = {f32: pg._core._get_gemm_blocking(f32=f32) for f32 in (False, True)}
    pg.set_backend("cpu")
    yield
    for f32, b in saved.items():
        pg._core._set_gemm_blocking(b["mc"], b["kc"], b["nc"], f32)
    pg.set_num_threads(prev_threads)
    pg.set_backend(prev_backend)


def _shapes(dtype):
    kc = pg._core._get_gemm_blocking(f32=dtype == np.float32)["kc"]
    return [
        (103, 200, 129),  # one pc step
        (205, 2 * kc + 3, 300),  # three pc steps, the last a short one
        (97, kc + 5, 203),  # two pc steps; several jc blocks once NC is lowered
    ]


def _blockings(dtype):
    d = pg._core._get_gemm_blocking(f32=dtype == np.float32)
    return [
        (mc, d["kc"], nc)
        for mc in (MR, d["mc"], 2 * d["mc"])
        for nc in (d["nc"], 5 * NR[dtype])
    ]


def _operands(dtype, m, k, n):
    rng = np.random.default_rng([m, k, n])
    return rng.standard_normal((m, k)).astype(dtype), rng.standard_normal((k, n)).astype(dtype)


def _with(dtype, blk, threads, a, b):
    pg._core._set_gemm_blocking(*blk, dtype == np.float32)
    pg.set_num_threads(threads)
    return pg.matmul(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_blocking_and_team_size_never_change_bits(dtype):
    default = pg._core._get_gemm_blocking(f32=dtype == np.float32)
    default = (default["mc"], default["kc"], default["nc"])
    for m, k, n in _shapes(dtype):
        a, b = _operands(dtype, m, k, n)
        ref = _with(dtype, default, 1, a, b)
        assert_matmul_close(ref, a @ b, a, b)
        differ = [
            (blk, threads)
            for blk in _blockings(dtype)
            for threads in (1, 2, 3, 6)
            if _with(dtype, blk, threads, a, b).tobytes() != ref.tobytes()
        ]
        assert differ == [], f"{(m, k, n)}: {differ}"


def _digest(r):
    return hashlib.blake2b(np.ascontiguousarray(r).tobytes()).hexdigest()


@pytest.mark.parametrize("dtype", DTYPES)
def test_concurrent_callers_over_many_steps(dtype):
    # Lowered NC and MC give each call many (jc, pc) steps and short ic blocks,
    # so the four teams sit in the per-step waits at the same time.
    d = pg._core._get_gemm_blocking(f32=dtype == np.float32)
    blk = (MR, d["kc"], 5 * NR[dtype])
    cases = [_operands(dtype, *shape) for shape in _shapes(dtype)]
    refs = [_digest(_with(dtype, blk, 1, a, b)) for a, b in cases]
    pg.set_num_threads(6)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(lambda a=a, b=b: _digest(pg.matmul(a, b))) for a, b in cases * 4]
        got = [f.result(timeout=120) for f in futures]
    assert got == refs * 4
