"""The output cache: large host results allocated through Peregrine's NumPy
memory handler, which keeps freed blocks for reuse.

A fresh large output is new pages from the system (the Windows heap maps every
request from about 1016 KiB up directly and unmaps it on free), and touching
them faults each page in. The handler named "peregrine" keeps freed blocks of
at least the threshold, keyed by size, up to a cap of cached bytes
(PEREGRINE_OUTPUT_CACHE_MB, default 256; 0 turns it off). It is installed only
around Peregrine's own large output allocations, on the fast and the Python
path alike, so results stay ordinary NumPy-owned arrays and every other
allocation keeps NumPy's default handler.
"""

import os
import subprocess
import sys
import threading

import numpy as np
import pytest
from numpy._core.multiarray import get_handler_name

import peregrine as pg
from peregrine import _core

DTYPES = [np.float32, np.float64]
DTYPE_IDS = ["f32", "f64"]
OPS = ["matmul", "transpose", "sum", "mean", "axpby", "fma3", "scaled_relu"]
STATS = _core._output_cache_stats()
THRESHOLD = STATS["threshold"]
ON = STATS["cap"] > 0
requires_cache = pytest.mark.skipif(not ON, reason="PEREGRINE_OUTPUT_CACHE_MB=0 in this process")


def _rand(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) ** 2 + 0.5).astype(dtype)


def _result(op, count, dtype):
    # A host result of exactly `count` elements from each op.
    if op == "matmul":
        return pg.matmul(_rand((1, 3), dtype, 0), _rand((3, count), dtype, 1))
    if op == "transpose":
        return pg.transpose(_rand((1, count), dtype, 0))
    if op in ("sum", "mean"):
        return getattr(pg, op)(_rand((3, count), dtype, 0), axis=0)
    xs = [_rand((1, count), dtype, i) for i in range(3)]
    if op == "axpby":
        return pg.axpby(xs[0], xs[1], a=2.0, b=3.0)
    if op == "fma3":
        return pg.fma3(*xs)
    return pg.scaled_relu(xs[0], scale=3.0)


@pytest.fixture(params=["fast", "python"])
def path(request, monkeypatch):
    if request.param == "python":
        for op in OPS:
            monkeypatch.setattr(pg, "_fast_" + op, pg._no_fast_entry)
    return request.param


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("op", OPS)
def test_handler_from_the_threshold_up(path, op, dtype):
    # The threshold is in bytes: exactly there the handler is Peregrine's, one
    # element below it NumPy's default, on both paths and for every op.
    at = THRESHOLD // np.dtype(dtype).itemsize
    large = _result(op, at, dtype)
    small = _result(op, at - 1, dtype)
    assert large.nbytes == THRESHOLD and small.nbytes < THRESHOLD
    assert get_handler_name(large) == ("peregrine" if ON else "default_allocator")
    assert get_handler_name(small) == "default_allocator"
    for r in (large, small):
        assert type(r) is np.ndarray and r.flags.owndata and r.base is None
    # The handler is never left installed for NumPy's own allocations.
    assert get_handler_name() == "default_allocator"
    assert get_handler_name(np.empty(at, dtype)) == "default_allocator"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("op", OPS)
def test_fast_and_python_paths_return_the_same_kind_of_array(monkeypatch, op, dtype):
    count = 2 * THRESHOLD // np.dtype(dtype).itemsize + 3
    fast = _result(op, count, dtype)
    for name in OPS:
        monkeypatch.setattr(pg, "_fast_" + name, pg._no_fast_entry)
    slow = _result(op, count, dtype)
    assert get_handler_name(fast) == get_handler_name(slow)
    assert fast.flags == slow.flags
    assert fast.base is None and slow.base is None
    assert fast.dtype == slow.dtype and fast.shape == slow.shape and fast.strides == slow.strides
    assert fast.tobytes() == slow.tobytes()


@requires_cache
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_a_freed_output_is_handed_out_again(path, dtype):
    a = _rand((1024, 1024), dtype, 0)
    first = pg.transpose(a)
    address = first.ctypes.data
    hits = _core._output_cache_stats()["hits"]
    del first
    again = pg.transpose(a)
    assert again.ctypes.data == address
    assert _core._output_cache_stats()["hits"] == hits + 1
    assert np.array_equal(again, a.T)


@requires_cache
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_a_reused_block_never_shows_old_values(path, dtype):
    # Fresh large pages read as zero, but a cached block still holds the result
    # it last carried, so every output must be written in full. A matmul with
    # k == 0 is the one whose whole output is a fill rather than a computation.
    m, n = 512, 1024
    junk = pg.transpose(np.full((n, m), 7.0, dtype))
    address = junk.ctypes.data
    del junk
    r = pg.matmul(np.ones((m, 0), dtype), np.ones((0, n), dtype))
    assert r.ctypes.data == address
    assert r.shape == (m, n) and not r.any()


@requires_cache
def test_resize_reallocates_a_cached_block():
    a = _rand((1024, 1024), np.float64, 0)
    r = pg.transpose(a)
    # A block that came back from the cache, so realloc runs on a reused pointer.
    del r
    r = pg.transpose(a)
    ref = np.ascontiguousarray(a.T)
    r.resize((2048, 1024), refcheck=False)
    assert get_handler_name(r) == "peregrine"
    assert np.array_equal(r[:1024], ref) and not r[1024:].any()
    r.resize((3, 7), refcheck=False)
    assert np.array_equal(r.ravel(), ref.ravel()[:21])
    del r
    # The grown and the shrunk block went back through the handler; large
    # outputs of every size involved still come out right.
    big = _rand((2048, 1024), np.float64, 1)
    for x in (a, big, a):
        assert np.array_equal(pg.transpose(x), x.T)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_concurrent_callers_get_correct_results(dtype):
    # Several threads allocate and free large outputs of a few sizes at once,
    # with the GIL dropped around each kernel, so the cache sees interleaved
    # requests. Every result must equal its single-threaded reference bitwise.
    shapes = [(512, 512), (700, 1024), (1024, 1024), (512, 512)]
    inputs = [[_rand(s, dtype, 10 * t + j) for j in range(3)] for t, s in enumerate(shapes)]
    refs = [(np.ascontiguousarray(x.T), pg.fma3(x, y, z)) for x, y, z in inputs]
    failures = []

    def worker(t):
        x, y, z = inputs[t]
        for _ in range(40):
            got_t = pg.transpose(x)
            got_f = pg.fma3(x, y, z)
            if not (np.array_equal(got_t, refs[t][0]) and got_f.tobytes() == refs[t][1].tobytes()):
                failures.append(t)
                return

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(len(shapes))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert failures == []
    s = _core._output_cache_stats()
    assert s["bytes"] <= s["cap"]


def _child(script, cache_mb):
    env = dict(os.environ)
    env.pop("PEREGRINE_OUTPUT_CACHE_MB", None)
    if cache_mb is not None:
        env["PEREGRINE_OUTPUT_CACHE_MB"] = cache_mb
    p = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return p.stdout


_CAP_SCRIPT = """
import numpy as np, peregrine as pg
from peregrine import _core
mib = 1 << 20
a = np.ones((3 * mib // 32, 2))      # transposes to 1.5 MiB of float64
held = [pg.transpose(a) for _ in range(5)]
addresses = [r.ctypes.data for r in held]
for i in range(5):
    held[i] = None                    # freed in allocation order
s = _core._output_cache_stats()
assert s["cap"] == 4 * mib, s
assert s["blocks"] == 2 and s["bytes"] == 3 * mib, s   # a third would pass the cap
first, second = pg.transpose(a), pg.transpose(a)
assert first.ctypes.data == addresses[4], "newest block first"
assert second.ctypes.data == addresses[3], "the oldest were evicted"
del first, second
big = pg.transpose(np.ones((5 * mib // 8, 1)))   # 5 MiB, more than the cap
del big
s = _core._output_cache_stats()
assert s["bytes"] == 3 * mib and s["blocks"] == 2, s
print("ok")
"""


def test_the_cap_bounds_the_cache_and_evicts_the_oldest():
    assert _child(_CAP_SCRIPT, "4").strip() == "ok"


_ENV_SCRIPT = """
import numpy as np, peregrine as pg
from numpy._core.multiarray import get_handler_name
from peregrine import _core
for _ in range(3):
    r = pg.transpose(np.ones((1024, 1024)))
    name = get_handler_name(r)
    del r
s = _core._output_cache_stats()
print(name, s["cap"], s["bytes"], s["blocks"], s["hits"])
"""


@pytest.mark.parametrize(
    "value, expected",
    [
        ("0", "default_allocator 0 0 0 0"),
        (None, f"peregrine {256 << 20} {8 << 20} 1 2"),
        ("abc", f"peregrine {256 << 20} {8 << 20} 1 2"),
        ("12x", f"peregrine {256 << 20} {8 << 20} 1 2"),
        ("7", f"peregrine {7 << 20} 0 0 0"),
        ("8", f"peregrine {8 << 20} {8 << 20} 1 2"),
    ],
    ids=["off", "unset", "invalid", "trailing", "below-block", "exact-block"],
)
def test_cache_size_from_the_environment(value, expected):
    # 0 is a pure pass-through: the handler is never installed, so results carry
    # NumPy's default handler and nothing is ever cached. An invalid value keeps
    # the default, and a cap below a block's size never caches that block.
    assert _child(_ENV_SCRIPT, value).strip() == expected


_EXIT_SCRIPT = """
import atexit, numpy as np, peregrine as pg
KEEP = [pg.transpose(np.ones((1024, 1024))) for _ in range(3)]
ONES = np.ones((1024, 1024), np.float32)
KEEP.append(pg.fma3(ONES, ONES, ONES))
CYCLE = [pg.transpose(np.ones((512, 1024)))]
CYCLE.append(CYCLE)
atexit.register(lambda: KEEP.pop())
print("ok")
"""


@pytest.mark.parametrize("value", [None, "0"], ids=["on", "off"])
def test_large_results_outlive_the_interpreter(value):
    # Arrays from the handler are freed during finalization (or never), after
    # the module's own teardown; the handler and its cache are never destroyed.
    assert _child(_EXIT_SCRIPT, value).strip() == "ok"
