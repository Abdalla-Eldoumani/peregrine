"""Thread-safety proof: thread count cannot change results, and concurrent
host threads cannot deadlock the native kernel.

Bitwise identity across thread counts is achievable and therefore required:
the kernel parallelizes over output tiles, and each element's accumulation
over the inner dimension runs strictly sequentially, so the thread count
decides only which thread computes a tile, never any element's rounding
order. The subprocess indirection exists because the OpenMP runtime reads
OMP_NUM_THREADS once at its initialization; the variable must be in the
environment before the process starts to take effect. Every child drops an
inherited PEREGRINE_NUM_THREADS, which outranks OMP_NUM_THREADS and would turn
a 1-versus-12 comparison into 1 versus 1.

In process, pg.set_num_threads(1) and (6) must give the same bits for every op
at sizes past any threading threshold, from one caller and from concurrent
callers, and a caller's FTZ/DAZ state must reach every worker. The environment
precedence and the physical-core default run in children, because the setting
is resolved once at import.

The executor test drives kernels that release the GIL into the OpenMP
runtime from four host threads at once. Each entering thread becomes its own
OpenMP root, so the worst plausible outcome is oversubscription, not
deadlock. The future.result timeout is not a real backstop: a worker wedged
in a native call never returns, so executor shutdown joins it forever and
the session hangs before pytest can report the TimeoutError; only the
harness-level timeout reaps that. The test proves the non-deadlocking run,
it does not convert a native deadlock into a loud failure.
"""

import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import peregrine as pg
from conftest import assert_matmul_close

# odd dimensions on purpose: 333 is a legacy-killer size and the row split
# across threads lands unevenly there. Inputs regenerate identically in each
# child from the seed; only the result crosses the process boundary
SCRIPT = """\
import sys

import numpy as np

import peregrine as pg

rng = np.random.default_rng(42)
a = rng.standard_normal((333, 257))
b = rng.standard_normal((257, 199))
np.save(sys.argv[1], pg.matmul(a, b))
"""

# 16 mixed tasks for 4 workers: the odd 333 triple, legacy-killer squares,
# tiny and rectangular shapes, so concurrent entries hit different row splits
EXECUTOR_SIZES = [
    (333, 257, 199),
    (1, 1, 1),
    (3, 7, 5),
    (7, 7, 7),
    (16, 16, 16),
    (50, 50, 50),
    (64, 32, 16),
    (33, 65, 17),
    (100, 1, 100),
    (1, 100, 1),
    (128, 64, 32),
    (250, 250, 250),
    (5, 333, 3),
    (81, 27, 9),
    (256, 128, 64),
    (13, 77, 205),
]


def _run_in_subprocess(tmp_path, omp_threads):
    out_path = tmp_path / f"got_{omp_threads}.npy"
    # full environment copy with the override merged in: a stripped env
    # breaks DLL resolution for the OpenMP runtime on Windows
    env = dict(os.environ, OMP_NUM_THREADS=str(omp_threads))
    env.pop("PEREGRINE_NUM_THREADS", None)
    p = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(out_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # stderr in the message so a child import failure reads as itself
    assert p.returncode == 0, p.stderr
    return np.load(out_path)


def test_thread_count_bitwise_identical(tmp_path):
    got_1 = _run_in_subprocess(tmp_path, 1)
    got_12 = _run_in_subprocess(tmp_path, 12)
    np.testing.assert_array_equal(got_1, got_12)


def _matmul_task(m, k, n, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    return a, b, pg.matmul(a, b)


def test_executor_no_deadlock():
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_matmul_task, m, k, n, seed)
            for seed, (m, k, n) in enumerate(EXECUTOR_SIZES)
        ]
        for future in futures:
            a, b, got = future.result(timeout=120)
            assert_matmul_close(got, a @ b, a, b)


@pytest.fixture(autouse=True)
def _restore_thread_setting():
    # The setting is process-wide, so a test that changes it must not leak
    # its value into the next one.
    previous = pg.get_num_threads()
    yield
    pg.set_num_threads(previous)


@pytest.mark.parametrize("n", [1, 2, 6, 64, np.int64(3)])
def test_set_get_round_trip(n):
    assert pg.set_num_threads(n) is None
    got = pg.get_num_threads()
    assert got == n
    assert type(got) is int


def test_set_num_threads_past_c_int_is_capped():
    pg.set_num_threads(2**40)
    assert pg.get_num_threads() == 2**31 - 1


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4", None])
def test_set_num_threads_rejects(bad):
    pg.set_num_threads(3)
    with pytest.raises(ValueError) as e:
        pg.set_num_threads(bad)
    assert str(e.value) == f"set_num_threads: n must be a positive int, got {bad!r}"
    assert pg.get_num_threads() == 3


# Physical cores in a process's affinity mask, found without peregrine and by a
# different route than its own: GetLogicalProcessorInformation core masks on
# Windows (peregrine reads the Ex form), and on Linux the sibling-list files
# grouped by their whole content (peregrine keys a core by its lowest sibling).
# cores_in_mask() returns one set of logical CPUs per core; confine() narrows
# the process affinity before peregrine is imported.
_TOPOLOGY = r"""
import os
import sys

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]

    class Info(ctypes.Structure):
        _fields_ = [("mask", ctypes.c_size_t), ("relationship", ctypes.c_uint32),
                    ("_union", ctypes.c_ubyte * 16)]

    def cores_in_mask():
        proc, system = ctypes.c_size_t(), ctypes.c_size_t()
        assert k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(system))
        size = ctypes.c_uint32(0)
        k32.GetLogicalProcessorInformation(None, ctypes.byref(size))
        buf = (Info * (size.value // ctypes.sizeof(Info)))()
        assert k32.GetLogicalProcessorInformation(buf, ctypes.byref(size))
        cores = [{c for c in range(64) if (i.mask & proc.value) >> c & 1}
                 for i in buf if i.relationship == 0]  # RelationProcessorCore
        return [c for c in cores if c]

    def confine(cpus):
        assert k32.SetProcessAffinityMask(k32.GetCurrentProcess(), sum(1 << c for c in cpus))
else:
    def cores_in_mask():
        cores = {}
        for cpu in sorted(os.sched_getaffinity(0)):
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                cores.setdefault(f.read().strip(), set()).add(cpu)
        return list(cores.values())

    def confine(cpus):
        os.sched_setaffinity(0, cpus)
"""

_SETTING_SCRIPT = _TOPOLOGY + r"""
before = cores_in_mask()
pick = {
    "one": {min(before[0])},
    "core": before[0],
    "firsts": {min(c) for c in before},
}.get(sys.argv[1])
if pick is not None:
    confine(pick)

import peregrine as pg

print(pg.get_num_threads(), len(cores_in_mask()), len(before))
"""


def _child_setting(mode="full", **env):
    """(setting, cores in the child's mask, cores before confining, stderr)."""
    child = {k: v for k, v in os.environ.items() if k not in ("PEREGRINE_NUM_THREADS", "OMP_NUM_THREADS")}
    child.update(env)
    p = subprocess.run(
        [sys.executable, "-c", _SETTING_SCRIPT, mode],
        env=child,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr
    got, cores, before = map(int, p.stdout.split())
    return got, cores, before, p.stderr


@pytest.mark.parametrize(
    "env, expect",
    [
        ({"PEREGRINE_NUM_THREADS": "3", "OMP_NUM_THREADS": "2"}, 3),
        ({"OMP_NUM_THREADS": "2"}, 2),
        # one count per nesting level; the outermost is the one a call uses
        ({"OMP_NUM_THREADS": "5,2"}, 5),
        ({"PEREGRINE_NUM_THREADS": " 4 "}, 4),
        ({"PEREGRINE_NUM_THREADS": "abc", "OMP_NUM_THREADS": "2"}, 2),
        # more than the logical CPUs is a valid setting, capped per call
        ({"PEREGRINE_NUM_THREADS": "64"}, 64),
    ],
)
def test_env_precedence(env, expect):
    got, _, _, _ = _child_setting(**env)
    assert got == expect


@pytest.mark.parametrize("value", ["", "0", "-1", "2.5", "abc", "4,2"])
def test_invalid_peregrine_num_threads_is_ignored_silently(value):
    got, cores, _, stderr = _child_setting(PEREGRINE_NUM_THREADS=value)
    assert got == cores
    assert stderr == ""


@pytest.mark.parametrize("value", ["0", "-3", "abc"])
def test_invalid_omp_num_threads_is_ignored(value):
    # stderr is not checked: libgomp itself warns about a malformed value
    got, cores, _, _ = _child_setting(OMP_NUM_THREADS=value)
    assert got == cores


@pytest.mark.parametrize("mode", ["full", "one", "core", "firsts"])
def test_default_is_physical_cores_in_affinity_mask(mode):
    got, cores, before, _ = _child_setting(mode)
    assert got == cores
    # SMT siblings are one core; the first CPU of each core keeps every core
    expect = {"full": before, "one": 1, "core": 1, "firsts": before}[mode]
    assert got == expect


# Every op at sizes past any threading threshold, both dtypes. Matmul takes the
# packed path (and the naive one under PEREGRINE_DISABLE_AVX2); the (3000, 2000)
# transpose input is a contiguous reshape of the (2000, 3000) operand.
_OPS = [
    "matmul-333x257x199", "matmul-1000x1000x1000",
    "sum-None", "sum-0", "sum-1", "mean-None", "mean-0", "mean-1",
    "transpose", "axpby", "fma3", "scaled_relu",
]
CASE_NAMES = [f"{op}-{dt}" for dt in ("float32", "float64") for op in _OPS]


@pytest.fixture(scope="module")
def cases():
    rng = np.random.default_rng(2024)
    out = {}
    for dt in ("float32", "float64"):
        for m, k, n in [(333, 257, 199), (1000, 1000, 1000)]:
            a = rng.standard_normal((m, k)).astype(dt)
            b = rng.standard_normal((k, n)).astype(dt)
            out[f"matmul-{m}x{k}x{n}-{dt}"] = lambda a=a, b=b: pg.matmul(a, b)
        x, y, z = (rng.standard_normal((2000, 3000)).astype(dt) for _ in range(3))
        for axis in (None, 0, 1):
            out[f"sum-{axis}-{dt}"] = lambda x=x, axis=axis: pg.sum(x, axis=axis)
            out[f"mean-{axis}-{dt}"] = lambda x=x, axis=axis: pg.mean(x, axis=axis)
        out[f"transpose-{dt}"] = lambda t=x.reshape(3000, 2000): pg.transpose(t)
        out[f"axpby-{dt}"] = lambda x=x, y=y: pg.axpby(x, y, a=0.75, b=-1.5)
        out[f"fma3-{dt}"] = lambda x=x, y=y, z=z: pg.fma3(x, y, z)
        out[f"scaled_relu-{dt}"] = lambda x=x: pg.scaled_relu(x, scale=1.25)
    assert sorted(out) == sorted(CASE_NAMES)
    return out


def _same_bits(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def _digest(r):
    r = np.asarray(r)
    return r.dtype.str, r.shape, hashlib.blake2b(r.tobytes()).hexdigest()


@pytest.mark.parametrize("name", CASE_NAMES)
def test_one_and_six_threads_bitwise(cases, name):
    pg.set_num_threads(1)
    one = cases[name]()
    pg.set_num_threads(6)
    six = cases[name]()
    assert _same_bits(one, six)


def test_concurrent_callers_bitwise(cases):
    # Each submitting thread roots its own OpenMP team. Digests keep the
    # reference set small while the pool holds its results.
    pg.set_num_threads(1)
    refs = {name: _digest(fn()) for name, fn in cases.items()}
    pg.set_num_threads(6)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {name: pool.submit(lambda fn=fn: _digest(fn())) for name, fn in cases.items()}
        differ = [name for name, f in futures.items() if f.result(timeout=120) != refs[name]]
    assert differ == []


_MXCSR_DEFAULT = 0x1F80  # every exception masked, round to nearest
_FTZ_DAZ = 0x8040  # flush-to-zero (bit 15) and denormals-are-zero (bit 6)


def _subnormal_case(op, dt):
    # Operands near 2**e, whose products are subnormal in dt. Every other row
    # is lifted so its products stay normal, and the result mixes both kinds.
    rng = np.random.default_rng(5)
    e, lift = (-520, 2.0**300) if dt == "float64" else (-70, 2.0**40)
    s = 2.0**e

    def operand(shape, signed=False):
        v = rng.uniform(1.0, 2.0, shape) * s
        if signed:
            v *= rng.choice([-1.0, 1.0], shape)
        v[::2] *= lift
        return v.astype(dt)

    if op == "matmul":
        a, b = operand((1000, 257)), (rng.uniform(1.0, 2.0, (257, 199)) * s).astype(dt)
        return lambda: pg.matmul(a, b)
    x, y = operand((2000, 3000), signed=True), operand((2000, 3000))
    if op == "axpby":
        return lambda: pg.axpby(x, y, a=s, b=s)
    if op == "fma3":
        # a subnormal addend, which DAZ reads as zero
        z = (rng.uniform(1.0, 2.0, x.shape) * s * s).astype(dt)
        return lambda: pg.fma3(x, y, z)
    return lambda: pg.scaled_relu(x, scale=s)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("op", ["matmul", "axpby", "fma3", "scaled_relu"])
def test_caller_ftz_daz_reaches_every_worker(op, dtype):
    fn = _subnormal_case(op, dtype)
    tiny = np.finfo(dtype).tiny
    pg.set_num_threads(6)
    plain = fn()
    previous = pg._core._set_mxcsr(_MXCSR_DEFAULT | _FTZ_DAZ)
    try:
        pg.set_num_threads(1)
        one = fn()
        pg.set_num_threads(6)
        six = fn()
    finally:
        pg._core._set_mxcsr(previous)
    # the default MXCSR keeps subnormals, so flushing them is observable
    assert np.any((plain != 0) & (np.abs(plain) < tiny))
    # a worker that kept its own MXCSR would leave subnormals in its rows
    for r in (one, six):
        assert not np.any((r != 0) & (np.abs(r) < tiny))
    assert _same_bits(one, six)


def test_team_size():
    ts = pg._core._team_size
    pg.set_num_threads(2**31 - 1)
    limit = ts(1e300, 0.0, 1.0, 2**62)  # OpenMP's own limit on this thread
    assert limit >= 1
    pg.set_num_threads(4)
    s = min(4, limit)
    assert ts(99.0, 100.0, 1.0, 1000) == 1  # below serial_below
    assert ts(100.0, 100.0, 1.0, 1000) == s  # at it
    assert ts(25.0, 0.0, 10.0, 1000) == min(3, s)  # ceil(25 / 10) threads have work
    assert ts(1e300, 0.0, 1.0, 2) == min(2, s)  # no more threads than units
    assert ts(1e300, 0.0, 1.0, 0) == 1
    assert ts(1e300, 0.0, 1.0, -5) == 1
    assert ts(0.0, 0.0, 1.0, 1000) == 1
    assert ts(-1.0, -2.0, 1.0, 1000) == 1
    assert ts(1e300, 0.0, 0.0, 1000) == s  # per_thread 0 bounds nothing
    pg.set_num_threads(1)
    assert ts(1e300, 0.0, 1.0, 1000) == 1


def test_team_size_honors_openmp_limit():
    # OpenMP's limit on the calling thread caps a call below the setting.
    # threadpoolctl's limit is process-wide under vcomp and this thread's under
    # libgomp; either way this thread sees it.
    threadpoolctl = pytest.importorskip("threadpoolctl")
    if not any(d.get("user_api") == "openmp" for d in threadpoolctl.threadpool_info()):
        pytest.skip("threadpoolctl does not see this build's OpenMP runtime")
    pg.set_num_threads(4)
    with threadpoolctl.threadpool_limits(1, user_api="openmp"):
        assert pg._core._team_size(1e300, 0.0, 1.0, 1000) == 1
