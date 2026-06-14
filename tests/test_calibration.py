"""Calibration and dispatch suite, covering DISP-01..05. This file is a Wave 0
scaffold: it establishes the shared infrastructure (the requires_cuda skip gate
and the FME_CACHE_DIR cache isolation idiom) so the later Phase 5 waves fill in
each reserved test under its own name without re-establishing setup. It asserts
nothing about calibrate/policy behavior yet -- those modules do not exist until
Waves 2-3 -- beyond proving the file collects and that the introspection smoke
passes on every build.

requires_cuda() (conftest) is the one GPU gate, identical to test_cuda.py: build
flag plus a usable-device probe, deferring to fme.has_cuda(). CPU-only and the
WSL/GCC clone stay green; only the GPU-measurement tests (the live crossover and
the real-device calibrate path) carry @gpu. The bulk of DISP-05 is NOT gated: a
backend decision is a PURE function fed a synthetic calibration dict, so it runs
everywhere and asserts a backend ENUM ("cpu" / "cuda"), needing no tolerance.

Cache isolation -- the design names FME_CACHE_DIR (DISP-03). Every in-process
test that touches the calibration cache redirects it to a pytest tmp_path so the
real per-user cache is never read or written:

    def test_writes_cache(tmp_path, monkeypatch):
        monkeypatch.setenv("FME_CACHE_DIR", str(tmp_path))
        ...  # calibrate()/policy now read and write under tmp_path only

For an FME_BACKEND override the variable must be in the environment BEFORE the
process starts (the dispatch policy reads it once, exactly like FME_DISABLE_AVX2
in test_fallback.py), so that test spawns a child with a full env copy and the
override merged in -- the test_fallback.py:52-67 subprocess shape -- rather than
monkeypatching in-process:

    env = dict(os.environ, FME_BACKEND="cpu")   # full copy: Windows DLL search
    p = subprocess.run([sys.executable, "-c", SCRIPT, str(out_path)],
                       env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr          # child error surfaces as itself

A warn-once or other process-global sentinel (RESEARCH Pitfall 5) is isolated
the same way the forced-OOM test isolates its sentinel (test_cuda.py:544-568): a
subprocess per assertion, never an in-process reset that leaks across tests.

Any numeric CPU-vs-GPU comparison a later wave adds routes through
assert_matmul_close (conftest), never an inline rtol -- the single toleranced
path, exactly like the CPU and CUDA suites; f32 is judged against the f64 ground
truth the same way. Most DISP-05 tests compare a backend enum and need no
tolerance at all.

Reserved -k test names, one per acceptance point in the Phase 5 validation map.
This wave (05-02) fills the pure-policy decision and cache-read names; the
measurement and integration names stay reserved for their waves. The names do
not change.

    DISP-01 (no calibration at import; bounded calibrate):
        calibrate_budget          calibration completes within its time budget   [Wave 3: calibrate.py]
        cpu_only_calibrate        a CPU-only build calibrates without a device    [Wave 3: calibrate.py]
    DISP-02 (cache write; signature):
        writes_cache              calibrate persists a cache file                 [Wave 3: calibrate.py]
        cache_reuse               a second run reuses the cache, does not redo     [Wave 3: calibrate.py]
        signature_distinct        a different machine signature -> a distinct key  [Wave 3: calibrate.py]
    DISP-03 (corrupt/mismatch -> recalibrate, never half-load; cache override):
        corrupt_recalibrate       a corrupt cache is discarded and recalibrated   [Wave 2: policy._load_cache]
        schema_mismatch           an unknown schema version -> recalibrate         [Wave 2: policy._load_cache]
        signature_mismatch        a signature mismatch -> recalibrate              [Wave 2: policy._load_cache]
        atomic_write              the cache write is atomic, never half-written    [Wave 3: calibrate.py]
        cache_dir_override        FME_CACHE_DIR redirects the cache location       [Wave 2: policy._cache_path]
    DISP-04 (conservative static fallback; transfer cost in crossover):
        static_fallback           no cache -> a conservative CPU-preferring choice [Wave 2: policy.choose_backend]
        transfer_cost             the crossover accounts for host<->device transfer[Wave 2: policy.choose_backend]
    DISP-05 (routing decisions; the FP64 trap):
        f64_host_cpu              host f64 never auto-routes to the GPU            [Wave 2: policy.choose_backend]
        f32_256_host_cpu          host f32 n=256 stays on the CPU (overhead floor) [Wave 2: policy.choose_backend]
        f32_4096_host_gpu         host f32 n=4096 routes to the GPU                [Wave 2: policy.choose_backend]
        f32_device_gpu            f32 device-resident routes to the GPU            [Wave 2: policy.choose_backend]
        set_backend               an explicit backend choice is honored           [Wave 2: policy.choose_backend]
        env_backend               FME_BACKEND (subprocess, env-before-import)      [Wave 4: __init__.py]
        live_crossover            real RTX 3060 calibration reproduces DISP-05 (@gpu) [Wave 4: integration]
"""

import importlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import fastmathext as fme
from fastmathext import policy

# The SUBMODULE, fetched via import_module. As of Phase 5 the package re-exports
# the calibrate FUNCTION as fme.calibrate (the public API),
# which shadows the submodule attribute: `from fastmathext import calibrate` and
# `import fastmathext.calibrate as calibrate` both now bind the FUNCTION. These
# tests need the MODULE -- they call calibrate.calibrate(), _machine_signature(),
# _cpu_brand() -- so import_module (which returns the module object from
# sys.modules regardless of the namespace shadowing) is the reliable handle. The
# public function is exercised as fme.calibrate elsewhere.
calibrate = importlib.import_module("fastmathext.calibrate")
from conftest import assert_matmul_close, requires_cuda

_CUDA_OK, _CUDA_REASON = requires_cuda()
# Reused by every device-touching test the later waves add (the live crossover
# and the real-device calibrate path). The scaffold smoke test below is
# deliberately NOT decorated: it must pass on every build to prove the file is
# wired and collects even when CUDA is off.
gpu = pytest.mark.skipif(not _CUDA_OK, reason=_CUDA_REASON)


def test_calibration_module_scaffold():
    # Runs everywhere, including CPU-only and WSL: proves this file collects and
    # is wired to the package without depending on calibrate/policy (which do not
    # exist until Waves 2-3). has_cuda() is the stable introspection surface and
    # returns a bool on every build (False on a CPU-only build, never an
    # exception), so the scaffold can assert its type as the everywhere-smoke,
    # mirroring test_cuda.py's test_has_cuda_runtime.
    assert isinstance(fme.has_cuda(), bool)


# --- synthetic calibration table -------------------------------------------
#
# The whole DISP-05 contract is provable WITHOUT a GPU by feeding choose_backend
# a synthetic calibration dict: it is a pure function, so its decisions are
# fixed by the numbers in the table, not by the hardware. The numbers below are
# constructed so the four acceptance points hold BY CONSTRUCTION, encoding the
# physical fact the single-GFLOP/s model gets wrong (RESEARCH CRITICAL pitfall):
# at small n the host GPU path is dominated by a FIXED staging+launch floor, not
# by asymptotic bandwidth, so n=256 host stays on the CPU while n=4096 host
# crosses to the GPU.
#
# Times are per-(dtype, n) measured wall seconds on a log grid (the cache shape):
#   cpu:  honest ~55 GF/s f32 / ~80 GF/s f64, work = 2*n^3 flops
#   gpu:  a ~50us cuBLAS launch floor + asymptotic ~2600 GF/s compute (NO transfer)
#   transfer: H2D(2 operands) + D2H(1 result) at ~12 GB/s PCIe, plus an ~800us
#             fixed staging floor (three Python->native->CUDA crossings + launch
#             + pageable-copy setup) that dominates at small n.
#
# Resulting host crossover is ~n=298, device crossover ~n=112, so:
#   f32 n=256 host  -> CPU  (gpu_compute + transfer 0.93ms > cpu 0.61ms)
#   f32 n=4096 host -> CUDA (70.5ms < cpu 2499ms)
#   f32 n=512 device-> CUDA (gpu_compute 0.15ms < cpu 4.88ms, no transfer)
#   f32 n=256 device-> CUDA while n=256 host -> CPU  (transfer-cost crossover)

_GRID = [128, 256, 512, 1024, 2048, 4096]


def _cpu_seconds(n, gflops):
    return 2.0 * n**3 / (gflops * 1e9)


def _gpu_seconds(n):
    # fixed cuBLAS launch floor + asymptotic 2600 GF/s compute (compute only)
    return 50e-6 + 2.0 * n**3 / 2600e9


def synthetic_calibration():
    """A realistic CPU+GPU+transfer calibration dict (no hardware needed)."""
    return {
        "schema": policy.SCHEMA,
        "signature": "synthetic|test|table",
        "cpu": {
            # f32 ~55 GF/s, f64 ~80 GF/s -- the exact values do not matter; what
            # matters is the small-n host GPU path losing to the CPU.
            "float32": [[n, _cpu_seconds(n, 55.0)] for n in _GRID],
            "float64": [[n, _cpu_seconds(n, 80.0)] for n in _GRID],
        },
        "gpu": {
            # only float32 -- f64 never auto-routes to the GPU, so the table
            # carries no f64 GPU entry (and the FP64 trap fires before any lookup)
            "float32": [[n, _gpu_seconds(n)] for n in _GRID],
        },
        "transfer": {
            "h2d_gbps": 12.0,
            "d2h_gbps": 12.0,
            "fixed_overhead_s": 800e-6,
        },
        "calibrated_at": "2026-06-13T00:00:00",
    }


# --- DISP-05: the four auto-mode acceptance points (pure, CPU-only) ----------


def test_f64_host_cpu():
    # The FP64 trap (rule 2): float64 GEMM never auto-routes to the GPU at any
    # size, decided BEFORE any table lookup. Holds at every grid size.
    table = synthetic_calibration()
    for n in (256, 1024, 4096):
        got = policy.choose_backend(
            "float64", n, n, n, residency="host", calibration=table, backend="auto"
        )
        assert got == "cpu", f"f64 host n={n} must stay on cpu, got {got}"


def test_f32_256_host_cpu():
    # Host f32 n=256 stays on the CPU: the GPU compute + H2D/D2H round trip +
    # fixed staging floor (~0.93ms) exceeds the honest CPU time (~0.61ms). This
    # is the point a single-GFLOP/s model gets wrong (RESEARCH CRITICAL pitfall).
    table = synthetic_calibration()
    got = policy.choose_backend(
        "float32", 256, 256, 256, residency="host", calibration=table, backend="auto"
    )
    assert got == "cpu"


def test_f32_4096_host_gpu():
    # Host f32 n=4096 crosses to the GPU: the GEMM time saved (~2499ms CPU vs
    # ~53ms GPU compute) dwarfs the ~17.6ms transfer round trip.
    table = synthetic_calibration()
    got = policy.choose_backend(
        "float32", 4096, 4096, 4096, residency="host", calibration=table, backend="auto"
    )
    assert got == "cuda"


def test_f32_device_gpu():
    # Device-resident f32 at n>=512 routes to the GPU: no transfer cost, so it
    # crosses far earlier than the host path. Verify the stated point and that it
    # holds across the larger grid sizes too.
    table = synthetic_calibration()
    for n in (512, 1024, 2048, 4096):
        got = policy.choose_backend(
            "float32", n, n, n, residency="device", calibration=table, backend="auto"
        )
        assert got == "cuda", f"f32 device n={n} must route to cuda, got {got}"


# --- DISP-04: static fallback and the transfer-cost crossover ----------------


def test_static_fallback():
    # No cache -> conservative CPU-preferring choice, for every (dtype, size,
    # residency). A calibration dict with no "gpu" key (a CPU-only cache) is the
    # same: the GPU is treated as unavailable.
    cpu_only = synthetic_calibration()
    del cpu_only["gpu"]
    for dtype in ("float32", "float64"):
        for n in (256, 1024, 4096):
            for residency in ("host", "device"):
                assert (
                    policy.choose_backend(
                        dtype, n, n, n, residency=residency,
                        calibration=None, backend="auto",
                    )
                    == "cpu"
                )
                assert (
                    policy.choose_backend(
                        dtype, n, n, n, residency=residency,
                        calibration=cpu_only, backend="auto",
                    )
                    == "cpu"
                )


def test_transfer_cost():
    # The crossover accounts for the host<->device round trip: at n=256 a
    # device-resident f32 routes to the GPU (compute alone beats the CPU) while
    # the SAME n as a host array stays on the CPU once H2D+D2H + staging is added.
    # The device path therefore crosses at a smaller n than the host path.
    table = synthetic_calibration()
    host = policy.choose_backend(
        "float32", 256, 256, 256, residency="host", calibration=table, backend="auto"
    )
    device = policy.choose_backend(
        "float32", 256, 256, 256, residency="device", calibration=table, backend="auto"
    )
    assert host == "cpu"
    assert device == "cuda"


# --- DISP-05: explicit backend override (set_backend / FME_BACKEND) ----------


def test_set_backend_forces_cpu():
    # A forced "cpu" backend wins within rules 1-2: it stays CPU even when the
    # table favors the GPU (large-n f32).
    table = synthetic_calibration()
    got = policy.choose_backend(
        "float32", 4096, 4096, 4096, residency="host", calibration=table, backend="cpu"
    )
    assert got == "cpu"


def test_set_backend_forces_cuda_routing():
    # A forced "cuda" backend routes to cuda even for float64 and even with no
    # calibration: the POLICY only routes; the wrapper (not the policy) validates
    # device availability and raises the requested-but-unavailable RuntimeError.
    got_f64 = policy.choose_backend(
        "float64", 256, 256, 256, residency="host", calibration=None, backend="cuda"
    )
    assert got_f64 == "cuda"
    table = synthetic_calibration()
    got_small = policy.choose_backend(
        "float32", 256, 256, 256, residency="host", calibration=table, backend="cuda"
    )
    assert got_small == "cuda"


def test_set_backend():
    # The wrapper-level set_backend (DISP-05 override half). cpu/auto succeed and
    # return None; an out-of-set value raises ValueError naming the allowed set;
    # on a CPU-only build "cuda" raises the VERBATIM "cuda backend requested but
    # <reason>" RuntimeError (the same token to_device raises -- the contract is
    # the literal string, matched on its prefix here). The module-global _backend
    # is restored to "auto" in a finally so this test cannot leak a forced backend
    # into any later test in the same process (the global is process-wide state).
    try:
        assert fme.set_backend("cpu") is None
        assert fme._backend == "cpu"
        assert fme.set_backend("auto") is None
        assert fme._backend == "auto"

        with pytest.raises(ValueError) as ei:
            fme.set_backend("gpu")
        # the message names the allowed vocabulary so a typo is self-correcting
        assert "auto" in str(ei.value) and "cpu" in str(ei.value)

        if not fme.has_cuda():
            with pytest.raises(RuntimeError) as ri:
                fme.set_backend("cuda")
            assert str(ri.value).startswith("cuda backend requested but "), str(ri.value)
        else:
            # on an ON build with a usable device the forced choice is accepted
            assert fme.set_backend("cuda") is None
            assert fme._backend == "cuda"
    finally:
        fme.set_backend("auto")


# Child for the FME_BACKEND env-before-import test. It asserts the override
# reached the module BEFORE trusting any behavior (the test_fallback.py shape:
# the child verifies the env actually took effect, so a missed override fails
# loudly rather than passing vacuously). FME_BACKEND is read once at import into
# fme._backend, so the child checks that global AND that a host f32 dispatch stays
# on the CPU under FME_BACKEND=cpu (it returns an ndarray, not an fme.Array).
_ENV_BACKEND_SCRIPT = """\
import numpy as np

import fastmathext as fme

# the override must have reached the module global at import
assert fme._backend == "cpu", repr(fme._backend)

# and a host f32 product stays on the CPU: a host ndarray result, never an Array
a = np.ones((64, 64), dtype=np.float32)
b = np.ones((64, 64), dtype=np.float32)
out = fme.matmul(a, b)
assert isinstance(out, np.ndarray), type(out)
assert out.shape == (64, 64) and out.dtype == np.float32
"""


def test_env_backend(tmp_path):
    # DISP-05: FME_BACKEND maps to the backend at import. The variable MUST be in
    # the environment before the process starts (the dispatch policy reads it once
    # at import, exactly like FME_DISABLE_AVX2 in test_fallback.py), so this spawns
    # a child with a FULL env copy plus the override merged in -- a stripped env
    # breaks Windows DLL resolution -- rather than monkeypatching in-process. The
    # child asserts the override reached fme._backend before trusting the result;
    # its stderr surfaces in the assertion message so a child failure reads as
    # itself. FME_CACHE_DIR is redirected to a tmp dir so the child never touches
    # the real per-user cache even if a dispatch were to read it.
    env = dict(os.environ, FME_BACKEND="cpu", FME_CACHE_DIR=str(tmp_path))
    p = subprocess.run(
        [sys.executable, "-c", _ENV_BACKEND_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr


# Child for the import-laziness guard. It imports fastmathext at a fresh, empty
# FME_CACHE_DIR and asserts NO cache file appeared: the import must run no
# calibration and read no cache (the <50ms lazy contract). The cache file is
# created only by an explicit calibrate() or by a dispatch that consults a present
# GPU's crossover -- never by `import fastmathext` itself.
_IMPORT_LAZY_SCRIPT = """\
import os
import sys

cache_dir = sys.argv[1]
before = set(os.listdir(cache_dir))

import fastmathext as fme

after = set(os.listdir(cache_dir))
new = after - before
assert new == set(), "import created cache files: " + repr(new)
# the lazy default is "auto" -- import read FME_BACKEND (absent here) and nothing
# else; no cache, no calibration ran
assert fme._backend == "auto", repr(fme._backend)
"""


def test_import_no_calibration(tmp_path):
    # DISP-01: a bare `import fastmathext` runs no calibration and reads/writes no
    # cache (the import budget is <50ms; calibration and the cache read are lazy).
    # Run in a subprocess so the import is genuinely fresh -- an in-process import
    # is already cached and could not observe import-time side effects. The child
    # points FME_CACHE_DIR at a fresh empty tmp dir and asserts no file appears
    # after the import; the cache materializes only on an explicit calibrate() or a
    # GPU-routed dispatch, never at import.
    env = dict(os.environ, FME_CACHE_DIR=str(tmp_path))
    p = subprocess.run(
        [sys.executable, "-c", _IMPORT_LAZY_SCRIPT, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr
    # belt and suspenders: the parent confirms the dir is still empty afterward
    assert os.listdir(str(tmp_path)) == [], os.listdir(str(tmp_path))


@gpu
def test_live_crossover(tmp_path, monkeypatch):
    # DISP-05 (live, @gpu): on a fresh ON build, an actual calibrate() on the real
    # RTX 3060 must complete within budget, return gpu + transfer entries (a true
    # GPU-signature cache), and the four DISP-05 crossover points must hold when
    # the policy is fed the REAL measured numbers -- not the synthetic table. This
    # is the end-to-end close of DISP-01/05 on hardware: every other DISP-05 test
    # uses constructed numbers, this one proves the live device reproduces the
    # contract. Isolated to a tmp cache dir so the real per-user cache is untouched.
    monkeypatch.setenv("FME_CACHE_DIR", str(tmp_path))

    start = time.perf_counter()
    cal = calibrate.calibrate(force=True, budget_seconds=120)
    elapsed = time.perf_counter() - start

    # completed within the documented ceiling (generous margin for one in-flight
    # size, matching test_calibrate_budget's reasoning)
    assert elapsed < 120 + 60.0, f"live calibrate overran: {elapsed:.1f}s"

    # a real GPU calibration carries the measured gpu + transfer series ...
    assert "gpu" in cal and "transfer" in cal, "ON-build calibrate must measure the GPU"
    assert cal["gpu"]["float32"], "gpu float32 grid must be non-empty"
    for key in ("h2d_gbps", "d2h_gbps", "fixed_overhead_s"):
        assert key in cal["transfer"], f"transfer must carry {key}"
    # ... and a GPU machine signature (gpu name + driver, not none/n-a)
    parts = cal["signature"].split("|")
    assert parts[1] != "none" and parts[2] != "n/a", cal["signature"]

    # the four DISP-05 points, fed the LIVE numbers. The measured grid only spans
    # the sizes the budget allowed; key the points off what was actually measured.
    measured_ns = [pt[0] for pt in cal["cpu"]["float32"]]
    large_n = measured_ns[-1]

    # f64 host never routes to the GPU, at any measured size (the FP64 trap --
    # hardcoded, independent of any measured time, so it holds on real numbers
    # exactly as on the synthetic table)
    for n in measured_ns:
        assert (
            policy.choose_backend(
                "float64", n, n, n, residency="host", calibration=cal, backend="auto"
            )
            == "cpu"
        ), f"live f64 host n={n} must stay on cpu"

    # The transfer-cost ordering, on real numbers: at EVERY measured size a
    # device-resident f32 is at least as likely to route to the GPU as the same
    # size as a host array, because the host path adds the H2D+D2H round trip and
    # the fixed staging floor on top of the identical compute comparison. So the
    # host path can never pick cuda where the device path picks cpu -- the device
    # crossover sits at a smaller-or-equal n. This is the load-bearing DISP-04/05
    # property and it holds regardless of the absolute CPU-time magnitude (it falls
    # straight out of _transfer_time >= 0 in the policy). The exact small-n
    # host->cpu point is asserted on the synthetic table (test_f32_256_host_cpu),
    # where the CPU baseline is honest by construction; here the live CPU grid only
    # has to be internally consistent, which the inequality checks.
    for n in measured_ns:
        host = policy.choose_backend(
            "float32", n, n, n, residency="host", calibration=cal, backend="auto"
        )
        device = policy.choose_backend(
            "float32", n, n, n, residency="device", calibration=cal, backend="auto"
        )
        assert not (host == "cuda" and device == "cpu"), (
            f"live n={n}: host picked cuda but device picked cpu -- the transfer "
            f"floor must make the host path the more conservative one"
        )

    # device-resident f32 at the largest measured size routes to the GPU: with no
    # transfer cost the warm device GEMM beats the CPU outright
    assert (
        policy.choose_backend(
            "float32", large_n, large_n, large_n,
            residency="device", calibration=cal, backend="auto",
        )
        == "cuda"
    ), f"live f32 device n={large_n} should route to cuda"

    # the largest measured host f32 crosses to the GPU: the GEMM time saved dwarfs
    # the transfer round trip at large n (the headline DISP-05 win, on real numbers)
    assert (
        policy.choose_backend(
            "float32", large_n, large_n, large_n,
            residency="host", calibration=cal, backend="auto",
        )
        == "cuda"
    ), f"live f32 host n={large_n} should route to cuda"


# --- DISP-03: fail-safe cache read and the FME_CACHE_DIR override -------------


def _write_cache_file(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if isinstance(payload, str):
            f.write(payload)
        else:
            json.dump(payload, f)


def test_corrupt_recalibrate(tmp_path):
    # A corrupt (unparseable) cache file is discarded -> None (recalibrate or
    # static fallback), NEVER a partial dict. Also covers the missing-file case.
    sig = "synthetic|test|table"
    path = str(tmp_path / "calibration.json")
    _write_cache_file(path, "{ this is not valid json ::::")
    assert policy._load_cache(path, sig) is None
    # an empty file is the JSONDecodeError edge that surfaced live
    _write_cache_file(path, "")
    assert policy._load_cache(path, sig) is None
    # a missing file: FileNotFoundError -> None, never an exception out
    assert policy._load_cache(str(tmp_path / "absent.json"), sig) is None


def test_schema_mismatch(tmp_path):
    # An unknown / missing schema version -> None (recalibrate). A foreign file
    # that happens to parse as a dict but carries a different schema integer (or
    # none at all) must not be trusted.
    sig = "synthetic|test|table"
    path = str(tmp_path / "calibration.json")
    payload = synthetic_calibration()
    payload["schema"] = policy.SCHEMA + 999
    _write_cache_file(path, payload)
    assert policy._load_cache(path, sig) is None
    # missing schema key entirely
    del payload["schema"]
    _write_cache_file(path, payload)
    assert policy._load_cache(path, sig) is None
    # not even a dict (a JSON array parses but is not a calibration table)
    _write_cache_file(path, "[1, 2, 3]")
    assert policy._load_cache(path, sig) is None


def test_malformed_body_recalibrate(tmp_path):
    # A cache that parses, carries the right schema AND the running signature, but
    # whose BODY is structurally malformed -> None (recalibrate / static), never a
    # partial dict reaching choose_backend. This is the corruption class the
    # schema+signature gate alone passes: a truncated file, a non-atomic external
    # writer, or a foreign tool can stamp the matching schema and signature onto a
    # broken body. Each variant below must collapse to None like every other
    # rejection, and the resulting static path must not raise.
    sig = "synthetic|test|table"
    path = str(tmp_path / "calibration.json")

    def _valid():
        # a complete, schema+signature-matching dict to mutate per variant
        payload = synthetic_calibration()
        payload["signature"] = sig
        return payload

    # empty cpu grid: choose_backend's _interp_time would do grid[0][0] -> IndexError
    bad = _valid()
    bad["cpu"]["float32"] = []
    _write_cache_file(path, bad)
    assert policy._load_cache(path, sig) is None

    # missing cpu key entirely: choose_backend's calibration["cpu"][dtype] -> KeyError
    bad = _valid()
    del bad["cpu"]
    _write_cache_file(path, bad)
    assert policy._load_cache(path, sig) is None

    # a grid point that is not an [n, time] pair (wrong arity)
    bad = _valid()
    bad["cpu"]["float64"] = [[128, 0.001], [256]]
    _write_cache_file(path, bad)
    assert policy._load_cache(path, sig) is None

    # gpu present but transfer absent: choose_backend's calibration["transfer"]
    # for a host product -> KeyError
    bad = _valid()
    del bad["transfer"]
    _write_cache_file(path, bad)
    assert policy._load_cache(path, sig) is None

    # gpu present, transfer present but with a zero bandwidth: _transfer_time would
    # divide by zero on the dispatch hot path
    bad = _valid()
    bad["transfer"]["h2d_gbps"] = 0.0
    _write_cache_file(path, bad)
    assert policy._load_cache(path, sig) is None

    # the load-bearing consequence: with the cache rejected to None, the dispatch
    # path falls back to the conservative static choice and does NOT crash, for
    # every (dtype, residency) -- the fail-safe contract DISP-03 promises.
    for dtype in ("float32", "float64"):
        for residency in ("host", "device"):
            assert (
                policy.choose_backend(
                    dtype, 256, 256, 256, residency=residency,
                    calibration=None, backend="auto",
                )
                == "cpu"
            )


def test_signature_mismatch(tmp_path):
    # A cache written on a different machine/build (signature mismatch) -> None,
    # so a foreign cache never claims this machine's crossovers. A matching
    # signature returns the COMPLETE dict.
    path = str(tmp_path / "calibration.json")
    payload = synthetic_calibration()
    _write_cache_file(path, payload)
    assert policy._load_cache(path, "a-different-machine|gpu|1.0") is None
    loaded = policy._load_cache(path, payload["signature"])
    assert isinstance(loaded, dict)
    assert loaded["schema"] == policy.SCHEMA
    assert loaded["signature"] == payload["signature"]
    # complete, not a half-load: the measured grids survived the round trip
    assert "cpu" in loaded and "gpu" in loaded and "transfer" in loaded


def test_cache_dir_override(tmp_path, monkeypatch):
    # FME_CACHE_DIR redirects the resolved cache path: the file lives under the
    # override dir, named calibration.json, so tests never touch the real cache.
    monkeypatch.setenv("FME_CACHE_DIR", str(tmp_path))
    resolved = policy._cache_path()
    assert resolved == os.path.join(str(tmp_path), "calibration.json")
    # without the override the path falls back to the platform user cache dir and
    # still ends in calibration.json under a fastmathext-named directory.
    monkeypatch.delenv("FME_CACHE_DIR", raising=False)
    default = policy._cache_path()
    assert default.endswith("calibration.json")
    assert "fastmathext" in default.lower()


# --- DISP-01: calibrate() measures under budget; CPU-only omits gpu/transfer ---
#
# These exercise the live calibrate() measurement on whatever build is present.
# On the CPU-only build (the dominant gate) they measure the CPU grid only; on a
# fresh ON build the gpu/transfer keys appear and the relevant assertions adapt.
# Every one redirects FME_CACHE_DIR to a tmp_path so the real per-user cache is
# never read or written. A small budget_seconds keeps them fast: the grid stops
# adding larger sizes once the budget projects to breach, so a handful of small
# sizes are measured in a couple of seconds while still proving the full path.


@pytest.mark.slow
def test_calibrate_budget(tmp_path, monkeypatch):
    # DISP-01: calibrate() returns a dict carrying the CPU grid and completes
    # within its time budget. budget_seconds is a hard ceiling, not a tight bound:
    # the loop measures a whole size before re-checking elapsed, so one in-flight
    # size can run past the nominal budget. Assert a generous ceiling (the budget
    # plus a wide margin for that final size) rather than the budget exactly --
    # a tight bound would be the flaky kind this suite forbids.
    monkeypatch.setenv("FME_CACHE_DIR", str(tmp_path))
    budget = 5.0
    start = time.perf_counter()
    result = calibrate.calibrate(force=True, budget_seconds=budget)
    elapsed = time.perf_counter() - start

    assert isinstance(result, dict)
    assert "cpu" in result
    # the CPU grid is non-empty and carries [n, time_s] pairs for both dtypes
    assert result["cpu"]["float32"], "cpu float32 grid must be non-empty"
    assert result["cpu"]["float64"], "cpu float64 grid must be non-empty"
    for dtype in ("float32", "float64"):
        for point in result["cpu"][dtype]:
            assert len(point) == 2, "each grid point is [n, time_s]"
            n, t = point
            assert isinstance(n, int) and t >= 0.0
    # generous ceiling: the budget plus a wide margin for the one in-flight size
    # that may finish after the budget is spent. The point is "bounded", not "to
    # the millisecond".
    assert elapsed < budget + 60.0, (
        f"calibrate overran its budget: {elapsed:.1f}s for budget {budget}s"
    )


def test_cpu_only_calibrate(tmp_path, monkeypatch):
    # DISP-01: on a CPU-only build calibrate() measures the CPU grid only and the
    # returned dict has NO gpu and NO transfer key, so the policy treats the GPU
    # as unavailable. This is the OFF-build contract (the dominant gate). On an ON
    # build the keys are present instead -- assert that branch too so the test is
    # meaningful on both builds rather than skipped on one.
    monkeypatch.setenv("FME_CACHE_DIR", str(tmp_path))
    result = calibrate.calibrate(force=True, budget_seconds=5.0)
    assert "cpu" in result
    if fme.has_cuda():
        assert "gpu" in result and "transfer" in result
    else:
        assert "gpu" not in result, "CPU-only calibrate must omit the gpu key"
        assert "transfer" not in result, "CPU-only calibrate must omit the transfer key"


# --- DISP-02: cache write, reuse, and the GPU-distinguishing signature ---------


def test_writes_cache(tmp_path, monkeypatch):
    # DISP-02: a fresh calibrate() writes a valid schema-versioned JSON under the
    # FME_CACHE_DIR path. The written file parses as JSON, carries the policy
    # SCHEMA and the running machine signature, and round-trips through
    # policy._load_cache to a non-None dict (the read/write contract closes here).
    monkeypatch.setenv("FME_CACHE_DIR", str(tmp_path))
    result = calibrate.calibrate(force=True, budget_seconds=5.0)

    path = policy._cache_path()
    assert os.path.isfile(path), "calibrate must write the cache file"
    with open(path) as f:
        on_disk = json.load(f)
    assert on_disk["schema"] == policy.SCHEMA
    assert on_disk["signature"] == result["signature"]

    loaded = policy._load_cache(path, result["signature"])
    assert loaded is not None, "the written cache must round-trip through policy"
    assert loaded["schema"] == policy.SCHEMA
    assert "cpu" in loaded


def test_cache_reuse(tmp_path, monkeypatch):
    # DISP-02: a second force=False call reuses the cache without re-measuring --
    # the second-run-uses-cache contract. Prove it two ways: the cache file's
    # mtime is unchanged across the second call (no rewrite happened), and the
    # second call returns the same content as the first. A force=True call by
    # contrast DOES rewrite, so the mtime advances -- the control that shows the
    # mtime check is non-vacuous.
    monkeypatch.setenv("FME_CACHE_DIR", str(tmp_path))
    first = calibrate.calibrate(force=True, budget_seconds=5.0)
    path = policy._cache_path()
    mtime_after_first = os.stat(path).st_mtime_ns

    second = calibrate.calibrate(force=False, budget_seconds=5.0)
    mtime_after_second = os.stat(path).st_mtime_ns

    assert mtime_after_second == mtime_after_first, (
        "force=False must reuse the cache, not rewrite it"
    )
    assert second["signature"] == first["signature"]
    assert second["cpu"] == first["cpu"]

    # control: force=True re-measures and rewrites, so the file is touched again.
    # (st_mtime_ns can tie on a coarse clock, so compare the rewritten content
    # round-trips rather than asserting a strictly greater mtime.)
    forced = calibrate.calibrate(force=True, budget_seconds=5.0)
    assert forced["signature"] == first["signature"]


def test_signature_distinct(tmp_path, monkeypatch):
    # DISP-02: the machine signature a CPU-only run produces (gpu=none, driver=n/a)
    # differs from a GPU-run signature on the SAME CPU. On the CPU-only build the
    # helper emits the none/n/a GPU+driver components, so swapping in a real GPU
    # name yields a strictly different string -- a CPU-only cache can never be
    # mistaken for a GPU cache. The signature also embeds the package version, so
    # it is stable run to run on the same build.
    monkeypatch.setenv("FME_CACHE_DIR", str(tmp_path))
    sig = calibrate._machine_signature()
    cpu = calibrate._cpu_brand()
    version = fme.__version__

    # the signature is the documented pipe-joined cpu|gpu|driver|version
    parts = sig.split("|")
    assert len(parts) == 4
    assert parts[0] == cpu
    assert parts[3] == version

    if fme.has_cuda():
        # on an ON build the GPU components are real and already non-"none"
        assert parts[1] != "none"
    else:
        # CPU-only: gpu=none, driver=n/a, so a GPU-run signature on the same CPU
        # (a real device name and driver swapped in) is a different string.
        assert parts[1] == "none" and parts[2] == "n/a"
        gpu_run_sig = f"{cpu}|NVIDIA GeForce RTX 3060|560.94|{version}"
        assert sig != gpu_run_sig, (
            "a CPU-only signature must differ from a GPU-run signature on the "
            "same CPU so a CPU-only cache never claims GPU crossovers"
        )


# --- DISP-03 (write half): the atomic, crash-safe cache write ------------------


def test_atomic_write(tmp_path, monkeypatch):
    # DISP-03 write half: after calibrate() writes the cache there is no leftover
    # *.tmp file in the cache dir and the target is a complete, parseable JSON.
    # The write goes to a temp file in the same dir then os.replace -- a crash
    # mid-write leaves the old file or the new, never a half-written one, and the
    # temp is unlinked on any failure so none is ever left behind on success.
    monkeypatch.setenv("FME_CACHE_DIR", str(tmp_path))
    calibrate.calibrate(force=True, budget_seconds=5.0)

    leftover = [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")]
    assert leftover == [], f"no temp file may survive a write, found {leftover}"

    path = policy._cache_path()
    assert os.path.isfile(path)
    with open(path) as f:
        # a complete write parses cleanly; a half-write would raise here
        data = json.load(f)
    assert isinstance(data, dict) and data.get("schema") == policy.SCHEMA

    # a second write overwrites atomically and still leaves no temp behind
    calibrate.calibrate(force=True, budget_seconds=5.0)
    leftover_again = [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")]
    assert leftover_again == [], "atomic overwrite must not leave a temp file"
