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
Later waves fill these in under these exact names; the names do not change.

    DISP-01 (no calibration at import; bounded calibrate):
        calibrate_budget          calibration completes within its time budget
        cpu_only_calibrate        a CPU-only build calibrates without a device
    DISP-02 (cache write; signature):
        writes_cache              calibrate persists a cache file
        cache_reuse               a second run reuses the cache, does not redo
        signature_distinct        a different machine signature -> a distinct key
    DISP-03 (corrupt/mismatch -> recalibrate, never half-load; cache override):
        corrupt_recalibrate       a corrupt cache is discarded and recalibrated
        schema_mismatch           an unknown schema version -> recalibrate
        signature_mismatch        a signature mismatch -> recalibrate
        atomic_write              the cache write is atomic, never half-written
        cache_dir_override        FME_CACHE_DIR redirects the cache location
    DISP-04 (conservative static fallback; transfer cost in crossover):
        static_fallback           no cache -> a conservative CPU-preferring choice
        transfer_cost             the crossover accounts for host<->device transfer
    DISP-05 (routing decisions; the FP64 trap):
        f64_host_cpu              host f64 never auto-routes to the GPU
        f32_256_host_cpu          host f32 n=256 stays on the CPU (overhead floor)
        f32_4096_host_gpu         host f32 n=4096 routes to the GPU
        f32_device_gpu            f32 device-resident routes to the GPU
        set_backend               an explicit backend choice is honored
        env_backend               FME_BACKEND (subprocess, env-before-import)
        live_crossover            real RTX 3060 calibration reproduces DISP-05 (@gpu)
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import fastmathext as fme
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
