"""CUDA suite skeleton. The whole file skips cleanly with a stated reason on a
CPU-only build or a machine without a usable device, so it stays green on the
WSL/GCC clone and on the default CPU-only Windows build, and only exercises the
device paths on the FME_ENABLE_CUDA=ON build (tests/CLAUDE.md: GPU tests skip
cleanly, never fail for absent hardware).

requires_cuda() (conftest) is the one gate: build flag plus a usable-device
probe. It migrates to fme.has_cuda() in 04-05 when that public runtime predicate
lands; the gate's meaning (built AND device present) does not change.

The -k area names (context, gemm, tolerance, roundtrip, residency, transfer,
routing, oom, fallback, has_cuda) match the phase research Test Map so the later
plans fill in each placeholder under its own name. They assert nothing about
device behavior yet beyond what this plan delivers: this plan proves the file
collects-and-skips cleanly and that the build-flag smoke passes everywhere.

Correctness tests added by later plans route every CPU-vs-GPU comparison through
assert_matmul_close (conftest), never an inline tolerance, exactly like the CPU
suite; f32 GPU compares against the f64 ground truth the same way.
"""

import os
import subprocess
import sys

import pytest

import fastmathext as fme
from conftest import requires_cuda

_CUDA_OK, _CUDA_REASON = requires_cuda()
# Reused by every device-touching test below. The build-flag smoke test is
# deliberately NOT decorated: it must pass on every build to prove the
# introspection surface exists and is a bool even when CUDA is off.
gpu = pytest.mark.skipif(not _CUDA_OK, reason=_CUDA_REASON)


def test_build_flag_smoke():
    # Runs everywhere, including CPU-only. has_cuda_build is the compile-time
    # flag; the runtime has_cuda() predicate is introduced in 04-05.
    assert isinstance(fme.has_cuda_build(), bool)


@gpu
def test_has_cuda_build_true_on_cuda_build():
    # Under the gate, reaching here means the build flag and a device are both
    # present, so the compile-time flag must be True.
    assert fme.has_cuda_build() is True


# A CUDA-built _core links cublas64_12.dll / cublasLt64_12.dll from the toolkit
# bin, which Python's secure DLL search ignores on PATH (3.8+). The in-process
# suite gets this for free from conftest, but the clean-shutdown child below is a
# bare interpreter that does NOT load conftest, so it registers the directory
# itself. This duplicates conftest's shim deliberately: it keeps the regression
# fence self-contained until 04-05 makes the package self-sufficient at import.
_CHILD_DLL_SETUP = (
    "import os\n"
    "_b = os.path.join(os.environ.get('CUDA_PATH', ''), 'bin')\n"
    "if os.name == 'nt' and _b.strip(os.sep) and os.path.isdir(_b):\n"
    "    os.add_dll_directory(_b)\n"
)

# Forces the context singleton to build (first _cuda_device_info call) and then
# exits normally. The atexit teardown must run cleanly on the way out: a
# static-destructor teardown would touch CUDA after the driver unloaded and
# intermittently nonzero-exit or print a CUDA shutdown error here. Asserting the
# device is present first guarantees the context actually built before exit.
_CLEAN_SHUTDOWN_SCRIPT = _CHILD_DLL_SETUP + (
    "import fastmathext as fme\n"
    "info = fme._core._cuda_device_info()\n"
    "assert info['present'] is True, info\n"
)


@gpu
def test_context_inits():
    # GPU-01: first use builds the process-lifetime singleton (device props,
    # streams, mempool, cublas/cublasLt handles). _cuda_device_info drives that
    # build and reports the bound device. Under the gate a usable device is
    # present, so the props must be populated; the dev box is sm_86 (cc 8.6) and
    # the contract floor is cc >= 7.0, so assert both the floor (the real
    # requirement) and the expected exact capability on this machine.
    info = fme._core._cuda_device_info()
    assert info["present"] is True, info
    assert (info["cc_major"], info["cc_minor"]) >= (7, 0), info
    assert (info["cc_major"], info["cc_minor"]) == (8, 6), info
    assert info["name"], "device name should be populated once props are queried"


@gpu
def test_context_clean_shutdown():
    # GPU-01 regression fence for the static-destruction-order trap: a child that
    # builds the context and exits must return 0 with no CUDA shutdown error in
    # stderr. Full env copy so Windows DLL resolution survives in the child
    # (the test_threads.py subprocess shape). If teardown ever regresses to a
    # static destructor, this is where it surfaces as a nonzero exit or a
    # cudaErrorCudartUnloading print.
    env = dict(os.environ)
    p = subprocess.run(
        [sys.executable, "-c", _CLEAN_SHUTDOWN_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr
    assert "cudaErrorCudartUnloading" not in p.stderr, p.stderr
    # Catch a native abort/crash at shutdown that still left returncode 0 on some
    # paths: none of these strings should appear on a clean teardown.
    for bad in ("Traceback", "terminate", "abort", "access violation"):
        assert bad not in p.stderr, p.stderr


@gpu
def test_gemm_device_placeholder():
    # GPU-02: f32/f64 device GEMM matches CPU via assert_matmul_close. 04-03.
    assert fme.has_cuda_build() is True


@gpu
def test_tolerance_placeholder():
    # GPU-02: TF32 off by default; result inside the f32 tolerance contract. 04-03.
    assert fme.has_cuda_build() is True


@gpu
def test_roundtrip_placeholder():
    # GPU-03: to_device -> from_device preserves bytes. 04-04.
    assert fme.has_cuda_build() is True


@gpu
def test_residency_placeholder():
    # GPU-03: matmul(Array, Array) -> Array; mixed residency raises TypeError. 04-04.
    assert fme.has_cuda_build() is True


@gpu
def test_transfer_placeholder():
    # GPU-04: async H2D/D2H with event sync before the host return. 04-04.
    assert fme.has_cuda_build() is True


@gpu
def test_routing_placeholder():
    # GPU-05: f32 device-resident routes to GPU; f64 never auto-routes. 04-05.
    assert fme.has_cuda_build() is True


@gpu
def test_oom_fallback_placeholder():
    # GPU-06: forced OOM (alloc > current free) -> warn once -> CPU fallback. 04-05.
    assert fme.has_cuda_build() is True
