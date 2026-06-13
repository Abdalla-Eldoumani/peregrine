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

import numpy as np
import pytest

import fastmathext as fme
from conftest import assert_matmul_close, requires_cuda

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


# Legacy-killer sizes (1, 7, 333 are not multiples of 4, where the archived
# kernel corrupted) plus round powers, all far inside the TDR budget (n=512 f32
# is microseconds; the watchdog is ~2s). The device GEMM must match NumPy at
# every one, square and rectangular.
_GEMM_SIZES = [1, 7, 64, 250, 333, 512]


@gpu
@pytest.mark.parametrize("n", _GEMM_SIZES)
def test_gemm_f32_matches_numpy(n):
    # GPU-02: f32 device GEMM via cublasSgemm and the no-transpose trick matches
    # NumPy. assert_matmul_close compares f32 against the f64 ground truth it
    # recomputes from a, b, exactly like the CPU suite -- the single toleranced
    # path, no inline rtol. Rectangular k so a wrong leading dimension in the
    # operand-swap would surface as a shape or value error, not hide behind n==k.
    rng = np.random.default_rng(20260613 + n)
    k = n + 5
    a = rng.standard_normal((n, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    got = fme._core._gemm_host(a, b)
    assert got.dtype == np.float32
    assert_matmul_close(got, a @ b, a, b)


@gpu
@pytest.mark.parametrize("n", _GEMM_SIZES)
def test_gemm_f64_matches_numpy(n):
    # GPU-02: f64 device GEMM via cublasDgemm, identical no-transpose layout. f64
    # is computed on the device for the forced-resident path even though it never
    # auto-routes here (the FP64 trap is a dispatch rule, 04-05); the kernel must
    # still be correct. Through the f64 arm of the same tolerance contract.
    rng = np.random.default_rng(40260613 + n)
    k = n + 5
    a = rng.standard_normal((n, k))
    b = rng.standard_normal((k, n))
    got = fme._core._gemm_host(a, b)
    assert got.dtype == np.float64
    assert_matmul_close(got, a @ b, a, b)


@gpu
def test_gemm_zero_dims():
    # GPU-02 / DESIGN_SYSTEM zero-sized dim semantics: the guards run before any
    # cuBLAS launch (an empty GEMM must not reach cuBLAS). (m,0)@(0,n) is an m x n
    # matrix of exact zeros; (0,k)@(k,n) is an empty (0,n). Exact equality, not
    # toleranced: a k=0 product is exactly zero and an empty result has no
    # elements. Both dtypes, since the guards are dtype-generic.
    for dtype in (np.float32, np.float64):
        a = np.zeros((4, 0), dtype=dtype)
        b = np.zeros((0, 3), dtype=dtype)
        got = fme._core._gemm_host(a, b)
        assert got.shape == (4, 3)
        assert got.dtype == dtype
        np.testing.assert_array_equal(got, np.zeros((4, 3), dtype=dtype))

        a0 = np.zeros((0, 5), dtype=dtype)
        b0 = (np.arange(15, dtype=dtype)).reshape(5, 3)
        got0 = fme._core._gemm_host(a0, b0)
        assert got0.shape == (0, 3)
        assert got0.dtype == dtype


# TF32 is opt-in (FME_ALLOW_TF32=1) and read ONCE at context init, so it must be
# set in the environment before import -- the subprocess + full-env-copy idiom
# from test_fallback/test_threads. The child reproduces the same seeded inputs,
# runs the f32 device GEMM under TF32, and writes both its result and the f64
# ground truth so the parent can measure the TF32 error against the DEFAULT_MATH
# error. The child DLL shim is required because a bare interpreter does not load
# conftest (which registers the toolkit bin for the cublas DLL).
_TF32_SCRIPT = _CHILD_DLL_SETUP + (
    "import sys\n"
    "import numpy as np\n"
    "import fastmathext as fme\n"
    "rng = np.random.default_rng(515)\n"
    "k = 517\n"
    "a = rng.standard_normal((512, k)).astype(np.float32)\n"
    "b = rng.standard_normal((k, 512)).astype(np.float32)\n"
    "got = fme._core._gemm_host(a, b)\n"
    "np.save(sys.argv[1], got)\n"
)


@gpu
def test_tf32_off_by_default_matches_contract(tmp_path):
    # GPU-02 math-mode policy. Two halves:
    #   1. The default path (no FME_ALLOW_TF32) is DEFAULT_MATH and passes the
    #      standard f32 tolerance contract.
    #   2. A subprocess with FME_ALLOW_TF32=1 takes the TF32 path. TF32's 10-bit
    #      mantissa is ~830x looser (measured), so its raw abs error against the
    #      f64 truth must MATERIALLY exceed the DEFAULT_MATH error. That proves
    #      TF32 is actually engaged and is why it is off by default. The TF32
    #      result is deliberately NOT run through assert_matmul_close: TF32 is
    #      opt-in and excluded from the standard contract and from headline
    #      numbers (DESIGN_SYSTEM numeric policy).
    rng = np.random.default_rng(515)
    k = 517
    a = rng.standard_normal((512, k)).astype(np.float32)
    b = rng.standard_normal((k, 512)).astype(np.float32)

    # Half 1: the in-process default build is DEFAULT_MATH; passes the contract.
    got_default = fme._core._gemm_host(a, b)
    assert_matmul_close(got_default, a @ b, a, b)

    # Half 2: the TF32 child reruns the identical seeded GEMM with the flag set.
    out_path = tmp_path / "got_tf32.npy"
    env = dict(os.environ, FME_ALLOW_TF32="1")
    p = subprocess.run(
        [sys.executable, "-c", _TF32_SCRIPT, str(out_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr
    got_tf32 = np.load(out_path)

    truth = a.astype(np.float64) @ b.astype(np.float64)
    err_default = float(np.abs(got_default.astype(np.float64) - truth).max())
    err_tf32 = float(np.abs(got_tf32.astype(np.float64) - truth).max())
    # Materially looser: an order of magnitude is a conservative floor against
    # the measured ~830x, enough to prove the mode actually changed without
    # pinning the exact ratio (which drifts with clocks and data).
    assert err_tf32 > 10.0 * err_default, (
        f"tf32 error {err_tf32:.3e} not materially looser than default "
        f"{err_default:.3e}; the FME_ALLOW_TF32 path may not be engaged"
    )


@gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_to_from_device_roundtrip(dtype):
    # GPU-03 / GPU-04: to_device copies a host array up to an fme.Array, from_device
    # copies it back (D2H on the transfer stream, synced before the return). A pure
    # copy must round-trip byte-IDENTICAL: this is a transfer, not a computation, so
    # exact equality is the right assertion (assert_matmul_close is for the kernel
    # path, not for a memcpy). Both dtypes -- to_device accepts f64 too (it is just
    # memory; the f64-never-auto-routes rule is a dispatch concern, 04-05). Rectangular
    # so a row/col swap in the shape plumbing would surface.
    a = np.random.default_rng(20260613).standard_normal((7, 5)).astype(dtype)
    x = fme._core.to_device(a)
    assert x.shape == a.shape
    assert x.dtype == np.dtype(dtype)
    b = fme._core.from_device(x)
    assert np.array_equal(a, b), "round-trip must preserve bytes exactly"


@gpu
def test_from_device_returns_ndarray():
    # GPU-03: from_device returns a numpy.ndarray, not an fme.Array -- the
    # residency-out type (ndarray in -> ndarray out). The device handle does not
    # leak back out of from_device.
    a = np.ones((3, 4), dtype=np.float32)
    b = fme._core.from_device(fme._core.to_device(a))
    assert isinstance(b, np.ndarray)
    assert not isinstance(b, type(fme._core.to_device(a)))


@gpu
def test_dlpack_device_is_cuda():
    # GPU-03: the fme.Array dlpack export reports a CUDA device. __dlpack_device__
    # returns (device_type, device_id); the DLPack enum for CUDA is kDLCUDA == 2.
    # __dlpack__ produces a dlpack-protocol object whose capsule is a "dltensor" on
    # that CUDA device. We assert the device tag and the capsule, NOT a
    # numpy.from_dlpack zero-copy: numpy has no device memory and cannot pull a CUDA
    # dltensor (RESEARCH line 271), and a device-aware consumer (cupy/torch) is not
    # required to be installed.
    a = np.ones((4, 4), dtype=np.float32)
    x = fme._core.to_device(a)

    dev_type, dev_id = x.__dlpack_device__()
    assert dev_type == 2, f"expected kDLCUDA (2), got {dev_type}"
    assert dev_id == 0

    # nanobind's array_api export returns an nb_ndarray that is itself a dlpack
    # producer; its __dlpack__ yields the actual PyCapsule named "dltensor". Reach
    # through one optional layer so the assertion holds whether __dlpack__ returns
    # the capsule directly or the producer object.
    exported = x.__dlpack__()
    capsule = exported.__dlpack__() if hasattr(exported, "__dlpack__") else exported
    assert "dltensor" in repr(capsule), f"expected a dltensor capsule, got {capsule!r}"


@gpu
def test_device_matmul_roundtrip():
    # GPU-03 (the REQUIRED device-resident entry): matmul(Array, Array) -> Array,
    # device-in/device-out. to_device both f32 operands, run the device GEMM on the
    # device buffers (no host staging), assert the result is an fme.Array, then
    # from_device it and compare to NumPy through assert_matmul_close (the single
    # toleranced path; f32 is judged against the f64 ground truth exactly like the
    # CPU suite). This proves gemm (04-03) + transfers + the device matmul entry
    # compose end to end on device buffers -- the entry 04-05's wrapper layers
    # residency semantics on top of. Rectangular k so a wrong leading dimension
    # would show as a value error, not hide behind a square shape.
    rng = np.random.default_rng(20260614)
    a = rng.standard_normal((33, 17)).astype(np.float32)
    b = rng.standard_normal((17, 29)).astype(np.float32)

    xa = fme._core.to_device(a)
    xb = fme._core.to_device(b)
    xc = fme._core.matmul(xa, xb)
    assert type(xc) is type(xa), "device matmul must return an fme.Array"
    assert xc.shape == (33, 29)

    got = fme._core.from_device(xc)
    assert isinstance(got, np.ndarray)
    assert_matmul_close(got, a @ b, a, b)


@gpu
def test_routing_placeholder():
    # GPU-05: f32 device-resident routes to GPU; f64 never auto-routes. 04-05.
    assert fme.has_cuda_build() is True


@gpu
def test_oom_fallback_placeholder():
    # GPU-06: forced OOM (alloc > current free) -> warn once -> CPU fallback. 04-05.
    assert fme.has_cuda_build() is True
