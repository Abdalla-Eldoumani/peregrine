"""CUDA suite skeleton. The whole file skips cleanly with a stated reason on a
CPU-only build or a machine without a usable device, so it stays green on the
WSL/GCC clone and on the default CPU-only Windows build, and only exercises the
device paths on the FME_ENABLE_CUDA=ON build (GPU tests skip cleanly, never fail
for absent hardware).

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


def test_has_cuda_runtime():
    # Runs everywhere, including CPU-only and WSL: the public runtime predicate
    # must exist and return a bool on every build, the cross-platform half of the
    # API-04 surface. has_cuda() composes the build flag AND a usable-device probe
    # (driver, count>0, cc>=7.0); on a CPU-only build it is simply False, never an
    # exception, so auto-mode code can branch on it without a try/except.
    assert isinstance(fme.has_cuda(), bool)


@gpu
def test_has_cuda_true_on_cuda_build_with_device():
    # Under the gate, the build flag and a usable device are both present, so the
    # public runtime predicate must be True. The private build flag is also True
    # here, but the public surface is has_cuda(); has_cuda_build is gone (API-04).
    assert fme.has_cuda() is True
    assert not hasattr(fme, "has_cuda_build")


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

# CR-01 regression: an fme.Array bound to a MODULE GLOBAL is still alive when the
# interpreter finalizes, so its ~Array runs AFTER the atexit teardown destroyed
# the context's transfer stream. Before the teardown-tolerant free_device, that
# ~Array called cudaFreeAsync on the destroyed stream (a use-after-teardown the
# sanitizer surfaces) through the throwing CHECK macro (a throw from a destructor
# at finalization). The child binds the Array at module scope, never deletes it,
# and exits: a clean teardown returns 0 with no CUDA shutdown error and no abort.
_GLOBAL_ARRAY_SHUTDOWN_SCRIPT = _CHILD_DLL_SETUP + (
    "import numpy as np\n"
    "import fastmathext as fme\n"
    # Module-global device array: deliberately never freed before exit so ~Array
    # runs at finalization, the dangerous post-teardown ordering.
    "g = fme.to_device(np.ones((8, 8), dtype=np.float32))\n"
    "assert g.shape == (8, 8), g.shape\n"
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
def test_module_global_array_clean_shutdown():
    # CR-01 regression fence: an fme.Array held in a module global is alive at
    # interpreter finalization, so ~Array runs AFTER the atexit teardown
    # destroyed the transfer stream. The teardown-tolerant free_device must make
    # that a no-op (g_torn_down skips the free; the live path never throws), so
    # the child exits 0 with no CUDA shutdown error and no destructor-throw abort.
    # Without the fix this is where the use-after-teardown + throw-from-destructor
    # surfaces (a cudaErrorCudartUnloading print, a nonzero exit, or an abort).
    env = dict(os.environ)
    p = subprocess.run(
        [sys.executable, "-c", _GLOBAL_ARRAY_SHUTDOWN_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr
    assert "cudaErrorCudartUnloading" not in p.stderr, p.stderr
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
    # GPU-02 zero-sized dim semantics: the guards run before any
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
    #      numbers (the numeric policy).
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
@pytest.mark.parametrize("n", [250, 512])
def test_device_matmul_transfer_ordering(n):
    # GPU-07 regression fence for the cross-stream use-before-ready race: to_device
    # allocates + copies operands on the TRANSFER stream while cuBLAS runs the GEMM
    # on the COMPUTE stream, and the matmul output buffer is allocated on transfer
    # too. With no ordering between the streams cuBLAS read operands and wrote the
    # output before the transfer-stream alloc/copy completed -- correct untimed
    # (the copy usually won the race) but flagged by compute-sanitizer
    # --track-stream-ordered-races all as use-before-alloc, returning garbage under
    # instrumentation. The fix fences the transfer stream before the GEMM and the
    # compute stream after it. This test runs the exact to_device -> device matmul
    # -> from_device path at a non-trivial size and asserts the result matches
    # NumPy through the single toleranced path; the hard proof is the sanitizer
    # gate (it FAILS under --track-stream-ordered-races all without the fences),
    # but a value check here keeps the path exercised in the normal suite too.
    rng = np.random.default_rng(20260617 + n)
    k = n + 3  # rectangular so a wrong leading dimension would surface as a value error
    a = rng.standard_normal((n, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    xc = fme._core.matmul(fme._core.to_device(a), fme._core.to_device(b))
    assert xc.shape == (n, n)
    assert_matmul_close(fme._core.from_device(xc), a @ b, a, b)


@gpu
def test_mixed_residency_raises():
    # GPU-06 residency: exactly one device-resident operand is the verbatim
    # mixed-residency TypeError, never a silent host<->device
    # transfer. Both operand orders, since the wrapper must reject either side
    # being the device one.
    rng = np.random.default_rng(7)
    a = rng.standard_normal((6, 4)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    xa = fme.to_device(a)
    with pytest.raises(TypeError, match="one input is on cuda and one on cpu"):
        fme.matmul(xa, b)
    with pytest.raises(TypeError, match="one input is on cuda and one on cpu"):
        fme.matmul(a, fme.to_device(b))


@gpu
def test_f32_device_routes_to_gpu():
    # GPU-05: two f32 device-resident operands route to the GPU. The return type
    # is the proof the device path was taken (residency-out: fme.Array in ->
    # fme.Array out). from_device of the result then matches NumPy through the
    # single toleranced path. Rectangular k so a wrong leading dimension surfaces.
    rng = np.random.default_rng(20260615)
    k = 37
    a = rng.standard_normal((29, k)).astype(np.float32)
    b = rng.standard_normal((k, 41)).astype(np.float32)
    xc = fme.matmul(fme.to_device(a), fme.to_device(b))
    assert isinstance(xc, fme.Array), "f32 device-resident matmul must return an fme.Array"
    assert xc.shape == (29, 41)
    assert_matmul_close(fme.from_device(xc), a @ b, a, b)


@gpu
def test_f64_never_auto_routes():
    # GPU-05 hard exclusion (the FP64 trap): a HOST f64 matmul returns an ndarray
    # computed on the CPU and never silently lands on the GPU (GA106 FP64 is 1/64
    # FP32, measured slower than this CPU). A forced device-resident f64 (both
    # operands explicitly to_device'd) is allowed and computes correctly -- forced
    # is permitted, auto is not. Both arms through assert_matmul_close.
    rng = np.random.default_rng(20260616)
    k = 23
    a = rng.standard_normal((31, k))
    b = rng.standard_normal((k, 19))
    assert a.dtype == np.float64 and b.dtype == np.float64

    # Host f64: ndarray out, CPU path. The return TYPE is the assertion that no
    # auto-route to the GPU happened -- a host f64 never becomes an fme.Array.
    got = fme.matmul(a, b)
    assert isinstance(got, np.ndarray)
    assert not isinstance(got, fme.Array)
    assert got.dtype == np.float64
    assert_matmul_close(got, a @ b, a, b)

    # Forced device-resident f64: allowed, computes correctly (slowly). Returns an
    # fme.Array, proving forced f64 reaches the device path; the exclusion is only
    # on AUTO-routing a host f64, not on a user's explicit device placement.
    xc = fme.matmul(fme.to_device(a), fme.to_device(b))
    assert isinstance(xc, fme.Array)
    assert_matmul_close(fme.from_device(xc), a @ b, a, b)


# Forced-OOM runs in a subprocess so the once-only warn sentinel and the large
# transient device state are isolated from the rest of the suite (a recurring
# sentinel would suppress the warning in a later test; the device allocation must
# be freed on a clean child exit, never wedging the display GPU). The child sizes
# the allocation against CURRENT free VRAM (nvidia-smi memory.free), NEVER the
# nominal 6144 (Pitfall 5): free fluctuates with the display. A rank-1 product
# (K=1) keeps the INPUTS tiny so the OOM fires on the matmul OUTPUT allocation,
# the path the auto-mode fallback wraps. The child DLL shim is required because a
# bare interpreter does not load conftest's toolkit-bin registration -- though
# 04-05 also gives the package its own import-time discovery, the shim stays
# belt-and-suspenders for a child that imports before any package code runs.
#
# The OOM is forced by RESERVING free VRAM in device-resident arrays first, so the
# matmul output need only exceed the residual free to OOM -- NOT the whole multi-GB
# free pool. This keeps the test bounded in TIME. The reserve is ADAPTIVE, not a
# single fixed-gap allocation: the mempool keeps ReleaseThreshold=UINT64_MAX
# (context.cu), so after prior GPU tests cudaFreeAsync returns freed buffers to the
# POOL not the OS and nvidia-smi over-reports truly-available VRAM. A one-shot
# free-minus-gap reserve then undershoots, the residual free stays above the gap,
# and the output alloc SUCCEEDS instead of OOMing -- the flake that passed this
# test in isolation but failed it in-suite. So after a free-read-seeded first
# chunk, the child PROBES by allocating an output-sized array in a bounded loop:
# while a probe succeeds it is retained as more reserve, driving the residual free
# down until a probe throws cudaErrorMemoryAllocation, which guarantees the real
# output alloc that follows OOMs regardless of pool-retained memory.
# The output is a SQUARE (N, N), rank-1 (K=1): a square output keeps BOTH the row
# AND column counts at ~N (N ~ 16384 for a ~1GB f32 output), so the CPU fallback
# is a normal cache-friendly outer product (seconds measured). A skinny (8, huge-n)
# output of the same BYTE size is pathological in the BLIS kernel -- millions of
# n-columns with M=8, K=1 packs and loops per column and ran for MINUTES, blowing
# past the subprocess timeout once free VRAM was large enough to size n into the
# tens of millions. Same byte size, same OOM trigger, vastly different CPU cost;
# the square shape is the one that stays bounded. The OOM path under test is
# identical -- alloc_device's cudaMemGetInfo pre-flight throws
# cudaErrorMemoryAllocation on the OUTPUT alloc because the reserve leaves less
# than request+margin free; the auto-mode fallback then computes the FULL host
# result (square, fast) before the corner check slices it.
_OOM_SCRIPT = _CHILD_DLL_SETUP + (
    "import subprocess, sys, warnings\n"
    "import numpy as np\n"
    "import fastmathext as fme\n"
    # SQUARE output (N, N) sized to gap+256MB = ~1GB: far past the 64MB pre-flight
    # margin once the reserve below leaves the residual free under this size, so the
    # OUTPUT alloc reliably throws cudaErrorMemoryAllocation, while N ~ 16384 keeps
    # the CPU fallback a fast cache-friendly outer product (seconds) rather than a
    # skinny millions-of-columns one (minutes, timeout-prone). Defined first so the
    # adaptive reserve can probe at exactly this byte size.
    "gap_mib = 768\n"
    "out_bytes = (gap_mib + 256) * 1024 * 1024\n"
    "N = int((out_bytes // 4) ** 0.5)\n"  # f32 square output (N, N)
    # Allocate the tiny rank-1 inputs FIRST, while VRAM is still plentiful, so the
    # exhaustion below can never starve them: (N, 1) and (1, N) are N floats each.
    # Only the (N, N) OUTPUT alloc then OOMs -- making the trigger deterministic
    # rather than racing the input allocs against the residual free. Seeded so the
    # parent need not see the data.
    "rng = np.random.default_rng(909)\n"
    "a = rng.standard_normal((N, 1)).astype(np.float32)\n"
    "b = rng.standard_normal((1, N)).astype(np.float32)\n"
    "xa = fme.to_device(a)\n"
    "xb = fme.to_device(b)\n"
    # Adaptive reserve: do NOT trust a single nvidia-smi free read. The mempool is
    # created with ReleaseThreshold=UINT64_MAX (context.cu), so after prior GPU
    # tests cudaFreeAsync returns freed buffers to the POOL, not the OS, and
    # nvidia-smi over-reports truly-available VRAM. A fixed free-gap reserve then
    # undershoots, the residual free exceeds the gap, and the output alloc SUCCEEDS
    # instead of OOMing (the in-suite flake). Instead, occupy VRAM in device-
    # resident chunks kept alive in `reserve`, then PROBE by trying to allocate an
    # output-sized (N, N) array: while a probe SUCCEEDS the residual free is still
    # >= the output request, so keep that probe as additional reserve and loop. The
    # moment a probe throws cudaErrorMemoryAllocation the residual free is below the
    # output request, so the real output alloc that follows is guaranteed to OOM --
    # regardless of how much memory the pool retained. np.empty (not RNG) keeps the
    # host cost to the allocation itself; each H2D at PCIe rate is sub-second.
    "out = subprocess.run(['nvidia-smi','--query-gpu=memory.free',"
    "'--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)\n"
    "free_mib = int(out.stdout.strip().splitlines()[0])\n"
    # Seed the reserve with one large chunk sized from the (untrusted) free read
    # down to the gap, so the common case reaches the probe loop in one allocation
    # rather than many small steps. max(0, ...) guards a free read already below the
    # gap; an empty (0,0) array allocates nothing and the probe loop does the work.
    "reserve = []\n"
    "seed_bytes = max(0, (free_mib - gap_mib)) * 1024 * 1024\n"
    "sside = int((seed_bytes // 4) ** 0.5)\n"
    "reserve.append(fme.to_device(np.empty((sside, sside), dtype=np.float32)))\n"
    # Probe at the output byte size. Each successful probe is retained as reserve
    # (occupying the residual free), so the loop monotonically drives the residual
    # free below the output request. Bounded iterations so a child can never wedge
    # the display GPU: at ~1GB per retained probe this covers the whole card many
    # times over, and on this hardware the first probe after the seed already fails.
    "for _ in range(64):\n"
    "    try:\n"
    "        probe = fme.to_device(np.empty((N, N), dtype=np.float32))\n"
    "    except RuntimeError:\n"
    # A probe alloc threw: residual free is now below the output request, so the
    # output alloc below will OOM. Stop growing the reserve.
    "        break\n"
    "    reserve.append(probe)\n"
    "with warnings.catch_warnings(record=True) as caught:\n"
    "    warnings.simplefilter('always')\n"
    "    got = fme.matmul(xa, xb)\n"
    # The fallback ran on the CPU: the result is a host ndarray, not an fme.Array.
    "assert isinstance(got, np.ndarray), type(got)\n"
    "assert not isinstance(got, fme.Array)\n"
    # Exactly one warning, carrying the verbatim fallback token with the cuda
    # error name (cudaErrorMemoryAllocation) inside the parentheses.
    "msgs = [str(w.message) for w in caught]\n"
    "fb = [s for s in msgs if 'falling back to cpu for this session' in s]\n"
    "assert len(fb) == 1, msgs\n"
    "assert fb[0].startswith('cuda matmul failed ('), fb[0]\n"
    "assert 'cudaErrorMemoryAllocation' in fb[0], fb[0]\n"
    # The CPU fallback result is correct: it is the rank-1 outer product. Verify a
    # small CORNER through the single toleranced path so the check stays cheap
    # (the full f64 ground truth of a free-sized result would double the memory).
    "corner = got[:, :64]\n"
    "ac = a\n"
    "bc = b[:, :64]\n"
    "from conftest import assert_matmul_close\n"
    "assert_matmul_close(corner, ac @ bc, ac, bc)\n"
    "print('OOM_FALLBACK_OK')\n"
)


@gpu
def test_forced_oom_warns_once_and_falls_back(tmp_path):
    # GPU-06: a device-resident f32 matmul whose output allocation exceeds CURRENT
    # free VRAM raises cudaErrorMemoryAllocation in the native pre-flight, which
    # the wrapper catches in auto mode, warns ONCE with the verbatim fallback
    # token, and recomputes on the CPU. The subprocess isolates the warn-once
    # sentinel and frees the device state on exit (no display wedge). The hard
    # gates: the warn-once token fired exactly once AND the CPU-fallback result is
    # correct (a corner verified through assert_matmul_close). The child runs the
    # whole assertion and prints a sentinel; the parent gates on a clean exit.
    env = dict(os.environ)  # full env copy: Windows DLL resolution in the child
    # conftest sets the tests dir on sys.path for the parent; the child needs it
    # too so `from conftest import assert_matmul_close` resolves.
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.abspath(__file__)), env.get("PYTHONPATH", "")]
    )
    p = subprocess.run(
        [sys.executable, "-c", _OOM_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert p.returncode == 0, p.stderr
    assert "OOM_FALLBACK_OK" in p.stdout, (p.stdout, p.stderr)
