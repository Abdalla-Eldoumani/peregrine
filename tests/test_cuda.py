"""CUDA suite skeleton. The whole file skips cleanly with a stated reason on a
CPU-only build or a machine without a usable device, so it stays green on the
WSL/GCC clone and on the default CPU-only Windows build, and only exercises the
device paths on the PG_ENABLE_CUDA=ON build (GPU tests skip cleanly, never fail
for absent hardware).

requires_cuda() (conftest) is the one gate: build flag plus a usable-device
probe, deferring to the public pg.has_cuda(); the gate means built AND device
present.

The -k area names are context, gemm, tolerance, roundtrip, residency, transfer,
routing, pinned, dlpack, staged, preflight, fallback, async_fault, teardown, and
has_cuda. Sanitizer runs deselect async_fault, which faults the device on
purpose.

Correctness tests route every CPU-vs-GPU comparison through
assert_matmul_close (conftest), never an inline tolerance, exactly like the CPU
suite; f32 GPU compares against the f64 ground truth the same way.
"""

import ctypes
import os
import subprocess
import sys

import numpy as np
import pytest

import peregrine as pg
from conftest import assert_matmul_close, requires_cuda

_CUDA_OK, _CUDA_REASON = requires_cuda()
# Reused by every device-touching test below. The build-flag smoke test is
# deliberately NOT decorated: it must pass on every build to prove the
# introspection surface exists and is a bool even when CUDA is off.
gpu = pytest.mark.skipif(not _CUDA_OK, reason=_CUDA_REASON)

_MiB = 1 << 20


# The CUDA runtime the package itself loaded, reached through ctypes so a test
# can consume a DLPack capsule and inspect host memory the way an independent
# CUDA library would, without one being installed.
def _cudart():
    if os.name == "nt":
        path = os.path.join(os.environ.get("CUDA_PATH", ""), "bin", "cudart64_12.dll")
    else:
        path = "libcudart.so.12"
    try:
        rt = ctypes.CDLL(path)
    except OSError as exc:
        pytest.skip(f"cuda runtime not loadable through ctypes: {exc}")
    rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    rt.cudaMemcpy.restype = ctypes.c_int
    rt.cudaPointerGetAttributes.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    rt.cudaPointerGetAttributes.restype = ctypes.c_int
    return rt


class _PointerAttributes(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("device", ctypes.c_int),
        ("devicePointer", ctypes.c_void_p),
        ("hostPointer", ctypes.c_void_p),
    ]


def _is_page_locked(arr):
    # cudaMemoryTypeHost (1) is memory from cudaHostAlloc; ordinary pageable
    # memory reads as cudaMemoryTypeUnregistered (0).
    attr = _PointerAttributes()
    assert _cudart().cudaPointerGetAttributes(ctypes.byref(attr), arr.ctypes.data) == 0
    return attr.type == 1


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_capsule_name = ctypes.pythonapi.PyCapsule_GetName
_capsule_name.argtypes = [ctypes.py_object]
_capsule_name.restype = ctypes.c_char_p
_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_pointer.restype = ctypes.c_void_p


def _consume_dltensor(capsule, dtype):
    # What a DLPack consumer does with a legacy capsule: check the name, read
    # the tensor fields, and copy the device data out. The capsule is not
    # renamed, so its own destructor still releases the producer.
    assert type(capsule).__name__ == "PyCapsule"
    assert _capsule_name(capsule) == b"dltensor"
    t = _DLManagedTensor.from_address(_capsule_pointer(capsule, b"dltensor")).dl_tensor
    assert (t.device.device_type, t.device.device_id) == (2, 0)
    assert t.ndim == 2
    itemsize = np.dtype(dtype).itemsize
    assert (t.dtype.code, t.dtype.bits, t.dtype.lanes) == (2, 8 * itemsize, 1)
    shape = (t.shape[0], t.shape[1])
    if t.strides:
        assert (t.strides[0], t.strides[1]) == (shape[1], 1)
    host = np.empty(shape, dtype)
    if host.size:
        # cudaMemcpyDeviceToHost; the legacy default stream it runs on is the
        # consumer stream of stream=1.
        assert _cudart().cudaMemcpy(host.ctypes.data, t.data + t.byte_offset, host.nbytes, 2) == 0
    return host


def test_has_cuda_runtime():
    # Runs everywhere, including CPU-only and WSL: the public runtime predicate
    # must exist and return a bool on every build, the cross-platform half of the
    # introspection surface. has_cuda() composes the build flag AND a usable-device probe
    # (driver, count>0, cc>=7.0); on a CPU-only build it is simply False, never an
    # exception, so auto-mode code can branch on it without a try/except.
    assert isinstance(pg.has_cuda(), bool)


@gpu
def test_has_cuda_true_on_cuda_build_with_device():
    # Under the gate, the build flag and a usable device are both present, so the
    # public runtime predicate must be True. The private build flag is also True
    # here, but the public surface is has_cuda(); has_cuda_build is gone.
    assert pg.has_cuda() is True
    assert not hasattr(pg, "has_cuda_build")


# A CUDA-built _core links cublas64_12.dll / cublasLt64_12.dll from the toolkit
# bin, which Python's secure DLL search ignores on PATH (3.8+). The in-process
# suite gets this for free from conftest, but the clean-shutdown child below is a
# bare interpreter that does NOT load conftest, so it registers the directory
# itself. This duplicates conftest's shim deliberately: it keeps the regression
# fence self-contained even before any package code runs.
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
    "import peregrine as pg\n"
    "info = pg._core._cuda_device_info()\n"
    "assert info['present'] is True, info\n"
)

# Regression: a pg.Array bound to a MODULE GLOBAL is still alive when the
# interpreter finalizes, so its ~Array runs AFTER the atexit teardown destroyed
# the context stream. Before the teardown-tolerant free_device, that
# ~Array called cudaFreeAsync on the destroyed stream (a use-after-teardown the
# sanitizer surfaces) through the throwing CHECK macro (a throw from a destructor
# at finalization). The child binds the Array at module scope, never deletes it,
# and exits: a clean teardown returns 0 with no CUDA shutdown error and no abort.
_GLOBAL_ARRAY_SHUTDOWN_SCRIPT = _CHILD_DLL_SETUP + (
    "import numpy as np\n"
    "import peregrine as pg\n"
    # Module-global device array: deliberately never freed before exit so ~Array
    # runs at finalization, the dangerous post-teardown ordering.
    "g = pg.to_device(np.ones((8, 8), dtype=np.float32))\n"
    "assert g.shape == (8, 8), g.shape\n"
)


@gpu
def test_context_inits():
    # First use builds the process-lifetime singleton (device props,
    # streams, mempool, cublas/cublasLt handles). _cuda_device_info drives that
    # build and reports the bound device. Under the gate a usable device is
    # present, so the props must be populated; the dev box is sm_86 (cc 8.6) and
    # the contract floor is cc >= 7.0, so assert both the floor (the real
    # requirement) and the expected exact capability on this machine.
    info = pg._core._cuda_device_info()
    assert info["present"] is True, info
    assert (info["cc_major"], info["cc_minor"]) >= (7, 0), info
    assert (info["cc_major"], info["cc_minor"]) == (8, 6), info
    assert info["name"], "device name should be populated once props are queried"


@gpu
def test_context_clean_shutdown():
    # Regression fence for the static-destruction-order trap: a child that
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
    # Regression fence: a pg.Array held in a module global is alive at
    # interpreter finalization, so ~Array runs AFTER the atexit teardown
    # destroyed the context stream. The teardown-tolerant free_device must make
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
    # f32 device GEMM via cublasSgemm and the no-transpose trick matches
    # NumPy. assert_matmul_close compares f32 against the f64 ground truth it
    # recomputes from a, b, exactly like the CPU suite -- the single toleranced
    # path, no inline rtol. Rectangular k so a wrong leading dimension in the
    # operand-swap would surface as a shape or value error, not hide behind n==k.
    rng = np.random.default_rng(20260613 + n)
    k = n + 5
    a = rng.standard_normal((n, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    got = pg._core._gemm_host(a, b)
    assert got.dtype == np.float32
    assert_matmul_close(got, a @ b, a, b)


@gpu
@pytest.mark.parametrize("n", _GEMM_SIZES)
def test_gemm_f64_matches_numpy(n):
    # f64 device GEMM via cublasDgemm, identical no-transpose layout. f64
    # is computed on the device for the forced-resident path even though it never
    # auto-routes here (the FP64 trap is a dispatch rule); the kernel must
    # still be correct. Through the f64 arm of the same tolerance contract.
    rng = np.random.default_rng(40260613 + n)
    k = n + 5
    a = rng.standard_normal((n, k))
    b = rng.standard_normal((k, n))
    got = pg._core._gemm_host(a, b)
    assert got.dtype == np.float64
    assert_matmul_close(got, a @ b, a, b)


@gpu
def test_gemm_zero_dims():
    # Zero-sized dim semantics: the guards run before any
    # cuBLAS launch (an empty GEMM must not reach cuBLAS). (m,0)@(0,n) is an m x n
    # matrix of exact zeros; (0,k)@(k,n) is an empty (0,n). Exact equality, not
    # toleranced: a k=0 product is exactly zero and an empty result has no
    # elements. Both dtypes, since the guards are dtype-generic.
    for dtype in (np.float32, np.float64):
        a = np.zeros((4, 0), dtype=dtype)
        b = np.zeros((0, 3), dtype=dtype)
        got = pg._core._gemm_host(a, b)
        assert got.shape == (4, 3)
        assert got.dtype == dtype
        np.testing.assert_array_equal(got, np.zeros((4, 3), dtype=dtype))

        a0 = np.zeros((0, 5), dtype=dtype)
        b0 = (np.arange(15, dtype=dtype)).reshape(5, 3)
        got0 = pg._core._gemm_host(a0, b0)
        assert got0.shape == (0, 3)
        assert got0.dtype == dtype


# TF32 is opt-in (PEREGRINE_ALLOW_TF32=1) and read ONCE at context init, so it must be
# set in the environment before import -- the subprocess + full-env-copy idiom
# from test_fallback/test_threads. The child reproduces the same seeded inputs,
# runs the f32 device GEMM under TF32, and writes both its result and the f64
# ground truth so the parent can measure the TF32 error against the DEFAULT_MATH
# error. The child DLL shim is required because a bare interpreter does not load
# conftest (which registers the toolkit bin for the cublas DLL).
_TF32_SCRIPT = _CHILD_DLL_SETUP + (
    "import sys\n"
    "import numpy as np\n"
    "import peregrine as pg\n"
    "rng = np.random.default_rng(515)\n"
    "k = 517\n"
    "a = rng.standard_normal((512, k)).astype(np.float32)\n"
    "b = rng.standard_normal((k, 512)).astype(np.float32)\n"
    "got = pg._core._gemm_host(a, b)\n"
    "np.save(sys.argv[1], got)\n"
)


@gpu
def test_tf32_off_by_default_matches_contract(tmp_path):
    # Math-mode policy. Two halves:
    #   1. The default path (no PEREGRINE_ALLOW_TF32) is DEFAULT_MATH and passes the
    #      standard f32 tolerance contract.
    #   2. A subprocess with PEREGRINE_ALLOW_TF32=1 takes the TF32 path. TF32's 10-bit
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
    got_default = pg._core._gemm_host(a, b)
    assert_matmul_close(got_default, a @ b, a, b)

    # Half 2: the TF32 child reruns the identical seeded GEMM with the flag set.
    out_path = tmp_path / "got_tf32.npy"
    env = dict(os.environ, PEREGRINE_ALLOW_TF32="1")
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
        f"{err_default:.3e}; the PEREGRINE_ALLOW_TF32 path may not be engaged"
    )


@gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_to_from_device_roundtrip(dtype):
    # to_device copies a host array up to a pg.Array, from_device
    # copies it back (D2H on the context stream, waited for before the return). A pure
    # copy must round-trip byte-IDENTICAL: this is a transfer, not a computation, so
    # exact equality is the right assertion (assert_matmul_close is for the kernel
    # path, not for a memcpy). Both dtypes -- to_device accepts f64 too (it is just
    # memory; the f64-never-auto-routes rule is a dispatch concern). Rectangular
    # so a row/col swap in the shape plumbing would surface.
    a = np.random.default_rng(20260613).standard_normal((7, 5)).astype(dtype)
    x = pg._core.to_device(a)
    assert x.shape == a.shape
    assert x.dtype == np.dtype(dtype)
    b = pg._core.from_device(x)
    assert np.array_equal(a, b), "round-trip must preserve bytes exactly"


@gpu
def test_from_device_returns_ndarray():
    # from_device returns a numpy.ndarray, not a pg.Array -- the
    # residency-out type (ndarray in -> ndarray out). The device handle does not
    # leak back out of from_device.
    a = np.ones((3, 4), dtype=np.float32)
    b = pg._core.from_device(pg._core.to_device(a))
    assert isinstance(b, np.ndarray)
    assert not isinstance(b, type(pg._core.to_device(a)))


@gpu
def test_from_device_result_ownership():
    # The result is a writeable NumPy array over a page-locked buffer lent by
    # peregrine's pool: NumPy does not own the memory, base holds the loan, and
    # the loan lasts as long as any view of the array does.
    a = np.random.default_rng(5).standard_normal((33, 17)).astype(np.float32)
    r = pg.from_device(pg.to_device(a))
    assert type(r) is np.ndarray and r.dtype == np.float32 and r.shape == (33, 17)
    assert r.flags.writeable and r.flags.c_contiguous and r.flags.aligned
    assert not r.flags.owndata
    assert r.base is not None
    assert _is_page_locked(r)
    assert r.tobytes() == a.tobytes()
    view = r[1:3]
    loaned = pg._core._pinned_pool_stats()["loaned"]
    del r
    assert pg._core._pinned_pool_stats()["loaned"] == loaned
    assert view.tobytes() == a[1:3].tobytes()
    del view
    assert pg._core._pinned_pool_stats()["loaned"] == loaned - 4096


@gpu
def test_pinned_pool_reuses_a_returned_buffer():
    x = pg.to_device(np.ones((256, 1024), np.float32))
    r = pg.from_device(x)
    address = r.ctypes.data
    stats = pg._core._pinned_pool_stats()
    del r
    after = pg._core._pinned_pool_stats()
    assert after == dict(stats, loaned=stats["loaned"] - _MiB)
    again = pg.from_device(x)
    assert again.ctypes.data == address
    assert _is_page_locked(again)
    assert pg._core._pinned_pool_stats() == stats


@gpu
def test_pinned_pool_cap_counts_loans_and_falls_back_to_pageable():
    cap = pg._core._pinned_pool_stats()["cap"]
    assert cap == 256 * _MiB
    block = 64 * _MiB
    ones = np.ones((64, _MiB // 4), np.float32)
    x = pg.to_device(ones)
    held = []
    # Buffers on loan count against the cap: once they fill it the next result
    # is ordinary pageable memory, and the pool's books do not change for it.
    for _ in range(cap // block + 1):
        r = pg.from_device(x)
        if not _is_page_locked(r):
            break
        held.append(r)
    stats = pg._core._pinned_pool_stats()
    assert not _is_page_locked(r)
    assert np.array_equal(r, ones)
    assert stats["loaned"] == stats["reserved"] <= cap < stats["loaned"] + block
    # Ending one loan makes room again, and the same buffer comes back.
    address = held[-1].ctypes.data
    del held[-1]
    again = pg.from_device(x)
    assert _is_page_locked(again) and again.ctypes.data == address
    assert np.array_equal(again, ones)
    # With nothing on loan the pool still holds the cap in cached buffers; a
    # result of another size gives some of them back to fit.
    del held, again, r
    small = pg.from_device(pg.to_device(np.full((3, 5), 2.0, np.float32)))
    assert _is_page_locked(small)
    assert np.array_equal(small, np.full((3, 5), 2.0, np.float32))
    assert pg._core._pinned_pool_stats()["reserved"] <= cap


@gpu
def test_to_device_from_page_locked_memory_copies_before_returning():
    # DMA reads a page-locked source after the copy call returns, unlike a
    # pageable one, which the driver stages first. to_device waits for such a
    # copy, so changing a from_device result right after uploading it cannot
    # reach the device copy.
    a = np.random.default_rng(3).standard_normal((2048, 4096)).astype(np.float32)
    r = pg.from_device(pg.to_device(a))
    assert _is_page_locked(r)
    y = pg.to_device(r)
    r[...] = 0.0
    assert pg.from_device(y).tobytes() == a.tobytes()


@gpu
def test_dlpack_device_is_cuda():
    # __dlpack_device__ is (kDLCUDA, 0), and __dlpack__ returns the capsule
    # itself, as the array API requires, not an object a consumer would have to
    # export again.
    a = np.ones((4, 4), dtype=np.float32)
    x = pg._core.to_device(a)

    dev_type, dev_id = x.__dlpack_device__()
    assert dev_type == 2, f"expected kDLCUDA (2), got {dev_type}"
    assert dev_id == 0

    capsule = x.__dlpack__()
    assert type(capsule).__name__ == "PyCapsule"
    assert _capsule_name(capsule) == b"dltensor"


@gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(7, 5), (1, 1), (0, 3)])
def test_dlpack_consumer_reads_the_device_buffer(dtype, shape):
    a = np.random.default_rng(31).standard_normal(shape).astype(dtype)
    x = pg.to_device(a)
    for stream in (None, 1, 2, 0x7FFF0000, -1):
        if stream == -1:
            pg._core._cuda_sync()  # stream=-1: ordering is the consumer's job
        got = _consume_dltensor(x.__dlpack__(stream=stream), dtype)
        assert got.tobytes() == a.tobytes(), stream
    # every argument the protocol defines, at a value the producer can honor
    capsule = x.__dlpack__(stream=None, max_version=(1, 0), dl_device=(2, 0), copy=False)
    assert _consume_dltensor(capsule, dtype).tobytes() == a.tobytes()


@gpu
def test_dlpack_capsule_keeps_the_buffer_alive():
    # The capsule owns a reference to the Array, so deleting the Array frees
    # nothing while the capsule lives; a same-size allocation made then cannot
    # land on the exported buffer.
    a = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    x = pg.to_device(a)
    capsule = x.__dlpack__()
    del x
    filler = [pg.to_device(np.full((64, 64), -1.0, np.float32)) for _ in range(4)]
    assert _consume_dltensor(capsule, np.float32).tobytes() == a.tobytes()
    del filler


@gpu
def test_dlpack_waits_for_queued_work():
    # A 2048-square float32 product takes milliseconds on this GPU and
    # is still running when the device matmul returns; __dlpack__ with any
    # stream but -1 returns only after it (with -1 the consumer orders it).
    a = np.ones((2048, 2048), np.float32)
    xa = pg.to_device(a)
    pg._core._cuda_sync()
    for stream in (None, 1, 2):
        xc = pg.matmul(xa, xa)
        xc.__dlpack__(stream=stream)
        assert pg._core._cuda_stream_idle(), stream


@gpu
def test_dlpack_argument_errors():
    x = pg.to_device(np.ones((2, 3), np.float32))
    with pytest.raises(ValueError, match="stream must be -1, 1, 2, or a stream handle"):
        x.__dlpack__(stream=0)
    with pytest.raises(ValueError, match="stream must be -1, 1, 2, or a stream handle"):
        x.__dlpack__(stream=-2)
    with pytest.raises(TypeError, match="stream must be None or an int"):
        x.__dlpack__(stream=1.0)
    with pytest.raises(TypeError, match="stream must be None or an int"):
        x.__dlpack__(stream=True)
    with pytest.raises(BufferError, match="copy=True is not supported"):
        x.__dlpack__(copy=True)
    for device in [(1, 0), (2, 1), (13, 0)]:
        with pytest.raises(BufferError, match="cannot be exported to another device"):
            x.__dlpack__(dl_device=device)
    # keyword-only, as the protocol defines it
    with pytest.raises(TypeError):
        x.__dlpack__(None)
    with pytest.raises(TypeError):
        x.__dlpack__(strem=None)
    # numpy has no device memory: its import refuses the CUDA tensor
    with pytest.raises(BufferError):
        np.from_dlpack(x)
    with pytest.raises(BufferError):
        np.from_dlpack(x, device="cpu")


@gpu
def test_device_matmul_roundtrip():
    # The REQUIRED device-resident entry: matmul(Array, Array) -> Array,
    # device-in/device-out. to_device both f32 operands, run the device GEMM on the
    # device buffers (no host staging), assert the result is a pg.Array, then
    # from_device it and compare to NumPy through assert_matmul_close (the single
    # toleranced path; f32 is judged against the f64 ground truth exactly like the
    # CPU suite). This proves gemm + transfers + the device matmul entry
    # compose end to end on device buffers -- the entry the Python wrapper layers
    # residency semantics on top of. Rectangular k so a wrong leading dimension
    # would show as a value error, not hide behind a square shape.
    rng = np.random.default_rng(20260614)
    a = rng.standard_normal((33, 17)).astype(np.float32)
    b = rng.standard_normal((17, 29)).astype(np.float32)

    xa = pg._core.to_device(a)
    xb = pg._core.to_device(b)
    xc = pg._core._matmul_device(xa, xb)
    assert type(xc) is type(xa), "device matmul must return a pg.Array"
    assert xc.shape == (33, 29)

    got = pg._core.from_device(xc)
    assert isinstance(got, np.ndarray)
    assert_matmul_close(got, a @ b, a, b)


@gpu
@pytest.mark.parametrize("n", [250, 512])
def test_device_matmul_transfer_ordering(n):
    # Regression fence for the use-before-ready race: the uploads, the output
    # allocation, the GEMM, and the download must reach the device in that order,
    # with no host wait between them. With two unordered streams cuBLAS once read
    # operands and wrote the output before their allocation and copy completed,
    # correct untimed but flagged by compute-sanitizer
    # --track-stream-ordered-races all as use-before-alloc. One stream orders
    # them now; this runs the exact to_device -> device matmul -> from_device
    # path at a non-trivial size and checks the values, and the sanitizer run
    # over this file is the hard proof of the ordering.
    rng = np.random.default_rng(20260617 + n)
    k = n + 3  # rectangular so a wrong leading dimension would surface as a value error
    a = rng.standard_normal((n, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    xc = pg._core._matmul_device(pg._core.to_device(a), pg._core.to_device(b))
    assert xc.shape == (n, n)
    assert_matmul_close(pg._core.from_device(xc), a @ b, a, b)


@gpu
def test_mixed_residency_raises():
    # Residency: exactly one device-resident operand is the verbatim
    # mixed-residency TypeError, never a silent host<->device
    # transfer. Both operand orders, since the wrapper must reject either side
    # being the device one.
    rng = np.random.default_rng(7)
    a = rng.standard_normal((6, 4)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    xa = pg.to_device(a)
    with pytest.raises(TypeError, match="one input is on cuda and one on cpu"):
        pg.matmul(xa, b)
    with pytest.raises(TypeError, match="one input is on cuda and one on cpu"):
        pg.matmul(a, pg.to_device(b))


@gpu
def test_f32_device_routes_to_gpu():
    # Two f32 device-resident operands route to the GPU. The return type
    # is the proof the device path was taken (residency-out: pg.Array in ->
    # pg.Array out). from_device of the result then matches NumPy through the
    # single toleranced path. Rectangular k so a wrong leading dimension surfaces.
    rng = np.random.default_rng(20260615)
    k = 37
    a = rng.standard_normal((29, k)).astype(np.float32)
    b = rng.standard_normal((k, 41)).astype(np.float32)
    xc = pg.matmul(pg.to_device(a), pg.to_device(b))
    assert isinstance(xc, pg.Array), "f32 device-resident matmul must return a pg.Array"
    assert xc.shape == (29, 41)
    assert_matmul_close(pg.from_device(xc), a @ b, a, b)


@gpu
def test_staged_host_operands_in_any_layout():
    # A host float32 pair staged to the device is uploaded in C order whatever
    # its layout, so F-ordered, transposed, and reversed operands give the bits
    # of the same values in C order, as on the CPU.
    rng = np.random.default_rng(20260618)
    a = rng.standard_normal((29, 37)).astype(np.float32)
    b = rng.standard_normal((37, 41)).astype(np.float32)
    pg.set_backend("cuda")
    try:
        ref = pg.matmul(a, b)
        assert isinstance(ref, np.ndarray)
        for x, y in [
            (np.asfortranarray(a), b),
            (a, np.ascontiguousarray(b.T).T),
            (np.ascontiguousarray(a[::-1])[::-1], np.ascontiguousarray(b[:, ::-1])[:, ::-1]),
        ]:
            assert pg.matmul(x, y).tobytes() == ref.tobytes()
    finally:
        pg.set_backend("auto")


# Legacy-killer sizes, a skinny product, and the empty and k=0 edges.
_STAGED_SHAPES = [(1, 1, 1), (7, 5, 3), (250, 333, 7), (512, 515, 333), (4, 0, 3), (0, 5, 3)]


@gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("m,k,n", _STAGED_SHAPES)
def test_staged_entry_bits_equal_the_three_crossing_path(dtype, m, k, n):
    # The single native staged call runs the same cuBLAS product on the same
    # stream as upload, device matmul, and download done one call at a time, so
    # the bits are the same, with or without out.
    rng = np.random.default_rng(m * 10007 + k * 101 + n)
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    three = pg._core.from_device(
        pg._core._matmul_device(pg._core.to_device(a), pg._core.to_device(b))
    )
    staged = pg._core._gemm_host(a, b)
    assert staged.dtype == dtype and staged.shape == (m, n)
    assert staged.tobytes() == three.tobytes()
    # a staged result is lent by the page-locked pool, like a from_device one
    assert not staged.flags.owndata and _is_page_locked(staged)
    out = np.full((m, n), np.nan, dtype)
    assert pg._core._gemm_host(a, b, out) is None
    assert out.tobytes() == three.tobytes()


@gpu
def test_f64_never_auto_routes():
    # Hard exclusion (the FP64 trap): a HOST f64 matmul returns an ndarray
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
    # auto-route to the GPU happened -- a host f64 never becomes a pg.Array.
    got = pg.matmul(a, b)
    assert isinstance(got, np.ndarray)
    assert not isinstance(got, pg.Array)
    assert got.dtype == np.float64
    assert_matmul_close(got, a @ b, a, b)

    # Forced device-resident f64: allowed, computes correctly (slowly). Returns an
    # pg.Array, proving forced f64 reaches the device path; the exclusion is only
    # on AUTO-routing a host f64, not on a user's explicit device placement.
    xc = pg.matmul(pg.to_device(a), pg.to_device(b))
    assert isinstance(xc, pg.Array)
    assert_matmul_close(pg.from_device(xc), a @ b, a, b)


# Device memory exhaustion runs in a child so the once-only warning and the
# large transient state stay out of this process. PEREGRINE_PREFLIGHT_FREE_CAP,
# read once when the child builds its CUDA context, caps the free VRAM the
# pre-flight believes in; at 0 every request of 256 MiB or more is refused
# before anything is allocated, whatever the display and other processes hold,
# so the refusal is deterministic. A rank-1 (N, 1) @ (1, N) product keeps the
# inputs tiny and puts the 256 MiB on the output, and its CPU fallback is an
# outer product that takes well under a second. The child DLL shim is required
# because a bare interpreter does not load conftest's toolkit-bin registration.
_OOM_SCRIPT = _CHILD_DLL_SETUP + (
    "import warnings\n"
    "import numpy as np\n"
    "import peregrine as pg\n"
    "from conftest import assert_matmul_close\n"
    "N = 8192\n"
    "rng = np.random.default_rng(909)\n"
    "a = rng.standard_normal((N, 1)).astype(np.float32)\n"
    "b = rng.standard_normal((1, N)).astype(np.float32)\n"
    "corner = (a, b[:, :64])\n"
    "pg.set_backend('cuda')\n"
    # Below 256 MiB nothing is checked, so this product runs on the GPU: its
    # result is lent by the page-locked pool, where a CPU result is NumPy's own.
    "small = pg.matmul(a[:64], b[:, :64])\n"
    "assert not small.flags.owndata\n"
    "with warnings.catch_warnings(record=True) as caught:\n"
    "    warnings.simplefilter('always')\n"
    "    got = pg.matmul(a, b)\n"
    "    out = np.full((N, N), np.nan, np.float32)\n"
    "    assert pg.matmul(a, b, out=out) is out\n"
    "msgs = [str(w.message) for w in caught]\n"
    "assert_matmul_close(got[:, :64], corner[0] @ corner[1], *corner)\n"
    "assert out.tobytes() == got.tobytes()\n"
    # The staged path still has the host operands: one warning naming the
    # error, and the products computed on the CPU (NumPy-owned results).
    "assert len(msgs) == 1, msgs\n"
    "assert msgs[0] == ('cuda matmul failed (cudaErrorMemoryAllocation), '\n"
    "                   'falling back to cpu for this session'), msgs[0]\n"
    "assert got.flags.owndata\n"
    # Device operands have no host copy to fall back on: the refusal raises.
    "xa, xb = pg.to_device(a), pg.to_device(b)\n"
    "for call in (lambda: pg.matmul(xa, xb), lambda: pg.to_device(np.empty((N, N), np.float32))):\n"
    "    try:\n"
    "        call()\n"
    "    except RuntimeError as exc:\n"
    "        assert str(exc).startswith('cudaErrorMemoryAllocation'), exc\n"
    "    else:\n"
    "        raise AssertionError('a 256 MiB device allocation passed the pre-flight')\n"
    "print('OOM_FALLBACK_OK')\n"
)


def _run_child(script, *args, env=None, timeout=300):
    env = dict(os.environ) if env is None else env  # full copy: DLL resolution
    # the tests dir on sys.path so the child can import conftest's helpers
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.abspath(__file__)), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@gpu
def test_preflight_refusal_warns_once_and_falls_back():
    env = dict(os.environ, PEREGRINE_PREFLIGHT_FREE_CAP="0")
    p = _run_child(_OOM_SCRIPT, env=env)
    assert p.returncode == 0, p.stderr
    assert "OOM_FALLBACK_OK" in p.stdout, (p.stdout, p.stderr)


# With the real free VRAM (about 5 GB here) the pre-flight admits requests over
# its threshold: a 256 MiB upload, and a staged product whose buffers total
# 256 MiB, run on the GPU without a warning. A matrix-vector product keeps the
# device work small enough for a sanitizer run.
_ADMIT_SCRIPT = _CHILD_DLL_SETUP + (
    "import warnings\n"
    "import numpy as np\n"
    "import peregrine as pg\n"
    "warnings.simplefilter('error')\n"
    "N = 8192\n"
    "pg.set_backend('cuda')\n"
    "a = np.full((N, N), 0.5, np.float32)\n"
    "x = pg.to_device(a)\n"
    "v = pg.matmul(a, np.ones((N, 1), np.float32))\n"
    "assert not v.flags.owndata\n"
    "assert np.array_equal(v, np.full((N, 1), N / 2, np.float32))\n"
    "print('ADMIT_OK')\n"
)


@gpu
def test_preflight_admits_what_fits():
    env = dict(os.environ)
    env.pop("PEREGRINE_PREFLIGHT_FREE_CAP", None)
    p = _run_child(_ADMIT_SCRIPT, env=env)
    assert p.returncode == 0, p.stderr
    assert "ADMIT_OK" in p.stdout, (p.stdout, p.stderr)


# A kernel that traps faults the device asynchronously: the launch returns, and
# the error surfaces at the next host fence. The fault is sticky, so it runs in
# a child. compute-sanitizer reports the trap it is built to catch, so sanitizer
# runs of this file deselect it (-k "not async_fault").
_FAULT_SCRIPT = _CHILD_DLL_SETUP + (
    "import warnings\n"
    "import numpy as np\n"
    "import peregrine as pg\n"
    "a = np.ones((64, 64), np.float32)\n"
    "x = pg.to_device(a)\n"
    "pg._core._cuda_inject_fault()\n"
    "def raises_named(call):\n"
    "    try:\n"
    "        call()\n"
    "    except RuntimeError as exc:\n"
    "        name = str(exc).split(' ', 1)[0]\n"
    "        assert name.startswith('cudaError'), exc\n"
    "        return name\n"
    "    raise AssertionError('no error surfaced')\n"
    # the next host fence reports the fault by its cuda error name
    "name = raises_named(lambda: pg.from_device(x))\n"
    "assert raises_named(lambda: x.__dlpack__()) == name\n"
    "assert raises_named(lambda: pg.matmul(x, x)) == name\n"
    # host operands: warn once, compute on the CPU, stay quiet afterwards
    "pg.set_backend('cuda')\n"
    "with warnings.catch_warnings(record=True) as caught:\n"
    "    warnings.simplefilter('always')\n"
    "    got = pg.matmul(a, a)\n"
    "    again = pg.matmul(a, a)\n"
    "msgs = [str(w.message) for w in caught]\n"
    "assert msgs == [f'cuda matmul failed ({name}), falling back to cpu for this session'], msgs\n"
    "assert np.array_equal(got, np.full((64, 64), 64.0, np.float32))\n"
    "assert np.array_equal(again, got)\n"
    "print('FAULT_OK', name)\n"
)


@gpu
def test_async_fault_surfaces_at_the_next_fence():
    p = _run_child(_FAULT_SCRIPT)
    assert p.returncode == 0, p.stderr
    assert "FAULT_OK cudaError" in p.stdout, (p.stdout, p.stderr)


# The exit teardown runs early here, then a NumPy array lent by the page-locked
# pool is released: the loan ends without the buffer going back to the pool
# (the process exit reclaims it) and without a CUDA call, and the child still
# exits cleanly through the real teardown, which finds nothing left to do.
_TEARDOWN_SCRIPT = _CHILD_DLL_SETUP + (
    "import numpy as np\n"
    "import peregrine as pg\n"
    "x = pg.to_device(np.ones((256, 1024), np.float32))\n"
    "r = pg.from_device(x)\n"
    "before = pg._core._pinned_pool_stats()\n"
    "assert before['loaned'] >= 1 << 20, before\n"
    "pg._core._cuda_teardown()\n"
    "del r\n"
    "after = pg._core._pinned_pool_stats()\n"
    "assert after == before, (before, after)\n"
    "print('TEARDOWN_OK')\n"
)


@gpu
def test_pinned_release_after_teardown_leaks_cleanly():
    p = _run_child(_TEARDOWN_SCRIPT)
    assert p.returncode == 0, p.stderr
    assert "TEARDOWN_OK" in p.stdout, (p.stdout, p.stderr)
    for bad in ("Traceback", "terminate", "abort", "access violation", "cudaError"):
        assert bad not in p.stderr, p.stderr
