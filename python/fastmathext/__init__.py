"""fastmathext: heterogeneous linear algebra with a NumPy-compatible surface.

The native core takes zero-copy views of C-contiguous float32 and float64
arrays. This wrapper supplies the NumPy-style ergonomics on top: dtype
promotion, contiguity normalization, and clear errors.
"""

from __future__ import annotations

import os
import threading
import warnings

import numpy as np

# Import-time CUDA DLL discovery (Windows). A CUDA-enabled _core links
# cublas64_12.dll / cublasLt64_12.dll from the toolkit bin, and since Python 3.8
# the secure DLL search for extension modules ignores PATH, so a bare
# `import fastmathext` under the ON build would raise "DLL load failed". Register
# the toolkit bin before importing _core so the package is self-sufficient at
# import on a CUDA build -- including under a bare `python -c` child that never
# loads the test conftest (test_threads.py / test_fallback.py spawn those).
# Best-effort and Windows-only: on the CPU-only build _core has no CUDA
# dependency so this changes nothing, and a missing CUDA_PATH simply skips. This
# is directory registration only (os.add_dll_directory), NOT a driver load or
# context init, so it costs microseconds and respects the <50ms import budget --
# "CUDA init" is the first device op, which this is not.
if os.name == "nt":
    _cuda_bin = os.path.join(os.environ.get("CUDA_PATH", ""), "bin")
    if _cuda_bin.strip(os.sep) and os.path.isdir(_cuda_bin):
        try:
            os.add_dll_directory(_cuda_bin)
        except OSError:
            pass

from . import policy
from ._core import _has_cuda_build, cpu_features
from ._core import axpby as _axpby_native
from ._core import fma3 as _fma3_native
from ._core import matmul as _matmul_native
from ._core import scaled_relu as _scaled_relu_native
from ._core import sum_all as _sum_all_native
from ._core import sum_axis as _sum_axis_native
from ._core import transpose as _transpose_native

# calibrate() is re-exported as the public fme.calibrate. The binding is explicit
# (not left to attribute fallthrough) BECAUSE a submodule named calibrate exists:
# `from .calibrate import calibrate` binds the FUNCTION into the package namespace,
# shadowing the submodule so `fme.calibrate(...)` is the callable the design doc
# names, not the module. calibrate.py runs nothing at import (no measurement, no
# device init -- it only defines functions and imports stdlib + this package), so
# this costs a one-time module compile and nothing measurable on the hot path; the
# cache read and any measurement stay lazy (calibrate() body / first dispatch).
from .calibrate import calibrate

# Device-side entries exist only on the CUDA build. Import them behind the build
# flag so the OFF build imports cleanly; bind Array to a private sentinel class
# nothing is an instance of when CUDA is absent, so the residency isinstance
# checks in matmul return False on OFF without a separate code path.
if _has_cuda_build():
    from ._core import Array
    from ._core import _cuda_device_probe
    from ._core import from_device as _from_device_native
    from ._core import to_device as _to_device_native
else:
    class Array:  # noqa: D401 - sentinel; the real device type is CUDA-build-only
        """Placeholder for fme.Array on a CPU-only build.

        The device array type exists only when fastmathext is built with
        ``FME_ENABLE_CUDA=ON``. On a CPU-only build this sentinel exists so the
        public name is importable and ``isinstance(x, fme.Array)`` is always
        False; constructing it raises, since there is no device to hold.
        """

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "cuda backend requested but no build "
                "(fastmathext was built with FME_ENABLE_CUDA=OFF)"
            )

    _cuda_device_probe = None

__version__ = "3.0.0.dev0"
__all__ = [
    "matmul",
    "transpose",
    "sum",
    "mean",
    "axpby",
    "fma3",
    "scaled_relu",
    "calibrate",
    "set_backend",
    "to_device",
    "from_device",
    "Array",
    "cpu_features",
    "has_cuda",
    "__version__",
]

# Checked per operand before np.result_type runs: promotion would silently
# launder float16 and bool up to an accepted float dtype, hiding a conversion
# the caller never asked for. complex and object have no kernel path at all.
_REJECT = frozenset(
    map(np.dtype, ("bool", "float16", "complex64", "complex128", "object"))
)

# The dtypes the native kernel takes directly, with no promotion. A pair already
# on one of these, both 2-D and C-contiguous, needs none of the policy chain, so
# the fast path forwards it to the kernel untouched.
_ACCEPTED = frozenset(map(np.dtype, ("float32", "float64")))


def _prepare(x, name: str, op: str = "matmul") -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{op}: {name} must be 2-dimensional, got ndim={arr.ndim}")
    return arr


def _normalize(arr: np.ndarray, common: np.dtype) -> np.ndarray:
    # ascontiguousarray is a no-op for arrays that already qualify, which is
    # the zero-copy fast path. Strided views pay one copy here until the
    # native kernels grow stride support.
    return np.ascontiguousarray(arr, dtype=common)


def _resolve_dtype(arr: np.ndarray, op: str) -> np.dtype:
    # The single-operand half of the one dtype policy, identical in spirit to
    # matmul's two-operand chain: reject the laundering set first, then take the
    # result dtype (a single operand's result_type is its own dtype), then the
    # integer-only float64 fallback. Reused verbatim by transpose, sum, and mean
    # so all four ops share one reject set and one promotion table.
    if arr.dtype in _REJECT:
        raise TypeError(
            f"{op}: unsupported dtype {arr.dtype.name}, "
            "expected float32 or float64 (ints promote)"
        )
    common = np.result_type(arr.dtype)
    if common not in (np.dtype(np.float32), np.dtype(np.float64)):
        # the fallback is integer-only, mirroring matmul: coercing an extended
        # float or complex dtype (longdouble, clongdouble where distinct) to
        # float64 would silently drop precision or imaginary parts
        if not np.issubdtype(common, np.integer):
            raise TypeError(
                f"{op}: unsupported dtype {common.name}, "
                "expected float32 or float64 (ints promote)"
            )
        common = np.dtype(np.float64)
    return common


def _resolve_dtype_multi(arrs, op: str) -> np.dtype:
    # The N-operand form of _resolve_dtype, the one dtype policy generalized
    # across the fused operands: reject the laundering set PER OPERAND first (so
    # promotion can never launder a float16/bool up into an accepted dtype),
    # naming the offending dtype, then np.result_type across every operand, then
    # the same integer-only float64 fallback. fma3 promotes across all three
    # arrays; axpby across two; scaled_relu across one. Reuses the exact reject
    # set and fallback so all ops -- matmul, transpose, sum, mean, and the fused
    # family -- share one reject table and one promotion table.
    for arr in arrs:
        if arr.dtype in _REJECT:
            raise TypeError(
                f"{op}: unsupported dtype {arr.dtype.name}, "
                "expected float32 or float64 (ints promote)"
            )
    common = np.result_type(*(arr.dtype for arr in arrs))
    if common not in (np.dtype(np.float32), np.dtype(np.float64)):
        # integer-only fallback, identical to _resolve_dtype / matmul: coercing an
        # extended float or complex result (longdouble, clongdouble where the
        # platform keeps them distinct) to float64 would silently drop precision
        # or imaginary parts, the exact laundering _REJECT exists to stop.
        if not np.issubdtype(common, np.integer):
            raise TypeError(
                f"{op}: unsupported dtype {common.name}, "
                "expected float32 or float64 (ints promote)"
            )
        common = np.dtype(np.float64)
    return common


def _check_same_shape(arrs, op: str) -> None:
    # v1 fused is same-shape, NO broadcast: a shape mismatch is a ValueError, not
    # a silent NumPy broadcast. This is the genuinely new validation with no
    # matmul analog (matmul checks inner-dimension conformability; fused checks
    # exact-shape equality across every operand). Raise on the first mismatch,
    # naming both shapes, in the design-doc error-model shape.
    s0 = arrs[0].shape
    for arr in arrs[1:]:
        if arr.shape != s0:
            raise ValueError(
                f"{op}: operands must have the same shape, got {s0} and {arr.shape}"
            )


def _check_axis(axis, op: str) -> None:
    # axis is keyword-only and accepts only None, 0, or 1: this is a 2-D library
    # and a positional or out-of-range axis is the accidental argument the
    # keyword wall exists to stop. bool is an int subclass, so reject it
    # explicitly; True would otherwise sneak through as axis 1.
    if axis is None:
        return
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise TypeError(f"{op}: axis must be None, 0, or 1, got {type(axis).__name__}")
    if axis not in (0, 1):
        raise ValueError(f"{op}: axis {axis} is out of bounds for 2-dimensional input")


# The exact mixed-residency token from the API design doc (DESIGN_SYSTEM.md error
# model). Verbatim, never paraphrased: tests match on it and it is the contract.
_MIXED_RESIDENCY_MSG = (
    "matmul: one input is on cuda and one on cpu, "
    "call to_device or from_device first"
)


def _mixed_residency_msg(op: str) -> str:
    # The fused per-op mixed-residency token. The matmul literal above is
    # matmul-prefixed and is its own contract string; the fused ops carry the
    # SAME wording but named for the op, never the matmul-prefixed literal, so a
    # caller mixing a host and a device operand to axpby/fma3/scaled_relu sees the
    # op they called. From the design-doc error model (same shape as matmul's).
    return (
        f"{op}: inputs must all be on the same device, "
        "call to_device or from_device first"
    )

# Set once per process the first time an auto-mode CUDA failure falls back to the
# CPU, so the warn-once contract holds: a long-running session that hits a
# recurring CUDA error warns a single time, not on every call. Module-level (not
# per-call) so the "for this session" wording is literally true. The lock makes
# the check-then-set atomic (WR-05): the library drops the GIL around kernels so
# callers thread around it, and `if not _cuda_fallback_warned: _cuda_fallback_warned = True`
# is two bytecode ops -- a thread switch between them under concurrent device
# failures would let two threads both warn, breaking the literal once-per-session
# claim. The lock is uncontended on the normal path (transfers/GEMM are the hot
# work, not fallback), so it costs nothing measurable.
_cuda_fallback_lock = threading.Lock()
_cuda_fallback_warned = False

# The active dispatch backend: "auto" runs the measured policy, "cpu"/"cuda"
# force that side. Read ONCE here at import from FME_BACKEND (the env hook the
# design doc names, mirroring FME_CACHE_DIR) defaulting to "auto", and validated
# against policy.BACKENDS so an out-of-set value can never reach choose_backend.
# This is a plain os.environ.get plus a membership test: microseconds, NO device
# probe and NO cache read, so the <50ms import budget holds (the probe runs only
# inside set_backend at call time, the cache read only on first dispatch). An
# invalid FME_BACKEND degrades to "auto" rather than raising at import: a bad env
# var must not make the package unimportable, and "auto" is the safe default. A
# bad value passed to set_backend at runtime DOES raise (the caller asked for it
# explicitly), so the strict-vs-lenient split is by who set it.
_backend = os.environ.get("FME_BACKEND", "auto")
if _backend not in policy.BACKENDS:
    _backend = "auto"


def _cuda_error_name(exc: Exception) -> str:
    """Extract the cuda error name from a native cuda_error message.

    The CHECK macros throw fme::cuda_error carrying "<cudaErrorName> at
    <file>:<line>" (or a plain sentence for the synthetic guards). The fallback
    warning names the error, so take the leading token up to " at " when present,
    otherwise the whole message. Keeps the design-doc token
    "cuda <op> failed (<cuda error name>), ..." readable rather than dumping a
    file path into it.
    """
    msg = str(exc)
    return msg.split(" at ", 1)[0].strip() or msg


def _cuda_unavailable_reason() -> str:
    """Return why the cuda backend is unavailable, or "" when it is usable.

    The reason strings are the ones the design doc names for the
    ``cuda backend requested but <reason>`` RuntimeError: "no build", "no
    device", "driver too old". Composed from the build flag and the cheap device
    probe (no context build, no driver load beyond the probe's own count query).
    """
    if not _has_cuda_build():
        return "no build"
    probe = _cuda_device_probe()
    if probe["present"]:
        return ""
    reason = probe["reason"]
    # Map the native probe's reason onto the design-doc vocabulary. The probe
    # already returns "driver too old" / "no device" / "compute capability too
    # low"; normalize the two device-absent variants to "no device" so the public
    # token stays in the doc's named set.
    if "driver" in reason:
        return "driver too old"
    return "no device"


def has_cuda() -> bool:
    """Return whether a usable CUDA device is available.

    True only when fastmathext was built with CUDA support AND a usable device
    is present: the driver loads, at least one device exists, and its compute
    capability is at least 7.0. False (never an exception) on a CPU-only build
    or a machine without a usable device, so auto-mode code can branch on it
    without a try/except and stays silent when there is no GPU.

    Returns
    -------
    bool
        True if a CUDA device is built-in and usable, otherwise False.

    Examples
    --------
    >>> import fastmathext as fme
    >>> isinstance(fme.has_cuda(), bool)
    True
    """
    # The probe runs lazily, on call, never at import: it queries the device
    # count and compute capability but does not build the context (no streams,
    # handles, or mempool), so importing fastmathext pays nothing for it and the
    # <50ms import budget holds.
    return _cuda_unavailable_reason() == ""


def set_backend(name: str) -> None:
    """Set the dispatch backend for the rest of the session.

    Overrides how matmul routes a host array pair. ``"auto"`` (the default, and
    the value at import unless FME_BACKEND says otherwise) runs the measured
    per-machine policy: a host float32 product routes to the GPU only when the
    calibrated crossover says the device wins after the host<->device transfer is
    paid, and float64 never auto-routes to the GPU. ``"cpu"`` forces every product
    onto the CPU. ``"cuda"`` forces a host float32 product onto the GPU (via a
    host->device staging copy), and raises immediately if no usable CUDA device is
    present, so the failure surfaces here at the set rather than later mid-matmul.

    This is one of only two functions that change global state (the other is
    calibrate). It sets a process-wide module global; it does not build a CUDA
    context, read the calibration cache, or run any measurement, so it is cheap
    and the cache read still happens lazily on the first dispatch.

    Parameters
    ----------
    name : {"auto", "cpu", "cuda"}
        The backend to use. ``"auto"`` consults the policy; ``"cpu"`` and
        ``"cuda"`` force that side.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``name`` is not one of ``"auto"``, ``"cpu"``, or ``"cuda"``.
    RuntimeError
        If ``name`` is ``"cuda"`` but no usable CUDA device is available; the
        message is ``"cuda backend requested but <reason>"`` (no build, no device,
        or driver too old), the same token to_device raises.

    Examples
    --------
    >>> import fastmathext as fme
    >>> fme.set_backend("cpu") is None
    True
    >>> fme.set_backend("auto") is None
    True
    """
    if name not in policy.BACKENDS:
        raise ValueError(
            f"set_backend: unknown backend {name!r}, "
            f"expected one of {', '.join(policy.BACKENDS)}"
        )
    if name == "cuda":
        # Validate device availability HERE, at the explicit request, reusing the
        # exact token to_device raises (__init__.py to_device): a forced "cuda"
        # with no usable device is a hard error, not a silent CPU fallback -- the
        # caller asked for the GPU by name. The reason ("no build" / "no device" /
        # "driver too old") comes from _cuda_unavailable_reason unchanged.
        reason = _cuda_unavailable_reason()
        if reason:
            raise RuntimeError(f"cuda backend requested but {reason}")
    global _backend
    _backend = name


def to_device(a) -> Array:
    """Copy a host array to the CUDA device, returning an fme.Array.

    The returned fme.Array holds a device-resident copy of ``a``; pass it to
    matmul for the device path, or back through from_device to retrieve the
    values. Accepts both float32 and float64: it is a memory transfer, not a
    routing decision. A float64 array is stored on the device, but a float64
    matmul never auto-routes to this GPU (its FP64 throughput is far below the
    CPU); a forced device-resident float64 matmul computes correctly and slowly.

    Parameters
    ----------
    a : array_like
        Input; must be 2-D with a float32, float64, or integer dtype (integers
        promote to float64, matching the host matmul promotion).

    Returns
    -------
    fme.Array
        A device-resident copy of ``a`` after promotion and contiguity
        normalization.

    Raises
    ------
    RuntimeError
        If no usable CUDA device is available.
    ValueError
        If ``a`` is not 2-D.
    TypeError
        If the dtype is unsupported.
    """
    reason = _cuda_unavailable_reason()
    if reason:
        raise RuntimeError(f"cuda backend requested but {reason}")
    arr = _prepare(a, "a", "to_device")
    common = _resolve_dtype(arr, "to_device")
    return _to_device_native(_normalize(arr, common))


def from_device(x: Array) -> np.ndarray:
    """Copy an fme.Array back to a host NumPy array.

    Performs a device-to-host copy and synchronizes before returning, so the
    result is fully populated. The returned array owns a fresh host buffer.

    Parameters
    ----------
    x : fme.Array
        A device-resident array produced by to_device or a device matmul.

    Returns
    -------
    numpy.ndarray
        A host copy of ``x``.

    Raises
    ------
    RuntimeError
        If no usable CUDA device is available.
    TypeError
        If ``x`` is not an fme.Array.
    """
    reason = _cuda_unavailable_reason()
    if reason:
        raise RuntimeError(f"cuda backend requested but {reason}")
    if not isinstance(x, Array):
        raise TypeError(
            f"from_device: expected an fme.Array, got {type(x).__name__}"
        )
    return _from_device_native(x)


def _matmul_cpu_fallback(a, b):
    # Recompute on the CPU after an auto-mode device failure: the operands are
    # device-resident fme.Arrays, so copy them back to host and run the host
    # matmul. The result is a host ndarray -- "falling back to cpu" means the
    # computation (and its residency) is now host-side; we cannot return a device
    # array when the device is exactly what failed. This is the correct,
    # testable fallback (the forced-OOM test compares it to NumPy through
    # assert_matmul_close, which operates on host arrays).
    return matmul(_from_device_native(a), _from_device_native(b))


def _matmul_gpu_staged(a, b):
    # Route a HOST array pair to the GPU by staging through the existing Phase-4
    # entries: H2D both operands (to_device, synced at the boundary), the device
    # matmul(Array, Array), then D2H the result (from_device, synced) back to a
    # host ndarray. Minimal-risk by reuse -- no new native code, no new sanitizer
    # surface; the three Python crossings are exactly the staging overhead the
    # policy already counted when it chose this path. a and b are already
    # promotion-/contiguity-normalized host ndarrays (the caller normalizes before
    # routing), so this only moves bytes and computes.
    #
    # The body is wrapped in the SAME warn-once fallback as the device-resident
    # path (the _cuda_fallback_lock / _cuda_fallback_warned mechanism), shared
    # intentionally: the "falling back to cpu for this session" contract is
    # process-wide, so whichever GPU path fails first claims the single warning
    # and every later failure (device-resident or host-staged) silently falls
    # back. On any CUDA runtime failure here, recompute on the HOST arrays a, b
    # directly via the normal host chain -- NOT _matmul_cpu_fallback, which
    # from_devices its operands (those are device Arrays; a and b here are host).
    try:
        xa = _to_device_native(a)
        xb = _to_device_native(b)
        xc = _matmul_native(xa, xb)
        return _from_device_native(xc)
    except RuntimeError as exc:
        global _cuda_fallback_warned
        with _cuda_fallback_lock:
            should_warn = not _cuda_fallback_warned
            _cuda_fallback_warned = True
        if should_warn:
            name = _cuda_error_name(exc)
            warnings.warn(
                f"cuda matmul failed ({name}), "
                "falling back to cpu for this session",
                RuntimeWarning,
                stacklevel=2,
            )
        # Host operands: recompute on the host chain directly. _normalize already
        # ran upstream, so a, b are accepted-dtype C-contiguous ndarrays and the
        # fast path forwards them to the CPU kernel untouched.
        return _matmul_native(a, b)


def _host_backend_for(a_arr, b_arr) -> str:
    # Decide CPU vs GPU for a normalized HOST pair, honoring the module-global
    # _backend. "cpu" forces the CPU; "cuda" forces the GPU staging path (its
    # device availability was validated when set_backend("cuda") was called, so a
    # runtime failure here falls back via the warn-once block). "auto" consults
    # the measured policy -- but only when a usable device actually exists: with
    # no GPU the policy can only return "cpu" (no "gpu" series in any cache it
    # could read), so short-circuit to skip the signature build and the cache read
    # entirely. This is what keeps the OFF build (and any no-GPU machine) on the
    # existing CPU fast path with zero added I/O, and it is why a bare import pays
    # nothing: the cache is read here, on the FIRST dispatch that could route to a
    # present GPU, never at import.
    if _backend == "cpu":
        return "cpu"
    if _backend == "cuda":
        return "cuda"
    # auto
    if not has_cuda():
        return "cpu"
    # A device is present: read the calibration cache lazily (first such dispatch)
    # and let the pure policy decide per measured (dtype, n) crossover, paying the
    # H2D+D2H round trip for a host pair. f64 never reaches the GPU -- the policy
    # hardcodes the FP64 trap, so the wrapper does not special-case it here.
    from .calibrate import _machine_signature

    calibration = policy._load_cache(policy._cache_path(), _machine_signature())
    dtype = a_arr.dtype.name
    m, k = a_arr.shape
    n = b_arr.shape[1]
    return policy.choose_backend(
        dtype, m, k, n, residency="host", calibration=calibration, backend="auto"
    )


def matmul(a, b, *, out=None):
    """Matrix product of two 2-D arrays.

    Matches numpy.matmul for float32 and float64 operands, including
    zero-sized dimensions and NaN/Inf propagation.

    The return type follows operand residency: two host arrays give a host
    ndarray, two device fme.Arrays give a device fme.Array (the float32 GPU
    path), and a mix of one host and one device operand raises TypeError rather
    than transferring silently. A device float64 product computes on the GPU only
    when both operands were explicitly placed there with to_device; a host
    float64 product never auto-routes to the GPU (its FP64 throughput is below
    the CPU). If a device computation fails at runtime, the call warns once and
    falls back to the CPU for the rest of the session, returning a host ndarray.

    Parameters
    ----------
    a : array_like or fme.Array
        Left operand; must be 2-D.
    b : array_like or fme.Array
        Right operand; must be 2-D with as many rows as a has columns, and on
        the same side (host or device) as a.
    out : None, optional
        Reserved for a future preallocated-output path. The keyword is part
        of the signature ahead of its implementation; only None is accepted.

    Returns
    -------
    numpy.ndarray or fme.Array
        The matrix product, on the host for host operands and on the device for
        device operands. The result dtype is np.result_type(a, b) when that
        lands on float32 or float64, so int8/int16/uint8/uint16 mixed with
        float32 give float32. Integer-with-integer products always promote to
        float64, unlike NumPy, which keeps int64: this is a float library, so
        every result is a float array.

    Raises
    ------
    ValueError
        If an operand is not 2-D, or the inner dimensions do not match.
    TypeError
        If exactly one operand is an fme.Array (mixed residency), or an operand
        dtype is bool, float16, complex64, complex128, or object, or if
        promotion lands on anything other than float32, float64, or an integer
        dtype (longdouble and clongdouble on platforms where they are distinct).
    NotImplementedError
        If out is anything other than None.

    Warns
    -----
    RuntimeWarning
        Once per session, if a device matmul fails at runtime and the call
        falls back to the CPU.

    Examples
    --------
    >>> import numpy as np
    >>> import fastmathext as fme
    >>> fme.matmul(np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2))
    array([[1., 2.],
           [3., 4.]])
    >>> fme.matmul(np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 1]])).dtype
    dtype('float64')
    """
    if out is not None:
        raise NotImplementedError("matmul: out= is not implemented yet")

    # Residency check, BEFORE the host fast path. Return type follows residency
    # (DESIGN_SYSTEM.md): both operands on the device returns an fme.Array, both
    # on the host returns an ndarray, and a mix is an error -- never a silent
    # transfer. fme.Array is the CUDA-build device type; on a CPU-only build it is
    # a sentinel nothing is an instance of, so both isinstance checks are False
    # and control falls straight through to the host path with zero overhead.
    a_is_dev = isinstance(a, Array)
    b_is_dev = isinstance(b, Array)
    if a_is_dev != b_is_dev:
        # Exactly one operand is device-resident: the verbatim mixed-residency
        # token, never a silent host<->device transfer the caller did not ask for.
        raise TypeError(_MIXED_RESIDENCY_MSG)
    if a_is_dev and b_is_dev:
        # Both device-resident: the f32 (and forced f64) device path. f64 reaches
        # here ONLY because the caller explicitly put both operands on the device
        # (to_device) -- a host f64 array never auto-routes to the GPU (it would
        # not be an fme.Array), so the FP64 trap rule is upheld upstream. Auto mode
        # catches a CUDA runtime failure, warns ONCE, and falls back to the CPU.
        try:
            return _matmul_native(a, b)
        except RuntimeError as exc:
            # The native cuda_error arm maps every FME_CUDA_CHECK / cuBLAS / OOM
            # failure to RuntimeError carrying the cuda error NAME. This is
            # recoverable: warn once with the verbatim fallback token (the cuda op
            # and error name come from the message) and recompute on the CPU. The
            # warn-once flag below is SHARED with the host-staging path
            # (_matmul_gpu_staged): whichever GPU path fails first this session
            # claims the single warning, by design (the "for this session" contract
            # is process-wide). Both the auto policy and a forced set_backend("cuda")
            # reach a device GEMM, and both fall back here on a runtime device
            # failure -- the forced choice still degrades rather than crashing mid
            # matmul, since set_backend already validated availability at the set.
            global _cuda_fallback_warned
            # Atomic check-then-set under the lock (WR-05) so exactly one thread
            # ever sees the flag as unset and wins the warning, even when several
            # threads hit a device failure at once. warnings.warn runs OUTSIDE the
            # lock: it can invoke arbitrary user warning filters, and holding a
            # lock across that is needless contention -- the only invariant the
            # lock must protect is the single flip of the sentinel.
            with _cuda_fallback_lock:
                should_warn = not _cuda_fallback_warned
                _cuda_fallback_warned = True
            if should_warn:
                name = _cuda_error_name(exc)
                warnings.warn(
                    f"cuda matmul failed ({name}), "
                    "falling back to cpu for this session",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return _matmul_cpu_fallback(a, b)

    # Fast path (CPU-06): when both operands are already 2-D C-contiguous
    # ndarrays of the same accepted float dtype, np.asarray, np.result_type, and
    # np.ascontiguousarray are all no-ops that only cost time, so skip the whole
    # chain and hand the kernel the arrays as-is. At n=8 the skipped helpers are
    # ~1.8 us, the entire 2x-NumPy budget, so this branch is what makes the small
    # win reachable rather than an optimization. It is a pre-branch, not a
    # replacement: any operand that is not an ndarray, not 2-D, not C-contiguous,
    # or a different/unaccepted dtype falls through to the full policy chain
    # below, which keeps the exact rejection, promotion, and error contract. The
    # native call still raises ValueError on mismatched inner dimensions, so the
    # shape contract holds here too. No copy is made: a qualifying input is
    # already contiguous, so the kernel takes a zero-copy view of it.
    if (
        type(a) is np.ndarray
        and type(b) is np.ndarray
        and a.ndim == 2
        and b.ndim == 2
        and a.dtype == b.dtype
        and a.dtype in _ACCEPTED
        and a.flags.c_contiguous
        and b.flags.c_contiguous
    ):
        # The pair is already a normalized host ndarray pair, so route it on the
        # module-global backend (Phase 5). _host_backend_for is "cpu" in one cheap
        # comparison when _backend is "cpu" OR when auto sees no usable device --
        # the OFF build and any no-GPU machine pay only that test and fall straight
        # into the existing CPU kernel call, so the fast path stays a no-overhead
        # pre-branch. A GPU route only happens for a host f32 pair the policy (or a
        # forced set_backend("cuda")) sends to the device.
        if _host_backend_for(a, b) == "cuda":
            return _matmul_gpu_staged(a, b)
        return _matmul_native(a, b)

    a_arr = _prepare(a, "a")
    b_arr = _prepare(b, "b")

    for arr in (a_arr, b_arr):
        if arr.dtype in _REJECT:
            raise TypeError(
                f"matmul: unsupported dtype {arr.dtype.name}, "
                "expected float32 or float64 (ints promote)"
            )

    common = np.result_type(a_arr.dtype, b_arr.dtype)
    if common not in (np.dtype(np.float32), np.dtype(np.float64)):
        # the fallback is integer-only: coercing an extended float or complex
        # result (longdouble, clongdouble where the platform keeps them
        # distinct) to float64 would silently drop precision or imaginary
        # parts, the exact laundering _REJECT exists to stop
        if not np.issubdtype(common, np.integer):
            raise TypeError(
                f"matmul: unsupported dtype {common.name}, "
                "expected float32 or float64 (ints promote)"
            )
        common = np.dtype(np.float64)

    # Normalize to the promoted dtype + C-contiguity, THEN route on the backend
    # (the staging path needs accepted-dtype contiguous host operands, which these
    # now are). Same backend decision as the fast path: CPU unless a present GPU's
    # measured crossover (or a forced "cuda") wins for this host f32 pair. The CPU
    # route is byte-identical to the prior single _matmul_native call.
    a_norm = _normalize(a_arr, common)
    b_norm = _normalize(b_arr, common)
    if _host_backend_for(a_norm, b_norm) == "cuda":
        return _matmul_gpu_staged(a_norm, b_norm)
    return _matmul_native(a_norm, b_norm)


def transpose(a) -> np.ndarray:
    """Transpose of a 2-D array, as an owned copy.

    Matches the values of ``numpy.transpose`` for float32 and float64
    operands, including zero-sized dimensions.

    Parameters
    ----------
    a : array_like
        Input; must be 2-D.

    Returns
    -------
    numpy.ndarray
        A new C-contiguous array, the transpose of ``a``. Unlike NumPy's
        ``a.T``, which is a view sharing memory with ``a``, this is always a
        fresh buffer: mutating the result never touches the input. The result
        dtype is the input dtype after promotion (integers become float64).

    Raises
    ------
    ValueError
        If ``a`` is not 2-D.
    TypeError
        If the dtype is bool, float16, complex64, complex128, object, or any
        promoted dtype other than float32 or float64.

    Examples
    --------
    >>> import numpy as np
    >>> import fastmathext as fme
    >>> fme.transpose(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    array([[1., 4.],
           [2., 5.],
           [3., 6.]])
    >>> a = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> t = fme.transpose(a)
    >>> t[0, 0] = 99.0  # owned copy: the input is untouched
    >>> float(a[0, 0])
    1.0
    """
    arr = _prepare(a, "a", "transpose")
    common = _resolve_dtype(arr, "transpose")
    return _transpose_native(_normalize(arr, common))


def sum(a, *, axis=None):
    """Sum of array elements over an axis.

    Matches ``numpy.sum`` for float32 and float64 operands, including
    zero-sized dimensions and NaN/Inf propagation, with summation performed
    in the input dtype (pairwise along a contiguous axis, sequential along
    the non-contiguous axis, the same algorithm class NumPy uses).

    Parameters
    ----------
    a : array_like
        Input; must be 2-D.
    axis : None, 0, or 1, optional
        Axis to reduce. None sums every element to a scalar; 0 reduces over
        rows to a length-n vector; 1 reduces over columns to a length-m
        vector. Keyword-only, and stricter than NumPy: only None, 0, and 1
        are accepted.

    Returns
    -------
    numpy.floating or numpy.ndarray
        A NumPy scalar of the result dtype for ``axis=None``, otherwise a 1-D
        array of the result dtype. The result dtype is the input dtype after
        promotion (integers become float64). The sum of an empty axis is 0.0,
        matching NumPy.

    Raises
    ------
    ValueError
        If ``a`` is not 2-D, or ``axis`` is an integer other than 0 or 1.
    TypeError
        If the dtype is unsupported, or ``axis`` is neither None nor an int.

    Examples
    --------
    >>> import numpy as np
    >>> import fastmathext as fme
    >>> float(fme.sum(np.array([[1.0, 2.0], [3.0, 4.0]])))
    10.0
    >>> fme.sum(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
    array([4., 6.])
    >>> fme.sum(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=1)
    array([3., 7.])
    """
    _check_axis(axis, "sum")
    arr = _prepare(a, "a", "sum")
    common = _resolve_dtype(arr, "sum")
    arr = _normalize(arr, common)
    if axis is None:
        # sum_all returns a Python float; cast it back to the result dtype so a
        # float32 input yields a float32 scalar (the round trip through a
        # double-precision Python float is lossless for a float32 value).
        return common.type(_sum_all_native(arr))
    return _sum_axis_native(arr, int(axis))


def mean(a, *, axis=None):
    """Arithmetic mean of array elements over an axis.

    Matches ``numpy.mean`` for float32 and float64 operands, including the
    empty-slice behavior: the mean of an empty axis is NaN and emits the
    same RuntimeWarnings NumPy does.

    Parameters
    ----------
    a : array_like
        Input; must be 2-D.
    axis : None, 0, or 1, optional
        Axis to reduce. None averages every element to a scalar; 0 reduces
        over rows to a length-n vector; 1 reduces over columns to a length-m
        vector. Keyword-only, and stricter than NumPy: only None, 0, and 1
        are accepted.

    Returns
    -------
    numpy.floating or numpy.ndarray
        A NumPy scalar of the result dtype for ``axis=None``, otherwise a 1-D
        array of the result dtype. The result dtype is the input dtype after
        promotion (integers become float64).

    Raises
    ------
    ValueError
        If ``a`` is not 2-D, or ``axis`` is an integer other than 0 or 1.
    TypeError
        If the dtype is unsupported, or ``axis`` is neither None nor an int.

    Warns
    -----
    RuntimeWarning
        When the reduced axis is empty and the output is non-empty, exactly
        as NumPy: a "Mean of empty slice" warning followed by an
        invalid-value divide warning, and the result is NaN.

    Examples
    --------
    >>> import numpy as np
    >>> import fastmathext as fme
    >>> float(fme.mean(np.array([[1.0, 2.0], [3.0, 4.0]])))
    2.5
    >>> fme.mean(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
    array([2., 3.])
    >>> fme.mean(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=1)
    array([1.5, 3.5])
    """
    _check_axis(axis, "mean")
    arr = _prepare(a, "a", "mean")
    common = _resolve_dtype(arr, "mean")
    arr = _normalize(arr, common)

    # mean is composed, not native: sum / count reproduces NumPy's exact dtype
    # and, on an empty reduced axis, its exact two-warning set for free. The
    # reduced count is the number of addends per output element.
    if axis is None:
        count = arr.size
        total = common.type(_sum_all_native(arr))
    elif axis == 0:
        count = arr.shape[0]
        total = _sum_axis_native(arr, 0)
    else:
        count = arr.shape[1]
        total = _sum_axis_native(arr, 1)

    # When the reduced axis is empty but the output is not, NumPy warns "Mean of
    # empty slice" before the division. The division itself, count == 0, then
    # produces NumPy's invalid-value divide warning and the NaN result on its
    # own. axis=1 on a (0, n) input has count == n > 0 and an empty (0,) output,
    # so neither warning fires: the guard below is exactly that case excluded.
    out_size = total.size if isinstance(total, np.ndarray) else 1
    if count == 0 and out_size > 0:
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)

    # True division, never reciprocal multiply: x / count and x * (1 / count)
    # round differently, and NumPy composes mean with a division.
    return total / count


def _fused_residency(operands, op: str) -> str:
    # Decide the residency of a fused call from its raw operands, BEFORE any host
    # normalization (the isinstance check must see the original objects). The
    # return-type-follows-residency rule (DESIGN_SYSTEM.md), extended from
    # matmul's two-operand check to N operands: every operand on the device routes
    # the device path (returns an fme.Array); every operand on the host routes the
    # host CPU path (returns an ndarray); ANY mix is a TypeError, never a silent
    # host<->device transfer. On a CPU-only build fme.Array is a sentinel nothing
    # is an instance of, so this is always "host" and the device branch is dead.
    dev = [isinstance(x, Array) for x in operands]
    if all(dev):
        return "device"
    if any(dev):
        raise TypeError(_mixed_residency_msg(op))
    return "host"


def axpby(x, y, *, a=1.0, b=1.0):
    """Elementwise ``a*x + b*y`` for two same-shape 2-D arrays.

    Computes the unfused expression ``a*x + b*y`` within the elementwise
    tolerance for float32 and float64, including exact NaN/Inf propagation
    (matching the NumPy expression positionally).

    The scalars ``a`` and ``b`` are keyword-only. The two arrays must have the
    same shape: v1 does not broadcast, so a shape mismatch raises ValueError.
    The return type follows operand residency (host arrays give a host ndarray);
    the device-resident path lands in a later plan.

    Parameters
    ----------
    x, y : array_like or fme.Array
        Operands; must be 2-D and the same shape, and on the same side (both
        host or both device).
    a, b : float, optional
        Scalar coefficients, keyword-only. Default 1.0 each.

    Returns
    -------
    numpy.ndarray
        ``a*x + b*y`` elementwise. The result dtype is np.result_type across the
        operands when that lands on float32 or float64; integer operands promote
        to float64 (this is a float library).

    Raises
    ------
    ValueError
        If an operand is not 2-D, or the operands differ in shape.
    TypeError
        If an operand dtype is bool, float16, complex64, complex128, or object,
        if promotion lands on anything other than float32/float64/integer, or if
        the operands are split across host and device.

    Examples
    --------
    >>> import numpy as np
    >>> import fastmathext as fme
    >>> x = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> y = np.array([[10.0, 20.0], [30.0, 40.0]])
    >>> fme.axpby(x, y, a=2.0, b=-1.0)
    array([[ -8., -16.],
           [-24., -32.]])
    """
    if _fused_residency((x, y), "axpby") == "device":
        # 06-04 wires the device-resident fused path (all operands fme.Array ->
        # device kernel -> fme.Array out). Until then this is a guarded stub, not
        # a silent CPU compute: the residency dispatch above already routed here,
        # so 06-04 replaces this raise with the device binding call and nothing
        # else moves.
        raise NotImplementedError(
            "axpby: device-resident fused ops are not implemented yet"
        )
    xa = _prepare(x, "x", "axpby")
    ya = _prepare(y, "y", "axpby")
    common = _resolve_dtype_multi((xa, ya), "axpby")
    _check_same_shape((xa, ya), "axpby")
    # Host fused ALWAYS computes on the CPU: v1 has no host->GPU auto-staging
    # (no _matmul_gpu_staged analog, no policy/choose_backend call). float(a) /
    # float(b) cast the scalars at the boundary, mirroring sum's int(axis).
    return _axpby_native(
        _normalize(xa, common), _normalize(ya, common), float(a), float(b)
    )


def fma3(x, y, z):
    """Elementwise fused multiply-add ``x*y + z`` for three same-shape arrays.

    Computes ``x*y + z`` in a single rounding (a true fused multiply-add), so it
    can be closer to the real-number result than the two-rounding unfused NumPy
    ``x*y + z``; both stay within the elementwise tolerance. NaN/Inf match the
    NumPy expression positionally, including ``inf*0 + z`` giving NaN.

    The three arrays must have the same shape: v1 does not broadcast, so a shape
    mismatch raises ValueError. The return type follows operand residency.

    Parameters
    ----------
    x, y, z : array_like or fme.Array
        Operands; must be 2-D and the same shape, and on the same side (all host
        or all device).

    Returns
    -------
    numpy.ndarray
        ``x*y + z`` elementwise. The result dtype is np.result_type across all
        three operands when that lands on float32 or float64; integer operands
        promote to float64.

    Raises
    ------
    ValueError
        If an operand is not 2-D, or the operands differ in shape.
    TypeError
        If an operand dtype is bool, float16, complex64, complex128, or object,
        if promotion lands on anything other than float32/float64/integer, or if
        the operands are split across host and device.

    Examples
    --------
    >>> import numpy as np
    >>> import fastmathext as fme
    >>> x = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> y = np.array([[5.0, 6.0], [7.0, 8.0]])
    >>> z = np.array([[1.0, 1.0], [1.0, 1.0]])
    >>> fme.fma3(x, y, z)
    array([[ 6., 13.],
           [22., 33.]])
    """
    if _fused_residency((x, y, z), "fma3") == "device":
        raise NotImplementedError(
            "fma3: device-resident fused ops are not implemented yet"
        )
    xa = _prepare(x, "x", "fma3")
    ya = _prepare(y, "y", "fma3")
    za = _prepare(z, "z", "fma3")
    common = _resolve_dtype_multi((xa, ya, za), "fma3")
    _check_same_shape((xa, ya, za), "fma3")
    return _fma3_native(
        _normalize(xa, common), _normalize(ya, common), _normalize(za, common)
    )


def scaled_relu(x, *, scale=1.0):
    """Elementwise ``maximum(scale*x, 0)`` for a 2-D array.

    Computes ``np.maximum(scale*x, 0)`` within the elementwise tolerance for
    float32 and float64. A NaN input PROPAGATES as NaN exactly like np.maximum
    (it does not collapse to 0); signed Inf matches positionally.

    The scalar ``scale`` is keyword-only. The return type follows operand
    residency (a host array gives a host ndarray).

    Parameters
    ----------
    x : array_like or fme.Array
        Input; must be 2-D.
    scale : float, optional
        Scalar multiplier applied before the rectifier, keyword-only. Default
        1.0.

    Returns
    -------
    numpy.ndarray
        ``maximum(scale*x, 0)`` elementwise. The result dtype is the input dtype
        after promotion (integers promote to float64).

    Raises
    ------
    ValueError
        If ``x`` is not 2-D.
    TypeError
        If the dtype is bool, float16, complex64, complex128, object, or any
        promoted dtype other than float32 or float64.

    Examples
    --------
    >>> import numpy as np
    >>> import fastmathext as fme
    >>> x = np.array([[-1.0, 2.0], [3.0, -4.0]])
    >>> fme.scaled_relu(x, scale=3.0)
    array([[0., 6.],
           [9., 0.]])
    """
    if _fused_residency((x,), "scaled_relu") == "device":
        raise NotImplementedError(
            "scaled_relu: device-resident fused ops are not implemented yet"
        )
    xa = _prepare(x, "x", "scaled_relu")
    common = _resolve_dtype_multi((xa,), "scaled_relu")
    return _scaled_relu_native(_normalize(xa, common), float(scale))
