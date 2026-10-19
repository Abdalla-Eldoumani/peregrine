"""peregrine: heterogeneous linear algebra with a NumPy-compatible surface.

The native core takes zero-copy views of C-contiguous float32 and float64
arrays. This wrapper supplies the NumPy-style ergonomics on top: dtype
promotion, contiguity normalization, and clear errors.
"""

from __future__ import annotations

import functools
import math
import os
import threading
import warnings
from importlib.metadata import PackageNotFoundError, version as _version

import numpy as np

# Import-time CUDA DLL discovery (Windows). A CUDA-enabled _core links
# cublas64_12.dll / cublasLt64_12.dll from the toolkit bin, and since Python 3.8
# the secure DLL search for extension modules ignores PATH, so a bare
# `import peregrine` under the ON build would raise "DLL load failed". Register
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

from . import _core, policy
from ._core import _clear_routing_facts, _has_cuda_build, _set_routing_facts, cpu_features
from ._core import axpby_into as _axpby_into
from ._core import fma3_into as _fma3_into
from ._core import matmul_into as _matmul_into
from ._core import scaled_relu_into as _scaled_relu_into
from ._core import sum_all as _sum_all_native
from ._core import sum_axis_into as _sum_axis_into
from ._core import transpose_into as _transpose_into


def _no_fast_entry(*args):
    return NotImplemented


def _fast_entry(name):
    # Each native fast entry takes the common host case of one op and returns
    # NotImplemented for anything else, which then runs the Python path below.
    # PEREGRINE_DISABLE_FAST_ENTRIES (any non-empty value, read once here)
    # binds the stub instead, so the Python path stays a fully tested path.
    if os.environ.get("PEREGRINE_DISABLE_FAST_ENTRIES"):
        return _no_fast_entry
    return getattr(_core, name, _no_fast_entry)


_fast_matmul = _fast_entry("_fast_matmul")
_fast_transpose = _fast_entry("_fast_transpose")
_fast_sum = _fast_entry("_fast_sum")
_fast_mean = _fast_entry("_fast_mean")
_fast_axpby = _fast_entry("_fast_axpby")
_fast_fma3 = _fast_entry("_fast_fma3")
_fast_scaled_relu = _fast_entry("_fast_scaled_relu")

# calibrate() is re-exported as the public pg.calibrate. The binding is explicit
# (not left to attribute fallthrough) BECAUSE a submodule named calibrate exists:
# `from .calibrate import calibrate` binds the FUNCTION into the package namespace,
# shadowing the submodule so `pg.calibrate(...)` is the callable users get, not
# the module. calibrate.py runs nothing at import (no measurement, no
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
    from ._core import axpby as _axpby_device
    from ._core import fma3 as _fma3_device
    from ._core import from_device as _from_device_native
    from ._core import matmul as _matmul_device
    from ._core import scaled_relu as _scaled_relu_device
    from ._core import to_device as _to_device_native
else:
    class Array:  # noqa: D401 - sentinel; the real device type is CUDA-build-only
        """Placeholder for pg.Array on a CPU-only build.

        The device array type exists only when peregrine is built with
        ``PG_ENABLE_CUDA=ON``. On a CPU-only build this sentinel exists so the
        public name is importable and ``isinstance(x, pg.Array)`` is always
        False; constructing it raises, since there is no device to hold.
        """

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "cuda backend requested but no build "
                "(peregrine was built with PG_ENABLE_CUDA=OFF)"
            )

    _cuda_device_probe = None

# Single source of the version: pyproject.toml carries it, and importlib.metadata
# reads it back from the installed package metadata so there is no second literal
# to drift. The fallback covers a source tree
# with no install metadata (e.g. running from a checkout that was never pip
# installed); it matches the pyproject value so an uninstalled tree still reports
# the release it belongs to.
try:
    __version__ = _version("peregrine")
except PackageNotFoundError:
    __version__ = "3.0.0"
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
    "set_num_threads",
    "get_num_threads",
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
    # require is a no-op for arrays that already qualify, which is the
    # zero-copy fast path. The reductions and elementwise ops compute on C
    # order (their bits depend on memory order), so strided views pay one copy
    # here, and so do unaligned ones: the kernels load whole elements, and a
    # misaligned element load is undefined behavior.
    return np.require(arr, dtype=common, requirements=("C", "A"))


def _matmul_operand(arr: np.ndarray, common: np.dtype) -> np.ndarray:
    # The matmul kernels read any element strides (negative and zero ones
    # included) and give the same bits for every layout, so an aligned native
    # operand of the result dtype goes in as it is. Only what they cannot read
    # is copied: another dtype or byte order, a misaligned element, or a byte
    # stride that is not a whole number of elements.
    if (
        arr.dtype == common
        and arr.flags.aligned
        and all(s % arr.itemsize == 0 for s in arr.strides)
    ):
        return arr
    return np.array(arr, dtype=common, order="C")


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
    # naming both shapes, in the same error shape as the other checks.
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


def _check_out(out, op: str, dtype: np.dtype, shape: tuple) -> None:
    # The kernel writes the result straight into out's buffer, so out must
    # already be exactly the array a fresh result would be. A dtype compare
    # includes byte order, and str() of a swapped dtype ('>f8') tells it apart
    # from the native one in the message.
    if type(out) is not np.ndarray:
        raise TypeError(
            f"{op}: out must be a numpy.ndarray of dtype {dtype.name}, "
            f"got {type(out).__name__}"
        )
    if out.dtype != dtype:
        raise TypeError(
            f"{op}: out must be a numpy.ndarray of dtype {dtype.name}, got {out.dtype}"
        )
    flags = out.flags
    if out.shape != shape:
        detail = f"shape {out.shape}"
    elif not flags.c_contiguous:
        detail = "a non-C-contiguous array"
    elif not flags.aligned:
        detail = "an unaligned array"
    elif not flags.writeable:
        detail = "a read-only array"
    else:
        return
    raise ValueError(
        f"{op}: out must be a writeable C-contiguous array of shape {shape}, "
        f"got {detail}"
    )


def _fused_out(out, arrs, op: str, dtype: np.dtype) -> np.ndarray:
    # Every fused kernel computes out[i] from the inputs at index i alone, so
    # an input that is out itself (same buffer, extent, and dtype) is an exact
    # in-place update. Any other overlap would read elements already
    # overwritten. may_share_memory without max_work compares byte bounds
    # only, the cheap test the fast path repeats in C.
    shape = arrs[0].shape
    if out is None:
        return np.empty(shape, dtype)
    _check_out(out, op, dtype, shape)
    ptr = out.__array_interface__["data"][0]
    for arr in arrs:
        if np.may_share_memory(arr, out) and not (
            arr.dtype == dtype
            and arr.flags.c_contiguous
            and arr.__array_interface__["data"][0] == ptr
        ):
            raise ValueError(
                f"{op}: out must be one of the inputs exactly or not overlap them"
            )
    return out


# The exact mixed-residency error string, part of the public API contract.
# Verbatim, never paraphrased: tests match on it and it is the contract.
_MIXED_RESIDENCY_MSG = (
    "matmul: one input is on cuda and one on cpu, "
    "call to_device or from_device first"
)


def _mixed_residency_msg(op: str) -> str:
    # The fused per-op mixed-residency token. The matmul literal above is
    # matmul-prefixed and is its own contract string; the fused ops carry the
    # SAME wording but named for the op, never the matmul-prefixed literal, so a
    # caller mixing a host and a device operand to axpby/fma3/scaled_relu sees the
    # op they called. Same shape as matmul's mixed-residency token above.
    return (
        f"{op}: inputs must all be on the same device, "
        "call to_device or from_device first"
    )

# Set once per process the first time an auto-mode CUDA failure falls back to the
# CPU, so the warn-once contract holds: a long-running session that hits a
# recurring CUDA error warns a single time, not on every call. Module-level (not
# per-call) so the "for this session" wording is literally true. The lock makes
# the check-then-set atomic: the library drops the GIL around kernels so
# callers thread around it, and `if not _cuda_fallback_warned: _cuda_fallback_warned = True`
# is two bytecode ops -- a thread switch between them under concurrent device
# failures would let two threads both warn, breaking the literal once-per-session
# claim. The lock is uncontended on the normal path (transfers/GEMM are the hot
# work, not fallback), so it costs nothing measurable. The auto router reads the
# flag too and stops staging host work to the device once it is set, so a device
# that failed once is not handed every later product only to fail again.
_cuda_fallback_lock = threading.Lock()
_cuda_fallback_warned = False

# The active dispatch backend: "auto" runs the measured policy, "cpu"/"cuda"
# force that side. Read ONCE here at import from PEREGRINE_BACKEND (the env hook,
# mirroring PEREGRINE_CACHE_DIR) defaulting to "auto", and validated
# against policy.BACKENDS so an out-of-set value can never reach choose_backend.
# This is a plain os.environ.get plus a membership test: microseconds, NO device
# probe and NO cache read, so the <50ms import budget holds (the probe runs only
# inside set_backend at call time, the cache read only on first dispatch). An
# invalid PEREGRINE_BACKEND degrades to "auto" rather than raising at import: a bad env
# var must not make the package unimportable, and "auto" is the safe default. A
# bad value passed to set_backend at runtime DOES raise (the caller asked for it
# explicitly), so the strict-vs-lenient split is by who set it.
_backend = os.environ.get("PEREGRINE_BACKEND", "auto")
if _backend not in policy.BACKENDS:
    _backend = "auto"

# The calibration the auto router uses, loaded on the first auto dispatch that
# could reach a GPU and kept for the session, so later dispatches do no file
# system work. None is a real value here (no usable cache, or no usable device
# to route to), hence the separate not-loaded-yet sentinel. calibrate() replaces
# it with what it wrote.
_UNSET = object()
_routing_calibration = _UNSET


def _cuda_error_name(exc: Exception) -> str:
    """Extract the cuda error name from a native cuda_error message.

    The CHECK macros throw pg::cuda_error carrying "<cudaErrorName> at
    <file>:<line>" (or a plain sentence for the synthetic guards). The fallback
    warning names the error, so take the leading token up to " at " when present,
    otherwise the whole message. Keeps the public warning token
    "cuda <op> failed (<cuda error name>), ..." readable rather than dumping a
    file path into it.
    """
    msg = str(exc)
    return msg.split(" at ", 1)[0].strip() or msg


@functools.cache
def _cuda_unavailable_reason() -> str:
    """Return why the cuda backend is unavailable, or "" when it is usable.

    The reason strings fill the
    ``cuda backend requested but <reason>`` RuntimeError: "no build", "no
    device", "driver too old". Composed from the build flag and the cheap device
    probe (no context build, no driver load beyond the probe's own count query).
    Memoized per process: every auto dispatch and every transfer asks, and the
    answer cannot change while the process runs. _reset_routing_state() clears it.
    """
    if not _has_cuda_build():
        return "no build"
    probe = _cuda_device_probe()
    if probe["present"]:
        return ""
    reason = probe["reason"]
    # Map the native probe's reason onto the public vocabulary. The probe
    # already returns "driver too old" / "no device" / "compute capability too
    # low"; normalize the two device-absent variants to "no device" so the public
    # token stays in the doc's named set.
    if "driver" in reason:
        return "driver too old"
    return "no device"


def has_cuda() -> bool:
    """Return whether a usable CUDA device is available.

    True only when peregrine was built with CUDA support AND a usable device
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
    >>> import peregrine as pg
    >>> isinstance(pg.has_cuda(), bool)
    True
    """
    # The probe runs lazily, on the first call, never at import: it queries the
    # device count and compute capability but does not build the context (no
    # streams, handles, or mempool), so importing peregrine pays nothing for it
    # and the <50ms import budget holds. Later calls reuse the first answer.
    return _cuda_unavailable_reason() == ""


def _push_routing_facts(auto_ready: bool, calibration=None) -> None:
    # The native matmul entry routes host products on these facts and declines
    # whatever they leave open, so they only ever hold a decision this module
    # would make itself. Until auto mode has looked at the device and the
    # calibration (auto_ready), the entry hands every auto product back here.
    usable = auto_ready and not _cuda_fallback_warned and has_cuda()
    bound = policy.f32_cpu_flops_bound(calibration) if usable else math.inf
    _set_routing_facts(_backend, auto_ready, usable, bound)


# A forced backend reaches the native entry from the first call; auto waits
# for its first dispatch, so import touches no device and reads no file.
_push_routing_facts(False)


def _install_routing_calibration(calibration) -> None:
    # The one place the auto router's calibration changes, so the native entry
    # always routes on the calibration this module routes on.
    global _routing_calibration
    _routing_calibration = calibration
    _push_routing_facts(True, calibration)


def _reset_routing_state() -> None:
    # Drops every per-process routing memo so the next dispatch re-derives it.
    # Tests point PEREGRINE_CACHE_DIR somewhere new per test and stub the device,
    # and a memo from the previous test would otherwise route this one.
    # The warn-once fallback flag stays set: it records a real device failure,
    # and clearing it would let a failing device take auto-routed work again.
    global _routing_calibration
    _routing_calibration = _UNSET
    _clear_routing_facts()
    _cuda_unavailable_reason.cache_clear()
    from .calibrate import _machine_signature

    _machine_signature.cache_clear()


def set_backend(name: str) -> None:
    """Set the dispatch backend for the rest of the session.

    Overrides how matmul routes a host array pair. ``"auto"`` (the default, and
    the value at import unless PEREGRINE_BACKEND says otherwise) runs the measured
    per-machine policy: a host float32 product routes to the GPU only when the
    calibrated crossover says the device wins after the host<->device transfer is
    paid, and float64 never auto-routes to the GPU. ``"cpu"`` forces every product
    onto the CPU. ``"cuda"`` forces a host float32 product onto the GPU (via a
    host->device staging copy), and raises immediately if no usable CUDA device is
    present, so the failure surfaces here at the set rather than later mid-matmul.

    This is one of three functions that change global state (with calibrate and
    set_num_threads). It sets a process-wide module global and forgets the
    calibration the auto router loaded, so the next auto dispatch reads the cache
    again; it does not build a CUDA context, read the calibration cache, or run
    any measurement itself, so it is cheap.

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
    >>> import peregrine as pg
    >>> pg.set_backend("cpu") is None
    True
    >>> pg.set_backend("auto") is None
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
    global _backend, _routing_calibration
    _backend = name
    _routing_calibration = _UNSET
    _push_routing_facts(False)


def set_num_threads(n: int) -> None:
    """Set how many CPU threads each later call may use.

    The setting is process-wide: it applies to every later call from any Python
    thread, and replaces the value chosen at import (see get_num_threads). A call
    uses at most ``n`` threads, fewer when OpenMP's own limit on the calling
    thread is lower (OMP_NUM_THREADS or threadpoolctl can lower it), and fewer
    for work too small to split. Results are bitwise identical for any thread
    count, so the setting changes speed only.

    Parameters
    ----------
    n : int
        A positive number of threads. Values above the logical CPU count are
        accepted; each call is still capped by OpenMP's limit.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``n`` is not a positive int (a bool is not accepted); the message is
        ``"set_num_threads: n must be a positive int, got <value>"``.

    Examples
    --------
    >>> import peregrine as pg
    >>> previous = pg.get_num_threads()
    >>> pg.set_num_threads(2)
    >>> pg.get_num_threads()
    2
    >>> pg.set_num_threads(previous)
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"set_num_threads: n must be a positive int, got {n!r}")
    # The native setting is a C int; a larger request is capped per call anyway.
    _core._set_num_threads(min(int(n), 2**31 - 1))


def get_num_threads() -> int:
    """Return the CPU thread setting.

    This is the value set_num_threads stored or, before any call to it, the
    value chosen at import: PEREGRINE_NUM_THREADS when it holds a positive int,
    else OMP_NUM_THREADS when it does (its first entry, if it lists one per
    nesting level), else the number of physical cores in the process affinity
    mask. Any other environment value is ignored. The setting is an upper bound
    per call, not the count a given call runs: OpenMP's limit on the calling
    thread also caps it, and small work runs on fewer threads.

    Returns
    -------
    int
        The thread setting, at least 1.

    Examples
    --------
    >>> import peregrine as pg
    >>> pg.get_num_threads() >= 1
    True
    """
    return _core._get_num_threads()


def to_device(a) -> Array:
    """Copy a host array to the CUDA device, returning an pg.Array.

    The returned pg.Array holds a device-resident copy of ``a``; pass it to
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
    pg.Array
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
    """Copy an pg.Array back to a host NumPy array.

    Performs a device-to-host copy and synchronizes before returning, so the
    result is fully populated. The returned array owns a fresh host buffer.

    Parameters
    ----------
    x : pg.Array
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
        If ``x`` is not an pg.Array.
    """
    reason = _cuda_unavailable_reason()
    if reason:
        raise RuntimeError(f"cuda backend requested but {reason}")
    if not isinstance(x, Array):
        raise TypeError(
            f"from_device: expected an pg.Array, got {type(x).__name__}"
        )
    return _from_device_native(x)


def _matmul_cpu_fallback(a, b):
    # Recompute on the CPU after an auto-mode device failure: the operands are
    # device-resident pg.Arrays, so copy them back to host and run the host
    # matmul. The result is a host ndarray -- "falling back to cpu" means the
    # computation (and its residency) is now host-side; we cannot return a device
    # array when the device is exactly what failed. This is the correct,
    # testable fallback (the forced-OOM test compares it to NumPy through
    # assert_matmul_close, which operates on host arrays).
    return matmul(_from_device_native(a), _from_device_native(b))


def _matmul_cpu(a, b, out=None):
    # a and b are normalized host operands; out is validated or None. NumPy
    # allocates every host result, so its provenance matches the fast path's.
    if out is None:
        out = np.empty((a.shape[0], b.shape[1]), a.dtype)
    _matmul_into(a, b, out)
    return out


def _matmul_gpu_staged(a, b, out=None):
    # Route a HOST array pair to the GPU by staging through the existing device
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
        xc = _matmul_device(xa, xb)
        r = _from_device_native(xc)
    except RuntimeError as exc:
        global _cuda_fallback_warned
        with _cuda_fallback_lock:
            should_warn = not _cuda_fallback_warned
            _cuda_fallback_warned = True
        _push_routing_facts(True)
        if should_warn:
            name = _cuda_error_name(exc)
            warnings.warn(
                f"cuda matmul failed ({name}), "
                "falling back to cpu for this session",
                RuntimeWarning,
                stacklevel=2,
            )
        # Host operands: recompute on the host chain directly. The caller already
        # made a, b accepted-dtype C-contiguous ndarrays, so they go to the CPU
        # kernel untouched.
        return _matmul_cpu(a, b, out)
    if out is None:
        return r
    np.copyto(out, r)
    return out


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
        # PEREGRINE_BACKEND=cuda is read at import, where raising would make the
        # package unimportable, so the device check set_backend("cuda") makes
        # happens here instead, with the same message.
        reason = _cuda_unavailable_reason()
        if reason:
            raise RuntimeError(f"cuda backend requested but {reason}")
        return "cuda"
    # auto. After a device failure this session has already been told the work
    # falls back to the cpu, so the device gets no more auto-routed work.
    if _cuda_fallback_warned or not has_cuda():
        if _routing_calibration is _UNSET:
            _install_routing_calibration(None)
        return "cpu"
    # A device is present: let the pure policy decide on the product's work and
    # the bytes its host pair moves each way. f64 never reaches
    # the GPU -- the policy hardcodes the FP64 trap, so the wrapper does not
    # special-case it here.
    calibration = _routing_calibration
    if calibration is _UNSET:
        from .calibrate import _machine_signature

        signature = _machine_signature()
        calibration = policy._load_cache(policy._cache_path(signature), signature)
        _install_routing_calibration(calibration)
    dtype = a_arr.dtype.name
    m, k = a_arr.shape
    n = b_arr.shape[1]
    return policy.choose_backend(
        dtype, m, k, n, residency="host", calibration=calibration, backend="auto"
    )


def _check_matmul_out(a, b, dtype: np.dtype, out) -> None:
    # The inner dimensions are checked here, before any output exists: a
    # mismatched pair must raise the shape error, never a MemoryError from
    # allocating an (m, n) result it could not have produced.
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul: inner dimensions do not match: {a.shape} @ {b.shape}")
    if out is not None:
        _check_out(out, "matmul", dtype, (a.shape[0], b.shape[1]))
        if np.may_share_memory(out, a) or np.may_share_memory(out, b):
            raise ValueError("matmul: out must not overlap an input")


def _matmul_host(a, b, out):
    if _host_backend_for(a, b) == "cuda":
        # The device upload takes C order; the CPU kernels read any strides.
        return _matmul_gpu_staged(np.ascontiguousarray(a), np.ascontiguousarray(b), out)
    return _matmul_cpu(a, b, out)


def matmul(a, b, *, out=None):
    """Matrix product of two 2-D arrays.

    Matches numpy.matmul for float32 and float64 operands, including
    zero-sized dimensions and NaN/Inf propagation.

    The return type follows operand residency: two host arrays give a host
    ndarray, two device pg.Arrays give a device pg.Array (the float32 GPU
    path), and a mix of one host and one device operand raises TypeError rather
    than transferring silently. A device float64 product computes on the GPU only
    when both operands were explicitly placed there with to_device; a host
    float64 product never auto-routes to the GPU (its FP64 throughput is below
    the CPU). If a device computation fails at runtime, the call warns once and
    falls back to the CPU for the rest of the session, returning a host ndarray.

    Parameters
    ----------
    a : array_like or pg.Array
        Left operand; must be 2-D.
    b : array_like or pg.Array
        Right operand; must be 2-D with as many rows as a has columns, and on
        the same side (host or device) as a.
    out : numpy.ndarray, optional
        Host operands only, keyword-only. The product is written into ``out``,
        which is returned. It must be an exact numpy.ndarray of the result
        dtype in native byte order, of shape ``(a.shape[0], b.shape[1])``,
        C-contiguous, aligned, and writeable, and its memory must not overlap
        either operand (a product cannot be computed in place).

    Returns
    -------
    numpy.ndarray or pg.Array
        The matrix product, on the host for host operands and on the device for
        device operands; ``out`` itself when given. The result dtype is
        np.result_type(a, b) when that lands on float32 or float64, so
        int8/int16/uint8/uint16 mixed with float32 give float32.
        Integer-with-integer products always promote to float64, unlike NumPy,
        which keeps int64: this is a float library, so every result is a float
        array.

    Raises
    ------
    ValueError
        If an operand is not 2-D, the inner dimensions do not match, ``out``
        has the wrong shape or is not a writeable, aligned C-contiguous array,
        or ``out`` overlaps an operand.
    TypeError
        If exactly one operand is an pg.Array (mixed residency), or device
        operands come with ``out``, or an operand dtype is bool, float16,
        complex64, complex128, or object, or if promotion lands on anything
        other than float32, float64, or an integer dtype (longdouble and
        clongdouble on platforms where they are distinct), or ``out`` is not
        a numpy.ndarray of the result dtype.

    Warns
    -----
    RuntimeWarning
        Once per session, if a device matmul fails at runtime and the call
        falls back to the CPU.

    Notes
    -----
    Host operands are read in place in any memory layout (C or Fortran
    order, transposed, sliced, or reversed views), and the result is bitwise
    the same as for C-ordered copies of them. Only a misaligned, byte-swapped,
    or other-dtype operand is copied first.

    Examples
    --------
    >>> import numpy as np
    >>> import peregrine as pg
    >>> pg.matmul(np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2))
    array([[1., 2.],
           [3., 4.]])
    >>> pg.matmul(np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 1]])).dtype
    dtype('float64')
    >>> out = np.empty((2, 2))
    >>> pg.matmul(np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2), out=out) is out
    True
    >>> out
    array([[1., 2.],
           [3., 4.]])
    """
    r = _fast_matmul(a, b, out)
    if r is not NotImplemented:
        return r

    # Residency check, BEFORE the host fast path. Return type follows residency:
    # both operands on the device returns an pg.Array, both
    # on the host returns an ndarray, and a mix is an error -- never a silent
    # transfer. pg.Array is the CUDA-build device type; on a CPU-only build it is
    # a sentinel nothing is an instance of, so both isinstance checks are False
    # and control falls straight through to the host path with zero overhead.
    a_is_dev = isinstance(a, Array)
    b_is_dev = isinstance(b, Array)
    if a_is_dev != b_is_dev or (a_is_dev and out is not None):
        # Exactly one operand is device-resident, or a device product was asked
        # to land in a host out: the verbatim mixed-residency token, never a
        # silent host<->device transfer the caller did not ask for.
        raise TypeError(_MIXED_RESIDENCY_MSG)
    if a_is_dev and b_is_dev:
        # Both device-resident: the f32 (and forced f64) device path. f64 reaches
        # here ONLY because the caller explicitly put both operands on the device
        # (to_device) -- a host f64 array never auto-routes to the GPU (it would
        # not be an pg.Array), so the FP64 trap rule is upheld upstream. Auto mode
        # catches a CUDA runtime failure, warns ONCE, and falls back to the CPU.
        try:
            return _matmul_device(a, b)
        except RuntimeError as exc:
            # The native cuda_error arm maps every PG_CUDA_CHECK / cuBLAS / OOM
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
            # Atomic check-then-set under the lock so exactly one thread
            # ever sees the flag as unset and wins the warning, even when several
            # threads hit a device failure at once. warnings.warn runs OUTSIDE the
            # lock: it can invoke arbitrary user warning filters, and holding a
            # lock across that is needless contention -- the only invariant the
            # lock must protect is the single flip of the sentinel.
            with _cuda_fallback_lock:
                should_warn = not _cuda_fallback_warned
                _cuda_fallback_warned = True
            _push_routing_facts(True)
            if should_warn:
                name = _cuda_error_name(exc)
                warnings.warn(
                    f"cuda matmul failed ({name}), "
                    "falling back to cpu for this session",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return _matmul_cpu_fallback(a, b)

    # Fast path: when both operands are already 2-D C-contiguous
    # ndarrays of the same accepted float dtype, np.asarray, np.result_type, and
    # np.ascontiguousarray are all no-ops that only cost time, so skip the whole
    # chain and hand the kernel the arrays as-is. At n=8 the skipped helpers are
    # ~1.8 us, the entire 2x-NumPy budget, so this branch is what makes the small
    # win reachable rather than an optimization. It is a pre-branch, not a
    # replacement: any operand that is not an ndarray, not 2-D, not C-contiguous,
    # or a different/unaccepted dtype falls through to the full policy chain
    # below, which keeps the exact rejection, promotion, and error contract. The
    # inner-dimension and out checks run here too, so the shape contract holds.
    # No copy is made: a qualifying input is already contiguous, so the kernel
    # takes a zero-copy view of it.
    if (
        type(a) is np.ndarray
        and type(b) is np.ndarray
        and a.ndim == 2
        and b.ndim == 2
        and a.dtype == b.dtype
        and a.dtype in _ACCEPTED
        and a.flags.c_contiguous
        and b.flags.c_contiguous
        and a.flags.aligned
        and b.flags.aligned
    ):
        # The pair is already a normalized host ndarray pair, so route it on the
        # module-global backend. _host_backend_for is "cpu" in one cheap
        # comparison when _backend is "cpu" OR when auto sees no usable device --
        # the OFF build and any no-GPU machine pay only that test and fall straight
        # into the existing CPU kernel call, so the fast path stays a no-overhead
        # pre-branch. A GPU route only happens for a host f32 pair the policy (or a
        # forced set_backend("cuda")) sends to the device.
        _check_matmul_out(a, b, a.dtype, out)
        return _matmul_host(a, b, out)

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

    # Bring the operands to the promoted dtype in a layout the kernels read,
    # THEN route on the backend. Same backend decision as the fast path: CPU
    # unless a present GPU's measured crossover (or a forced "cuda") wins for
    # this host f32 pair. The overlap check sees the operands as passed, before
    # any copy, so the rule is the same whatever layout the caller used.
    _check_matmul_out(a_arr, b_arr, common, out)
    return _matmul_host(_matmul_operand(a_arr, common), _matmul_operand(b_arr, common), out)


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
    >>> import peregrine as pg
    >>> pg.transpose(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    array([[1., 4.],
           [2., 5.],
           [3., 6.]])
    >>> a = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> t = pg.transpose(a)
    >>> t[0, 0] = 99.0  # owned copy: the input is untouched
    >>> float(a[0, 0])
    1.0
    """
    r = _fast_transpose(a)
    if r is not NotImplemented:
        return r
    arr = _prepare(a, "a", "transpose")
    common = _resolve_dtype(arr, "transpose")
    arr = _normalize(arr, common)
    out = np.empty(arr.shape[::-1], common)
    _transpose_into(arr, out)
    return out


def _sum_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    # axis 0 leaves one value per column, axis 1 one per row.
    out = np.empty(arr.shape[1 - axis], arr.dtype)
    _sum_axis_into(arr, axis, out)
    return out


def sum(a, *, axis=None):
    """Sum of array elements over an axis.

    Matches ``numpy.sum`` for float32 and float64 operands, including
    zero-sized dimensions and NaN/Inf propagation, with summation performed
    in the input dtype. Rows and the whole array sum pairwise, as NumPy sums
    a contiguous axis. Along axis 0, blocks of up to 128 rows sum
    sequentially, as NumPy does, and the blocks then add pairwise, so tall
    arrays round less than a running sum. The result does not depend on the
    thread count.

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
    >>> import peregrine as pg
    >>> float(pg.sum(np.array([[1.0, 2.0], [3.0, 4.0]])))
    10.0
    >>> pg.sum(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
    array([4., 6.])
    >>> pg.sum(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=1)
    array([3., 7.])
    """
    r = _fast_sum(a, axis)
    if r is not NotImplemented:
        return r
    _check_axis(axis, "sum")
    arr = _prepare(a, "a", "sum")
    common = _resolve_dtype(arr, "sum")
    arr = _normalize(arr, common)
    if axis is None:
        # sum_all returns a Python float; cast it back to the result dtype so a
        # float32 input yields a float32 scalar (the round trip through a
        # double-precision Python float is lossless for a float32 value).
        return common.type(_sum_all_native(arr))
    return _sum_axis(arr, int(axis))


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
    >>> import peregrine as pg
    >>> float(pg.mean(np.array([[1.0, 2.0], [3.0, 4.0]])))
    2.5
    >>> pg.mean(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
    array([2., 3.])
    >>> pg.mean(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=1)
    array([1.5, 3.5])
    """
    r = _fast_mean(a, axis)
    if r is not NotImplemented:
        return r
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
        total = _sum_axis(arr, 0)
    else:
        count = arr.shape[1]
        total = _sum_axis(arr, 1)

    # When the reduced axis is empty but the output is not, NumPy warns "Mean of
    # empty slice" before the division. The division itself, count == 0, then
    # produces NumPy's invalid-value divide warning and the NaN result on its
    # own. axis=1 on a (0, n) input has count == n > 0 and an empty (0,) output,
    # so neither warning fires: the guard below is exactly that case excluded.
    out_size = total.size if isinstance(total, np.ndarray) else 1
    if count == 0 and out_size > 0:
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)

    # True division, never reciprocal multiply: x / count and x * (1 / count)
    # round differently, and NumPy composes mean with a division. count is a
    # Python int, so NumPy's weak-scalar rule converts it to the result dtype
    # and divides in that dtype (float32 stays float32, never a double divide).
    return total / count


def _fused_residency(operands, op: str) -> str:
    # Decide the residency of a fused call from its raw operands, BEFORE any host
    # normalization (the isinstance check must see the original objects). The
    # return-type-follows-residency rule, extended from
    # matmul's two-operand check to N operands: every operand on the device routes
    # the device path (returns an pg.Array); every operand on the host routes the
    # host CPU path (returns an ndarray); ANY mix is a TypeError, never a silent
    # host<->device transfer. On a CPU-only build pg.Array is a sentinel nothing
    # is an instance of, so this is always "host" and the device branch is dead.
    dev = [isinstance(x, Array) for x in operands]
    if all(dev):
        return "device"
    if any(dev):
        raise TypeError(_mixed_residency_msg(op))
    return "host"


def axpby(x, y, *, a=1.0, b=1.0, out=None):
    """Elementwise ``a*x + b*y`` for two same-shape 2-D arrays.

    Computes the unfused expression ``a*x + b*y`` within the elementwise
    tolerance for float32 and float64, including exact NaN/Inf propagation
    (matching the NumPy expression positionally).

    The scalars ``a`` and ``b`` are keyword-only. The two arrays must have the
    same shape: v1 does not broadcast, so a shape mismatch raises ValueError.
    The return type follows operand residency: host arrays give a host ndarray,
    pg.Array operands give a pg.Array.

    Parameters
    ----------
    x, y : array_like or pg.Array
        Operands; must be 2-D and the same shape, and on the same side (both
        host or both device).
    a, b : float, optional
        Scalar coefficients, keyword-only. Default 1.0 each. Each converts to
        the result dtype before the multiply, as NumPy does for a Python
        scalar, so float32 operands give a float32 result computed with
        float32 coefficients.
    out : numpy.ndarray, optional
        Host operands only, keyword-only. The result is written into ``out``,
        which is returned. It must be an exact numpy.ndarray of the result
        dtype in native byte order, of the operands' shape, C-contiguous,
        aligned, and writeable. It may be exactly one of the operands (same
        buffer, shape, and dtype) for an in-place update; any other overlap
        with an operand raises ValueError.

    Returns
    -------
    numpy.ndarray
        ``a*x + b*y`` elementwise; ``out`` itself when given. The result dtype
        is np.result_type across the operands when that lands on float32 or
        float64; integer operands promote to float64 (this is a float library).

    Raises
    ------
    ValueError
        If an operand is not 2-D, or the operands differ in shape,
        or ``out`` has the wrong shape, is not a writeable aligned C-contiguous
        array, or overlaps an operand other than exactly.
    TypeError
        If an operand dtype is bool, float16, complex64, complex128, or object,
        if promotion lands on anything other than float32/float64/integer, if
        the operands are split across host and device, or if
        ``out`` is not a numpy.ndarray of the result dtype, or device operands
        come with ``out``.

    Examples
    --------
    >>> import numpy as np
    >>> import peregrine as pg
    >>> x = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> y = np.array([[10.0, 20.0], [30.0, 40.0]])
    >>> pg.axpby(x, y, a=2.0, b=-1.0)
    array([[ -8., -16.],
           [-24., -32.]])
    >>> pg.axpby(x, y, a=2.0, b=-1.0, out=x) is x
    True
    >>> x
    array([[ -8., -16.],
           [-24., -32.]])
    """
    r = _fast_axpby(x, y, a, b, out)
    if r is not NotImplemented:
        return r
    if _fused_residency((x, y), "axpby") == "device":
        # Device-resident path: both operands are pg.Array, so the native
        # axpby Array overload runs the grid-stride kernel device-in/device-out and
        # returns an pg.Array. Scalars cross as Python floats; no host->GPU staging
        # (v1 is device-resident only). dtype/shape validation lives in the binding
        # (a mismatch raises ValueError via the shape_error translator). A device
        # result cannot land in a host out without a transfer nobody asked for.
        if out is not None:
            raise TypeError(_mixed_residency_msg("axpby"))
        return _axpby_device(x, y, float(a), float(b))
    xa = _prepare(x, "x", "axpby")
    ya = _prepare(y, "y", "axpby")
    common = _resolve_dtype_multi((xa, ya), "axpby")
    _check_same_shape((xa, ya), "axpby")
    out = _fused_out(out, (xa, ya), "axpby", common)
    # Host fused ALWAYS computes on the CPU: v1 has no host->GPU auto-staging
    # (no _matmul_gpu_staged analog, no policy/choose_backend call). float(a) /
    # float(b) cast the scalars at the boundary, mirroring sum's int(axis); the
    # binding narrows them to the result dtype.
    _axpby_into(
        _normalize(xa, common), _normalize(ya, common), float(a), float(b), out
    )
    return out


def fma3(x, y, z, *, out=None):
    """Elementwise fused multiply-add ``x*y + z`` for three same-shape arrays.

    Computes ``x*y + z`` in a single rounding (a true fused multiply-add), so it
    can be closer to the real-number result than the two-rounding unfused NumPy
    ``x*y + z``; both stay within the elementwise tolerance. NaN/Inf match the
    NumPy expression positionally, including ``inf*0 + z`` giving NaN.

    The three arrays must have the same shape: v1 does not broadcast, so a shape
    mismatch raises ValueError. The return type follows operand residency.

    Parameters
    ----------
    x, y, z : array_like or pg.Array
        Operands; must be 2-D and the same shape, and on the same side (all host
        or all device).
    out : numpy.ndarray, optional
        Host operands only, keyword-only. The result is written into ``out``,
        which is returned. It must be an exact numpy.ndarray of the result
        dtype in native byte order, of the operands' shape, C-contiguous,
        aligned, and writeable. It may be exactly one of the operands (same
        buffer, shape, and dtype) for an in-place update; any other overlap
        with an operand raises ValueError.

    Returns
    -------
    numpy.ndarray
        ``x*y + z`` elementwise; ``out`` itself when given. The result dtype is
        np.result_type across all three operands when that lands on float32 or
        float64; integer operands promote to float64.

    Raises
    ------
    ValueError
        If an operand is not 2-D, or the operands differ in shape,
        or ``out`` has the wrong shape, is not a writeable aligned C-contiguous
        array, or overlaps an operand other than exactly.
    TypeError
        If an operand dtype is bool, float16, complex64, complex128, or object,
        if promotion lands on anything other than float32/float64/integer, if
        the operands are split across host and device, or if
        ``out`` is not a numpy.ndarray of the result dtype, or device operands
        come with ``out``.

    Examples
    --------
    >>> import numpy as np
    >>> import peregrine as pg
    >>> x = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> y = np.array([[5.0, 6.0], [7.0, 8.0]])
    >>> z = np.array([[1.0, 1.0], [1.0, 1.0]])
    >>> pg.fma3(x, y, z)
    array([[ 6., 13.],
           [22., 33.]])
    >>> out = np.empty((2, 2))
    >>> pg.fma3(x, y, z, out=out) is out
    True
    """
    r = _fast_fma3(x, y, z, out)
    if r is not NotImplemented:
        return r
    if _fused_residency((x, y, z), "fma3") == "device":
        # Device-resident path: all three operands are pg.Array, so the
        # native fma3 Array overload runs the device kernel and returns an
        # pg.Array. No host->GPU staging.
        if out is not None:
            raise TypeError(_mixed_residency_msg("fma3"))
        return _fma3_device(x, y, z)
    xa = _prepare(x, "x", "fma3")
    ya = _prepare(y, "y", "fma3")
    za = _prepare(z, "z", "fma3")
    common = _resolve_dtype_multi((xa, ya, za), "fma3")
    _check_same_shape((xa, ya, za), "fma3")
    out = _fused_out(out, (xa, ya, za), "fma3", common)
    _fma3_into(
        _normalize(xa, common), _normalize(ya, common), _normalize(za, common), out
    )
    return out


def scaled_relu(x, *, scale=1.0, out=None):
    """Elementwise ``maximum(scale*x, 0)`` for a 2-D array.

    Computes ``np.maximum(scale*x, 0)`` within the elementwise tolerance for
    float32 and float64. A NaN input PROPAGATES as NaN exactly like np.maximum
    (it does not collapse to 0); signed Inf matches positionally.

    The scalar ``scale`` is keyword-only. The return type follows operand
    residency (a host array gives a host ndarray).

    Parameters
    ----------
    x : array_like or pg.Array
        Input; must be 2-D.
    scale : float, optional
        Scalar multiplier applied before the rectifier, keyword-only. Default
        1.0. It converts to the result dtype before the multiply, as NumPy does
        for a Python scalar, so a float32 input gives a float32 result.
    out : numpy.ndarray, optional
        Host operands only, keyword-only. The result is written into ``out``,
        which is returned. It must be an exact numpy.ndarray of the result
        dtype in native byte order, of the input's shape, C-contiguous,
        aligned, and writeable. It may be exactly ``x`` (same
        buffer, shape, and dtype) for an in-place update; any other overlap
        with ``x`` raises ValueError.

    Returns
    -------
    numpy.ndarray
        ``maximum(scale*x, 0)`` elementwise; ``out`` itself when given. The
        result dtype is the input dtype after promotion (integers promote to
        float64).

    Raises
    ------
    ValueError
        If ``x`` is not 2-D, or ``out`` has the wrong shape, is not a writeable
        aligned C-contiguous array, or overlaps ``x`` other than exactly.
    TypeError
        If the dtype is bool, float16, complex64, complex128, object, or any
        promoted dtype other than float32 or float64, or if
        ``out`` is not a numpy.ndarray of the result dtype, or device operands
        come with ``out``.

    Examples
    --------
    >>> import numpy as np
    >>> import peregrine as pg
    >>> x = np.array([[-1.0, 2.0], [3.0, -4.0]])
    >>> pg.scaled_relu(x, scale=3.0)
    array([[0., 6.],
           [9., 0.]])
    >>> pg.scaled_relu(x, scale=3.0, out=x) is x
    True
    >>> x
    array([[0., 6.],
           [9., 0.]])
    """
    r = _fast_scaled_relu(x, scale, out)
    if r is not NotImplemented:
        return r
    if _fused_residency((x,), "scaled_relu") == "device":
        # Device-resident path: the operand is an pg.Array, so the native
        # scaled_relu Array overload runs the device kernel (NaN-propagating, like
        # the CPU path) and returns an pg.Array. No host->GPU staging.
        if out is not None:
            raise TypeError(_mixed_residency_msg("scaled_relu"))
        return _scaled_relu_device(x, float(scale))
    xa = _prepare(x, "x", "scaled_relu")
    common = _resolve_dtype_multi((xa,), "scaled_relu")
    out = _fused_out(out, (xa,), "scaled_relu", common)
    _scaled_relu_into(_normalize(xa, common), float(scale), out)
    return out
