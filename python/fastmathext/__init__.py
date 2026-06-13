"""fastmathext: heterogeneous linear algebra with a NumPy-compatible surface.

The native core takes zero-copy views of C-contiguous float32 and float64
arrays. This wrapper supplies the NumPy-style ergonomics on top: dtype
promotion, contiguity normalization, and clear errors.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._core import cpu_features, has_cuda_build
from ._core import matmul as _matmul_native
from ._core import sum_all as _sum_all_native
from ._core import sum_axis as _sum_axis_native
from ._core import transpose as _transpose_native

__version__ = "3.0.0.dev0"
__all__ = [
    "matmul",
    "transpose",
    "sum",
    "mean",
    "cpu_features",
    "has_cuda_build",
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


def matmul(a, b, *, out=None) -> np.ndarray:
    """Matrix product of two 2-D arrays.

    Matches numpy.matmul for float32 and float64 operands, including
    zero-sized dimensions and NaN/Inf propagation.

    Parameters
    ----------
    a : array_like
        Left operand; must be 2-D.
    b : array_like
        Right operand; must be 2-D with as many rows as a has columns.
    out : None, optional
        Reserved for a future preallocated-output path. The keyword is part
        of the signature ahead of its implementation; only None is accepted.

    Returns
    -------
    numpy.ndarray
        The matrix product. The result dtype is np.result_type(a, b) when
        that lands on float32 or float64, so int8/int16/uint8/uint16 mixed
        with float32 give float32. Integer-with-integer products always
        promote to float64, unlike NumPy, which keeps int64: this is a
        float library, so every result is a float array.

    Raises
    ------
    ValueError
        If an operand is not 2-D, or the inner dimensions do not match.
    TypeError
        If an operand dtype is bool, float16, complex64, complex128, or
        object, or if promotion lands on anything other than float32,
        float64, or an integer dtype (longdouble and clongdouble on
        platforms where they are distinct).
    NotImplementedError
        If out is anything other than None.

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

    return _matmul_native(_normalize(a_arr, common), _normalize(b_arr, common))


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
