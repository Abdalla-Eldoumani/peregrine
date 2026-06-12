"""fastmathext: heterogeneous linear algebra with a NumPy-compatible surface.

The native core takes zero-copy views of C-contiguous float32 and float64
arrays. This wrapper supplies the NumPy-style ergonomics on top: dtype
promotion, contiguity normalization, and clear errors.
"""

from __future__ import annotations

import numpy as np

from ._core import cpu_features, has_cuda_build
from ._core import matmul as _matmul_native

__version__ = "3.0.0.dev0"
__all__ = ["matmul", "cpu_features", "has_cuda_build", "__version__"]

# Checked per operand before np.result_type runs: promotion would silently
# launder float16 and bool up to an accepted float dtype, hiding a conversion
# the caller never asked for. complex and object have no kernel path at all.
_REJECT = frozenset(
    map(np.dtype, ("bool", "float16", "complex64", "complex128", "object"))
)


def _prepare(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"matmul: {name} must be 2-dimensional, got ndim={arr.ndim}")
    return arr


def _normalize(arr: np.ndarray, common: np.dtype) -> np.ndarray:
    # ascontiguousarray is a no-op for arrays that already qualify, which is
    # the zero-copy fast path. Strided views pay one copy here until the
    # native kernels grow stride support.
    return np.ascontiguousarray(arr, dtype=common)


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
        The matrix product. float32 when both operands are float32,
        otherwise float64. Integer operands always promote to float64,
        unlike NumPy, which keeps int64: this is a float library, so every
        result is a float array.

    Raises
    ------
    ValueError
        If an operand is not 2-D, or the inner dimensions do not match.
    TypeError
        If an operand dtype is bool, float16, complex64, complex128, or
        object.
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
        common = np.dtype(np.float64)

    return _matmul_native(_normalize(a_arr, common), _normalize(b_arr, common))
