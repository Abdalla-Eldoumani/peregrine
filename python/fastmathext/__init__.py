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


def _prepare(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"matmul: {name} must be 2-dimensional, got ndim={arr.ndim}")
    return arr


def matmul(a, b) -> np.ndarray:
    """Matrix product of two 2-D arrays.

    Matches numpy.matmul for float32 and float64 operands, including
    zero-sized dimensions. Integer and mixed inputs promote with
    numpy.result_type, so the result dtype is what NumPy would produce.
    """
    a_arr = _prepare(a, "a")
    b_arr = _prepare(b, "b")

    common = np.result_type(a_arr.dtype, b_arr.dtype)
    if common not in (np.dtype(np.float32), np.dtype(np.float64)):
        common = np.dtype(np.float64)

    # ascontiguousarray is a no-op for arrays that already qualify, which is
    # the zero-copy fast path. Strided views pay one copy here until the
    # native kernels grow stride support.
    a_c = np.ascontiguousarray(a_arr, dtype=common)
    b_c = np.ascontiguousarray(b_arr, dtype=common)

    return _matmul_native(a_c, b_c)
