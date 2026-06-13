"""Fused elementwise suite skeleton: axpby, fma3, scaled_relu, both backends.

Wave-0 scaffold. The kernels do not exist yet (they land in the later Phase 6
plans); this file proves it collects-and-passes cleanly on the OFF build so the
op bodies drop into a live module. It imports the single fused toleranced path
assert_fused_close (conftest) and the requires_cuda gate, defines the size grid,
and reserves the -k area names from the phase Test Map so each later plan fills
its own placeholder under its own name:

    cpu and oracle  -> FUSE-02 CPU AVX2 matches unfused NumPy, all dtypes/sizes
    relu and nan    -> FUSE-02 scaled_relu NaN propagation (the np.maximum trap)
    thread          -> FUSE-02 bitwise thread stability (OMP 1 vs N subprocess)
    error           -> FUSE-02 dtype rejection / promotion / same-shape errors
    gpu             -> FUSE-03 device grid-stride/float4 matches the same oracle
    property        -> FUSE-04 hypothesis property over drawn shapes + large @example

Every CPU-vs-NumPy and GPU-vs-NumPy comparison routes through assert_fused_close,
never an inline tolerance, exactly like the matmul and reduction suites. fma3
compares against the unfused NumPy x*y + z and the helper allows the
single-rounding fused result to be at least as accurate (DESIGN_SYSTEM
elementwise-fused clause). GPU tests skip cleanly with a stated reason on a
CPU-only build or a machine without a device (tests/CLAUDE.md), so the whole file
stays green on the WSL/GCC clone and the default CPU-only Windows build.
"""

import numpy as np
import pytest

import fastmathext as fme  # noqa: F401  (op wrappers land in later plans; kept for the live module)
from conftest import assert_fused_close, requires_cuda

# The legacy-killer sizes never leave the grid: the archived kernel corrupted
# results at sizes whose tail was not divisible by four, and the scalar remainder
# below the 8-lane (f32) / 4-lane (f64) AVX2 block and the float4 CUDA tail are
# the analogous danger here. Round sizes exercise the full-vector path.
LEGACY_KILLER_SIZES = [1, 3, 7, 50, 250, 333, 750]
ROUND_SIZES = [4, 8, 16, 64, 128, 256]
SIZES = LEGACY_KILLER_SIZES + ROUND_SIZES
DTYPES = [np.float32, np.float64]

# The oracle plan adds 16M and 16M-1 as explicit examples (the 0..16M range and
# the odd-tail float4 probe) rather than across the full cartesian grid, to keep
# suite runtime sane (CONTEXT: mind 16M runtime; RESEARCH FUSE-04 Pitfall 4).
LARGE_SIZE = 16 * 1024 * 1024
LARGE_ODD_SIZE = 16 * 1024 * 1024 - 1

# The @gpu skip marker, copied verbatim from test_cuda.py: requires_cuda() is the
# one gate (build flag AND a usable device). The bodies under it land in the GPU
# plan; the marker resolves to a clean skip on CPU-only and WSL.
_CUDA_OK, _CUDA_REASON = requires_cuda()
gpu = pytest.mark.skipif(not _CUDA_OK, reason=_CUDA_REASON)


def test_fused_helper_smoke():
    """The single toleranced path is importable and passes a trivial identity.

    Deliberately not gated: it must pass on every build so the scaffold is
    non-empty and collectable before any kernel exists. The op-specific oracle,
    NaN, thread, error, and property bodies land in the later Phase 6 plans under
    the reserved -k names documented in the module docstring.
    """
    x = np.zeros((2, 2), np.float64)
    assert_fused_close(x, x.copy())
