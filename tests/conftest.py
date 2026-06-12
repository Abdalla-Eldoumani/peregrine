"""Shared pytest infrastructure: hypothesis profiles, the slow marker, the
machine manifest header, and assert_matmul_close, the only toleranced
comparison path in the suite. An inline rtol/atol anywhere else is a defect.

Tolerance contract rationale, float64. A pure relative bound cannot hold for
matmul: elements produced by cancellation have references near zero, so a
difference of one or two ulps in absolute terms becomes an unbounded relative
one. The divergence class is mechanical, not a kernel bug. MSVC under
/fp:precise stopped emitting FMA contractions in VS 2022, while NumPy's BLAS
kernels use FMA explicitly, so on Windows the naive kernel takes two roundings
per multiply-add step where the reference takes one. Each of the k
accumulation steps can therefore differ by at most one extra rounding of
magnitude eps64 * amax(a) * amax(b), and the worst-case difference grows
linearly in k. The Higham-style absolute term 4 * k * eps64 * amax(a) *
amax(b) bounds exactly that class, with margin constant 4. The
smallest-subnormal floor exists because the product term underflows to exactly
zero for deep-denormal operands, which would silently reduce the contract to
the vacuous relative-only check; one rounding step in the subnormal regime is
quantized at one denormal ulp, so the floor is the principled minimum.

float32 results are held to a different bar: both results are compared against
the float64 ground truth and ours must stay within a small multiple of NumPy's
own rounding error, with a floor of k * eps32 scaled by the operands' largest
finite magnitudes. The scaling exists for the same reason as the float64 atol
term: one float32 rounding step is eps32-sized at data scale, and NumPy's FMA
path can land arbitrarily close to the truth on cancellation elements, so an
unscaled floor is vacuous away from unit-scale data. A fixed rtol is exactly
what failed at k=750 historically.

Special values are exact: NaN positions, Inf positions, and Inf signs must
match NumPy positionally. The tolerance contract applies only to the mutually
finite remainder.
"""

import os
import platform

import numpy as np
from hypothesis import settings

import fastmathext as fme

try:
    from threadpoolctl import threadpool_info
except ImportError:
    threadpool_info = None

# hypothesis never reads the HYPOTHESIS_PROFILE env var on its own; only the
# --hypothesis-profile CLI flag is plugin-handled, and that flag cleanly
# overrides this load if ever passed. The explicit load below is therefore
# mandatory, not a convenience. deadline=None in both profiles: spike shapes
# breach the default 200 ms per-example deadline.
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile(
    "ci", max_examples=200, derandomize=True, print_blob=True, deadline=None
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))

EPS64 = float(np.finfo(np.float64).eps)
EPS32 = float(np.finfo(np.float32).eps)
SMALLEST_SUBNORMAL = float(np.finfo(np.float64).smallest_subnormal)
C_HIGHAM = 4.0


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running property spikes")


def pytest_report_header(config):
    if threadpool_info is None:
        pools = "threadpoolctl not installed"
    else:
        pools = "; ".join(
            f"{d.get('internal_api')} {d.get('version') or '?'} "
            f"({d.get('num_threads')} threads)"
            for d in threadpool_info()
        )
    return [
        f"machine: {platform.platform()} | python {platform.python_version()}",
        f"numpy {np.__version__} | fme {fme.__version__} "
        f"| cpu_features {fme.cpu_features()}",
        f"threadpools: {pools}",
    ]


def assert_matmul_close(got, ref, a, b):
    """Assert got matches ref under the suite tolerance contract.

    float64, elementwise, on all-finite results:
        |got - ref| <= 1e-12 * |ref| + atol
        atol = max(4 * k * eps64 * amax(a) * amax(b), smallest subnormal)
    float32: both results against the float64 ground truth a64 @ b64; ours
    within 4 * max(NumPy's own max abs error, k * eps32 * amax(a) * amax(b)).
    Non-finite entries must match positionally, including each Inf's sign;
    the toleranced comparison then covers the mutually finite remainder only,
    with amax taken over finite entries.

    Pass the operands actually multiplied as a and b: the float32 arm
    recomputes the ground truth from them and the float64 atol scales with
    their magnitudes and the inner dimension k = a.shape[1]. A non-finite
    amax(a) * amax(b) product fails outright: the contract is unenforceable
    at such magnitudes and must never pass vacuously.
    """
    assert got.shape == ref.shape, f"shape mismatch: {got.shape} vs {ref.shape}"
    assert got.dtype == ref.dtype, f"dtype mismatch: {got.dtype} vs {ref.dtype}"

    finite = None
    if not (np.isfinite(got).all() and np.isfinite(ref).all()):
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        np.testing.assert_array_equal(np.isinf(got), np.isinf(ref))
        inf_positions = np.isinf(ref)
        # value equality at the inf positions checks each inf's sign
        np.testing.assert_array_equal(got[inf_positions], ref[inf_positions])
        finite = np.isfinite(got) & np.isfinite(ref)
        got = got[finite]
        ref = ref[finite]
        if got.size == 0:
            return

    k = a.shape[1]
    # amax runs over finite entries only: an injected NaN or Inf in an
    # operand would otherwise poison the bound for finite remainder elements
    # whose accumulations never touch it. Every reduction takes initial=0.0:
    # a bare .max() raises ValueError on zero-size operands, and k=0 must
    # yield an exactly-zero bound. The product runs through Python float
    # (float64), never the operand dtype: a float32 product overflows to inf
    # above roughly 1.8e19 per operand, and an infinite bound would pass any
    # result whatsoever
    amax_prod = float(np.abs(a[np.isfinite(a)]).max(initial=0.0)) * float(
        np.abs(b[np.isfinite(b)]).max(initial=0.0)
    )
    assert np.isfinite(amax_prod), (
        "operand magnitudes overflow the tolerance bound; contract unenforceable"
    )
    if got.dtype == np.float64:
        atol = max(C_HIGHAM * k * EPS64 * amax_prod, SMALLEST_SUBNORMAL)
        diff = np.abs(got - ref)
        bound = 1e-12 * np.abs(ref) + atol
        ok = diff <= bound
        if not ok.all():
            worst = int(np.argmax(diff - bound))
            raise AssertionError(
                f"f64 contract violated at {int((~ok).sum())} of {diff.size} "
                f"elements: max abs diff {float(diff.max(initial=0.0)):.3e}, "
                f"worst element diff {float(diff.flat[worst]):.3e} vs bound "
                f"{float(bound.flat[worst]):.3e} (k={k}, atol={atol:.3e})"
            )
    else:
        # float32 summation order differs from BLAS, so a fixed rtol fails
        # once k grows (k=750 broke rtol=1e-5). Compare both results against
        # the float64 ground truth and require ours to stay within a small
        # multiple of NumPy's own rounding error. The floor scales with the
        # operands' magnitudes for the same reason the float64 atol does:
        # one rounding step is eps32-sized at data scale, and NumPy's FMA
        # path can land arbitrarily closer to the truth on cancellation
        # elements, so an unscaled k*eps32 floor is vacuous off unit scale.
        truth = a.astype(np.float64) @ b.astype(np.float64)
        if finite is not None:
            truth = truth[finite]
        err_got = np.abs(got.astype(np.float64) - truth).max(initial=0.0)
        err_ref = np.abs(ref.astype(np.float64) - truth).max(initial=0.0)
        bound = 4.0 * max(err_ref, k * EPS32 * amax_prod)
        assert err_got <= bound, (
            f"f32 max abs error {err_got:.3e} exceeds bound {bound:.3e} "
            f"(numpy's own error vs float64 truth: {err_ref:.3e})"
        )
