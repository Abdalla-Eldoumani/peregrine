"""Shared pytest infrastructure: hypothesis profiles, the slow marker, the
machine manifest header, and the two toleranced comparison paths in the suite,
assert_matmul_close and its reduction sibling assert_reduce_close. An inline
rtol/atol anywhere else is a defect.

The reduction contract (assert_reduce_close) is linear in the reduced count r
and scaled by S, the sum of magnitudes along the reduced axis, not a product of
operand extremes: reductions add, they do not multiply. NumPy reduces
sequentially along the non-contiguous axis (its axis=0 column sums err 1.44 at
r=1e5 where a pairwise bound predicts ~1e-3), so a logarithmic bound is vacuous;
Higham's sequential (r-1)*eps*S bounds both sides and therefore their
difference. Mean divides the sum atol by r. The full clause lives in the helper
docstring.

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
SMALLEST_SUBNORMAL32 = float(np.finfo(np.float32).smallest_subnormal)
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


def assert_reduce_close(got, ref, a, axis, kind="sum"):
    """Assert a reduction result matches ref under the suite tolerance contract.

    The sibling of assert_matmul_close and the only other toleranced path in
    the suite. The bound is LINEAR in the reduced count r, scaled by S, the sum
    of magnitudes along the reduced axis:

        |got - ref| <= 1e-12 * |ref| + atol
        r    = a.size for axis=None, a.shape[0] for axis=0, a.shape[1] for axis=1
        S    = max over outputs of sum(|finite addends|) along the reduced axis
        sum  f64 atol = max(4 * r * eps64 * S, smallest subnormal)
        sum  f32 atol = max(4 * r * eps32 * S, smallest subnormal32)
        mean atol     = sum atol / r

    The scale is S, not amax(a) * amax(b): reductions add rather than multiply,
    so the relevant magnitude is the running sum, not a product of operand
    extremes. The bound is linear, not logarithmic, because NumPy reduces
    sequentially along the non-contiguous axis (its axis=0 column sums err 1.44
    at r=1e5 where a pairwise bound predicts ~1e-3); Higham's sequential bound
    (r-1) * eps * S bounds both NumPy's accumulation and ours, so it bounds
    their difference, with c = 4 for 2x margin. Pure addition has no FMA
    asymmetry, so there is no dual-ground-truth arm.

    Pass the operand actually reduced as a; S is computed from it over finite
    addends only, so an injected NaN or Inf cannot poison the finite-remainder
    bound. A non-finite S fails outright: an infinite bound passes any result.
    For kind="mean" the atol is the sum atol divided by r; the final division
    adds one rounding absorbed by the rtol term. r = 0 sum compares exactly
    (atol collapses to the subnormal floor); the r = 0 mean is the exact-NaN /
    "Mean of empty slice" warning path, tested separately, and never reaches
    this toleranced branch.
    """
    assert got.shape == ref.shape, f"shape mismatch: {got.shape} vs {ref.shape}"
    assert got.dtype == ref.dtype, f"dtype mismatch: {got.dtype} vs {ref.dtype}"

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

    # r is the reduced element count per output. axis None reduces the whole
    # array; axis 0 reduces over rows (r = number of rows); axis 1 reduces over
    # columns (r = number of columns).
    if axis is None:
        r = a.size
        reduce_axis = None
    elif axis == 0:
        r = a.shape[0]
        reduce_axis = 0
    elif axis == 1:
        r = a.shape[1]
        reduce_axis = 1
    else:
        raise ValueError(f"assert_reduce_close: axis must be None, 0, or 1, got {axis!r}")

    # S is the largest per-output sum of absolute finite addends. Non-finite
    # operand entries are zeroed before the sum, never masked away, so the count
    # of addends (and therefore r) is unchanged: only their magnitude is dropped
    # so an injected special cannot inflate S. Through Python float, never the
    # operand dtype, mirroring the matmul helper's amax_prod: an f32 column sum
    # can overflow to inf, and an infinite bound passes any result.
    finite_addends = np.where(np.isfinite(a), np.abs(a), 0.0)
    scale = float(finite_addends.sum(axis=reduce_axis, dtype=np.float64).max(initial=0.0))
    assert np.isfinite(scale), (
        "reduction addend magnitudes overflow the tolerance bound; "
        "contract unenforceable"
    )

    eps = EPS64 if got.dtype == np.float64 else EPS32
    floor = SMALLEST_SUBNORMAL if got.dtype == np.float64 else SMALLEST_SUBNORMAL32
    sum_atol = max(C_HIGHAM * r * eps * scale, floor)
    # mean = sum / count: the division shrinks both the value and its error by r,
    # so the absolute atol divides by r too (r >= 1 here; the r = 0 mean never
    # reaches this branch, it is the exact-NaN/warning path).
    atol = sum_atol / r if kind == "mean" else sum_atol

    diff = np.abs(got - ref)
    bound = 1e-12 * np.abs(ref) + atol
    ok = diff <= bound
    if not ok.all():
        worst = int(np.argmax(diff - bound))
        raise AssertionError(
            f"{kind} reduction contract violated at {int((~ok).sum())} of "
            f"{diff.size} elements: max abs diff {float(diff.max(initial=0.0)):.3e}, "
            f"worst element diff {float(diff.flat[worst]):.3e} vs bound "
            f"{float(bound.flat[worst]):.3e} (r={r}, S={scale:.3e}, atol={atol:.3e})"
        )
