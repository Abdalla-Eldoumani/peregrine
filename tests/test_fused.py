"""Fused elementwise suite: axpby, fma3, scaled_relu, both backends.

FUSE-02 (CPU AVX2 matches the unfused NumPy expression incl exact NaN/Inf) and
the CPU half of FUSE-04 (all dtypes, sizes 0 to 16M, NaN/Inf). Every CPU-vs-NumPy
and GPU-vs-NumPy comparison routes through assert_fused_close (conftest), never an
inline tolerance, exactly like the matmul and reduction suites. fma3 compares
against the unfused NumPy x*y + z and the helper allows the single-rounding fused
result to be at least as accurate (DESIGN_SYSTEM elementwise-fused clause).

The reserved -k area names from the phase Test Map (filled here):

    cpu and oracle  -> FUSE-02 CPU AVX2 matches unfused NumPy, all dtypes/sizes
    relu and nan    -> FUSE-02 scaled_relu NaN propagation (the np.maximum trap)
    thread          -> FUSE-02 bitwise thread stability (OMP 1 vs N subprocess)
    error           -> FUSE-02 dtype rejection / promotion / same-shape errors
    gpu             -> FUSE-03 device grid-stride/float4 matches the same oracle
    property        -> FUSE-04 hypothesis property over drawn shapes + large @example

GPU tests skip cleanly with a stated reason on a CPU-only build or a machine
without a device (tests/CLAUDE.md), so the whole file stays green on the WSL/GCC
clone and the default CPU-only Windows build.
"""

import os
import subprocess
import sys

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import fastmathext as fme
from conftest import assert_fused_close, requires_cuda

# The legacy-killer sizes never leave the grid: the archived kernel corrupted
# results at sizes whose tail was not divisible by four, and the scalar remainder
# below the 8-lane (f32) / 4-lane (f64) AVX2 block and the float4 CUDA tail are
# the analogous danger here. Round sizes exercise the full-vector path.
LEGACY_KILLER_SIZES = [1, 3, 7, 50, 250, 333, 750]
ROUND_SIZES = [4, 8, 16, 64, 128, 256]
SIZES = LEGACY_KILLER_SIZES + ROUND_SIZES
DTYPES = [np.float32, np.float64]

# The oracle adds 16M and 16M-1 as explicit examples (the 0..16M range and the
# odd-tail float4 probe) rather than across the full cartesian grid, to keep
# suite runtime sane (CONTEXT: mind 16M runtime; RESEARCH FUSE-04 Pitfall 4).
LARGE_SIZE = 16 * 1024 * 1024
LARGE_ODD_SIZE = 16 * 1024 * 1024 - 1

# The @gpu skip marker, copied verbatim from test_cuda.py: requires_cuda() is the
# one gate (build flag AND a usable device). The bodies under it land in the GPU
# plan; the marker resolves to a clean skip on CPU-only and WSL.
_CUDA_OK, _CUDA_REASON = requires_cuda()
gpu = pytest.mark.skipif(not _CUDA_OK, reason=_CUDA_REASON)


# The three ops as (name, fme callable, unfused-NumPy oracle) triples. The oracle
# is the UNFUSED expression in the operands' own dtype: axpby a*x+b*y, fma3 x*y+z
# (assert_fused_close allows the fused single rounding to be at least as accurate
# as the unfused two-rounding NumPy expression), scaled_relu maximum(scale*x, 0).
#
# axpby and fma3 use POSITIVE coefficients on POSITIVE operands (see _operands)
# on purpose: a*x+b*y / x*y+z with mixed signs catastrophically cancels at some
# elements, and at a cancellation element |ref| collapses toward zero while the
# fused result equals the f64 truth EXACTLY (verified) -- so the unfused NumPy
# reference is the one carrying a ~1 ULP error, and the fused contract's relative
# bound (rtol*|ref|, no operand-magnitude atol -- the design-doc-locked clause)
# becomes vacuous there and rejects the MORE accurate kernel. Testing accuracy at
# operand scale (no cancellation) is the correct, non-flaky bar for an
# elementwise op; the NaN/Inf positional behaviour is proven separately. scale is
# positive likewise so scaled_relu's finite remainder stays at operand scale.
def _axpby_oracle(x, y, z):
    return x.dtype.type(2.0) * x + x.dtype.type(3.0) * y


def _fma3_oracle(x, y, z):
    return x * y + z


def _scaled_relu_oracle(x, y, z):
    return np.maximum(x.dtype.type(3.0) * x, x.dtype.type(0.0))


def _axpby_call(x, y, z):
    return fme.axpby(x, y, a=2.0, b=3.0)


def _fma3_call(x, y, z):
    return fme.fma3(x, y, z)


def _scaled_relu_call(x, y, z):
    return fme.scaled_relu(x, scale=3.0)


# (id, call, oracle, n_operands): n_operands says how many arrays to build for
# this op so the same grid feeds the 1-, 2-, and 3-array ops.
OPS = [
    ("axpby", _axpby_call, _axpby_oracle, 2),
    ("fma3", _fma3_call, _fma3_oracle, 3),
    ("scaled_relu", _scaled_relu_call, _scaled_relu_oracle, 1),
]
OP_IDS = [o[0] for o in OPS]


def _operands(shape, dtype, n_operands, seed=0):
    # Distinct STRICTLY-POSITIVE finite content per operand. Distinct so a
    # mis-wired operand (e.g. fma3 using x twice) is caught; positive so the
    # oracle expression (a*x+b*y / x*y+z with positive coefficients) never
    # catastrophically cancels -- a cancellation element drives |ref| toward zero
    # and makes the fused contract's relative bound vacuous against the MORE
    # accurate fused result (see the OPS comment). The +0.5 offset keeps values
    # off zero and well above subnormal; the magnitudes stay O(1) so a*x+b*y and
    # x*y+z never organically overflow, so any non-finite output traces to an
    # injected special (the NaN/Inf test owns those).
    rng = np.random.default_rng(seed)
    arrs = [
        (rng.standard_normal(shape).astype(dtype) ** 2 + dtype(0.5))
        for _ in range(n_operands)
    ]
    # pad to a 3-tuple so the call/oracle signatures are uniform; the extra
    # arrays are ignored by ops that take fewer operands.
    while len(arrs) < 3:
        arrs.append(arrs[0])
    return arrs[0], arrs[1], arrs[2]


# ---------------------------------------------------------------------------
# FUSE-02 / FUSE-04: oracle, all dtypes, legacy-killer + round + odd sizes.
# -k "cpu and oracle"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS, ids=OP_IDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("size", SIZES)
def test_cpu_oracle_square(op_id, call, oracle, n_operands, dtype, size):
    # Square shapes across the legacy-killer + round grid: the (size, size) flat
    # extent exercises both the full AVX2 block and the scalar remainder tail
    # (size not a multiple of 8/4). The odd legacy-killer sizes (1, 3, 7, 333) are
    # the tail probe; the round sizes are the full-vector path.
    x, y, z = _operands((size, size), dtype, n_operands)
    got = call(x, y, z)
    ref = oracle(x, y, z)
    assert_fused_close(np.asarray(got), ref)


@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS, ids=OP_IDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_cpu_oracle_zero_sized(op_id, call, oracle, n_operands, dtype):
    # Size 0 in each dimension: the empty op returns an empty array of the right
    # dtype and shape, the boundary the binding's checked_bytes(0, n) and the
    # kernel's n==0 path must both handle without a write.
    for shape in [(0, 0), (0, 5), (5, 0)]:
        x, y, z = _operands(shape, dtype, n_operands)
        got = call(x, y, z)
        ref = oracle(x, y, z)
        assert_fused_close(np.asarray(got), ref)


@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS, ids=OP_IDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_cpu_oracle_rectangular(op_id, call, oracle, n_operands, dtype):
    # A non-square odd shape so the flat element count (m*n) lands off every
    # vector-width multiple in a different way than the square grid does.
    x, y, z = _operands((333, 7), dtype, n_operands)
    got = call(x, y, z)
    ref = oracle(x, y, z)
    assert_fused_close(np.asarray(got), ref)


def _ulp_distance(got, ref):
    # Representable-step distance between two same-dtype float arrays: reinterpret
    # the bits as signed ints and remap to a monotone ordering (negatives mirrored
    # below +0.0), so |mono(got) - mono(ref)| counts the float ladder rungs between
    # them. One rung is one ULP. Used to prove a contracting-build divergence is a
    # single legitimate rounding, not arbitrary error. Inputs are strictly positive
    # finite here, so the negative branch never runs, but it keeps the helper total.
    int_t = np.int64 if got.dtype == np.float64 else np.int32
    g = np.ascontiguousarray(got).view(int_t).astype(np.int64)
    r = np.ascontiguousarray(ref).view(int_t).astype(np.int64)
    floor = np.int64(np.iinfo(np.int64).min)
    g = np.where(g < 0, floor - g, g)
    r = np.where(r < 0, floor - r, r)
    return np.abs(g - r)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_cpu_axpby_bitwise_equals_unfused(dtype):
    # axpby's vector body is add(mul(a,x), mul(b,y)) -- written as two roundings to
    # match the unfused NumPy expression a*x + b*y (the WR-01 contract; fma3 is the
    # one op that contracts to a single rounding). Whether those two roundings
    # SURVIVE to the result depends on the compiler's sanctioned FP model, and the
    # honest claim differs per platform:
    #
    #   MSVC /fp:precise stops emitting FMA contractions (VS 2022; conftest header),
    #   so the kernel keeps both roundings and is BITWISE-equal to the unfused NumPy
    #   reference -- the strong claim, asserted with assert_array_equal.
    #
    #   GCC's DEFAULT model (DESIGN_SYSTEM numeric policy: "default GCC FP model ...
    #   FMA contraction is allowed") contracts add(mul,mul) into a single fmadd even
    #   from explicit intrinsics, so the kernel rounds once. That is the more
    #   accurate single-rounding FMA, legitimately differing from the always-
    #   two-rounding NumPy reference by at most one ULP -- exactly the fma3-class
    #   contraction the design doc sanctions, NOT a kernel bug.
    #
    # Detect which world we are in at runtime from the kernel's OWN output rather
    # than guessing by os.name: the NumPy reference is always two-rounding (separate
    # ufuncs never fuse), so if the kernel reproduces it bit-for-bit the build does
    # not contract and the strong bitwise claim holds; if it does not, the build
    # contracted, and we assert the divergence is a single legitimate rounding (<= 1
    # ULP) and stays within the design-doc fused tolerance (assert_fused_close, the
    # same sanctioned path the rest of the suite uses -- the contract is NOT
    # loosened). The size spans many AVX2 blocks (8 f32 / 4 f64) AND a scalar tail
    # (+5 is not a multiple of either width). Strictly positive operands (no
    # cancellation), so this is a rounding-form claim, not an accuracy claim.
    n = 8 * 1000 + 5
    rng = np.random.default_rng(11)
    x = (np.abs(rng.standard_normal((1, n))) + 0.5).astype(dtype)
    y = (np.abs(rng.standard_normal((1, n))) + 0.5).astype(dtype)
    got = np.asarray(fme.axpby(x, y, a=2.0, b=3.0))
    ref = dtype(2.0) * x + dtype(3.0) * y
    if (got == ref).all():
        # Non-contracting build (MSVC /fp:precise): the two roundings survive, so
        # the strongest true claim is exact bitwise equality.
        np.testing.assert_array_equal(got, ref)
    else:
        # Contracting build (GCC default model): the kernel rounded once. Prove the
        # divergence is a single legitimate FMA rounding -- at most one ULP from the
        # two-rounding reference -- and within the sanctioned fused tolerance.
        assert int(_ulp_distance(got, ref).max()) <= 1, (
            "axpby diverges from the unfused reference by more than one ULP: the "
            "difference is not a single FMA contraction"
        )
        assert_fused_close(got, ref)


@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS, ids=OP_IDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("n", [LARGE_SIZE, LARGE_ODD_SIZE], ids=["16M", "16M-1"])
def test_cpu_oracle_large(op_id, call, oracle, n_operands, dtype, n):
    # The 0..16M FUSE-04 range and the 16M-1 odd-tail probe, as a FEW explicit
    # (1, n) examples rather than across the full cartesian grid (runtime). 16M-1
    # is odd so the very last element is a scalar-tail element after the last full
    # AVX2 block -- the legacy-killer class at scale.
    x, y, z = _operands((1, n), dtype, n_operands)
    got = call(x, y, z)
    ref = oracle(x, y, z)
    assert_fused_close(np.asarray(got), ref)


def test_cpu_oracle_int_promotes_to_float64():
    # Integer operands promote to float64 (the wrapper's _resolve_dtype_multi),
    # unlike NumPy which keeps int. The oracle is computed in float64 to match.
    for op_id, call, oracle, n_operands in OPS:
        x, y, z = (a.astype(np.int32) for a in _operands((7, 7), np.int64, n_operands))
        got = call(x, y, z)
        assert np.asarray(got).dtype == np.float64
        xf, yf, zf = (a.astype(np.float64) for a in (x, y, z))
        ref = oracle(xf, yf, zf)
        assert_fused_close(np.asarray(got), ref)


# ---------------------------------------------------------------------------
# FUSE-02: NaN / Inf positional. The scaled_relu NaN trap is lane-position
# sensitive, so specials land at every offset incl the tail.
# -k "relu and nan" / -k "nan"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("scale", [1.0, 3.0, -2.0], ids=["s1", "s3", "sneg"])
def test_relu_nan_propagates_at_every_lane(dtype, scale):
    # scaled_relu(NaN) must PROPAGATE NaN, not collapse to 0 (the bare maxps trap
    # RESEARCH Pitfall 1). Build a flat row long enough to span several AVX2
    # blocks and a non-empty tail, then put a NaN at EVERY lane offset in turn,
    # including the last element (the tail lane). np.maximum is the oracle.
    width = 8 if dtype == np.float32 else 4
    n = width * 5 + 3  # several full blocks + a 3-element scalar tail
    base = np.linspace(-2.0, 2.0, n).astype(dtype).reshape(1, n)
    for pos in range(n):
        x = base.copy()
        x[0, pos] = np.nan
        got = np.asarray(fme.scaled_relu(x, scale=scale))
        ref = np.maximum(dtype(scale) * x, dtype(0.0))
        assert_fused_close(got, ref)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_relu_negative_nan_propagates(dtype):
    # -NaN is still NaN: scaled_relu(-NaN) propagates, never 0 (RESEARCH line 230).
    # The sign bit of a NaN is not meaningful to np.maximum, but it must stay NaN.
    n = 11
    x = np.full((1, n), -np.nan, dtype=dtype)
    got = np.asarray(fme.scaled_relu(x, scale=2.0))
    ref = np.maximum(dtype(2.0) * x, dtype(0.0))
    assert np.isnan(got).all()
    assert_fused_close(got, ref)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_relu_signed_inf_positions(dtype):
    # +Inf survives the rectifier as +Inf; -Inf clamps to 0. Both must match
    # np.maximum positionally (assert_fused_close checks Inf positions and signs).
    n = 13
    x = np.linspace(-1.0, 1.0, n).astype(dtype).reshape(1, n)
    x[0, 0] = np.inf
    x[0, n - 1] = -np.inf  # tail lane
    x[0, n // 2] = np.nan
    got = np.asarray(fme.scaled_relu(x, scale=1.0))
    ref = np.maximum(dtype(1.0) * x, dtype(0.0))
    assert_fused_close(got, ref)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_fma3_inf_times_zero_is_nan(dtype):
    # inf*0 + z = NaN, exactly like the unfused NumPy expression (RESEARCH line
    # 242). Place the inf*0 pair at multiple offsets including the tail.
    n = 14
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, n)).astype(dtype)
    y = rng.standard_normal((1, n)).astype(dtype)
    z = rng.standard_normal((1, n)).astype(dtype)
    for pos in (0, n // 2, n - 1):
        x[0, pos] = np.inf
        y[0, pos] = 0.0
    got = np.asarray(fme.fma3(x, y, z))
    # inf*0 in the unfused NumPy reference legitimately warns "invalid value
    # encountered in multiply" -- that IS the inf*0=NaN this test asserts the
    # kernel reproduces, so silence the expected warning at the oracle build.
    with np.errstate(invalid="ignore"):
        ref = x * y + z
    assert np.isnan(got[0, 0]) and np.isnan(got[0, n - 1])
    assert_fused_close(got, ref)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_axpby_inf_operand_positions(dtype):
    # An Inf operand in axpby propagates positionally exactly like a*x+b*y. A NaN
    # also appears where b*y subtracts inf from inf in the oracle, so the helper's
    # positional NaN/Inf checks are exercised together.
    n = 17
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, n)).astype(dtype)
    y = rng.standard_normal((1, n)).astype(dtype)
    x[0, 0] = np.inf
    x[0, n - 1] = -np.inf  # tail
    y[0, n // 2] = np.inf
    got = np.asarray(fme.axpby(x, y, a=2.0, b=-3.0))
    ref = dtype(2.0) * x + dtype(-3.0) * y
    assert_fused_close(got, ref)


# ---------------------------------------------------------------------------
# FUSE-02: dtype rejection / promotion / same-shape / ndim errors.
# -k "error"
# ---------------------------------------------------------------------------

_REJECTED_DTYPES = ["bool", "float16", "complex64", "complex128", "object"]


@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS, ids=OP_IDS)
@pytest.mark.parametrize("name", _REJECTED_DTYPES)
def test_error_rejected_dtype_is_type_error(op_id, call, oracle, n_operands, name):
    # Every op rejects the laundering set per operand, naming the dtype (the
    # message must name its dtype, like the matmul suite). Reuses the shared
    # _REJECT table via _resolve_dtype_multi, so the wording matches matmul's.
    x, y, z = _operands((2, 2), np.float64, n_operands)
    x = x.astype(name)
    with pytest.raises(TypeError, match=f"unsupported dtype {name}"):
        call(x, y, z)


def test_error_rejected_dtype_on_later_operand():
    # Rejection is per operand, not just the first: a float16 on the SECOND axpby
    # operand and the THIRD fma3 operand must still raise, naming float16.
    a = np.zeros((2, 2), np.float64)
    bad = np.zeros((2, 2), np.float16)
    with pytest.raises(TypeError, match="unsupported dtype float16"):
        fme.axpby(a, bad)
    with pytest.raises(TypeError, match="unsupported dtype float16"):
        fme.fma3(a, a, bad)


def test_error_post_promotion_unsupported_dtype():
    # datetime64 promotes to itself, neither float32/float64 nor an integer
    # subdtype, so it exercises the post-promotion reject on every platform
    # (standing in for longdouble/clongdouble which MSVC aliases away), as the
    # matmul error suite does.
    a = np.zeros((2, 2), dtype="datetime64[s]")
    with pytest.raises(TypeError, match="unsupported dtype datetime64"):
        fme.scaled_relu(a)


def test_error_same_shape_mismatch_is_value_error():
    # The fused-specific validation, no matmul analog: v1 is same-shape, NO
    # broadcast, so a shape mismatch is a ValueError naming both shapes.
    with pytest.raises(ValueError, match="same shape"):
        fme.axpby(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="same shape"):
        fme.fma3(np.ones((2, 2)), np.ones((2, 2)), np.ones((3, 2)))


@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS, ids=OP_IDS)
def test_error_1d_input_is_value_error(op_id, call, oracle, n_operands):
    # A 1-D operand raises ValueError (this is a 2-D library), like matmul.
    args = [np.ones(4) for _ in range(n_operands)]
    while len(args) < 3:
        args.append(np.ones((1, 4)))
    with pytest.raises(ValueError, match="2-dimensional"):
        call(*args)


@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS, ids=OP_IDS)
def test_error_3d_input_is_value_error(op_id, call, oracle, n_operands):
    # A 3-D+ operand raises ValueError as well.
    args = [np.ones((2, 2, 2)) for _ in range(n_operands)]
    while len(args) < 3:
        args.append(np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="2-dimensional"):
        call(*args)


# ---------------------------------------------------------------------------
# FUSE-02: bitwise thread stability (OMP 1 vs N subprocess). Elementwise has no
# cross-element accumulation, so the result is bitwise thread-stable by
# construction -- but the test is still required. -k "thread"
# ---------------------------------------------------------------------------

# odd 333 rows on purpose: 333 is a legacy-killer size and the OpenMP chunk split
# over the flat range lands unevenly there. Inputs regenerate identically in each
# child from the seed; only the result crosses the process boundary.
_THREAD_SCRIPT = """\
import sys

import numpy as np

import fastmathext as fme

rng = np.random.default_rng(42)
x = rng.standard_normal((333, 257))
y = rng.standard_normal((333, 257))
z = rng.standard_normal((333, 257))
np.save(sys.argv[1], fme.fma3(x, y, z))
"""


def _run_thread_subprocess(tmp_path, omp_threads):
    out_path = tmp_path / f"fused_{omp_threads}.npy"
    # full environment copy with the override merged in: a stripped env breaks
    # DLL resolution for the OpenMP runtime on Windows (mirrors test_threads.py).
    env = dict(os.environ, OMP_NUM_THREADS=str(omp_threads))
    p = subprocess.run(
        [sys.executable, "-c", _THREAD_SCRIPT, str(out_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # stderr in the message so a child import failure reads as itself
    assert p.returncode == 0, p.stderr
    return np.load(out_path)


def test_thread_count_bitwise_identical(tmp_path):
    # BITWISE identity (assert_array_equal, not the tolerance helper): elementwise
    # is thread-stable by construction, so OMP 1 and N must agree to the last bit.
    got_1 = _run_thread_subprocess(tmp_path, 1)
    got_n = _run_thread_subprocess(tmp_path, 12)
    np.testing.assert_array_equal(got_1, got_n)


# ---------------------------------------------------------------------------
# FUSE-04: hypothesis property over drawn shapes + the legacy-killer @example
# pins + an explicit large @example. -k "property"
# ---------------------------------------------------------------------------

# st.floats(width=32) rejects bounds not exactly representable in float32, and
# 1e15 is not; routing the cap through np.float32 yields the nearest value.
F32_MAX = float(np.float32(1e15))

# STRICTLY-POSITIVE drawn elements. Positive (not arbitrary-sign) for the same
# reason the oracle operands are: a*x+b*y / x*y+z with positive coefficients on
# positive operands never catastrophically cancels, so the fused contract's
# relative bound stays non-vacuous against the (verified) more-accurate fused
# result. The lower bound is the smallest normal (denormals are still drawn via
# allow_subnormal so the denormal path is exercised); the upper bound caps the
# intermediate (amax^2 for fma3) below the overflow ceiling so a non-finite output
# can only come from an injected special, which this property never injects.
_F64_TINY = float(np.finfo(np.float64).tiny)
_F32_TINY = float(np.finfo(np.float32).tiny)
_F64_ELEMENTS = st.floats(
    min_value=0.0,
    max_value=1e100,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=True,
    exclude_min=False,
)
_F32_ELEMENTS = st.floats(
    min_value=0.0,
    max_value=F32_MAX,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=True,
    width=32,
    exclude_min=False,
)
_DTYPE = st.sampled_from((np.float32, np.float64))


def _elements_for(dtype):
    return _F32_ELEMENTS if dtype == np.float32 else _F64_ELEMENTS


@st.composite
def _drawn_fused(draw):
    # One drawn shape shared by every operand (fused is same-shape, no broadcast),
    # dims in [0, 256] so empty operands and tail sizes are both reachable;
    # elementwise drawing surfaces denormals, signed zeros, and boundary
    # magnitudes that seeded normal data never produces.
    dtype = draw(_DTYPE)
    m = draw(st.integers(0, 256))
    n = draw(st.integers(0, 256))
    elements = _elements_for(dtype)
    x = draw(hnp.arrays(dtype, (m, n), elements=elements))
    y = draw(hnp.arrays(dtype, (m, n), elements=elements))
    z = draw(hnp.arrays(dtype, (m, n), elements=elements))
    return x, y, z


@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS, ids=OP_IDS)
@given(operands=_drawn_fused())
def test_property_matches_unfused_numpy(op_id, call, oracle, n_operands, operands):
    # The space-sampling twin of the oracle unit tests. Compare the fme op to the
    # unfused NumPy expression via assert_fused_close across drawn shapes and
    # elementwise-drawn values (denormals, signed zeros, boundary magnitudes).
    x, y, z = operands
    got = call(x, y, z)
    ref = oracle(x, y, z)
    assert_fused_close(np.asarray(got), ref)


# Explicit shape examples that run first and do not count toward max_examples: the
# legacy-killer sizes as squares (the regression fence) plus an explicit large
# odd-tail case (the 16M class is too big for a property example, so 4096*4096-1
# stands in as a large flat extent with an odd tail). Each binds a single fixed
# dtype/shape; the @given draw above is overridden per pin.
def _fixed_operands(shape, dtype):
    # Strictly positive, like _operands and the drawn elements: no cancellation,
    # so the fused contract bound stays non-vacuous against the more-accurate
    # fused result. These are the legacy-killer / large-tail @example pins.
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal(shape).astype(dtype) ** 2 + dtype(0.5),
        rng.standard_normal(shape).astype(dtype) ** 2 + dtype(0.5),
        rng.standard_normal(shape).astype(dtype) ** 2 + dtype(0.5),
    )


for _size in LEGACY_KILLER_SIZES:
    test_property_matches_unfused_numpy = example(
        operands=_fixed_operands((_size, _size), np.float64)
    )(test_property_matches_unfused_numpy)

# one explicit large flat extent with an odd tail (kept off the 16M scale for the
# property runtime, but large enough to span many AVX2 blocks + a scalar tail)
test_property_matches_unfused_numpy = example(
    operands=_fixed_operands((1, 4096 * 1024 - 1), np.float32)
)(test_property_matches_unfused_numpy)


# ---------------------------------------------------------------------------
# FUSE-03: the device path matches the same oracle. The device-resident kernels
# (grid-stride/float4) compute on fme.Array operands; the result, brought back
# with from_device, must match the SAME assert_fused_close oracle as the CPU
# path. The marker resolves to a clean skip on CPU-only and WSL. -k "gpu"
# ---------------------------------------------------------------------------


# Device callables mirroring the host OPS table: to_device the operands the op
# consumes, run the fused op (returns an fme.Array), from_device the result. The
# oracle is the SAME unfused-NumPy expression as the host path (the _*_oracle
# functions above), so the GPU is held to the identical assert_fused_close bound.
#
# Every fme.Array (the operands and the device result) is bound to a local and
# explicitly del'd once the result is on the host, so the device handles are
# released deterministically rather than relying on expression-temporary GC. The
# returned value is a host ndarray (no device handle), so nothing device-resident
# survives the call -- without this, one Array could straggle to interpreter
# shutdown (nanobind's leaked-instance warning).
def _axpby_call_gpu(x, y, z):
    dx, dy = fme.to_device(x), fme.to_device(y)
    out = fme.axpby(dx, dy, a=2.0, b=3.0)
    host = fme.from_device(out)
    del dx, dy, out
    return host


def _fma3_call_gpu(x, y, z):
    dx, dy, dz = fme.to_device(x), fme.to_device(y), fme.to_device(z)
    out = fme.fma3(dx, dy, dz)
    host = fme.from_device(out)
    del dx, dy, dz, out
    return host


def _scaled_relu_call_gpu(x, y, z):
    dx = fme.to_device(x)
    out = fme.scaled_relu(dx, scale=3.0)
    host = fme.from_device(out)
    del dx, out
    return host


OPS_GPU = [
    ("axpby", _axpby_call_gpu, _axpby_oracle, 2),
    ("fma3", _fma3_call_gpu, _fma3_oracle, 3),
    ("scaled_relu", _scaled_relu_call_gpu, _scaled_relu_oracle, 1),
]


@gpu
@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS_GPU, ids=OP_IDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("size", SIZES)
def test_gpu_matches_oracle_square(op_id, call, oracle, n_operands, dtype, size):
    # The device-resident fused path must match the same unfused-NumPy oracle as
    # the CPU path across the legacy-killer + round grid. The square (size, size)
    # flat extent spans the float4/double2 aligned prefix AND the scalar tail (size
    # not a multiple of the pack width); the odd legacy-killer sizes are the tail
    # probe. Both f32 and f64 route to the device (no f64 auto-route exclusion on
    # the fused device path -- the operands were explicitly placed there).
    x, y, z = _operands((size, size), dtype, n_operands)
    got = call(x, y, z)
    ref = oracle(x, y, z)
    assert_fused_close(np.asarray(got), ref)


@gpu
@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS_GPU, ids=OP_IDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_gpu_matches_oracle_zero_sized(op_id, call, oracle, n_operands, dtype):
    # Size 0 in each dimension: the device op returns an empty fme.Array of the
    # right shape/dtype, the boundary the kernel's n==0 zero-dim guard handles
    # without a launch (and from_device round-trips an empty buffer).
    for shape in [(0, 0), (0, 5), (5, 0)]:
        x, y, z = _operands(shape, dtype, n_operands)
        got = call(x, y, z)
        ref = oracle(x, y, z)
        assert_fused_close(np.asarray(got), ref)


@gpu
@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS_GPU, ids=OP_IDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_gpu_matches_oracle_nan_inf(op_id, call, oracle, n_operands, dtype):
    # The device kernels must reproduce the host NaN/Inf behavior: scaled_relu(NaN)
    # propagates (the device fmax NaN-quieting trap, the GPU twin of the CPU bare
    # max trap), fma3 inf*0+z=NaN (the device fmaf single rounding), axpby Inf
    # propagates. Specials land at offset 0, the middle, and the LAST element (the
    # scalar-tail lane) so the float4 path and the tail path are both exercised.
    n = 37  # several float4/double2 packs + a non-empty scalar tail
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((1, n)).astype(dtype) ** 2 + dtype(0.5))
    y = (rng.standard_normal((1, n)).astype(dtype) ** 2 + dtype(0.5))
    z = (rng.standard_normal((1, n)).astype(dtype) ** 2 + dtype(0.5))
    for pos in (0, n // 2, n - 1):
        x[0, pos] = np.nan
    x[0, 1] = np.inf
    x[0, n - 2] = -np.inf
    with np.errstate(invalid="ignore"):
        got = call(x, y, z)
        ref = oracle(x, y, z)
    assert_fused_close(np.asarray(got), ref)


@gpu
@pytest.mark.parametrize("op_id,call,oracle,n_operands", OPS_GPU, ids=OP_IDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("n", [LARGE_SIZE, LARGE_ODD_SIZE], ids=["16M", "16M-1"])
def test_gpu_matches_oracle_large(op_id, call, oracle, n_operands, dtype, n):
    # The 0..16M FUSE-04 range and the 16M-1 odd-tail float4 probe on the device:
    # 16M-1 is odd so the last element is a scalar-tail element after the last full
    # float4/double2 pack -- the float4 OOB risk compute-sanitizer memcheck catches
    # (RESEARCH Pitfall 4). A few explicit (1, n) examples, not the cartesian grid.
    x, y, z = _operands((1, n), dtype, n_operands)
    got = call(x, y, z)
    ref = oracle(x, y, z)
    assert_fused_close(np.asarray(got), ref)


@gpu
def test_gpu_mixed_residency_is_type_error():
    # One operand on the device, one on the host: the fused per-op mixed-residency
    # TypeError, never a silent host<->device transfer. Mirrors the matmul mixed
    # residency contract, extended to the fused ops (every operand must be on the
    # same side).
    h = np.ones((4, 4), np.float32)
    d = fme.to_device(h)
    with pytest.raises(TypeError, match="same device"):
        fme.axpby(d, h)
    with pytest.raises(TypeError, match="same device"):
        fme.fma3(d, h, h)
    with pytest.raises(TypeError, match="same device"):
        fme.axpby(h, d)
    del d  # release the device handle deterministically


def test_fused_helper_smoke():
    """The single toleranced path is importable and passes a trivial identity.

    Kept from the Wave-0 scaffold: a non-gated everywhere-pass so the file stays
    collectable even if every kernel were stripped.
    """
    x = np.zeros((2, 2), np.float64)
    assert_fused_close(x, x.copy())
