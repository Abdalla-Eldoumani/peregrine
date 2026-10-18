"""Backend-decision policy: read the calibration cache and choose a backend.

This module is the dispatch brain split out from the wrapper. It owns three
things and nothing else:

* ``SCHEMA`` -- the cache-format integer (single source of truth; the calibrator
  imports it from here and stamps the same value into every cache it writes).
* the fail-safe cache read (``_cache_path`` + ``_load_cache``): resolve the cache
  location, read it, and return a COMPLETE calibration dict or ``None`` -- never
  a partially populated dict. ``None`` means "recalibrate or use static
  thresholds." Any parse error, an unknown schema, or a foreign machine
  signature all collapse to ``None``.
* ``choose_backend`` -- a PURE function: op/dtype/shape/residency/calibration/
  backend in, a ``"cpu"`` / ``"cuda"`` enum out. No file I/O, no warnings, no
  logging, no global mutation. The wrapper owns side effects; this only decides.

The load-bearing design fact (the reason this module exists rather than a single
threshold constant): the crossover compares MEASURED wall times interpolated at
the product's work, NOT one asymptotic GFLOP/s figure. A single-constant model --
cpu_time = work/CPU_GFLOPS vs gpu_time = work/GPU_GFLOPS + transfer using the warm
large-n throughput -- routes EVERYTHING, including a 256x256 host GEMM, to the
GPU, because at small n the GPU never reaches that warm throughput: a fixed
launch + cuBLAS setup + host-staging floor dominates. Interpolating the measured
times encodes that floor by construction, so a small host matrix correctly stays
on the CPU while a large one crosses to the GPU. The times are looked up by the
work 2*m*k*n and the host transfer by the bytes each operand moves, never by one
dimension alone, so a matrix-vector product with a large n is not mistaken for a
large square one.

Security: the cache is read with ``json.load`` ONLY, never ``pickle`` -- pickle
deserialization executes arbitrary code, JSON cannot. After loading, the schema
integer and the signature are validated before any field is trusted, so a
tampered or foreign cache yields at worst wrong routing (a performance bug),
never code execution.

Deserialization invariant: the cache read is lazy (first dispatch decision,
never at import) and the platformdirs import is deferred into ``_cache_path``, so
importing peregrine pays nothing for this module and the <50ms import budget
holds.
"""

import json
import math
import os

# Bump on ANY change to the calibration-dict fields; the loader rejects unknown
# schemas (a cache written by an older or newer build is discarded and the
# machine recalibrates). This is the single source of truth: calibrate.py
# imports SCHEMA from here and stamps it into every cache it writes, so the
# writer and the reader can never drift apart. Schema 2 adds the "staged" series;
# schema 1 caches also carry a CPU grid that was timed through the auto router and
# so includes its per-call routing cost, which must never be trusted again.
SCHEMA = 2

# The two manual-override backends route directly within rules 1-2 (correctness
# and the hard exclusions) without consulting the table; "auto" runs the
# measured policy. Exposed for the wrapper to validate PEREGRINE_BACKEND / set_backend
# against this set.
BACKENDS = ("auto", "cpu", "cuda")


def _cache_path(signature: str) -> str:
    """Resolve the calibration cache file path for one machine signature.

    ``PEREGRINE_CACHE_DIR`` overrides the location (the test-isolation hook: tests
    point it at a tmp dir so the real per-user cache is never touched).
    Otherwise the platform user cache dir is used with ``appauthor=False`` for
    the clean single-level ``.../peregrine/`` directory -- the default
    double-nests the app name as both author and name. The file name carries a
    short hash of the signature, so a CPU-only build and a CUDA build sharing one
    cache directory keep separate caches instead of overwriting each other on
    every calibration. platformdirs and hashlib are imported here, not at module
    top, so the import cost is paid on first dispatch, never at
    ``import peregrine``.

    Parameters
    ----------
    signature : str
        The machine/build signature the cache belongs to.

    Returns
    -------
    str
        Absolute path to ``calibration-<hash>.json`` under the resolved cache dir.
    """
    import hashlib

    override = os.environ.get("PEREGRINE_CACHE_DIR")
    if override:
        base = override
    else:
        import platformdirs

        base = platformdirs.user_cache_dir("peregrine", appauthor=False)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    return os.path.join(base, f"calibration-{digest}.json")


def _load_cache(path: str, expected_signature: str):
    """Read the calibration cache, returning a complete dict or ``None``.

    The result is ALWAYS either a fully-formed calibration dict or ``None`` --
    never a partially populated dict. ``None`` is the recalibrate/static-fallback
    signal. The rejection order is fixed:

    1. the file is missing, unreadable, or not valid JSON -> ``None``;
    2. the parsed value is not a dict, or its schema integer is missing or does
       not match ``SCHEMA`` (a foreign / older / newer cache) -> ``None``;
    3. the signature does not match the running machine/build -> ``None``;
    4. the body is structurally malformed -- a missing or empty ``cpu`` grid, a
       grid point that is not an ``[n, time]`` pair of positive numbers, a grid
       whose ``n`` does not strictly ascend, or a ``gpu`` series present without
       a well-formed ``staged`` series and ``transfer`` triple -> ``None``.

    The schema integer promises the fields exist and are well-formed, but a
    truncated file, a non-atomic external writer, or a foreign tool can stamp the
    matching schema and signature onto a broken body. Step 4 is the reader-side
    defense: a structurally malformed body collapses to ``None`` like every other
    rejection so a partial dict never reaches :func:`choose_backend`, where
    ``grid[0][0]`` / ``calibration["cpu"][dtype]`` / ``calibration["transfer"]``
    would otherwise raise ``IndexError`` / ``KeyError``, and the log-log
    interpolation would raise on a zero time or a repeated ``n``, on the
    dispatch hot path.

    Only ``json.load`` is used -- never ``pickle`` -- so a tampered cache cannot
    execute code; it can only be rejected or, at worst, route sub-optimally.

    Parameters
    ----------
    path : str
        Cache file path, from :func:`_cache_path`.
    expected_signature : str
        The running machine/build signature; a mismatch discards the cache.

    Returns
    -------
    dict or None
        The complete calibration dict when the file parses, matches the schema,
        and matches the signature; otherwise ``None``.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # missing or corrupt -> recalibrate, never a half-load
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        # not a calibration dict, or a missing / unknown schema version
        return None
    if data.get("signature") != expected_signature:
        # a different machine or build wrote this cache
        return None
    # Structural validation (step 4): the schema + signature gate does not prove
    # the body is well-formed. A non-empty per-dtype [n, time] CPU grid is
    # required, and if a "gpu" series is present then the "staged" series its
    # fixed cost is fitted from and the "transfer" triple must be present and
    # complete (a positive bandwidth pair so the _transfer_time division below
    # cannot hit a zero). Any defect collapses to None -- recalibrate or static
    # fallback -- exactly like a parse/schema/signature failure, so a malformed
    # body never reaches choose_backend.
    cpu = data.get("cpu")
    if not isinstance(cpu, dict):
        return None
    if not (_is_grid(cpu.get("float32")) and _is_grid(cpu.get("float64"))):
        return None
    gpu = data.get("gpu")
    if gpu is not None:
        staged = data.get("staged")
        if not (
            isinstance(gpu, dict)
            and _is_grid(gpu.get("float32"))
            and isinstance(staged, dict)
            and _is_grid(staged.get("float32"))
        ):
            return None
        transfer = data.get("transfer")
        if not isinstance(transfer, dict) or not all(
            k in transfer for k in ("h2d_gbps", "d2h_gbps", "fixed_overhead_s")
        ):
            return None
        if not (transfer["h2d_gbps"] > 0 and transfer["d2h_gbps"] > 0):
            return None
    return data


def _is_grid(grid) -> bool:
    # The log-log interpolation divides by logs of neighbouring sizes and times,
    # so it needs positive values and strictly ascending sizes.
    return (
        isinstance(grid, list)
        and len(grid) > 0
        and all(isinstance(p, list) and len(p) == 2 and p[0] > 0 and p[1] > 0 for p in grid)
        and all(p[0] < q[0] for p, q in zip(grid, grid[1:]))
    )


def _interp_time(grid, flops) -> float:
    """Log-log interpolate a measured wall time at ``flops`` of work.

    ``grid`` is one ascending ``[n, time_s]`` series of square products, so the
    point at ``n`` sits at ``F = 2*n**3``. Interpolating log time against log F
    is the same as against the log of the equivalent cube side
    ``(F/2)**(1/3)``, since one log is an affine function of the other, so the
    grid's own sizes serve as the knots. GEMM cost is close to a power of its
    work, which a straight line in log-log space follows and a straight line in
    linear space does not.

    Below the smallest point the time is clamped: a smaller product is never
    slower, and nothing measured says how much faster it is. Above the largest
    point the time follows the last segment's slope rather than clamping:
    clamping would freeze both compute times while a host product's transfer
    kept growing, so past some size every square product would turn back to
    the CPU although the GPU's lead was widening where the measurements stopped.

    Parameters
    ----------
    grid : list of [int, float]
        Ascending ``[n, time_s]`` measurements of square products.
    flops : float
        The work ``2*m*k*n`` to estimate a time for.

    Returns
    -------
    float
        The interpolated, clamped, or extrapolated wall time in seconds.
    """
    side = (flops / 2.0) ** (1.0 / 3.0)
    if side <= grid[0][0] or len(grid) == 1:
        return grid[0][1]
    i = 1
    while i < len(grid) - 1 and side > grid[i][0]:
        i += 1
    (n0, t0), (n1, t1) = grid[i - 1], grid[i]
    return t0 * (side / n0) ** (math.log(t1 / t0) / math.log(n1 / n0))


def _transfer_time(transfer, m, k, n) -> float:
    """Estimate the transfers a host float32 GEMM routed to the GPU pays, in seconds.

    The two operands go up (``4*(m*k + k*n)`` bytes at the H2D bandwidth) and
    the product comes back (``4*m*n`` bytes at the D2H bandwidth). Only float32
    reaches the GPU from the host, so the width is fixed. Device-resident inputs
    pay none of this, which is why their crossover is at less work.

    Parameters
    ----------
    transfer : dict
        ``{"h2d_gbps", "d2h_gbps", ...}`` measured bandwidths in GB/s.
    m, k, n : int
        GEMM dimensions (M x K times K x N).

    Returns
    -------
    float
        The upload plus download time in seconds.
    """
    # Floor the bandwidth before dividing: _load_cache rejects a cache whose
    # bandwidths are <= 0, but a degenerate-but-positive reading (a fast device
    # against a coarse timer) could still round low, and this function must never
    # raise ZeroDivisionError on the dispatch hot path. A 1e-12 GB/s floor makes a
    # near-zero bandwidth read as effectively infinite transfer time (the GPU
    # loses), which is the conservative direction.
    up = 4.0 * (m * k + k * n) / (max(transfer["h2d_gbps"], 1e-12) * 1e9)
    down = 4.0 * m * n / (max(transfer["d2h_gbps"], 1e-12) * 1e9)
    return up + down


def _staging_floor(calibration) -> float:
    """Fit the fixed cost of the staged GPU path from its measured series.

    The model prices a staged square product at ``gpu_t + transfer + t_fixed``.
    ``t_fixed`` is the value that minimizes the summed squared relative error of
    that model against the staged series, ``sum(((model_i - staged_i) /
    staged_i)**2)``, whose closed form is the mean of the residuals
    ``staged_i - gpu_i - transfer_i`` weighted by ``1/staged_i**2``. A relative
    fit counts every size alike; an absolute one would be settled by the
    largest sizes, where the residual is millisecond-scale error in the
    transfer model rather than a fixed cost, while the fixed cost only decides
    anything at the small sizes. A fixed cost cannot be negative, so the fit is
    floored at zero, which also keeps a host product's GPU cost at or above the
    compute-only cost a device-resident product is judged by.

    Parameters
    ----------
    calibration : dict
        A calibration dict carrying ``gpu``, ``staged``, and ``transfer``.

    Returns
    -------
    float
        The fitted fixed cost in seconds.
    """
    gpu = calibration["gpu"]["float32"]
    # calibrate() times both series at the same sizes, so the lookup is exact
    # and the interpolation only serves a cache written some other way.
    gpu_at = dict(gpu)
    # A square product moves n**2 times the bytes of a 1x1x1 one each way.
    unit_transfer = _transfer_time(calibration["transfer"], 1, 1, 1)
    num = den = 0.0
    for n, staged in calibration["staged"]["float32"]:
        g = gpu_at[n] if n in gpu_at else _interp_time(gpu, 2.0 * n**3)
        w = 1.0 / staged**2
        num += w * (staged - g - unit_transfer * n * n)
        den += w
    return max(num / den, 0.0)


def choose_backend(dtype, m, k, n, *, residency, calibration, backend) -> str:
    """Choose the backend for one GEMM: ``"cpu"`` or ``"cuda"``.

    A pure decision function -- inputs in, a backend enum out, no side effects.
    It implements the dispatch rule order (a lower rule never overrides a higher
    one):

    1. a forced ``backend`` of ``"cpu"`` or ``"cuda"`` routes directly (rule 5,
       within rules 1-2): the wrapper, not the policy, validates that a requested
       CUDA device actually exists and raises if not -- here ``"cuda"`` only
       routes, even for float64;
    2. ``dtype == "float64"`` -> ``"cpu"`` always (rule 2, the FP64 trap: this
       hardware's float64 throughput is a small fraction of its float32, so a
       float64 GEMM never auto-routes to the GPU at any size). This is hardcoded,
       never measured;
    3. no calibration, or a calibration with no ``"gpu"`` series (a CPU-only
       build's cache) -> ``"cpu"`` (rule 4, the conservative static fallback:
       prefer the CPU when there is nothing measured to justify the GPU);
    4. otherwise the measured crossover (rule 4) at the product's work
       ``F = 2*m*k*n``: the CPU and GPU compute times are interpolated at ``F``
       from the square calibration grids. A device-resident input compares
       compute alone. A host input adds the upload of ``4*(m*k + k*n)`` bytes,
       the download of ``4*m*n`` bytes, and the staged path's fixed cost fitted
       from the staged series, so it crosses to the GPU at more work than a
       device-resident input does, and a product with one small dimension,
       whose bytes are large for its work, stays on the CPU.

    Parameters
    ----------
    dtype : str
        ``"float32"`` or ``"float64"``.
    m, k, n : int
        GEMM dimensions (M x K times K x N).
    residency : str
        ``"host"`` (operands in host memory, transfer cost applies) or
        ``"device"`` (already on the device, no transfer cost).
    calibration : dict or None
        A calibration dict from :func:`_load_cache`, or ``None`` for the static
        fallback. The ``"gpu"``, ``"staged"``, and ``"transfer"`` keys are
        absent on a CPU-only build.
    backend : str
        ``"auto"`` (run the policy), ``"cpu"`` or ``"cuda"`` (forced).

    Returns
    -------
    str
        ``"cpu"`` or ``"cuda"``.

    Examples
    --------
    With no calibration the policy is conservative, and float64 never auto-routes
    to the GPU -- both decidable without a device:

    >>> from peregrine import policy
    >>> policy.choose_backend("float64", 4096, 4096, 4096,
    ...                       residency="host", calibration=None, backend="auto")
    'cpu'
    >>> policy.choose_backend("float32", 256, 256, 256,
    ...                       residency="host", calibration=None, backend="auto")
    'cpu'
    """
    # Rule 5 first, but only the forced backends (they win within rules 1-2). A
    # forced "cuda" routes even for float64 here: the wrapper validates device
    # availability and raises the requested-but-unavailable error, so the policy
    # must not silently rewrite the user's explicit choice into "cpu".
    if backend == "cpu":
        return "cpu"
    if backend == "cuda":
        return "cuda"

    # Rule 2 (hard exclusion): float64 GEMM never auto-routes to the GPU on this
    # hardware. Decided before any table lookup so no measured crossover can
    # override it -- the FP64 trap is upheld here exactly as the wrapper upholds
    # it upstream for device-resident operands.
    if dtype == "float64":
        return "cpu"

    # Rule 4, fallback half: no cache, or a CPU-only cache (no "gpu" series) ->
    # prefer the CPU. A conservative default is correct when nothing measured
    # justifies paying the GPU's launch + transfer cost.
    if calibration is None or not calibration.get("gpu"):
        return "cpu"

    # Rule 4, measured half. The work is formed in double in the same order the
    # native entry forms it, so both sides compare the same number to the bound.
    flops = 2.0 * m * k * n
    cpu_t = _interp_time(calibration["cpu"][dtype], flops)
    gpu_t = _interp_time(calibration["gpu"][dtype], flops)

    # Rule 3 (residency): device-resident inputs pay no transfer cost, so the
    # comparison is compute-vs-compute and the GPU wins at far less work.
    if residency == "device":
        return "cuda" if gpu_t < cpu_t else "cpu"

    # Host inputs pay their bytes both ways plus the staged path's fixed cost.
    # That fixed cost is why a small host matrix (e.g. n=256) stays on the CPU
    # even though the GPU's compute alone is faster, and the bytes are why a
    # matrix-vector product stays there at any size.
    host_t = (
        gpu_t
        + _transfer_time(calibration["transfer"], m, k, n)
        + _staging_floor(calibration)
    )
    return "cuda" if host_t < cpu_t else "cpu"


# The bound's walk runs over a log grid in F with this ratio between points
# (the equivalent cube side moves about 0.7 percent per point), from the
# smallest product up to the work of a cube of side 2**20. A product past that
# needs at least 12*(2**20)**2 bytes, 13 TB, of operands (the AM-GM bound
# below), so none exists.
_BOUND_STEP = 2.0 ** (1.0 / 32.0)
_BOUND_TOP = 2.0 * 2.0**60


def f32_cpu_flops_bound(calibration) -> float:
    """Return a work bound below which every host float32 GEMM stays on the CPU.

    The native matmul entry computes a host float32 product on the CPU, without
    asking :func:`choose_backend`, when its ``2*m*k*n`` is at most this bound.
    The bound must therefore never admit a shape that :func:`choose_backend`
    would send to the GPU, whatever its proportions. For work ``F`` no shape's
    host GPU cost is below ``gpu_t(F) + t_fixed + 12*(F/2)**(2/3) / bw``, where
    ``bw`` is the faster of the two bandwidths: by the AM-GM inequality
    ``4*(m*k + k*n + m*n) >= 12*(m*k*n)**(2/3)``, with equality for a cube, and
    no byte moves faster than the faster direction. The bound is the largest
    ``F`` such that ``cpu_t(F') <=`` that cost for every ``F' <= F``.

    It is found on a fine log grid in ``F`` that includes every calibration
    size, so both interpolated times are monotone between neighbouring points
    and their extremes over each interval sit at its ends. An interval passes
    only if the CPU's slowest end is no slower than the GPU's fastest end plus
    the cost terms at the interval's start, which proves the inequality across
    the whole interval, not only at the points. The walk stops at the first
    interval that fails, which keeps it to a few hundred points for a
    calibration with a crossover.

    Parameters
    ----------
    calibration : dict or None
        A calibration dict from :func:`_load_cache`, or ``None``.

    Returns
    -------
    float
        The bound in flops: ``0.0`` when even the smallest product may go to
        the GPU, and ``inf`` when no calibration or no ``"gpu"`` series exists
        or no product goes to the GPU.

    Examples
    --------
    >>> from peregrine import policy
    >>> policy.f32_cpu_flops_bound(None)
    inf
    """
    if calibration is None or not calibration.get("gpu"):
        return math.inf
    cpu = calibration["cpu"]["float32"]
    gpu = calibration["gpu"]["float32"]
    fixed = _staging_floor(calibration)
    transfer = calibration["transfer"]
    bandwidth = max(transfer["h2d_gbps"], transfer["d2h_gbps"], 1e-12) * 1e9
    steps = math.ceil(math.log(_BOUND_TOP / 2.0) / math.log(_BOUND_STEP))
    points = sorted(
        {2.0 * _BOUND_STEP**i for i in range(steps + 1)}
        | {2.0 * n**3 for n, _ in cpu + gpu if 2.0 * n**3 < _BOUND_TOP}
    )
    bound = 0.0
    lo = points[0]
    cpu_lo, gpu_lo = _interp_time(cpu, lo), _interp_time(gpu, lo)
    for hi in points[1:]:
        cpu_hi, gpu_hi = _interp_time(cpu, hi), _interp_time(gpu, hi)
        floor = min(gpu_lo, gpu_hi) + fixed + 12.0 * (lo / 2.0) ** (2.0 / 3.0) / bandwidth
        if max(cpu_lo, cpu_hi) > floor:
            return bound
        bound, lo, cpu_lo, gpu_lo = hi, hi, cpu_hi, gpu_hi
    return math.inf
