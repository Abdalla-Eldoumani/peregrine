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
threshold constant): the crossover compares per-(dtype, n) MEASURED wall times,
NOT one asymptotic GFLOP/s figure. A single-constant model -- cpu_time =
work/CPU_GFLOPS vs gpu_time = work/GPU_GFLOPS + transfer using the warm large-n
throughput -- routes EVERYTHING, including a 256x256 host GEMM, to the GPU,
because at small n the GPU never reaches that warm throughput: a fixed launch +
cuBLAS setup + host-staging floor dominates. Storing the measured per-n times and
interpolating between them encodes that floor by construction, so a small host
matrix correctly stays on the CPU while a large one crosses to the GPU.

Security: the cache is read with ``json.load`` ONLY, never ``pickle`` -- pickle
deserialization executes arbitrary code, JSON cannot. After loading, the schema
integer and the signature are validated before any field is trusted, so a
tampered or foreign cache yields at worst wrong routing (a performance bug),
never code execution.

Deserialization invariant: the cache read is lazy (first dispatch decision,
never at import) and the platformdirs import is deferred into ``_cache_path``, so
importing fastmathext pays nothing for this module and the <50ms import budget
holds.
"""

import json
import os

# Bump on ANY change to the calibration-dict fields; the loader rejects unknown
# schemas (a cache written by an older or newer build is discarded and the
# machine recalibrates). This is the single source of truth: calibrate.py
# imports SCHEMA from here and stamps it into every cache it writes, so the
# writer and the reader can never drift apart.
SCHEMA = 1

# The two manual-override backends route directly within rules 1-2 (correctness
# and the hard exclusions) without consulting the table; "auto" runs the
# measured policy. Exposed for the wrapper to validate FME_BACKEND / set_backend
# against the design-doc vocabulary.
BACKENDS = ("auto", "cpu", "cuda")


def _cache_path() -> str:
    """Resolve the calibration cache file path.

    ``FME_CACHE_DIR`` overrides the location (the test-isolation hook: tests
    point it at a tmp dir so the real per-user cache is never touched).
    Otherwise the platform user cache dir is used with ``appauthor=False`` for
    the clean single-level ``.../fastmathext/calibration.json`` path -- the
    default double-nests the app name as both author and name. platformdirs is
    imported here, not at module top, so the import cost is paid on first
    dispatch, never at ``import fastmathext``.

    Returns
    -------
    str
        Absolute path to ``calibration.json`` under the resolved cache dir.
    """
    override = os.environ.get("FME_CACHE_DIR")
    if override:
        base = override
    else:
        import platformdirs

        base = platformdirs.user_cache_dir("fastmathext", appauthor=False)
    return os.path.join(base, "calibration.json")


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
       grid point that is not an ``[n, time]`` pair, or a ``gpu`` series present
       without a well-formed ``transfer`` triple -> ``None``.

    The schema integer promises the fields exist and are well-formed, but a
    truncated file, a non-atomic external writer, or a foreign tool can stamp the
    matching schema and signature onto a broken body. Step 4 is the reader-side
    defense: a structurally malformed body collapses to ``None`` like every other
    rejection so a partial dict never reaches :func:`choose_backend`, where
    ``grid[0][0]`` / ``calibration["cpu"][dtype]`` / ``calibration["transfer"]``
    would otherwise raise ``IndexError`` / ``KeyError`` on the dispatch hot path.

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
    # required, and if a "gpu" series is present then the "transfer" triple it
    # depends on must be present and complete (a positive bandwidth pair so the
    # _transfer_time division below cannot hit a zero). Any defect collapses to
    # None -- recalibrate or static fallback -- exactly like a parse/schema/
    # signature failure, so a malformed body never reaches choose_backend.
    cpu = data.get("cpu")
    if not isinstance(cpu, dict):
        return None
    for dt in ("float32", "float64"):
        grid = cpu.get(dt)
        if not isinstance(grid, list) or not grid:
            return None
        if not all(isinstance(p, list) and len(p) == 2 for p in grid):
            return None
    gpu = data.get("gpu")
    if gpu is not None:
        if not (isinstance(gpu, dict) and gpu.get("float32")):
            return None
        transfer = data.get("transfer")
        if not isinstance(transfer, dict) or not all(
            k in transfer for k in ("h2d_gbps", "d2h_gbps", "fixed_overhead_s")
        ):
            return None
        if not (transfer["h2d_gbps"] > 0 and transfer["d2h_gbps"] > 0):
            return None
    return data


def _interp_time(grid, n) -> float:
    """Linear-interpolate a measured wall time at ``n`` from a ``[n, time_s]`` grid.

    ``grid`` is the ascending-by-n list of measured ``[n, time_s]`` points for
    one (dtype, residency-independent compute) series. Below the smallest grid n
    the time is clamped to the smallest point and above the largest to the
    largest point: extrapolating a cubic GEMM cost off the measured range would
    be less trustworthy than clamping, and the calibration grid spans the sizes
    that matter. Between grid points the time is linearly interpolated.

    Parameters
    ----------
    grid : list of [int, float]
        Ascending ``[n, time_s]`` measurements.
    n : int
        The size to estimate a time for.

    Returns
    -------
    float
        The interpolated (or clamped) wall time in seconds.
    """
    if n <= grid[0][0]:
        return grid[0][1]
    if n >= grid[-1][0]:
        return grid[-1][1]
    for i in range(1, len(grid)):
        n1, t1 = grid[i]
        if n <= n1:
            n0, t0 = grid[i - 1]
            # linear between the bracketing measured points
            frac = (n - n0) / (n1 - n0)
            return t0 + frac * (t1 - t0)
    return grid[-1][1]


def _transfer_time(transfer, dtype, n) -> float:
    """Estimate the host round-trip a host-array GPU GEMM pays, in seconds.

    A host GEMM routed to the GPU stages two operands up (H2D) and one result
    back (D2H), then pays a fixed staging overhead -- the three Python ->
    native -> CUDA crossings plus the kernel launch and the pageable-copy setup.
    That fixed floor is what keeps a small host matrix on the CPU even though the
    GPU's compute alone is faster. Device-resident inputs pay none of this, which
    is why their crossover is at a much smaller n.

    Parameters
    ----------
    transfer : dict
        ``{"h2d_gbps", "d2h_gbps", "fixed_overhead_s"}`` measured bandwidths and
        the fixed staging floor.
    dtype : str
        ``"float32"`` or ``"float64"`` -- sets the per-element byte width.
    n : int
        The square operand dimension.

    Returns
    -------
    float
        The H2D(2 operands) + D2H(1 result) round-trip time plus the fixed
        overhead, in seconds.
    """
    itemsize = 8 if dtype == "float64" else 4
    nbytes = n * n * itemsize
    # Floor the bandwidth before dividing: _load_cache rejects a cache whose
    # bandwidths are <= 0, but a degenerate-but-positive reading (a fast device
    # against a coarse timer) could still round low, and this function must never
    # raise ZeroDivisionError on the dispatch hot path. A 1e-12 GB/s floor makes a
    # near-zero bandwidth read as effectively infinite transfer time (the GPU
    # loses), which is the conservative direction.
    h2d = 2 * nbytes / (max(transfer["h2d_gbps"], 1e-12) * 1e9)
    d2h = nbytes / (max(transfer["d2h_gbps"], 1e-12) * 1e9)
    return h2d + d2h + transfer["fixed_overhead_s"]


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
    4. otherwise the measured per-(dtype, n) crossover (rule 4): a device-
       resident input compares GPU compute vs CPU time with NO transfer cost; a
       host input adds the H2D+D2H round trip plus the fixed staging floor, so it
       crosses to the GPU at a larger n than a device-resident input does.

    Parameters
    ----------
    dtype : str
        ``"float32"`` or ``"float64"``.
    m, k, n : int
        GEMM dimensions (M x K times K x N); ``n`` keys the measured grids.
    residency : str
        ``"host"`` (operands in host memory, transfer cost applies) or
        ``"device"`` (already on the device, no transfer cost).
    calibration : dict or None
        A calibration dict from :func:`_load_cache`, or ``None`` for the static
        fallback. The ``"gpu"`` and ``"transfer"`` keys are absent on a CPU-only
        build.
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

    >>> from fastmathext import policy
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

    # Rule 4, measured half: compare per-n MEASURED wall times, not a single
    # GFLOP/s. cpu_t and gpu_compute_t are interpolated from the calibration
    # grids at this n.
    cpu_t = _interp_time(calibration["cpu"][dtype], n)
    gpu_compute_t = _interp_time(calibration["gpu"][dtype], n)

    # Rule 3 (residency): device-resident inputs pay no transfer cost, so the
    # comparison is compute-vs-compute and the GPU wins at a far smaller n.
    if residency == "device":
        return "cuda" if gpu_compute_t < cpu_t else "cpu"

    # Host inputs pay the H2D+D2H round trip plus the fixed staging floor. That
    # floor is exactly why a small host matrix (e.g. n=256) stays on the CPU even
    # though the GPU's compute alone is faster -- the single-GFLOP/s model misses
    # it and wrongly routes small host GEMMs to the GPU.
    xfer_t = _transfer_time(calibration["transfer"], dtype, n)
    return "cuda" if (gpu_compute_t + xfer_t) < cpu_t else "cpu"
