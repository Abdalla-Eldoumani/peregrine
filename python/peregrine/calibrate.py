"""Per-machine calibration: measure the CPU/GPU/transfer grid and write the cache.

This module is the measurement half of the dispatch split (``policy.py`` is the
read + decide half). It owns three things and nothing else:

* the measurement orchestration: time a square GEMM at each grid size for both
  float dtypes on the CPU, and -- only when a usable device is present -- the
  GPU compute time per size, the host-staged GPU time per size, and the H2D/D2H
  transfer bandwidth;
* the machine signature: CPU brand + GPU name + driver version + package
  version, assembled so a CPU-only run (gpu=``none``, driver=``n/a``) and a GPU
  run on the SAME CPU produce different strings, so a CPU-only cache can never be
  mistaken for a GPU cache and never claims GPU crossovers it did not measure;
* the atomic, schema-versioned cache write: a temp file in the cache dir, fsync,
  then ``os.replace`` -- a crash mid-write leaves the old cache or the new, never
  a half-written one.

``SCHEMA`` and ``_cache_path`` are imported from ``policy`` -- one source of
truth. The writer stamps the same ``SCHEMA`` the reader checks, so they cannot
drift, and both resolve the cache location through the same ``_cache_path``.

The load-bearing measurement fact (the reason the cache stores per-(dtype, n)
wall TIMES, not a single GFLOP/s constant): at small n the GPU never reaches its
warm large-n throughput -- a fixed launch + cuBLAS setup + host-staging floor
dominates -- so a single-constant model wrongly routes a small host GEMM to the
GPU. Storing the measured per-n times lets the policy interpolate the real
small-n floor instead of assuming the asymptote. See ``policy.choose_backend``.

What each series times, and why:

* ``cpu``: the native CPU kernel with the backend forced to ``"cpu"``. Timing
  ``pg.matmul`` through the auto router would bill the router's own per-call
  cost to the CPU, and on a machine with a cached GPU crossover it would time the
  GPU instead.
* ``gpu``: the device-resident GEMM alone, one cudaEvent pair per repetition.
* ``staged``: a host float32 pair through the same staged path auto mode takes
  (upload, device GEMM, download), timed on the wall clock because that path
  synchronizes before it returns.
* ``transfer``: the native upload and download entries, without the public
  wrappers' per-call checks.

Every point is the minimum over repetitions after warmup: interference from the
OS, the display, or a clock change only ever adds time, so the minimum is the
estimate of the work itself. The display GPU idles at low clocks between
bursts, so the GPU runs continuous GEMMs before the first measurement until two
consecutive batches agree, and briefly again before each size.

Budget: calibration uses LIGHTER reps than the publishable benchmark (warmup 2,
a handful of reps that scale down as n grows). ``budget_seconds`` bounds the
grid: the elapsed wall time is checked BEFORE each size, and once it has been
spent no further size is started, so a slow machine returns a smaller-but-valid
grid. The check is between sizes, not projected, so the one size already in
flight when the budget is reached runs to completion -- the budget can be
overrun by a single size measurement, never by more. The whole CPU-only pass
completes in well under the default 120s.

GPU symbols (``_cuda_time_matmul``, ``to_device``, ``from_device``, ``Array``)
exist ONLY on a CUDA build; on a CPU-only build they are absent. Every GPU
measurement is therefore gated behind ``pg.has_cuda()``, and the cache a
CPU-only build writes omits the ``gpu``, ``staged``, and ``transfer`` keys
entirely.

Import budget: this module runs NOTHING at import -- no measurement, no device
init, no thread spin-up, no NVML load. ``calibrate()`` runs only when called.

Security: the cache is written with ``json.dump`` (never ``pickle``). The
driver version comes from NVML, loaded from the driver's own library by
absolute path, so no subprocess runs and no search path can substitute a
different library; any NVML failure degrades to ``"n/a"``.
"""

from __future__ import annotations

import datetime
import functools
import json
import os
import platform
import time

import numpy as np

import peregrine as pg

from .policy import SCHEMA, _cache_path

# The coarse square sizes, ascending. The small end is where the per-call floor
# dominates and the CPU wins; the large end is where the GPU pulls ahead. The
# grid is refined afterwards wherever the CPU and the staged GPU path trade
# places, so the coarse grid only has to bracket each crossover.
_GRID = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

# Midpoints inserted into each bracket where cpu - staged changes sign, spaced
# geometrically because GEMM cost grows as a power of n.
_REFINE_POINTS = 3

# Lighter than the publishable bench (warmup 5 / reps 30): calibration needs one
# representative time per grid point, not a quotable distribution, and the full
# protocol reps would put a single n=4096 f32 point at tens of seconds. reps
# scale DOWN as n grows -- the signal is strong and the per-call cost is high at
# large n, so fewer reps there keep the budget while staying representative.
_WARMUP = 2
_REPS_SMALL = 5  # n <= 1024
_REPS_LARGE = 3  # n > 1024
_LARGE_N = 1024

# Device event-timed repetitions and the transfer warmups.
_GPU_WARMUP = 2
_GPU_REPS = 5

# (minimum, maximum) seconds of continuous GEMMs. The first warmup lifts the GPU
# out of its idle clock state; the short one before each size undoes the decay
# while the CPU side of that size was being measured.
_FIRST_WARM = (2.0, 4.0)
_REWARM = (0.2, 1.0)
# Two consecutive warmup batches within this fraction of each other mean the
# clocks have settled.
_WARM_AGREE = 0.05
# One warmup batch lasts about this long, so a batch is a stable sample.
_WARM_BATCH_S = 0.05


def _reps_for(n: int) -> int:
    """Return the measured-rep count for size ``n`` (fewer at large n)."""
    return _REPS_LARGE if n > _LARGE_N else _REPS_SMALL


def _min_time(fn, reps: int, warmup: int) -> float:
    """Return the minimum wall time of ``fn`` over ``reps`` runs, in seconds.

    Warmup runs are untimed: they pay the first-touch page faults and let the
    clocks settle so the measured reps reflect steady-state work.
    ``perf_counter_ns`` is the monotonic high-resolution clock the bench protocol
    names for synchronous work.

    The callable MUST build its inputs OUTSIDE the timed region -- this only
    times ``fn()`` itself -- so input construction never bills the measurement.

    Parameters
    ----------
    fn : callable
        A zero-argument callable performing the operation to time.
    reps : int
        Number of measured (timed) runs; the minimum is taken over these.
    warmup : int
        Number of untimed warmup runs before timing begins.

    Returns
    -------
    float
        The smallest elapsed time across the measured runs, in seconds.
    """
    for _ in range(warmup):
        fn()
    best = None
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        t = (time.perf_counter_ns() - t0) / 1e9
        best = t if best is None else min(best, t)
    return best


def _cpu_brand() -> str:
    """Return the human-readable CPU brand string for the machine signature.

    On Windows ``platform.processor()`` returns a coarse family/model string
    ("Intel64 Family 6 Model 165..."), useless as a machine key; the registry
    ``ProcessorNameString`` carries the real brand ("Intel(R) Core(TM) i7-10750H
    CPU @ 2.60GHz"). On Linux/WSL the ``model name`` line in ``/proc/cpuinfo`` is
    the equivalent. ``platform.processor()`` is the last-resort fallback when
    neither source is readable, so the signature is always SOME stable string.

    Returns
    -------
    str
        The CPU brand string, or ``"unknown"`` if no source is readable.
    """
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            try:
                brand, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            finally:
                winreg.CloseKey(key)
            if brand:
                return brand.strip()
        except OSError:
            pass
    else:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _load_nvml():
    """Load the NVML library that ships with the NVIDIA driver.

    The driver installer does not put NVML on the DLL search path, and NVIDIA
    documents loading it by absolute path: ``System32`` for DCH drivers, then
    the ``NVSMI`` directory for standard drivers. On Linux the driver package
    provides the versioned soname; the unversioned name usually ships only with
    development packages. Raises ``OSError`` when no candidate loads.
    """
    import ctypes

    if os.name != "nt":
        return ctypes.CDLL("libnvidia-ml.so.1")
    candidates = []
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidates.append(os.path.join(system_root, "System32", "nvml.dll"))
    program_files = os.environ.get("ProgramW6432")
    if program_files:
        candidates.append(
            os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvml.dll")
        )
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise OSError("nvml library not found")


def _nvml_driver_version() -> str:
    """Return the installed NVIDIA driver version, or ``"n/a"`` on any failure.

    The driver version keys the cache because a driver update can change GEMM
    and transfer speed. ``cudaDriverGetVersion`` cannot stand in for it: it
    reports the highest CUDA API version the driver supports (12080-style),
    which many driver releases share. NVML init is reference counted, so the
    shutdown in ``finally`` never tears down another user's session.

    Returns
    -------
    str
        The driver version string (for example ``"610.47"``), or ``"n/a"``.
    """
    import ctypes

    try:
        nvml = _load_nvml()
        if nvml.nvmlInit_v2() != 0:
            return "n/a"
        try:
            # NVML documents 80 bytes, terminator included, as the maximum.
            buf = ctypes.create_string_buffer(80)
            if nvml.nvmlSystemGetDriverVersion(buf, ctypes.c_uint(80)) != 0:
                return "n/a"
            return buf.value.decode("ascii").strip() or "n/a"
        finally:
            nvml.nvmlShutdown()
    except (OSError, AttributeError, ValueError):
        return "n/a"


@functools.cache
def _machine_signature() -> str:
    """Assemble the machine/build signature string that keys the cache.

    The signature is ``"{cpu}|{gpu}|{driver}|{version}"`` -- a human-debuggable
    raw string, not a hash, so a cache file can be read and understood. The GPU
    and driver components are gated on ``pg.has_cuda()``: a CPU-only build (or a
    machine with no usable device) yields ``gpu="none"`` / ``driver="n/a"``,
    while a GPU run on the SAME CPU reads the real device name and driver. The two
    therefore DIFFER by construction, so a CPU-only cache is never mistaken for a
    GPU cache and a CPU-only machine never inherits GPU crossovers it could not
    have measured.

    The GPU name comes from the device probe, which never builds the CUDA
    context. Memoized per process because the auto router needs the signature
    to find its cache and the registry read and NVML load would otherwise run on
    every dispatch; ``pg._reset_routing_state()`` clears it.

    Returns
    -------
    str
        The pipe-joined ``cpu|gpu|driver|version`` signature.
    """
    cpu = _cpu_brand()
    if pg.has_cuda():
        gpu = pg._cuda_device_probe()["name"] or "none"
        driver = _nvml_driver_version()
    else:
        gpu = "none"
        driver = "n/a"
    return f"{cpu}|{gpu}|{driver}|{pg.__version__}"


def _write_cache_atomic(path: str, payload: dict) -> None:
    """Write ``payload`` as JSON to ``path`` atomically and crash-safely.

    The cache must never be left half-written: a partial JSON would be a corrupt
    cache the loader has to reject (at best) or could misread (at worst). The
    write goes to a uniquely-named temp file in the SAME directory as the target
    (so ``os.replace`` is a same-volume rename, not a copy), is flushed and
    ``fsync``'d to durable storage, then atomically swapped in with
    ``os.replace`` -- which overwrites an existing target on Windows where
    ``os.rename`` would raise. On ANY failure the temp file is unlinked
    best-effort before the error propagates, so no stray ``.tmp`` is left behind.

    Parameters
    ----------
    path : str
        Destination cache file path.
    payload : dict
        The calibration dict to serialize as JSON.
    """
    import tempfile

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave a temp behind on a failed write -- a stray .tmp would
        # accumulate and could be mistaken for the cache. Best-effort unlink, then
        # re-raise the original error unchanged.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _seeded_square(n: int, dtype) -> np.ndarray:
    """Return an ``(n, n)`` standard-normal array of ``dtype``, fixed seed.

    Built OUTSIDE any timed region. The seed is fixed so the calibration is
    reproducible run to run; the values are immaterial to a timing measurement,
    only the shape and dtype set the work.
    """
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, n)).astype(dtype)


def _time_cpu(n: int, dtype) -> float:
    """Return the CPU GEMM time at size ``n`` through the native kernel."""
    a = _seeded_square(n, dtype)
    b = _seeded_square(n, dtype)
    return _min_time(lambda: pg._matmul_native(a, b), _reps_for(n), _WARMUP)


def _gpu_warm(xa, xb, min_seconds: float, max_seconds: float) -> None:
    """Run back-to-back device GEMMs until the GPU clocks settle.

    Stops once at least ``min_seconds`` have passed and two consecutive batches
    agree within ``_WARM_AGREE`` per GEMM, or at ``max_seconds`` regardless.
    """
    start = time.perf_counter()
    reps = 1
    prev = None
    while True:
        ms = pg._core._cuda_time_matmul(xa, xb, reps, 0)
        elapsed = time.perf_counter() - start
        settled = prev is not None and abs(ms - prev) <= _WARM_AGREE * prev
        if (settled and elapsed >= min_seconds) or elapsed >= max_seconds:
            return
        prev = ms
        reps = max(1, int(_WARM_BATCH_S / max(ms / 1e3, 1e-7)))


def _time_gpu(n: int, warm: tuple) -> tuple:
    """Return ``(device_s, staged_s)`` for a float32 GEMM at size ``n``.

    Only float32 is measured: a float64 GEMM never auto-routes to this GPU (the
    FP64 trap), so a GPU float64 series would be dead data.
    """
    a = _seeded_square(n, np.float32)
    b = _seeded_square(n, np.float32)
    xa = pg._to_device_native(a)
    xb = pg._to_device_native(b)
    _gpu_warm(xa, xb, *warm)
    # One event pair per call, so the minimum is taken per repetition.
    device_ms = min(
        pg._core._cuda_time_matmul(xa, xb, 1, 0) for _ in range(_GPU_REPS)
    )
    staged = _min_time(lambda: pg._matmul_gpu_staged(a, b), _reps_for(n), _WARMUP)
    return device_ms / 1e3, staged


def _refine_sizes(lo: int, hi: int) -> list:
    """Return the geometric midpoints strictly inside ``(lo, hi)``."""
    steps = _REFINE_POINTS + 1
    mids = {round(lo * (hi / lo) ** (i / steps)) for i in range(1, steps)}
    return sorted(mids - {lo, hi})


def _measure_grid(budget_seconds: float, start: float, cuda: bool) -> dict:
    """Measure every series on the coarse grid, then refine each crossover.

    The elapsed wall time is checked BEFORE each size: once it has reached
    ``budget_seconds`` no further size is started, so a slow machine returns a
    smaller-but-valid grid. The size already started when the budget is reached
    runs to completion, so the elapsed time can exceed ``budget_seconds`` by at
    most one size measurement. At least the smallest size is always measured so
    the grid is never empty.

    Parameters
    ----------
    budget_seconds : float
        The wall-time budget for the whole calibration.
    start : float
        The ``perf_counter`` value captured at the start of calibration.
    cuda : bool
        Whether to measure the GPU series too.

    Returns
    -------
    dict
        ``{n: (cpu_f32_s, cpu_f64_s, gpu_s, staged_s)}`` for every measured size;
        the GPU entries are None when ``cuda`` is False.
    """
    points = {}

    def out_of_budget() -> bool:
        return bool(points) and (time.perf_counter() - start) >= budget_seconds

    def measure(n: int, warm: tuple) -> None:
        cpu32 = _time_cpu(n, np.float32)
        cpu64 = _time_cpu(n, np.float64)
        gpu, staged = _time_gpu(n, warm) if cuda else (None, None)
        points[n] = (cpu32, cpu64, gpu, staged)

    for n in _GRID:
        if out_of_budget():
            break
        measure(n, _REWARM if points else _FIRST_WARM)

    if cuda:
        coarse = sorted(points)
        for lo, hi in zip(coarse, coarse[1:]):
            cpu_wins_lo = points[lo][0] < points[lo][3]
            cpu_wins_hi = points[hi][0] < points[hi][3]
            if cpu_wins_lo == cpu_wins_hi:
                continue
            for n in _refine_sizes(lo, hi):
                if out_of_budget():
                    break
                measure(n, _REWARM)
    return points


def _measure_transfer(sizes: list) -> dict:
    """Measure H2D / D2H bandwidth and the fixed staging floor, in seconds.

    Caller-gated on ``pg.has_cuda()``. The native upload and download entries
    both synchronize at the boundary, so a ``perf_counter`` round trip around
    them is an honest transfer time. Bandwidth is derived from the largest
    measured size (where the fixed per-call overhead is the smallest fraction of
    the time, giving the cleanest GB/s), and the fixed staging floor is recovered
    from the smallest size as the part of its round-trip time NOT explained by
    the measured bandwidth -- the allocation, crossing, and copy-setup floor that
    keeps a small host GEMM on the CPU. Both are clamped non-negative so a
    noisy tiny-size sample can never produce a nonsensical negative floor.

    Parameters
    ----------
    sizes : list of int
        The measured grid; the smallest and largest anchor the floor and the
        bandwidth respectively.

    Returns
    -------
    dict
        ``{"h2d_gbps", "d2h_gbps", "fixed_overhead_s"}``.
    """
    def _h2d(arr) -> float:
        return _min_time(lambda: pg._to_device_native(arr), _GPU_REPS, _GPU_WARMUP)

    def _d2h(dev) -> float:
        return _min_time(lambda: pg._from_device_native(dev), _GPU_REPS, _GPU_WARMUP)

    big = sizes[-1]
    a_big = _seeded_square(big, np.float32)
    dev_big = pg._to_device_native(a_big)
    big_bytes = big * big * 4
    h2d_big = _h2d(a_big)
    d2h_big = _d2h(dev_big)
    # Floor the measured time before dividing: a degenerate 0.0 reading (a coarse
    # timer against a fast transfer, or a stubbed transfer in a test) would raise
    # ZeroDivisionError here. The 1e-9s floor caps the derived bandwidth instead of
    # crashing; _load_cache and _transfer_time defend the read side symmetrically.
    h2d_gbps = big_bytes / max(h2d_big, 1e-9) / 1e9
    d2h_gbps = big_bytes / max(d2h_big, 1e-9) / 1e9

    small = sizes[0]
    a_small = _seeded_square(small, np.float32)
    dev_small = pg._to_device_native(a_small)
    small_bytes = small * small * 4
    # The smallest size's round trip minus the part the measured bandwidth
    # explains is the fixed floor that is constant regardless of size.
    round_trip_small = _h2d(a_small) + _d2h(dev_small)
    # Floor the bandwidths here too (matching policy._transfer_time): if a big-size
    # reading floored to a near-zero bandwidth above, dividing by it here would
    # raise. A near-zero bandwidth makes "explained" huge, so the clamped-non-
    # negative floor below reads as ~0 -- the conservative degenerate result.
    explained = small_bytes / (max(h2d_gbps, 1e-12) * 1e9) + \
        small_bytes / (max(d2h_gbps, 1e-12) * 1e9)
    fixed_overhead_s = max(round_trip_small - explained, 0.0)

    return {
        "h2d_gbps": h2d_gbps,
        "d2h_gbps": d2h_gbps,
        "fixed_overhead_s": fixed_overhead_s,
    }


def calibrate(force: bool = False, budget_seconds: float = 120) -> dict:
    """Measure this machine's GEMM performance and write the calibration cache.

    Times a square GEMM on a log grid of sizes for both float dtypes on the CPU
    and -- only when a usable CUDA device is present -- the device GEMM time, the
    host-staged GPU time (upload, GEMM, download), and the H2D/D2H transfer
    bandwidth. Wherever the CPU and the staged GPU path trade places between two
    grid sizes, three more sizes are measured inside that bracket so the
    crossover is located closely. Each point is the minimum over repetitions
    after warmup. Assembles the machine signature, writes the result atomically
    as a schema-versioned JSON under the cache directory (``PEREGRINE_CACHE_DIR``
    if set, else the per-user cache dir) in a file named for the signature, makes
    it the calibration the auto router uses for the rest of the session, and
    returns the same dict.

    On a CPU-only build the returned dict (and the cache) contain only the
    ``cpu`` grid: the ``gpu``, ``staged``, and ``transfer`` keys are absent,
    which the policy reads as "no GPU available." Because the signature also
    records ``gpu=none`` in that case, a CPU-only cache is never mistaken for a
    GPU one.

    Calibration uses lighter reps than the publishable benchmark. No further
    size is started once ``budget_seconds`` of wall time has elapsed, checked
    between sizes -- so on a slow machine the call returns close to the budget,
    overrunning by at most the one size measurement already in flight when the
    budget is reached. The whole CPU-only pass completes in well under the 120s
    default.

    Parameters
    ----------
    force : bool, optional
        When False (default) and a valid cache for this machine already exists,
        return it without re-measuring (the second-run-uses-cache contract). When
        True, ignore any existing cache and re-measure unconditionally.
    budget_seconds : float, optional
        Wall-time budget for the measurement, default 120. Checked between sizes:
        no further size starts once this much time has elapsed, so the one size in
        flight at that point still completes (overrun by at most a single size
        measurement).

    Returns
    -------
    dict
        The calibration dict: ``{"schema", "signature", "cpu", "calibrated_at"}``
        always, plus ``"gpu"``, ``"staged"``, and ``"transfer"`` when a CUDA
        device is present. Also written to the cache file.

    Notes
    -----
    This measures; it never decides. The backend choice lives in
    :func:`peregrine.policy.choose_backend`, a pure function fed this dict.

    Examples
    --------
    A calibration with a tiny budget, isolated to a temp cache dir so the real
    per-user cache is untouched, returns a dict carrying the ``cpu`` grid:

    >>> import os, tempfile
    >>> from peregrine.calibrate import calibrate
    >>> with tempfile.TemporaryDirectory() as d:
    ...     os.environ["PEREGRINE_CACHE_DIR"] = d
    ...     try:
    ...         result = calibrate(force=True, budget_seconds=1)
    ...     finally:
    ...         del os.environ["PEREGRINE_CACHE_DIR"]
    >>> isinstance(result, dict)
    True
    >>> "cpu" in result
    True
    """
    signature = _machine_signature()
    path = _cache_path(signature)

    if not force:
        # Lazy import to keep this module free of a hard policy-read dependency at
        # the top and to honor the single-source cache-read path: a valid cache
        # for THIS machine short-circuits the whole measurement.
        from .policy import _load_cache

        cached = _load_cache(path, signature)
        if cached is not None:
            pg._routing_calibration = cached
            return cached

    start = time.perf_counter()
    cuda = pg.has_cuda()
    # The CPU series must time the CPU even under a forced or cached GPU route.
    saved_backend = pg._backend
    pg._backend = "cpu"
    try:
        points = _measure_grid(budget_seconds, start, cuda)
        sizes = sorted(points)
        transfer = _measure_transfer(sizes) if cuda else None
    finally:
        pg._backend = saved_backend

    payload = {
        "schema": SCHEMA,
        "signature": signature,
        "cpu": {
            "float32": [[n, points[n][0]] for n in sizes],
            "float64": [[n, points[n][1]] for n in sizes],
        },
        "calibrated_at": datetime.datetime.now().isoformat(),
    }

    # GPU, staged, and transfer series exist ONLY on a build with a usable
    # device. On a CPU-only build the keys stay absent and the policy treats the
    # GPU as unavailable.
    if cuda:
        payload["gpu"] = {"float32": [[n, points[n][2]] for n in sizes]}
        payload["staged"] = {"float32": [[n, points[n][3]] for n in sizes]}
        payload["transfer"] = transfer

    _write_cache_atomic(path, payload)
    pg._routing_calibration = payload
    return payload
