"""Per-machine calibration: measure the CPU/GPU/transfer grid and write the cache.

This module is the measurement half of the dispatch split (``policy.py`` is the
read + decide half). It owns three things and nothing else:

* the measurement orchestration: time a square GEMM at each grid size for both
  float dtypes on the CPU, and -- only when a usable device is present -- the
  GPU compute time per size and the H2D/D2H transfer bandwidth;
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

Budget: calibration uses LIGHTER reps than the publishable benchmark (warmup 2,
a handful of reps that scale down as n grows). ``budget_seconds`` bounds the
grid: the elapsed wall time is checked BEFORE each size, and once it has been
spent no larger size is started, so a slow machine returns a smaller-but-valid
grid. The check is between sizes, not projected, so the one size already in
flight when the budget is reached runs to completion -- the budget can be
overrun by a single largest-size measurement, never by more. The whole CPU-only
pass completes in well under the default 120s.

GPU symbols (``_cuda_time_matmul``, ``to_device``, ``from_device``, ``Array``)
exist ONLY on a CUDA build; on a CPU-only build they are absent. Every GPU
measurement is therefore gated behind ``fme.has_cuda()``, and the cache a
CPU-only build writes omits the ``gpu`` and ``transfer`` keys entirely.

Import budget: this module runs NOTHING at import -- no measurement, no device
init, no thread spin-up. ``calibrate()`` runs only when called, so importing
fastmathext (which does not import this module) pays nothing and the package
import stays under its 50ms budget.

Security: the cache is written with ``json.dump`` (never ``pickle``), and the
driver version is read from ``nvidia-smi`` via a list-form argv with
``shell=False`` and a fixed timeout, parsed by string partition and never
eval'd; a missing or failing ``nvidia-smi`` degrades to ``("none", "n/a")``.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import statistics
import subprocess
import time

import numpy as np

import fastmathext as fme

from .policy import SCHEMA, _cache_path

# The square sizes the grid sweeps, ascending. A log-ish span from small (where
# the per-call overhead floor dominates and the CPU wins) to large (where the GPU
# pulls ahead) is what lets the policy interpolate a faithful crossover. The
# largest sizes are the expensive ones; the budget ceiling drops them first on a
# slow machine (a smaller-but-valid grid beats an overrun). Verified live: the
# whole both-dtype CPU sweep over this grid is ~12-22s on this machine, well
# under the 120s default.
_GRID = (128, 256, 512, 1024, 2048, 4096)

# Lighter than the publishable bench (warmup 5 / reps 30): calibration needs one
# representative time per grid point, not a quotable distribution, and the full
# protocol reps would put a single n=4096 f32 point at tens of seconds. reps
# scale DOWN as n grows -- the signal is strong and the per-call cost is high at
# large n, so fewer reps there keep the budget while staying representative.
_WARMUP = 2
_REPS_SMALL = 5  # n <= 1024
_REPS_LARGE = 3  # n > 1024
_LARGE_N = 1024

# GPU event-timer reps. _cuda_time_matmul times the GEMM only (no transfer) via
# cudaEvent pairs after a sync, so a handful of warm reps is a stable number; the
# device path is exercised only on a fresh ON build.
_GPU_WARMUP = 2
_GPU_REPS = 5


def _reps_for(n: int) -> int:
    """Return the measured-rep count for size ``n`` (fewer at large n)."""
    return _REPS_LARGE if n > _LARGE_N else _REPS_SMALL


def _median_time(fn, reps: int, warmup: int) -> float:
    """Return the median wall time of ``fn`` over ``reps`` runs, in seconds.

    The bench timing core thinned to the median (calibration needs one
    representative time per point, not the p25/p75/CV the publishable bench
    reports). Warmup runs are untimed: they pay the first-touch page faults and
    let the clocks settle so the measured reps reflect steady-state work.
    ``perf_counter_ns`` is the monotonic high-resolution clock the bench protocol
    names for synchronous CPU work.

    The callable MUST build its inputs OUTSIDE the timed region -- this only
    times ``fn()`` itself -- so input construction never bills the measurement.

    Parameters
    ----------
    fn : callable
        A zero-argument callable performing the operation to time.
    reps : int
        Number of measured (timed) runs; the median is taken over these.
    warmup : int
        Number of untimed warmup runs before timing begins.

    Returns
    -------
    float
        The median elapsed time across the measured runs, in seconds.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e9)
    return statistics.median(times)


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


def _gpu_driver() -> tuple[str, str]:
    """Return ``(gpu_name, driver_version)`` via nvidia-smi, best-effort.

    The GPU name from ``_cuda_device_info`` and the driver version are two
    different sources: the device-info dict carries the name and compute
    capability but NOT the driver version, so the driver comes from
    ``nvidia-smi`` -- the project's established driver source -- queried with a
    list-form argv and ``shell=False`` so no string ever reaches a shell, with a
    fixed timeout, the output parsed by ``str.partition`` and never eval'd. A
    missing ``nvidia-smi``, a non-zero exit, empty output, or any subprocess
    error all degrade to ``("none", "n/a")`` so the signature assembly never
    raises on a machine without the tool.

    This is called only behind ``fme.has_cuda()``; on a CPU-only build the
    caller substitutes ``("none", "n/a")`` directly and never invokes this.

    Returns
    -------
    tuple of (str, str)
        ``(gpu_name, driver_version)``, each degrading to ``"none"`` / ``"n/a"``.
    """
    import shutil

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return "none", "n/a"
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "none", "n/a"
    if out.returncode != 0 or not out.stdout.strip():
        return "none", "n/a"
    first = out.stdout.strip().splitlines()[0]
    name, _, driver = first.partition(",")
    return name.strip() or "none", driver.strip() or "n/a"


def _machine_signature() -> str:
    """Assemble the machine/build signature string that keys the cache.

    The signature is ``"{cpu}|{gpu}|{driver}|{version}"`` -- a human-debuggable
    raw string, not a hash, so a cache file can be read and understood. The GPU
    and driver components are gated on ``fme.has_cuda()``: a CPU-only build (or a
    machine with no usable device) yields ``gpu="none"`` / ``driver="n/a"``,
    while a GPU run on the SAME CPU reads the real device name and driver. The two
    therefore DIFFER by construction, so a CPU-only cache is never mistaken for a
    GPU cache and a CPU-only machine never inherits GPU crossovers it could not
    have measured.

    Returns
    -------
    str
        The pipe-joined ``cpu|gpu|driver|version`` signature.
    """
    cpu = _cpu_brand()
    if fme.has_cuda():
        info = fme._core._cuda_device_info()
        gpu = info.get("name") or "none"
        _, driver = _gpu_driver()
    else:
        gpu = "none"
        driver = "n/a"
    return f"{cpu}|{gpu}|{driver}|{fme.__version__}"


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


def _measure_cpu(budget_seconds: float, start: float) -> tuple[dict, list]:
    """Measure the CPU per-(dtype, n) wall-time grid under the time budget.

    Both float dtypes are measured at each grid size until the budget is spent.
    The elapsed wall time is checked BEFORE each size: once it has reached
    ``budget_seconds`` the grid stops growing, so a slow machine returns a
    smaller-but-valid grid. The check is between sizes, not projected, so the size
    already chosen when the budget is reached runs to completion -- the elapsed
    time can therefore exceed ``budget_seconds`` by at most one largest-size
    measurement. At least the smallest size is always measured so the grid is
    never empty.

    Parameters
    ----------
    budget_seconds : float
        The wall-time budget for the whole calibration. Checked between sizes, so
        the one size in flight when it is reached still completes (overrun by at
        most a single largest-size measurement).
    start : float
        The ``perf_counter`` value captured at the start of calibration.

    Returns
    -------
    tuple of (dict, list)
        The ``{"float32": [[n, t], ...], "float64": [...]}`` CPU grid, and the
        ascending list of sizes actually measured (the GPU pass reuses it so the
        two grids stay aligned under the budget).
    """
    cpu = {"float32": [], "float64": []}
    measured_sizes = []
    for n in _GRID:
        # Stop adding larger sizes once the budget is spent. The smallest size is
        # always taken (measured_sizes empty) so a valid grid is guaranteed even
        # on a machine slower than the budget allows.
        if measured_sizes and (time.perf_counter() - start) >= budget_seconds:
            break
        reps = _reps_for(n)
        for dtype_name, dtype in (("float32", np.float32), ("float64", np.float64)):
            a = _seeded_square(n, dtype)
            b = _seeded_square(n, dtype)
            t = _median_time(lambda: fme.matmul(a, b), reps, _WARMUP)
            cpu[dtype_name].append([n, t])
        measured_sizes.append(n)
    return cpu, measured_sizes


def _measure_gpu(sizes: list) -> dict:
    """Measure the GPU float32 per-n compute-time grid (device-resident).

    Caller-gated on ``fme.has_cuda()``. The operands are staged to the device
    ONCE per size (outside the timed region), then ``_cuda_time_matmul`` returns
    the warm GEMM time in milliseconds via cudaEvent pairs after a sync -- the
    only honest GPU timer, with no H2D/D2H inside the timed region. Only float32
    is measured: a float64 GEMM never auto-routes to this GPU (the FP64 trap), so
    the policy never consults a GPU float64 grid and measuring one would be dead
    data.

    Parameters
    ----------
    sizes : list of int
        The sizes to measure, the same grid the CPU pass actually covered.

    Returns
    -------
    dict
        ``{"float32": [[n, time_s], ...]}`` of device GEMM times in seconds.
    """
    series = []
    for n in sizes:
        a = _seeded_square(n, np.float32)
        b = _seeded_square(n, np.float32)
        xa = fme.to_device(a)
        xb = fme.to_device(b)
        warm_ms = fme._core._cuda_time_matmul(xa, xb, _GPU_REPS, _GPU_WARMUP)
        series.append([n, warm_ms / 1e3])
    return {"float32": series}


def _measure_transfer(sizes: list) -> dict:
    """Measure H2D / D2H bandwidth and the fixed staging floor, in seconds.

    Caller-gated on ``fme.has_cuda()``. ``to_device`` (H2D) and ``from_device``
    (D2H) both synchronize at the boundary, so a ``perf_counter`` round trip
    around them is an honest transfer time. Bandwidth is derived from the largest
    measured size (where the fixed per-call overhead is the smallest fraction of
    the time, giving the cleanest GB/s), and the fixed staging floor is recovered
    from the smallest size as the part of its round-trip time NOT explained by
    the measured bandwidth -- exactly the launch + crossing + pageable-copy floor
    that keeps a small host GEMM on the CPU. Both are clamped non-negative so a
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
        return _median_time(lambda: fme.to_device(arr), _GPU_REPS, _GPU_WARMUP)

    def _d2h(dev) -> float:
        return _median_time(lambda: fme.from_device(dev), _GPU_REPS, _GPU_WARMUP)

    big = sizes[-1]
    a_big = _seeded_square(big, np.float32)
    dev_big = fme.to_device(a_big)
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
    dev_small = fme.to_device(a_small)
    small_bytes = small * small * 4
    # The smallest size's round trip minus the part the measured bandwidth
    # explains is the fixed floor: the launch + Python<->native<->CUDA crossings
    # + pageable-copy setup that is constant regardless of size.
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

    Times a square GEMM at each size on a log grid for both float dtypes on the
    CPU and -- only when a usable CUDA device is present -- the GPU compute time
    per size and the H2D/D2H transfer bandwidth. Assembles the machine signature,
    writes the result atomically as a schema-versioned JSON under the cache path
    (``FME_CACHE_DIR`` if set, else the per-user cache dir), and returns the same
    dict.

    On a CPU-only build the returned dict (and the cache) contain only the
    ``cpu`` grid: the ``gpu`` and ``transfer`` keys are absent, which the policy
    reads as "no GPU available." Because the signature also records ``gpu=none``
    in that case, a CPU-only cache is never mistaken for a GPU one.

    Calibration uses lighter reps than the publishable benchmark. The grid stops
    adding larger sizes once ``budget_seconds`` of wall time has elapsed, checked
    between sizes -- so on a slow machine the call returns close to the budget,
    overrunning by at most the one largest-size measurement already in flight when
    the budget is reached. The whole CPU-only pass completes in well under the
    120s default.

    Parameters
    ----------
    force : bool, optional
        When False (default) and a valid cache for this machine already exists,
        return it without re-measuring (the second-run-uses-cache contract). When
        True, ignore any existing cache and re-measure unconditionally.
    budget_seconds : float, optional
        Wall-time budget for the measurement, default 120. Checked between sizes:
        the grid stops growing once this much time has elapsed, so the one size in
        flight at that point still completes (overrun by at most a single
        largest-size measurement).

    Returns
    -------
    dict
        The calibration dict: ``{"schema", "signature", "cpu", "calibrated_at"}``
        always, plus ``"gpu"`` and ``"transfer"`` when a CUDA device is present.
        Also written to the cache file.

    Notes
    -----
    This measures; it never decides. The backend choice lives in
    :func:`fastmathext.policy.choose_backend`, a pure function fed this dict.

    Examples
    --------
    A CPU-only calibration with a tiny budget, isolated to a temp cache dir so the
    real per-user cache is untouched, returns a dict carrying the ``cpu`` grid:

    >>> import os, tempfile
    >>> from fastmathext.calibrate import calibrate
    >>> with tempfile.TemporaryDirectory() as d:
    ...     os.environ["FME_CACHE_DIR"] = d
    ...     try:
    ...         result = calibrate(force=True, budget_seconds=1)
    ...     finally:
    ...         del os.environ["FME_CACHE_DIR"]
    >>> isinstance(result, dict)
    True
    >>> "cpu" in result
    True
    """
    path = _cache_path()
    signature = _machine_signature()

    if not force:
        # Lazy import to keep this module free of a hard policy-read dependency at
        # the top and to honor the single-source cache-read path: a valid cache
        # for THIS machine short-circuits the whole measurement.
        from .policy import _load_cache

        cached = _load_cache(path, signature)
        if cached is not None:
            return cached

    start = time.perf_counter()
    cpu, measured_sizes = _measure_cpu(budget_seconds, start)

    payload = {
        "schema": SCHEMA,
        "signature": signature,
        "cpu": cpu,
        "calibrated_at": datetime.datetime.now().isoformat(),
    }

    # GPU + transfer are measured ONLY on a build with a usable device. On a
    # CPU-only build the symbols do not exist and has_cuda() is False, so the keys
    # stay absent and the policy treats the GPU as unavailable.
    if fme.has_cuda():
        payload["gpu"] = _measure_gpu(measured_sizes)
        payload["transfer"] = _measure_transfer(measured_sizes)

    _write_cache_atomic(path, payload)
    return payload
