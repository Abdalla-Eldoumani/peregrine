"""Apples-to-apples matmul benchmark and the shared measurement core.

Methodology, in order of importance:
1. Inputs are already-constructed contiguous ndarrays. No list conversion is
   timed for either side. The legacy harness timed np.dot on Python lists,
   which billed NumPy 13 to 122 ms of conversion per call at these sizes.
2. Thread counts are pinned and recorded. An unpinned comparison measures
   scheduler luck, not kernels.
3. Median of many reps after warmup, with the spread reported. Means hide
   bimodal thermal behavior on laptops.
4. Every benchmarked op is verified against NumPy in the same run. A fast
   wrong kernel scores zero.

The second half of this module is the measurement protocol every published
number comes from (interleaved ABBA sampling, shared batch size, per-sample
bitwise verification, the CV gate, clock discards, pinned threads, and the run
manifest). benchmarks/bench_suite.py drives it over the cell matrix; the quick
sequential `run` below stays as a smoke check.

Usage:
    python benchmarks/bench_matmul.py --sizes 256 512 1024 --dtype float64
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

import peregrine as pg

# OpenMP reads OMP_NUM_THREADS once, when its runtime initializes, so the value
# that counts is the one present when this process started, not whatever a
# later caller writes into os.environ.
_START_OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS")

# The tolerance contract lives in one place, tests/conftest.py. Routing the
# per-run verification through its helpers keeps every bench on the single
# sanctioned toleranced path; a bench that carries its own rtol can loosen it
# until a wrong kernel passes.
_TESTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from conftest import assert_matmul_close  # noqa: E402

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
    threadpool_info = None
    threadpool_limits = None


def _blas_identity() -> str:
    # threadpoolctl's blas entry is the one place that names the BLAS NumPy
    # actually loaded; scipy-openblas surfaces here as its internal_api plus
    # version, so no separate probe is needed.
    if threadpool_info is None:
        return "unknown"
    for d in threadpool_info():
        if d.get("user_api") == "blas":
            return f"{d.get('internal_api')} {d.get('version') or '?'}"
    return "unknown"


def _power_profile() -> str:
    # Windows power profile via powercfg, list-form argv and shell=False so no
    # string ever reaches a shell; the output is parsed, never eval'd. A missing
    # powercfg (non-Windows, or stripped PATH) degrades to "unknown" rather than
    # failing the whole run: the manifest records what it can prove.
    try:
        out = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _gpu_manifest() -> tuple[str, str]:
    # (gpu, driver) for the benchmark manifest, best-effort and never
    # raising. nvidia-smi ships with the driver and is queried for the device
    # name and driver version; a CPU-only build, no device, or no nvidia-smi all
    # degrade to ("none", "n/a") so the manifest records what it can prove rather
    # than failing the run. Records the device regardless of has_cuda(): the
    # manifest is provenance, so a GPU present but below the cc floor still
    # belongs in the record.
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


def _redact_threadpools(pools: list) -> list:
    # threadpoolctl reports each loaded pool's absolute on-disk filepath, which on
    # a dev machine is an installed-package path under the user's home directory
    # (the OS username and home structure). That is PII and must never be written
    # into a committed results JSON. Keep only the library basename: the version,
    # prefix, and architecture fields already document which BLAS ran, and the
    # absolute path adds nothing a reader needs. Each entry is copied so the live
    # threadpoolctl dict is left untouched.
    redacted = []
    for entry in pools:
        item = dict(entry)
        if "filepath" in item and item["filepath"]:
            item["filepath"] = os.path.basename(item["filepath"])
        redacted.append(item)
    return redacted


def _machine_manifest() -> dict:
    # No manifest, no merge: every results JSON carries the machine identity, the
    # BLAS the comparison ran against, the power profile (thermal state changes
    # throughput on a laptop), and a timestamp.
    gpu, driver = _gpu_manifest()
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "peregrine": pg.__version__,
        "cpu_features": pg.cpu_features(),
        # The BUILD flag, not the runtime has_cuda(): the manifest records what
        # the binary was compiled with (whether the CUDA path exists at all), and
        # the gpu/driver fields below record the device that was actually present.
        # A run on a CUDA build with the GPU busy elsewhere should still read
        # cuda_build True.
        "cuda_build": pg._has_cuda_build(),
        "gpu": gpu,
        "driver": driver,
        "blas": _blas_identity(),
        "power_profile": _power_profile(),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if threadpool_info is not None:
        info["threadpools"] = _redact_threadpools(threadpool_info())
    return info


def _measure_once(fn, reps: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e9)
    median = statistics.median(times)
    # Coefficient of variation, stdev over median, reported as a readout so a
    # noisy run is visible at the console; _bench's opt-in cv_gate wrapper owns
    # the rerun-after-cooldown machinery. stdev needs two samples and a
    # non-zero median.
    cv = (
        statistics.stdev(times) / median
        if reps >= 2 and median > 0
        else float("nan")
    )
    return {
        "median_s": median,
        "p25_s": statistics.quantiles(times, n=4)[0],
        "p75_s": statistics.quantiles(times, n=4)[2],
        "min_s": min(times),
        "cv": cv,
        "reps": reps,
    }


def _bench(
    fn,
    reps: int,
    warmup: int,
    *,
    cv_gate: bool = False,
    cv_threshold: float = 0.05,
    max_reruns: int = 2,
    cooldown_s: float = 30,
    cooldown_fn=time.sleep,
) -> dict:
    # The sequential timing layer the older scripts (bench_fused, bench_gpu,
    # scaling, sweep_blocking, calibrate) call positionally. The DEFAULT path
    # (cv_gate False) returns _measure_once unchanged, so they see a stable dict
    # shape (median_s, p25_s, p75_s, min_s, cv, reps).
    result = _measure_once(fn, reps, warmup)
    if not cv_gate:
        return result
    # A series whose CV exceeds the threshold is rerun after a cooldown, at
    # least 30 s between heavy series on this laptop; after a bounded number of
    # reruns a still-noisy series is recorded rather than dropped. cooldown_fn
    # is injectable (default time.sleep) so a unit test stubs it to a no-op and
    # never waits 30 real seconds.
    reruns = 0
    while result["cv"] > cv_threshold and reruns < max_reruns:
        cooldown_fn(cooldown_s)
        result = _measure_once(fn, reps, warmup)
        reruns += 1
    # A high_cv series is still verified: the noise flag documents the
    # residual instead of hiding it, and the series is never stripped, because
    # the antivirus-vs-OpenMP barrier noise on this machine makes the CPU sweep
    # high_cv every run. The gate exists so a noisy median is never silently
    # published.
    result["high_cv"] = result["cv"] > cv_threshold
    result["reruns"] = reruns
    result["verified"] = True
    return result


def _gpu_series(sizes: list[int], reps: int, warmup: int) -> list[dict]:
    # The device-resident GPU f32 series. Gated on has_cuda() so a CPU-only build
    # or a machine without a usable device simply omits it and the CPU bench
    # above still runs. Timing uses pg._core._cuda_time_matmul, the
    # cudaEvent-timed warm GEMM: operands are device-resident and the timed
    # region is the GEMM only (no H2D/D2H), so this measures device work, not a
    # wall-clock around an async launch. The transfer happens ONCE, outside the
    # timed region; cold and warm are both event-timed (cold = warmups=0/reps=1
    # before the clocks warm, warm = the warmups/reps result). verified=true is
    # published only after assert_matmul_close passes.
    if not pg.has_cuda():
        return []
    series = []
    for n in sizes:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((n, n)).astype(np.float32)
        b = rng.standard_normal((n, n)).astype(np.float32)

        xa = pg.to_device(a)
        xb = pg.to_device(b)

        # Verify ONCE before any timing, on the single sanctioned toleranced path:
        # from_device the device result and compare to the NumPy reference. f32 is
        # judged against the f64 ground truth assert_matmul_close recomputes,
        # exactly like the CPU suite -- no inline rtol.
        got = pg.from_device(pg.matmul(xa, xb))
        assert_matmul_close(got, a @ b, a, b)

        gflop = 2 * n**3 / 1e9
        # Cold: a single event-timed call before the clocks warm (low-clock first
        # launch after idle). Warm: the warmups/reps event-timed mean.
        # Both numbers come from cudaEvent pairs inside _cuda_time_matmul, never
        # wall-clock; the timed region is transfer-free.
        cold_ms = pg._core._cuda_time_matmul(xa, xb, 1, 0)
        warm_ms = pg._core._cuda_time_matmul(xa, xb, reps, warmup)
        warm_gflops = gflop / (warm_ms / 1e3)
        # NumPy CPU f32 at the same n, on the wall-clock path (a CPU GEMM is
        # synchronous, so perf_counter is the honest timer here). The ratio is
        # warm device-resident f32 GFLOP/s over NumPy CPU f32 GFLOP/s, an
        # f32-vs-f32 ratio independent of the CPU series --dtype.
        numpy_t = _bench(lambda: a @ b, reps, warmup)
        numpy_gflops = gflop / numpy_t["median_s"]
        series.append(
            {
                "n": n,
                "gflop": gflop,
                "cold_ms": cold_ms,
                "median_ms": warm_ms,
                "gflops": warm_gflops,
                "cold_gflops": gflop / (cold_ms / 1e3),
                "numpy_cpu_f32_gflops": numpy_gflops,
                "ratio_vs_numpy_cpu_f32": warm_gflops / numpy_gflops,
                "reps": reps,
                "warmup": warmup,
                # True only because assert_matmul_close passed above. The bench
                # never publishes a verified-false series (no result with verified
                # false ever reaches a results file).
                "verified": True,
            }
        )
        print(
            f"n={n:5d}  gpu  warm {warm_ms:9.3f} ms ({warm_gflops:8.1f} GF/s)"
            f"  cold {cold_ms:9.3f} ms ({series[-1]['cold_gflops']:8.1f} GF/s)"
            f"  numpy-cpu-f32 ({numpy_gflops:7.1f} GF/s)"
            f"  ratio {series[-1]['ratio_vs_numpy_cpu_f32']:.1f}x  (device-resident, transfer excluded)"
        )
    return series


def run(sizes: list[int], dtype: str, reps: int, warmup: int) -> dict:
    dt = np.dtype(dtype)
    results = {"manifest": _machine_manifest(), "dtype": dtype, "cases": []}
    for n in sizes:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((n, n)).astype(dt)
        b = rng.standard_normal((n, n)).astype(dt)

        ref = a @ b
        got = pg.matmul(a, b)
        # Verify on the single sanctioned toleranced path before any time is
        # recorded: fast and wrong is just wrong.
        assert_matmul_close(got, ref, a, b)

        gflop = 2 * n**3 / 1e9
        ours = _bench(lambda: pg.matmul(a, b), reps, warmup)
        numpy_t = _bench(lambda: a @ b, reps, warmup)
        case = {
            "n": n,
            "gflop": gflop,
            "peregrine": ours,
            "numpy": numpy_t,
            "pg_gflops": gflop / ours["median_s"],
            "numpy_gflops": gflop / numpy_t["median_s"],
            "speedup_vs_numpy": numpy_t["median_s"] / ours["median_s"],
            # The floor (min) ratio sits beside the median ratio because the
            # median on this machine is dragged by antivirus preemption of an
            # OpenMP worker at the barrier; the floor is the least-contaminated
            # estimate.
            "speedup_floor": numpy_t["min_s"] / ours["min_s"],
            # True because assert_matmul_close passed above before any time was
            # recorded. report.py's loader requires a stored verified flag on
            # every case; the bench never writes a verified-false series.
            "verified": True,
        }
        results["cases"].append(case)
        print(
            f"n={n:5d}  pg {ours['median_s']*1e3:9.2f} ms ({case['pg_gflops']:7.1f} GF/s)"
            f"  numpy {numpy_t['median_s']*1e3:9.2f} ms ({case['numpy_gflops']:7.1f} GF/s)"
            f"  speedup {case['speedup_vs_numpy']:.2f}x"
            f"  cv {ours['cv']*100:4.1f}%/{numpy_t['cv']*100:4.1f}%"
        )
    # The device-resident GPU f32 series runs over the same size grid,
    # event-timed and transfer-free; omitted cleanly on a CPU-only build or a
    # machine without a usable device. Reported under its own key so the CPU
    # cases above are never conflated with device-resident GPU numbers.
    gpu_cases = _gpu_series(sizes, reps, warmup)
    if gpu_cases:
        results["gpu_cases"] = gpu_cases
    return results


# ---------------------------------------------------------------------------
# The measurement protocol.
#
# A cell is one (op, dtype, shape, layout, variant, threads, seed) point with
# two or three contenders. Everything below exists so that a ratio written by
# run_cell can be trusted: same inputs, same batch size K and the same loop
# code for every contender, interleaved ABBA blocks so drift lands on both
# sides, every sample's output checked bitwise outside the timed region, a CV
# gate that reruns and finally closes a cell as noisy instead of averaging it,
# and a CPU clock reading that can discard samples taken off-frequency.
# ---------------------------------------------------------------------------

PEREGRINE = "peregrine"


class VerifyFailed(Exception):
    """An output differed bitwise from its first output, or the oracle failed."""


@dataclass
class Contender:
    name: str
    # One call of the operation; returns its output.
    fn: Callable[[], object]
    # Device contenders time themselves with cudaEvents: timer(k) returns the
    # per-call seconds of k back-to-back calls, and fn is then only the untimed
    # verification path.
    timer: Callable[[int], float] | None = None
    to_host: Callable[[object], np.ndarray] = np.asarray


@dataclass
class Protocol:
    samples: int = 30
    warmups: int = 5
    block: int = 5
    batch_target_s: float = 200e-6
    switch_gap_s: float = 0.3
    cv_limit: float = 0.05
    max_strikes: int = 3
    strike_cooldown_s: float = 30.0
    fresh_cooldown_s: float = 120.0
    clock_floor: float | None = None
    clock_ceiling: float | None = None
    clock_group_s: float = 0.05
    discard_limit: float = 0.20
    # "last" checks each batch's last call; "all" keeps and checks every call.
    # Keeping K results alive moves the timing it is meant to verify: paired
    # on this machine, +50 to +177 percent per call for allocating contenders
    # at 64 to 256 KB outputs and K of 5 to 22 (fresh memory every call), and
    # -7 to +8 percent in opposite directions for the two libraries at outputs
    # under 1 KB. With K == 1 the two modes are the same loop.
    verify_mode: str = "last"


def abba_order(names: list[str], cell_id: str) -> list[str]:
    # Seeding from the cell id (str seeding hashes with sha512, independent of
    # PYTHONHASHSEED) makes the start reproducible per cell while still spreading
    # first place across contenders over the matrix.
    order = list(names)
    random.Random(cell_id).shuffle(order)
    return order


def abba_blocks(order: list[str], rounds: int) -> list[str]:
    # Mirrored rounds: A B | B A | A B ... (A B C | C B A for three). Each
    # contender sits at every position equally often, so a slow drift in clock
    # or temperature cannot favor whoever always runs first.
    seq: list[str] = []
    for r in range(rounds):
        seq.extend(order if r % 2 == 0 else order[::-1])
    return seq


def batch_size(per_call_s: dict, target_s: float) -> int:
    # Sized from the FASTEST contender so even its samples reach the target;
    # slower contenders simply take longer per sample with the same K.
    fastest = min(per_call_s.values())
    if fastest >= target_s:
        return 1
    # The epsilon keeps an exact quotient such as 200e-6 / 1e-6 from rounding
    # up one call on float noise.
    return math.ceil(target_s / fastest - 1e-9)


def bitwise_equal(got, ref) -> bool:
    got = np.asarray(got)
    ref = np.asarray(ref)
    return (
        got.dtype == ref.dtype
        and got.shape == ref.shape
        and bool(np.array_equal(got, ref, equal_nan=True))
    )


def _batch_all(fn, k, slots, now):
    # Every call's result is stored by index so all of them can be verified
    # after the sample; the stores are part of the identical loop each
    # contender runs.
    t0 = now()
    for i in range(k):
        slots[i] = fn()
    return now() - t0


def _batch_last(fn, k, slots, now):
    # Only the last result survives the loop; earlier results are freed inside
    # the timed region exactly as in a plain user loop.
    t0 = now()
    for _ in range(k):
        r = fn()
    t1 = now()
    slots[0] = r
    return t1 - t0


_LOOPS = {"all": _batch_all, "last": _batch_last}


def _sample(c: Contender, k: int, mode: str, now) -> tuple[float, list]:
    # Returns (per-call seconds, outputs to verify).
    if c.timer is not None:
        return c.timer(k), [c.fn()]
    slots = [None] * (k if mode == "all" else 1)
    ns = _LOOPS[mode](c.fn, k, slots, now)
    return ns / 1e9 / k, slots


def _verify(c: Contender, outs: list, ref: np.ndarray) -> None:
    # Out-buffer contenders return the same object every call, so identical
    # objects are checked once: that is the buffer state after the sample.
    seen = set()
    for o in outs:
        if id(o) in seen:
            continue
        seen.add(id(o))
        if not bitwise_equal(c.to_host(o), ref):
            raise VerifyFailed(
                f"{c.name}: a sample output differs bitwise from its first output"
            )


def summarize(samples: list[float]) -> dict:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return {"samples_s": [], "n_samples": 0}
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    cv = statistics.stdev(samples) / med if arr.size >= 2 and med > 0 else float("nan")
    return {
        "samples_s": [float(s) for s in samples],
        "median_s": float(med),
        "q1_s": float(q1),
        "q3_s": float(q3),
        "min_s": float(arr.min()),
        "cv": float(cv),
        "n_samples": int(arr.size),
    }


def _out_of_band(reading: float, proto: Protocol) -> bool:
    if proto.clock_floor is not None and reading < proto.clock_floor:
        return True
    return proto.clock_ceiling is not None and reading > proto.clock_ceiling


def measure_cell(
    cell_id: str,
    contenders: list[Contender],
    check_first: Callable[[dict], None],
    proto: Protocol,
    *,
    refs: dict,
    sleep=time.sleep,
    clock=None,
    now=time.perf_counter_ns,
    log=print,
) -> dict:
    """One attempt at a cell. Raises VerifyFailed; returns stats otherwise."""
    names = [c.name for c in contenders]
    by = {c.name: c for c in contenders}
    mode = proto.verify_mode

    # The first output of each contender is its bitwise reference for every
    # later sample, kept across CV reruns so a rerun cannot quietly change what
    # "correct" means for the cell.
    if not refs:
        for c in contenders:
            refs[c.name] = np.array(c.to_host(c.fn()), copy=True)
        try:
            check_first(refs)
        except AssertionError as e:
            raise VerifyFailed(f"oracle: {e}") from e

    per_call = {}
    for c in contenders:
        k = 1
        while True:
            pc, _ = _sample(c, k, mode, now)
            if pc * k >= proto.batch_target_s or k >= 1 << 16:
                break
            k *= 2
        per_call[c.name] = pc
    K = batch_size(per_call, proto.batch_target_s)

    order = abba_order(names, cell_id)
    # A rate counter read over less than ~50 ms says little, so short samples
    # are grouped and read together; a block holds whole groups.
    group = {
        n: max(1, math.ceil(proto.clock_group_s / max(per_call[n] * K, 1e-9)))
        if clock is not None
        else 1
        for n in names
    }
    block = max(g * math.ceil(proto.block / g) for g in group.values())
    rounds = max(2, math.ceil(proto.samples / block))
    rounds += rounds % 2
    planned = rounds * block * len(names)

    for n in order:
        for _ in range(proto.warmups):
            _sample(by[n], K, mode, now)
    last = order[-1]

    samples = {n: [] for n in names}
    discards = {n: 0 for n in names}
    readings = {n: [] for n in names}
    clock_noisy = False
    for n in abba_blocks(order, rounds):
        if clock_noisy:
            break
        c = by[n]
        if n != last:
            # Idle gap at every switch so the previous library's spinning
            # worker threads have released the cores before the next one runs.
            sleep(proto.switch_gap_s)
            last = n
            if c.timer is not None:
                # The GPU drops to its idle clocks while the CPU contender
                # runs; the first event-timed chain after 3 s idle read 21
                # percent slow here, so one untimed sample absorbs the ramp.
                c.timer(K)
        taken = 0
        while taken < block:
            g = min(group[n], block - taken)
            if clock is not None:
                clock.start()
            got = []
            for _ in range(g):
                pc, outs = _sample(c, K, mode, now)
                _verify(c, outs, refs[n])
                got.append(pc)
            reading = clock.stop() if clock is not None else None
            if reading is not None and _out_of_band(reading, proto):
                discards[n] += g
                log(f"  discard {g} {n} sample(s): clock {reading:.1f}%")
                if sum(discards.values()) > proto.discard_limit * planned:
                    clock_noisy = True
                    break
                continue
            samples[n].extend(got)
            readings[n].append(reading)
            taken += g

    stats = {}
    for n in names:
        s = summarize(samples[n])
        s["discards"] = discards[n]
        if clock is not None:
            s["clock_pct"] = readings[n]
        stats[n] = s
    return {
        "batch_k": K,
        "warmups": proto.warmups,
        "block": block,
        "rounds": rounds,
        "order": order,
        # Recorded as "all" whenever every call was in fact checked.
        "verify_mode": "all" if K == 1 else mode,
        "contenders": stats,
        "clock_noisy": clock_noisy,
    }


def _close(m: dict, status: str, strikes: int, attempts: list, reason=None) -> dict:
    out = dict(m)
    out.pop("clock_noisy", None)
    out["status"] = status
    out["verified"] = True
    out["strikes"] = strikes
    out["attempts"] = attempts
    if reason:
        out["reason"] = reason
    if status == "ok" and PEREGRINE in out["contenders"]:
        pg_med = out["contenders"][PEREGRINE]["median_s"]
        ratio = {
            n: s["median_s"] / pg_med
            for n, s in out["contenders"].items()
            if n != PEREGRINE
        }
        if ratio:
            out["ratio"] = ratio
    return out


def run_cell(
    cell_id: str,
    contenders: list[Contender],
    check_first: Callable[[dict], None],
    proto: Protocol,
    *,
    sleep=time.sleep,
    clock=None,
    now=time.perf_counter_ns,
    fresh: Callable[[], dict] | None = None,
    log=print,
) -> dict:
    """A cell through the CV gate: up to max_strikes attempts, then one fresh
    process, then "noisy". Never averages attempts together."""
    refs: dict = {}
    attempts: list = []
    strikes = 0
    m = None
    while True:
        try:
            m = measure_cell(
                cell_id, contenders, check_first, proto,
                refs=refs, sleep=sleep, clock=clock, now=now, log=log,
            )
        except VerifyFailed as e:
            return {
                "status": "verify_failed",
                "verified": False,
                "reason": str(e),
                "strikes": strikes,
                "attempts": attempts,
            }
        cvs = {n: s.get("cv") for n, s in m["contenders"].items()}
        attempts.append({"cv": cvs, "discards": {n: s["discards"] for n, s in m["contenders"].items()}})
        if m["clock_noisy"]:
            return _close(m, "noisy", strikes, attempts, "clock discards above the limit")
        # A NaN CV (one sample, or a zero median) fails the comparison, so it
        # strikes rather than passing silently.
        if all(cv is not None and cv <= proto.cv_limit for cv in cvs.values()):
            return _close(m, "ok", strikes, attempts)
        strikes += 1
        log(f"  strike {strikes}: cv {', '.join(f'{n} {v:.3f}' for n, v in cvs.items())}")
        if strikes >= proto.max_strikes:
            break
        sleep(proto.strike_cooldown_s)
    if fresh is not None:
        sleep(proto.fresh_cooldown_s)
        try:
            child = fresh()
        except RuntimeError as e:
            return _close(m, "noisy", strikes, attempts, f"fresh process failed: {e}")
        child = dict(child)
        child["strikes"] = strikes + child.get("strikes", 0)
        child["attempts"] = attempts + child.get("attempts", [])
        child["fresh_process"] = True
        return child
    return _close(m, "noisy", strikes, attempts, "cv above the limit on every attempt")


class PdhClock:
    """`% Processor Performance` of all logical processors, via pdh.dll.

    The counter is a rate between two collects: start() and stop() bracket an
    interval and stop() returns the percentage of nominal frequency over it
    (above 100 under turbo, hence NOCAP100). The classic \\Processor object is
    absent on this machine; \\Processor Information is the one that exists.
    """

    PATH = r"\Processor Information(_Total)\% Processor Performance"
    PDH_FMT_DOUBLE_NOCAP100 = 0x8200

    def __init__(self):
        self.available = False
        self.reason = ""
        if os.name != "nt":
            self.reason = "PDH is Windows-only; no clock discards on this OS"
            return
        try:
            from ctypes import wintypes

            self._pdh = ctypes.WinDLL("pdh.dll")
            self._query = ctypes.c_void_p()
            self._counter = ctypes.c_void_p()
            rc = self._pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query))
            if rc == 0:
                rc = self._pdh.PdhAddEnglishCounterW(
                    self._query, self.PATH, None, ctypes.byref(self._counter)
                )
            if rc != 0:
                self.reason = f"PDH counter unavailable (0x{rc & 0xFFFFFFFF:08x})"
                return

            class _Value(ctypes.Structure):
                _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

            self._value_t = _Value
            self._dword = wintypes.DWORD
            self._pdh.PdhCollectQueryData(self._query)
            self.available = True
        except OSError as e:
            self.reason = f"PDH unavailable: {e}"

    def start(self) -> None:
        self._pdh.PdhCollectQueryData(self._query)

    def stop(self) -> float | None:
        self._pdh.PdhCollectQueryData(self._query)
        v = self._value_t()
        kind = self._dword()
        rc = self._pdh.PdhGetFormattedCounterValue(
            self._counter, self.PDH_FMT_DOUBLE_NOCAP100, ctypes.byref(kind), ctypes.byref(v)
        )
        # An unreadable interval is recorded as None and never discards.
        if rc != 0 or v.CStatus not in (0, 1):
            return None
        return float(v.doubleValue)


def _pools(user_api: str) -> list[dict]:
    if threadpool_info is None:
        return []
    return [d for d in threadpool_info() if d.get("user_api") == user_api]


def assert_threads(n: int) -> dict:
    """Pin peregrine to n threads where possible and prove both sides run n."""
    if threadpool_info is None:
        raise RuntimeError("threadpoolctl is required to pin and verify thread counts")
    if hasattr(pg, "set_num_threads"):
        pg.set_num_threads(n)
        source = "pg.set_num_threads"
    else:
        # Without a setter the only lever is OMP_NUM_THREADS before startup.
        if _START_OMP_NUM_THREADS != str(n):
            raise RuntimeError(
                f"this peregrine build has no set_num_threads, so the process must "
                f"start with OMP_NUM_THREADS={n}; it started with "
                f"OMP_NUM_THREADS={_START_OMP_NUM_THREADS!r}. Rerun this selection "
                f"as its own invocation with OMP_NUM_THREADS={n} exported."
            )
        source = "OMP_NUM_THREADS"
    blas = [d.get("num_threads") for d in _pools("blas")]
    omp = [d.get("num_threads") for d in _pools("openmp")]
    if not blas or any(t != n for t in blas):
        raise RuntimeError(f"BLAS pools run {blas} threads, expected {n}")
    if hasattr(pg, "get_num_threads"):
        pg_threads = int(pg.get_num_threads())
    elif omp:
        pg_threads = omp[0]
    else:
        raise RuntimeError("no OpenMP pool is loaded, so peregrine's thread count cannot be verified")
    if pg_threads != n or any(t != n for t in omp):
        raise RuntimeError(f"peregrine runs {pg_threads} threads (OpenMP pools {omp}), expected {n}")
    return {"blas": blas, "openmp": omp, "peregrine": pg_threads, "source": source}


@contextlib.contextmanager
def pinned_threads(n: int):
    if threadpool_limits is None:
        raise RuntimeError("threadpoolctl is required to pin and verify thread counts")
    with threadpool_limits(n, user_api="blas"):
        yield assert_threads(n)


def _home_pattern() -> re.Pattern | None:
    home = os.path.expanduser("~")
    if not home or home == "~":
        return None
    parts = [re.escape(p) for p in re.split(r"[\\/]+", home) if p]
    if not parts:
        return None
    lead = r"[\\/]" if home[0] in "\\/" else ""
    return re.compile(lead + r"[\\/]+".join(parts), re.IGNORECASE)


def redact_paths(obj, pattern: re.Pattern | None = None):
    """Replace the home directory in every string with ~, recursively."""
    pattern = pattern if pattern is not None else _home_pattern()
    if pattern is None:
        return obj
    if isinstance(obj, str):
        return pattern.sub("~", obj)
    if isinstance(obj, dict):
        return {k: redact_paths(v, pattern) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_paths(v, pattern) for v in obj]
    return obj


def _cpu_brand() -> str:
    try:
        if os.name == "nt":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_state(path: str) -> dict:
    # Tracked files only: an untracked notes folder must not mark every run
    # dirty, while any edit to a tracked source file does.
    def git(*args):
        p = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else None

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        commit = status = None
    return {
        "commit": commit or "unknown",
        "dirty": None if status is None else bool(status),
    }


def protocol_manifest(*, argv: list[str], series_seed: int | None, clock, extra=None) -> dict:
    """The run manifest: machine, toolchain, libraries, threads, source state."""
    info = _machine_manifest()
    blas = _pools("blas")
    omp = _pools("openmp")
    build = getattr(pg._core, "_build_info", None)
    build = build() if callable(build) else {}
    info.update(
        {
            "machine": platform.node(),
            "os_build": platform.platform(),
            "cpu_brand": _cpu_brand(),
            "power_plan": info["power_profile"],
            "compiler": build.get("compiler", "unknown"),
            "compiler_flags": build.get("flags", "unknown"),
            "openmp_runtime": os.path.basename(omp[0]["filepath"]) if omp and omp[0].get("filepath") else "unknown",
            "openblas_version": blas[0].get("version") if blas else "unknown",
            "blas_core_type": blas[0].get("architecture") if blas else "unknown",
            "blas_threads": blas[0].get("num_threads") if blas else None,
            "peregrine_threads": (
                int(pg.get_num_threads()) if hasattr(pg, "get_num_threads")
                else (omp[0].get("num_threads") if omp else None)
            ),
            "omp_num_threads_at_start": _START_OMP_NUM_THREADS,
            "peregrine_version": pg.__version__,
            "source_tree": _git_state(os.path.dirname(os.path.abspath(pg.__file__))),
            "harness_tree": _git_state(os.path.dirname(os.path.abspath(__file__))),
            "datetime": datetime.datetime.now().astimezone().isoformat(),
            "series_order_seed": series_seed,
            "cli": list(argv),
            "clock_counter": PdhClock.PATH if clock is not None and clock.available else None,
            "clock_counter_note": "" if clock is None else clock.reason,
        }
    )
    if extra:
        info.update(extra)
    return redact_paths(info)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 512, 1024])
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    # At least 5 warmup and 30 measured reps by default; numbers from fewer
    # reps are a smoke check, never a result.
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    out = run(args.sizes, args.dtype, args.reps, args.warmup)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.save}")
