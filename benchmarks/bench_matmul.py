"""Apples-to-apples matmul benchmark: ndarray in, ndarray out, both sides.

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

Usage:
    python benchmarks/bench_matmul.py --sizes 256 512 1024 --dtype float64
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

import fastmathext as fme

# The tolerance contract lives in one place, tests/conftest.py. Routing the
# per-run verification through assert_matmul_close keeps the bench on the single
# sanctioned toleranced path; an inline rtol here is exactly what bench-protocol
# rule 14 and the single-path rule forbid (a bench that loosens tolerance to
# pass is a bench that lies).
_TESTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from conftest import assert_matmul_close

try:
    from threadpoolctl import threadpool_info
except ImportError:
    threadpool_info = None


def _blas_identity() -> str:
    # The comparator BLAS identity comes from threadpoolctl's blas entry, the
    # same source bench-protocol rule 9 names. scipy-openblas surfaces here as
    # its internal_api plus version; no separate probe.
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
    # (gpu, driver) for the DESIGN_SYSTEM manifest keys, best-effort and never
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


def _machine_manifest() -> dict:
    # bench-protocol rule 9: no manifest, no merge. Every results JSON carries
    # the machine identity, the BLAS the comparison ran against, the power
    # profile (thermal state changes throughput on a laptop), and a timestamp.
    gpu, driver = _gpu_manifest()
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fastmathext": fme.__version__,
        "cpu_features": fme.cpu_features(),
        # The BUILD flag, not the runtime has_cuda(): the manifest records what
        # the binary was compiled with (whether the CUDA path exists at all), and
        # the gpu/driver fields below record the device that was actually present.
        # A run on a CUDA build with the GPU busy elsewhere should still read
        # cuda_build True.
        "cuda_build": fme._has_cuda_build(),
        "gpu": gpu,
        "driver": driver,
        "blas": _blas_identity(),
        "power_profile": _power_profile(),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if threadpool_info is not None:
        info["threadpools"] = threadpool_info()
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
    # Coefficient of variation, stdev over median. bench-protocol rule 6 gates a
    # series at CV > 5 percent; this body reports it as a readout so a noisy run
    # is visible at the console, while _bench's opt-in cv_gate wrapper owns the
    # rerun-after-cooldown machinery. stdev needs two samples and a non-zero
    # median.
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
    # The single shared timing core. The DEFAULT path (cv_gate False) returns
    # _measure_once unchanged, so every existing reuser -- bench_fused's
    # `from bench_matmul import _bench`, scaling.py, sweep_blocking.py,
    # calibrate.py -- calls it positionally and sees a byte-identical dict shape
    # (median_s, p25_s, p75_s, min_s, cv, reps). They are untouched by this gate.
    result = _measure_once(fn, reps, warmup)
    if not cv_gate:
        return result
    # bench-protocol rule 6: a series whose CV exceeds the threshold is
    # invalidated and rerun after a cooldown (rule 10: >=30s between heavy
    # series); after a bounded number of reruns a still-noisy series is recorded
    # rather than dropped. cooldown_fn is injectable (default time.sleep) so a
    # unit test stubs it to a no-op and never waits 30 real seconds.
    reruns = 0
    while result["cv"] > cv_threshold and reruns < max_reruns:
        cooldown_fn(cooldown_s)
        result = _measure_once(fn, reps, warmup)
        reruns += 1
    # rule 6/9/11: a high_cv series is STILL verified. The floor (min_s) is the
    # honest headline statistic regardless of CV; high_cv documents the residual
    # noise rather than hiding it behind a footnote, and the series is never
    # stripped -- 07-03's load_series must accept the high_cv+verified pair (the
    # AV-noisy CPU sweep on this machine is high_cv every run). The gate exists
    # so a noisy MEDIAN is never silently published, not to "fix" the
    # AV-vs-OpenMP barrier noise that is structural here.
    result["high_cv"] = result["cv"] > cv_threshold
    result["reruns"] = reruns
    result["verified"] = True
    return result


def _gpu_series(sizes: list[int], reps: int, warmup: int) -> list[dict]:
    # The device-resident GPU f32 series (GPU-08). Gated on has_cuda() so a
    # CPU-only build or a machine without a usable device simply omits it and the
    # CPU bench above still runs. Timing CONSUMES fme._core._cuda_time_matmul (the
    # cudaEvent-timed warm GEMM created in 04-05): operands are device-resident,
    # the timed region is the GEMM only (no H2D/D2H), so this measures device work,
    # not a wall-clock around an async launch (which would report nonsense). The
    # transfer happens ONCE, outside the timed region; cold and warm are both
    # event-timed (cold = warmups=0/reps=1 before the clocks warm, warm = the
    # protocol warmups/reps result). verified=true is published only after the
    # single sanctioned toleranced path (assert_matmul_close) passes -- a bench
    # that loosens tolerance or times a transfer is a bench that lies.
    if not fme.has_cuda():
        return []
    series = []
    for n in sizes:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((n, n)).astype(np.float32)
        b = rng.standard_normal((n, n)).astype(np.float32)

        xa = fme.to_device(a)
        xb = fme.to_device(b)

        # Verify ONCE before any timing, on the single sanctioned toleranced path:
        # from_device the device result and compare to the NumPy reference. f32 is
        # judged against the f64 ground truth assert_matmul_close recomputes,
        # exactly like the CPU suite -- no inline rtol.
        got = fme.from_device(fme.matmul(xa, xb))
        assert_matmul_close(got, a @ b, a, b)

        gflop = 2 * n**3 / 1e9
        # Cold: a single event-timed call before the clocks warm (low-clock first
        # launch after idle). Warm: the protocol warmups/reps event-timed mean.
        # Both numbers come from cudaEvent pairs inside _cuda_time_matmul, never
        # wall-clock; the timed region is transfer-free.
        cold_ms = fme._core._cuda_time_matmul(xa, xb, 1, 0)
        warm_ms = fme._core._cuda_time_matmul(xa, xb, reps, warmup)
        warm_gflops = gflop / (warm_ms / 1e3)
        # NumPy CPU f32 at the same n, on the wall-clock path (a CPU GEMM is
        # synchronous, so perf_counter is the honest timer here). This is the
        # GPU-08 denominator: warm device-resident f32 GFLOP/s vs NumPy CPU f32
        # GFLOP/s, an f32-vs-f32 ratio independent of the CPU series --dtype.
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
                # never publishes a verified-false series (benchmarks/CLAUDE.md).
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
        got = fme.matmul(a, b)
        # Verify on the single sanctioned toleranced path before any time is
        # recorded (bench-protocol rule 11: fast and wrong is just wrong).
        assert_matmul_close(got, ref, a, b)

        gflop = 2 * n**3 / 1e9
        ours = _bench(lambda: fme.matmul(a, b), reps, warmup)
        numpy_t = _bench(lambda: a @ b, reps, warmup)
        case = {
            "n": n,
            "gflop": gflop,
            "fastmathext": ours,
            "numpy": numpy_t,
            "fme_gflops": gflop / ours["median_s"],
            "numpy_gflops": gflop / numpy_t["median_s"],
            "speedup_vs_numpy": numpy_t["median_s"] / ours["median_s"],
        }
        results["cases"].append(case)
        print(
            f"n={n:5d}  fme {ours['median_s']*1e3:9.2f} ms ({case['fme_gflops']:7.1f} GF/s)"
            f"  numpy {numpy_t['median_s']*1e3:9.2f} ms ({case['numpy_gflops']:7.1f} GF/s)"
            f"  speedup {case['speedup_vs_numpy']:.2f}x"
            f"  cv {ours['cv']*100:4.1f}%/{numpy_t['cv']*100:4.1f}%"
        )
    # The device-resident GPU f32 series (GPU-08) runs over the same size grid,
    # event-timed and transfer-free; omitted cleanly on a CPU-only build or a
    # machine without a usable device. Reported under its own key so the CPU
    # cases above are never conflated with device-resident GPU numbers.
    gpu_cases = _gpu_series(sizes, reps, warmup)
    if gpu_cases:
        results["gpu_cases"] = gpu_cases
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 512, 1024])
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    # Protocol floors (bench-protocol rule 5): at least 5 warmup, at least 30
    # measured reps. The old 3/15 defaults are below the floor and STATE.md
    # flags numbers from them as not quotable.
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    out = run(args.sizes, args.dtype, args.reps, args.warmup)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.save}")
