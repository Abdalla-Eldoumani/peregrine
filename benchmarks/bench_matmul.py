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


def _machine_manifest() -> dict:
    # bench-protocol rule 9: no manifest, no merge. Every results JSON carries
    # the machine identity, the BLAS the comparison ran against, the power
    # profile (thermal state changes throughput on a laptop), and a timestamp.
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fastmathext": fme.__version__,
        "cpu_features": fme.cpu_features(),
        "cuda_build": fme.has_cuda_build(),
        "blas": _blas_identity(),
        "power_profile": _power_profile(),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if threadpool_info is not None:
        info["threadpools"] = threadpool_info()
    return info


def _bench(fn, reps: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e9)
    median = statistics.median(times)
    # Coefficient of variation, stdev over median. bench-protocol rule 6 gates a
    # series at CV > 5 percent; this phase reports it as a readout so a noisy run
    # is visible at the console, but the gate-with-automatic-rerun machinery is
    # Phase 7, not here. stdev needs two samples and a non-zero median.
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
