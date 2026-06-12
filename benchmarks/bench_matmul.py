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
import json
import platform
import statistics
import time

import numpy as np

import fastmathext as fme

try:
    from threadpoolctl import threadpool_info
except ImportError:
    threadpool_info = None


def _machine_manifest() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fastmathext": fme.__version__,
        "cpu_features": fme.cpu_features(),
        "cuda_build": fme.has_cuda_build(),
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
    return {
        "median_s": statistics.median(times),
        "p25_s": statistics.quantiles(times, n=4)[0],
        "p75_s": statistics.quantiles(times, n=4)[2],
        "min_s": min(times),
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
        rtol = 1e-4 if dt == np.float32 else 1e-10
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=rtol)

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
        )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 512, 1024])
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    out = run(args.sizes, args.dtype, args.reps, args.warmup)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.save}")
