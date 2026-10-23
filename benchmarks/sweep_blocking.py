"""MC/KC/NC blocking sweep for the packed GEMM, per dtype.

The blocking constants are loop bounds, not unroll factors, so the native
_set_gemm_blocking hook walks the whole grid in one process; a compile-time
sweep would mean a rebuild per point. The grid is the gemm-optimization skill's:
MC {48,72,96,144} x KC {192,256,320} x NC {2048,4080}, 24 points.

Two methodology points specific to a blocking sweep:

1. OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES are set at the top of this file
   BEFORE peregrine is imported. The OpenMP runtime reads them once at its
   initialization, so setting them after the first parallel region measures
   unpinned noise. The defaults pin 6 threads (the physical
   core count) to cores; override with the flags to measure SMT or fewer threads.
2. KC repartitions each output element's k accumulation, so results differ
   bitwise across KC values. Every grid point is therefore verified through
   assert_matmul_close (the single sanctioned toleranced path), never bitwise;
   a bitwise reference would read a correct kernel as broken.

3. Each grid point is timed twice, once on a forward walk of the grid and once
   on the reverse walk, and ranked on the median of both runs' samples, so a
   laptop's slow thermal drift over the sweep biases no point.

Usage:
    python benchmarks/sweep_blocking.py --dtype float32 --save benchmarks/results/tuning/sweep_refbox_f32.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time

# bench-protocol rule 8: pin threads in the environment
# BEFORE the first import that can touch the OpenMP runtime. setdefault so an
# operator who exported a different OMP_NUM_THREADS to test another thread count
# is honored rather than overridden.
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np  # noqa: E402

import peregrine as pg  # noqa: E402
import peregrine._core as _core  # noqa: E402

# The bench harness is the analog for manifest, timing, verification, and the
# committed-vs-local JSON split; reuse its pieces rather than reimplement them so
# the sweep and the headline bench measure identically. The verification helper
# is imported transitively through bench_matmul (which adds tests/ to sys.path).
from bench_matmul import (  # noqa: E402
    _machine_manifest,
    assert_matmul_close,
)
from bench_suite import _confine_to_physical_cores  # noqa: E402

# The skill's starting grid. The chosen winner is committed into gemm_blis.cpp's
# current_blocking() default with this grid's measured GFLOP/s in a comment.
MC_GRID = [48, 72, 96, 144]
KC_GRID = [192, 256, 320]
NC_GRID = [2048, 4080]
AFFINITY = None


def _omp_settings() -> dict:
    # Record the thread count and placement so a reader knows how the grid ran.
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "peregrine_threads": pg.get_num_threads(),
        "affinity_mask": AFFINITY,
    }


def _samples(fn, reps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - t0) / 1e9)
    return out


def run(dtype: str, sizes: list[int], reps: int, warmup: int, grid: dict) -> dict:
    f32 = dtype == "float32"
    manifest = _machine_manifest()
    manifest["omp"] = _omp_settings()
    # Restore the committed default after the sweep so the process is not left in
    # a swept state if anything imports this module.
    default = _core._get_gemm_blocking(f32)
    points = [(mc, kc, nc) for mc in grid["mc"] for kc in grid["kc"] for nc in grid["nc"]]
    results = {
        "manifest": manifest,
        "dtype": dtype,
        "grid": grid,
        "default_blocking": default,
        "sizes": sizes,
        "cases": [],
    }
    try:
        for n in sizes:
            rng = np.random.default_rng(0)
            a = rng.standard_normal((n, n)).astype(dtype)
            b = rng.standard_normal((n, n)).astype(dtype)
            ref = a @ b
            gflop = 2 * n**3 / 1e9
            samples = {p: [] for p in points}
            for walk in (points, points[::-1]):
                for mc, kc, nc in walk:
                    _core._set_gemm_blocking(mc, kc, nc, f32)
                    # Toleranced, never bitwise: KC changes the k-chunking and so
                    # the result bitwise, within the matmul contract.
                    assert_matmul_close(pg.matmul(a, b), ref, a, b)
                    samples[(mc, kc, nc)] += _samples(lambda: pg.matmul(a, b), reps, warmup)
            for (mc, kc, nc), t in samples.items():
                q1, med, q3 = np.percentile(t, [25, 50, 75])
                case = {
                    "n": n, "mc": mc, "kc": kc, "nc": nc, "gflop": gflop,
                    "median_s": float(med), "iqr_s": float(q3 - q1), "min_s": float(min(t)),
                    "gflops": gflop / med, "verified": True,
                }
                results["cases"].append(case)
                print(f"n={n:5d}  mc={mc:3d} kc={kc:3d} nc={nc:4d}  med {med*1e3:8.3f} ms"
                      f" ({gflop / med:6.1f} GF/s)  iqr {(q3 - q1)*1e3:6.3f} ms")
    finally:
        _core._set_gemm_blocking(default["mc"], default["kc"], default["nc"], f32)

    # Per size: the median winner, and every point whose median is within the
    # winner's IQR (a tie the sweep cannot separate).
    results["winners"] = {}
    for n in sizes:
        cases = [c for c in results["cases"] if c["n"] == n]
        best = min(cases, key=lambda c: c["median_s"])
        ties = [c for c in cases if c["median_s"] <= best["median_s"] + best["iqr_s"]]
        results["winners"][str(n)] = {
            "best": {k: best[k] for k in ("mc", "kc", "nc", "gflops")},
            "ties": [{k: c[k] for k in ("mc", "kc", "nc", "gflops")} for c in ties],
        }
        print(f"  n={n:5d} best mc={best['mc']} kc={best['kc']} nc={best['nc']} "
              f"{best['gflops']:.1f} GF/s; {len(ties)} points within its IQR")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    # The skill sweeps blocking at n=512 (Bp/Ap fit the cache budget there) and
    # cross-checks n=1024; both default in so one run produces the committed grid.
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--mc", type=int, nargs="+", default=MC_GRID)
    p.add_argument("--kc", type=int, nargs="+", default=KC_GRID)
    p.add_argument("--nc", type=int, nargs="+", default=NC_GRID)
    # Protocol floors (bench-protocol rule 5): >= 5 warmup, >= 30 measured reps.
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    if args.reps < 30 or args.warmup < 5:
        print(
            f"warning: reps={args.reps} warmup={args.warmup} is below the protocol "
            "floor (>=30 reps, >=5 warmup); numbers from this run are not quotable"
        )

    AFFINITY = _confine_to_physical_cores()
    pg.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    out = run(args.dtype, args.sizes, args.reps, args.warmup, {"mc": args.mc, "kc": args.kc, "nc": args.nc})
    out["measured_at"] = datetime.datetime.now().isoformat()
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.save}")
