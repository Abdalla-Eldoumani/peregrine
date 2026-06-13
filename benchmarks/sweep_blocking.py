"""MC/KC/NC blocking sweep for the f64 BLIS kernel (CPU-04).

The blocking constants are loop bounds, not unroll factors, so the native
_set_gemm_blocking hook walks the whole grid in one process; a compile-time
sweep would mean a rebuild per point. The grid is the gemm-optimization skill's:
MC {48,72,96,144} x KC {192,256,320} x NC {2048,4080}, 24 points.

Two methodology points specific to a blocking sweep:

1. OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES are set at the top of this file
   BEFORE fastmathext is imported. The OpenMP runtime reads them once at its
   initialization, so setting them after the first parallel region measures
   unpinned noise (RESEARCH Pitfall 8). The defaults pin 6 threads (the physical
   core count) to cores; override with the flags to measure SMT or fewer threads.
2. KC repartitions each output element's k accumulation, so results differ
   bitwise across KC values. Every grid point is therefore verified through
   assert_matmul_close (the single sanctioned toleranced path), never bitwise;
   a bitwise reference would read a correct kernel as broken (RESEARCH Pitfall 4).

Usage:
    python benchmarks/sweep_blocking.py --save benchmarks/results/tuning/sweep_zpicy.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import time

# bench-protocol rule 8 / RESEARCH Pitfall 8: pin threads in the environment
# BEFORE the first import that can touch the OpenMP runtime. setdefault so an
# operator who exported a different OMP_NUM_THREADS to test another thread count
# is honored rather than overridden.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

import numpy as np  # noqa: E402

import fastmathext as fme  # noqa: E402
import fastmathext._core as _core  # noqa: E402

# The bench harness is the analog for manifest, timing, verification, and the
# committed-vs-local JSON split; reuse its pieces rather than reimplement them so
# the sweep and the headline bench measure identically. The verification helper
# is imported transitively through bench_matmul (which adds tests/ to sys.path).
from bench_matmul import (  # noqa: E402
    _bench,
    _machine_manifest,
    assert_matmul_close,
)

# The skill's starting grid. The chosen winner is committed into gemm_blis.cpp's
# current_blocking() default with this grid's measured GFLOP/s in a comment.
MC_GRID = [48, 72, 96, 144]
KC_GRID = [192, 256, 320]
NC_GRID = [2048, 4080]


def _omp_settings() -> dict:
    # Record the exact thread pinning in the manifest so a reader knows the grid
    # was measured pinned, not on scheduler luck (bench-protocol rule 8).
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", "unset"),
        "OMP_PLACES": os.environ.get("OMP_PLACES", "unset"),
    }


def run(sizes: list[int], reps: int, warmup: int) -> dict:
    manifest = _machine_manifest()
    manifest["omp"] = _omp_settings()
    # Restore the committed default after the sweep so the process is not left in
    # a swept state if anything imports this module.
    default = _core._get_gemm_blocking()
    results = {
        "manifest": manifest,
        "dtype": "float64",
        "grid": {"mc": MC_GRID, "kc": KC_GRID, "nc": NC_GRID},
        "default_blocking": default,
        "sizes": sizes,
        "cases": [],
    }
    try:
        for n in sizes:
            rng = np.random.default_rng(0)
            a = rng.standard_normal((n, n))
            b = rng.standard_normal((n, n))
            ref = a @ b
            gflop = 2 * n**3 / 1e9
            for mc in MC_GRID:
                for kc in KC_GRID:
                    for nc in NC_GRID:
                        _core._set_gemm_blocking(mc, kc, nc)
                        got = fme.matmul(a, b)
                        # Toleranced, never bitwise: KC changes the k-chunking and
                        # so the result bitwise, within the matmul contract. A
                        # verified-false point never reaches the JSON.
                        assert_matmul_close(got, ref, a, b)
                        timing = _bench(lambda: fme.matmul(a, b), reps, warmup)
                        gflops = gflop / timing["median_s"]
                        case = {
                            "n": n,
                            "mc": mc,
                            "kc": kc,
                            "nc": nc,
                            "gflop": gflop,
                            "timing": timing,
                            "gflops": gflops,
                            "verified": True,
                        }
                        results["cases"].append(case)
                        print(
                            f"n={n:5d}  mc={mc:3d} kc={kc:3d} nc={nc:4d}"
                            f"  {timing['median_s']*1e3:8.2f} ms ({gflops:7.1f} GF/s)"
                            f"  cv {timing['cv']*100:4.1f}%  verified True"
                        )
    finally:
        _core._set_gemm_blocking(default["mc"], default["kc"], default["nc"])

    # Winner by median GFLOP/s across all sizes is reported per size and overall;
    # the operator commits the overall winner (or the per-size winner the protocol
    # run agrees on) into the kernel default with the measured grid in a comment.
    best = max(results["cases"], key=lambda c: c["gflops"])
    results["winner"] = {
        "mc": best["mc"],
        "kc": best["kc"],
        "nc": best["nc"],
        "n": best["n"],
        "gflops": best["gflops"],
    }
    print(
        f"\nwinner: mc={best['mc']} kc={best['kc']} nc={best['nc']}"
        f" at n={best['n']}  {best['gflops']:.1f} GF/s"
    )
    for n in sizes:
        per = max((c for c in results["cases"] if c["n"] == n), key=lambda c: c["gflops"])
        print(
            f"  n={n:5d} winner: mc={per['mc']} kc={per['kc']} nc={per['nc']}"
            f"  {per['gflops']:.1f} GF/s"
        )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    # The skill sweeps blocking at n=512 (Bp/Ap fit the cache budget there) and
    # cross-checks n=1024; both default in so one run produces the committed grid.
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
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

    out = run(args.sizes, args.reps, args.warmup)
    out["measured_at"] = datetime.datetime.now().isoformat()
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.save}")
