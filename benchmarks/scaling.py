"""Thread-scaling measurement for the f64 BLIS kernel (CPU-05).

CPU-05 requires >= 4x f64 throughput from 1 to 6 threads at n=1024 under
/openmp:llvm. At n=1024 with NC=4080 there is exactly one jc block, so all of
the scaling comes from the parallel ic loop; a jc-only parallelization would
measure 1.0x (RESEARCH Pitfall 3).

The OpenMP runtime reads OMP_NUM_THREADS once at its initialization, so a single
process cannot remeasure at different thread counts. This mirrors the
tests/test_threads.py subprocess model: one child per thread count, the OMP env
set in the child's environment BEFORE the child imports fastmathext, a full
os.environ copy so Windows DLL resolution for the OpenMP runtime survives, and
the measured median crossing the process boundary as JSON on stdout. Each child
verifies its own result through the single toleranced path before reporting; a
verified-false child fails the run rather than contributing a number.

Usage:
    python benchmarks/scaling.py --save benchmarks/results/tuning/scaling_zpicy.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

# The manifest comes from the bench harness; threads are recorded per child.
# The parent does not import fastmathext (it never benches), so no OMP env is
# pinned here: only the children measure, and they pin their own.
from bench_matmul import _machine_manifest

# Child program: pin already happened via the inherited env (set before launch),
# import, build seeded inputs outside timing, verify, bench, print one JSON line.
# n and the protocol floors arrive as argv so the parent owns them.
_CHILD = """\
import json
import sys
import time
import statistics

import numpy as np

import fastmathext as fme

_TESTS = sys.argv[4]
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from conftest import assert_matmul_close

n = int(sys.argv[1])
reps = int(sys.argv[2])
warmup = int(sys.argv[3])

rng = np.random.default_rng(0)
a = rng.standard_normal((n, n))
b = rng.standard_normal((n, n))
ref = a @ b
got = fme.matmul(a, b)
assert_matmul_close(got, ref, a, b)

for _ in range(warmup):
    fme.matmul(a, b)
times = []
for _ in range(reps):
    t0 = time.perf_counter_ns()
    fme.matmul(a, b)
    times.append((time.perf_counter_ns() - t0) / 1e9)
median = statistics.median(times)
cv = statistics.stdev(times) / median if reps >= 2 and median > 0 else float("nan")
gflop = 2 * n ** 3 / 1e9
print(json.dumps({
    "n": n,
    "median_s": median,
    "cv": cv,
    "gflops": gflop / median,
    "reps": reps,
    "verified": True,
}))
"""


def _tests_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"
    )


def _measure(threads: int, n: int, reps: int, warmup: int) -> dict:
    # Full env copy with the thread count and pinning merged in BEFORE launch:
    # the child's OpenMP runtime reads them at import. A stripped env breaks the
    # Windows DLL search for the OpenMP runtime (the test_threads.py lesson).
    env = dict(
        os.environ,
        OMP_NUM_THREADS=str(threads),
        OMP_PROC_BIND="close",
        OMP_PLACES="cores",
    )
    p = subprocess.run(
        [sys.executable, "-c", _CHILD, str(n), str(reps), str(warmup), _tests_dir()],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # stderr in the message so a child import or verification failure reads as
    # itself rather than as an opaque JSON parse error.
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def run(threads: list[int], n: int, reps: int, warmup: int) -> dict:
    manifest = _machine_manifest()
    manifest["omp"] = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
    results = {
        "manifest": manifest,
        "dtype": "float64",
        "n": n,
        "thread_counts": threads,
        "series": [],
    }
    by_threads: dict[int, float] = {}
    for t in threads:
        m = _measure(t, n, reps, warmup)
        m["threads"] = t
        results["series"].append(m)
        by_threads[t] = m["gflops"]
        print(
            f"threads={t:2d}  n={n}  {m['median_s']*1e3:8.2f} ms"
            f"  ({m['gflops']:7.1f} GF/s)  cv {m['cv']*100:4.1f}%  verified True"
        )

    # CPU-05 is the 6-vs-1 ratio at n=1024. Report it plus every available ratio
    # against 1 thread so SMT (12) and the intermediate points are visible.
    if 1 in by_threads:
        base = by_threads[1]
        results["scaling_vs_1"] = {
            str(t): by_threads[t] / base for t in threads if t in by_threads
        }
        if 6 in by_threads:
            ratio = by_threads[6] / base
            results["ratio_6_vs_1"] = ratio
            print(f"\n6-vs-1 thread scaling at n={n}: {ratio:.2f}x  (CPU-05 target >= 4x)")
            if ratio < 4.0:
                print(
                    "  below the 4x target: confirm ic parallelism, pinning, and "
                    "cooldowns before accepting (CPU-05 is a requirement)"
                )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    # 1,2,4,6 are the CPU-05 series; 12 measures SMT (the skill predicts 6 wins
    # for FMA-bound code but it is measured, not assumed).
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6, 12])
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    if args.reps < 30 or args.warmup < 5:
        print(
            f"warning: reps={args.reps} warmup={args.warmup} is below the protocol "
            "floor (>=30 reps, >=5 warmup); numbers from this run are not quotable"
        )

    out = run(args.threads, args.n, args.reps, args.warmup)
    out["measured_at"] = datetime.datetime.now().isoformat()
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.save}")
