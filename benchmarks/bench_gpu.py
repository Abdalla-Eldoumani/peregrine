"""The full GPU matrix: with/without-transfer x cold/warm as four labeled series.

bench-protocol rule 7 wants cold and warm as separate labeled series, and the
dispatch-cost story needs the host round-trip a user actually pays, not only the
device-resident slice. Only the device-resident "without-transfer" slice exists
today (bench_matmul._gpu_series, the GPU-08 measurement); this file adds the
cold/warm labeling and the with-transfer host round-trip, and saves the four
series to a results JSON the README and plots draw from.

The two regimes are timed by DIFFERENT clocks, on purpose:
1. without-transfer (device-resident): event-timed via _cuda_time_matmul. The
   operands live on the device, the timed region is the GEMM only (no H2D/D2H),
   so this measures device work, never a wall-clock around an async launch (rule
   12). This regime IS bench_matmul._gpu_series verbatim -- reused, not
   re-derived, so the per-size record carries the _gpu_series keys exactly and
   07-03 consumes it as gpu_cases by construction.
2. with-transfer (host round-trip): wall-clocked via the CV-gated _bench. The
   round-trip crosses Python/native/CUDA three times and from_device owns the
   D2H sync, so the round-trip is SYNCHRONOUS and perf_counter is the honest
   timer here -- the ONE sanctioned wall-clock GPU exception (rule 12). This is
   the dispatch cost _matmul_gpu_staged pays for a host array.

CuPy is ABSENT on this machine (verified), so per rule 15 ("where installed")
the comparison is GPU vs NumPy CPU f32; cupy is never imported and the absence
is recorded in the methodology block.

Every device entry sits behind has_cuda() so the OFF build (where
_cuda_time_matmul does not exist) imports this module with no AttributeError --
each series function returns an empty list before touching any device entry.

Usage:
    python benchmarks/bench_gpu.py --sizes 256 512 1024 --reps 30 --warmup 5 --save out.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import peregrine as pg

# Reuse the matmul harness rather than re-deriving the protocol: the manifest
# capture, the CV-gated timing core, AND the device-resident _gpu_series (the
# without-transfer regime IS that series). A copy would be a second place the
# protocol could drift. bench_fused.py line 48 is the same import pattern.
from bench_matmul import _bench, _gpu_series, _machine_manifest

# The single sanctioned toleranced path. bench_matmul put tests/ on sys.path at
# import for assert_matmul_close, so the same import works here. Routing per-run
# verification through it keeps the bench on the one tolerance contract; an
# inline rtol is exactly what bench-protocol rule 14 forbids (a bench that
# loosens tolerance to pass lies).
from conftest import assert_matmul_close

# CuPy is ABSENT on this machine (verified: ModuleNotFoundError). Per rule 15
# ("compare against the strong baseline WHERE installed") the GPU comparison is
# against NumPy CPU f32, not CuPy. This flag is recorded in the methodology block
# so a reader knows the denominator; cupy is never imported.
_CUPY_AVAILABLE = False


def _without_transfer_series(sizes: list[int], reps: int, warmup: int) -> list[dict]:
    # The device-resident regime (GPU-08): operands live on the device, the timed
    # region is the GEMM only, both cold and warm event-timed via
    # _cuda_time_matmul. This IS bench_matmul._gpu_series: reuse it so each record
    # carries the _gpu_series keys VERBATIM (n, gflop, cold_ms, median_ms warm,
    # gflops, cold_gflops, numpy_cpu_f32_gflops, ratio_vs_numpy_cpu_f32, reps,
    # warmup, verified). That dict is exactly the gpu_cases shape 07-03 reads, so
    # do NOT re-derive or rename it here. The has_cuda() gate lives inside
    # _gpu_series, which returns [] on the OFF build before reaching
    # _cuda_time_matmul; the redundant guard here keeps this function honest about
    # never touching a device entry on OFF and makes the gate visible at this call
    # site too.
    if not pg.has_cuda():
        return []
    series = []
    for record in _gpu_series(sizes, reps, warmup):
        # Attach the rule-7 cold/warm labels alongside the reused record WITHOUT
        # altering its keys: median_ms is the warm event-timed value and cold_ms
        # the cold one, both already present, so the labels select which of the
        # two the consumer reads. Copy so the originals stay byte-identical to
        # what _gpu_series emits (the gpu_cases consumer contract).
        warm = dict(record)
        warm["label"] = "without-transfer"
        warm["phase"] = "warm"
        cold = dict(record)
        cold["label"] = "without-transfer"
        cold["phase"] = "cold"
        series.append(warm)
        series.append(cold)
    return series


def _with_transfer_series(sizes: list[int], reps: int, warmup: int) -> list[dict]:
    # The host round-trip regime (the dispatch-cost story): to_device both
    # operands, matmul on the device, from_device the result. from_device owns the
    # D2H sync, so the whole round-trip is SYNCHRONOUS and perf_counter (_bench) is
    # the honest timer -- the one sanctioned wall-clock GPU exception (rule 12).
    # Gated on has_cuda() so the OFF build returns [] before reaching to_device /
    # the device matmul overload, which do not exist there.
    if not pg.has_cuda():
        return []
    series = []
    for n in sizes:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((n, n)).astype(np.float32)
        b = rng.standard_normal((n, n)).astype(np.float32)

        # Verify ONCE before any timing, on the single sanctioned toleranced path
        # (rule 11): run the round-trip, compare the host result to the NumPy
        # reference. f32 is judged against the f64 ground truth assert_matmul_close
        # recomputes, exactly like the CPU suite -- no inline rtol. Bind the
        # device handles to locals and drop them after the verify so they do not
        # straggle to interpreter finalization (the 06-04 nanobind-leak lesson).
        xa = pg.to_device(a)
        xb = pg.to_device(b)
        got = pg.from_device(pg.matmul(xa, xb))
        assert_matmul_close(got, a @ b, a, b)
        del xa, xb, got

        gflop = 2 * n**3 / 1e9

        def round_trip():
            # The full host round-trip the dispatch story is about: H2D both
            # operands, the device GEMM, D2H the result. from_device syncs the
            # boundary, so the closure returns only after all device work
            # completes -- perf_counter around it is honest.
            ra = pg.to_device(a)
            rb = pg.to_device(b)
            return pg.from_device(pg.matmul(ra, rb))

        # Cold FIRST, before any warm rep touches the device: a single round-trip
        # is the honest first-launch-after-idle number. It is timed directly with
        # perf_counter (not _bench) because _bench's _measure_once computes
        # quantiles, which need >=2 samples; a cold measurement is exactly one
        # sample. The round-trip is synchronous (from_device syncs), so
        # perf_counter is honest here (the rule-12 wall-clock exception).
        _t0 = time.perf_counter_ns()
        round_trip()
        cold_s = (time.perf_counter_ns() - _t0) / 1e9
        # Warm: the protocol warmups/reps wall-clock, CV-gated (the round-trip
        # runs on the CPU-visible timeline, so the AV-vs-OpenMP noise that gates
        # CPU series applies; the gate reruns a noisy median and records high_cv,
        # never dropping the series).
        warm = _bench(round_trip, reps, warmup, cv_gate=True)

        # NumPy CPU f32 at the same n on the wall-clock floor (a CPU GEMM is
        # synchronous, so perf_counter's min is the honest denominator). CuPy is
        # absent, so this is the rule-15 comparison baseline. The ratio is the
        # floor round-trip vs the NumPy CPU floor: an end-to-end host number, the
        # honest "is the GPU worth the transfer for a host array" answer.
        numpy_t = _bench(lambda: a @ b, reps, warmup)
        warm_ms = warm["min_s"] * 1e3
        cold_ms = cold_s * 1e3
        numpy_floor_ms = numpy_t["min_s"] * 1e3
        warm_gflops = gflop / warm["min_s"]
        numpy_gflops = gflop / numpy_t["min_s"]

        # The with-transfer record is its OWN shape (it does not have to match the
        # without-transfer gpu_cases keys). Two rows, cold and warm, each labeled
        # for rule 7. verified is True only because assert_matmul_close passed
        # above; the bench never publishes a verified-false series.
        common = {
            "n": n,
            "gflop": gflop,
            "numpy_cpu_f32_floor_ms": numpy_floor_ms,
            "numpy_cpu_f32_gflops": numpy_gflops,
            "reps": reps,
            "warmup": warmup,
            "label": "with-transfer",
            "verified": True,
        }
        warm_row = dict(common)
        warm_row.update(
            {
                "phase": "warm",
                "round_trip_ms": warm_ms,
                "gflops": warm_gflops,
                "ratio_vs_numpy_cpu_f32": warm_gflops / numpy_gflops,
                "cv": warm["cv"],
                "high_cv": warm.get("high_cv", False),
                "reruns": warm.get("reruns", 0),
            }
        )
        cold_row = dict(common)
        cold_row.update(
            {
                "phase": "cold",
                "round_trip_ms": cold_ms,
                "gflops": gflop / cold_s,
                "ratio_vs_numpy_cpu_f32": (gflop / cold_s) / numpy_gflops,
            }
        )
        series.append(warm_row)
        series.append(cold_row)
        print(
            f"n={n:5d}  gpu+xfer  warm {warm_ms:9.3f} ms ({warm_gflops:7.1f} GF/s)"
            f"  cold {cold_ms:9.3f} ms"
            f"  numpy-cpu-f32 floor {numpy_floor_ms:8.2f} ms"
            f"  ratio {warm_row['ratio_vs_numpy_cpu_f32']:.2f}x"
            f"  (host round-trip, transfer INCLUDED)"
        )
    return series


def run(sizes: list[int], reps: int, warmup: int) -> dict:
    # The four labeled series under their two regime keys. The without-transfer
    # regime is saved under gpu_cases (the GPU-08 source 07-03 reads as its
    # consumer contract); the with-transfer regime is its own key. Both omit
    # cleanly on a CPU-only build (each series function returns []). The
    # methodology block records the cupy-absent denominator and the timing
    # justification for each regime.
    results = {
        "manifest": _machine_manifest(),
        "benchmark": "gpu-matrix",
        "methodology": {
            "cupy_available": _CUPY_AVAILABLE,
            "cupy_absent_note": (
                "CuPy is not installed on this machine; the GPU comparison "
                "baseline is NumPy CPU f32, not CuPy."
            ),
            "without_transfer_timer": (
                "event-timed via _cuda_time_matmul (cudaEvent pair after a sync, "
                "transfer outside the timed region); never a wall-clock around an "
                "async launch."
            ),
            "with_transfer_timer": (
                "wall-clocked via the CV-gated _bench; the to_device + matmul + "
                "from_device round-trip is synchronous (from_device owns the D2H "
                "sync), the one sanctioned wall-clock GPU exception."
            ),
            "statistic": "floor(min) for the with-transfer ratio; CV is a readout/gate, not the claim.",
        },
    }
    gpu_cases = _without_transfer_series(sizes, reps, warmup)
    if gpu_cases:
        # The without-transfer regime under gpu_cases: the GPU-08 source 07-03
        # consumes, carrying the _gpu_series shape verbatim plus the cold/warm
        # label.
        results["gpu_cases"] = gpu_cases
    with_transfer = _with_transfer_series(sizes, reps, warmup)
    if with_transfer:
        results["with_transfer_cases"] = with_transfer
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    # The size grid includes non-multiples of the vector width (rule 3): a kernel
    # that only benches at tile multiples hides its tail path. The default keeps
    # 250/333/750/1000 alongside round sizes.
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[256, 333, 512, 750, 1024, 2048],
    )
    # Protocol floors (rule 5): at least 5 warmup, at least 30 measured reps.
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    if not pg.has_cuda():
        # The OFF build has no device entries; the series functions return [] and
        # the run carries only the manifest. Say so plainly rather than writing an
        # empty-series file silently.
        print(
            "has_cuda() is False (OFF build or no usable device): the GPU series "
            "are omitted. Rebuild with PG_ENABLE_CUDA=ON to measure the matrix."
        )

    _start = time.perf_counter()
    out = run(args.sizes, args.reps, args.warmup)
    print(f"gpu matrix run took {time.perf_counter() - _start:.1f}s")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.save}")
