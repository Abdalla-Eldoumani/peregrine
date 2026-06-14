"""FUSE-05: the 3-op fused chain at 8M float32, >= 2x NumPy on the floor.

The chain is scaled_relu(fma3(axpby(x, y, a, b), z)) -- one fused pass per op
where NumPy materializes a temporary at every step. At a memory-bound size the
win is structural: NumPy walks the 8M-element arrays once per pass (axpby, the
multiply, the add, the maximum), our kernels fuse the arithmetic into far fewer
passes over memory. Research measured NumPy's chain floor at ~57.6 ms against a
~6 ms ideal single pass (~9.3x headroom), so 2x clears on the floor honestly.

Methodology, in order of importance (the bench-protocol skill is the law):
1. Inputs are already-constructed contiguous ndarrays, built once with a seeded
   default_rng outside the timed region. No conversion is ever billed to either
   side.
2. The CPU number is the FLOOR (min over many reps), NOT the median. A persistent
   background antivirus on this machine preempts an OpenMP worker at the
   fork/join barrier and drags the median/max of every multi-threaded CPU timing
   (verified up to 670 ms on a 3 ms kernel); the floor is the noise-free best
   case and the honest statistic, exactly as Phases 3-5 quote it. CV is reported
   as a readout only -- it WILL exceed 5% here; Phase 7 owns the CV gate.
3. The GPU number is device-resident and cudaEvent-timed via _cuda_time_fused
   (the 3-operand chain timer from plan 04): operands live on the device, the
   timed region is the chain only (no H2D/D2H), so it measures device work and is
   immune to the CPU/AV noise.
4. The CPU and GPU series time the IDENTICAL chain composition (the same three
   ops and the same scalar constants A/B/SCALE below), so the two recorded ratios
   describe the same computation on different backends.
5. Every series is verified against the unfused NumPy chain through
   assert_fused_close (the single fused toleranced path) before any time is
   recorded. A series with verified false never reaches a results file.

Usage:
    python benchmarks/bench_fused.py --sizes 8000000 --reps 30 --warmup 5 --save out.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import fastmathext as fme

# Reuse the matmul harness rather than re-deriving the protocol: the manifest
# capture, the warmup/reps timing core (which records min_s, the floor this bench
# quotes), and the --save shape are identical honest-measurement machinery. A
# fused-specific copy would be a second place the protocol could drift.
from bench_matmul import _bench, _machine_manifest

# The single fused toleranced path lives in tests/conftest.py; bench_matmul put
# tests/ on sys.path at import for assert_matmul_close, so the same import works
# here. Routing per-run verification through assert_fused_close keeps the bench on
# the one sanctioned tolerance contract -- an inline rtol is exactly what
# bench-protocol rule 14 forbids (a bench that loosens tolerance to pass lies).
from conftest import assert_fused_close

# The fixed chain constants. These MUST match the native device chain timer
# (src/cuda/fused.cu time_fused_chain uses a=2, b=3, scale=1) so the CPU floor
# series and the GPU device series time the same arithmetic. They are also all
# POSITIVE, with strictly-positive operands built below, on purpose: with mixed
# signs a*x+b*y / t*y+z catastrophically cancels at some elements, where the
# single-rounding fused result equals the f64 truth EXACTLY while unfused NumPy
# carries ~1 ULP, and the design-doc fused bound (rtol*|ref| + subnormal floor,
# no operand-magnitude atol) goes vacuous as |ref|->0 and would reject the MORE
# accurate kernel. Positive operands keep |ref| at operand scale (the same gotcha
# the fused oracle/property tests document); do NOT recenter the operands to zero.
A = 2.0
B = 3.0
SCALE = 1.0


def _numpy_chain(x, y, z):
    # The unfused reference: three temp-materializing passes, the exact
    # composition the fused kernels collapse. Step for step it mirrors the native
    # device chain (axpby -> fma3(t, y, z) = t*y + z -> scaled_relu): fma3's middle
    # step reuses y and z as the multiply and add operands, NOT a fresh pair, so
    # the CPU/NumPy reference computes t*y + z to match the GPU timer exactly.
    t = A * x + B * y
    t = t * y + z
    return np.maximum(SCALE * t, 0.0)


def _fme_chain(x, y, z):
    # The fused composition: scaled_relu(fma3(axpby(x, y, a, b), z)). fma3 is the
    # true 3-operand x*y + z, so the middle call passes (t, y, z) -> t*y + z,
    # reusing y and z exactly as _numpy_chain and the device timer do. Each op is
    # one fused pass; NumPy's _numpy_chain above materializes a temporary at each
    # arithmetic step.
    t = fme.axpby(x, y, a=A, b=B)
    t = fme.fma3(t, y, z)
    return fme.scaled_relu(t, scale=SCALE)


def _make_operands(elems: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 8M f32 operands as a near-square 2-D array (the library is 2-D only). Built
    # once, seeded, outside any timed region. STRICTLY POSITIVE (the +1.0 shift on
    # a standard normal's magnitude) so the fused tolerance bound stays
    # non-vacuous -- see the constant block above. A near-square factorization
    # keeps the shape honest (a 1xN row would hide any row-loop overhead); fall
    # back to (1, elems) only for a prime element count.
    side = int(round(elems**0.5))
    while side > 1 and elems % side != 0:
        side -= 1
    rows, cols = (side, elems // side) if side > 1 else (1, elems)
    rng = np.random.default_rng(0)
    x = (np.abs(rng.standard_normal((rows, cols))) + 1.0).astype(np.float32)
    y = (np.abs(rng.standard_normal((rows, cols))) + 1.0).astype(np.float32)
    z = (np.abs(rng.standard_normal((rows, cols))) + 1.0).astype(np.float32)
    return x, y, z


def _cpu_chain_series(sizes: list[int], reps: int, warmup: int) -> list[dict]:
    # The CPU FUSE-05 series, quoted on the FLOOR. Both the fused chain and the
    # NumPy chain are timed through _bench; the speedup is numpy_min / fme_min (the
    # floor ratio), NOT a median ratio -- the AV-vs-OpenMP noise makes the median
    # meaningless on this machine. Verified ONCE per size before timing via
    # assert_fused_close against the unfused NumPy chain; a verified-false series
    # is never recorded. Run this on the OFF build (CONTEXT): no _cuda_time_fused
    # reference is reachable here, so the script imports and runs CPU-only clean.
    series = []
    for elems in sizes:
        x, y, z = _make_operands(elems)

        # Verify on the single sanctioned toleranced path before any time is
        # recorded (bench-protocol rule 11: fast and wrong is just wrong). The
        # fused chain result is compared to the unfused NumPy chain in the same
        # f32 dtype; the single-rounding fma3 stays within the two-sided bound.
        got = _fme_chain(x, y, z)
        assert_fused_close(np.asarray(got), _numpy_chain(x, y, z))

        fme_t = _bench(lambda: _fme_chain(x, y, z), reps, warmup)
        numpy_t = _bench(lambda: _numpy_chain(x, y, z), reps, warmup)
        # The headline statistic is the floor ratio. min_s is the noise-free best
        # case for both contenders; quoting it (not the median) is the per-call-
        # floor verdict the research locked for this exact machine.
        speedup_floor = numpy_t["min_s"] / fme_t["min_s"]
        case = {
            "elems": elems,
            "shape": list(x.shape),
            "dtype": "float32",
            "chain": "scaled_relu(fma3(axpby(x,y,a,b),z))",
            "constants": {"a": A, "b": B, "scale": SCALE},
            "fastmathext": fme_t,
            "numpy": numpy_t,
            # The quoted FUSE-05 CPU number: the floor (min) ratio. The median
            # ratio is recorded too, but only as a noise readout -- it is NOT the
            # claim (Pitfall 2: the median is dragged by AV-vs-OpenMP preemption).
            "speedup_floor": speedup_floor,
            "speedup_median_readout": numpy_t["median_s"] / fme_t["median_s"],
            "statistic": "floor(min)",
            "floor_caveat": (
                "CPU quoted on the floor(min): a background antivirus preempts an "
                "OpenMP worker at the barrier and inflates the median/max on this "
                "machine; the floor is the noise-free best case (the per-call-floor "
                "verdict). CV exceeds 5% by design and is a readout, not a gate."
            ),
            # True only because assert_fused_close passed above; the bench never
            # publishes a verified-false series (benchmarks/CLAUDE.md).
            "verified": True,
        }
        series.append(case)
        print(
            f"elems={elems:>9d}  cpu  fme floor {fme_t['min_s']*1e3:8.2f} ms"
            f"  numpy floor {numpy_t['min_s']*1e3:8.2f} ms"
            f"  speedup {speedup_floor:5.2f}x (floor)"
            f"  cv {fme_t['cv']*100:5.1f}%/{numpy_t['cv']*100:5.1f}% (readout)"
        )
    return series


def _gpu_chain_series(sizes: list[int], reps: int, warmup: int) -> list[dict]:
    # The device-resident GPU FUSE-05 series, cudaEvent-timed. Gated on
    # has_cuda() so a CPU-only build (or a machine without a usable device) omits
    # it and NO fme._core._cuda_time_fused reference is ever reached on the OFF
    # build -- that entry does not exist there. This mirrors bench_matmul._gpu_series,
    # which returns [] before touching _cuda_time_matmul. Inside the gate: operands
    # are device-resident (to_device ONCE, outside any timed region), verified ONCE
    # before timing, then the chain is event-timed transfer-free via the plan-04
    # 3-operand chain timer. Same chain + same constants as the CPU series above.
    if not fme.has_cuda():
        return []
    series = []
    for elems in sizes:
        x, y, z = _make_operands(elems)

        xd = fme.to_device(x)
        yd = fme.to_device(y)
        zd = fme.to_device(z)

        # Verify ONCE before any timing, on the single sanctioned toleranced path:
        # run the fused chain device-resident, from_device the result, and compare
        # to the unfused NumPy chain. The device kernels match the same oracle the
        # CPU path does, so this is the same assert_fused_close bound -- no inline
        # rtol, no loosened tolerance.
        dt = fme.axpby(xd, yd, a=A, b=B)
        dt = fme.fma3(dt, yd, zd)
        dout = fme.scaled_relu(dt, scale=SCALE)
        got = fme.from_device(dout)
        assert_fused_close(got, _numpy_chain(x, y, z))
        # Release the device handles the verify pass created before timing so they
        # do not straggle to interpreter finalization (the 06-04 nanobind-leak
        # lesson: bind device Arrays to locals and drop them deterministically).
        del dt, dout, got

        # The timed region is the device chain only (no transfer). _cuda_time_fused
        # runs warmups then reps of axpby->fma3->scaled_relu on compute-stream
        # scratch inside a cudaEvent pair and returns warm ms PER REP for the whole
        # chain. Device-side timing is immune to the CPU/AV noise (RESEARCH).
        warm_ms = fme._core._cuda_time_fused(xd, yd, zd, reps, warmup)
        # Cold: a single event-timed chain before the clocks warm.
        cold_ms = fme._core._cuda_time_fused(xd, yd, zd, 1, 0)
        # NumPy CPU f32 chain on the wall-clock floor: a CPU chain is synchronous,
        # so perf_counter's min is the honest denominator. The device-resident
        # ratio is warm device chain ms vs the NumPy CPU chain floor ms, the same
        # f32 chain on both sides.
        numpy_t = _bench(lambda: _numpy_chain(x, y, z), reps, warmup)
        numpy_floor_ms = numpy_t["min_s"] * 1e3
        # Bandwidth sanity (the bench-lies tripwire): the chain touches roughly
        # 3 input + 1 output streams of 4*elems bytes over the 3 passes; a number
        # implying far past the ~300 GB/s effective bound means the timer is wrong.
        approx_bytes = 4.0 * elems * 4.0  # ~4 array-passes of f32 over the chain
        gbps = approx_bytes / (warm_ms / 1e3) / 1e9
        case = {
            "elems": elems,
            "shape": list(x.shape),
            "dtype": "float32",
            "chain": "scaled_relu(fma3(axpby(x,y,a,b),z))",
            "constants": {"a": A, "b": B, "scale": SCALE},
            "cold_ms": cold_ms,
            "warm_ms": warm_ms,
            "numpy_cpu_f32_floor_ms": numpy_floor_ms,
            "approx_effective_gbps": gbps,
            # Device chain vs NumPy CPU chain floor, the same 3-op chain both sides.
            "ratio_vs_numpy_cpu_f32": numpy_floor_ms / warm_ms,
            "reps": reps,
            "warmup": warmup,
            # True only because assert_fused_close passed above.
            "verified": True,
        }
        series.append(case)
        print(
            f"elems={elems:>9d}  gpu  warm {warm_ms:8.3f} ms"
            f"  cold {cold_ms:8.3f} ms"
            f"  numpy-cpu-f32 floor {numpy_floor_ms:8.2f} ms"
            f"  ratio {case['ratio_vs_numpy_cpu_f32']:5.1f}x"
            f"  (~{gbps:.0f} GB/s, device-resident, transfer excluded)"
        )
        # Release the operand device handles before the next size's to_device so a
        # multi-size sweep does not hold three prior-iteration buffers (and the
        # final iteration's operands do not straggle to interpreter finalization,
        # the 06-04 nanobind leaked-instance warning the verify temporaries above
        # are already dropped to avoid).
        del xd, yd, zd
    return series


def run(sizes: list[int], reps: int, warmup: int) -> dict:
    results = {
        "manifest": _machine_manifest(),
        "benchmark": "fuse-05",
        "chain": "scaled_relu(fma3(axpby(x,y,a,b),z))",
        "constants": {"a": A, "b": B, "scale": SCALE},
        "cpu_cases": _cpu_chain_series(sizes, reps, warmup),
    }
    # The device-resident GPU chain series (the FUSE-05 GPU number) runs over the
    # same sizes, event-timed and transfer-free; omitted cleanly on a CPU-only
    # build or a machine without a usable device. Reported under its own key so the
    # CPU floor cases are never conflated with device-resident GPU numbers.
    gpu_cases = _gpu_chain_series(sizes, reps, warmup)
    if gpu_cases:
        results["gpu_cases"] = gpu_cases
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    # FUSE-05 is the 8M-f32 chain; the size grid defaults to that one point but
    # accepts more for exploration. Sizes are element counts (the chain is 1-D in
    # spirit; _make_operands factors each into a near-square 2-D array).
    p.add_argument("--sizes", type=int, nargs="+", default=[8_000_000])
    # Protocol floors (bench-protocol rule 5): at least 5 warmup, at least 30
    # measured reps. The defaults sit at the floor.
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    out = run(args.sizes, args.reps, args.warmup)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.save}")
