"""The orchestrator: run every benchmark regime to a single results JSON.

benchmarks/CLAUDE.md names this file. It composes the existing regimes -- the
matmul CPU pairs + device-resident GPU slice (bench_matmul), the fused 3-op chain
CPU + GPU series (bench_fused), and the full GPU with/without-transfer x cold/warm
matrix (bench_gpu) -- into one dict with the shared manifest, so a single command
captures the whole measurement set. It reuses the protocol machinery; it does not
re-derive any of it (the manifest, the CV-gated timing core, and every series
function are imported, the bench_fused line-48 pattern).

Regime/build discipline (RESEARCH Pitfall 4): the CPU regimes (matmul CPU pairs,
the fused CPU chain) MUST run on the OFF build -- the CUDA-ON build's bare matmul
carries a size-independent per-call floor that inflates a CPU re-measure. The GPU
regimes (the device-resident slice, the four-series GPU matrix) run on the ON
build. This orchestrator runs whatever the active build exposes: on OFF the GPU
series omit cleanly (each is has_cuda()-gated), on ON every regime populates.

The size grid includes non-multiples of the vector width (250/333/750/1000)
alongside round sizes (rule 3): a kernel that only benches at tile multiples
hides its tail path.

Usage:
    python benchmarks/bench_suite.py --sizes 256 512 1024 --reps 30 --warmup 5 --save out.json
"""

from __future__ import annotations

import argparse
import json
import time

import fastmathext as fme

# Reuse, do not re-derive: the manifest and the CV-gated timing core come from
# bench_matmul (the bench_fused line-48 import pattern). The regime series
# functions are imported from each bench module so this file only orchestrates.
from bench_matmul import _machine_manifest
from bench_matmul import run as _matmul_run
from bench_fused import _cpu_chain_series, _gpu_chain_series
from bench_gpu import _with_transfer_series, _without_transfer_series

# The fused chain's default element count (FUSE-05 is the 8M-f32 chain). The
# matmul grid sizes are square n; the fused grid is element counts, a different
# axis, so the orchestrator carries its own fused-size default rather than
# reusing the square grid.
_FUSED_ELEMS = [8_000_000]


def run(
    sizes: list[int],
    reps: int,
    warmup: int,
    *,
    dtype: str = "float64",
    fused_elems: list[int] | None = None,
) -> dict:
    # One dict, the shared manifest, every regime under its own key. Each series
    # function already verifies per size and publishes verified=true only after
    # the toleranced path passes, so the assembled dict carries no verified-false
    # series (benchmarks/CLAUDE.md). The matmul regime returns CPU cases plus the
    # device-resident gpu_cases (on ON); the fused regime returns cpu_cases plus
    # gpu_cases; the GPU matrix returns the without-transfer regime under
    # gpu_cases and the with-transfer regime under its own key.
    fused_elems = fused_elems if fused_elems is not None else _FUSED_ELEMS
    results = {
        "manifest": _machine_manifest(),
        "benchmark": "suite",
        "size_grid": list(sizes),
        "fused_elems": list(fused_elems),
    }

    # The matmul regime (CPU pairs + the device-resident GPU slice). _matmul_run
    # already assembles {manifest, dtype, cases, gpu_cases?}; nest it under
    # "matmul" but drop its inner manifest (the suite carries one shared
    # manifest, rule 9, not one per regime).
    matmul = _matmul_run(sizes, dtype, reps, warmup)
    matmul.pop("manifest", None)
    results["matmul"] = matmul

    # The fused 3-op chain regime (CPU floor chain + device-resident GPU chain).
    # The GPU chain omits cleanly on the OFF build (has_cuda()-gated).
    fused = {"cpu_cases": _cpu_chain_series(fused_elems, reps, warmup)}
    fused_gpu = _gpu_chain_series(fused_elems, reps, warmup)
    if fused_gpu:
        fused["gpu_cases"] = fused_gpu
    results["fused"] = fused

    # The full GPU matrix regime: four labeled series (with/without-transfer x
    # cold/warm). The without-transfer regime sits under gpu_cases (the GPU-08
    # source 07-03 reads), the with-transfer regime under its own key. Both omit
    # cleanly on the OFF build.
    gpu_matrix = {}
    without = _without_transfer_series(sizes, reps, warmup)
    if without:
        gpu_matrix["gpu_cases"] = without
    with_transfer = _with_transfer_series(sizes, reps, warmup)
    if with_transfer:
        gpu_matrix["with_transfer_cases"] = with_transfer
    if gpu_matrix:
        results["gpu_matrix"] = gpu_matrix

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    # The default grid carries the non-multiples 250/333/750/1000 (rule 3)
    # alongside round sizes.
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 250, 256, 333, 512, 750, 1000, 1024, 2048],
    )
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument(
        "--fused-elems",
        type=int,
        nargs="+",
        default=_FUSED_ELEMS,
    )
    # Protocol floors (rule 5): at least 5 warmup, at least 30 measured reps.
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    if not fme.has_cuda():
        print(
            "has_cuda() is False (OFF build): the GPU regimes are omitted; only "
            "the CPU matmul and fused CPU chain run. Run the GPU regimes on an "
            "FME_ENABLE_CUDA=ON build."
        )

    _start = time.perf_counter()
    out = run(
        args.sizes,
        args.reps,
        args.warmup,
        dtype=args.dtype,
        fused_elems=args.fused_elems,
    )
    print(f"suite run took {time.perf_counter() - _start:.1f}s")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.save}")
