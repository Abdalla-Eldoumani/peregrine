"""The measurement matrix: every cell the project quotes, run under one protocol.

A cell is (op, dtype, shape, layout, variant, threads, seed). Each cell builds
its inputs once from a seeded default_rng, hands the same input objects to
every contender (peregrine, numpy, and for the fused ops an allocation-free
NumPy sequence too), and runs bench_matmul.run_cell: interleaved ABBA blocks,
one shared batch size, bitwise verification of every sample, the CV gate with
a fresh-process last chance, and clock discards. One JSON per invocation.

Work is selected by --group so a long matrix splits into background windows
well under two hours each; --list prints a selection's cells and an estimated
duration without running anything.

Usage:
    python benchmarks/bench_suite.py --group quick --out benchmarks/results/local/v3.1/quick.json
    python benchmarks/bench_suite.py --group matmul-square --list
    OMP_NUM_THREADS=1 python benchmarks/bench_suite.py --group gemm-1t
    python benchmarks/bench_suite.py --probe-spin
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np

import peregrine as pg

from bench_matmul import (
    PEREGRINE,
    Contender,
    PdhClock,
    Protocol,
    batch_size,
    bitwise_equal,
    pinned_threads,
    protocol_manifest,
    redact_paths,
    run_cell,
)
# The device chain timer hard-codes these constants, so the NumPy chain the
# device chain is compared with must use the same ones.
from bench_fused import A as CHAIN_A, B as CHAIN_B, SCALE as CHAIN_SCALE, _numpy_chain
from conftest import assert_fused_close, assert_matmul_close, assert_reduce_close

HEADLINE_THREADS = 6
# A cell that keeps the CPU busy this long has spent the laptop's short-term
# turbo budget (about 28 s at PL2 on the i7-10750H), so the next cell waits out
# a cooldown as if a new series started.
HEAVY_CELL_S = 10.0
RESULTS_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "local", "v3.1")
DTYPES = ("float32", "float64")

# Square sizes mix tile multiples with 250/333/750/1000/1500/3000 so the
# kernel's tail paths are always on the grid.
SQUARE = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 250, 256, 333, 384,
          512, 640, 750, 1000, 1024, 1500, 2048, 3000, 4096)
RECT = ((1, 4096, 4096), (4096, 4096, 1), (4, 4096, 4096), (4096, 4096, 4),
        (4096, 64, 4096), (64, 4096, 64), (4096, 256, 256), (256, 4096, 256),
        (256, 256, 4096), (65536, 32, 32), (1000, 333, 750))
LAYOUT_SHAPES = ((64, 64, 64), (512, 512, 512), (2048, 2048, 2048), (1000, 333, 750), (4096, 256, 256))
# c: both C-contiguous; at_b: a is the transposed view a.T; a_bt: b is b.T;
# ff: both operands F-contiguous.
LAYOUTS = ("c", "at_b", "a_bt", "ff")
TRANSPOSE = tuple((n, n) for n in (8, 32, 64, 128, 256, 512, 1024, 2048, 4096)) + ((1000, 3000), (4096, 64))
REDUCE = ((8, 8), (64, 64), (256, 256), (1024, 1024), (4096, 4096), (16, 1000000), (1000000, 16))
AXES = ("axis=None", "axis=0", "axis=1")
FUSED_OPS = ("axpby", "fma3", "scaled_relu")
FUSED = ((8, 8), (32, 32), (128, 128), (512, 512), (1024, 1024), (2896, 2896), (5793, 5793))
GEMM_1T = (64, 128, 256, 512, 1024, 2048)
GPU_DEVICE = (256, 512, 1024, 2048, 4096)
GPU_AUTO = tuple((n, n, n) for n in (128, 256, 512, 1024, 2048, 4096)) + ((1, 4096, 4096), (4096, 64, 4096))
# 1M, 4M, 16M and 32M elements.
GPU_ELEMS = ((1024, 1024), (2048, 2048), (4096, 4096), (4096, 8192))
GPU_OPS = ("gpu_matmul", "to_device", "from_device", "gpu_chain")
GROUPS = ("quick", "overhead", "matmul-square", "matmul-rect", "matmul-layout",
          "transpose", "reduce", "fused", "gemm-1t", "gpu")


@dataclass(frozen=True)
class Cell:
    group: str
    op: str
    dtype: str
    shape: tuple
    layout: str = "c"
    variant: str = "-"
    threads: int = HEADLINE_THREADS
    seed: int = 0

    @property
    def id(self) -> str:
        # The group is left out on purpose: the same cell reached from two
        # groups gets the same id, hence the same ABBA start.
        dims = "x".join(str(d) for d in self.shape)
        return f"{self.op}/{self.dtype}/{dims}/{self.layout}/{self.variant}/t{self.threads}/s{self.seed}"

    @property
    def series(self) -> tuple:
        return (self.group, self.op, self.dtype, self.layout, self.variant, self.threads)

    def spec(self) -> dict:
        d = dataclasses.asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_spec(cls, d: dict) -> "Cell":
        d = dict(d)
        d["shape"] = tuple(d["shape"])
        return cls(**d)


def select(groups: list[str], seeds: list[int], cuda: bool) -> list[Cell]:
    cells: list[Cell] = []
    for group in groups:
        def add(op, dtype, shape, **kw):
            cells.extend(Cell(group, op, dtype, tuple(shape), seed=s, **kw) for s in seeds)

        if group == "quick":
            for dt in DTYPES:
                for n in (4, 16, 64, 256, 1024, 2048):
                    add("matmul", dt, (n, n, n))
                add("matmul", dt, (1, 4096, 4096))
                add("matmul", dt, (512, 512, 512), layout="at_b")
                for n in (64, 2048):
                    add("transpose", dt, (n, n))
                for shape in ((64, 64), (4096, 4096)):
                    for axis in ("axis=None", "axis=0"):
                        add("sum", dt, shape, variant=axis)
                for shape in ((32, 32), (2896, 2896)):
                    for v in ("alloc", "out"):
                        add("axpby", dt, shape, variant=v)
            if cuda:
                for n in (256, 2048):
                    add("matmul", "float32", (n, n, n), variant="auto")
        elif group == "overhead":
            # Per-call cost: sizes where the work itself is a few nanoseconds.
            for dt in DTYPES:
                for n in (1, 2, 4, 8):
                    add("matmul", dt, (n, n, n))
                add("transpose", dt, (8, 8))
                for op in ("sum", "mean"):
                    for axis in AXES:
                        add(op, dt, (8, 8), variant=axis)
                for op in FUSED_OPS:
                    for v in ("alloc", "out"):
                        add(op, dt, (8, 8), variant=v)
        elif group == "matmul-square":
            for dt in DTYPES:
                for n in SQUARE:
                    add("matmul", dt, (n, n, n))
        elif group == "matmul-rect":
            for dt in DTYPES:
                for shape in RECT:
                    add("matmul", dt, shape)
        elif group == "matmul-layout":
            for dt in DTYPES:
                for lay in LAYOUTS:
                    for shape in LAYOUT_SHAPES:
                        add("matmul", dt, shape, layout=lay)
        elif group == "transpose":
            for dt in DTYPES:
                for shape in TRANSPOSE:
                    add("transpose", dt, shape)
        elif group == "reduce":
            for op in ("sum", "mean"):
                for dt in DTYPES:
                    for axis in AXES:
                        for shape in REDUCE:
                            add(op, dt, shape, variant=axis)
        elif group == "fused":
            for op in FUSED_OPS:
                for dt in DTYPES:
                    for v in ("alloc", "out"):
                        for shape in FUSED:
                            add(op, dt, shape, variant=v)
        elif group == "gemm-1t":
            for dt in DTYPES:
                for n in GEMM_1T:
                    add("matmul", dt, (n, n, n), threads=1)
        elif group == "gpu":
            for dt in DTYPES:
                for n in GPU_DEVICE:
                    add("gpu_matmul", dt, (n, n, n), variant="device")
            for shape in GPU_AUTO:
                add("matmul", "float32", shape, variant="auto")
            for op in ("to_device", "from_device"):
                for shape in GPU_ELEMS:
                    add(op, "float32", shape)
            for shape in GPU_ELEMS:
                add("gpu_chain", "float32", shape, variant="device")
        else:
            raise ValueError(f"unknown group {group!r}")
    return cells


def needs_cuda(cell: Cell) -> bool:
    return cell.op in GPU_OPS or cell.variant == "auto"


def unsupported(cell: Cell) -> str | None:
    if cell.op in FUSED_OPS and cell.variant == "out":
        fn = getattr(pg, cell.op)
        if "out" not in inspect.signature(fn).parameters:
            return f"pg.{cell.op} has no out= parameter in this build"
    return None


# --- Builders: inputs once per seed, contenders over the same objects. ---


def _normal(rng, shape, dt):
    return rng.standard_normal(shape).astype(dt)


def _positive(rng, shape, dt):
    # The fused suite's distribution: strictly positive operands with positive
    # coefficients never cancel, so the elementwise relative bound stays
    # meaningful (with mixed signs |ref| collapses toward zero and the bound
    # rejects the more accurate fused result).
    return rng.standard_normal(shape).astype(dt) ** 2 + dt.type(0.5)


def _build_matmul(cell: Cell):
    m, k, n = cell.shape
    dt = np.dtype(cell.dtype)
    rng = np.random.default_rng(cell.seed)
    a = _normal(rng, (k, m), dt).T if cell.layout == "at_b" else _normal(rng, (m, k), dt)
    b = _normal(rng, (n, k), dt).T if cell.layout == "a_bt" else _normal(rng, (k, n), dt)
    if cell.layout == "ff":
        a, b = np.asfortranarray(a), np.asfortranarray(b)
    contenders = [
        Contender(PEREGRINE, lambda: pg.matmul(a, b)),
        Contender("numpy", lambda: np.matmul(a, b)),
    ]

    def check(refs):
        assert_matmul_close(refs[PEREGRINE], refs["numpy"], a, b)

    meta = {"flop": 2 * m * k * n}
    if cell.variant == "auto" and hasattr(pg, "_host_backend_for"):
        meta["route"] = pg._host_backend_for(a, b)
    return contenders, check, meta


def _build_transpose(cell: Cell):
    dt = np.dtype(cell.dtype)
    a = _normal(np.random.default_rng(cell.seed), cell.shape, dt)
    # pg.transpose returns an owned C-contiguous copy, so the fair NumPy
    # counterpart materializes one too; a.T alone is a free view.
    contenders = [
        Contender(PEREGRINE, lambda: pg.transpose(a)),
        Contender("numpy", lambda: np.ascontiguousarray(a.T)),
    ]

    def check(refs):
        # Transposition moves values without arithmetic, so the oracle is exact.
        if not bitwise_equal(refs[PEREGRINE], refs["numpy"]):
            raise AssertionError("pg.transpose differs from np.ascontiguousarray(a.T)")

    return contenders, check, {"bytes": 2 * a.nbytes}


def _build_reduce(cell: Cell):
    dt = np.dtype(cell.dtype)
    a = _normal(np.random.default_rng(cell.seed), cell.shape, dt)
    axis = {"axis=None": None, "axis=0": 0, "axis=1": 1}[cell.variant]
    pg_fn, np_fn = (pg.sum, np.sum) if cell.op == "sum" else (pg.mean, np.mean)
    contenders = [
        Contender(PEREGRINE, lambda: pg_fn(a, axis=axis)),
        Contender("numpy", lambda: np_fn(a, axis=axis)),
    ]

    def check(refs):
        assert_reduce_close(refs[PEREGRINE], refs["numpy"], a, axis, kind=cell.op)

    return contenders, check, {"bytes": a.nbytes}


def _build_fused(cell: Cell):
    dt = np.dtype(cell.dtype)
    rng = np.random.default_rng(cell.seed)
    n_in = {"axpby": 2, "fma3": 3, "scaled_relu": 1}[cell.op]
    ops = [_positive(rng, cell.shape, dt) for _ in range(n_in)]
    x = ops[0]
    o = np.empty_like(x)
    po = np.empty_like(x)
    # The suite's coefficients: axpby a=2, b=3; scaled_relu scale=3.
    two, three, zero = dt.type(2.0), dt.type(3.0), dt.type(0.0)
    use_out = cell.variant == "out"

    if cell.op == "axpby":
        y = ops[1]
        t = np.empty_like(x)

        def pg_call():
            if use_out:
                pg.axpby(x, y, a=2.0, b=3.0, out=po)
                return po
            return pg.axpby(x, y, a=2.0, b=3.0)

        def idiom():
            return two * x + three * y

        def noalloc():
            np.multiply(x, two, out=o)
            np.multiply(y, three, out=t)
            np.add(o, t, out=o)
            return o
    elif cell.op == "fma3":
        y, z = ops[1], ops[2]

        def pg_call():
            if use_out:
                pg.fma3(x, y, z, out=po)
                return po
            return pg.fma3(x, y, z)

        def idiom():
            return x * y + z

        def noalloc():
            np.multiply(x, y, out=o)
            np.add(o, z, out=o)
            return o
    else:

        def pg_call():
            if use_out:
                pg.scaled_relu(x, scale=3.0, out=po)
                return po
            return pg.scaled_relu(x, scale=3.0)

        def idiom():
            return np.maximum(three * x, zero)

        def noalloc():
            np.multiply(x, three, out=o)
            np.maximum(o, zero, out=o)
            return o

    contenders = [
        Contender(PEREGRINE, pg_call),
        Contender("numpy", idiom),
        Contender("numpy_noalloc", noalloc),
    ]

    def check(refs):
        assert_fused_close(refs[PEREGRINE], refs["numpy"])
        # The buffered sequence must be the same roundings as the expression,
        # or the second NumPy contender would be timing a different computation.
        if not bitwise_equal(refs["numpy_noalloc"], refs["numpy"]):
            raise AssertionError("allocation-free NumPy sequence differs from the expression")

    return contenders, check, {"bytes": (n_in + 1) * x.nbytes}


def _build_gpu_matmul(cell: Cell):
    m, k, n = cell.shape
    dt = np.dtype(cell.dtype)
    rng = np.random.default_rng(cell.seed)
    a, b = _normal(rng, (m, k), dt), _normal(rng, (k, n), dt)
    xa, xb = pg.to_device(a), pg.to_device(b)
    # Wall clock around an async launch measures the launch, so the device side
    # is timed by cudaEvents inside _cuda_time_matmul (transfer-free); fn is
    # only the untimed verification path.
    contenders = [
        Contender(
            PEREGRINE,
            lambda: pg.from_device(pg.matmul(xa, xb)),
            timer=lambda kk: pg._core._cuda_time_matmul(xa, xb, kk, 0) / 1e3,
        ),
        Contender("numpy", lambda: np.matmul(a, b)),
    ]

    def check(refs):
        assert_matmul_close(refs[PEREGRINE], refs["numpy"], a, b)

    return contenders, check, {"flop": 2 * m * k * n, "timer": "cudaEvent"}


def _build_transfer(cell: Cell):
    dt = np.dtype(cell.dtype)
    a = _normal(np.random.default_rng(cell.seed), cell.shape, dt)
    if cell.op == "to_device":
        c = Contender(PEREGRINE, lambda: pg.to_device(a), to_host=pg.from_device)
    else:
        xd = pg.to_device(a)
        c = Contender(PEREGRINE, lambda: pg.from_device(xd))

    def check(refs):
        if not bitwise_equal(refs[PEREGRINE], a):
            raise AssertionError(f"{cell.op} round trip is not exact")

    # Both calls synchronize before returning, so wall clock is honest here.
    return [c], check, {"bytes": a.nbytes}


def _build_gpu_chain(cell: Cell):
    dt = np.dtype(cell.dtype)
    rng = np.random.default_rng(cell.seed)
    x, y, z = (_positive(rng, cell.shape, dt) for _ in range(3))
    xd, yd, zd = pg.to_device(x), pg.to_device(y), pg.to_device(z)

    def device_chain():
        t = pg.axpby(xd, yd, a=CHAIN_A, b=CHAIN_B)
        t = pg.fma3(t, yd, zd)
        return pg.from_device(pg.scaled_relu(t, scale=CHAIN_SCALE))

    contenders = [
        Contender(
            PEREGRINE,
            device_chain,
            timer=lambda kk: pg._core._cuda_time_fused(xd, yd, zd, kk, 0) / 1e3,
        ),
        Contender("numpy", lambda: _numpy_chain(x, y, z)),
    ]

    def check(refs):
        assert_fused_close(refs[PEREGRINE], refs["numpy"])

    return contenders, check, {"timer": "cudaEvent"}


BUILDERS = {
    "matmul": _build_matmul,
    "transpose": _build_transpose,
    "sum": _build_reduce,
    "mean": _build_reduce,
    "axpby": _build_fused,
    "fma3": _build_fused,
    "scaled_relu": _build_fused,
    "gpu_matmul": _build_gpu_matmul,
    "to_device": _build_transfer,
    "from_device": _build_transfer,
    "gpu_chain": _build_gpu_chain,
}


# The oracle each op's first peregrine output is checked against: a conftest
# helper, or exact equality for ops that move values without arithmetic.
ORACLE = {
    "matmul": "assert_matmul_close",
    "gpu_matmul": "assert_matmul_close",
    "sum": "assert_reduce_close",
    "mean": "assert_reduce_close",
    "axpby": "assert_fused_close",
    "fma3": "assert_fused_close",
    "scaled_relu": "assert_fused_close",
    "gpu_chain": "assert_fused_close",
    "transpose": "exact",
    "to_device": "exact",
    "from_device": "exact",
}


def _base(cell: Cell) -> dict:
    return {
        "id": cell.id,
        "group": cell.group,
        "op": cell.op,
        "dtype": cell.dtype,
        "shape": list(cell.shape),
        "layout": cell.layout,
        "variant": cell.variant,
        "threads": cell.threads,
        "seed": cell.seed,
        "oracle": ORACLE[cell.op],
        "strikes": 0,
    }


def run_one(cell: Cell, proto: Protocol, *, sleep=time.sleep, clock=None, fresh_runner=None, log=print) -> dict:
    base = _base(cell)
    if needs_cuda(cell) and not pg.has_cuda():
        return {**base, "status": "skipped", "verified": False, "reason": "no usable CUDA device or a CPU-only build"}
    why = unsupported(cell)
    if why:
        return {**base, "status": "unsupported", "verified": False, "reason": why}
    meta = {}
    with pinned_threads(cell.threads) as observed:
        try:
            contenders, check, meta = BUILDERS[cell.op](cell)
            fresh = fresh_runner(cell, proto) if fresh_runner is not None else None
            res = run_cell(cell.id, contenders, check, proto, sleep=sleep, clock=clock, fresh=fresh, log=log)
        except Exception as e:
            # An op that raises cannot be verified; it fails the run like a
            # wrong answer would.
            res = {"status": "verify_failed", "verified": False, "reason": f"error: {type(e).__name__}: {e}"}
    out = {**base, **res, "threads_observed": observed}
    for key in ("route", "timer"):
        if key in meta:
            out[key] = meta[key]
    if out["status"] == "ok":
        for key, unit in (("flop", "gflops"), ("bytes", "gbps")):
            if key in meta:
                out[unit] = {n: meta[key] / s["median_s"] / 1e9 for n, s in out["contenders"].items()}
    return out


def _proto_argv(proto: Protocol, smoke: bool) -> list[str]:
    a = [
        "--samples", str(proto.samples),
        "--warmups", str(proto.warmups),
        "--switch-gap-ms", repr(proto.switch_gap_s * 1e3),
        "--rewarm-ms", repr(proto.rewarm_s * 1e3),
        "--batch-target-us", repr(proto.batch_target_s * 1e6),
        "--verify-mode", proto.verify_mode,
    ]
    if proto.clock_floor is not None:
        a += ["--clock-floor", repr(proto.clock_floor)]
    if proto.clock_ceiling is not None:
        a += ["--clock-ceiling", repr(proto.clock_ceiling)]
    if smoke:
        a.append("--smoke")
    return a


_AFFINITY = None


def _physical_core_mask() -> int | None:
    # The first logical processor of every physical core, from the OS topology.
    # Windows otherwise parks two of six workers on SMT siblings for seconds at
    # a time: 6-thread GEMM CV here fell from 6 to 8 percent to 1 to 2 percent
    # for Peregrine and roughly halved for OpenBLAS once both were confined.
    if os.name != "nt":
        return None
    import ctypes

    class Info(ctypes.Structure):
        _fields_ = [("mask", ctypes.c_size_t), ("relationship", ctypes.c_uint32),
                    ("_union", ctypes.c_ubyte * 16)]

    k32 = ctypes.windll.kernel32
    size = ctypes.c_uint32(0)
    k32.GetLogicalProcessorInformation(None, ctypes.byref(size))
    buf = (Info * (size.value // ctypes.sizeof(Info)))()
    if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(size)):
        return None
    mask = 0
    for info in buf:
        if info.relationship == 0 and info.mask:  # RelationProcessorCore
            mask |= info.mask & -info.mask
    return mask or None


def _confine_to_physical_cores() -> str | None:
    mask = _physical_core_mask()
    if mask is None:
        return None
    import ctypes

    k32 = ctypes.windll.kernel32
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return hex(mask) if k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask) else None


def fresh_process_runner(smoke: bool):
    """The last chance for a noisy cell: this script re-invoked for that cell alone."""

    def make(cell: Cell, proto: Protocol):
        def run() -> dict:
            with tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, "cell.json")
                cmd = [
                    sys.executable, os.path.abspath(__file__),
                    "--cell", json.dumps(cell.spec()),
                    "--out", out, "--fresh-child",
                    *_proto_argv(proto, smoke),
                    *(["--all-logical-cpus"] if _AFFINITY is None else []),
                ]
                # The child inherits the environment so its OpenMP runtime
                # starts with the same OMP_NUM_THREADS as this process did.
                p = subprocess.run(cmd, env=dict(os.environ), capture_output=True, text=True, timeout=7200)
                for line in p.stdout.splitlines():
                    print(f"  fresh process: {line}")
                if not os.path.exists(out):
                    raise RuntimeError(f"exit {p.returncode}: {p.stderr.strip()[-800:]}")
                with open(out) as f:
                    return json.load(f)["cells"][0]

        return run

    return make


# --- Duration estimate for --list. ---
# Planning rates for this laptop (i7-10750H, RTX 3060 Laptop), rounded down from
# the committed v3.0.0 curves; they size background windows, they are not
# results.
_CPU_GEMM = {(6, "float64"): 150e9, (6, "float32"): 300e9, (1, "float64"): 40e9, (1, "float32"): 80e9}
_GPU_GEMM = {"float32": 5e12, "float64": 0.15e12}
_MEM_BW = 8e9
_PCIE = 10e9
_CALL = 3e-6


def _est_per_call(cell: Cell) -> dict:
    item = np.dtype(cell.dtype).itemsize
    if cell.op in ("matmul", "gpu_matmul"):
        m, k, n = cell.shape
        cpu = 2 * m * k * n / _CPU_GEMM[(cell.threads, cell.dtype)] + _CALL
        if cell.op == "gpu_matmul":
            return {PEREGRINE: 2 * m * k * n / _GPU_GEMM[cell.dtype] + 10e-6, "numpy": cpu}
        return {PEREGRINE: cpu, "numpy": cpu}
    nbytes = math.prod(cell.shape) * item
    if cell.op == "transpose":
        t = 2 * nbytes / _MEM_BW + _CALL
        return {PEREGRINE: t, "numpy": t}
    if cell.op in ("sum", "mean"):
        t = nbytes / _MEM_BW + _CALL
        return {PEREGRINE: t, "numpy": t}
    if cell.op in FUSED_OPS:
        t = 3 * nbytes / _MEM_BW + _CALL
        return {PEREGRINE: t, "numpy": 2 * t, "numpy_noalloc": 1.5 * t}
    if cell.op in ("to_device", "from_device"):
        return {PEREGRINE: nbytes / _PCIE + 20e-6}
    # gpu_chain: three device passes against roughly ten NumPy array passes.
    return {PEREGRINE: 12 * nbytes / 300e9, "numpy": 10 * nbytes / _MEM_BW}


def estimate_cell_s(cell: Cell, proto: Protocol, clock_on: bool) -> float:
    if needs_cuda(cell) and not pg.has_cuda() or unsupported(cell):
        return 0.0
    pcs = _est_per_call(cell)
    K = batch_size(pcs, proto.batch_target_s)
    sample = {n: pc * K for n, pc in pcs.items()}
    group = {n: max(1, math.ceil(proto.clock_group_s / s)) if clock_on else 1 for n, s in sample.items()}
    block = max(g * math.ceil(proto.block / g) for g in group.values())
    rounds = max(2, math.ceil(proto.samples / block))
    rounds += rounds % 2
    measured = sum(s * (rounds * block + proto.warmups + 3) for s in sample.values())
    switches = (len(pcs) - 1) * rounds + 1
    oracle = 0.0
    if cell.op in ("matmul", "gpu_matmul") and cell.dtype == "float32":
        m, k, n = cell.shape
        oracle = 2 * m * k * n / _CPU_GEMM[(HEADLINE_THREADS, "float64")]
    return measured + switches * proto.switch_gap_s + oracle


def series_order(cells: list[Cell], seed: int) -> list[list[Cell]]:
    by: dict = {}
    for c in cells:
        by.setdefault(c.series, []).append(c)
    series = list(by.values())
    # Randomized once per run so no series always runs on a cold (or a hot)
    # chip; the seed is recorded in the manifest so the order can be replayed.
    random.Random(seed).shuffle(series)
    return series


def _fmt_s(s: float) -> str:
    return f"{s / 60:.1f} min" if s >= 120 else f"{s:.1f} s"


def list_cells(groups, seeds, proto: Protocol, cooldown_s: float, clock_on: bool, cuda: bool,
               cell_cooldown_s: float = 10.0) -> None:
    grand = 0.0
    for g in groups:
        cells = select([g], seeds, cuda)
        n_series = len({c.series for c in cells})
        est = 0.0
        heavy = 0
        for c in cells:
            e = estimate_cell_s(c, proto, clock_on)
            est += e
            heavy += e >= HEAVY_CELL_S
            tag = "" if e else ("  (skipped)" if needs_cuda(c) else "  (unsupported)")
            print(f"  {c.id:52s} {e:8.1f} s{tag}")
        cool = max(0, n_series - 1) * cooldown_s + heavy * min(cooldown_s, cell_cooldown_s)
        grand += est + cool
        print(
            f"group {g}: {len(cells)} cells in {n_series} series, est {_fmt_s(est + cool)}"
            f" (cells {_fmt_s(est)} + cooldowns {_fmt_s(cool)}; CV reruns not included)"
        )
    if len(groups) > 1:
        print(f"total est {_fmt_s(grand)}")


# --- Spin-wait probe. ---


def _settle(trace: list, window_s: float, floor: float) -> tuple[float, float | None]:
    # process_time advances in scheduler ticks (about 15.6 ms per thread on
    # Windows), so a rate over one 5 ms step is mostly 0 or several cores. A
    # trailing window long enough that one stray tick stays under the floor is
    # the readable rate; the settle point is where the last busy window began.
    last_busy = None
    first_drop = None
    for i in range(1, len(trace)):
        t, cpu = trace[i]
        dt_step = t - trace[i - 1][0]
        if first_drop is None and dt_step > 0 and (cpu - trace[i - 1][1]) / dt_step < floor:
            first_drop = t
        j = 0
        while j + 1 < i and trace[j + 1][0] <= t - window_s:
            j += 1
        span = t - trace[j][0]
        if span > 0 and (cpu - trace[j][1]) / span >= floor:
            last_busy = t
    settle = 0.0 if last_busy is None else max(0.0, last_busy - window_s)
    return settle, first_drop


def probe_spin(
    *, threads: int = HEADLINE_THREADS, n: int = 1024, reps: int = 5, span_s: float = 1.5,
    step_s: float = 0.005, floor: float = 0.1, window_s: float = 0.2, sleep=time.sleep,
) -> dict:
    rng = np.random.default_rng(0)
    a, b = _normal(rng, (n, n), np.dtype("float64")), _normal(rng, (n, n), np.dtype("float64"))
    calls = {"numpy": lambda: np.matmul(a, b), PEREGRINE: lambda: pg.matmul(a, b)}
    out = {}
    with pinned_threads(threads) as observed:
        for name, fn in calls.items():
            fn()
            sleep(span_s)
            runs = []
            for _ in range(reps):
                fn()
                w0, c0 = time.perf_counter(), time.process_time()
                trace = [(0.0, 0.0)]
                while trace[-1][0] < span_s:
                    sleep(step_s)
                    trace.append((time.perf_counter() - w0, time.process_time() - c0))
                settle, first_drop = _settle(trace, window_s, floor)
                runs.append({"settle_s": settle, "first_step_below_s": first_drop, "trace": trace})
                print(f"  {name}: settle {settle * 1e3:.0f} ms, first 5 ms step below {floor} core at "
                      f"{(first_drop or 0) * 1e3:.0f} ms, cpu over {span_s} s {trace[-1][1]:.3f} s")
            out[name] = runs
    worst = max(r["settle_s"] for runs in out.values() for r in runs)
    return {
        "threads": threads,
        "threads_observed": observed,
        "n": n,
        "dtype": "float64",
        "step_s": step_s,
        "span_s": span_s,
        "rate_window_s": window_s,
        "rate_floor_cores": floor,
        "results": out,
        "max_settle_s": worst,
        # Rounded up to 50 ms with one step of 50 ms headroom.
        "suggested_switch_gap_ms": int(50 * (math.ceil(worst * 1000 / 50) + 1)),
    }


# --- CLI. ---


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", nargs="+", choices=GROUPS, default=["quick"])
    p.add_argument("--list", action="store_true", help="print the cells and an estimated duration, run nothing")
    p.add_argument("--out", default=None, help="result JSON (default under benchmarks/results/local/v3.1/)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--samples", type=int, default=None, help="samples per contender per cell (min 30)")
    p.add_argument("--warmups", type=int, default=None, help="warmup samples per contender per cell (min 5)")
    p.add_argument("--switch-gap-ms", type=float, default=None, help="idle gap at every contender switch (default 300)")
    # 100 ms, far above the 200 us floor: averaging each sample over many calls is
    # what settled small and single-thread cells here, while 250 ms samples made a
    # 6-thread cell outlast the laptop's turbo window (clocks fell from about 145
    # to 121 percent in the last block, for both libraries).
    p.add_argument("--batch-target-us", type=float, default=100000.0)
    p.add_argument("--all-logical-cpus", action="store_true",
                   help="do not restrict the process to one logical CPU per physical core")
    p.add_argument("--rewarm-ms", type=float, default=None,
                   help="untimed running of each contender before sampling, after every switch (default 200)")
    p.add_argument("--cooldown-s", type=float, default=30.0, help="between series and before each CV rerun")
    p.add_argument("--cell-cooldown-s", type=float, default=10.0,
                   help="after any cell that ran 10 s or more, within a series")
    p.add_argument("--no-cooldown", action="store_true")
    p.add_argument("--clock-floor", type=float, default=None, help="discard samples below this %% Processor Performance")
    p.add_argument("--clock-ceiling", type=float, default=None, help="discard samples above this %% Processor Performance")
    p.add_argument("--verify-mode", choices=["last", "all"], default="last",
                   help="check each batch's last call (default) or keep and check every call")
    p.add_argument("--series-seed", type=int, default=None)
    p.add_argument("--filter", default=None, help="keep only cells whose id matches this regex")
    p.add_argument("--smoke", action="store_true", help="few samples, no cooldowns; output is never evidence")
    p.add_argument("--probe-spin", action="store_true")
    p.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    p.add_argument("--fresh-child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not args.smoke and (
        (args.samples is not None and args.samples < 30) or (args.warmups is not None and args.warmups < 5)
    ):
        p.error("--samples must be >= 30 and --warmups >= 5 (use --smoke for a non-evidence check)")
    return args


def _protocol(args) -> tuple[Protocol, float]:
    quiet = args.smoke or args.no_cooldown
    cooldown = 0.0 if quiet else args.cooldown_s
    gap_ms = args.switch_gap_ms if args.switch_gap_ms is not None else (0.0 if args.smoke else 300.0)
    proto = Protocol(
        samples=args.samples if args.samples is not None else (10 if args.smoke else 30),
        warmups=args.warmups if args.warmups is not None else (2 if args.smoke else 5),
        batch_target_s=args.batch_target_us * 1e-6,
        switch_gap_s=gap_ms / 1e3,
        rewarm_s=(args.rewarm_ms if args.rewarm_ms is not None else (0.0 if args.smoke else 200.0)) / 1e3,
        strike_cooldown_s=cooldown,
        fresh_cooldown_s=0.0 if quiet else 120.0,
        clock_floor=args.clock_floor,
        clock_ceiling=args.clock_ceiling,
        verify_mode=args.verify_mode,
        max_strikes=1 if args.fresh_child else 3,
    )
    return proto, cooldown


def _write(path: str, result: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(redact_paths(result), f, indent=1)
    os.replace(tmp, path)


def _line(r: dict) -> str:
    s = f"{r['id']:52s} {r['status']:13s}"
    if "batch_k" in r:
        s += f" K={r['batch_k']:<5d}"
    for n, v in r.get("ratio", {}).items():
        s += f" {n} {v:6.3f}x"
    if r["status"] not in ("ok",) and r.get("reason"):
        s += f"  ({r['reason'][:120]})"
    return s


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse(argv)
    global _AFFINITY
    _AFFINITY = None if args.all_logical_cpus else _confine_to_physical_cores()
    proto, cooldown = _protocol(args)
    clock = PdhClock()
    clock_obj = clock if clock.available else None
    cuda = pg.has_cuda()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.probe_spin:
        out_path = args.out or os.path.join(RESULTS_LOCAL, f"probe-spin-{stamp}.json")
        probe = probe_spin()
        manifest = protocol_manifest(argv=["bench_suite.py", *argv], series_seed=None, clock=clock)
        _write(out_path, {"manifest": manifest, "probe_spin": probe})
        print(f"max settle {probe['max_settle_s'] * 1e3:.0f} ms; suggested --switch-gap-ms "
              f"{probe['suggested_switch_gap_ms']}; saved {out_path}")
        return 0

    if args.list:
        list_cells(args.group, args.seeds, proto, cooldown, clock_obj is not None, cuda, args.cell_cooldown_s)
        return 0

    cells = [Cell.from_spec(json.loads(args.cell))] if args.cell else select(args.group, args.seeds, cuda)
    if args.filter:
        cells = [c for c in cells if re.search(args.filter, c.id)]
    try:
        for t in sorted({c.threads for c in cells}):
            with pinned_threads(t):
                pass
    except RuntimeError as e:
        print(f"thread check failed: {e}", file=sys.stderr)
        return 2

    seed = args.series_seed if args.series_seed is not None else random.SystemRandom().randrange(2**32)
    out_path = args.out or os.path.join(RESULTS_LOCAL, f"{'+'.join(args.group)}-{stamp}.json")
    manifest = protocol_manifest(
        argv=["bench_suite.py", *argv], series_seed=seed, clock=clock,
        extra={"smoke": args.smoke, "protocol": dataclasses.asdict(proto), "series_cooldown_s": cooldown,
               "affinity_mask": _AFFINITY},
    )
    result = {"manifest": manifest, "cells": []}
    fresh = None if args.fresh_child else fresh_process_runner(args.smoke)
    cell_cooldown = 0.0 if cooldown == 0 else args.cell_cooldown_s
    wait = 0.0
    for i, series in enumerate(series_order(cells, seed)):
        if i > 0:
            wait = max(wait, cooldown)
        for cell in series:
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
            try:
                r = run_one(cell, proto, clock=clock_obj, fresh_runner=fresh)
            except RuntimeError as e:
                print(f"thread check failed: {e}", file=sys.stderr)
                return 2
            wait = cell_cooldown if time.perf_counter() - t0 >= HEAVY_CELL_S else 0.0
            result["cells"].append(r)
            _write(out_path, result)
            print(_line(r))
    print(f"saved {out_path}")
    return 1 if any(c["status"] == "verify_failed" for c in result["cells"]) else 0


if __name__ == "__main__":
    sys.exit(main())
