"""Render the benchmark tables in README.md and docs/benchmarks.md from the results JSON.

    python benchmarks/report.py --write    rewrite every block in both files in place
    python benchmarks/report.py --check    exit 1 if any block differs from a fresh render

A block is the text between a `<!-- report:NAME -->` line and its
`<!-- /report:NAME -->` line. Every number in a block is read from a committed
benchmarks/results/*.json file with json.load: a results file is data, so
nothing here evals or unpickles one. A missing file renders as an explicit
"Not measured" line. A cell below 1.0x against any baseline must match a
REASONS entry or rendering raises, so no loss is published without its reason.
A cell the protocol closed as noisy is listed by id and never shown with a
number.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "benchmarks", "results")

DTYPES = ("float32", "float64")
FUSED_OPS = ("axpby", "fma3", "scaled_relu")

# (group, build): one results file and one docs table each.
MATRIX = (
    ("matmul_square", "cpu"), ("matmul_rect", "cpu"), ("matmul_layout", "cpu"),
    ("transpose", "cpu"), ("reduce", "cpu"), ("fused", "cpu"), ("overhead", "cpu"),
    ("gemm_1t", "cpu"), ("gpu", "cuda"), ("overhead", "cuda"),
)
TITLES = {
    "matmul_square": "matmul, square", "matmul_rect": "matmul, rectangular",
    "matmul_layout": "matmul, operand layouts", "transpose": "transpose",
    "reduce": "sum and mean", "fused": "fused elementwise ops", "overhead": "per-call overhead",
    "gemm_1t": "matmul, one thread", "gpu": "GPU",
}
BUILDS = {"cpu": "CPU-only build", "cuda": "CUDA build"}
CEILINGS = "refbox_ceilings.json"

DOCS = {
    "README.md": ("headline", "losses", "compare", "environment", "hardware"),
    "docs/benchmarks.md": ("environment", "compare_full") + tuple(f"matrix_{g}_{b}" for g, b in MATRIX),
}

# A ratio below 1.0 against any baseline needs a reason here (first match on the
# cell id wins), or rendering raises.
REASONS = (
    (r"^matmul/float32/4096x4096x1/",
     "n = 1 makes this a matrix-vector product, bound by streaming the 64 MiB left operand, "
     "and this build runs it through the packed GEMM path"),
    (r"^gpu_matmul/float64/",
     "this GPU runs float64 at 1/64 of its float32 rate, below the CPU's own float64 throughput, "
     "which is why float64 never auto-routes to it"),
    (r"^matmul/float32/[^/]+/c/auto/",
     "auto runs the route it picks plus the routing check, so it trails that forced route by "
     "the dispatch cost; its target is the band around the faster route"),
    (r"^(axpby|fma3|scaled_relu)/[^/]+/[^/]+/c/alloc/",
     "a new output pays for its allocation and first touch, which the preallocated NumPy "
     "sequence never does"),
)

# A cell that misses its target and matches one of these is reported with the
# closure's kind instead of "below". The evidence is always computed from the
# JSON; "ceiling" selects the ceilings-file record that bounds the cell, and an
# "at ceiling" closure with one must hold at 90 percent of it and 0.97x NumPy.
CLOSURES = (
    (r"^scaling/float(32|64)/2048x2048x2048/",
     {"kind": "at ceiling",
      "why": "per-thread timers show the six-thread compute is bound by the all-core clock under "
             "the laptop's power limit, which puts the schedule's own speedup bound below the "
             "gate; three approaches to the remaining synchronization cost were measured"}),
    (r"^fma3/float64/1024x1024/c/out/",
     {"kind": "at ceiling", "ceiling": ("memory", "triad", "nontemporal", 6),
      "why": "memory bound: Peregrine moves its four operands at or above the measured "
             "triad bandwidth"}),
)

HEADLINE = (
    tuple(("matmul_square", "cpu", f"matmul/{dt}/{n}x{n}x{n}/c/-/t6")
          for dt in DTYPES for n in (16, 64, 256, 512, 2048))
    + (("matmul_rect", "cpu", "matmul/float32/1x4096x4096/c/-/t6"),
       ("matmul_rect", "cpu", "matmul/float32/4096x64x4096/c/-/t6"),
       ("matmul_rect", "cpu", "matmul/float64/1000x333x750/c/-/t6"))
    + tuple(("matmul_layout", "cpu", f"matmul/float32/512x512x512/{lay}/-/t6") for lay in ("at_b", "a_bt", "ff"))
    + (("transpose", "cpu", "transpose/float32/64x64/c/-/t6"),
       ("transpose", "cpu", "transpose/float32/2048x2048/c/-/t6"),
       ("transpose", "cpu", "transpose/float64/1000x3000/c/-/t6"),
       ("reduce", "cpu", "sum/float32/64x64/c/axis=None/t6"),
       ("reduce", "cpu", "sum/float64/4096x4096/c/axis=None/t6"),
       ("reduce", "cpu", "mean/float32/1024x1024/c/axis=0/t6"),
       ("reduce", "cpu", "mean/float64/1000000x16/c/axis=1/t6"))
    + tuple(("fused", "cpu", f"{op}/float32/1024x1024/c/{v}/t6") for op in FUSED_OPS for v in ("alloc", "out"))
    + tuple(("gpu", "cuda", f"matmul/float32/{n}x{n}x{n}/c/auto/t6") for n in (2048, 4096))
)

# The README's short comparison; every cell that got slower is added to it.
COMPARE_SHORT = (
    "matmul/float32/64x64x64/c/-/t6", "matmul/float32/2048x2048x2048/c/-/t6",
    "matmul/float64/64x64x64/c/-/t6", "matmul/float64/2048x2048x2048/c/-/t6",
    "matmul/float32/1x4096x4096/c/-/t6", "matmul/float32/512x512x512/at_b/-/t6",
    "transpose/float32/2048x2048/c/-/t6", "sum/float64/4096x4096/c/axis=None/t6",
    "axpby/float32/2896x2896/c/alloc/t6", "matmul/float32/2048x2048x2048/c/auto/t6",
)

LAYOUT_ORDER = {"c": 0, "at_b": 1, "a_bt": 2, "ff": 3}
LAYOUT_TEXT = {"c": "a @ b", "at_b": "a.T @ b", "a_bt": "a @ b.T", "ff": "a @ b, both F-order"}
VARIANT_TEXT = {"alloc": "new output", "out": "out=", "auto": "auto route"}
BASE_TEXT = {
    "numpy": "NumPy", "numpy_noalloc": "NumPy, preallocated", "cupy": "CuPy",
    "forced": "faster forced route", "own_c": "own C-layout time", "v3.0.0": "v3.0.0",
}
COLUMN_TEXT = {
    "numpy": "NumPy", "numpy_noalloc": "NumPy prealloc", "cupy": "CuPy",
    "peregrine_cpu": "forced CPU", "peregrine_gpu": "forced GPU", "peregrine": "Peregrine",
}
RATIO_TEXT = {
    "numpy": "ratio vs NumPy", "numpy_noalloc": "ratio vs prealloc", "cupy": "ratio vs CuPy",
    "forced": "auto vs faster forced", "own_c": "time vs own C", "v3.0.0": "ratio vs v3.0.0",
}
# Ratios that compare Peregrine with itself, shown as a percentage, never a loss.
SELF = ("forced", "own_c")


# --- Reading. ---


def load(name: str) -> dict | None:
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def matrix_file(group: str, build: str) -> str:
    return f"refbox_{group}_{build}_v310.json"


def quick_file(build: str, version: str) -> str:
    return f"refbox_quick_{build}_{version}.json"


def not_measured(name: str) -> str:
    return f"Not measured: `benchmarks/results/{name}` is absent."


def summarize(data: dict) -> dict:
    """Rows keyed by cell id without its seed; seeds aggregate by median over the ok ones."""
    by_key: dict = {}
    for c in data["cells"]:
        by_key.setdefault(c["id"].rsplit("/s", 1)[0], []).append(c)
    return {k: _row(k, cs) for k, cs in by_key.items()}


def _row(key: str, cells: list) -> dict:
    c0 = cells[0]
    ok = [c for c in cells if c["status"] == "ok"]
    statuses = [c["status"] for c in cells]
    row = {
        "key": key, "op": c0["op"], "dtype": c0["dtype"], "shape": tuple(c0["shape"]),
        "layout": c0["layout"], "variant": c0["variant"], "threads": c0["threads"],
        "timer": c0.get("timer"), "names": list(c0.get("contenders", {})),
        "seeds": f"{len(ok)}/{len(cells)}",
        "status": "ok" if ok else ("noisy" if "noisy" in statuses else statuses[0]),
        "stats": {}, "ratio": {}, "rate": {},
    }
    if not ok:
        return row
    med = statistics.median
    for name in ok[0]["contenders"]:
        per = [c["contenders"][name] for c in ok]
        row["stats"][name] = {
            "median": med([s["median_s"] for s in per]),
            "iqr": med([s["q3_s"] - s["q1_s"] for s in per]),
            "min": min(s["min_s"] for s in per),
        }
    for base in ok[0].get("ratio", {}):
        row["ratio"][base] = med([c["ratio"][base] for c in ok])
    for unit in ("gflops", "gbps"):
        if unit in ok[0]:
            row["rate"][unit] = {n: med([c[unit][n] for c in ok]) for n in ok[0][unit]}
    if {"peregrine_cpu", "peregrine_gpu"} <= row["ratio"].keys():
        row["ratio"]["forced"] = min(row["ratio"]["peregrine_cpu"], row["ratio"]["peregrine_gpu"])
    return row


def _key_row(key: str) -> dict:
    op, dtype, shape, layout, variant, threads = key.split("/")
    return {"key": key, "op": op, "dtype": dtype, "shape": tuple(int(d) for d in shape.split("x")),
            "layout": layout, "variant": variant, "threads": int(threads[1:]), "status": "not measured",
            "stats": {}, "ratio": {}, "rate": {}, "names": []}


def _pg(row: dict) -> float:
    return row["stats"]["peregrine"]["median"]


def matrix_rows(group: str, build: str) -> dict | None:
    data = load(matrix_file(group, build))
    if data is None:
        return None
    rows = summarize(data)
    if group == "matmul_layout":
        for row in rows.values():
            if row["layout"] == "c":
                continue
            parts = row["key"].split("/")
            parts[3] = "c"
            ref = rows.get("/".join(parts))
            if ref is None:
                continue
            row["has_ref"] = True
            if ref["status"] == "ok" and row["status"] == "ok":
                row["ref_median"] = _pg(ref)
                row["ratio"]["own_c"] = _pg(ref) / _pg(row)
    if build == "cuda":
        old = load(quick_file("cuda", "v300"))
        old_rows = summarize(old) if old else {}
        for row in rows.values():
            prev = old_rows.get(row["key"])
            if row["variant"] != "device" or prev is None:
                continue
            row["has_v300"] = True
            if prev["status"] == "ok" and row["status"] == "ok":
                row["v300_median"] = _pg(prev)
                row["ratio"]["v3.0.0"] = _pg(prev) / _pg(row)
    return rows


# --- Targets. ---


def square_target(n: int) -> float:
    if n <= 16:
        return 3.0
    if n <= 96:
        return 1.5
    if n <= 384:
        return 1.15
    if n <= 768:
        return 1.05
    return 0.97


def targets(row: dict) -> list:
    """[(baseline, target)]: a pair is met when baseline time / Peregrine time >= target."""
    op, shape, variant = row["op"], row["shape"], row["variant"]
    if op == "matmul" and variant == "auto":
        m, k, n = shape
        t = [("forced", 1 / 1.05)]
        if m == k == n and n >= 2048:
            t.append(("numpy", 5.0))
        return t
    if op == "matmul":
        m, k, n = shape
        if row["threads"] == 1:
            # The one-thread cells are the GEMM diagnostic, held to single-thread
            # OpenBLAS at the two sizes its gate names; the size targets are for six.
            return [("numpy", 0.95)] if m == k == n and n in (1024, 2048) else []
        t = [("numpy", square_target(n) if m == k == n else 1.0)]
        if row.get("has_ref"):
            t.append(("own_c", 1 / 1.10))
        return t
    if op == "transpose":
        return [("numpy", 4.0 if min(shape) >= 512 else 2.0)]
    if op in ("sum", "mean"):
        m, n = shape
        return [("numpy", 3.0 if m <= 64 and n <= 64 else 1.5)]
    if op in FUSED_OPS:
        # Like for like on allocation above 64x64: out= against preallocated
        # NumPy, a new output against the expression that also allocates.
        m, n = shape
        if m <= 64 and n <= 64:
            return [("numpy", 3.0), ("numpy_noalloc", 2.0)]
        return [("numpy_noalloc", 2.0)] if variant == "out" else [("numpy", 2.0)]
    if row.get("has_v300"):
        return [("v3.0.0", 1.0)]
    return []


def _match(table: tuple, key: str):
    return next((v for rx, v in table if re.search(rx, key)), None)


def status(row: dict) -> str:
    if row["status"] != "ok":
        return row["status"].replace("_", " ")
    tg = targets(row)
    if not tg:
        return "no target"
    if any(b not in row["ratio"] for b, _ in tg):
        return "reference noisy"
    if all(row["ratio"][b] >= t for b, t in tg):
        return "met"
    entry = _match(CLOSURES, row["key"])
    return entry["kind"] if entry else "below"


def losses() -> list:
    """(file, row, baseline, ratio, reason) for every ratio below 1.0; one without a reason raises."""
    out = []
    for group, build in MATRIX:
        rows = matrix_rows(group, build)
        for row in (rows or {}).values():
            for base, r in row["ratio"].items():
                if base in SELF or r >= 1.0:
                    continue
                reason = _match(REASONS, row["key"])
                if reason is None:
                    raise ValueError(f"{row['key']} is {r:.2f}x against {base} and has no REASONS entry")
                out.append((matrix_file(group, build), row, base, r, reason))
    return out


# --- Formatting. ---


def fmt_time(s: float) -> str:
    for unit, scale in (("ns", 1e-9), ("us", 1e-6), ("ms", 1e-3), ("s", 1.0)):
        v = s / scale
        if v < 999.5 or unit == "s":
            return f"{v:.{0 if v >= 99.95 else 1 if v >= 9.995 else 2}f} {unit}"


def fmt_ratio(r: float) -> str:
    return f"{r:.2f}x"


def fmt_pair(base: str, r: float) -> str:
    return f"{(1 / r - 1) * 100:+.1f}%" if base in SELF else fmt_ratio(r)


def target_text(base: str, t: float) -> str:
    if base == "forced":
        return f"within {(1 / t - 1) * 100:.0f}% of the faster forced route"
    if base == "own_c":
        return f"within {(1 / t - 1) * 100:.0f}% of own C-layout time"
    return f">= {t:.2f}x {BASE_TEXT[base]}"


def op_text(row: dict) -> str:
    op = row["op"]
    if op == "matmul":
        op = f"matmul {LAYOUT_TEXT[row['layout']]}"
    elif op == "gpu_matmul":
        op = "matmul, device-resident"
    elif op == "gpu_chain":
        op = "fused chain, device-resident"
    extra = VARIANT_TEXT.get(row["variant"], "" if row["variant"] in ("-", "device") else row["variant"])
    return f"{op}, {extra}" if extra else op


def shape_text(row: dict) -> str:
    return "x".join(str(d) for d in row["shape"])


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def _sort_key(row: dict) -> tuple:
    return (row["op"], row["dtype"], row["variant"], math.prod(row["shape"]), row["shape"],
            LAYOUT_ORDER.get(row["layout"], 9))


def _base_median(row: dict, base: str) -> float:
    if base == "forced":
        return min(row["stats"]["peregrine_cpu"]["median"], row["stats"]["peregrine_gpu"]["median"])
    if base == "own_c":
        return row["ref_median"]
    if base == "v3.0.0":
        return row["v300_median"]
    return row["stats"][base]["median"]


# --- Blocks. ---


def render_headline() -> str:
    cache: dict = {}
    out = []
    for group, build, key in HEADLINE:
        if (group, build) not in cache:
            cache[group, build] = matrix_rows(group, build)
        rows = cache[group, build]
        row = rows.get(key) if rows else None
        if row is None:
            row = _key_row(key)
            out.append([op_text(row), row["dtype"], shape_text(row), row["threads"]] + ["-"] * 5 + ["not measured"])
            continue
        cell = status(row)
        for base, t in targets(row):
            r = row["ratio"].get(base)
            if r is None:
                out.append([op_text(row), row["dtype"], shape_text(row), row["threads"], BASE_TEXT[base],
                            "-", "-", "noisy" if cell == "noisy" else "-", target_text(base, t), cell])
                continue
            verdict = "met" if r >= t else (cell if cell in ("at ceiling", "deferred") else "below")
            out.append([op_text(row), row["dtype"], shape_text(row), row["threads"], BASE_TEXT[base],
                        fmt_time(_base_median(row, base)), fmt_time(_pg(row)), fmt_pair(base, r),
                        target_text(base, t), verdict])
    missing = sorted({matrix_file(g, b) for g, b, _ in HEADLINE if cache.get((g, b)) is None})
    caption = (
        "Representative cells from the full matrix. Times are medians of the timed samples; both "
        "libraries ran at the thread count in the threads column. The ratio is the baseline's median "
        "time divided by Peregrine's, so above 1.00x Peregrine is faster; a percentage compares "
        "Peregrine with its own reference route. The auto rows come from the CUDA build, the rest "
        "from the CPU-only build."
    )
    header = ["operation", "dtype", "shape", "threads", "baseline", "baseline median",
              "Peregrine median", "ratio", "target", "status"]
    return "\n\n".join([caption, table(header, out)] + [not_measured(n) for n in missing])


def scaling_rows() -> list | None:
    one, six, ceil = matrix_rows("gemm_1t", "cpu"), matrix_rows("matmul_square", "cpu"), load(CEILINGS)
    if one is None or six is None or ceil is None:
        return None
    out = []
    for dt in DTYPES:
        fma = {f["threads"]: f["total_gflops"] for f in ceil["fma"] if f["dtype"] == dt}
        lo, hi = min(fma), max(fma)
        r1 = one.get(f"matmul/{dt}/2048x2048x2048/c/-/t{lo}")
        r6 = six.get(f"matmul/{dt}/2048x2048x2048/c/-/t{hi}")
        row = {"key": f"scaling/{dt}/2048x2048x2048/c/-/t{lo}-t{hi}", "dtype": dt, "lo": lo, "hi": hi,
               "fma": fma[hi] / fma[lo]}
        if r1 is None or r6 is None:
            row["status"] = "not measured"
        elif r1["status"] != "ok" or r6["status"] != "ok":
            row["status"] = "noisy"
        else:
            row["pg"] = _pg(r1) / _pg(r6)
            row["blas"] = r1["stats"]["numpy"]["median"] / r6["stats"]["numpy"]["median"]
            entry = _match(CLOSURES, row["key"])
            row["status"] = "met" if row["pg"] >= 0.9 * row["fma"] else (entry["kind"] if entry else "below")
        out.append(row)
    return out


def _ceiling(ceil: dict, spec: tuple) -> tuple:
    if spec[0] == "fma":
        _, dtype, threads = spec
        f = next(f for f in ceil["fma"] if f["dtype"] == dtype and f["threads"] == threads)
        return f["total_gflops"], "GF/s", f"{f['threads']}-thread {dtype} FMA peak"
    _, kernel, stores, threads = spec
    run = next(r for r in ceil["memory"]["runs"] if r["threads"] == threads)
    k = next(k for k in run["kernels"] if k["kernel"] == kernel and k["stores"] == stores)
    return k["median_gbs"], "GB/s", f"{run['threads']}-thread {kernel} bandwidth ({stores} stores)"


def _closure_evidence(row: dict, entry: dict, ceil: dict | None) -> str:
    worst = min(targets(row), key=lambda bt: row["ratio"][bt[0]] / bt[1])
    text = f"{fmt_pair(worst[0], row['ratio'][worst[0]])} against {BASE_TEXT[worst[0]]} (target {target_text(*worst)})"
    if "ceiling" not in entry:
        return text
    if ceil is None:
        return f"{text}; {not_measured(CEILINGS)}"
    value, unit, name = _ceiling(ceil, entry["ceiling"])
    rate = row["rate"]["gflops" if unit == "GF/s" else "gbps"]["peregrine"]
    vs_numpy = [row["ratio"][b] for b, _ in targets(row) if b not in SELF]
    if entry["kind"] == "at ceiling" and (rate < 0.9 * value or min(vs_numpy, default=1.0) < 0.97):
        raise ValueError(f"{row['key']} does not meet the at-ceiling rule: {rate:.1f} {unit} against {value:.1f}")
    return f"{text}; Peregrine at {rate:.1f} {unit}, {rate / value:.0%} of the {name}, {value:.1f} {unit}"


def render_losses() -> str:
    loss = losses()
    parts = ["**Below 1.00x.** Every measured ratio below one: the baseline's median time divided by "
             "Peregrine's, both libraries at the thread count in the cell id, with the reason."]
    if loss:
        parts.append(table(["cell", "results file", "baseline", "ratio", "reason"],
                           [[f"`{row['key']}`", f"`{f}`", BASE_TEXT[b], fmt_ratio(r), why]
                            for f, row, b, r, why in loss]))
    else:
        parts.append("No measured cell is below 1.00x against any baseline.")

    ceil = load(CEILINGS)
    closed, below, noisy, missing = [], [], [], []
    for group, build in MATRIX:
        data = load(matrix_file(group, build))
        if data is None:
            missing.append(matrix_file(group, build))
            continue
        ids = [c["id"] for c in data["cells"] if c["status"] == "noisy"]
        if ids:
            noisy.append(f"- {TITLES[group]}, {BUILDS[build]}: " + ", ".join(f"`{i}`" for i in ids))
        name = f"`{matrix_file(group, build)}`"
        for row in sorted(matrix_rows(group, build).values(), key=_sort_key):
            st = status(row)
            if st in ("at ceiling", "deferred"):
                entry = _match(CLOSURES, row["key"])
                closed.append([f"`{row['key']}`", name, st, _closure_evidence(row, entry, ceil), entry["why"]])
            elif st == "below":
                tg = [(b, t) for b, t in targets(row) if row["ratio"][b] < t]
                below.append(f"`{row['key']}` in {name} (" + "; ".join(
                    f"{fmt_pair(b, row['ratio'][b])}, target {target_text(b, t)}" for b, t in tg) + ")")
    for row in scaling_rows() or ():
        name = f"`{matrix_file('gemm_1t', 'cpu')}`, `{matrix_file('matmul_square', 'cpu')}`"
        if row["status"] in ("at ceiling", "deferred"):
            entry = _match(CLOSURES, row["key"])
            closed.append([f"`{row['key']}`", name, row["status"],
                           f"Peregrine {fmt_ratio(row['pg'])} from {row['lo']} to {row['hi']} threads, OpenBLAS "
                           f"{fmt_ratio(row['blas'])}, measured FMA peak ratio {fmt_ratio(row['fma'])}",
                           entry["why"]])
        elif row["status"] == "below":
            below.append(f"`{row['key']}` in {name} ({fmt_ratio(row['pg'])}, gate {fmt_ratio(0.9 * row['fma'])})")

    parts.append("**Closed below target.** Cells that miss their target and are closed at a measured "
                 "ceiling or deferred; every number is computed from the results files.")
    parts.append(table(["cell", "results file", "closure", "evidence", "reason"], closed) if closed else "None.")
    parts.append("**Below target, not closed.**")
    parts.append("\n".join(f"- {b}" for b in below) if below else "None.")
    parts.append("**Noisy.** These cells stayed above the CV limit on every attempt, including the "
                 "fresh-process retry, so the protocol publishes no number for them.")
    parts.append("\n".join(noisy) if noisy else "None.")
    return "\n\n".join(parts + [not_measured(n) for n in missing])


def render_hardware() -> str:
    ceil = load(CEILINGS)
    cuda = next((d for d in (load(matrix_file(g, b)) for g, b in MATRIX if b == "cuda") if d), None)
    if ceil is None:
        return not_measured(CEILINGS)
    cpu = ceil["cpu"]
    rows = [["CPU", f"{cpu['brand']}, {cpu['physical_cores']} cores, {cpu['logical_processors']} logical processors"]]
    if cuda is not None:
        m = cuda["manifest"]
        rt = ceil.get("pcie", {}).get("cuda_runtime_version")
        runtime = f", CUDA runtime {rt // 1000}.{rt % 1000 // 10}" if rt else ""
        rows.append(["GPU", f"{m.get('gpu')}, driver {m.get('driver')}{runtime}"])
    parts = [table(["component", "value"], rows)]
    if cuda is None:
        parts.append(not_measured(matrix_file("gpu", "cuda")))

    lo, hi = min(ceil["params"]["threads"]), max(ceil["params"]["threads"])
    fma = {(f["dtype"], f["threads"]): f["total_gflops"] for f in ceil["fma"]}
    parts.append(f"FMA peak in GF/s, measured by the ceiling tool at {lo} and {hi} threads (one per core):")
    parts.append(table(["dtype", f"{lo} thread", f"{hi} threads"],
                       [[dt, f"{fma[dt, lo]:.1f} GF/s", f"{fma[dt, hi]:.1f} GF/s"] for dt in DTYPES]))

    mem = ceil["memory"]
    runs = {r["threads"]: {(k["kernel"], k["stores"]): k["median_gbs"] for k in r["kernels"]} for r in mem["runs"]}
    parts.append(f"Memory bandwidth in GB/s, median of {mem['trials']} trials over a "
                 f"{mem['array_bytes'] // 2**20} MiB {mem['element']} array, counting the bytes each kernel "
                 f"reads and writes:")
    parts.append(table(["kernel", "stores", f"{lo} thread", f"{hi} threads"],
                       [[k, s, f"{runs[lo][k, s]:.1f} GB/s", f"{runs[hi][k, s]:.1f} GB/s"] for k, s in runs[lo]]))

    pcie = ceil.get("pcie")
    if pcie:
        pinned = [t for t in pcie["transfers"] if t["host"] == "pinned"]
        big = max(t["bytes"] for t in pinned)
        lat = {(x["host"], x["direction"]): x["latency_us"] for x in pcie["latency"]}
        parts.append(f"PCIe to the {pcie['device']}, pinned host memory, median of {pcie['reps']} transfers "
                     f"of {big // 2**20} MiB timed with CUDA events:")
        parts.append(table(["direction", "bandwidth", "latency"],
                           [[t["direction"].upper(), f"{t['median_gbs']:.1f} GB/s",
                             fmt_time(lat["pinned", t["direction"]] * 1e-6)]
                            for t in pinned if t["bytes"] == big]))
    return "\n\n".join(parts)


def _power_plan(text: str) -> str:
    m = re.search(r"\(([^)]+)\)", text or "")
    return m.group(1) if m else str(text)


ENV_FIELDS = (
    ("Peregrine threads", lambda m: m.get("peregrine_threads")),
    ("OpenBLAS threads", lambda m: m.get("blas_threads")),
    ("OMP_NUM_THREADS at start", lambda m: m.get("omp_num_threads_at_start")),
    ("process priority", lambda m: m.get("priority")),
    ("affinity mask (one logical CPU per core)", lambda m: m.get("affinity_mask")),
    ("CPU", lambda m: m.get("cpu_brand")),
    ("OS", lambda m: m.get("os_build")),
    ("power plan", lambda m: _power_plan(m.get("power_plan"))),
    ("Python", lambda m: m.get("python")),
    ("NumPy", lambda m: m.get("numpy")),
    ("OpenBLAS", lambda m: f"{m.get('openblas_version')} ({m.get('blas_core_type')} kernels)"),
    ("compiler", lambda m: m.get("compiler")),
    ("compiler flags", lambda m: f"`{m.get('compiler_flags')}`"),
    ("OpenMP runtime", lambda m: m.get("openmp_runtime")),
    ("GPU and driver (CUDA build)", lambda m: f"{m.get('gpu')}, driver {m.get('driver')}" if m.get("cuda_build") else None),
    ("samples per contender", lambda m: m["protocol"]["samples"]),
    ("warmup samples", lambda m: m["protocol"]["warmups"]),
    ("samples per ABBA block", lambda m: m["protocol"]["block"]),
    ("batch target per sample", lambda m: fmt_time(m["protocol"]["batch_target_s"])),
    ("rewarm after a switch", lambda m: fmt_time(m["protocol"]["rewarm_s"])),
    ("rewarm after a switch, GPU contenders", lambda m: fmt_time(m["protocol"]["gpu_rewarm_s"])),
    ("idle gap at a switch", lambda m: fmt_time(m["protocol"]["switch_gap_s"])),
    ("CV limit", lambda m: f"{m['protocol']['cv_limit']:.0%}"),
    ("attempts before the fresh-process retry", lambda m: m["protocol"]["max_strikes"]),
    ("clock-discard band", lambda m: "off" if m["protocol"]["clock_floor"] is None and m["protocol"]["clock_ceiling"] is None
     else f"{m['protocol']['clock_floor']} to {m['protocol']['clock_ceiling']} percent processor performance"),
)


def render_environment() -> str:
    files = [matrix_file(g, b) for g, b in MATRIX] + [quick_file(b, v) for b in ("cpu", "cuda") for v in ("v300", "v310")]
    loaded = {f: load(f) for f in files}
    present = {f: d for f, d in loaded.items() if d is not None}
    if not present:
        return "\n".join(not_measured(f) for f in files)
    per_file = []
    for f, d in present.items():
        m = d["manifest"]
        per_file.append([f"`{f}`", "CUDA" if m.get("cuda_build") else "CPU-only", m.get("peregrine_version"),
                         str((m.get("source_tree") or {}).get("commit"))[:12], str(m.get("datetime"))[:16],
                         m.get("priority") or "not recorded"])
    matrix = [d["manifest"] for f, d in present.items() if "_v310" in f and "quick" not in f]
    conditions = []
    for label, get in ENV_FIELDS:
        values = []
        for m in matrix:
            v = get(m)
            if v is not None and str(v) not in values:
                values.append(str(v))
        conditions.append([label, ", ".join(values) if values else "not recorded"])
    parts = ["Each results file and the run that wrote it:",
             table(["results file", "build", "Peregrine", "commit", "date", "priority"], per_file),
             "Measurement conditions recorded in the manifests of the matrix files (one value means every "
             "file agrees):",
             table(["condition", "value"], conditions)]
    return "\n\n".join(parts + [not_measured(f) for f in files if loaded[f] is None])


def _compare_cell(r: dict | None, name: str) -> str:
    if r is None:
        return "not measured"
    if r["status"] != "ok":
        return r["status"].replace("_", " ")
    return fmt_time(r["stats"][name]["median"])


def render_compare(short: bool) -> str:
    out, missing = [], []
    for build in ("cpu", "cuda"):
        old_d, new_d = load(quick_file(build, "v300")), load(quick_file(build, "v310"))
        missing += [quick_file(build, v) for v, d in (("v300", old_d), ("v310", new_d)) if d is None]
        if old_d is None or new_d is None:
            continue
        old, new = summarize(old_d), summarize(new_d)
        for key in sorted({**new, **old}, key=lambda k: _sort_key(old.get(k) or new[k])):
            a, b = old.get(key), new.get(key)
            speedup = _pg(a) / _pg(b) if a and b and a["status"] == b["status"] == "ok" else None
            if short and key not in COMPARE_SHORT and not (speedup is not None and speedup < 1.0):
                continue
            row = a or b
            out.append([op_text(row), row["dtype"], shape_text(row), row["threads"],
                        _compare_cell(a, "numpy"), _compare_cell(b, "numpy"),
                        _compare_cell(a, "peregrine"), _compare_cell(b, "peregrine"),
                        fmt_ratio(speedup) if speedup is not None else "-"])
    caption = (
        "Peregrine's median time in the v3.0.0 and v3.1.0 runs of the quick subset, same harness and "
        "machine, with NumPy's median in each run as a check that the machine state matched. The "
        "speedup is the v3.0.0 time divided by the v3.1.0 time; both libraries ran at the thread count "
        "in the threads column."
    )
    if short:
        caption += " A selection, plus every cell that got slower; the full list is in docs/benchmarks.md."
    header = ["operation", "dtype", "shape", "threads", "NumPy, v3.0.0 run", "NumPy, v3.1.0 run",
              "Peregrine v3.0.0", "Peregrine v3.1.0", "speedup"]
    parts = [caption, table(header, out)] if out else []
    return "\n\n".join(parts + [not_measured(n) for n in missing])


SECTIONS = ("", "Device-resident, Peregrine timed with CUDA events",
            "Host float32 matmul, auto route against both forced routes", "Transfers")


def _section(row: dict) -> str:
    if row["variant"] == "auto":
        return SECTIONS[2]
    if row["op"] in ("to_device", "from_device"):
        return SECTIONS[3]
    return SECTIONS[1] if row["variant"] == "device" else SECTIONS[0]


def _applies(row: dict, base: str) -> bool:
    if base == "forced":
        return row["variant"] == "auto"
    if base == "own_c":
        return bool(row.get("has_ref"))
    if base == "v3.0.0":
        return bool(row.get("has_v300"))
    return base in row["names"]


def matrix_table(rows: list) -> str:
    names = [n for n in COLUMN_TEXT if any(n in r["names"] for r in rows)]
    auto = any(r["variant"] == "auto" for r in rows)
    bases = [b for b in ("numpy", "numpy_noalloc", "cupy") if b in names]
    bases += [b for b, flag in (("forced", auto), ("own_c", any(r.get("has_ref") for r in rows)),
                                ("v3.0.0", any(r.get("has_v300") for r in rows))) if flag]
    rate = names == ["peregrine"]
    header = ["op", "dtype", "shape", "layout", "variant", "threads", "seeds"]
    for n in names:
        label = "auto" if auto and n == "peregrine" else COLUMN_TEXT[n]
        header += [f"{label} median"] if n in ("peregrine_cpu", "peregrine_gpu") else \
            [f"{label} median", f"{label} IQR", f"{label} min"]
    header += [RATIO_TEXT[b] for b in bases] + (["Peregrine GB/s"] if rate else []) + ["target", "status"]
    out = []
    for row in rows:
        ok = row["status"] == "ok"
        line = [row["op"], row["dtype"], shape_text(row), row["layout"], row["variant"], row["threads"], row["seeds"]]
        for n in names:
            s = row["stats"].get(n)
            keys = ("median",) if n in ("peregrine_cpu", "peregrine_gpu") else ("median", "iqr", "min")
            line += [fmt_time(s[k]) if s else "-" for k in keys]
        for b in bases:
            r = row["ratio"].get(b)
            noisy = row["status"] == "noisy" and _applies(row, b)
            line.append(fmt_pair(b, r) if r is not None else ("noisy" if noisy else "-"))
        if rate:
            line.append(f"{row['rate']['gbps']['peregrine']:.1f} GB/s" if ok else "-")
        line += ["; ".join(target_text(b, t) for b, t in targets(row)) or "-", status(row)]
        out.append(line)
    return table(header, out)


def render_matrix(group: str, build: str) -> str:
    name = matrix_file(group, build)
    rows = matrix_rows(group, build)
    if rows is None:
        return not_measured(name)
    parts = [f"{TITLES[group]}, {BUILDS[build]}, from `{name}`. Medians, IQRs and minimums of the timed "
             "samples (each sample is the mean of a batch of calls); both libraries at the thread count in "
             "the threads column. A ratio is the baseline's median time divided by Peregrine's; a percentage "
             "compares Peregrine with its own reference. Seeds counts the seeds that passed the CV gate; a "
             "noisy cell shows no number."]
    ordered = sorted(rows.values(), key=_sort_key)
    for title in sorted({_section(r) for r in ordered}, key=SECTIONS.index):
        if title:
            parts.append(f"**{title}**")
        parts.append(matrix_table([r for r in ordered if _section(r) == title]))
    if group == "gemm_1t":
        sc = scaling_rows()
        if sc is None:
            parts.append("Thread scaling: " + not_measured(f"{CEILINGS}` or `{matrix_file('matmul_square', 'cpu')}"))
        else:
            lo, hi = sc[0]["lo"], sc[0]["hi"]
            parts.append(f"**Thread scaling at n = 2048**: Peregrine's and OpenBLAS's {lo}-thread median divided "
                         f"by their {hi}-thread median, against the ceiling tool's FMA peak ratio between the same "
                         "thread counts; the gate is 90 percent of that ratio.")
            parts.append(table(
                ["dtype", "Peregrine speedup", "OpenBLAS speedup", "FMA peak ratio", "gate", "status"],
                [[r["dtype"], fmt_ratio(r["pg"]) if "pg" in r else "-", fmt_ratio(r["blas"]) if "blas" in r else "-",
                  fmt_ratio(r["fma"]), fmt_ratio(0.9 * r["fma"]), r["status"]] for r in sc]))
    return "\n\n".join(parts)


RENDER = {
    "headline": render_headline,
    "losses": render_losses,
    "hardware": render_hardware,
    "environment": render_environment,
    "compare": lambda: render_compare(short=True),
    "compare_full": lambda: render_compare(short=False),
}
RENDER.update({f"matrix_{g}_{b}": (lambda g=g, b=b: render_matrix(g, b)) for g, b in MATRIX})


def render(name: str) -> str:
    return RENDER[name]()


# --- The docs files. ---

BLOCK = re.compile(r"<!-- report:(\w+) -->\n(.*?)<!-- /report:\1 -->", re.S)


def _read(rel: str) -> tuple:
    path = os.path.join(ROOT, *rel.split("/"))
    with open(path, encoding="utf-8", newline="") as f:
        raw = f.read()
    # Keep the file's own line endings so a rewrite only touches the blocks.
    return path, raw.replace("\r\n", "\n"), "\r\n" if "\r\n" in raw else "\n"


def fill(text: str) -> str:
    return BLOCK.sub(lambda m: f"<!-- report:{m[1]} -->\n{render(m[1])}\n<!-- /report:{m[1]} -->", text)


def stale() -> list:
    """Every problem --check reports: a block out of date, or the set of blocks in a file changed."""
    out = []
    for rel, names in DOCS.items():
        found = {m[1]: m[2] for m in BLOCK.finditer(_read(rel)[1])}
        if tuple(found) != names:
            out.append(f"{rel}: has blocks {list(found)}, expected {list(names)}")
        out += [f"{rel}: block {n} is stale" for n, body in found.items() if n in RENDER and body != render(n) + "\n"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="rewrite every block in place")
    mode.add_argument("--check", action="store_true", help="exit 1 if any block is stale")
    args = p.parse_args(argv)
    if args.write:
        for rel in DOCS:
            path, text, nl = _read(rel)
            with open(path, "w", encoding="utf-8", newline=nl) as f:
                f.write(fill(text))
    problems = stale()
    for line in problems:
        print(line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
