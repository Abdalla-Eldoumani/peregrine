"""Draw the results charts from the committed results JSON, headless.

    python benchmarks/plots.py    writes gflops_vs_n.png, ratio_heatmap.png and crossover.png
                                  to benchmarks/results/

Every point comes from report.py's reading of the same files, so a chart and
its table cannot disagree. A cell the protocol closed as noisy is never drawn as
a number: it is left out of the line charts and drawn gray in the heatmap. A
missing results file draws a "not measured" note instead of a chart.
"""

from __future__ import annotations

import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")  # before pyplot, so no display is ever needed

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors, patches  # noqa: E402

import report  # noqa: E402

SERIES_COLORS = {"cpu": "#1f77b4", "cuda": "#2ca02c", "numpy": "#7f7f7f", "cupy": "#ff7f0e"}
NOISY_COLOR = "#c8c8c8"
NOTE = "Medians; noisy cells are not drawn."

# Heatmap rows: (label, group, filter). Out of every ratio a cell carries, the
# heatmap shows the like-for-like one: out= against preallocated NumPy.
HEAT_ROWS = (
    ("matmul f32", "matmul_square", lambda r: r["dtype"] == "float32"),
    ("matmul f64", "matmul_square", lambda r: r["dtype"] == "float64"),
    ("matmul rect", "matmul_rect", lambda r: True),
    ("matmul layouts", "matmul_layout", lambda r: r["layout"] != "c"),
    ("transpose", "transpose", lambda r: True),
    ("sum", "reduce", lambda r: r["op"] == "sum"),
    ("mean", "reduce", lambda r: r["op"] == "mean"),
) + tuple(
    (f"{op}{' out=' if v == 'out' else ''}", "fused", lambda r, op=op, v=v: r["op"] == op and r["variant"] == v)
    for op in report.FUSED_OPS for v in ("alloc", "out")
)
# Size is the geometric mean of the dimensions (n for a square matmul); the
# edges follow the square matmul regimes.
BUCKETS = ((16, "≤ 16"), (96, "24 to 96"), (384, "128 to 384"), (768, "512 to 768"), (math.inf, "≥ 1000"))


def _note(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=9, wrap=True)


def _ok(rows: dict | None, keep) -> list:
    return sorted((r for r in (rows or {}).values() if r["status"] == "ok" and keep(r)), key=lambda r: r["shape"])


def plot_gflops_vs_n(path: str) -> str:
    cpu = report.matrix_rows("matmul_square", "cpu")
    gpu = report.matrix_rows("gpu", "cuda")
    ceil = report.load(report.CEILINGS)
    fig, axes = plt.subplots(1, len(report.DTYPES), figsize=(10, 4.2))
    for ax, dt in zip(axes, report.DTYPES):
        drawn = False
        for rows, keep, name, color, label in (
            (cpu, lambda r: r["op"] == "matmul", "peregrine", SERIES_COLORS["cpu"], "Peregrine CPU"),
            (cpu, lambda r: r["op"] == "matmul", "numpy", SERIES_COLORS["numpy"], "NumPy CPU"),
            (gpu, lambda r: r["op"] == "gpu_matmul", "peregrine", SERIES_COLORS["cuda"], "Peregrine GPU, device-resident"),
            (gpu, lambda r: r["op"] == "gpu_matmul", "cupy", SERIES_COLORS["cupy"], "CuPy, device-resident"),
        ):
            pts = [(r["shape"][0], r["rate"]["gflops"][name]) for r in _ok(rows, lambda r: keep(r) and r["dtype"] == dt)
                   if name in r["rate"].get("gflops", {})]
            if pts:
                threads = {r["threads"] for r in _ok(rows, lambda r: keep(r) and r["dtype"] == dt)}
                ax.plot(*zip(*pts), marker="o", ms=3, color=color,
                        label=f"{label}, {'/'.join(map(str, sorted(threads)))} threads" if rows is cpu else label)
                drawn = True
        if ceil is not None:
            top = max(f["threads"] for f in ceil["fma"])
            peak = next(f["total_gflops"] for f in ceil["fma"] if f["dtype"] == dt and f["threads"] == top)
            ax.axhline(peak, color=SERIES_COLORS["cpu"], linestyle="--", linewidth=1)
            ax.text(1, peak * 1.08, f"{top}-thread FMA peak {peak:.0f} GF/s", color=SERIES_COLORS["cpu"], fontsize=8)
        missing = [f for f, d in ((report.matrix_file("matmul_square", "cpu"), cpu),
                                  (report.matrix_file("gpu", "cuda"), gpu), (report.CEILINGS, ceil)) if d is None]
        if missing:
            ax.text(0.02, 0.02, "not measured: " + ", ".join(missing), transform=ax.transAxes, fontsize=7)
        if drawn:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_title(f"square matmul, {dt}")
        ax.set_xlabel("n")
        ax.set_ylabel("GF/s (2n³ / median time)")
        ax.grid(True, which="both", alpha=0.3)
    fig.text(0.01, 0.01, NOTE, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _bucket(shape: tuple) -> int:
    size = round(math.prod(shape) ** (1 / len(shape)), 6)
    return next(i for i, (edge, _) in enumerate(BUCKETS) if size <= edge)


def _heat_ratio(row: dict) -> float:
    return row["ratio"]["numpy_noalloc" if row["variant"] == "out" else "numpy"]


def plot_ratio_heatmap(path: str) -> str:
    groups = {g: report.matrix_rows(g, "cpu") for g in dict.fromkeys(g for _, g, _ in HEAT_ROWS)}
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    cmap = plt.get_cmap("RdBu")
    values = {}
    for i, (label, group, keep) in enumerate(HEAT_ROWS):
        rows = [r for r in (groups[group] or {}).values() if keep(r)]
        for j in range(len(BUCKETS)):
            cell = [r for r in rows if _bucket(r["shape"]) == j]
            ok = [_heat_ratio(r) for r in cell if r["status"] == "ok"]
            if ok:
                values[i, j] = statistics.median(ok)
            elif cell:
                values[i, j] = None
    span = max([abs(math.log10(v)) for v in values.values() if v] + [math.log10(4)])
    norm = colors.Normalize(-span, span)
    for i, (label, group, _) in enumerate(HEAT_ROWS):
        for j in range(len(BUCKETS)):
            if (i, j) not in values:
                text, face, ink = ("not measured" if groups[group] is None else ""), "white", "black"
            elif values[i, j] is None:
                text, face, ink = "noisy", NOISY_COLOR, "black"
            else:
                level = norm(math.log10(values[i, j]))
                text, face = report.fmt_ratio(values[i, j]), cmap(level)
                ink = "white" if abs(level - 0.5) > 0.32 else "black"
            ax.add_patch(patches.Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white"))
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=8, color=ink)
    ax.set_xlim(0, len(BUCKETS))
    ax.set_ylim(len(HEAT_ROWS), 0)
    ax.set_xticks([j + 0.5 for j in range(len(BUCKETS))], [b for _, b in BUCKETS], fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(HEAT_ROWS))], [r[0] for r in HEAT_ROWS], fontsize=8)
    ax.set_xlabel("size: n for square matmul, geometric mean of the dimensions otherwise")
    ax.set_title("NumPy time / Peregrine time, median per bucket (CPU-only build)", fontsize=10)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04)
    ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32) if abs(math.log10(t)) <= span]
    bar.set_ticks([math.log10(t) for t in ticks], labels=[f"{t:g}x" for t in ticks])
    fig.text(0.01, 0.01, "Out= rows against preallocated NumPy, the rest against the NumPy expression; "
             "gray buckets hold only noisy cells.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_crossover(path: str) -> str:
    gpu = report.matrix_rows("gpu", "cuda")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    square = lambda r: r["variant"] == "auto" and len(set(r["shape"])) == 1  # noqa: E731
    rows = _ok(gpu, square)
    drawn = False
    for name, label, color, style in (
        ("peregrine_cpu", "forced CPU", SERIES_COLORS["cpu"], "-"),
        ("peregrine_gpu", "forced GPU, transfers staged", SERIES_COLORS["cuda"], "-"),
        ("peregrine", "auto", "black", "--"),
        ("numpy", "NumPy", SERIES_COLORS["numpy"], "-"),
        ("cupy", "CuPy", SERIES_COLORS["cupy"], "-"),
    ):
        pts = [(r["shape"][0], r["stats"][name]["median"]) for r in rows if name in r["stats"]]
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=3, color=color, linestyle=style, label=label)
            drawn = True
    if gpu is None:
        _note(ax, f"not measured: {report.matrix_file('gpu', 'cuda')}")
    elif not drawn:
        _note(ax, "every auto-routed cell in this file is noisy, so none is drawn")
    else:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    ax.set_xlabel("n (square float32 host arrays)")
    ax.set_ylabel("median time per call (s)")
    ax.set_title("Host float32 matmul: auto route against both forced routes (CUDA build)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.text(0.01, 0.01, NOTE, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_all(out_dir: str | None = None) -> list:
    out_dir = out_dir or report.RESULTS_DIR
    return [
        plot_gflops_vs_n(os.path.join(out_dir, "gflops_vs_n.png")),
        plot_ratio_heatmap(os.path.join(out_dir, "ratio_heatmap.png")),
        plot_crossover(os.path.join(out_dir, "crossover.png")),
    ]


if __name__ == "__main__":
    for p in generate_all():
        print(f"wrote {p}")
