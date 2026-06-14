"""Generate the three README charts from the committed results JSON, headless.

bench-protocol rule 19: the report carries three plots with fixed series colors,
GFLOP/s vs n per backend, speedup bars per regime, and a crossover chart. This
module produces exactly those three, reading only committed
benchmarks/results/*.json (rule 17, the same round-trip rule the table renderer
obeys) and writing PNGs back to benchmarks/results/.

The Agg backend is selected BEFORE importing pyplot so the figures render with no
display (verified headless: savefig writes a non-empty PNG with no DISPLAY). The
four series colors are the exact DESIGN_SYSTEM tokens, consistent across every
chart; the cupy color stays defined for consistency even though cupy is absent on
this machine (the GPU baseline is NumPy CPU per rule 15, so no cupy series is
plotted).

json.load only (Security V5): every source file is first-party data read with
json, never eval or pickle. The matmul / gpu / fused / scaling files go through
report.load_series (manifest + verified gate); the calibration file is the
calibration-cache shape (schema / signature / cpu / gpu / transfer, no manifest),
read with a plain json.load plus a schema check.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless; MUST precede the pyplot import (rule 19 / Agg)

import matplotlib.pyplot as plt  # noqa: E402

import report  # noqa: E402

# The exact DESIGN_SYSTEM series colors (lines 262-263). Consistent across every
# chart. cupy stays defined for consistency though no cupy series is plotted
# (cupy is absent; the GPU baseline is NumPy CPU per rule 15).
SERIES_COLORS = {
    "cpu": "#1f77b4",
    "cuda": "#2ca02c",
    "numpy": "#7f7f7f",
    "cupy": "#ff7f0e",
}

RESULTS_DIR = report.RESULTS_DIR


def _results_path(*parts: str) -> str:
    return os.path.join(RESULTS_DIR, *parts)


def plot_gflops_vs_n(out_path: str) -> str:
    """GFLOP/s vs n per backend: CPU f64, GPU device-resident f32, NumPy CPU f64.

    CPU floor GFLOP/s comes from the cpu02 f64 cases (gflop / fastmathext.min_s),
    NumPy floor from the same cases (gflop / numpy.min_s), and the GPU
    device-resident warm GFLOP/s from the committed gpu matrix without-transfer
    warm series. Every value is read from JSON (rule 17); the colors are the
    fixed series tokens.
    """
    cpu = report.load_series(_results_path("cpu02_f64_zpicy.json"))
    gpu = report.load_series(_results_path("zpicy_gpu_matrix.json"))

    cpu_n = [c["n"] for c in cpu["cases"]]
    cpu_gflops = [c["gflop"] / c["fastmathext"]["min_s"] for c in cpu["cases"]]
    numpy_gflops = [c["gflop"] / c["numpy"]["min_s"] for c in cpu["cases"]]

    warm = sorted(
        (
            c
            for c in gpu["gpu_cases"]
            if c["label"] == "without-transfer" and c["phase"] == "warm"
        ),
        key=lambda c: c["n"],
    )
    gpu_n = [c["n"] for c in warm]
    gpu_gflops = [c["gflops"] for c in warm]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        cpu_n, cpu_gflops, marker="o", color=SERIES_COLORS["cpu"],
        label="FastMathExt CPU f64 (floor)",
    )
    ax.plot(
        cpu_n, numpy_gflops, marker="s", color=SERIES_COLORS["numpy"],
        label="NumPy CPU f64 (floor)",
    )
    ax.plot(
        gpu_n, gpu_gflops, marker="^", color=SERIES_COLORS["cuda"],
        label="FastMathExt GPU f32 (device-resident, warm)",
    )
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("GFLOP/s vs n per backend")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_speedup_bars(out_path: str) -> str:
    """Speedup bars per regime, wins and losses side by side.

    Each bar is a regime's headline speedup read from JSON: CPU-02 f64 floor at
    n=2048 (a loss, below 1.0), CPU-06 small at n=64 (a loss), the GPU
    device-resident win, the GPU with-transfer small loss, and the fused CPU and
    GPU wins. The 1.0 parity line marks win vs loss; loss bars are not hidden
    (rule 16).
    """
    cpu02 = report.load_series(_results_path("cpu02_f64_zpicy.json"))
    cpu06 = report.load_series(_results_path("cpu06_f32_zpicy.json"))
    gpu = report.load_series(_results_path("zpicy_gpu_matrix.json"))
    fuse_cpu = report.load_series(_results_path("fuse05_cpu_f32_zpicy.json"))
    fuse_gpu = report.load_series(_results_path("fuse05_gpu_f32_zpicy.json"))

    cpu02_2048 = report._case_by_n(cpu02["cases"], 2048)
    cpu02_floor = (
        cpu02_2048["numpy"]["min_s"] / cpu02_2048["fastmathext"]["min_s"]
    )
    cpu06_64 = report._case_by_n(cpu06["cases"], 64)["speedup_vs_numpy"]
    gpu08 = report._gpu08_ratio(gpu)
    transfer_small = report._case_by_n(
        [c for c in gpu["with_transfer_cases"] if c["phase"] == "warm"], 256
    )["ratio_vs_numpy_cpu_f32"]
    fuse_cpu_floor = fuse_cpu["cpu_cases"][0]["speedup_floor"]
    fuse_gpu_ratio = fuse_gpu["gpu_cases"][0]["ratio_vs_numpy_cpu_f32"]

    labels = [
        "CPU f64\nn=2048",
        "CPU small\nn=64",
        "GPU f32\ndevice",
        "GPU f32\n+transfer n=256",
        "Fused\nCPU",
        "Fused\nGPU",
    ]
    values = [
        cpu02_floor,
        cpu06_64,
        gpu08,
        transfer_small,
        fuse_cpu_floor,
        fuse_gpu_ratio,
    ]
    # Color a bar by backend: CPU regimes cpu-blue, GPU regimes cuda-green.
    colors = [
        SERIES_COLORS["cpu"],
        SERIES_COLORS["cpu"],
        SERIES_COLORS["cuda"],
        SERIES_COLORS["cuda"],
        SERIES_COLORS["cpu"],
        SERIES_COLORS["cuda"],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.axhline(1.0, color=SERIES_COLORS["numpy"], linestyle="--", label="parity")
    ax.set_ylabel("speedup vs NumPy (x)")
    ax.set_title("Speedup per regime (wins and losses)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _load_calibration(path: str) -> dict:
    """Read the calibration-cache JSON (the crossover source); json.load only.

    The calibration cache is a different shape from the results files: it carries
    schema / signature / cpu / gpu / transfer, NOT a manifest + cases, so
    report.load_series does not apply. Validate the fields the crossover needs are
    present. json.load only (Security V5): a cache file is data, never code.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    assert "cpu" in data and "gpu" in data and "transfer" in data, (
        "calibration cache missing cpu/gpu/transfer (crossover source)"
    )
    return data


def plot_crossover(out_path: str, calibration_path: str | None = None):
    """Crossover chart: host CPU time vs GPU compute + host transfer, per n (f32).

    Shows the n where the GPU wins for a HOST float32 array once the H2D+D2H
    transfer is paid, which is the decision the auto policy makes. CPU time is the
    calibrated cpu.float32 curve; the GPU host cost is gpu.float32 compute plus the
    modeled transfer time (bytes / h2d_gbps + bytes / d2h_gbps + fixed_overhead),
    all read from the committed calibration cache.

    Returns the out_path on success, or None when the calibration cache is absent
    (the chart is skipped with a clear reason rather than fabricating data).
    """
    if calibration_path is None:
        calibration_path = _results_path("zpicy_calibration.json")
    if not os.path.exists(calibration_path):
        # No calibration cache committed: skip rather than invent a curve. The
        # caller logs the skip; a fabricated crossover would be an orphan number.
        return None
    cal = _load_calibration(calibration_path)

    cpu_pts = sorted(cal["cpu"]["float32"], key=lambda p: p[0])
    gpu_pts = sorted(cal["gpu"]["float32"], key=lambda p: p[0])
    transfer = cal["transfer"]
    h2d = transfer["h2d_gbps"]
    d2h = transfer["d2h_gbps"]
    fixed = transfer["fixed_overhead_s"]

    ns = [n for n, _ in cpu_pts]
    cpu_t = [t for _, t in cpu_pts]
    gpu_by_n = {n: t for n, t in gpu_pts}
    gpu_host_t = []
    for n in ns:
        # f32 square: two input matrices in, one out; bytes moved is the H2D of
        # both operands plus the D2H of the result. 4 bytes per f32 element.
        in_bytes = 2 * n * n * 4
        out_bytes = n * n * 4
        transfer_s = (
            in_bytes / (h2d * 1e9) + out_bytes / (d2h * 1e9) + fixed
        )
        gpu_host_t.append(gpu_by_n.get(n, float("nan")) + transfer_s)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        ns, cpu_t, marker="o", color=SERIES_COLORS["cpu"],
        label="CPU compute (host)",
    )
    ax.plot(
        ns, gpu_host_t, marker="^", color=SERIES_COLORS["cuda"],
        label="GPU compute + host transfer",
    )
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("time per matmul (s)")
    ax.set_title("Host float32 crossover: CPU vs GPU (with transfer)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def generate_all(out_dir: str | None = None) -> list:
    """Write the three charts to out_dir (default benchmarks/results/).

    Returns the list of PNG paths actually written. The crossover is omitted from
    the list (with a printed reason) when the calibration cache is absent.
    """
    if out_dir is None:
        out_dir = RESULTS_DIR
    written = []
    written.append(plot_gflops_vs_n(os.path.join(out_dir, "gflops_vs_n.png")))
    written.append(plot_speedup_bars(os.path.join(out_dir, "speedup_bars.png")))
    crossover = plot_crossover(os.path.join(out_dir, "crossover.png"))
    if crossover is None:
        print(
            "crossover chart skipped: no calibration cache at "
            f"{_results_path('zpicy_calibration.json')}"
        )
    else:
        written.append(crossover)
    return written


if __name__ == "__main__":
    for path in generate_all():
        print(f"wrote {path}")
