"""The results pipeline: benchmarks/report.py, benchmarks/plots.py, and the docs they fill.

Pinned here: every generated block in README.md and docs/benchmarks.md equals a
fresh render; outside the blocks and code fences neither file carries a
performance number; every loss has a reason; noisy cells never show a number;
the target rules, on small synthetic results files; the plots' backend, palette
and output; json.load as the only way either module reads a results file; and
no reference to private tooling paths in the published docs.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT_PATH = pathlib.Path(_REPO_ROOT)
_BENCH_DIR = os.path.join(_REPO_ROOT, "benchmarks")
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

import report  # noqa: E402

# matplotlib lives in the [bench] extra, outside [test], so the plot tests skip
# cleanly on a [test]-only install; the report tests need no matplotlib.
try:
    import matplotlib  # noqa: F401

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

requires_matplotlib = pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed; pip install -e .[bench]")

_DOCS = [os.path.join(_REPO_ROOT, *rel.split("/")) for rel in report.DOCS]

# Private tooling paths that must never appear in a tracked public file. The
# tokens are assembled from fragments so this file does not itself contain the
# strings it forbids.
_DOT = "."
_MD = _DOT + "md"
_C = "cl" + "au" + "de"
_INFRA_TOKENS = (
    _DOT + "ag" + "ent",
    _DOT + "plan" + "ning",
    _DOT + _C,
    _C.upper() + _MD,
    "g" + "sd",
    "get-" + "shit-" + "done",
)

# A performance quantity: a ratio such as 2.5x, or a number with a rate or time unit.
_PERF_NUMBER = re.compile(r"\b\d+(?:\.\d+)?x\b|\b\d+(?:\.\d+)?\s?(?:GF/s|GFLOP/s|GB/s|ms|us|µs|ns)\b")
_FENCE = re.compile(r"^```.*?^```", re.S | re.M)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def readme_hygiene(text: str) -> list:
    """The private tooling tokens found in text; empty means clean."""
    return [token for token in _INFRA_TOKENS if token in text]


def _prose(text: str) -> str:
    return _FENCE.sub("", report.BLOCK.sub("", text))


# --- Synthetic results files. ---


def _cell(op, dtype, shape, times, *, layout="c", variant="-", threads=6, seed=0, status="ok", **extra):
    cid = f"{op}/{dtype}/{'x'.join(map(str, shape))}/{layout}/{variant}/t{threads}/s{seed}"
    cell = {"id": cid, "op": op, "dtype": dtype, "shape": list(shape), "layout": layout, "variant": variant,
            "threads": threads, "seed": seed, "status": status,
            "contenders": {n: {"median_s": t, "q1_s": t * 0.99, "q3_s": t * 1.01, "min_s": t * 0.98}
                           for n, t in times.items()}, **extra}
    if status == "ok" and len(times) > 1:
        cell["ratio"] = {n: t / times["peregrine"] for n, t in times.items() if n != "peregrine"}
    return cell


def _write(tmp_path, name, cells):
    (tmp_path / name).write_text(json.dumps({"manifest": {}, "cells": cells}), encoding="utf-8")


def _statuses(group, build="cpu"):
    return {k: report.status(r) for k, r in report.matrix_rows(group, build).items()}


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_DIR", str(tmp_path))
    return tmp_path


# --- The docs blocks. ---


def test_blocks_round_trip():
    # Every block in both files equals a fresh render, and each file carries
    # exactly its expected blocks in order.
    assert report.stale() == []


def test_no_performance_number_outside_blocks():
    assert _PERF_NUMBER.findall("2.5x faster, 10 GB/s, 3 ms, 1.2 us, 734 GF/s") == \
        ["2.5x", "10 GB/s", "3 ms", "1.2 us", "734 GF/s"]
    assert _PERF_NUMBER.findall("a 64x64 matrix on Python 3.12 with 4 KiB pages") == []
    for path in _DOCS:
        assert _PERF_NUMBER.findall(_prose(_read_text(path))) == [], path


def test_every_loss_has_reason():
    for _, row, base, ratio, reason in report.losses():
        assert ratio < 1.0 and reason, row["key"]
    assert "Below 1.00x" in report.render("losses")


def test_loss_without_reason_raises(synthetic, monkeypatch):
    _write(synthetic, "refbox_matmul_rect_cpu_v310.json",
           [_cell("matmul", "float32", (4096, 4096, 1), {"peregrine": 2.0, "numpy": 1.0})])
    monkeypatch.setattr(report, "REASONS", ())
    with pytest.raises(ValueError, match="no REASONS entry"):
        report.render("losses")
    monkeypatch.setattr(report, "REASONS", ((r"^matmul/", "the reason"),))
    text = report.render("losses")
    assert "0.50x" in text and "the reason" in text


def test_noisy_cells_render_without_a_number(synthetic):
    _write(synthetic, "refbox_matmul_square_cpu_v310.json", [
        _cell("matmul", "float32", (512, 512, 512), {"peregrine": 0.0123, "numpy": 0.0456}, status="noisy"),
        _cell("matmul", "float32", (64, 64, 64), {"peregrine": 1e-6, "numpy": 2e-6}),
    ])
    published = ("12.3 ms", "45.6 ms", "3.71x")  # its medians and ratio, were they shown
    for name in ("matrix_matmul_square_cpu", "headline"):
        lines = [ln for ln in report.render(name).splitlines() if "512x512x512" in ln and "not measured" not in ln]
        assert lines and all("| noisy |" in ln for ln in lines), name
        assert not [s for s in published for ln in lines if s in ln], name
    losses = report.render("losses")
    assert "`matmul/float32/512x512x512/c/-/t6/s0`" in losses
    assert not [s for s in published if s in losses]


def test_missing_files_render_not_measured(synthetic):
    for name in report.RENDER:
        assert "Not measured" in report.render(name), name


# --- Target rules on synthetic files. ---


def test_a_failing_render_leaves_the_docs_untouched(tmp_path, monkeypatch):
    for rel in report.DOCS:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((_REPO_ROOT_PATH / rel).read_bytes())
    before = {rel: (tmp_path / rel).read_bytes() for rel in report.DOCS}
    monkeypatch.setattr(report, "ROOT", str(tmp_path))

    def boom(name):
        raise ValueError("a loss without a reason")

    monkeypatch.setattr(report, "render", boom)
    with pytest.raises(ValueError):
        report.main(["--write"])
    assert {rel: (tmp_path / rel).read_bytes() for rel in report.DOCS} == before


def test_square_regime_boundaries(synthetic):
    # Each ratio passes under the cell's own regime and fails under the next
    # regime's target, or the reverse, so a shifted boundary flips a status.
    cases = {16: (2.9, "below"), 24: (1.6, "met"), 96: (1.4, "below"), 128: (1.2, "met"),
             384: (1.1, "below"), 512: (1.1, "met"), 768: (1.0, "below"), 1000: (1.0, "met")}
    # float32, because the float64 n = 12 and 16 cells carry a deferral closure.
    _write(synthetic, "refbox_matmul_square_cpu_v310.json",
           [_cell("matmul", "float32", (n, n, n), {"peregrine": 1.0, "numpy": r}) for n, (r, _) in cases.items()]
           + [_cell("matmul", "float64", (16, 16, 16), {"peregrine": 1.0, "numpy": 2.9})])
    got = _statuses("matmul_square")
    assert {n: got[f"matmul/float32/{n}x{n}x{n}/c/-/t6"] for n in cases} == {n: s for n, (_, s) in cases.items()}
    assert got["matmul/float64/16x16x16/c/-/t6"] == "deferred"


def test_other_regimes(synthetic):
    _write(synthetic, "refbox_matmul_rect_cpu_v310.json", [
        _cell("matmul", "float32", (1, 4096, 4096), {"peregrine": 1.0, "numpy": 1.01}),
        _cell("matmul", "float32", (4096, 64, 4096), {"peregrine": 1.0, "numpy": 0.99}),
    ])
    _write(synthetic, "refbox_transpose_cpu_v310.json", [
        _cell("transpose", "float32", shape, {"peregrine": 1.0, "numpy": 3.0})
        for shape in ((256, 256), (512, 512), (1000, 3000), (4096, 64))])
    _write(synthetic, "refbox_reduce_cpu_v310.json", [
        _cell("sum", "float32", shape, {"peregrine": 1.0, "numpy": 2.0}, variant="axis=0")
        for shape in ((64, 64), (64, 128))])
    _write(synthetic, "refbox_gemm_1t_cpu_v310.json", [
        _cell("matmul", "float64", (n, n, n), {"peregrine": 1.0, "numpy": 0.96}, threads=1) for n in (512, 1024)])
    assert _statuses("matmul_rect") == {"matmul/float32/1x4096x4096/c/-/t6": "met",
                                        "matmul/float32/4096x64x4096/c/-/t6": "below"}
    assert _statuses("transpose") == {"transpose/float32/256x256/c/-/t6": "met",
                                      "transpose/float32/512x512/c/-/t6": "below",
                                      "transpose/float32/1000x3000/c/-/t6": "below",
                                      "transpose/float32/4096x64/c/-/t6": "met"}
    assert _statuses("reduce") == {"sum/float32/64x64/c/axis=0/t6": "below",
                                   "sum/float32/64x128/c/axis=0/t6": "met"}
    assert _statuses("gemm_1t") == {"matmul/float64/512x512x512/c/-/t1": "no target",
                                    "matmul/float64/1024x1024x1024/c/-/t1": "met"}


def test_fused_pairing(synthetic):
    # Up to 64x64 both baselines hold; above, each variant meets only the
    # baseline that does its allocation work, and the other ratio is still a
    # loss when it falls below one.
    _write(synthetic, "refbox_fused_cpu_v310.json", [
        _cell("fma3", "float32", (64, 64), {"peregrine": 1.0, "numpy": 3.1, "numpy_noalloc": 1.9}, variant="alloc"),
        _cell("fma3", "float32", (64, 64), {"peregrine": 1.0, "numpy": 3.1, "numpy_noalloc": 2.1}, variant="out"),
        _cell("fma3", "float32", (128, 128), {"peregrine": 1.0, "numpy": 2.1, "numpy_noalloc": 0.9}, variant="alloc"),
        _cell("fma3", "float32", (128, 128), {"peregrine": 1.0, "numpy": 5.0, "numpy_noalloc": 1.9}, variant="out"),
    ])
    assert _statuses("fused") == {"fma3/float32/64x64/c/alloc/t6": "below", "fma3/float32/64x64/c/out/t6": "met",
                                  "fma3/float32/128x128/c/alloc/t6": "met", "fma3/float32/128x128/c/out/t6": "below"}
    assert [(row["key"], base) for _, row, base, _, _ in report.losses()] == \
        [("fma3/float32/128x128/c/alloc/t6", "numpy_noalloc")]


def test_layout_ten_percent_rule(synthetic):
    shape = (512, 512, 512)
    _write(synthetic, "refbox_matmul_layout_cpu_v310.json", [
        _cell("matmul", "float32", shape, {"peregrine": 1.00, "numpy": 1.3}),
        _cell("matmul", "float32", shape, {"peregrine": 1.09, "numpy": 1.3}, layout="at_b"),
        _cell("matmul", "float32", shape, {"peregrine": 1.11, "numpy": 1.3}, layout="a_bt"),
        _cell("matmul", "float32", shape, {"peregrine": 1.30, "numpy": 1.3}, layout="ff"),
        _cell("matmul", "float32", (64, 64, 64), {"peregrine": 1.0, "numpy": 2.0}, status="noisy"),
        _cell("matmul", "float32", (64, 64, 64), {"peregrine": 1.0, "numpy": 2.0}, layout="ff"),
    ])
    got = _statuses("matmul_layout")
    assert got["matmul/float32/512x512x512/c/-/t6"] == "met"
    assert got["matmul/float32/512x512x512/at_b/-/t6"] == "met"
    assert got["matmul/float32/512x512x512/a_bt/-/t6"] == "below"  # ratio passes, 11 percent over own C
    assert got["matmul/float32/512x512x512/ff/-/t6"] == "below"  # ratio 1.00x fails the 1.05x target too
    assert got["matmul/float32/64x64x64/ff/-/t6"] == "reference noisy"


def test_auto_five_percent_rule(synthetic):
    def auto(n, auto_t, cpu_t, gpu_t, numpy_t):
        return _cell("matmul", "float32", (n, n, n), {"peregrine": auto_t, "numpy": numpy_t,
                                                      "peregrine_cpu": cpu_t, "peregrine_gpu": gpu_t},
                     variant="auto")

    _write(synthetic, "refbox_gpu_cuda_v310.json", [
        auto(512, 1.04, 1.0, 2.0, 1.5), auto(1024, 1.06, 2.0, 1.0, 9.0),
        auto(2048, 1.0, 3.0, 1.02, 4.9), auto(4096, 1.03, 9.0, 1.0, 6.18),
    ])
    assert _statuses("gpu", "cuda") == {"matmul/float32/512x512x512/c/auto/t6": "met",
                                        "matmul/float32/1024x1024x1024/c/auto/t6": "below",
                                        "matmul/float32/2048x2048x2048/c/auto/t6": "below",
                                        "matmul/float32/4096x4096x4096/c/auto/t6": "met"}
    assert "+4.0%" in report.render("matrix_gpu_cuda")


def test_closure_numbers_come_from_json(synthetic, monkeypatch):
    key = "fma3/float64/1024x1024/c/out/t6"
    _write(synthetic, "refbox_fused_cpu_v310.json", [
        _cell("fma3", "float64", (1024, 1024), {"peregrine": 1.0, "numpy": 4.7, "numpy_noalloc": 1.85},
              variant="out", gbps={"peregrine": 36.3, "numpy": 7.7, "numpy_noalloc": 19.6})])
    monkeypatch.setattr(report, "CLOSURES", ((re.escape(key), {
        "kind": "at ceiling", "ceiling": ("memory", "triad", "nontemporal", 6), "why": "memory bound"}),))

    def ceilings(gbs):
        (synthetic / report.CEILINGS).write_text(json.dumps({"memory": {"runs": [{"threads": 6, "kernels": [
            {"kernel": "triad", "stores": "nontemporal", "median_gbs": gbs}]}]}}), encoding="utf-8")

    ceilings(21.3)
    assert _statuses("fused")[key] == "at ceiling"
    text = report.render("losses")
    assert all(s in text for s in ("1.85x", "36.3 GB/s", "21.3 GB/s", "170%", "memory bound"))
    ceilings(50.0)  # Peregrine at 73 percent of this ceiling cannot close at it
    with pytest.raises(ValueError, match="at-ceiling"):
        report.render("losses")


# --- What the modules may read. ---


def test_results_read_with_json_load_only():
    for name in ("report.py", "plots.py"):
        tree = ast.parse(_read_text(os.path.join(_BENCH_DIR, name)))
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not imported & {"pickle", "marshal", "shelve", "yaml", "dill"}, name
        calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert not calls & {"eval", "exec", "compile", "__import__"}, name
        # Only committed results: no string in code names the gitignored local folder.
        docs = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
        literals = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in docs]
        assert all("local" not in lit for lit in literals), name
    assert "json.load(" in _read_text(os.path.join(_BENCH_DIR, "report.py"))


# --- The plots. ---


@requires_matplotlib
def test_plot_colors():
    import plots

    assert plots.SERIES_COLORS == {"cpu": "#1f77b4", "cuda": "#2ca02c", "numpy": "#7f7f7f", "cupy": "#ff7f0e"}


@requires_matplotlib
def test_plots_use_agg_backend():
    import matplotlib

    import plots  # noqa: F401  (the import selects Agg)

    assert matplotlib.get_backend().lower() == "agg"


@requires_matplotlib
def test_plots_smoke(tmp_path):
    import plots

    paths = plots.generate_all(str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ["gflops_vs_n.png", "ratio_heatmap.png", "crossover.png"]
    for path in paths:
        assert os.path.getsize(path) > 0


# --- Hygiene. ---


def test_readme_hygiene_helper():
    planning_tok = "." + "plan" + "ning"
    doc_tok = _C.upper() + "." + "md"
    assert readme_hygiene("plain technical prose with no internal path") == []
    assert planning_tok in readme_hygiene(f"see {planning_tok}/" + "STATE" + ".md for details")
    assert doc_tok in readme_hygiene(f"the rule is in {doc_tok}")


def test_docs_no_agent_infra_path():
    for path in _DOCS + [os.path.join(_BENCH_DIR, "report.py"), os.path.join(_BENCH_DIR, "plots.py")]:
        hits = readme_hygiene(_read_text(path))
        assert hits == [], f"{path} references private tooling paths: {hits}"
