"""Render the README per-regime tables from the committed results JSON.

bench-protocol rule 17: a benchmark number enters the README only by being
regenerated from the saved JSON, never hand-typed. This module is that
regenerator. Every value in every table it emits is read from a
benchmarks/results/*.json file with json.load (Security V5: a results file is
data, never code, so never eval or pickle one), so the README and the JSON
cannot drift.

bench-protocol rule 16: the results section MUST carry the regimes where NumPy
or OpenBLAS wins or ties. float64 mid-size parity is the stated target, not a
win, so a table with no loss row is the red flag the protocol exists to catch.
The loss rows here are CPU-02 (f64 floor below OpenBLAS), CPU-05 (thread scaling
below the 4x ideal), and CPU-06 (the small-matrix path losing to NumPy).

The four result shapes the renderer branches on (the schema differs per file):
- matmul / gpu matrix: top-level ``cases`` (CPU pairs, each with nested
  ``fastmathext`` / ``numpy`` timing dicts, plus per-row ``speedup_floor`` and
  ``speedup_vs_numpy`` derived from that per-rep timing) plus ``gpu_cases``
  (device-resident, each carrying ``ratio_vs_numpy_cpu_f32``) and optional
  ``with_transfer_cases``. The CPU-02 n=2048 parity headline is recomputed from
  the per-rep ``cases`` of the fresh matmul f64 sweep, not from any hand-entered
  summary scalar.
- fused: top-level ``cpu_cases`` / ``gpu_cases`` with precomputed
  ``speedup_floor`` (CPU) and ``ratio_vs_numpy_cpu_f32`` (GPU).
- scaling: top-level ``series`` plus a precomputed ``ratio_6_vs_1_floor``.

The floor (min) is the headline statistic, not the median: a background
antivirus preempts the OpenMP barrier on this machine and inflates the median of
every multi-threaded CPU series (the per-call-floor verdict). The renderer reads
the floor fields the saved JSON already carries.
"""

from __future__ import annotations

import json
import os

# The committed results directory. results/local/ is gitignored scratch and is
# never read here: every file this module opens must survive a fresh clone, or
# the README number it feeds would be an orphan. The device-resident GPU row in
# particular reads results/refbox_gpu_matrix.json (the committed output), NOT
# results/local/gpu08.json (absent on a clone).
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _results_path(*parts: str) -> str:
    return os.path.join(RESULTS_DIR, *parts)


def _case_is_verified(case: dict) -> bool:
    # rule 11 acceptance, accommodating the two committed CPU-case shapes:
    #   1. The bench_matmul / gpu / fused / scaling shape carries an explicit
    #      per-case ``verified`` boolean (True to publish).
    #   2. The paired-timing shape (cpu02_f64_refbox / cpu06_f32_refbox)
    #      records each case as a paired ``fastmathext`` + ``numpy`` timing dict
    #      with NO per-case ``verified`` flag; the per-rep verification ran
    #      against assert_matmul_close (the bench writes no unverified series: no
    #      result with verified false ever reaches a results file), and the
    #      analysis-block verdict documents it. A paired CPU case is therefore
    #      verified by construction.
    # An EXPLICIT verified=False is always a reject (rule 11), regardless of
    # shape: a series the bench marked unverified must never be rendered.
    if case.get("verified") is False:
        return False
    if case.get("verified") is True:
        return True
    return "fastmathext" in case and "numpy" in case


def load_series(path: str) -> dict:
    """Load a results JSON and assert it is publishable; return the dict.

    rule 9: no manifest, no merge. rule 11: a verified-false series never
    reaches a results file, so it must never be rendered. A ``high_cv=true``
    case is STILL ``verified=true`` (the noise flag documents the residual, it
    never strips the series), so this validator passes that combination through
    unchanged; the AV-noisy CPU sweeps this module consumes are high_cv on every
    real run.

    Two committed CPU-case shapes exist (see ``_case_is_verified``): the
    bench_matmul / gpu / fused / scaling shape with an explicit per-case
    ``verified`` boolean, and the paired-timing shape
    (cpu02_f64_refbox / cpu06_f32_refbox) whose paired ``fastmathext`` + ``numpy``
    timing case is verified-per-rep by construction with no per-case flag. Both
    are accepted; an EXPLICIT ``verified: False`` is rejected in either shape.

    json.load ONLY (Security V5): a results file is untrusted data on disk;
    deserializing it with eval or pickle would execute arbitrary code, json
    cannot. There is no cupy import anywhere in this path either (cupy is absent
    on this machine; the GPU baseline is NumPy CPU per rule 15).

    Parameters
    ----------
    path : str
        Absolute path to a results JSON file.

    Returns
    -------
    dict
        The parsed results dictionary, unchanged.

    Raises
    ------
    AssertionError
        If the file carries no manifest, or any case in cases / cpu_cases /
        gpu_cases / with_transfer_cases / series is not verified. The scaling
        file stores its rows under ``series``, so that key is gated too: a
        verified-false scaling row must not feed the README scaling line.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    assert "manifest" in data, f"rule 9: no manifest, no merge ({path})"
    for key in ("cases", "cpu_cases", "gpu_cases", "with_transfer_cases", "series"):
        for case in data.get(key, []):
            assert _case_is_verified(
                case
            ), f"rule 11: verified-false series in {path}"
    return data


def _case_by_n(cases: list, n: int) -> dict:
    for case in cases:
        if case.get("n") == n:
            return case
    raise KeyError(f"no case with n={n}")


def _fmt(value: float, places: int = 2) -> str:
    return f"{value:.{places}f}"


def render_matmul_f64_table(data: dict, sweep: dict) -> str:
    """The f64 matmul floor table (CPU-02). A LOSS regime (rule 16).

    The per-size floor ratio is ``numpy.min_s / fastmathext.min_s`` from each
    ``cases`` entry (the floor, not the median: AV noise inflates the median).

    The headline n=2048 parity figure comes from ``sweep`` -- the fresh 07-02
    matmul f64 sweep (refbox_matmul_f64.json), which carries full per-rep timing
    behind every row (``fastmathext.min_s`` / ``numpy.min_s`` over 30 reps,
    ``verified: true``). Both the floor ratio (``speedup_floor``, the noise-robust
    headline) and the median ratio (``speedup_vs_numpy``) are recomputed from that
    per-rep data, so the quoted number regenerates from saved measurement, not
    from a hand-entered summary scalar. The honest story is unchanged: at f64
    mid-size FastMathExt does not beat OpenBLAS.
    """
    cases = data["cases"]
    lines = [
        "| n | FastMathExt floor (GFLOP/s) | OpenBLAS floor (GFLOP/s) "
        "| floor ratio |",
        "| --- | --- | --- | --- |",
    ]
    for case in cases:
        n = case["n"]
        gflop = case["gflop"]
        fme_min = case["fastmathext"]["min_s"]
        np_min = case["numpy"]["min_s"]
        fme_gflops = gflop / fme_min
        np_gflops = gflop / np_min
        ratio = np_min / fme_min
        lines.append(
            f"| {n} | {_fmt(fme_gflops, 1)} | {_fmt(np_gflops, 1)} "
            f"| {_fmt(ratio)} |"
        )
    n2048 = _case_by_n(sweep["cases"], 2048)
    floor_ratio = n2048["speedup_floor"]
    median_ratio = n2048["speedup_vs_numpy"]
    lines.append("")
    lines.append(
        f"At n=2048 the fresh per-rep sweep measures {_fmt(floor_ratio)} of "
        f"OpenBLAS on the floor and {_fmt(median_ratio)} on the median. float64 "
        "mid-size parity is the target; FastMathExt does not beat OpenBLAS here."
    )
    return "\n".join(lines)


def render_small_matrix_table(data: dict) -> str:
    """The small-matrix f32 table (CPU-06). A LOSS regime (rule 16).

    ``speedup_vs_numpy`` per ``cases`` entry is end-to-end Python time
    (numpy_median / fme_median). Below 1.0 at every size: the in-tree small path
    loses to NumPy's OpenBLAS small-GEMM dispatch plus the irreducible
    Python+nanobind round-trip floor.
    """
    cases = data["cases"]
    lines = [
        "| n | speedup vs NumPy | outcome |",
        "| --- | --- | --- |",
    ]
    for case in cases:
        n = case["n"]
        speedup = case["speedup_vs_numpy"]
        outcome = "loses to NumPy" if speedup < 1.0 else "faster than NumPy"
        lines.append(f"| {n} | {_fmt(speedup)}x | {outcome} |")
    return "\n".join(lines)


def render_scaling_line(data: dict) -> str:
    """The thread-scaling line (CPU-05). A LOSS regime vs the 4x ideal (rule 16).

    ``ratio_6_vs_1_floor`` is the floor speedup from 1 to 6 threads, well below a
    linear 6x and below the 4x the four physical-core plateau would suggest: the
    kernel stops scaling past four threads on this part.
    """
    ratio = data["ratio_6_vs_1_floor"]
    n = data["n"]
    return (
        f"1 to 6 thread scaling at n={n} (f64): {_fmt(ratio)}x floor speedup, "
        "below the 4x the four-core plateau would give and far below linear. "
        "More threads do not help past four cores here."
    )


def _gpu08_ratio(data: dict) -> float:
    """The GPU-08 device-resident f32 headline ratio from the committed matrix.

    Reads ``gpu_cases[i].ratio_vs_numpy_cpu_f32`` for the largest warm
    without-transfer case (n=2048, the GPU-08 measurement size), from the
    COMMITTED results/refbox_gpu_matrix.json. Not a literal, and not
    results/local/gpu08.json (gitignored, absent on a fresh clone):
    reading the local file would orphan this number on a clone.
    """
    warm = [
        c
        for c in data["gpu_cases"]
        if c.get("label") == "without-transfer" and c.get("phase") == "warm"
    ]
    largest = max(warm, key=lambda c: c["n"])
    return largest["ratio_vs_numpy_cpu_f32"]


def render_gpu_table(data: dict) -> str:
    """The GPU device-resident + with-transfer table (GPU-08). The win + a loss.

    Without-transfer warm is the device-resident headline (GPU-08, ~23x NumPy
    CPU f32 at n=2048). The with-transfer warm rows are the dispatch-cost story:
    the small sizes lose to the host round-trip (n=256 ~0.35x), which is why the
    auto policy keeps small host arrays on the CPU. CuPy is absent on this
    machine, so the baseline is NumPy CPU f32 (rule 15).
    """
    warm = [
        c
        for c in data["gpu_cases"]
        if c.get("label") == "without-transfer" and c.get("phase") == "warm"
    ]
    warm.sort(key=lambda c: c["n"])
    lines = [
        "Device-resident float32 (without transfer), warm, vs NumPy CPU f32:",
        "",
        "| n | GFLOP/s | speedup vs NumPy CPU f32 |",
        "| --- | --- | --- |",
    ]
    for case in warm:
        lines.append(
            f"| {case['n']} | {_fmt(case['gflops'], 0)} "
            f"| {_fmt(case['ratio_vs_numpy_cpu_f32'])}x |"
        )

    with_transfer = [
        c
        for c in data.get("with_transfer_cases", [])
        if c.get("phase") == "warm"
    ]
    if with_transfer:
        with_transfer.sort(key=lambda c: c["n"])
        lines.append("")
        lines.append(
            "With the host round-trip (to_device + matmul + from_device), warm:"
        )
        lines.append("")
        lines.append("| n | speedup vs NumPy CPU f32 | outcome |")
        lines.append("| --- | --- | --- |")
        for case in with_transfer:
            ratio = case["ratio_vs_numpy_cpu_f32"]
            outcome = (
                "loses to transfer cost" if ratio < 1.0 else "wins after transfer"
            )
            lines.append(f"| {case['n']} | {_fmt(ratio)}x | {outcome} |")
    return "\n".join(lines)


def render_fused_table(cpu_data: dict, gpu_data: dict) -> str:
    """The fused 3-op chain table (FUSE-05). Both wins.

    The CPU floor speedup (``cpu_cases[0].speedup_floor``, ~3.2x) and the GPU
    device-resident ratio (``gpu_cases[0].ratio_vs_numpy_cpu_f32``, ~71x NumPy
    CPU f32) for the chain scaled_relu(fma3(axpby(x,y,a,b),z)) at 8M float32.
    The fused kernel makes one memory pass where the unfused NumPy chain makes
    several, which is the whole speedup.
    """
    chain = cpu_data["chain"]
    cpu_floor = cpu_data["cpu_cases"][0]["speedup_floor"]
    gpu_ratio = gpu_data["gpu_cases"][0]["ratio_vs_numpy_cpu_f32"]
    elems = cpu_data["cpu_cases"][0]["elems"]
    lines = [
        f"Chain `{chain}` at {elems // 1_000_000}M float32 elements:",
        "",
        "| backend | speedup vs NumPy unfused chain |",
        "| --- | --- |",
        f"| CPU (floor) | {_fmt(cpu_floor)}x |",
        f"| GPU device-resident | {_fmt(gpu_ratio)}x |",
    ]
    return "\n".join(lines)


def render_blocking_note(data: dict) -> str:
    """The autotuned blocking winner, cited from the saved sweep JSON (rule 17).

    PITFALL 5: this reads ``winner`` from the saved sweep, which selected
    mc=48 on this machine. The renderer cites what the JSON contains, never a
    hand-picked value.
    """
    winner = data["winner"]
    return (
        "Autotuned CPU blocking on this machine (from the saved sweep): "
        f"mc={winner['mc']}, kc={winner['kc']}, nc={winner['nc']} "
        f"(selected on {winner['selected_on']})."
    )


def _manifest(data: dict) -> dict:
    return data["manifest"]


def render_manifest_table(data: dict) -> str:
    """The hardware manifest table (rule 9 keys) from any results file's manifest.

    Reads the manifest the bench already captured; the GPU/driver fields come
    from a CUDA-build result (the GPU matrix file), the CPU/BLAS fields from any
    file. Every value is read, never typed.
    """
    man = _manifest(data)
    rows = [
        ("Platform", man.get("platform", "n/a")),
        ("CPU", man.get("processor", "n/a")),
        ("GPU", man.get("gpu", "n/a")),
        ("CUDA driver", man.get("driver", "n/a")),
        ("Python", man.get("python", "n/a")),
        ("NumPy", man.get("numpy", "n/a")),
        ("BLAS", man.get("blas", "n/a")),
    ]
    lines = ["| Component | Value |", "| --- | --- |"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _load_optional(*parts: str) -> dict | None:
    """load_series the file at parts, or return None when it is absent.

    Mirrors the crossover chart's os.path.exists skip: a missing input on a fresh
    clone or a partial-results run degrades the corresponding README section to a
    printed skip rather than crashing the whole render with FileNotFoundError. A
    file that EXISTS but is unverified or manifest-less still raises through
    load_series -- absence is the only condition softened here, not invalidity.
    """
    path = _results_path(*parts)
    if not os.path.exists(path):
        return None
    return load_series(path)


def render_report() -> str:
    """Render every per-regime section from the committed JSON, as one blob.

    This is the single function the README author and the round-trip test both
    call: the README pastes these sections, and the test asserts every headline
    number appears in this string. Each section reads its own committed file; a
    file that exists but is verified-false or manifest-less fails through
    load_series before any number is emitted. A file that is simply ABSENT (a
    fresh clone, a partial-results run) degrades that one section to a printed
    skip line so the rest of the report still renders.
    """
    cpu02 = _load_optional("cpu02_f64_refbox.json")
    matmul_f64 = _load_optional("refbox_matmul_f64.json")
    cpu06 = _load_optional("cpu06_f32_refbox.json")
    scaling = _load_optional("tuning", "scaling_refbox.json")
    gpu_matrix = _load_optional("refbox_gpu_matrix.json")
    fuse_cpu = _load_optional("fuse05_cpu_f32_refbox.json")
    fuse_gpu = _load_optional("fuse05_gpu_f32_refbox.json")
    sweep = _load_optional("tuning", "sweep_refbox.json")

    def _skip(name: str) -> str:
        return f"_(skipped: {name} not found in results/; regenerate the bench)_"

    sections = ["## float64 matmul (CPU)"]
    if cpu02 is not None and matmul_f64 is not None:
        sections.append(render_matmul_f64_table(cpu02, matmul_f64))
    else:
        sections.append(_skip("cpu02_f64_refbox.json / refbox_matmul_f64.json"))

    sections.append("## Small-matrix float32 (CPU)")
    sections.append(
        render_small_matrix_table(cpu06)
        if cpu06 is not None
        else _skip("cpu06_f32_refbox.json")
    )

    sections.append("## Thread scaling (CPU)")
    sections.append(
        render_scaling_line(scaling)
        if scaling is not None
        else _skip("tuning/scaling_refbox.json")
    )

    sections.append("## GPU matmul")
    sections.append(
        render_gpu_table(gpu_matrix)
        if gpu_matrix is not None
        else _skip("refbox_gpu_matrix.json")
    )

    sections.append("## Fused 3-op chain")
    if fuse_cpu is not None and fuse_gpu is not None:
        sections.append(render_fused_table(fuse_cpu, fuse_gpu))
    else:
        sections.append(
            _skip("fuse05_cpu_f32_refbox.json / fuse05_gpu_f32_refbox.json")
        )

    sections.append("## Autotuned blocking")
    sections.append(
        render_blocking_note(sweep)
        if sweep is not None
        else _skip("tuning/sweep_refbox.json")
    )

    sections.append("## Hardware")
    sections.append(
        render_manifest_table(gpu_matrix)
        if gpu_matrix is not None
        else _skip("refbox_gpu_matrix.json")
    )
    return "\n\n".join(sections)


if __name__ == "__main__":
    print(render_report())
