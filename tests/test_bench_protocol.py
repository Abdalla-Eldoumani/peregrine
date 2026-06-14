"""BNCH-01 unit coverage for the bench-protocol CV-gate (rule 6).

bench-protocol rule 6 is the last unenforced protocol rule: a series whose
coefficient of variation exceeds 5 percent is invalidated and rerun after a
cooldown, and a still-noisy series after the rerun bound is recorded with
verified=true but flagged high_cv=true. The floor (min) stays the honest
statistic regardless; the gate exists so a noisy MEDIAN is never silently
published, not to "fix" the AV-vs-OpenMP noise that is structural on this
machine (rules 6, 9, 11).

These tests exercise the gate threaded into the single shared timing core
benchmarks/bench_matmul.py _bench, as opt-in keyword arguments with an
INJECTABLE cooldown callable. The cooldown is always a counter here, never a
real time.sleep, so the unit suite never waits 30 real seconds (the 07
validation latency budget / Nyquist constraint): a test that slept the
protocol cooldown would be unrunnable.

Coverage:
- the CV-gate reruns on CV>5 percent up to the rerun bound, then records
  high_cv + verified together (the floor stays honest, the series is never
  dropped for being noisy);
- the cooldown is the injected callable, invoked with cooldown_s between
  reruns, so the whole path finishes well under a second;
- a low-CV series does not rerun and is not flagged;
- the DEFAULT (no cv_gate) path returns exactly today's keys so the reusers
  bench_fused / scaling / calibrate stay untouched;
- a load_series-style schema validator accepts a saved series whose case
  carries manifest + verified=true, AND accepts the high_cv=true + verified=true
  combination unchanged (the noise flag never strips a verified series from the
  README; the AV-noisy CPU sweep that 07-03 consumes WILL be high_cv every run).
"""

from __future__ import annotations

import os
import sys

import pytest

# No test currently imports a benchmarks module. Mirror the inverse sys.path
# insert bench_matmul.py does to reach tests/ (its lines 39-42), pointing at
# benchmarks/ instead, then import the shared timing core under test.
_BENCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"
)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

import bench_matmul  # noqa: E402
from bench_matmul import _bench  # noqa: E402

# The @gpu skip marker, copied verbatim from test_cuda.py lines 30-36, for the
# bench_gpu_series slot 07-02 fills. The CV-gate / schema / default-path tests
# below are NOT gpu-gated: they run on every build, including the WSL CPU-only
# clone, with no device.
from conftest import requires_cuda  # noqa: E402

_CUDA_OK, _CUDA_REASON = requires_cuda()
gpu = pytest.mark.skipif(not _CUDA_OK, reason=_CUDA_REASON)


def _force_timings(monkeypatch, durations_ns):
    """Drive _measure_once's rep loop with deterministic per-rep durations.

    _measure_once reads time.perf_counter_ns() twice per rep (t0, then again
    after fn()); the recorded duration is the difference. Feeding a clock whose
    successive readings are (0, d0, 0, d1, ...) makes rep i last exactly
    durations_ns[i] nanoseconds, so CV is forced high or low with no real
    wall-clock variance (the validation Nyquist note: do NOT rely on jitter).

    The warmup loop calls fn() but never reads the clock, so warmup does not
    consume the sequence. Reruns re-enter _measure_once and re-read from the
    same iterator, so a per-rerun durations list reproduces a fixed CV each
    pass.
    """
    seq = []
    for d in durations_ns:
        seq.append(0)
        seq.append(d)
    it = iter(seq)
    monkeypatch.setattr(bench_matmul.time, "perf_counter_ns", lambda: next(it))


class _Cooldown:
    """An injectable cooldown that records its calls instead of sleeping.

    The production default is time.sleep(>=30s); the unit suite passes this so
    the rerun loop never blocks. It appends each requested cooldown_s so a test
    can assert both the call count and the argument the gate passed.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


# --- The shared schema validator the later bench/release plans import. ---
# Defined here (07-01) so 07-02/07-03 reuse the one validator rather than
# re-deriving it. json.load-only by contract (Security V5: a results file is
# data, never code); these tests pass it dicts directly, but the rule it
# enforces is rule 9 (manifest present) + rule 11 (every series verified=true),
# and CRITICALLY it must accept a high_cv+verified case unchanged.


def load_series(d):
    """Validate a saved-series dict is publishable; return it unchanged.

    rule 9: no manifest, no merge. rule 11: a verified-false series never
    reaches a results file. A high_cv=true case is STILL verified=true (rule
    6/9/11): the noise flag documents the residual, it never strips the series,
    so this validator must pass it through. 07-03's load_series accepts the same
    shape (the CPU matmul sweep on this AV-noisy machine is high_cv every run).
    """
    assert "manifest" in d, "rule 9: no manifest, no merge"
    for key in ("cases", "cpu_cases", "gpu_cases"):
        for case in d.get(key, []):
            assert (
                case.get("verified", False) is True
            ), "rule 11: verified-false series never published"
    return d


def _stable_fn():
    # The timed function itself is irrelevant: the clock is monkeypatched, so
    # fn's real duration never enters the measurement. A no-op keeps the loop
    # fast and the intent clear (the forced timings, not fn, set CV).
    return None


def test_cv_gate_reruns_then_flags(monkeypatch):
    # Forced timings whose CV stays above the 5 percent threshold every pass:
    # widely spread values give stdev/median well over 0.05, so the gate reruns
    # to the bound and then records the residual. Each pass re-reads the same
    # spread, so CV is identically high on every measurement.
    durations = [100, 300, 100, 300, 100, 300]  # CV ~ 0.47, far above 0.05
    _force_timings(monkeypatch, durations * 8)  # enough for 1 + max_reruns passes
    cooldown = _Cooldown()

    result = _bench(
        _stable_fn,
        reps=6,
        warmup=0,
        cv_gate=True,
        max_reruns=2,
        cooldown_s=30,
        cooldown_fn=cooldown,
    )

    # Rerun bound hit, residual noise recorded, series still verified (never
    # dropped): the floor stays the honest statistic, high_cv documents it.
    assert result["reruns"] == 2
    assert len(cooldown.calls) == 2
    assert result["high_cv"] is True
    assert result["verified"] is True
    assert "min_s" in result  # the floor is still reported regardless of CV


def test_cv_gate_cooldown_injected(monkeypatch):
    # The cooldown callable is invoked with cooldown_s between reruns, and the
    # whole test finishes well under a second because it is a counter, not a
    # real time.sleep(30). This is the proof there is no 30s wait on the path.
    import time as _wallclock

    durations = [100, 400, 100, 400, 100, 400]  # persistently high CV
    _force_timings(monkeypatch, durations * 8)
    cooldown = _Cooldown()

    t0 = _wallclock.perf_counter()
    result = _bench(
        _stable_fn,
        reps=6,
        warmup=0,
        cv_gate=True,
        max_reruns=2,
        cooldown_s=30,
        cooldown_fn=cooldown,
    )
    elapsed = _wallclock.perf_counter() - t0

    assert cooldown.calls == [30, 30]  # called with cooldown_s, once per rerun
    assert elapsed < 1.0  # no real 30s sleep on the path
    assert result["reruns"] == 2


def test_cv_gate_low_cv_no_rerun(monkeypatch):
    # Stable forced timings (CV under 5 percent) must not rerun: reruns 0, the
    # cooldown never fires, and high_cv is False (the flag is only set when the
    # residual exceeds the threshold).
    durations = [200, 201, 200, 201, 200, 201]  # CV ~ 0.0025, well under 0.05
    _force_timings(monkeypatch, durations * 4)
    cooldown = _Cooldown()

    result = _bench(
        _stable_fn,
        reps=6,
        warmup=0,
        cv_gate=True,
        max_reruns=2,
        cooldown_s=30,
        cooldown_fn=cooldown,
    )

    assert result["reruns"] == 0
    assert cooldown.calls == []
    assert result["high_cv"] is False


def test_bench_default_path_unchanged(monkeypatch):
    # _bench WITHOUT cv_gate returns EXACTLY today's keys and adds neither
    # high_cv nor reruns. This is the default reuse contract: bench_fused,
    # scaling, and calibrate call _bench positionally with no new kwargs and
    # must see the identical dict shape they see today.
    durations = [100, 300, 100, 300, 100, 300]  # high CV, but the gate is OFF
    _force_timings(monkeypatch, durations)
    cooldown = _Cooldown()

    result = _bench(_stable_fn, reps=6, warmup=0)

    assert set(result.keys()) == {
        "median_s",
        "p25_s",
        "p75_s",
        "min_s",
        "cv",
        "reps",
    }
    assert "high_cv" not in result
    assert "reruns" not in result
    assert "verified" not in result
    # The OFF path must not consult a cooldown at all.
    assert cooldown.calls == []


def test_results_schema_verified():
    # The schema validator accepts a saved series carrying the manifest and a
    # verified=true case (rule 9 + rule 11).
    good = {
        "manifest": {"platform": "x", "blas": "y"},
        "cases": [{"n": 256, "verified": True}],
    }
    assert load_series(good) is good

    # rule 9: a series with no manifest is rejected.
    with pytest.raises(AssertionError):
        load_series({"cases": [{"n": 256, "verified": True}]})

    # rule 11: a verified-false case is rejected (never published).
    with pytest.raises(AssertionError):
        load_series(
            {"manifest": {}, "cases": [{"n": 256, "verified": False}]}
        )

    # THE high_cv+verified survival contract: a case flagged high_cv=true that
    # is ALSO verified=true PASSES the validator unchanged. The noise flag never
    # strips a verified series from the README; on this AV-noisy machine the CPU
    # matmul sweep 07-03 consumes WILL be high_cv every real run, so load_series
    # must accept exactly this shape.
    noisy_but_verified = {
        "manifest": {"platform": "x", "blas": "y"},
        "cases": [{"n": 256, "verified": True, "high_cv": True, "reruns": 2}],
    }
    assert load_series(noisy_but_verified) is noisy_but_verified
    # The case is not dropped or mutated by validation.
    assert noisy_but_verified["cases"][0]["high_cv"] is True
    assert noisy_but_verified["cases"][0]["verified"] is True


@gpu
def test_bench_gpu_series():
    # Reserved slot: 07-02 fills this with the bench_gpu with/without-transfer x
    # cold/warm matrix (4 labeled series, rule 7). Kept under @gpu so the -k name
    # resolves on every build and 07-02 fills the body without renaming.
    pytest.xfail("bench_gpu series filled in 07-02")
