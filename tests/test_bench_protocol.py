"""Unit coverage for the benchmark measurement protocol.

Two layers are covered. The sequential _bench layer the older scripts use: a
series whose coefficient of variation exceeds 5 percent is rerun after a
cooldown, and a still-noisy series after the rerun bound is recorded with
verified=true but flagged high_cv=true. And the protocol core every published
number comes from (bench_matmul.run_cell driven by bench_suite): ABBA order and
its seeded start, the shared batch size, per-sample bitwise verification, the
CV strike sequence ending in a fresh process, clock discards and the noisy
closure, the unsupported and skipped paths, and the manifest.

Every timer, clock reading, and sleep here is injected, so nothing waits on a
real cooldown and no assertion depends on this machine's timing jitter.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys

import numpy as np
import pytest

# Mirror the sys.path insert bench_matmul.py does to reach tests/, pointing at
# benchmarks/ instead, then import the modules under test.
_BENCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"
)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

import bench_matmul  # noqa: E402
import bench_suite  # noqa: E402
from bench_matmul import (  # noqa: E402
    Contender,
    Protocol,
    _bench,
    abba_blocks,
    abba_order,
    batch_size,
    bitwise_equal,
    redact_paths,
    run_cell,
)

import peregrine as pg  # noqa: E402

# The GPU series test below runs only where a usable device exists; every other
# test runs on every build, including a CPU-only Linux clone.
from conftest import requires_cuda  # noqa: E402

_CUDA_OK, _CUDA_REASON = requires_cuda()
gpu = pytest.mark.skipif(not _CUDA_OK, reason=_CUDA_REASON)


def _force_timings(monkeypatch, durations_ns):
    """Drive _measure_once's rep loop with deterministic per-rep durations.

    _measure_once reads time.perf_counter_ns() twice per rep (t0, then again
    after fn()); the recorded duration is the difference. Feeding a clock whose
    successive readings are (0, d0, 0, d1, ...) makes rep i last exactly
    durations_ns[i] nanoseconds, so CV is forced high or low with no real
    wall-clock variance.

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
    """An injectable sleep that records its calls instead of sleeping."""

    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def load_series(d):
    """Validate a saved-series dict in the older shape; return it unchanged.

    No manifest, no merge; a verified-false series never reaches a results
    file. A high_cv=true case is still verified=true: the noise flag
    documents the residual and never strips the series, so this validator
    must pass it through.
    """
    assert "manifest" in d, "no manifest, no merge"
    for key in ("cases", "cpu_cases", "gpu_cases"):
        for case in d.get(key, []):
            assert (
                case.get("verified", False) is True
            ), "a verified-false series is never published"
    return d


def _stable_fn():
    # The clock is monkeypatched, so fn's real duration never enters the
    # measurement; the forced timings set CV.
    return None


def test_cv_gate_reruns_then_flags(monkeypatch):
    # Widely spread forced timings keep CV far above 5 percent on every pass,
    # so the gate reruns to the bound and then records the residual.
    durations = [100, 300, 100, 300, 100, 300]  # CV ~ 0.47
    _force_timings(monkeypatch, durations * 8)
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

    assert result["reruns"] == 2
    assert len(cooldown.calls) == 2
    assert result["high_cv"] is True
    assert result["verified"] is True
    assert "min_s" in result


def test_cv_gate_cooldown_injected(monkeypatch):
    # The cooldown callable is invoked with cooldown_s between reruns, and the
    # whole test finishes well under a second, which proves no real 30 s sleep
    # sits on the path.
    import time as _wallclock

    durations = [100, 400, 100, 400, 100, 400]
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

    assert cooldown.calls == [30, 30]
    assert elapsed < 1.0
    assert result["reruns"] == 2


def test_cv_gate_low_cv_no_rerun(monkeypatch):
    durations = [200, 201, 200, 201, 200, 201]  # CV ~ 0.0025
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
    # Without cv_gate, _bench returns exactly the keys bench_fused, scaling,
    # sweep_blocking, and calibrate read, and never consults a cooldown.
    durations = [100, 300, 100, 300, 100, 300]
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
    assert cooldown.calls == []


def test_results_schema_verified():
    good = {
        "manifest": {"platform": "x", "blas": "y"},
        "cases": [{"n": 256, "verified": True}],
    }
    assert load_series(good) is good

    with pytest.raises(AssertionError):
        load_series({"cases": [{"n": 256, "verified": True}]})

    with pytest.raises(AssertionError):
        load_series(
            {"manifest": {}, "cases": [{"n": 256, "verified": False}]}
        )

    # A case flagged high_cv that is also verified passes unchanged.
    noisy_but_verified = {
        "manifest": {"platform": "x", "blas": "y"},
        "cases": [{"n": 256, "verified": True, "high_cv": True, "reruns": 2}],
    }
    assert load_series(noisy_but_verified) is noisy_but_verified
    assert noisy_but_verified["cases"][0]["high_cv"] is True
    assert noisy_but_verified["cases"][0]["verified"] is True


@gpu
def test_bench_gpu_series():
    # The GPU matrix is four labeled series, with/without transfer by cold/warm.
    # A tiny size grid keeps the device timing short; the assertions are on
    # the shape, not the magnitudes.
    import bench_gpu

    sizes = [64]
    without = bench_gpu._without_transfer_series(sizes, reps=2, warmup=1)
    with_transfer = bench_gpu._with_transfer_series(sizes, reps=2, warmup=1)

    combos = {(r["label"], r["phase"]) for r in without + with_transfer}
    assert combos == {
        ("without-transfer", "warm"),
        ("without-transfer", "cold"),
        ("with-transfer", "warm"),
        ("with-transfer", "cold"),
    }

    for record in without + with_transfer:
        assert record["verified"] is True

    # report.py reads these keys from each gpu_cases record; a missing one
    # would break the renderer.
    for record in without:
        for key in ("n", "ratio_vs_numpy_cpu_f32", "cold_ms", "median_ms"):
            assert key in record, f"without-transfer record missing {key}"

    assert load_series({"manifest": {"x": 1}, "gpu_cases": without}) is not None


# --- The protocol core. ---


class _FakeTime:
    """A nanosecond clock that only moves when a fake contender runs."""

    def __init__(self):
        self.ns = 0

    def now(self):
        return self.ns

    def advance(self, seconds):
        self.ns += int(round(seconds * 1e9))


def _timed(ft, durations, value=None):
    """A contender body whose i-th call lasts durations(i) seconds."""
    state = {"i": 0}
    out = np.array([1.0]) if value is None else value

    def fn():
        ft.advance(durations(state["i"]))
        state["i"] += 1
        return out

    fn.state = state
    return fn


def _ok_check(refs):
    return None


def _proto(**kw):
    base = dict(samples=10, warmups=5, switch_gap_s=0.0, rewarm_s=0.0, strike_cooldown_s=30.0, fresh_cooldown_s=120.0)
    base.update(kw)
    return Protocol(**base)


def test_abba_order_seeded_start():
    names = ["peregrine", "numpy"]
    # Reproducible per cell id, across calls and processes (str seeding).
    assert abba_order(names, "matmul/float64/64x64x64/c/-/t6/s0") == abba_order(
        names, "matmul/float64/64x64x64/c/-/t6/s0"
    )
    starts = {abba_order(names, f"cell-{i}")[0] for i in range(40)}
    assert starts == {"peregrine", "numpy"}
    assert sorted(abba_order(["a", "b", "c"], "x")) == ["a", "b", "c"]


def test_abba_blocks_mirror():
    assert abba_blocks(["A", "B"], 4) == ["A", "B", "B", "A", "A", "B", "B", "A"]
    assert abba_blocks(["A", "B", "C"], 2) == ["A", "B", "C", "C", "B", "A"]


def test_rewarm_keeps_cold_clock_calls_out_of_the_samples():
    # A contender that runs 4x slow for its first 150 ms of activity after each
    # switch, the way a core idled through the gap ramps its clock back.
    ft = _FakeTime()
    state = {"since_switch": 0.0}

    def body(slow_s, fast_s):
        def fn():
            d = slow_s if state["since_switch"] < 0.15 else fast_s
            ft.advance(d)
            state["since_switch"] += d
            return np.array([1.0])
        return fn

    def gap(_s):
        state["since_switch"] = 0.0

    a = Contender("peregrine", body(400e-6, 100e-6))
    b = Contender("numpy", body(800e-6, 200e-6))
    res = run_cell("cell", [a, b], _ok_check, _proto(rewarm_s=0.2, switch_gap_s=0.3), sleep=gap, now=ft.now)
    assert res["status"] == "ok"
    for name, fast in (("peregrine", 100e-6), ("numpy", 200e-6)):
        assert max(res["contenders"][name]["samples_s"]) == pytest.approx(fast)


def test_batch_size_from_fastest():
    target = 200e-6
    assert batch_size({"peregrine": 1e-6, "numpy": 5e-6}, target) == 200
    assert batch_size({"peregrine": 150e-6, "numpy": 1e-3}, target) == 2
    assert batch_size({"peregrine": 300e-6, "numpy": 1e-3}, target) == 1


def test_shared_k_identical_loop_and_per_call_time():
    ft = _FakeTime()
    fast = Contender("peregrine", _timed(ft, lambda i: 1e-6))
    slow = Contender("numpy", _timed(ft, lambda i: 4e-6))
    sleep = _Cooldown()
    res = run_cell("cell", [fast, slow], _ok_check, _proto(switch_gap_s=0.3), sleep=sleep, now=ft.now)

    assert res["status"] == "ok"
    # K comes from the fastest contender and is shared by both.
    assert res["batch_k"] == 200
    for name, per_call in (("peregrine", 1e-6), ("numpy", 4e-6)):
        s = res["contenders"][name]
        assert s["n_samples"] >= 10
        assert s["median_s"] == pytest.approx(per_call)
        assert all(x == pytest.approx(per_call) for x in s["samples_s"])
    assert res["ratio"]["numpy"] == pytest.approx(4.0)
    # One gap per contender switch, counting the switch from the last warmed
    # contender into the first measured block.
    gaps = [s for s in sleep.calls if s == 0.3]
    blocks = abba_blocks(res["order"], res["rounds"])
    switches = 1 + sum(1 for a, b in zip(blocks, blocks[1:]) if a != b)
    assert len(gaps) == switches


def test_device_contender_rewarmed_after_each_switch():
    ft = _FakeTime()
    calls = []

    def timer(k):
        calls.append(k)
        return 1e-3

    dev = Contender("peregrine", lambda: np.array([1.0]), timer=timer)
    cpu = Contender("numpy", _timed(ft, lambda i: 1e-3))
    res = run_cell("cell", [dev, cpu], _ok_check, _proto(), now=ft.now)
    blocks = abba_blocks(res["order"], res["rounds"])
    prev = [res["order"][-1]] + blocks[:-1]
    entries = sum(1 for b, p in zip(blocks, prev) if b == "peregrine" and p != b)
    # pre-pilot warm (one call) + pilot (one call: 1 ms is over the target) +
    # warmups + samples + one re-warm per entry into the device contender's block.
    assert len(calls) == 1 + 1 + 5 + res["contenders"]["peregrine"]["n_samples"] + entries


def test_verify_last_mode_checks_last_call():
    ft = _FakeTime()
    c = Contender("peregrine", _timed(ft, lambda i: 1e-6))
    res = run_cell("cell", [c], _ok_check, _proto(verify_mode="last"), now=ft.now)
    assert res["status"] == "ok"
    assert res["verify_mode"] == "last"


def test_unbatched_cells_report_every_call_verified():
    ft = _FakeTime()
    batched = run_cell("c", [Contender("peregrine", _timed(ft, lambda i: 1e-6))], _ok_check, _proto(), now=ft.now)
    single = run_cell("c", [Contender("peregrine", _timed(ft, lambda i: 300e-6))], _ok_check, _proto(), now=ft.now)
    assert (batched["batch_k"], batched["verify_mode"]) == (200, "last")
    assert (single["batch_k"], single["verify_mode"]) == (1, "all")


@pytest.mark.parametrize("mode,status", [("all", "verify_failed"), ("last", "ok")])
def test_mid_batch_error_caught_only_when_all_calls_are_kept(mode, status):
    ft = _FakeTime()
    good, bad = np.array([1.0]), np.array([9.0])
    state = {"i": 0}

    def fn():
        # 1 us calls: the pilot ends at k=256 (calls 1..511), K=200, warmups are
        # calls 512..1511, so call 1600 sits mid-way through the first sample.
        ft.advance(1e-6)
        i = state["i"]
        state["i"] += 1
        return bad if i == 1600 else good

    res = run_cell("c", [Contender("peregrine", fn)], _ok_check, _proto(verify_mode=mode), now=ft.now)
    assert res["status"] == status


def test_cv_strikes_then_fresh_process(monkeypatch):
    ft = _FakeTime()
    # K=1 (every call is over the 200 us target) and alternating 1 ms / 3 ms
    # calls, so every attempt's CV is far above 5 percent.
    noisy = Contender("peregrine", _timed(ft, lambda i: 1e-3 if i % 2 else 3e-3))
    steady = Contender("numpy", _timed(ft, lambda i: 1e-3))
    sleep = _Cooldown()
    calls = []

    def fresh():
        calls.append(1)
        return {
            "status": "ok", "verified": True, "strikes": 0,
            "attempts": [{"cv": {"peregrine": 0.01, "numpy": 0.01}}],
            "contenders": {"peregrine": {"median_s": 1e-3}, "numpy": {"median_s": 2e-3}},
            "ratio": {"numpy": 2.0},
        }

    res = run_cell("cell", [noisy, steady], _ok_check, _proto(), sleep=sleep, now=ft.now, fresh=fresh)

    assert calls == [1]
    assert res["status"] == "ok"
    assert res["fresh_process"] is True
    assert res["strikes"] == 3
    assert len(res["attempts"]) == 4
    # Two cooldowns between the three in-process attempts, then the long one
    # before the fresh process.
    assert [s for s in sleep.calls if s > 0] == [30.0, 30.0, 120.0]


def test_cv_strikes_fresh_still_noisy_keeps_samples():
    ft = _FakeTime()
    noisy = Contender("peregrine", _timed(ft, lambda i: 1e-3 if i % 2 else 3e-3))
    steady = Contender("numpy", _timed(ft, lambda i: 1e-3))

    def fresh_noisy():
        # The child closes noisy on its single attempt, raw samples kept.
        return {
            "status": "noisy", "verified": True, "strikes": 1,
            "attempts": [{"cv": {"peregrine": 0.4, "numpy": 0.0}}],
            "contenders": {"peregrine": {"samples_s": [1e-3, 3e-3], "median_s": 2e-3, "cv": 0.4}},
        }

    res = run_cell("cell", [noisy, steady], _ok_check, _proto(), sleep=_Cooldown(), now=ft.now, fresh=fresh_noisy)
    assert res["status"] == "noisy"
    assert res["strikes"] == 4
    assert "ratio" not in res
    assert res["contenders"]["peregrine"]["samples_s"]

    # Without a fresh runner (the child itself) three strikes close noisy, no ratio.
    res = run_cell("cell", [noisy, steady], _ok_check, _proto(), sleep=_Cooldown(), now=ft.now)
    assert res["status"] == "noisy"
    assert res["strikes"] == 3
    assert "ratio" not in res
    assert len(res["contenders"]["peregrine"]["samples_s"]) >= 10


def test_fresh_process_failure_closes_noisy():
    ft = _FakeTime()
    noisy = Contender("peregrine", _timed(ft, lambda i: 1e-3 if i % 2 else 3e-3))

    def broken():
        raise RuntimeError("exit 3: child died")

    res = run_cell("cell", [noisy], _ok_check, _proto(), sleep=_Cooldown(), now=ft.now, fresh=broken)
    assert res["status"] == "noisy"
    assert "fresh process failed" in res["reason"]


class _FakeClock:
    """Clock readings by group index; `low` lists the groups read off-band."""

    def __init__(self, low, value=150.0, off=50.0):
        self.low = set(low)
        self.value = value
        self.off = off
        self.n = 0

    def start(self):
        pass

    def stop(self):
        i = self.n
        self.n += 1
        return self.off if i in self.low else self.value


def test_clock_discards_counted_and_retaken():
    ft = _FakeTime()
    # 60 ms calls: each sample is read on its own (over the 50 ms grouping).
    a = Contender("peregrine", _timed(ft, lambda i: 60e-3))
    b = Contender("numpy", _timed(ft, lambda i: 60e-3))
    clock = _FakeClock(low={1, 4, 9})
    res = run_cell("cell", [a, b], _ok_check, _proto(clock_floor=100.0), sleep=_Cooldown(), now=ft.now, clock=clock)

    assert res["status"] == "ok"
    assert sum(s["discards"] for s in res["contenders"].values()) == 3
    for s in res["contenders"].values():
        # Discarded samples are re-taken, so the planned count still lands.
        assert s["n_samples"] == 10
        assert all(r == 150.0 for r in s["clock_pct"])


def test_clock_discards_over_limit_close_noisy():
    ft = _FakeTime()
    a = Contender("peregrine", _timed(ft, lambda i: 60e-3))
    b = Contender("numpy", _timed(ft, lambda i: 60e-3))
    # 20 planned samples; a fifth discard passes the 20 percent limit.
    clock = _FakeClock(low=set(range(0, 40, 2)))
    res = run_cell("cell", [a, b], _ok_check, _proto(clock_floor=100.0), sleep=_Cooldown(), now=ft.now, clock=clock)

    assert res["status"] == "noisy"
    assert "clock" in res["reason"]
    assert "ratio" not in res
    assert sum(s["discards"] for s in res["contenders"].values()) == 5
    assert res["strikes"] == 0


def test_clock_ceiling_and_no_bounds():
    ft = _FakeTime()
    a = Contender("peregrine", _timed(ft, lambda i: 60e-3))
    # Off-band readings with no floor or ceiling are recorded, never discarded.
    res = run_cell("cell", [a], _ok_check, _proto(), sleep=_Cooldown(), now=ft.now, clock=_FakeClock(low={0, 1, 2}))
    assert res["contenders"]["peregrine"]["discards"] == 0
    assert 50.0 in res["contenders"]["peregrine"]["clock_pct"]
    # A ceiling discards readings above it.
    res = run_cell(
        "cell", [a], _ok_check, _proto(clock_ceiling=100.0), sleep=_Cooldown(), now=ft.now,
        clock=_FakeClock(low={0}, value=50.0, off=190.0),
    )
    assert res["contenders"]["peregrine"]["discards"] == 1


def test_bitwise_mismatch_is_verify_failed():
    ft = _FakeTime()
    good, bad = np.array([1.0, 2.0]), np.array([1.0, 2.0 + 2**-40])
    state = {"i": 0}

    def drifting():
        ft.advance(1e-3)
        state["i"] += 1
        return good if state["i"] < 12 else bad

    res = run_cell(
        "cell", [Contender("peregrine", drifting), Contender("numpy", _timed(ft, lambda i: 1e-3))],
        _ok_check, _proto(), now=ft.now,
    )
    assert res["status"] == "verify_failed"
    assert res["verified"] is False
    assert "bitwise" in res["reason"]
    assert "ratio" not in res and "contenders" not in res


def test_out_buffer_checked_after_each_sample():
    ft = _FakeTime()
    buf = np.zeros(4)
    state = {"i": 0}

    def into_buffer():
        ft.advance(1e-6)
        state["i"] += 1
        # The same object every call; its contents go wrong late in the run.
        buf[:] = 1.0 if state["i"] < 3000 else 2.0
        return buf

    res = run_cell("cell", [Contender("peregrine", into_buffer)], _ok_check, _proto(samples=60), now=ft.now)
    assert res["status"] == "verify_failed"


def test_nan_outputs_compare_equal():
    ft = _FakeTime()
    nan = np.array([np.nan, 1.0])
    res = run_cell("cell", [Contender("peregrine", _timed(ft, lambda i: 1e-3, value=nan))], _ok_check, _proto(), now=ft.now)
    assert res["status"] == "ok"
    assert bitwise_equal(np.float32(np.nan), np.float32(np.nan))
    assert not bitwise_equal(np.zeros(2, np.float32), np.zeros(2, np.float64))
    assert not bitwise_equal(np.zeros((2, 1)), np.zeros((1, 2)))


def test_oracle_failure_is_verify_failed():
    ft = _FakeTime()

    def wrong(refs):
        raise AssertionError("f64 contract violated")

    res = run_cell("cell", [Contender("peregrine", _timed(ft, lambda i: 1e-3))], wrong, _proto(), now=ft.now)
    assert res["status"] == "verify_failed"
    assert res["reason"].startswith("oracle:")


@contextlib.contextmanager
def _no_pin(n):
    yield {"blas": [n], "openmp": [n], "peregrine": n, "source": "test"}


def test_driver_wrong_peregrine_result_is_verify_failed(monkeypatch):
    # The real driver path end to end on a tiny cell, with peregrine's matmul
    # monkeypatched to return a result 0.1 percent off.
    monkeypatch.setattr(bench_suite, "pinned_threads", _no_pin)
    real = pg.matmul
    monkeypatch.setattr(pg, "matmul", lambda a, b: real(a, b) * 1.001)
    cell = bench_suite.Cell("quick", "matmul", "float64", (16, 16, 16))
    res = bench_suite.run_one(cell, _proto(switch_gap_s=0.0), sleep=_Cooldown())
    assert res["status"] == "verify_failed"
    assert res["oracle"] == "assert_matmul_close"
    assert res["verified"] is False


def test_driver_unsupported_and_skipped(monkeypatch):
    monkeypatch.setattr(bench_suite, "pinned_threads", _no_pin)

    def axpby_without_out(x, y, *, a=1.0, b=1.0):
        return a * x + b * y

    def axpby_with_out(x, y, *, a=1.0, b=1.0, out=None):
        return a * x + b * y

    cell = bench_suite.Cell("fused", "axpby", "float64", (8, 8), variant="out")
    monkeypatch.setattr(pg, "axpby", axpby_without_out)
    res = bench_suite.run_one(cell, _proto())
    assert res["status"] == "unsupported"
    assert "out=" in res["reason"]
    assert "ratio" not in res and res["verified"] is False
    monkeypatch.setattr(pg, "axpby", axpby_with_out)
    assert bench_suite.unsupported(cell) is None

    monkeypatch.setattr(pg, "has_cuda", lambda: False)
    res = bench_suite.run_one(bench_suite.Cell("gpu", "gpu_matmul", "float32", (256, 256, 256), variant="device"), _proto())
    assert res["status"] == "skipped"


def test_assert_threads_mismatch(monkeypatch):
    monkeypatch.delattr(pg, "set_num_threads", raising=False)
    monkeypatch.setattr(bench_matmul, "_START_OMP_NUM_THREADS", "999")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS=6"):
        bench_matmul.assert_threads(6)


def test_redact_paths_nested():
    home = os.path.expanduser("~")
    fwd = home.replace("\\", "/")
    doc = {"cli": ["--out", os.path.join(home, "r.json"), fwd + "/x"], "deep": [{"p": home.upper()}], "n": 3}
    red = redact_paths(doc)
    text = json.dumps(red)
    for variant in (home, fwd, home.upper(), json.dumps(home)[1:-1]):
        assert variant not in text
    assert red["n"] == 3
    assert red["cli"][0] == "--out"


def test_manifest_keys_and_redaction():
    home = os.path.expanduser("~")
    m = bench_matmul.protocol_manifest(
        argv=["bench_suite.py", "--out", os.path.join(home, "out.json")], series_seed=7, clock=None
    )
    for key in (
        "machine", "os_build", "cpu_brand", "power_plan", "compiler", "compiler_flags",
        "openmp_runtime", "python", "numpy", "openblas_version", "blas_core_type",
        "blas_threads", "peregrine_threads", "cuda_build", "gpu", "driver",
        "peregrine_version", "source_tree", "harness_tree", "datetime",
        "series_order_seed", "cli", "threadpools",
    ):
        assert key in m, key
    assert m["series_order_seed"] == 7
    assert set(m["source_tree"]) == {"commit", "dirty"}
    if hasattr(pg._core, "_build_info"):
        for key in ("compiler", "compiler_flags", "openmp_runtime"):
            assert m[key] != "unknown", key
    text = json.dumps(m)
    assert home not in text and home.replace("\\", "/") not in text
    assert json.dumps(home)[1:-1] not in text


def test_child_protocol_round_trips_through_argv():
    proto = Protocol(samples=40, warmups=6, switch_gap_s=0.25, batch_target_s=300e-6,
                     clock_floor=120.0, verify_mode="last")
    args = bench_suite._parse(bench_suite._proto_argv(proto, smoke=False) + ["--cell", "{}", "--fresh-child"])
    child, _ = bench_suite._protocol(args)
    for f in ("samples", "warmups", "switch_gap_s", "clock_floor", "clock_ceiling", "verify_mode"):
        assert getattr(child, f) == getattr(proto, f), f
    assert child.batch_target_s == pytest.approx(proto.batch_target_s)
    # The child gets one attempt and no fresh process of its own.
    assert child.max_strikes == 1


def test_cli_floors_and_smoke():
    with pytest.raises(SystemExit):
        bench_suite._parse(["--samples", "10"])
    proto, cooldown = bench_suite._protocol(bench_suite._parse(["--smoke"]))
    assert proto.samples < 30 and cooldown == 0.0 and proto.fresh_cooldown_s == 0.0
    proto, cooldown = bench_suite._protocol(bench_suite._parse([]))
    assert (proto.samples, proto.warmups, proto.switch_gap_s, cooldown, proto.fresh_cooldown_s) == (30, 5, 0.3, 30.0, 120.0)


def test_groups_select():
    for g in bench_suite.GROUPS:
        cells = bench_suite.select([g], [0, 1, 2], cuda=True)
        assert cells, g
        assert len({c.id for c in cells}) == len(cells), g
        assert {c.threads for c in cells} == ({1} if g == "gemm-1t" else {6}), g
    quick_cpu = bench_suite.select(["quick"], [0], cuda=False)
    quick_gpu = bench_suite.select(["quick"], [0], cuda=True)
    assert len(quick_gpu) - len(quick_cpu) == 2
    assert not any(bench_suite.needs_cuda(c) for c in quick_cpu)
    ids = {c.id for c in quick_cpu}
    assert "matmul/float64/1x4096x4096/c/-/t6/s0" in ids
    assert "matmul/float32/512x512x512/at_b/-/t6/s0" in ids
    cell = quick_cpu[0]
    assert bench_suite.Cell.from_spec(json.loads(json.dumps(cell.spec()))) == cell


def test_series_order_reproducible():
    cells = bench_suite.select(["reduce"], [0], cuda=False)
    a = [s[0].series for s in bench_suite.series_order(cells, 11)]
    assert a == [s[0].series for s in bench_suite.series_order(cells, 11)]
    assert a != [s[0].series for s in bench_suite.series_order(cells, 12)]


def test_spin_settle_from_quantized_trace():
    # Five threads spinning until 190 ms, charged in 15.6 ms scheduler ticks.
    tick = 0.015625
    trace = []
    for i in range(301):
        t = i * 0.005
        trace.append((t, (int(5 * min(t, 0.19) / tick)) * tick))
    settle, _ = bench_suite._settle(trace, window_s=0.2, floor=0.1)
    assert 0.15 <= settle <= 0.2
    idle = [(i * 0.005, 0.0) for i in range(301)]
    assert bench_suite._settle(idle, window_s=0.2, floor=0.1)[0] == 0.0
