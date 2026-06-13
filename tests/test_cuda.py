"""CUDA suite skeleton. The whole file skips cleanly with a stated reason on a
CPU-only build or a machine without a usable device, so it stays green on the
WSL/GCC clone and on the default CPU-only Windows build, and only exercises the
device paths on the FME_ENABLE_CUDA=ON build (tests/CLAUDE.md: GPU tests skip
cleanly, never fail for absent hardware).

requires_cuda() (conftest) is the one gate: build flag plus a usable-device
probe. It migrates to fme.has_cuda() in 04-05 when that public runtime predicate
lands; the gate's meaning (built AND device present) does not change.

The -k area names (context, gemm, tolerance, roundtrip, residency, transfer,
routing, oom, fallback, has_cuda) match the phase research Test Map so the later
plans fill in each placeholder under its own name. They assert nothing about
device behavior yet beyond what this plan delivers: this plan proves the file
collects-and-skips cleanly and that the build-flag smoke passes everywhere.

Correctness tests added by later plans route every CPU-vs-GPU comparison through
assert_matmul_close (conftest), never an inline tolerance, exactly like the CPU
suite; f32 GPU compares against the f64 ground truth the same way.
"""

import pytest

import fastmathext as fme
from conftest import requires_cuda

_CUDA_OK, _CUDA_REASON = requires_cuda()
# Reused by every device-touching test below. The build-flag smoke test is
# deliberately NOT decorated: it must pass on every build to prove the
# introspection surface exists and is a bool even when CUDA is off.
gpu = pytest.mark.skipif(not _CUDA_OK, reason=_CUDA_REASON)


def test_build_flag_smoke():
    # Runs everywhere, including CPU-only. has_cuda_build is the compile-time
    # flag; the runtime has_cuda() predicate is introduced in 04-05.
    assert isinstance(fme.has_cuda_build(), bool)


@gpu
def test_has_cuda_build_true_on_cuda_build():
    # Under the gate, reaching here means the build flag and a device are both
    # present, so the compile-time flag must be True.
    assert fme.has_cuda_build() is True


@gpu
def test_context_singleton_placeholder():
    # GPU-01: context singleton init and clean atexit shutdown. Filled in 04-02.
    assert fme.has_cuda_build() is True


@gpu
def test_gemm_device_placeholder():
    # GPU-02: f32/f64 device GEMM matches CPU via assert_matmul_close. 04-03.
    assert fme.has_cuda_build() is True


@gpu
def test_tolerance_placeholder():
    # GPU-02: TF32 off by default; result inside the f32 tolerance contract. 04-03.
    assert fme.has_cuda_build() is True


@gpu
def test_roundtrip_placeholder():
    # GPU-03: to_device -> from_device preserves bytes. 04-04.
    assert fme.has_cuda_build() is True


@gpu
def test_residency_placeholder():
    # GPU-03: matmul(Array, Array) -> Array; mixed residency raises TypeError. 04-04.
    assert fme.has_cuda_build() is True


@gpu
def test_transfer_placeholder():
    # GPU-04: async H2D/D2H with event sync before the host return. 04-04.
    assert fme.has_cuda_build() is True


@gpu
def test_routing_placeholder():
    # GPU-05: f32 device-resident routes to GPU; f64 never auto-routes. 04-05.
    assert fme.has_cuda_build() is True


@gpu
def test_oom_fallback_placeholder():
    # GPU-06: forced OOM (alloc > current free) -> warn once -> CPU fallback. 04-05.
    assert fme.has_cuda_build() is True
