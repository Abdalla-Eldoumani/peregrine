"""Thread-safety proof: thread count cannot change results, and concurrent
host threads cannot deadlock the native kernel.

Bitwise identity across thread counts is achievable and therefore required:
the kernel parallelizes over output rows, and each element's accumulation
over the inner dimension runs strictly sequentially, so the thread count
decides only which thread computes a row, never any element's rounding
order. The subprocess indirection exists because the OpenMP runtime reads
OMP_NUM_THREADS once at its initialization; the variable must be in the
environment before the process starts to take effect.

The executor test drives kernels that release the GIL into the OpenMP
runtime from four host threads at once. Each entering thread becomes its own
OpenMP root, so the worst plausible outcome is oversubscription, not
deadlock. The future.result timeout is not a real backstop: a worker wedged
in a native call never returns, so executor shutdown joins it forever and
the session hangs before pytest can report the TimeoutError; only the
harness-level timeout reaps that. The test proves the non-deadlocking run,
it does not convert a native deadlock into a loud failure.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import fastmathext as fme
from conftest import assert_matmul_close

# odd dimensions on purpose: 333 is a legacy-killer size and the row split
# across threads lands unevenly there. Inputs regenerate identically in each
# child from the seed; only the result crosses the process boundary
SCRIPT = """\
import sys

import numpy as np

import fastmathext as fme

rng = np.random.default_rng(42)
a = rng.standard_normal((333, 257))
b = rng.standard_normal((257, 199))
np.save(sys.argv[1], fme.matmul(a, b))
"""

# 16 mixed tasks for 4 workers: the odd 333 triple, legacy-killer squares,
# tiny and rectangular shapes, so concurrent entries hit different row splits
EXECUTOR_SIZES = [
    (333, 257, 199),
    (1, 1, 1),
    (3, 7, 5),
    (7, 7, 7),
    (16, 16, 16),
    (50, 50, 50),
    (64, 32, 16),
    (33, 65, 17),
    (100, 1, 100),
    (1, 100, 1),
    (128, 64, 32),
    (250, 250, 250),
    (5, 333, 3),
    (81, 27, 9),
    (256, 128, 64),
    (13, 77, 205),
]


def _run_in_subprocess(tmp_path, omp_threads):
    out_path = tmp_path / f"got_{omp_threads}.npy"
    # full environment copy with the override merged in: a stripped env
    # breaks DLL resolution for the OpenMP runtime on Windows
    env = dict(os.environ, OMP_NUM_THREADS=str(omp_threads))
    p = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(out_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # stderr in the message so a child import failure reads as itself
    assert p.returncode == 0, p.stderr
    return np.load(out_path)


def test_thread_count_bitwise_identical(tmp_path):
    got_1 = _run_in_subprocess(tmp_path, 1)
    got_12 = _run_in_subprocess(tmp_path, 12)
    np.testing.assert_array_equal(got_1, got_12)


def _matmul_task(m, k, n, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    return a, b, fme.matmul(a, b)


def test_executor_no_deadlock():
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_matmul_task, m, k, n, seed)
            for seed, (m, k, n) in enumerate(EXECUTOR_SIZES)
        ]
        for future in futures:
            a, b, got = future.result(timeout=120)
            assert_matmul_close(got, a @ b, a, b)
