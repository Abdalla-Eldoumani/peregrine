"""Fallback reachability proof: PEREGRINE_DISABLE_AVX2 forces the naive kernel on this
AVX2 machine, and the forced-naive result matches the blis result within the
suite tolerance.

The override is the only way to exercise the no-AVX2 path on hardware that has
AVX2, so this test is what keeps the permanent fallback honest. The subprocess
indirection exists because the features are memoized at the first detect() call
(module import); PEREGRINE_DISABLE_AVX2 must be in the environment before the process
starts to take effect, exactly like the OMP_NUM_THREADS pattern in test_threads.

The comparison is toleranced, never bitwise: the blis path chunks its k
accumulation by KC while naive chunks by KB=64, so the two accumulate each
output element in a different order and land a few ulps apart. That divergence
is the mechanical FMA/order class the tolerance contract was built for, not a
kernel bug, so assert_matmul_close is the correct gate.
"""

import os
import subprocess
import sys

import numpy as np

import peregrine as pg
from conftest import assert_matmul_close

# odd dimensions on purpose: 333 is a legacy-killer size and 257/199 are not
# multiples of the 6x8 register tile, so the child's reference run and the
# parent's blis run both drive the masked C-edge path. Inputs regenerate
# identically from the seed in both the child and the parent; only the child's
# result crosses the process boundary as a .npy. The child also asserts the
# override actually reached feature detection before trusting the result.
SEED = 42
M, K, N = 333, 257, 199

SCRIPT = """\
import sys

import numpy as np

import peregrine as pg

assert pg.cpu_features()["avx2"] is False, pg.cpu_features()

rng = np.random.default_rng(42)
a = rng.standard_normal((333, 257))
b = rng.standard_normal((257, 199))
np.save(sys.argv[1], pg.matmul(a, b))
"""


def _run_forced_naive(tmp_path):
    out_path = tmp_path / "got_naive.npy"
    # full environment copy with the override merged in: a stripped env breaks
    # DLL resolution for the OpenMP runtime on Windows
    env = dict(os.environ, PEREGRINE_DISABLE_AVX2="1")
    p = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(out_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # stderr in the message so a child import or assertion failure reads as
    # itself (the cpu_features assertion fires here if the override is missed)
    assert p.returncode == 0, p.stderr
    return np.load(out_path)


def test_disable_avx2_forces_naive_and_matches_blis(tmp_path):
    got_naive = _run_forced_naive(tmp_path)

    # same seed in-process: this machine has AVX2+FMA, so this runs the blis path
    rng = np.random.default_rng(SEED)
    a = rng.standard_normal((M, K))
    b = rng.standard_normal((K, N))
    got_blis = pg.matmul(a, b)

    # toleranced, not bitwise: KC vs KB chunking differ across the two paths
    assert_matmul_close(got_naive, got_blis, a, b)
