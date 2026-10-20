"""Import an installed wheel on a machine without Visual Studio and run each op.

Run inside a clean Windows container with the wheel and NumPy installed into a
target directory: python -I tools/smoke_import.py <site-dir>. It first proves
the container has no system copy of the runtimes the wheel bundles, so a pass
means the wheel's own copies were loaded.
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    present = [d for d in ("vcomp140.dll", "msvcp140.dll") if os.path.exists(os.path.join(system32, d))]
    if present:
        print(f"not a clean machine: {', '.join(present)} in {system32}")
        return 1
    sys.path.insert(0, argv[0])
    import numpy as np

    import peregrine as pg

    rng = np.random.default_rng(0)
    for dt in (np.float32, np.float64):
        a = rng.standard_normal((67, 45)).astype(dt)
        b = rng.standard_normal((45, 33)).astype(dt)
        c = pg.matmul(a, b)
        assert np.allclose(c, a @ b, rtol=1e-4, atol=1e-4), "matmul"
        assert np.array_equal(pg.transpose(a), a.T), "transpose"
        assert np.allclose(pg.sum(a, axis=0), a.sum(axis=0), rtol=1e-5, atol=1e-5), "sum"
        assert np.allclose(pg.fma3(a, a, a), a * a + a, rtol=1e-5, atol=1e-5), "fma3"
    print(f"peregrine {pg.__version__} imported and ran from {os.path.dirname(pg.__file__)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
