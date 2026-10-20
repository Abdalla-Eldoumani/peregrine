"""Fail if a wheel or sdist holds anything outside the files it may ship.

An sdist may hold tracked source only (every member is a path `git ls-files`
lists, or the PKG-INFO the build writes); a wheel may hold the package, its
compiled module, the runtime libraries the repair step bundles, and its
dist-info. Both rules are allowlists, so a stray file of any name fails the
check whatever it is called.

Usage:
    python tools/check_dist_contents.py sdist dist/peregrine-*.tar.gz
    python tools/check_dist_contents.py wheel wheelhouse/*.whl
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tarfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per file, not per directory: the dist-info and the bundled-library folder are
# named after the project, and each library carries the repair tool's hash.
WHEEL_ALLOWED = [
    r"peregrine/[A-Za-z_][A-Za-z0-9_]*\.py",
    r"peregrine/_core\.(pyd|abi3\.so)",
    r"peregrine\.libs/(msvcp140|vcomp140)-[0-9a-f]{32}\.dll",
    r"peregrine\.libs/libgomp-[0-9a-f]{8}\.so(\.[0-9]+)*",
    r"peregrine-[0-9][0-9.]*(\.dev[0-9]+)?\.dist-info/(METADATA|WHEEL|RECORD|DELVEWHEEL)",
    r"peregrine-[0-9][0-9.]*(\.dev[0-9]+)?\.dist-info/licenses/[A-Za-z_.-]+",
]


def tracked_files() -> set[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return {p for p in out.decode("utf-8").split("\0") if p}


def check_sdist(path: str) -> list[str]:
    tracked = tracked_files()
    bad = []
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if m.isdir():
                continue
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
            if rel != "PKG-INFO" and rel not in tracked:
                bad.append(m.name)
    return bad


def check_wheel(path: str) -> list[str]:
    allowed = [re.compile(p + r"\Z") for p in WHEEL_ALLOWED]
    with zipfile.ZipFile(path) as whl:
        return [n for n in whl.namelist() if not n.endswith("/") and not any(a.match(n) for a in allowed)]


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("sdist", "wheel"):
        print(__doc__, file=sys.stderr)
        return 2
    check = check_sdist if argv[0] == "sdist" else check_wheel
    failed = False
    for path in argv[1:]:
        bad = check(path)
        print(f"{os.path.basename(path)}: {'ok' if not bad else f'{len(bad)} files outside the allowlist'}")
        for name in bad:
            print(f"  {name}")
        failed = failed or bool(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
