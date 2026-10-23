"""Fail if an unrepaired Windows wheel needs a runtime library we may not ship.

`delvewheel show` lists the DLLs the repair step would copy into the wheel and
always exits 0, so its report is parsed here. Only Microsoft's redistributable
C++ and OpenMP runtimes may be bundled; any other DLL, or one it cannot find,
fails the build before the repair runs. LLVM's libomp140, which /openmp:llvm
links, is not redistributable and would be copied silently, so it must never
appear.

Usage: python tools/check_wheel_dlls.py <unrepaired.whl>
"""

from __future__ import annotations

import re
import subprocess
import sys

ALLOWED = {"msvcp140.dll", "vcomp140.dll"}


def bundled(report: str) -> list[str]:
    # The copy list is the indented block after its heading, up to the next
    # blank line or heading.
    lines = report.splitlines()
    out = []
    for i, line in enumerate(lines):
        if line.startswith("The following DLLs will be copied into the wheel"):
            for item in lines[i + 1:]:
                if not item.startswith((" ", "\t")) or not item.strip():
                    break
                out.append(item.strip())
            break
    return out


def offending(report: str) -> list[str]:
    bad = []
    for item in bundled(report):
        name = re.split(r"\s+\(", item, maxsplit=1)[0].lower()
        if name not in ALLOWED or "not found" in item.lower():
            bad.append(item)
    return bad


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    report = subprocess.run([sys.executable, "-m", "delvewheel", "show", argv[0]], capture_output=True,
                            text=True, check=True).stdout
    print(report)
    bad = offending(report)
    if bad:
        print("runtime libraries outside the allowlist:", *bad, sep="\n  ")
        return 1
    print("bundled runtime libraries: all redistributable")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
