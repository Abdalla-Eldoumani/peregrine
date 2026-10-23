"""Fail if any object built from a fallback source contains a ymm instruction.

The fallback sources are the module's .cpp files that are not in PG_AVX2_SOURCES,
both lists read from CMakeLists.txt so this check follows the build. MSVC compiles
AVX2 intrinsics without /arch:AVX2 silently, and such code would fault at import
on a CPU without AVX2, so the objects themselves are disassembled.

Usage: python tools/check_no_ymm.py <build-dir>
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cmake_block(text: str, opener: str) -> str:
    start = text.index(opener)
    return text[start:text.index(")", start)]


def fallback_stems() -> list[str]:
    text = open(os.path.join(ROOT, "CMakeLists.txt"), encoding="utf-8").read()
    module = re.findall(r"src/[\w/]+\.cpp", _cmake_block(text, "nanobind_add_module(_core"))
    avx2 = set(re.findall(r"src/[\w/]+\.cpp", _cmake_block(text, "set(PG_AVX2_SOURCES")))
    stems = [os.path.splitext(os.path.basename(p))[0] for p in module if p not in avx2]
    if not stems:
        sys.exit("check_no_ymm: found no fallback sources in CMakeLists.txt")
    return stems


def find_object(build_dir: str, stem: str) -> str:
    hits = []
    for pattern in (f"{stem}.obj", f"{stem}.cpp.obj", f"{stem}.cpp.o", f"{stem}.o"):
        hits += glob.glob(os.path.join(build_dir, "**", pattern), recursive=True)
    if not hits:
        sys.exit(f"check_no_ymm: no object for {stem} under {build_dir}")
    return max(hits, key=os.path.getmtime)


def disassembler() -> list[str]:
    if os.name != "nt":
        return ["objdump", "-d", "--no-show-raw-insn"]
    found = shutil.which("dumpbin")
    if not found:
        vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                               "Microsoft Visual Studio", "Installer", "vswhere.exe")
        out = subprocess.run([vswhere, "-latest", "-products", "*", "-find",
                              r"VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe"],
                             capture_output=True, text=True, check=False).stdout.splitlines()
        found = out[0] if out else None
    if not found:
        sys.exit("check_no_ymm: dumpbin not found")
    return [found, "/disasm:nobytes"]


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tool = disassembler()
    bad = 0
    for stem in fallback_stems():
        obj = find_object(sys.argv[1], stem)
        listing = subprocess.run([*tool, obj], capture_output=True, text=True, check=True).stdout
        hits = [line.strip() for line in listing.splitlines() if re.search(r"\bymm\d", line, re.I)]
        print(f"{stem}: {len(hits)} ymm instructions ({os.path.relpath(obj, sys.argv[1])})")
        for line in hits[:5]:
            print(f"    {line}")
        bad += bool(hits)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
