"""Prepare and publish a GitHub release from built distributions.

One script for both the pull-request dry run and the tag run, so everything but
the upload is exercised before a tag exists:

    python tools/release.py --dist dist --dry-run
    python tools/release.py --dist dist --tag v3.1.0 --prepare
    python tools/release.py --dist dist --tag v3.1.0 --publish

--prepare (implied by --dry-run) checks that the tag, pyproject.toml, and every
file name carry the same version, writes the release notes (the version's
CHANGELOG.md section) and SHA256SUMS into the dist directory, and prints the
`gh release create` command. --publish runs that command; the workflow attests
the files between the two steps.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_version() -> str:
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]["version"]


def changelog_section(version: str) -> str:
    text = open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8").read()
    head = re.compile(rf"^## \[?{re.escape(version)}\]?(\s|$)", re.M)
    m = head.search(text)
    if m is None:
        raise SystemExit(f"CHANGELOG.md has no section for {version}")
    end = re.compile(r"^## ", re.M).search(text, m.end())
    body = text[m.end():end.start() if end else len(text)].strip()
    if not body:
        raise SystemExit(f"CHANGELOG.md section for {version} is empty")
    return body + "\n"


def distributions(dist: str, version: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(dist, "*.whl")) + glob.glob(os.path.join(dist, "*.tar.gz")))
    if not files:
        raise SystemExit(f"no wheels or sdist in {dist}")
    for f in files:
        name = os.path.basename(f)
        if not name.startswith(f"peregrine-{version}-") and name != f"peregrine-{version}.tar.gz":
            raise SystemExit(f"{name} does not carry version {version}")
    return files


def sha256sums(files: list[str]) -> str:
    lines = []
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {os.path.basename(f)}")
    return "\n".join(lines) + "\n"


def prepare(dist: str, tag: str | None) -> list[str]:
    version = project_version()
    expected = f"v{version}"
    if tag is not None and tag != expected:
        raise SystemExit(f"tag {tag} does not match pyproject.toml version {version} (expected {expected})")
    files = distributions(dist, version)
    notes = os.path.join(dist, "RELEASE_NOTES.md")
    sums = os.path.join(dist, "SHA256SUMS")
    with open(notes, "w", encoding="utf-8", newline="\n") as f:
        f.write(changelog_section(version))
    with open(sums, "w", encoding="utf-8", newline="\n") as f:
        f.write(sha256sums(files))
    return ["gh", "release", "create", expected, *files, sums, "--verify-tag", "--title", f"peregrine {version}",
            "--notes-file", notes]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dist", required=True)
    p.add_argument("--tag")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--publish", action="store_true")
    args = p.parse_args(argv)
    cmd = prepare(args.dist, args.tag)
    print(" ".join(cmd))
    if args.publish:
        return subprocess.run(cmd, check=False).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
