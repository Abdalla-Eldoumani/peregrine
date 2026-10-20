"""Release-readiness checks for the 3.1.0 cut. Pure-Python and ungated: no GPU,
no native build, so this file runs on every build including the WSL CPU-only
clone (it is the release-readiness unit coverage the fresh-clone gate leans on).

The version single-source proof: pg.__version__ must equal both the literal
"3.1.0" and importlib.metadata.version("peregrine"), so a drift between
pyproject and the package would fail here rather than ship. pyproject is read
with tomllib (stdlib 3.11+), json.load's TOML sibling, never eval'd.
"""

from importlib.metadata import version as _metadata_version
import hashlib
import importlib.util
import os
import tomllib
import zipfile

import pytest

import peregrine as pg

# pyproject.toml sits at the repo root, two levels up from this test file.
_PYPROJECT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml"
)


def _load_pyproject() -> dict:
    # tomllib is the stdlib TOML reader (3.11+); it parses, it does not execute,
    # so reading the project's own pyproject is safe (the Security V5 json-only
    # rule's TOML analog). Binary mode is required by tomllib.load.
    with open(_PYPROJECT, "rb") as f:
        return tomllib.load(f)


def test_version_single_source():
    # The genuine single source: __version__ is read from importlib.metadata in
    # __init__.py, so it must equal both the released literal and the installed
    # package metadata. If pyproject and the package ever drift, this fails.
    assert pg.__version__ == "3.1.0"
    assert pg.__version__ == _metadata_version("peregrine")


def test_pyproject_version():
    # pyproject is the one place the version is typed; the metadata read above
    # resolves against it. The [project] table version must be the 3.1.0 release.
    data = _load_pyproject()
    assert data["project"]["version"] == "3.1.0"


def test_pyproject_readme_key():
    # The [project] readme key drives the long-description on a build; the
    # release cut restores it. Its presence and value are the contract.
    data = _load_pyproject()
    assert data["project"]["readme"] == "README.md"


def test_matplotlib_in_bench_not_test():
    # matplotlib belongs in the [bench] extras (plot regeneration), NOT [test]:
    # the WSL CPU-only test gate installs [test] and must stay lean (the suite
    # itself imports no matplotlib). A package-name prefix check tolerates the
    # version pin (e.g. "matplotlib==3.11.2").
    extras = _load_pyproject()["project"]["optional-dependencies"]
    bench = [d for d in extras.get("bench", []) if d.startswith("matplotlib")]
    test = [d for d in extras.get("test", []) if d.startswith("matplotlib")]
    assert bench, "matplotlib must be pinned in the [bench] extras"
    assert not test, "matplotlib must not be in [test] (keep the WSL gate lean)"


# The release tools are scripts, not package modules, so they load by path.
_TOOLS = os.path.join(os.path.dirname(_PYPROJECT), "tools")


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_TOOLS, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changelog_has_a_section_for_this_version():
    release = _tool("release")
    body = release.changelog_section(release.project_version())
    assert body.strip() and "\n## " not in body


def test_release_prepare_writes_notes_and_checksums(tmp_path):
    release = _tool("release")
    v = release.project_version()
    files = [tmp_path / f"peregrine-{v}-cp312-abi3-win_amd64.whl", tmp_path / f"peregrine-{v}.tar.gz"]
    for i, f in enumerate(files):
        f.write_bytes(bytes([i]) * 100)
    cmd = release.prepare(str(tmp_path), f"v{v}")
    assert cmd[:4] == ["gh", "release", "create", f"v{v}"] and "--verify-tag" in cmd
    sums = (tmp_path / "SHA256SUMS").read_text().splitlines()
    want = sorted(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in files)
    assert sorted(sums) == want
    assert (tmp_path / "RELEASE_NOTES.md").read_text(encoding="utf-8") == release.changelog_section(v)


def test_release_refuses_a_mismatched_tag_or_file(tmp_path):
    release = _tool("release")
    v = release.project_version()
    (tmp_path / f"peregrine-{v}.tar.gz").write_bytes(b"x")
    with pytest.raises(SystemExit, match="does not match"):
        release.prepare(str(tmp_path), "v0.0.1")
    (tmp_path / "peregrine-0.0.1-cp312-abi3-win_amd64.whl").write_bytes(b"x")
    with pytest.raises(SystemExit, match="does not carry version"):
        release.prepare(str(tmp_path), None)


def test_wheel_allowlist_names_only_the_strays(tmp_path):
    check = _tool("check_dist_contents")
    ok = [
        "peregrine/__init__.py",
        "peregrine/_core.pyd",
        "peregrine/_core.abi3.so",
        "peregrine.libs/vcomp140-" + "0" * 32 + ".dll",
        "peregrine.libs/libgomp-a34b3233.so.1.0.0",
        "peregrine-3.1.0.dist-info/RECORD",
        "peregrine-3.1.0.dist-info/licenses/LICENSE",
        "peregrine-3.1.0.dist-info/sboms/auditwheel.cdx.json",
    ]
    strays = ["peregrine/_core.pdb", "peregrine.libs/libomp140.x86_64-" + "0" * 32 + ".dll", "notes/todo.txt"]
    whl = tmp_path / "w.whl"
    with zipfile.ZipFile(whl, "w") as z:
        for name in ok + strays:
            z.writestr(name, b"")
        z.writestr("peregrine/", b"")
    assert check.check_wheel(str(whl)) == strays


_SHOW = """Analyzing peregrine-3.1.0-cp312-abi3-win_amd64.whl

The following DLLs will be copied into the wheel.
    msvcp140.dll (C:/Windows/System32/msvcp140.dll)
    vcomp140.dll (C:/Windows/System32/vcomp140.dll)
{extra}
The following DLLs are assumed to be present in the end user's environment and will not be copied into the wheel.
    kernel32.dll
    python3.dll
"""


def test_dll_check_allows_only_the_redistributable_runtimes():
    check = _tool("check_wheel_dlls")
    assert check.offending(_SHOW.format(extra="")) == []
    llvm = "    libomp140.x86_64.dll (C:/Windows/System32/libomp140.x86_64.dll)\n"
    assert check.offending(_SHOW.format(extra=llvm)) == [llvm.strip()]
    missing = "    vcomp140d.dll (Error: Not Found)\n"
    assert check.offending(_SHOW.format(extra=missing)) == [missing.strip()]
