"""Release-readiness checks for the 3.0.0 cut. Pure-Python and ungated: no GPU,
no native build, so this file runs on every build including the WSL CPU-only
clone (it is the BNCH-04 unit coverage the fresh-clone gate leans on).

The version single-source proof: fme.__version__ must equal both the literal
"3.0.0" and importlib.metadata.version("fastmathext"), so a drift between
pyproject and the package would fail here rather than ship. pyproject is read
with tomllib (stdlib 3.11+), json.load's TOML sibling, never eval'd.
"""

from importlib.metadata import version as _metadata_version
import os
import tomllib

import fastmathext as fme

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
    assert fme.__version__ == "3.0.0"
    assert fme.__version__ == _metadata_version("fastmathext")


def test_pyproject_version():
    # pyproject is the one place the version is typed; the metadata read above
    # resolves against it. The [project] table version must be the 3.0.0 release.
    data = _load_pyproject()
    assert data["project"]["version"] == "3.0.0"


def test_pyproject_readme_key():
    # The [project] readme key drives the long-description on a build; BNCH-04
    # restores it for the release. Its presence and value are the contract.
    data = _load_pyproject()
    assert data["project"]["readme"] == "README.md"


def test_matplotlib_in_bench_not_test():
    # matplotlib belongs in the [bench] extras (plot regeneration), NOT [test]:
    # the WSL CPU-only test gate installs [test] and must stay lean (the suite
    # itself imports no matplotlib). A package-name prefix check tolerates the
    # version pin (e.g. "matplotlib==3.9.1").
    extras = _load_pyproject()["project"]["optional-dependencies"]
    bench = [d for d in extras.get("bench", []) if d.startswith("matplotlib")]
    test = [d for d in extras.get("test", []) if d.startswith("matplotlib")]
    assert bench, "matplotlib must be pinned in the [bench] extras"
    assert not test, "matplotlib must not be in [test] (keep the WSL gate lean)"
