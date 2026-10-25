"""Tests of what the package publishes: its extras, and the files its wheel carries."""

from __future__ import annotations

import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXTRAS: dict[str, list[str]] = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "optional-dependencies"
]


def test_the_all_extra_is_exactly_the_union_of_the_others() -> None:
    """``[all]`` is hand-listed, so a package or a floor changed in one extra must change there too (#513)."""
    others = {requirement.replace(" ", "") for name, group in EXTRAS.items() if name != "all" for requirement in group}
    listed = {requirement.replace(" ", "") for requirement in EXTRAS["all"]}
    assert listed == others, (
        f"[all] lacks {sorted(others - listed)} and has {sorted(listed - others)} that no other extra declares"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_the_wheel_carries_every_file_of_the_package(tmp_path: Path) -> None:
    """The built wheel holds each file under ``src/process_improve``: the code, and the data it reads at run time.

    A data file left out of the wheel passes every other test, because they import the package from the
    source tree, and fails only for a user: a dataset loader cannot find its CSV, or the design knowledge
    base its YAML.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv builds the wheel, and it is not on PATH")
    subprocess.run(  # noqa: S603 - the uv found on PATH, with fixed arguments
        [uv, "build", "--wheel", "--out-dir", str(tmp_path)], cwd=ROOT, check=True, capture_output=True
    )
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        packaged = set(archive.namelist())

    source = ROOT / "src"
    expected = {
        path.relative_to(source).as_posix()
        for path in (source / "process_improve").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    }
    assert {"process_improve/py.typed", "process_improve/datasets/multivariate/LDPE/LDPE.csv"} <= expected
    assert any(name.startswith("process_improve/experiments/knowledge/data/") for name in expected)
    assert not expected - packaged, f"the wheel lacks {sorted(expected - packaged)}"
