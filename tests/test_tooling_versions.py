"""The pre-commit hooks run the tool versions that CI pins (#39).

CI's lint and typecheck jobs run ruff and mypy at the versions ``pyproject.toml``
allows; pre-commit (locally, and on pre-commit.ci) runs whatever revision
``.pre-commit-config.yaml`` names. When the two drift apart, code the hook formatted
can still fail CI's ``ruff format --check .``. pre-commit.ci's autoupdate moves the
hook revisions on its own schedule, so these tests fail such an update until the pin
in ``pyproject.toml`` moves with it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]

#: The hook repository that runs each tool CI also pins.
HOOK_REPOSITORIES = {
    "ruff": "https://github.com/astral-sh/ruff-pre-commit",
    "mypy": "https://github.com/pre-commit/mirrors-mypy",
}


def _config() -> dict:
    return yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))


def _hook_version(repository: str) -> Version:
    (revision,) = [entry["rev"] for entry in _config()["repos"] if entry["repo"] == repository]
    return Version(revision.removeprefix("v"))


def _pins(tool: str) -> list[Requirement]:
    """Every requirement on ``tool`` in pyproject.toml, wherever it is declared."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return [Requirement(spec) for spec in re.findall(rf'"({tool}\s*[<>=!~][^"]*)"', text)]


@pytest.mark.parametrize("tool", sorted(HOOK_REPOSITORIES))
def test_the_hook_runs_a_version_ci_allows(tool: str) -> None:
    version, pins = _hook_version(HOOK_REPOSITORIES[tool]), _pins(tool)
    assert pins, f"pyproject.toml does not pin {tool}."
    for pin in pins:
        assert pin.specifier.contains(version), (
            f".pre-commit-config.yaml runs {tool} {version}, outside the pyproject.toml pin {pin}. "
            "Move the pin with the hook, or CI and pre-commit will disagree about the same code."
        )


def test_pre_commit_ci_skips_only_hooks_that_exist() -> None:
    """A misspelt id in ``ci.skip`` would silently leave that hook running."""
    config = _config()
    hook_ids = {hook["id"] for entry in config["repos"] for hook in entry["hooks"]}
    assert set(config["ci"]["skip"]) <= hook_ids
