"""The pre-commit hooks run the tool versions that CI pins (#39), and CI mutates every module mutmut lists.

CI's lint and typecheck jobs run ruff and mypy at the versions ``pyproject.toml``
allows; pre-commit (locally, and on pre-commit.ci) runs whatever revision
``.pre-commit-config.yaml`` names. When the two drift apart, code the hook formatted
can still fail CI's ``ruff format --check .``. pre-commit.ci's autoupdate moves the
hook revisions on its own schedule, so these tests fail such an update until the pin
in ``pyproject.toml`` moves with it.
"""

from __future__ import annotations

import ast
import re
import tomllib
from fnmatch import fnmatch
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


def _mutant_stems(module: str) -> list[str]:
    """Return the name mutmut gives the first mutant of each function and method of ``module``.

    A function's mutants are named ``<module>.x_<function>__mutmut_<n>``, a method's
    ``<module>.xǁ<class>ǁ<method>__mutmut_<n>``; a nested function is mutated with its parent.
    """
    dotted = module.removeprefix("src/").removesuffix(".py").replace("/", ".")
    stems = []
    for node in ast.parse((ROOT / module).read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stems.append(f"{dotted}.x_{node.name}__mutmut_1")
        elif isinstance(node, ast.ClassDef):
            stems.extend(
                f"{dotted}.x\u01c1{node.name}\u01c1{item.name}__mutmut_1"
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return stems


def test_every_mutated_function_is_in_exactly_one_mutation_job() -> None:
    """Each function of every module in ``[tool.mutmut]`` is mutated by exactly one job of ``mutation.yml``.

    The workflow runs the modules in parallel jobs, each naming its mutants by pattern; a
    large module may be split between jobs by function, so every function is checked.
    """
    modules = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["mutmut"]["only_mutate"]
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "mutation.yml").read_text(encoding="utf-8"))
    groups = workflow["jobs"]["mutmut"]["strategy"]["matrix"]["shard"]
    for module in modules:
        stems = _mutant_stems(module)
        assert stems, f"{module} has no function for mutmut to mutate."
        for mutant in stems:
            jobs = [
                group["name"] for group in groups if any(fnmatch(mutant, glob) for glob in group["patterns"].split())
            ]
            assert len(jobs) == 1, f"{mutant} is mutated by {jobs or 'no job'} of mutation.yml; it needs exactly one."
