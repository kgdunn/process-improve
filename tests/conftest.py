"""Suite-wide pytest options: run only what a change affects, or one shard of the suite.

``pytest --affected-since=origin/main``
    Run only the test files whose import graph reaches a file changed since the
    merge base with ``origin/main`` (committed, staged, unstaged and untracked
    changes). Runs the whole suite when the change touches something the graph
    cannot reason about. See ``tests/_selection.py``.

``pytest --shard=2/4``
    Run the second of four duration-balanced slices. The four slices together
    are exactly the full suite. Durations come from ``tests/.test_durations.json``;
    refresh that file with ``pytest --store-durations``.

Both work under xdist (``-n auto``): each worker computes the same selection, so
the workers agree on what was collected.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import pytest

from tests._selection import affected_test_files, assign_shard, load_durations

ROOT = Path(__file__).resolve().parents[1]
DURATIONS = Path(__file__).with_name(".test_durations.json")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the selection options."""
    group = parser.getgroup("selection", "test selection (tests/_selection.py)")
    group.addoption("--affected-since", metavar="REF", help="run only tests affected by changes since REF")
    group.addoption("--shard", metavar="K/N", help="run shard K of N (1-based), balanced by recorded durations")
    group.addoption("--store-durations", action="store_true", help=f"write per-file durations to {DURATIONS.name}")


def _git(*args: str) -> list[str]:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)  # noqa: S603, S607
    return result.stdout.splitlines()


@cache
def _affected(ref: str) -> tuple[list[str], set[Path] | None]:
    """Files changed since the merge base with ``ref``, and the test files they affect."""
    base = _git("merge-base", ref, "HEAD")[0]
    changed = sorted({*_git("diff", "--name-only", base), *_git("ls-files", "--others", "--exclude-standard")})
    return changed, affected_test_files(changed, ROOT)


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Say what was selected, once, on the controller."""
    lines = []
    if ref := config.getoption("affected_since"):
        changed, affected = _affected(ref)
        scope = "the whole suite" if affected is None else f"{len(affected)} test file(s)"
        lines.append(f"--affected-since={ref}: {len(changed)} changed file(s) select {scope}")
    if spec := config.getoption("shard"):
        lines.append(f"--shard={spec}: balanced by {DURATIONS.relative_to(ROOT)}")
    return lines


def _keep_files(config: pytest.Config, items: list[pytest.Item], keep: set[str]) -> None:
    """Keep only items from files in ``keep``; report the rest as deselected."""
    dropped = [item for item in items if item.nodeid.split("::")[0] not in keep]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [item for item in items if item.nodeid.split("::")[0] in keep]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply ``--affected-since`` first, then ``--shard`` to whatever remains."""
    if ref := config.getoption("affected_since"):
        _, affected = _affected(ref)
        if affected is not None:
            keep = {item.nodeid.split("::")[0] for item in items if item.path.resolve() in affected}
            _keep_files(config, items, keep)

    if spec := config.getoption("shard"):
        shard, n_shards = (int(x) for x in spec.split("/"))
        tests_per_file = Counter(item.nodeid.split("::")[0] for item in items)
        _keep_files(config, items, assign_shard(tests_per_file, load_durations(DURATIONS), shard, n_shards))


_seconds: defaultdict[str, float] = defaultdict(float)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Accumulate wall time per test file: setup, call and teardown."""
    _seconds[report.nodeid.split("::")[0]] += report.duration


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Write per-file durations when asked. Only the xdist controller writes."""
    if session.config.getoption("store_durations") and not hasattr(session.config, "workerinput"):
        durations = {name: round(seconds, 2) for name, seconds in sorted(_seconds.items())}
        DURATIONS.write_text(json.dumps(durations, indent=1) + "\n", encoding="utf-8")
