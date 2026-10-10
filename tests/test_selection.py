"""Tests for the test-selection plugin in ``tests/_selection.py``."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests._selection import affected_test_files, assign_shard, build_import_graph, load_durations

# A miniature repository that exercises every kind of edge the graph draws.
FAKE_REPO = {
    "src/process_improve/__init__.py": "",
    "src/process_improve/core.py": "def f(): ...",
    "src/process_improve/user.py": "from process_improve.core import f",
    "src/process_improve/lazy.py": "MODULES = ['process_improve.plugin']",
    "src/process_improve/plugin.py": "",
    "src/process_improve/loader.py": "PATH = 'tables/table.csv'",
    "src/process_improve/tables/table.csv": "a,b\n1,2\n",
    "src/process_improve/pkg/__init__.py": "from .inner import g",
    "src/process_improve/pkg/inner.py": "def g(): ...",
    "tests/__init__.py": "",
    "tests/test_core.py": "from process_improve.core import f",
    "tests/test_user.py": "from process_improve.user import f",
    "tests/test_lazy.py": "import process_improve.lazy",
    "tests/test_loader.py": "from process_improve import loader",
    "tests/test_pkg.py": "from process_improve.pkg import g",
    "tests/test_subprocess.py": """
        import subprocess, sys
        CODE = '''
        from process_improve.plugin import h
        print(h)
        '''
        subprocess.run([sys.executable, "-c", CODE], check=True)
    """,
    "tests/test_readme.py": "exec(open('README.md').read())",
    "tests/sub/__init__.py": "",
    "tests/sub/conftest.py": "from process_improve import plugin",
    "tests/sub/test_nothing_imported.py": "",
    "README.md": "Examples that test_readme.py runs.",
    "CHANGELOG.md": "No Python names this file.",
    "pyproject.toml": "",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Write ``FAKE_REPO`` under a temporary directory."""
    for name, text in FAKE_REPO.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    return tmp_path.resolve()


def _affected(repo: Path, *changed: str) -> set[str] | None:
    files = affected_test_files(list(changed), repo)
    return None if files is None else {p.relative_to(repo).as_posix() for p in files}


def test_a_direct_and_a_transitive_importer_are_both_affected(repo: Path) -> None:
    assert _affected(repo, "src/process_improve/core.py") == {
        "tests/test_core.py",
        "tests/test_user.py",  # through user.py
        "tests/test_readme.py",  # opaque: runs code the graph cannot see
    }


def test_a_module_named_in_a_string_literal_is_an_edge(repo: Path) -> None:
    affected = _affected(repo, "src/process_improve/plugin.py")
    assert affected is not None
    assert "tests/test_lazy.py" in affected  # the lazy registry names it
    assert "tests/test_subprocess.py" in affected  # the python -c snippet imports it


def test_an_affected_conftest_affects_every_test_below_it(repo: Path) -> None:
    affected = _affected(repo, "src/process_improve/plugin.py")
    assert affected is not None
    assert "tests/sub/test_nothing_imported.py" in affected


def test_a_relative_import_resolves_against_its_package(repo: Path) -> None:
    affected = _affected(repo, "src/process_improve/pkg/inner.py")
    assert affected == {"tests/test_pkg.py", "tests/test_readme.py"}


def test_a_data_file_affects_the_code_that_names_it(repo: Path) -> None:
    affected = _affected(repo, "src/process_improve/tables/table.csv")
    assert affected == {"tests/test_loader.py", "tests/test_readme.py"}


def test_a_changed_test_file_selects_itself(repo: Path) -> None:
    assert _affected(repo, "tests/test_core.py") == {"tests/test_core.py"}


def test_a_document_affects_the_test_that_names_it(repo: Path) -> None:
    assert _affected(repo, "README.md") == {"tests/test_readme.py"}


def test_a_file_nothing_names_affects_no_test(repo: Path) -> None:
    assert _affected(repo, "CHANGELOG.md") == set()


@pytest.mark.parametrize("changed", ["pyproject.toml", "src/process_improve/deleted.py"])
def test_a_build_file_or_a_deleted_file_selects_the_whole_suite(repo: Path, changed: str) -> None:
    assert _affected(repo, changed) is None


def test_only_test_modules_that_run_unseen_code_are_opaque(repo: Path) -> None:
    graph = build_import_graph(repo)
    assert {p.relative_to(repo).as_posix() for p in graph.opaque} == {"tests/test_readme.py"}


def test_the_real_repository_parses() -> None:
    """Every Python file in this repository goes through the parser without error."""
    root = Path(__file__).resolve().parents[1]
    graph = build_import_graph(root)
    assert root / "tests" / "test_selection.py" in graph.edges


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

TESTS_PER_FILE = {f"t{i}.py": 10 for i in range(9)}
DURATIONS = {"t0.py": 50.0, "t1.py": 40.0, "t2.py": 20.0, "t3.py": 20.0, "t4.py": 10.0, "t5.py": 10.0}


@pytest.mark.parametrize("n_shards", [1, 2, 3, 4, 12])
def test_the_shards_partition_the_suite(n_shards: int) -> None:
    shards = [assign_shard(TESTS_PER_FILE, DURATIONS, k, n_shards) for k in range(1, n_shards + 1)]
    assert set().union(*shards) == set(TESTS_PER_FILE)
    assert sum(len(s) for s in shards) == len(TESTS_PER_FILE)


def test_the_shards_are_balanced_by_duration() -> None:
    # The three unrecorded files are estimated at the median rate, 2 s per test, so
    # 20 s each: 210 s in all. Every cost is a multiple of 10 s, so no split beats
    # 110 s / 100 s, and that is the split the greedy rule finds.
    cost = {**DURATIONS, "t6.py": 20.0, "t7.py": 20.0, "t8.py": 20.0}
    loads = [sum(cost[f] for f in assign_shard(TESTS_PER_FILE, DURATIONS, k, 2)) for k in (1, 2)]
    assert sorted(loads) == [100.0, 110.0]


def test_missing_durations_fall_back_to_test_counts(tmp_path: Path) -> None:
    assert load_durations(tmp_path / "absent.json") == {}
    shards = [assign_shard({"a.py": 30, "b.py": 10, "c.py": 10, "d.py": 10}, {}, k, 2) for k in (1, 2)]
    assert sorted(shards, key=len) == [{"a.py"}, {"b.py", "c.py", "d.py"}]
