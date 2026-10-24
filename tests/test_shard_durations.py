"""The coverage shards' durations: combined into one file at one speed, and a warning when they are stale."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "shard_durations.py"


@pytest.fixture(scope="module")
def durations():
    """Import `tools/shard_durations.py`, which is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("shard_durations_tool", _TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["shard_durations_tool"] = module
    spec.loader.exec_module(module)
    return module


COMMITTED = {"a.py": 10.0, "b.py": 20.0, "c.py": 30.0, "e.py": 8.0, "gone.py": 5.0}


def test_each_file_takes_the_value_of_the_shard_that_ran_it_at_the_committed_speed(durations):
    # Shard 1's runner took twice as long on a.py and b.py, so its speed is 2; c.py took three
    # times as long, so it really grew. Shard 2's runner was twice as fast, and ran the new d.py.
    shard_1 = {**COMMITTED, "a.py": 20.0, "b.py": 40.0, "c.py": 90.0}
    shard_2 = {**COMMITTED, "e.py": 4.0, "d.py": 3.5}
    combined, shards = durations.combine(COMMITTED, [shard_1, shard_2])
    assert combined == {"a.py": 10.0, "b.py": 20.0, "c.py": 45.0, "d.py": 7.0, "e.py": 8.0, "gone.py": 5.0}
    assert list(combined) == sorted(combined)
    assert [(s.speed, s.load) for s in shards] == [(2.0, 75.0), (0.5, 15.0)]


def test_a_shard_of_new_files_only_is_taken_at_its_own_speed(durations):
    combined, shards = durations.combine(COMMITTED, [{**COMMITTED, "new.py": 4.0}])
    assert combined["new.py"] == 4.0
    assert [(s.speed, s.load) for s in shards] == [(1.0, 4.0)]


@pytest.mark.parametrize(
    ("loads", "expected"),
    [([100.0, 90.0, 80.0, 76.0], False), ([100.0, 90.0, 74.0], True), ([0.0, 0.0], False), ([], False)],
)
def test_the_durations_are_stale_when_the_shards_differ_by_more_than_a_quarter(durations, loads, expected):
    assert durations.stale(loads) is expected


def test_the_script_writes_the_combined_file_and_warns_when_stale(durations, tmp_path, capsys):
    # Shard 1's a.py grew ninefold against its other files, which set the runner's speed at 1.1.
    shard_1 = {**COMMITTED, "a.py": 100.0, "b.py": 22.0, "c.py": 33.0}
    shard_2 = {**COMMITTED, "e.py": 8.5}
    paths = []
    for name, content in [("committed", COMMITTED), ("s1", shard_1), ("s2", shard_2)]:
        paths.append(tmp_path / f"{name}.json")
        paths[-1].write_text(json.dumps(content), encoding="utf-8")
    output = tmp_path / "combined.json"
    durations.main([*map(str, paths), "-o", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))["a.py"] == 90.91
    printed = capsys.readouterr().out
    assert "shard 1: 141 s of tests at the committed speed; its runner took 1.10x" in printed
    assert "shard 2: 8 s of tests" in printed
    assert "::warning title=Stale test durations::" in printed
