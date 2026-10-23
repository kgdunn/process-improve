"""The coverage shards' durations: combined into one file, and a warning when they are stale."""

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


COMMITTED = {"a.py": 10.0, "b.py": 20.0, "c.py": 30.0, "gone.py": 5.0}


def test_each_file_takes_the_value_of_the_shard_that_ran_it(durations):
    # Shard 1 ran a.py and c.py, shard 2 ran b.py and the new file d.py; nothing ran gone.py.
    shard_1 = {**COMMITTED, "a.py": 12.0, "c.py": 33.0}
    shard_2 = {**COMMITTED, "b.py": 18.0, "d.py": 7.0}
    combined, loads = durations.combine(COMMITTED, [shard_1, shard_2])
    assert combined == {"a.py": 12.0, "b.py": 18.0, "c.py": 33.0, "d.py": 7.0, "gone.py": 5.0}
    assert list(combined) == sorted(combined)
    assert loads == [45.0, 25.0]


def test_a_file_measured_again_at_its_committed_value_keeps_it(durations):
    combined, loads = durations.combine(COMMITTED, [dict(COMMITTED)])
    assert combined == COMMITTED
    assert loads == [0.0]


@pytest.mark.parametrize(
    ("loads", "expected"),
    [([100.0, 90.0, 80.0, 76.0], False), ([100.0, 90.0, 74.0], True), ([0.0, 0.0], False), ([], False)],
)
def test_the_durations_are_stale_when_the_shards_differ_by_more_than_a_quarter(durations, loads, expected):
    assert durations.stale(loads) is expected


def test_the_script_writes_the_combined_file_and_warns_when_stale(durations, tmp_path, capsys):
    paths = []
    for name, content in [("committed", COMMITTED), ("s1", {**COMMITTED, "a.py": 100.0}), ("s2", COMMITTED)]:
        paths.append(tmp_path / f"{name}.json")
        paths[-1].write_text(json.dumps(content), encoding="utf-8")
    output = tmp_path / "combined.json"
    durations.main([*map(str, paths), "-o", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))["a.py"] == 100.0
    printed = capsys.readouterr().out
    assert "shard 1: 100 s of tests" in printed
    assert "shard 2: 0 s of tests" in printed
    assert "::warning title=Stale test durations::" in printed
