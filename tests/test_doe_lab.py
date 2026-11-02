"""Tests for scripts/doe_lab.py: the blind laboratory an agent under evaluation makes its runs in."""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from process_improve.experiments.benchmark import CASES

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

ADHESION = next(case for case in CASES if case.name == "adhesion, DSD")
IMPACT = next(case for case in CASES if case.name == "impact (03), hypothesised")
CENTRE = dict.fromkeys("ABCDEFGH", 0.0)


@pytest.fixture(scope="module")
def doe_lab() -> ModuleType:
    """Return the script, loaded as a module."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "doe_lab.py"
    spec = importlib.util.spec_from_file_location("doe_lab", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_the_problem_names_the_factors_and_keeps_the_process_to_itself(doe_lab: ModuleType, tmp_path: Path) -> None:
    lab = doe_lab.Lab(IMPACT, seed=0, budget=60, log=tmp_path / "lab.jsonl")
    problem = lab.problem()
    assert [f["name"] for f in problem["factors"]] == [f"X{i:02d}" for i in range(1, 11)]
    assert problem["factors"][8] | {"description": ""} == {
        "name": "X09",
        "low": 216.0,
        "high": 241.0,
        "units": "",
        "description": "",
    }
    assert "Peak metal temperature" in problem["factors"][8]["description"]
    assert problem["goal"] == "maximize"
    assert problem["budget"] == 60
    assert "Intercept" not in json.dumps(problem)  # nothing of the true model


def test_runs_are_measured_logged_and_charged(doe_lab: ModuleType, tmp_path: Path) -> None:
    log = tmp_path / "lab.jsonl"
    lab = doe_lab.Lab(ADHESION, seed=0, budget=10, log=log)
    made = lab.make([CENTRE, {**CENTRE, "A": 1.0, "C": 1.0, "G": -1.0}])
    assert made["runs_used"] == 2
    assert made["runs_left"] == 8
    assert made["runs"][1]["y"] > made["runs"][0]["y"]  # 131 against 100, noise sd 3
    assert [r["event"] for r in _records(log)] == ["start", "runs"]


@pytest.mark.parametrize(
    ("runs", "message"),
    [
        ([{**CENTRE, "A": 1.5}], "inside its range"),
        ([{"A": 0.0}], "a number for every factor"),
        ([{**CENTRE, "B": "high"}], "a number for every factor"),
        ([CENTRE] * 11, "Only 10 runs are left"),
    ],
)
def test_runs_outside_the_ranges_or_the_budget_are_refused_and_not_charged(
    doe_lab: ModuleType, tmp_path: Path, runs: list[dict[str, Any]], message: str
) -> None:
    lab = doe_lab.Lab(ADHESION, seed=0, budget=10, log=tmp_path / "lab.jsonl")
    assert message in lab.make(runs)["error"]
    assert lab.runs == []


def test_an_answer_at_the_true_best_falls_short_by_nothing(doe_lab: ModuleType, tmp_path: Path) -> None:
    log = tmp_path / "lab.jsonl"
    lab = doe_lab.Lab(ADHESION, seed=0, budget=10, log=log)
    lab.make([CENTRE] * 3)
    lab.answer({"settings": {**CENTRE, "A": 1.0, "C": 1.0, "G": -1.0}, "predicted": 130.0, "summary": "A, C, G"})
    scored = doe_lab.score(log)
    assert scored["runs_used"] == 3
    assert scored["inside_ranges"] is True
    assert scored["true_value"] == pytest.approx(131.0)
    assert scored["shortfall_in_sigma"] == pytest.approx(0.0)


def test_only_the_last_answer_counts_and_a_missing_one_scores_nothing(doe_lab: ModuleType, tmp_path: Path) -> None:
    log = tmp_path / "lab.jsonl"
    lab = doe_lab.Lab(ADHESION, seed=0, budget=10, log=log)
    assert doe_lab.score(log)["answered"] is False
    lab.answer({"settings": {**CENTRE, "A": 1.0, "C": 1.0, "G": -1.0}, "predicted": 131.0})
    lab.answer({"settings": CENTRE, "predicted": 100.0})
    assert doe_lab.score(log)["shortfall_in_sigma"] == pytest.approx(31 / 3)


@pytest.fixture
def served(doe_lab: ModuleType, tmp_path: Path) -> Iterator[str]:
    """Serve a lab on a free port and return its address."""
    lab = doe_lab.Lab(ADHESION, seed=0, budget=5, log=tmp_path / "lab.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", 0), doe_lab._handler(lab))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join()


def _call(url: str, body: bytes | None = None) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, data=body, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_endpoints_answer_over_http(served: str) -> None:
    status, problem = _call(f"{served}/problem")
    assert status == 200
    assert problem["budget"] == 5
    status, made = _call(f"{served}/runs", json.dumps({"runs": [CENTRE]}).encode())
    assert status == 200
    assert made["runs_used"] == 1
    assert _call(f"{served}/history")[1]["runs"] == made["runs"]
    assert _call(f"{served}/answer", json.dumps({"settings": CENTRE}).encode()) == (200, {"received": True})


@pytest.mark.parametrize(
    ("path", "body", "status"),
    [
        ("/runs", json.dumps({"runs": [{**CENTRE, "A": 9}]}).encode(), 400),
        ("/runs", b"not json", 400),
        ("/model", None, 404),
        ("/model", b"{}", 404),
    ],
)
def test_bad_requests_are_refused(served: str, path: str, body: bytes | None, status: int) -> None:
    assert _call(f"{served}{path}", body)[0] == status
