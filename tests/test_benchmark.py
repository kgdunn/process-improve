"""Tests for the study benchmark: the next step followed round by round on known processes."""

from __future__ import annotations

import pandas as pd
import pytest

from process_improve.experiments import Factor, TrueProcess
from process_improve.experiments.benchmark import CASES, BenchmarkCase, run_benchmark, run_study

BY_NAME = {case.name: case for case in CASES}


def test_the_cases_are_named_once_and_use_only_their_own_factors() -> None:
    assert len(BY_NAME) == len(CASES)
    for case in CASES:
        assert set(case.truth.factors) <= {f.name for f in case.factors}, case.name


def test_the_hypothesised_cases_say_they_are_not_the_hidden_models() -> None:
    for name in ("hardness (02), hypothesised", "impact (03), hypothesised"):
        assert "not the study's hidden model" in BY_NAME[name].note


def test_settings_are_coded_on_the_first_ranges() -> None:
    case = BY_NAME["impact (03), hypothesised"]
    corners = pd.DataFrame([{f.name: f.low for f in case.factors}, {f.name: f.high for f in case.factors}])
    assert case.coded(corners).to_numpy().tolist() == [[-1.0] * 10, [1.0] * 10]


def test_an_interior_optimum_is_confirmed_close_to_the_true_best() -> None:
    study = run_study(BY_NAME["interior optimum, 2^3 + centre"], random_state=0)
    assert study["actions"] == ["augment_to_rsm", "confirm", "stop"]
    assert study["shortfall_in_sigma"] < 0.1
    assert study["runs_used"] > 11  # the eleven runs of the first design, and what followed


def test_an_optimum_outside_the_ranges_is_reached_by_moving() -> None:
    study = run_study(BY_NAME["optimum outside the ranges"], random_state=0)
    assert "move_region" in study["actions"]
    assert study["ended"] == "stop"
    assert study["true_best"] == pytest.approx(98.75)
    assert study["shortfall_in_sigma"] < 0.5


def test_noise_widens_the_ranges_and_stops() -> None:
    study = run_study(BY_NAME["noise only"], random_state=0)
    assert study["actions"] == ["widen_ranges", "stop"]
    assert study["shortfall"] == 0.0


def test_a_target_is_met() -> None:
    case = BenchmarkCase(
        "target",
        (Factor(name="A", low=10, high=20), Factor(name="B", low=0, high=1)),
        TrueProcess({"Intercept": 50, "A": 5, "B": -2}, sigma=0.5),
        {"design_type": "full_factorial", "n_center_points": 3},
        goal="target",
        target=52.0,
    )
    study = run_study(case, random_state=1)
    assert study["ended"] == "stop"
    assert abs(study["true_value"] - 52.0) < 1.0


def test_the_summary_has_a_row_per_case() -> None:
    cases = [BY_NAME["interior optimum, 2^3 + centre"], BY_NAME["noise only"]]
    table = run_benchmark(cases, n_seeds=2)
    assert table["case"].tolist() == [c.name for c in cases]
    assert list(table.columns) == ["case", "shortfall [sd]", "worst [sd]", "runs", "ends", "commonest path"]
    assert table.loc[1, "ends"] == {"stop": 2}


@pytest.mark.slow
@pytest.mark.parametrize("name", [c.name for c in CASES])
def test_every_case_ends_within_its_budget(name: str) -> None:
    case = BY_NAME[name]
    study = run_study(case, random_state=3)
    assert study["ended"] in ("stop", "investigate")
    assert study["runs_used"] <= case.max_runs
    assert study["shortfall_in_sigma"] >= 0.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", ["adhesion, DSD", "adhesion, Plackett-Burman", "curvature in 8 factors, 2^(8-4) + centre"]
)
def test_the_screening_cases_reach_their_best(name: str) -> None:
    study = run_study(BY_NAME[name], random_state=3)
    assert study["ended"] == "stop"
    assert study["shortfall_in_sigma"] < 0.1
