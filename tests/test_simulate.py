"""Tests for simulate_design_performance: designs judged by simulated studies on a hypothesised process."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import process_improve.experiments.tools  # noqa: F401 - registers the experiments tools
from process_improve.experiments import Factor, TrueProcess, generate_design, simulate_design_performance
from process_improve.tool_spec import execute_tool_call

ADHESION = TrueProcess({"Intercept": 100, "A": 10, "C": 7, "G": -6, "A:C": 8}, sigma=3)


def _factors(names: str) -> list[Factor]:
    return [Factor(name=n, low=-1, high=1) for n in names]


def _coded(names: str, **design: object) -> pd.DataFrame:
    return generate_design(_factors(names), **design).design[list(names)]  # type: ignore[arg-type]


class TestWhatTheStudiesFind:
    """The shares of studies that find the true terms, by what each design can and cannot separate."""

    def test_a_dsd_finds_the_adhesion_process(self) -> None:
        report = simulate_design_performance(
            {"DSD": _coded("ABCDEFGH", design_type="dsd", n_center_points=5)}, ADHESION, n_simulations=40
        )
        dsd = report["designs"]["DSD [n=21]"]
        assert dsd["found"] == {"A": 1.0, "C": 1.0, "G": 1.0, "A:C": 1.0}
        assert dsd["found_all"] == 1.0
        assert dsd["found_exactly"] <= dsd["found_all"]
        assert dsd["judged_by"] == {"the staged analysis of a definitive screening design (analyze_omars)": 1.0}
        assert dsd["summary"].startswith("Finds every true term in 100% of studies (A 100%, C 100%, G 100%, A:C 100%)")

    @pytest.mark.slow
    def test_twelve_runs_without_centre_runs_miss_what_a_dsd_finds(self) -> None:
        """In a 12-run Plackett-Burman design the A:C interaction biases every main effect outside it."""
        designs = {
            "PB": _coded("ABCDEFGH", design_type="plackett_burman", n_center_points=0),
            "DSD": _coded("ABCDEFGH", design_type="dsd", n_center_points=5),
        }
        report = simulate_design_performance(designs, ADHESION, n_simulations=60)
        pb, dsd = report["designs"]["PB [n=12]"], report["designs"]["DSD [n=21]"]
        assert pb["found_all"] < 0.8 < dsd["found_all"]
        assert pb["nothing_stood_out"] > 0.2

    def test_a_term_found_through_its_exact_alias_counts_as_found(self) -> None:
        """In 2^(7-4) with D = AB the analysis names D for A:B: the design can do no better; it is no false alarm."""
        truth = TrueProcess({"Intercept": 50, "A": 6, "B": 5, "A:B": 4}, sigma=0.8)
        runs = _coded("ABCDEFG", design_type="fractional_factorial", resolution=3, n_center_points=0)
        report = simulate_design_performance(runs, truth, n_simulations=30)
        (only,) = report["designs"].values()
        assert only["found"]["A:B"] > 0.9
        assert "D" not in only["false_alarm_terms"]

    def test_a_square_is_found_through_the_curvature_of_the_centre_runs(self) -> None:
        truth = TrueProcess({"Intercept": 60, "A": 3, "I(A ** 2)": -5}, sigma=0.5)
        runs = _coded("ABC", design_type="full_factorial", n_center_points=3)
        report = simulate_design_performance(runs, truth, n_simulations=30)
        (only,) = report["designs"].values()
        assert only["found"]["I(A ** 2)"] > 0.9
        assert "curvature" not in only["false_alarm_terms"]

    def test_noise_alone_rarely_stands_out(self) -> None:
        report = simulate_design_performance(
            _coded("ABCD", design_type="full_factorial", n_center_points=3),
            TrueProcess({"Intercept": 20}, sigma=2),
            n_simulations=60,
        )
        (only,) = report["designs"].values()
        assert only["found"] == {}
        assert only["nothing_stood_out"] > 0.85
        assert only["false_alarms"] < 0.3


class TestTowardAGoal:
    """How far the best settings a study predicts fall short of the true best inside the ranges."""

    @pytest.mark.parametrize(("goal", "target"), [("maximize", None), ("minimize", None), ("target", 55.0)])
    def test_the_true_optimum_is_found_inside_the_ranges(self, goal: str, target: float | None) -> None:
        truth = TrueProcess({"Intercept": 60, "A": 2, "B": -1.5, "I(A ** 2)": -6, "I(B ** 2)": -4}, sigma=0.7)
        runs = _coded("AB", design_type="ccd", alpha="face_centered", n_center_points=3)
        best = simulate_design_performance(runs, truth, goal=goal, target=target, n_simulations=2)["true_optimum"]  # type: ignore[arg-type]
        if goal == "maximize":
            assert best["coded"] == pytest.approx({"A": 1 / 6, "B": -1.5 / 8}, abs=1e-6)
        elif goal == "minimize":
            assert best["response"] == pytest.approx(60 - 2 - 6 - 1.5 - 4)  # A = -1, B = +1
        else:
            assert best["response"] == pytest.approx(55.0, abs=1e-6)

    def test_a_response_surface_design_gets_close_to_the_best(self) -> None:
        truth = TrueProcess({"Intercept": 60, "A": 2, "B": -1.5, "I(A ** 2)": -6, "I(B ** 2)": -4}, sigma=0.7)
        runs = _coded("AB", design_type="ccd", alpha="face_centered", n_center_points=3)
        report = simulate_design_performance(runs, truth, goal="maximize", n_simulations=40)
        (only,) = report["designs"].values()
        assert only["shortfall"]["mean_in_sigma"] < 0.5
        assert 0 <= only["shortfall"]["median"] <= only["shortfall"]["p90"] <= only["shortfall"]["worst"]
        assert "fall short of the true best by" in only["summary"]

    def test_a_shortfall_of_nothing_reads_as_the_best(self) -> None:
        report = simulate_design_performance(
            {"DSD": _coded("ABCDEFGH", design_type="dsd", n_center_points=5)},
            ADHESION,
            goal="maximize",
            n_simulations=20,
        )
        dsd = report["designs"]["DSD [n=21]"]
        assert dsd["shortfall"]["worst"] == 0.0
        assert dsd["summary"].endswith("its predicted best settings are the true best in every study.")


class TestInputs:
    def test_the_same_arguments_give_the_same_report(self) -> None:
        runs = _coded("ABC", design_type="full_factorial", n_center_points=3)
        truth = TrueProcess({"Intercept": 10, "A": 1, "B:C": 0.5}, sigma=1)
        first = simulate_design_performance(runs, truth, n_simulations=20, random_state=7)
        assert simulate_design_performance(runs, truth, n_simulations=20, random_state=7) == first

    def test_a_design_result_and_a_list_of_them_are_labelled_by_their_type(self) -> None:
        factors = _factors("ABC")
        result = generate_design(factors, design_type="box_behnken", n_center_points=3)
        truth = TrueProcess({"Intercept": 10, "A": 1}, sigma=1)
        assert list(simulate_design_performance([result], truth, n_simulations=2)["designs"]) == ["box_behnken [n=15]"]
        assert list(simulate_design_performance(result, truth, n_simulations=2)["designs"]) == ["design [n=15]"]

    @pytest.mark.parametrize(
        ("change", "error", "message"),
        [
            ({"truth": TrueProcess({"Intercept": 1, "Z": 1}, sigma=1)}, ValueError, r"factor\(s\) \['Z'\]"),
            ({"truth": {"A": 1}}, TypeError, "must be a TrueProcess"),
            ({"goal": "best"}, ValueError, "goal must be"),
            ({"goal": "target"}, ValueError, "needs its target"),
            ({"n_simulations": 0}, ValueError, "n_simulations"),
            ({"alpha": 1.5}, ValueError, "alpha"),
            ({"designs": {}}, ValueError, "at least one design"),
        ],
    )
    def test_bad_inputs_are_refused(self, change: dict, error: type, message: str) -> None:
        arguments = {
            "designs": _coded("ABC", design_type="full_factorial", n_center_points=1),
            "truth": TrueProcess({"Intercept": 1, "A": 1}, sigma=1),
            "n_simulations": 2,
        } | change
        with pytest.raises(error, match=message):
            simulate_design_performance(**arguments)


class TestThroughTheTools:
    def test_simulate_returns_json(self) -> None:
        designs = {
            "2^3+3c": _coded("ABC", design_type="full_factorial", n_center_points=3).to_dict(orient="records"),
            "BBD": _coded("ABC", design_type="box_behnken", n_center_points=3).to_dict(orient="records"),
        }
        result = execute_tool_call(
            "simulate_design_performance",
            {
                "designs": designs,
                "true_coefficients": {"Intercept": 60, "A": 2, "A^2": -6},
                "sigma": 0.7,
                "goal": "maximize",
                "n_simulations": 10,
            },
        )
        assert "error" not in result
        json.dumps(result)
        assert set(result["designs"]) == {"2^3+3c [n=11]", "BBD [n=15]"}
        assert set(result["designs"]["BBD [n=15]"]["found"]) == {"A", "I(A ** 2)"}

    def test_simulate_reports_an_error(self) -> None:
        result = execute_tool_call(
            "simulate_design_performance",
            {"designs": {"one": [{"A": -1}, {"A": 1}]}, "true_coefficients": {"B": 1}, "sigma": 1},
        )
        assert "which the designs do not have" in result["error"]

    def test_compare_returns_the_table_without_the_correlations(self) -> None:
        designs = {
            "BBD": _coded("ABC", design_type="box_behnken", n_center_points=3).to_dict(orient="records"),
            "CCD": _coded("ABC", design_type="ccd", alpha="face_centered", n_center_points=3).to_dict(orient="records"),
        }
        result = execute_tool_call("compare_designs", {"designs": designs, "model": "quadratic"})
        assert "error" not in result
        json.dumps(result)
        assert "correlations" not in result
        assert result["designs"] == ["BBD [n=15]", "CCD [n=17]"]
        assert result["table"][0] == {"Measure": "Runs", "Better": "", "BBD [n=15]": "15", "CCD [n=17]": "17"}
        with_them = execute_tool_call(
            "compare_designs", {"designs": designs, "model": "quadratic", "include_correlations": True}
        )
        assert set(with_them["correlations"]) == {"BBD [n=15]", "CCD [n=17]"}

    def test_compare_reports_an_error(self) -> None:
        result = execute_tool_call("compare_designs", {"designs": {"one": [{"A": "x"}, {"A": 1}]}})
        assert "error" in result


def test_the_true_optimum_of_many_factors_is_searched_from_a_sample() -> None:
    """Five levels of eight factors is 390625 points, more than the grid searched whole."""
    names = "ABCDEFGH"
    truth = TrueProcess({"Intercept": 0, **dict.fromkeys(names, 1.0)}, sigma=1)
    runs = _coded(names, design_type="plackett_burman", n_center_points=0)
    best = simulate_design_performance(runs, truth, goal="maximize", n_simulations=1)["true_optimum"]
    assert best["response"] == pytest.approx(8.0)
    assert np.allclose(list(best["coded"].values()), 1.0)
