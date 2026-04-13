"""Tests for the optimize_responses() API (Tool 4)."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments.optimization import (
    _build_model_evaluator,
    _canonical_analysis,
    _composite_desirability,
    _desirability_maximize,
    _desirability_minimize,
    _desirability_target,
    _extract_b_and_B,
    _find_stationary_point,
    _individual_desirability,
    _parse_term,
    _steepest_path,
    evaluate_model,
    optimize_responses,
)

# ---------------------------------------------------------------------------
# Fixtures — reusable model coefficient dicts
# ---------------------------------------------------------------------------


def _quadratic_2f_coeffs() -> list[dict]:
    """Two-factor quadratic model: y = 40 + 5.25*A - 2*B - 3*A^2 - 1.5*B^2 + 1.5*A:B.

    Stationary point is a maximum (both eigenvalues of B negative).
    """
    return [
        {"term": "Intercept", "coefficient": 40.0},
        {"term": "A", "coefficient": 5.25},
        {"term": "B", "coefficient": -2.0},
        {"term": "I(A ** 2)", "coefficient": -3.0},
        {"term": "I(B ** 2)", "coefficient": -1.5},
        {"term": "A:B", "coefficient": 1.5},
    ]


def _linear_2f_coeffs() -> list[dict]:
    """Two-factor first-order model: y = 30 + 4*A + 3*B."""
    return [
        {"term": "Intercept", "coefficient": 30.0},
        {"term": "A", "coefficient": 4.0},
        {"term": "B", "coefficient": 3.0},
    ]


def _saddle_2f_coeffs() -> list[dict]:
    """Two-factor quadratic with saddle point: y = 50 + 2*A - A^2 + 3*B^2."""
    return [
        {"term": "Intercept", "coefficient": 50.0},
        {"term": "A", "coefficient": 2.0},
        {"term": "B", "coefficient": 0.0},
        {"term": "I(A ** 2)", "coefficient": -1.0},
        {"term": "I(B ** 2)", "coefficient": 3.0},
    ]


def _minimum_2f_coeffs() -> list[dict]:
    """Two-factor quadratic with minimum: y = 10 - 1*A + 2*A^2 + 3*B^2."""
    return [
        {"term": "Intercept", "coefficient": 10.0},
        {"term": "A", "coefficient": -1.0},
        {"term": "B", "coefficient": 0.0},
        {"term": "I(A ** 2)", "coefficient": 2.0},
        {"term": "I(B ** 2)", "coefficient": 3.0},
    ]


FACTOR_NAMES_2F = ["A", "B"]
FACTOR_RANGES_2F = {"A": {"low": 150, "high": 200}, "B": {"low": 1, "high": 5}}


# ---------------------------------------------------------------------------
# Term parser
# ---------------------------------------------------------------------------


class TestParseTerm:
    def test_intercept(self) -> None:
        assert _parse_term("Intercept") == ()

    def test_linear(self) -> None:
        assert _parse_term("A") == ("A",)
        assert _parse_term("Temperature") == ("Temperature",)

    def test_interaction(self) -> None:
        assert _parse_term("A:B") == ("A", "B")

    def test_quadratic(self) -> None:
        assert _parse_term("I(A ** 2)") == ("A", "A")

    def test_three_way_interaction(self) -> None:
        assert _parse_term("A:B:C") == ("A", "B", "C")


# ---------------------------------------------------------------------------
# Model evaluator
# ---------------------------------------------------------------------------


class TestModelEvaluator:
    def test_intercept_only(self) -> None:
        coeffs = [{"term": "Intercept", "coefficient": 42.0}]
        val = evaluate_model(coeffs, ["A"], {"A": 0.0})
        assert val == pytest.approx(42.0)

    def test_linear_model_at_origin(self) -> None:
        coeffs = _linear_2f_coeffs()
        val = evaluate_model(coeffs, FACTOR_NAMES_2F, {"A": 0.0, "B": 0.0})
        assert val == pytest.approx(30.0)

    def test_linear_model_at_plus_one(self) -> None:
        coeffs = _linear_2f_coeffs()
        # y = 30 + 4*1 + 3*1 = 37
        val = evaluate_model(coeffs, FACTOR_NAMES_2F, {"A": 1.0, "B": 1.0})
        assert val == pytest.approx(37.0)

    def test_quadratic_model_at_origin(self) -> None:
        coeffs = _quadratic_2f_coeffs()
        # At origin: y = 40 (intercept only, all x=0)
        val = evaluate_model(coeffs, FACTOR_NAMES_2F, {"A": 0.0, "B": 0.0})
        assert val == pytest.approx(40.0)

    def test_quadratic_model_at_corner(self) -> None:
        coeffs = _quadratic_2f_coeffs()
        # At (1,1): y = 40 + 5.25 - 2 - 3 - 1.5 + 1.5 = 40.25
        val = evaluate_model(coeffs, FACTOR_NAMES_2F, {"A": 1.0, "B": 1.0})
        assert val == pytest.approx(40.25)

    def test_build_evaluator_returns_callable(self) -> None:
        f = _build_model_evaluator(_linear_2f_coeffs(), FACTOR_NAMES_2F)
        assert callable(f)
        assert f(np.array([0.0, 0.0])) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Extract b and B
# ---------------------------------------------------------------------------


class TestExtractBandB:
    def test_linear_model_has_zero_B(self) -> None:
        b0, b, B = _extract_b_and_B(_linear_2f_coeffs(), FACTOR_NAMES_2F)
        assert b0 == pytest.approx(30.0)
        assert b[0] == pytest.approx(4.0)
        assert b[1] == pytest.approx(3.0)
        assert np.allclose(B, 0)

    def test_quadratic_model_B_is_symmetric(self) -> None:
        _b0, _b, B = _extract_b_and_B(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert B[0, 1] == pytest.approx(B[1, 0])

    def test_quadratic_diagonals(self) -> None:
        _b0, _b, B = _extract_b_and_B(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert B[0, 0] == pytest.approx(-3.0)
        assert B[1, 1] == pytest.approx(-1.5)

    def test_interaction_split(self) -> None:
        _b0, _b, B = _extract_b_and_B(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        # Interaction coeff 1.5 is split: B[0,1] = B[1,0] = 0.75
        assert B[0, 1] == pytest.approx(0.75)
        assert B[1, 0] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Stationary point
# ---------------------------------------------------------------------------


class TestStationaryPoint:
    def test_maximum_classification(self) -> None:
        result = _find_stationary_point(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "maximum"

    def test_saddle_classification(self) -> None:
        result = _find_stationary_point(_saddle_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "saddle_point"

    def test_minimum_classification(self) -> None:
        result = _find_stationary_point(_minimum_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "minimum"

    def test_stationary_point_keys(self) -> None:
        result = _find_stationary_point(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert "stationary_point_coded" in result
        assert "predicted_response" in result
        assert "classification" in result
        assert "eigenvalues" in result
        assert "inside_design_space" in result

    def test_predicted_response_is_float(self) -> None:
        result = _find_stationary_point(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert isinstance(result["predicted_response"], float)

    def test_with_factor_ranges(self) -> None:
        result = _find_stationary_point(_quadratic_2f_coeffs(), FACTOR_NAMES_2F, FACTOR_RANGES_2F)
        assert "stationary_point_actual" in result
        actual = result["stationary_point_actual"]
        assert "A" in actual
        assert "B" in actual

    def test_linear_model_errors(self) -> None:
        result = _find_stationary_point(_linear_2f_coeffs(), FACTOR_NAMES_2F)
        assert "error" in result

    def test_eigenvalues_count(self) -> None:
        result = _find_stationary_point(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert len(result["eigenvalues"]) == 2


# ---------------------------------------------------------------------------
# Canonical analysis
# ---------------------------------------------------------------------------


class TestCanonicalAnalysis:
    def test_maximum_classification(self) -> None:
        result = _canonical_analysis(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "maximum"

    def test_saddle_classification(self) -> None:
        result = _canonical_analysis(_saddle_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "saddle_point"

    def test_minimum_classification(self) -> None:
        result = _canonical_analysis(_minimum_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "minimum"

    def test_eigenvalues_sorted_by_absolute(self) -> None:
        result = _canonical_analysis(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        evs = result["eigenvalues"]
        assert abs(evs[0]) >= abs(evs[1])

    def test_eigenvectors_present(self) -> None:
        result = _canonical_analysis(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert "eigenvectors" in result
        assert len(result["eigenvectors"]) == 2

    def test_canonical_form_description(self) -> None:
        result = _canonical_analysis(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert "canonical_form_description" in result
        assert len(result["canonical_form_description"]) == 2

    def test_linear_model_errors(self) -> None:
        result = _canonical_analysis(_linear_2f_coeffs(), FACTOR_NAMES_2F)
        assert "error" in result


# ---------------------------------------------------------------------------
# Steepest ascent / descent
# ---------------------------------------------------------------------------


class TestSteepestPath:
    def test_ascent_direction(self) -> None:
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F, direction="ascent")
        dv = result["direction_vector"]
        # For y = 30 + 4A + 3B, direction should be positive for both
        assert dv["A"] > 0
        assert dv["B"] > 0

    def test_descent_direction(self) -> None:
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F, direction="descent")
        dv = result["direction_vector"]
        assert dv["A"] < 0
        assert dv["B"] < 0

    def test_step_count(self) -> None:
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F, n_steps=5)
        # n_steps + 1 because step 0 (center) is included
        assert len(result["steps"]) == 6

    def test_first_step_is_center(self) -> None:
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F)
        step0 = result["steps"][0]
        assert step0["step"] == 0
        assert step0["coded"]["A"] == pytest.approx(0.0)
        assert step0["coded"]["B"] == pytest.approx(0.0)

    def test_predicted_response_increases_for_ascent(self) -> None:
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F, direction="ascent")
        responses = [s["predicted_response"] for s in result["steps"]]
        # Each step should give a higher predicted response
        for i in range(1, len(responses)):
            assert responses[i] > responses[i - 1]

    def test_actual_values_with_factor_ranges(self) -> None:
        result = _steepest_path(
            _linear_2f_coeffs(), FACTOR_NAMES_2F, factor_ranges=FACTOR_RANGES_2F
        )
        step1 = result["steps"][1]
        assert "actual" in step1
        assert "A" in step1["actual"]
        assert "B" in step1["actual"]

    def test_zero_coefficients_error(self) -> None:
        coeffs = [
            {"term": "Intercept", "coefficient": 10.0},
            {"term": "A", "coefficient": 0.0},
            {"term": "B", "coefficient": 0.0},
        ]
        result = _steepest_path(coeffs, FACTOR_NAMES_2F)
        assert "error" in result


# ---------------------------------------------------------------------------
# Desirability functions
# ---------------------------------------------------------------------------


class TestDesirabilityMaximize:
    def test_below_low(self) -> None:
        assert _desirability_maximize(5.0, 10.0, 20.0) == 0.0

    def test_above_high(self) -> None:
        assert _desirability_maximize(25.0, 10.0, 20.0) == 1.0

    def test_at_midpoint(self) -> None:
        assert _desirability_maximize(15.0, 10.0, 20.0) == pytest.approx(0.5)

    def test_weight_effect(self) -> None:
        d_linear = _desirability_maximize(15.0, 10.0, 20.0, weight=1.0)
        d_concave = _desirability_maximize(15.0, 10.0, 20.0, weight=0.5)
        # weight < 1 → concave → higher d at midpoint
        assert d_concave > d_linear


class TestDesirabilityMinimize:
    def test_below_low(self) -> None:
        assert _desirability_minimize(5.0, 10.0, 20.0) == 1.0

    def test_above_high(self) -> None:
        assert _desirability_minimize(25.0, 10.0, 20.0) == 0.0

    def test_at_midpoint(self) -> None:
        assert _desirability_minimize(15.0, 10.0, 20.0) == pytest.approx(0.5)


class TestDesirabilityTarget:
    def test_at_target(self) -> None:
        assert _desirability_target(15.0, 10.0, 15.0, 20.0) == pytest.approx(1.0)

    def test_below_low(self) -> None:
        assert _desirability_target(5.0, 10.0, 15.0, 20.0) == 0.0

    def test_above_high(self) -> None:
        assert _desirability_target(25.0, 10.0, 15.0, 20.0) == 0.0

    def test_between_low_and_target(self) -> None:
        d = _desirability_target(12.5, 10.0, 15.0, 20.0)
        assert 0.0 < d < 1.0

    def test_between_target_and_high(self) -> None:
        d = _desirability_target(17.5, 10.0, 15.0, 20.0)
        assert 0.0 < d < 1.0


class TestIndividualDesirability:
    def test_maximize_goal(self) -> None:
        goal = {"goal": "maximize", "low": 10.0, "high": 20.0}
        assert _individual_desirability(25.0, goal) == 1.0

    def test_minimize_goal(self) -> None:
        goal = {"goal": "minimize", "low": 10.0, "high": 20.0}
        assert _individual_desirability(5.0, goal) == 1.0

    def test_target_goal(self) -> None:
        goal = {"goal": "target", "low": 10.0, "high": 20.0, "target": 15.0}
        assert _individual_desirability(15.0, goal) == pytest.approx(1.0)

    def test_unknown_goal_raises(self) -> None:
        goal = {"goal": "unknown", "low": 10.0, "high": 20.0}
        with pytest.raises(ValueError, match="Unknown goal"):
            _individual_desirability(15.0, goal)


class TestCompositeDesirability:
    def test_all_ones(self) -> None:
        assert _composite_desirability([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_any_zero_gives_zero(self) -> None:
        assert _composite_desirability([1.0, 0.0, 1.0]) == 0.0

    def test_geometric_mean(self) -> None:
        # D = (0.5 * 0.8)^(1/2) = sqrt(0.4)
        d = _composite_desirability([0.5, 0.8])
        assert d == pytest.approx(np.sqrt(0.4))

    def test_weighted(self) -> None:
        d = _composite_desirability([0.5, 0.8], importances=[2.0, 1.0])
        expected = np.exp((2.0 * np.log(0.5) + 1.0 * np.log(0.8)) / 3.0)
        assert d == pytest.approx(expected)

    def test_empty_list(self) -> None:
        assert _composite_desirability([]) == 0.0
