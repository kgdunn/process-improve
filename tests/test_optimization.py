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
