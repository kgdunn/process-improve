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
