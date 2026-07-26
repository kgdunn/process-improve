"""Tests for the optimize_responses() API (Tool 4)."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments._desirability import (
    composite_desirability,
    desirability_maximize,
    desirability_minimize,
    desirability_target,
    individual_desirability,
)
from process_improve.experiments.optimization import (
    _build_model_evaluator,
    _canonical_analysis,
    _extract_b_and_B,
    _find_stationary_point,
    _parse_term,
    _steepest_path,
    evaluate_model,
    optimize_responses,
)
from process_improve.tool_spec import get_tool_specs

# ---------------------------------------------------------------------------
# Fixtures - reusable model coefficient dicts
# ---------------------------------------------------------------------------


def _quadratic_2f_coeffs() -> list[dict]:
    """Two-factor quadratic: y = 40 + 5.25*A - 2*B - 3*A^2 - 1.5*B^2 + 1.5*A:B."""
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
    """Verify _parse_term classifies coefficient term names correctly."""

    def test_intercept(self) -> None:
        """Intercept returns empty tuple."""
        assert _parse_term("Intercept") == ()

    def test_linear(self) -> None:
        """Linear terms return single-element tuple."""
        assert _parse_term("A") == ("A",)
        assert _parse_term("Temperature") == ("Temperature",)

    def test_interaction(self) -> None:
        """Interaction A:B returns two-element tuple."""
        assert _parse_term("A:B") == ("A", "B")

    def test_quadratic(self) -> None:
        """Quadratic I(A ** 2) returns (A, A)."""
        assert _parse_term("I(A ** 2)") == ("A", "A")

    def test_quadratic_np_power_form(self) -> None:
        """``np.power(A, 2)`` / ``power(A, 2)`` (SEC-27 #276) parse identically to ``I(A ** 2)``.

        Newer statsmodels emits the ``np.power`` form. If the regex misses it,
        a quadratic term silently falls through to the linear branch and the
        downstream surface / optimisation produces wrong predictions.
        """
        assert _parse_term("np.power(A, 2)") == ("A", "A")
        assert _parse_term("power(A, 2)") == ("A", "A")
        # Whitespace tolerance.
        assert _parse_term("np.power(A,2)") == ("A", "A")

    def test_three_way_interaction(self) -> None:
        """Three-way interaction A:B:C returns three-element tuple."""
        assert _parse_term("A:B:C") == ("A", "B", "C")


# ---------------------------------------------------------------------------
# Model evaluator
# ---------------------------------------------------------------------------


class TestModelEvaluator:
    """Verify polynomial evaluation at known coded points."""

    def test_intercept_only(self) -> None:
        """Intercept-only model returns constant."""
        coeffs = [{"term": "Intercept", "coefficient": 42.0}]
        val = evaluate_model(coeffs, ["A"], {"A": 0.0})
        assert val == pytest.approx(42.0)

    def test_linear_model_at_origin(self) -> None:
        """Linear model at origin returns intercept."""
        coeffs = _linear_2f_coeffs()
        val = evaluate_model(coeffs, FACTOR_NAMES_2F, {"A": 0.0, "B": 0.0})
        assert val == pytest.approx(30.0)

    def test_linear_model_at_plus_one(self) -> None:
        """Linear model at (1,1): y = 30 + 4 + 3 = 37."""
        coeffs = _linear_2f_coeffs()
        val = evaluate_model(coeffs, FACTOR_NAMES_2F, {"A": 1.0, "B": 1.0})
        assert val == pytest.approx(37.0)

    def test_quadratic_model_at_origin(self) -> None:
        """Quadratic at origin returns intercept (all x=0)."""
        coeffs = _quadratic_2f_coeffs()
        val = evaluate_model(coeffs, FACTOR_NAMES_2F, {"A": 0.0, "B": 0.0})
        assert val == pytest.approx(40.0)

    def test_quadratic_model_at_corner(self) -> None:
        """Quadratic at (1,1): 40 + 5.25 - 2 - 3 - 1.5 + 1.5 = 40.25."""
        coeffs = _quadratic_2f_coeffs()
        val = evaluate_model(coeffs, FACTOR_NAMES_2F, {"A": 1.0, "B": 1.0})
        assert val == pytest.approx(40.25)

    def test_build_evaluator_returns_callable(self) -> None:
        """_build_model_evaluator returns a callable accepting numpy array."""
        f = _build_model_evaluator(_linear_2f_coeffs(), FACTOR_NAMES_2F)
        assert callable(f)
        assert f(np.array([0.0, 0.0])) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Extract b and B
# ---------------------------------------------------------------------------


class TestExtractBandB:
    """Verify extraction of intercept, linear vector b, and quadratic matrix B."""

    def test_linear_model_has_zero_b_matrix(self) -> None:
        """Linear model produces zero B matrix."""
        b0, b, b_mat = _extract_b_and_B(_linear_2f_coeffs(), FACTOR_NAMES_2F)
        assert b0 == pytest.approx(30.0)
        assert b[0] == pytest.approx(4.0)
        assert b[1] == pytest.approx(3.0)
        assert np.allclose(b_mat, 0)

    def test_quadratic_model_b_matrix_is_symmetric(self) -> None:
        """Quadratic B matrix is symmetric."""
        _b0, _b, b_mat = _extract_b_and_B(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert b_mat[0, 1] == pytest.approx(b_mat[1, 0])

    def test_quadratic_diagonals(self) -> None:
        """Diagonal entries of B match quadratic coefficients."""
        _b0, _b, b_mat = _extract_b_and_B(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert b_mat[0, 0] == pytest.approx(-3.0)
        assert b_mat[1, 1] == pytest.approx(-1.5)

    def test_interaction_split(self) -> None:
        """Interaction coeff 1.5 splits to B[0,1]=B[1,0]=0.75."""
        _b0, _b, b_mat = _extract_b_and_B(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert b_mat[0, 1] == pytest.approx(0.75)
        assert b_mat[1, 0] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Stationary point
# ---------------------------------------------------------------------------


class TestStationaryPoint:
    """Verify stationary point computation and classification."""

    def test_maximum_classification(self) -> None:
        """Quadratic model with all-negative eigenvalues classified as maximum."""
        result = _find_stationary_point(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "maximum"

    def test_saddle_classification(self) -> None:
        """Model with mixed-sign eigenvalues classified as saddle point."""
        result = _find_stationary_point(_saddle_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "saddle_point"

    def test_minimum_classification(self) -> None:
        """Model with all-positive eigenvalues classified as minimum."""
        result = _find_stationary_point(_minimum_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "minimum"

    def test_stationary_point_keys(self) -> None:
        """Result contains all expected keys."""
        result = _find_stationary_point(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert "stationary_point_coded" in result
        assert "predicted_response" in result
        assert "classification" in result
        assert "eigenvalues" in result
        assert "inside_design_space" in result

    def test_predicted_response_is_float(self) -> None:
        """Predicted response at stationary point is a Python float."""
        result = _find_stationary_point(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert isinstance(result["predicted_response"], float)

    def test_with_factor_ranges(self) -> None:
        """Factor ranges trigger coded-to-actual conversion."""
        result = _find_stationary_point(_quadratic_2f_coeffs(), FACTOR_NAMES_2F, FACTOR_RANGES_2F)
        assert "stationary_point_actual" in result
        actual = result["stationary_point_actual"]
        assert "A" in actual
        assert "B" in actual

    def test_linear_model_errors(self) -> None:
        """First-order model with no quadratic terms returns error."""
        result = _find_stationary_point(_linear_2f_coeffs(), FACTOR_NAMES_2F)
        assert "error" in result

    def test_eigenvalues_count(self) -> None:
        """Number of eigenvalues matches number of factors."""
        result = _find_stationary_point(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert len(result["eigenvalues"]) == 2


# ---------------------------------------------------------------------------
# Canonical analysis
# ---------------------------------------------------------------------------


class TestCanonicalAnalysis:
    """Verify canonical analysis eigenvalue decomposition."""

    def test_maximum_classification(self) -> None:
        """Quadratic with all-negative eigenvalues → maximum."""
        result = _canonical_analysis(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "maximum"

    def test_saddle_classification(self) -> None:
        """Mixed-sign eigenvalues → saddle point."""
        result = _canonical_analysis(_saddle_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "saddle_point"

    def test_minimum_classification(self) -> None:
        """All-positive eigenvalues → minimum."""
        result = _canonical_analysis(_minimum_2f_coeffs(), FACTOR_NAMES_2F)
        assert result["classification"] == "minimum"

    def test_eigenvalues_sorted_by_absolute(self) -> None:
        """Eigenvalues are sorted largest-absolute first."""
        result = _canonical_analysis(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        evs = result["eigenvalues"]
        assert abs(evs[0]) >= abs(evs[1])

    def test_eigenvectors_present(self) -> None:
        """Result includes eigenvector list matching factor count."""
        result = _canonical_analysis(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert "eigenvectors" in result
        assert len(result["eigenvectors"]) == 2

    def test_canonical_form_description(self) -> None:
        """Canonical form description has one entry per eigenvalue."""
        result = _canonical_analysis(_quadratic_2f_coeffs(), FACTOR_NAMES_2F)
        assert "canonical_form_description" in result
        assert len(result["canonical_form_description"]) == 2

    def test_linear_model_errors(self) -> None:
        """First-order model returns error for canonical analysis."""
        result = _canonical_analysis(_linear_2f_coeffs(), FACTOR_NAMES_2F)
        assert "error" in result


# ---------------------------------------------------------------------------
# Steepest ascent / descent
# ---------------------------------------------------------------------------


class TestSteepestPath:
    """Verify steepest ascent/descent path generation."""

    def test_ascent_direction(self) -> None:
        """For y=30+4A+3B, ascent direction is positive for both factors."""
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F, direction="ascent")
        dv = result["direction_vector"]
        assert dv["A"] > 0
        assert dv["B"] > 0

    def test_descent_direction(self) -> None:
        """Descent direction is negative for both factors."""
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F, direction="descent")
        dv = result["direction_vector"]
        assert dv["A"] < 0
        assert dv["B"] < 0

    def test_step_count(self) -> None:
        """Steps list has n_steps+1 entries (including step 0 at centre)."""
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F, n_steps=5)
        assert len(result["steps"]) == 6

    def test_first_step_is_center(self) -> None:
        """Step 0 is at the design centre (all coded values zero)."""
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F)
        step0 = result["steps"][0]
        assert step0["step"] == 0
        assert step0["coded"]["A"] == pytest.approx(0.0)
        assert step0["coded"]["B"] == pytest.approx(0.0)

    def test_predicted_response_increases_for_ascent(self) -> None:
        """Each ascent step gives a higher predicted response."""
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F, direction="ascent")
        responses = [s["predicted_response"] for s in result["steps"]]
        for i in range(1, len(responses)):
            assert responses[i] > responses[i - 1]

    def test_actual_values_with_factor_ranges(self) -> None:
        """Factor ranges trigger actual-unit conversion in step entries."""
        result = _steepest_path(
            _linear_2f_coeffs(), FACTOR_NAMES_2F, factor_ranges=FACTOR_RANGES_2F
        )
        step1 = result["steps"][1]
        assert "actual" in step1
        assert "A" in step1["actual"]
        assert "B" in step1["actual"]

    def test_zero_coefficients_error(self) -> None:
        """All-zero linear coefficients return an error."""
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
    """Verify one-sided maximise desirability function."""

    def test_below_low(self) -> None:
        """Value below low bound gives d=0."""
        assert desirability_maximize(5.0, 10.0, 20.0) == 0.0

    def test_above_high(self) -> None:
        """Value above high bound gives d=1."""
        assert desirability_maximize(25.0, 10.0, 20.0) == 1.0

    def test_at_midpoint(self) -> None:
        """Midpoint gives d=0.5 with linear weight."""
        assert desirability_maximize(15.0, 10.0, 20.0) == pytest.approx(0.5)

    def test_weight_effect(self) -> None:
        """Weight < 1 (concave) gives higher d at midpoint than linear."""
        d_linear = desirability_maximize(15.0, 10.0, 20.0, weight=1.0)
        d_concave = desirability_maximize(15.0, 10.0, 20.0, weight=0.5)
        assert d_concave > d_linear


class TestDesirabilityMinimize:
    """Verify one-sided minimise desirability function."""

    def test_below_low(self) -> None:
        """Value below low bound gives d=1."""
        assert desirability_minimize(5.0, 10.0, 20.0) == 1.0

    def test_above_high(self) -> None:
        """Value above high bound gives d=0."""
        assert desirability_minimize(25.0, 10.0, 20.0) == 0.0

    def test_at_midpoint(self) -> None:
        """Midpoint gives d=0.5 with linear weight."""
        assert desirability_minimize(15.0, 10.0, 20.0) == pytest.approx(0.5)


class TestDesirabilityTarget:
    """Verify two-sided target desirability function."""

    def test_at_target(self) -> None:
        """At target value, d=1."""
        assert desirability_target(15.0, 10.0, 15.0, 20.0) == pytest.approx(1.0)

    def test_below_low(self) -> None:
        """Below low bound, d=0."""
        assert desirability_target(5.0, 10.0, 15.0, 20.0) == 0.0

    def test_above_high(self) -> None:
        """Above high bound, d=0."""
        assert desirability_target(25.0, 10.0, 15.0, 20.0) == 0.0

    def test_between_low_and_target(self) -> None:
        """Between low and target, 0 < d < 1."""
        d = desirability_target(12.5, 10.0, 15.0, 20.0)
        assert 0.0 < d < 1.0

    def test_between_target_and_high(self) -> None:
        """Between target and high, 0 < d < 1."""
        d = desirability_target(17.5, 10.0, 15.0, 20.0)
        assert 0.0 < d < 1.0


class TestIndividualDesirability:
    """Verify individual_desirability dispatch to correct function."""

    def test_maximize_goal(self) -> None:
        """Maximize goal above high gives d=1."""
        goal = {"goal": "maximize", "low": 10.0, "high": 20.0}
        assert individual_desirability(25.0, goal) == 1.0

    def test_minimize_goal(self) -> None:
        """Minimize goal below low gives d=1."""
        goal = {"goal": "minimize", "low": 10.0, "high": 20.0}
        assert individual_desirability(5.0, goal) == 1.0

    def test_target_goal(self) -> None:
        """Target goal at target gives d=1."""
        goal = {"goal": "target", "low": 10.0, "high": 20.0, "target": 15.0}
        assert individual_desirability(15.0, goal) == pytest.approx(1.0)

    def test_unknown_goal_raises(self) -> None:
        """Unknown goal type raises ValueError."""
        goal = {"goal": "unknown", "low": 10.0, "high": 20.0}
        with pytest.raises(ValueError, match="Unknown goal"):
            individual_desirability(15.0, goal)


class TestCompositeDesirability:
    """Verify weighted geometric mean composite desirability."""

    def test_all_ones(self) -> None:
        """All d=1 gives composite D=1."""
        assert composite_desirability([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_any_zero_gives_zero(self) -> None:
        """Any d=0 makes composite D=0."""
        assert composite_desirability([1.0, 0.0, 1.0]) == 0.0

    def test_geometric_mean(self) -> None:
        """Unweighted: D = sqrt(0.5 * 0.8) = sqrt(0.4)."""
        d = composite_desirability([0.5, 0.8])
        assert d == pytest.approx(np.sqrt(0.4))

    def test_weighted(self) -> None:
        """Weighted geometric mean with importances [2, 1]."""
        d = composite_desirability([0.5, 0.8], importances=[2.0, 1.0])
        expected = np.exp((2.0 * np.log(0.5) + 1.0 * np.log(0.8)) / 3.0)
        assert d == pytest.approx(expected)

    def test_empty_list(self) -> None:
        """Empty list returns 0."""
        assert composite_desirability([]) == 0.0


# ---------------------------------------------------------------------------
# Desirability optimisation (end-to-end)
# ---------------------------------------------------------------------------


class TestOptimizeDesirability:
    """Verify end-to-end desirability optimisation via scipy."""

    def test_single_response_maximize(self) -> None:
        """Single-response maximize yields positive composite desirability."""
        model = {
            "response_name": "yield",
            "coefficients": _quadratic_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        goals = [{"response": "yield", "goal": "maximize", "low": 30.0, "high": 50.0}]
        result = optimize_responses([model], goals=goals, method="desirability")
        d_result = result["desirability"]
        assert "optimal_coded" in d_result
        assert "composite_desirability" in d_result
        assert d_result["composite_desirability"] > 0.0

    def test_two_response_desirability(self) -> None:
        """Two-response optimisation returns predictions for both responses."""
        model1 = {
            "response_name": "yield",
            "coefficients": _quadratic_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        model2 = {
            "response_name": "purity",
            "coefficients": [
                {"term": "Intercept", "coefficient": 80.0},
                {"term": "A", "coefficient": -3.0},
                {"term": "B", "coefficient": 2.0},
                {"term": "I(A ** 2)", "coefficient": -1.0},
                {"term": "I(B ** 2)", "coefficient": -2.0},
                {"term": "A:B", "coefficient": 0.5},
            ],
            "factor_names": FACTOR_NAMES_2F,
        }
        goals = [
            {"response": "yield", "goal": "maximize", "low": 30.0, "high": 50.0},
            {"response": "purity", "goal": "maximize", "low": 70.0, "high": 90.0},
        ]
        result = optimize_responses([model1, model2], goals=goals, method="desirability")
        d_result = result["desirability"]
        assert "predicted_responses" in d_result
        assert "yield" in d_result["predicted_responses"]
        assert "purity" in d_result["predicted_responses"]

    def test_with_factor_ranges(self) -> None:
        """Factor ranges produce actual-unit optimal settings."""
        model = {
            "response_name": "yield",
            "coefficients": _quadratic_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        goals = [{"response": "yield", "goal": "maximize", "low": 30.0, "high": 50.0}]
        result = optimize_responses(
            [model], goals=goals, method="desirability", factor_ranges=FACTOR_RANGES_2F
        )
        d_result = result["desirability"]
        assert "optimal_actual" in d_result

    def test_random_state_is_configurable(self) -> None:
        """SEC-33 (#282) sub-item 5: ``random_state`` is now a public kwarg.

        The previous implementation hard-coded ``np.random.default_rng(42)``
        inside the multi-start loop, which meant callers could not get
        reproducible *or* truly-random behaviour from a different seed.
        The fix moves the seed onto the public ``optimize_responses`` /
        ``_grid_search_desirability`` signature.
        """
        from process_improve.experiments.optimization import _optimize_desirability

        model = {
            "response_name": "yield",
            "coefficients": _quadratic_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        goals = [{"response": "yield", "goal": "maximize", "low": 30.0, "high": 50.0}]
        out_a = _optimize_desirability(
            [model], goals=goals, factor_names=FACTOR_NAMES_2F, random_state=1
        )
        out_b = _optimize_desirability(
            [model], goals=goals, factor_names=FACTOR_NAMES_2F, random_state=1
        )
        out_c = _optimize_desirability(
            [model], goals=goals, factor_names=FACTOR_NAMES_2F, random_state=2
        )

        # Same seed -> bit-identical optimum.
        assert out_a["optimal_coded"] == pytest.approx(out_b["optimal_coded"])
        # Different seed -> the multi-start may pick a different (still
        # optimal) point, so the *value* of the composite desirability is
        # what's reproducible; both should be high.
        assert out_a["composite_desirability"] > 0.5
        assert out_c["composite_desirability"] > 0.5


# ---------------------------------------------------------------------------
# Where the optimum actually lands
# ---------------------------------------------------------------------------


class TestDesirabilityOptimumLocation:
    """Pin down where the optimiser lands, not just that it produced a number.

    The other desirability tests assert only that keys are present and that the
    composite is above zero, which a badly wrong optimum would also satisfy.
    """

    @staticmethod
    def _plane(name: str, intercept: float, slope_a: float, slope_b: float) -> dict:
        """Build a plane in A and B, so the optimum is known without solving anything."""
        return {
            "response_name": name,
            "coefficients": [
                {"term": "Intercept", "coefficient": intercept},
                {"term": "A", "coefficient": slope_a},
                {"term": "B", "coefficient": slope_b},
            ],
            "factor_names": ["A", "B"],
        }

    def test_single_response_lands_on_the_known_corner(self) -> None:
        """Maximising a plane drives both factors to the corner that maximises it."""
        model = self._plane("y", intercept=0.0, slope_a=1.0, slope_b=-1.0)
        goals = [{"response": "y", "goal": "maximize", "low": -2.0, "high": 2.0}]
        out = optimize_responses([model], goals=goals, method="desirability")["desirability"]
        assert out["optimal_coded"]["A"] == pytest.approx(1.0, abs=1e-4)
        assert out["optimal_coded"]["B"] == pytest.approx(-1.0, abs=1e-4)
        assert out["predicted_responses"]["y"] == pytest.approx(2.0, abs=1e-4)
        assert out["composite_desirability"] == pytest.approx(1.0, abs=1e-4)

    def test_two_responses_compromise_between_their_optima(self) -> None:
        """Conflicting responses settle strictly between their individual optima.

        y1 wants A at +1, y2 wants A at -1, and both are indifferent to B. The
        compromise must therefore sit strictly inside the A range.
        """
        y1 = self._plane("y1", intercept=0.0, slope_a=1.0, slope_b=0.0)
        y2 = self._plane("y2", intercept=0.0, slope_a=-1.0, slope_b=0.0)
        goals = [
            {"response": "y1", "goal": "maximize", "low": -1.0, "high": 1.0},
            {"response": "y2", "goal": "maximize", "low": -1.0, "high": 1.0},
        ]
        out = optimize_responses([y1, y2], goals=goals, method="desirability")["desirability"]
        assert out["optimal_coded"]["A"] == pytest.approx(0.0, abs=1e-3)

    def test_importance_pulls_the_optimum_toward_the_favoured_response(self) -> None:
        """Raising one response's importance moves the compromise its way."""
        y1 = self._plane("y1", intercept=0.0, slope_a=1.0, slope_b=0.0)
        y2 = self._plane("y2", intercept=0.0, slope_a=-1.0, slope_b=0.0)
        goals = [
            {"response": "y1", "goal": "maximize", "low": -1.0, "high": 1.0},
            {"response": "y2", "goal": "maximize", "low": -1.0, "high": 1.0},
        ]
        balanced = optimize_responses([y1, y2], goals=goals, method="desirability")["desirability"]
        favoured = optimize_responses(
            [y1, y2], goals=goals, method="desirability", response_importance=[5.0, 1.0]
        )["desirability"]
        assert favoured["optimal_coded"]["A"] > balanced["optimal_coded"]["A"]


class TestGoalMatching:
    """Goals should follow their response name, not their list position."""

    @staticmethod
    def _models() -> list[dict]:
        return [
            {
                "response_name": "yield",
                "coefficients": [{"term": "Intercept", "coefficient": 0.0}, {"term": "A", "coefficient": 1.0}],
                "factor_names": ["A", "B"],
            },
            {
                "response_name": "cost",
                "coefficients": [{"term": "Intercept", "coefficient": 0.0}, {"term": "A", "coefficient": -1.0}],
                "factor_names": ["A", "B"],
            },
        ]

    def test_goal_order_does_not_change_the_answer(self) -> None:
        """Reordering goals relative to models used to silently invert the problem."""
        yield_goal = {"response": "yield", "goal": "maximize", "low": -1.0, "high": 1.0}
        cost_goal = {"response": "cost", "goal": "minimize", "low": -1.0, "high": 1.0}

        in_order = optimize_responses(
            self._models(), goals=[yield_goal, cost_goal], method="desirability"
        )["desirability"]
        reversed_order = optimize_responses(
            self._models(), goals=[cost_goal, yield_goal], method="desirability"
        )["desirability"]

        assert in_order["optimal_coded"]["A"] == pytest.approx(reversed_order["optimal_coded"]["A"], abs=1e-6)
        # Both goals push A to +1: yield rises with A, and cost falls with A.
        assert in_order["optimal_coded"]["A"] == pytest.approx(1.0, abs=1e-4)

    def test_mismatched_length_is_rejected(self) -> None:
        """One goal per model, or the pairing is undefined."""
        goals = [{"response": "yield", "goal": "maximize", "low": -1.0, "high": 1.0}]
        with pytest.raises(ValueError, match="correspond one to one"):
            optimize_responses(self._models(), goals=goals, method="desirability")

    def test_unnamed_goals_fall_back_to_position(self, caplog: pytest.LogCaptureFixture) -> None:
        """Without names on both sides, position is the only reading available."""
        goals = [
            {"goal": "maximize", "low": -1.0, "high": 1.0},
            {"goal": "minimize", "low": -1.0, "high": 1.0},
        ]
        with caplog.at_level("WARNING"):
            out = optimize_responses(self._models(), goals=goals, method="desirability")
        assert "by position" in caplog.text
        assert out["desirability"]["optimal_coded"]["A"] == pytest.approx(1.0, abs=1e-4)


class TestResponseImportanceNaming:
    """The old kwarg name said 'weights' but carried importances."""

    @staticmethod
    def _model() -> dict:
        return {
            "response_name": "y",
            "coefficients": [{"term": "Intercept", "coefficient": 0.0}, {"term": "A", "coefficient": 1.0}],
            "factor_names": ["A", "B"],
        }

    def test_deprecated_alias_still_works(self) -> None:
        """desirability_weights keeps working, with a warning."""
        goals = [{"response": "y", "goal": "maximize", "low": -1.0, "high": 1.0}]
        with pytest.warns(DeprecationWarning, match="response_importance"):
            out = optimize_responses(
                [self._model()], goals=goals, method="desirability", desirability_weights=[1.0]
            )
        assert out["desirability"]["composite_desirability"] > 0.0

    def test_both_names_together_is_an_error(self) -> None:
        """Passing both leaves the intent ambiguous."""
        goals = [{"response": "y", "goal": "maximize", "low": -1.0, "high": 1.0}]
        with pytest.raises(ValueError, match="not both"):
            optimize_responses(
                [self._model()],
                goals=goals,
                method="desirability",
                response_importance=[1.0],
                desirability_weights=[2.0],
            )

    def test_result_carries_responses_for_the_overlay_plot(self) -> None:
        """The desirability result is directly consumable by the overlay plot."""
        goals = [{"response": "y", "goal": "maximize", "low": -1.0, "high": 1.0}]
        out = optimize_responses([self._model()], goals=goals, method="desirability")["desirability"]
        assert out["responses"][0]["name"] == "y"
        assert out["responses"][0]["low"] == -1.0
        assert out["responses"][0]["high"] == 1.0
        assert out["responses"][0]["coefficients"]


class TestIntervalsAtOptimum:
    """Uncertainty at the optimum, when the fitted model objects are supplied."""

    @staticmethod
    def _fit() -> tuple[dict, object]:
        """Fit a small two-factor model on coded factors and return both forms."""
        import pandas as pd
        import statsmodels.formula.api as smf

        design = pd.DataFrame({
            "A": [-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "B": [-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        })
        design["y"] = 40.0 + 5.0 * design["A"] - 2.0 * design["B"] + [0.3, -0.2, 0.1, -0.1, 0.2, -0.3, 0.1, 0.0]
        ols_result = smf.ols("y ~ A + B", data=design).fit()
        model = {
            "response_name": "y",
            "coefficients": [
                {"term": term, "coefficient": float(value)} for term, value in ols_result.params.items()
            ],
            "factor_names": ["A", "B"],
        }
        return model, ols_result

    def test_no_intervals_without_fitted_results(self) -> None:
        """Behaviour is unchanged when the fitted objects are not supplied."""
        model, _ = self._fit()
        goals = [{"response": "y", "goal": "maximize", "low": 30.0, "high": 50.0}]
        out = optimize_responses([model], goals=goals, method="desirability")["desirability"]
        assert "response_intervals" not in out

    def test_intervals_are_reported_and_ordered(self) -> None:
        """The prediction interval contains the confidence interval, which contains the fit."""
        model, fitted = self._fit()
        goals = [{"response": "y", "goal": "maximize", "low": 30.0, "high": 50.0}]
        out = optimize_responses(
            [model], goals=goals, method="desirability", fitted_results=[fitted]
        )["desirability"]

        interval = out["response_intervals"]["y"]
        ci_low, ci_high = interval["confidence_interval"]
        pi_low, pi_high = interval["prediction_interval"]
        predicted = interval["predicted"]

        assert ci_low < predicted < ci_high
        assert pi_low < ci_low
        assert pi_high > ci_high
        assert interval["confidence_level"] == pytest.approx(0.95)
        # The optimizer and the fitted model must agree on the predicted value.
        assert predicted == pytest.approx(out["predicted_responses"]["y"], abs=1e-6)

    def test_significance_level_widens_the_interval(self) -> None:
        """A smaller alpha gives a wider interval."""
        model, fitted = self._fit()
        goals = [{"response": "y", "goal": "maximize", "low": 30.0, "high": 50.0}]

        def width(alpha: float) -> float:
            out = optimize_responses(
                [model],
                goals=goals,
                method="desirability",
                fitted_results=[fitted],
                significance_level=alpha,
            )["desirability"]
            low, high = out["response_intervals"]["y"]["confidence_interval"]
            return high - low

        assert width(0.01) > width(0.05) > width(0.20)

    def test_mismatched_fitted_results_length_is_rejected(self) -> None:
        """One fitted result per model, in the same order."""
        model, fitted = self._fit()
        goals = [{"response": "y", "goal": "maximize", "low": 30.0, "high": 50.0}]
        with pytest.raises(ValueError, match="correspond one to one"):
            optimize_responses(
                [model], goals=goals, method="desirability", fitted_results=[fitted, fitted]
            )


class TestSearchBounds:
    """The searched region defaults to the cube, but need not be the cube.

    A two-level design covers the factorial cube, so (-1, 1) is right for it.
    A central composite design reaches further: its axial runs sit at plus or
    minus alpha. Searching only the cube there refuses to consider settings the
    experiment actually covered.
    """

    @staticmethod
    def _rising_plane() -> dict:
        """Return a plane rising with A, so the optimum sits at the upper bound."""
        return {
            "response_name": "y",
            "coefficients": [{"term": "Intercept", "coefficient": 0.0}, {"term": "A", "coefficient": 1.0}],
            "factor_names": ["A", "B"],
        }

    @staticmethod
    def _goals() -> list[dict]:
        return [{"response": "y", "goal": "maximize", "low": -2.0, "high": 2.0}]

    def test_default_is_the_factorial_cube(self) -> None:
        """Unchanged behaviour when nothing is passed."""
        out = optimize_responses([self._rising_plane()], goals=self._goals(), method="desirability")
        assert out["desirability"]["optimal_coded"]["A"] == pytest.approx(1.0, abs=1e-4)

    def test_widening_the_region_moves_the_optimum_out(self) -> None:
        """A central composite design's axial reach is searchable."""
        out = optimize_responses(
            [self._rising_plane()],
            goals=self._goals(),
            method="desirability",
            search_bounds=(-1.41, 1.41),
        )
        assert out["desirability"]["optimal_coded"]["A"] == pytest.approx(1.41, abs=1e-4)

    def test_per_factor_bounds(self) -> None:
        """One factor can be widened without widening the others."""
        model = {
            "response_name": "y",
            "coefficients": [
                {"term": "Intercept", "coefficient": 0.0},
                {"term": "A", "coefficient": 1.0},
                {"term": "B", "coefficient": 1.0},
            ],
            "factor_names": ["A", "B"],
        }
        # The ramp is deliberately wider than the region can reach, so the
        # desirability never saturates and the optimum stays unique.
        goals = [{"response": "y", "goal": "maximize", "low": -5.0, "high": 5.0}]
        out = optimize_responses(
            [model], goals=goals, method="desirability", search_bounds={"A": (-1.41, 1.41)}
        )
        coded = out["desirability"]["optimal_coded"]
        assert coded["A"] == pytest.approx(1.41, abs=1e-4)
        assert coded["B"] == pytest.approx(1.0, abs=1e-4)

    def test_stationary_point_region_test_respects_the_bounds(self) -> None:
        """A point outside the cube can still be inside a composite design's region.

        The quadratic below has its maximum at A = 1.2, which is outside the
        factorial cube but well within the axial reach of a rotatable
        two-factor central composite design.
        """
        model = {
            "response_name": "y",
            "coefficients": [
                {"term": "Intercept", "coefficient": 0.0},
                {"term": "A", "coefficient": 2.4},
                {"term": "B", "coefficient": 0.0},
                {"term": "I(A ** 2)", "coefficient": -1.0},
                {"term": "I(B ** 2)", "coefficient": -1.0},
            ],
            "factor_names": ["A", "B"],
        }
        cube = optimize_responses([model], method="stationary_point")["stationary_point"]
        assert cube["stationary_point_coded"]["A"] == pytest.approx(1.2, abs=1e-6)
        assert cube["inside_design_space"] is False

        composite = optimize_responses(
            [model], method="stationary_point", search_bounds=(-1.41, 1.41)
        )["stationary_point"]
        assert composite["inside_design_space"] is True

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            ((1.0, -1.0), "low < high"),
            ((0.0, 0.0), "low < high"),
            ((float("-inf"), 1.0), "finite"),
            ((1.0,), "pair of numbers"),
            ("wide", "pair of numbers"),
        ],
    )
    def test_malformed_bounds_are_rejected(self, bad: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            optimize_responses(
                [self._rising_plane()], goals=self._goals(), method="desirability", search_bounds=bad
            )

    def test_unknown_factor_in_bounds_is_rejected(self) -> None:
        """A typo in a factor name would otherwise be silently ignored."""
        with pytest.raises(ValueError, match="unknown factor"):
            optimize_responses(
                [self._rising_plane()],
                goals=self._goals(),
                method="desirability",
                search_bounds={"Temperature": (-2.0, 2.0)},
            )


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class TestStubs:
    """Verify stub methods return appropriate not-implemented messages."""

    def test_ridge_analysis_stub(self) -> None:
        """Ridge analysis returns error with stub status."""
        model = {
            "response_name": "yield",
            "coefficients": _quadratic_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        result = optimize_responses([model], method="ridge_analysis")
        assert "error" in result["ridge_analysis"]
        assert result["ridge_analysis"]["status"] == "stub"

    def test_pareto_front_stub(self) -> None:
        """Pareto front returns error with stub status."""
        model = {
            "response_name": "yield",
            "coefficients": _quadratic_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        goals = [{"response": "yield", "goal": "maximize", "low": 30.0, "high": 50.0}]
        result = optimize_responses([model], goals=goals, method="pareto_front")
        assert "error" in result["pareto_front"]
        assert result["pareto_front"]["status"] == "stub"


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    """Verify optimize_responses routes to the correct method."""

    def test_stationary_point_via_dispatcher(self) -> None:
        """Stationary point method produces stationary_point key."""
        model = {
            "response_name": "yield",
            "coefficients": _quadratic_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        result = optimize_responses([model], method="stationary_point")
        assert result["method"] == "stationary_point"
        assert "stationary_point" in result

    def test_canonical_via_dispatcher(self) -> None:
        """Canonical analysis also includes stationary point for context."""
        model = {
            "response_name": "yield",
            "coefficients": _quadratic_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        result = optimize_responses([model], method="canonical_analysis")
        assert "canonical_analysis" in result
        assert "stationary_point" in result

    def test_steepest_ascent_via_dispatcher(self) -> None:
        """Steepest ascent produces steepest_path key."""
        model = {
            "response_name": "yield",
            "coefficients": _linear_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        result = optimize_responses([model], method="steepest_ascent", step_size=0.5, n_steps=5)
        assert "steepest_path" in result

    def test_steepest_descent_via_dispatcher(self) -> None:
        """Steepest descent sets direction to descent."""
        model = {
            "response_name": "yield",
            "coefficients": _linear_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        result = optimize_responses([model], method="steepest_descent")
        assert "steepest_path" in result
        assert result["steepest_path"]["direction"] == "descent"

    def test_unknown_method_raises(self) -> None:
        """Unknown method name raises ValueError."""
        model = {
            "response_name": "y",
            "coefficients": _linear_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        with pytest.raises(ValueError, match="Unknown method"):
            optimize_responses([model], method="bogus")

    def test_empty_models_raises(self) -> None:
        """Empty fitted_models list raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            optimize_responses([], method="stationary_point")

    def test_desirability_without_goals_raises(self) -> None:
        """Desirability method without goals raises ValueError."""
        model = {
            "response_name": "y",
            "coefficients": _linear_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        with pytest.raises(ValueError, match="Goals are required"):
            optimize_responses([model], method="desirability")

    def test_factor_names_in_result(self) -> None:
        """Result always includes factor_names."""
        model = {
            "response_name": "y",
            "coefficients": _linear_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        result = optimize_responses([model], method="steepest_ascent")
        assert result["factor_names"] == FACTOR_NAMES_2F


# ---------------------------------------------------------------------------
# Tool wrapper (JSON round-trip)
# ---------------------------------------------------------------------------


class TestToolWrapper:
    """Verify the @tool_spec wrapper for optimize_responses."""

    def test_tool_returns_dict(self) -> None:
        """Tool wrapper returns a JSON-serialisable dict."""
        from process_improve.tool_spec import execute_tool_call

        result = execute_tool_call(
            "optimize_responses",
            {
                "fitted_models": [{
                    "response_name": "yield",
                    "coefficients": _quadratic_2f_coeffs(),
                    "factor_names": FACTOR_NAMES_2F,
                }],
                "method": "stationary_point",
            },
        )
        assert isinstance(result, dict)
        assert "method" in result

    def test_tool_error_handling(self) -> None:
        """Missing required args returns error dict instead of raising."""
        from process_improve.tool_spec import execute_tool_call

        result = execute_tool_call(
            "optimize_responses",
            {
                "fitted_models": [{
                    "coefficients": _linear_2f_coeffs(),
                    "factor_names": FACTOR_NAMES_2F,
                }],
                "method": "desirability",
            },
        )
        assert "error" in result

    def test_tool_registered(self) -> None:
        """optimize_responses appears in the experiments tool registry."""
        specs = get_tool_specs(category="experiments")
        names = [s["name"] for s in specs]
        assert "optimize_responses" in names
