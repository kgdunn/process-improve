"""Tests for the optimize_responses() API (Tool 4)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from process_improve.experiments._desirability import (
    composite_desirability,
    desirability_maximize,
    desirability_minimize,
    desirability_target,
    individual_desirability,
)
from process_improve.experiments.factor import Factor
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
from process_improve.experiments.region import DesignRegion
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

    @pytest.mark.parametrize("method", ["stationary_point", "canonical_analysis", "ridge_analysis"])
    def test_three_factor_term_is_refused_not_dropped(self, method: str) -> None:
        """A:B:C used to be skipped, so the 'stationary point' was not stationary for the fitted model."""
        coefficients = [
            {"term": "Intercept", "coefficient": 0.0},
            *({"term": f, "coefficient": 1.0} for f in "ABC"),
            *({"term": f"I({f} ** 2)", "coefficient": -1.0} for f in "ABC"),
            {"term": "A:B:C", "coefficient": 5.0},
        ]
        model = {"response_name": "y", "factor_names": ["A", "B", "C"], "coefficients": coefficients}
        with pytest.raises(ValueError, match=r"second-order model.*\['A:B:C'\]"):
            optimize_responses([model], method=method)

    @pytest.mark.parametrize("term", ["I(A ** 3)", "C(A)[T.1]", "Z"])
    @pytest.mark.parametrize("method", ["stationary_point", "steepest_ascent"])
    def test_unrecognised_term_names_itself(self, term: str, method: str) -> None:
        """Terms that are not products of the factors used to surface as a bare KeyError."""
        model = {"response_name": "y", "factor_names": FACTOR_NAMES_2F, "coefficients": _quadratic_2f_coeffs()}
        model["coefficients"].append({"term": term, "coefficient": 1.0})
        with pytest.raises(ValueError, match=re.escape(f"[{term!r}]")):
            optimize_responses([model], method=method)

    def test_three_factor_term_is_evaluated_by_desirability(self) -> None:
        """The general evaluator multiplies any number of factors, so desirability keeps working."""
        coefficients = [{"term": "Intercept", "coefficient": 1.0}, {"term": "A:B:C", "coefficient": 2.0}]
        assert evaluate_model(coefficients, ["A", "B", "C"], {"A": 0.5, "B": -1.0, "C": 2.0}) == pytest.approx(-1.0)


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


class TestPathInputs:
    """Inputs that used to reverse or empty a path, or ignore the bounds, without a word."""

    @staticmethod
    def _model() -> dict:
        return {"response_name": "y", "factor_names": FACTOR_NAMES_2F, "coefficients": _linear_2f_coeffs()}

    @pytest.mark.parametrize(("step_size", "n_steps"), [(-1.0, 3), (0.0, 3), (float("nan"), 3), (0.5, 0), (0.5, -3)])
    def test_steepest_path_rejects_a_non_positive_step_or_count(self, step_size: float, n_steps: int) -> None:
        """step_size=-1 walked steepest ascent downhill; n_steps=-3 returned an empty path."""
        with pytest.raises(ValueError, match=r"step_size|n_steps"):
            optimize_responses([self._model()], method="steepest_ascent", step_size=step_size, n_steps=n_steps)

    def test_step_size_is_a_euclidean_distance(self) -> None:
        """The documented convention: successive points are step_size apart."""
        steps = optimize_responses([self._model()], method="steepest_ascent", step_size=0.5, n_steps=2)
        coded = [np.array(list(s["coded"].values())) for s in steps["steepest_path"]["steps"]]
        assert np.linalg.norm(coded[2] - coded[1]) == pytest.approx(0.5)

    def test_ridge_flags_points_outside_asymmetric_bounds(self) -> None:
        """The ridge follows spheres; with A bounded to (0, 2) it reaches A = -1.96, which is now flagged."""
        model = {
            "response_name": "y",
            "factor_names": FACTOR_NAMES_2F,
            "coefficients": [
                {"term": "Intercept", "coefficient": 0.0},
                {"term": "A", "coefficient": -1.0},
                {"term": "B", "coefficient": 0.2},
                {"term": "I(A ** 2)", "coefficient": -1.0},
                {"term": "I(B ** 2)", "coefficient": -1.0},
            ],
        }
        bounds = {"A": (0.0, 2.0), "B": (-2.0, 2.0)}
        with pytest.warns(UserWarning, match="cannot follow search_bounds"):
            out = optimize_responses([model], method="ridge_analysis", search_bounds=bounds, n_steps=4)
        path = out["ridge_analysis"]["path"]
        assert path[0]["inside_search_bounds"] is True
        assert path[-1]["coded"]["A"] < 0
        assert path[-1]["inside_search_bounds"] is False

    def test_ridge_with_symmetric_bounds_does_not_warn(self) -> None:
        """The usual case, one symmetric pair for every factor, is traced as before and stays inside."""
        model = {"response_name": "y", "factor_names": FACTOR_NAMES_2F, "coefficients": _quadratic_2f_coeffs()}
        out = optimize_responses([model], method="ridge_analysis", search_bounds=(-1.41, 1.41))
        assert all(entry["inside_search_bounds"] for entry in out["ridge_analysis"]["path"])


class TestRidgeSystems:
    """A zero eigenvalue of B is a ridge, not a saddle (Myers, Montgomery and Anderson-Cook, sec. 6.4)."""

    @staticmethod
    def _model(b_a: float, b_b: float, b_bb: float) -> dict:
        """Build 10 + b_a*A + b_b*B - A^2 + b_bb*B^2, flat along B when b_bb is 0."""
        return {
            "response_name": "y",
            "factor_names": FACTOR_NAMES_2F,
            "coefficients": [
                {"term": "Intercept", "coefficient": 10.0},
                {"term": "A", "coefficient": b_a},
                {"term": "B", "coefficient": b_b},
                {"term": "I(A ** 2)", "coefficient": -1.0},
                {"term": "I(B ** 2)", "coefficient": b_bb},
            ],
        }

    @pytest.mark.parametrize("b_bb", [0.0, 1e-12])
    def test_stationary_ridge_of_maxima(self, b_bb: float) -> None:
        """The surface 10 + 2A - A^2 has a line of maxima at A = 1; it used to be called a saddle."""
        out = optimize_responses([self._model(2.0, 0.0, b_bb)], method="canonical_analysis")
        assert out["canonical_analysis"]["classification"] == "stationary_ridge"
        assert out["canonical_analysis"]["ridge_of"] == "maxima"
        assert out["canonical_analysis"]["canonical_form_description"][1].endswith("(flat)")
        point = out["stationary_point"]
        assert point["classification"] == "stationary_ridge"
        assert point["stationary_point_coded"] == pytest.approx({"A": 1.0, "B": 0.0})
        assert point["predicted_response"] == pytest.approx(11.0)

    def test_rising_ridge_has_no_stationary_point(self) -> None:
        """Adding a linear B term makes the response keep rising along the flat direction."""
        out = optimize_responses([self._model(2.0, 1.0, 0.0)], method="canonical_analysis")
        assert out["canonical_analysis"]["classification"] == "rising_ridge"
        assert out["stationary_point"]["classification"] == "rising_ridge"
        assert "No stationary point" in out["stationary_point"]["error"]

    def test_curved_surfaces_are_unchanged(self) -> None:
        """A small but real eigenvalue keeps its sign: this is a maximum, not a ridge."""
        out = optimize_responses([self._model(2.0, 0.0, -0.01)], method="stationary_point")["stationary_point"]
        assert out["classification"] == "maximum"
        assert out["stationary_point_coded"] == pytest.approx({"A": 1.0, "B": 0.0})


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
        result = _steepest_path(_linear_2f_coeffs(), FACTOR_NAMES_2F, factor_ranges=FACTOR_RANGES_2F)
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

    @pytest.mark.parametrize(
        ("goal", "match"),
        [
            ({"goal": "maximize", "low": 80.0, "high": 60.0}, "low < high"),
            ({"goal": "maximize", "low": 1.0, "high": 1.0}, "low < high"),
            ({"goal": "maximize", "low": 0.0, "high": float("inf")}, "finite"),
            ({"goal": "maximize", "high": 1.0}, "needs both 'low' and 'high'"),
            ({"goal": "target", "low": 0.0, "high": 50.0, "target": 100.0}, "strictly between"),
            ({"goal": "target", "low": 0.0, "high": 50.0, "target": 0.0}, "strictly between"),
            ({"goal": "maximize", "low": 0.0, "high": 1.0, "weight": -1.0}, "positive 'weight'"),
            ({"goal": "maximize", "low": 0.0, "high": 1.0, "weight": 0.0}, "positive 'weight'"),
            ({"goal": "target", "low": 0.0, "high": 2.0, "target": 1.0, "weight_high": -2.0}, "positive 'weight_high'"),
        ],
    )
    def test_malformed_goal_raises(self, goal: dict, match: str) -> None:
        """Derringer-Suich ramps need low < target < high and positive exponents."""
        with pytest.raises(ValueError, match=match):
            individual_desirability(0.5, goal)

    def test_error_names_the_response(self) -> None:
        """A bad goal in a multi-response problem says which response it belongs to."""
        with pytest.raises(ValueError, match="'purity'"):
            individual_desirability(0.5, {"response": "purity", "goal": "minimize", "low": 2.0, "high": 1.0})


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

    @pytest.mark.parametrize(
        ("importances", "match"),
        [
            ([2.0, -1.0], "non-negative"),
            ([0.0, 0.0], "at least one positive"),
            ([1.0], "one per response"),
        ],
    )
    def test_bad_importances_raise(self, importances: list[float], match: str) -> None:
        """A negative importance can push D above 1; a short list used to fail inside zip()."""
        with pytest.raises(ValueError, match=match):
            composite_desirability([0.5, 0.8], importances=importances)

    def test_importance_length_checked_before_optimising(self) -> None:
        """optimize_responses names the length mismatch rather than surfacing a zip() error."""
        model = {
            "response_name": "y",
            "factor_names": ["A"],
            "coefficients": [{"term": "Intercept", "coefficient": 0.0}, {"term": "A", "coefficient": 1.0}],
        }
        goals = [{"response": n, "goal": "maximize", "low": 0.0, "high": 1.0} for n in ("y", "z")]
        with pytest.raises(ValueError, match="1 importance\\(s\\) for 2 response"):
            optimize_responses(
                [model, dict(model, response_name="z")],
                goals=goals,
                method="desirability",
                response_importance=[1.0],
            )


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
        result = optimize_responses([model], goals=goals, method="desirability", factor_ranges=FACTOR_RANGES_2F)
        d_result = result["desirability"]
        assert "optimal_actual" in d_result

    def test_result_values_are_plain_python_floats(self) -> None:
        """Settings print as 189.47, not np.float64(189.47), and serialise without conversion."""
        model = {"response_name": "yield", "coefficients": _quadratic_2f_coeffs(), "factor_names": FACTOR_NAMES_2F}
        goals = [{"response": "yield", "goal": "maximize", "low": 30.0, "high": 50.0}]
        out = optimize_responses([model], goals=goals, method="desirability", factor_ranges=FACTOR_RANGES_2F)
        d_result = out["desirability"]
        for key in ("optimal_actual", "optimal_coded", "predicted_responses", "individual_desirability"):
            assert all(type(v) is float for v in d_result[key].values()), key
        assert type(d_result["composite_desirability"]) is float
        assert "np.float64" not in repr(d_result["optimal_actual"])

        stationary = optimize_responses([model], method="stationary_point", factor_ranges=FACTOR_RANGES_2F)
        assert all(type(v) is float for v in stationary["stationary_point"]["stationary_point_actual"].values())
        path = optimize_responses([model], method="steepest_ascent", factor_ranges=FACTOR_RANGES_2F)
        assert all(type(v) is float for v in path["steepest_path"]["steps"][1]["actual"].values())

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
        out_a = _optimize_desirability([model], goals=goals, factor_names=FACTOR_NAMES_2F, random_state=1)
        out_b = _optimize_desirability([model], goals=goals, factor_names=FACTOR_NAMES_2F, random_state=1)
        out_c = _optimize_desirability([model], goals=goals, factor_names=FACTOR_NAMES_2F, random_state=2)

        # Same seed -> bit-identical optimum.
        assert out_a["optimal_coded"] == pytest.approx(out_b["optimal_coded"])
        # Different seed -> the multi-start may pick a different (still
        # optimal) point, so the *value* of the composite desirability is
        # what's reproducible; both should be high.
        assert out_a["composite_desirability"] > 0.5
        assert out_c["composite_desirability"] > 0.5

    @pytest.mark.parametrize("method", ["desirability", "pareto_front"])
    def test_random_state_reaches_the_multistart(self, method: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """optimize_responses passes its random_state to the multistart search (reproducibility.rst rule 1)."""
        from process_improve.experiments import optimization

        seen: list[object] = []

        def spy(random_state: object) -> np.random.Generator:
            seen.append(random_state)
            return np.random.default_rng(0)

        monkeypatch.setattr(optimization, "check_random_state", spy)
        model = {"response_name": "y", "coefficients": _quadratic_2f_coeffs(), "factor_names": FACTOR_NAMES_2F}
        models = [model, dict(model, response_name="z")]
        goals = [{"response": n, "goal": "maximize", "low": 30.0, "high": 50.0} for n in ("y", "z")]
        rng = np.random.default_rng(7)
        optimize_responses(models, goals=goals, method=method, random_state=rng)
        assert seen == [rng]


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

    def test_tight_limits_are_found_when_every_random_start_scores_zero(self) -> None:
        """Limits [1.9, 2.0] on y = A + B: only the corner (1, 1) meets them.

        Every random start lands where d = 0 and the gradient is 0, so the search
        used to stay there and report D = 0 with optimizer_success=True.
        """
        model = self._plane("y", intercept=0.0, slope_a=1.0, slope_b=1.0)
        goals = [{"response": "y", "goal": "maximize", "low": 1.9, "high": 2.0}]
        out = optimize_responses([model], goals=goals, method="desirability")["desirability"]
        assert out["composite_desirability"] == pytest.approx(1.0, abs=1e-6)
        assert out["optimal_coded"]["A"] == pytest.approx(1.0, abs=1e-4)
        assert out["optimal_coded"]["B"] == pytest.approx(1.0, abs=1e-4)

    def test_narrow_target_window_between_two_responses_is_found(self) -> None:
        """Two responses whose acceptable windows overlap only in a thin sliver of the box."""
        y1 = self._plane("y1", intercept=0.0, slope_a=1.0, slope_b=1.0)
        y2 = self._plane("y2", intercept=0.0, slope_a=1.0, slope_b=-1.0)
        goals = [
            {"response": "y1", "goal": "target", "low": 1.2, "target": 1.3, "high": 1.4},
            {"response": "y2", "goal": "target", "low": 0.5, "target": 0.6, "high": 0.7},
        ]
        out = optimize_responses([y1, y2], goals=goals, method="desirability")["desirability"]
        assert out["composite_desirability"] == pytest.approx(1.0, abs=1e-4)
        assert out["optimal_coded"]["A"] == pytest.approx(0.95, abs=1e-3)
        assert out["optimal_coded"]["B"] == pytest.approx(0.35, abs=1e-3)

    def test_unreachable_limits_warn_and_report_the_closest_setting(self) -> None:
        """When no setting in the box gives D > 0, say so, and report the nearest miss rather than the centre."""
        model = self._plane("y", intercept=0.0, slope_a=1.0, slope_b=1.0)
        goals = [{"response": "y", "goal": "maximize", "low": 5.0, "high": 6.0}]
        with pytest.warns(UserWarning, match="composite desirability is 0") as record:
            out = optimize_responses([model], goals=goals, method="desirability")["desirability"]
        assert record[0].filename == __file__
        assert out["composite_desirability"] == 0.0
        assert out["optimal_coded"]["A"] == pytest.approx(1.0, abs=1e-4)
        assert out["optimal_coded"]["B"] == pytest.approx(1.0, abs=1e-4)

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
        favoured = optimize_responses([y1, y2], goals=goals, method="desirability", response_importance=[5.0, 1.0])[
            "desirability"
        ]
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

        in_order = optimize_responses(self._models(), goals=[yield_goal, cost_goal], method="desirability")[
            "desirability"
        ]
        reversed_order = optimize_responses(self._models(), goals=[cost_goal, yield_goal], method="desirability")[
            "desirability"
        ]

        assert in_order["optimal_coded"]["A"] == pytest.approx(reversed_order["optimal_coded"]["A"], abs=1e-6)
        # Both goals push A to +1: yield rises with A, and cost falls with A.
        assert in_order["optimal_coded"]["A"] == pytest.approx(1.0, abs=1e-4)

    def test_mismatched_length_is_rejected(self) -> None:
        """One goal per model, or the pairing is undefined."""
        goals = [{"response": "yield", "goal": "maximize", "low": -1.0, "high": 1.0}]
        with pytest.raises(ValueError, match="correspond one to one"):
            optimize_responses(self._models(), goals=goals, method="desirability")

    def test_unnamed_goals_fall_back_to_position(self) -> None:
        """Without names on both sides, position is the only reading available."""
        goals = [
            {"goal": "maximize", "low": -1.0, "high": 1.0},
            {"goal": "minimize", "low": -1.0, "high": 1.0},
        ]
        with pytest.warns(UserWarning, match="by position") as record:
            out = optimize_responses(self._models(), goals=goals, method="desirability")
        assert len([w for w in record if "by position" in str(w.message)]) == 1
        assert record[0].filename == __file__
        assert out["desirability"]["optimal_coded"]["A"] == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("method", ["desirability", "pareto_front"])
    def test_mismatched_names_raise_instead_of_pairing_by_position(self, method: str) -> None:
        """'Yield' is not 'yield': pairing by position would minimise yield and maximise cost."""
        goals = [
            {"response": "Cost", "goal": "minimize", "low": -1.0, "high": 1.0},
            {"response": "yield", "goal": "maximize", "low": -1.0, "high": 1.0},
        ]
        with pytest.raises(ValueError, match=r"\['Cost'\] match no model.*\['cost'\] match no goal"):
            optimize_responses(self._models(), goals=goals, method=method)

    def test_duplicate_goal_names_raise(self) -> None:
        """Two goals for one response leave the other response without one."""
        goals = [
            {"response": "yield", "goal": "maximize", "low": -1.0, "high": 1.0},
            {"response": "yield", "goal": "minimize", "low": -1.0, "high": 1.0},
        ]
        with pytest.raises(ValueError, match="one to one"):
            optimize_responses(self._models(), goals=goals, method="desirability")

    def test_one_unnamed_model_and_goal_pair_without_a_warning(self, recwarn: pytest.WarningsRecorder) -> None:
        """A single model and a single goal can only go together, so pairing them by position is not flagged."""
        unnamed = {key: value for key, value in self._models()[0].items() if key != "response_name"}
        goals = [{"goal": "maximize", "low": -1.0, "high": 1.0}]
        out = optimize_responses([unnamed], goals=goals, method="desirability")
        assert out["desirability"]["optimal_coded"]["A"] == pytest.approx(1.0, abs=1e-4)
        assert not [w for w in recwarn if "by position" in str(w.message)]


class TestNoConvergedStart:
    """When every SLSQP start fails, the desirability search says so instead of returning a setting."""

    @pytest.mark.parametrize(
        ("region", "message"),
        [
            pytest.param(None, "optimization produced no result", id="search-box"),
            pytest.param(
                DesignRegion([Factor(name="A", low=-1, high=1), Factor(name="B", low=-1, high=1)]),
                "no start converged inside the region",
                id="design-region",
            ),
        ],
    )
    def test_no_start_converging_raises(
        self, monkeypatch: pytest.MonkeyPatch, region: DesignRegion | None, message: str
    ) -> None:
        from process_improve.experiments import optimization

        monkeypatch.setattr(optimization, "_closest_to_specification", lambda *_args: None)
        monkeypatch.setattr(optimization, "_multistart_slsqp", lambda *_args, **_kwargs: None)
        model = {"response_name": "y", "coefficients": _quadratic_2f_coeffs(), "factor_names": FACTOR_NAMES_2F}
        goals = [{"response": "y", "goal": "maximize", "low": 30.0, "high": 45.0}]
        with pytest.raises(RuntimeError, match=f"^{message}$"):
            optimize_responses([model], goals=goals, method="desirability", region=region)


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
            out = optimize_responses([self._model()], goals=goals, method="desirability", desirability_weights=[1.0])
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

        design = pd.DataFrame(
            {
                "A": [-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "B": [-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        design["y"] = 40.0 + 5.0 * design["A"] - 2.0 * design["B"] + [0.3, -0.2, 0.1, -0.1, 0.2, -0.3, 0.1, 0.0]
        ols_result = smf.ols("y ~ A + B", data=design).fit()
        model = {
            "response_name": "y",
            "coefficients": [{"term": term, "coefficient": float(value)} for term, value in ols_result.params.items()],
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
        out = optimize_responses([model], goals=goals, method="desirability", fitted_results=[fitted])["desirability"]

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
            optimize_responses([model], goals=goals, method="desirability", fitted_results=[fitted, fitted])

    def test_a_result_that_cannot_predict_is_reported_for_its_response(self, caplog: pytest.LogCaptureFixture) -> None:
        """An object that is not a fitted model gives that response an error entry, and a logged warning."""
        model, _ = self._fit()
        goals = [{"response": "y", "goal": "maximize", "low": 30.0, "high": 50.0}]
        with caplog.at_level("WARNING", logger="process_improve.experiments.optimization"):
            out = optimize_responses([model], goals=goals, method="desirability", fitted_results=[object()])
        assert set(out["desirability"]["response_intervals"]["y"]) == {"error"}
        assert "Could not compute intervals for response 'y'" in caplog.text


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
        out = optimize_responses([model], goals=goals, method="desirability", search_bounds={"A": (-1.41, 1.41)})
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

        composite = optimize_responses([model], method="stationary_point", search_bounds=(-1.41, 1.41))[
            "stationary_point"
        ]
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
            optimize_responses([self._rising_plane()], goals=self._goals(), method="desirability", search_bounds=bad)

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


class TestFormerStubs:
    """Both methods were stubs returning ``status: "stub"`` until #208.

    The behaviour they now have is covered in
    ``tests/test_optimization_ridge_pareto.py``; these only pin that the stub
    reply is gone, so nothing downstream is still branching on it.
    """

    def test_ridge_analysis_traces_a_path(self) -> None:
        model = {
            "response_name": "yield",
            "coefficients": _quadratic_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        result = optimize_responses([model], method="ridge_analysis", n_steps=5)["ridge_analysis"]
        assert "status" not in result
        assert len(result["path"]) == 6
        assert result["path"][0]["radius"] == 0.0

    def test_pareto_front_returns_a_front(self) -> None:
        models = [
            {"response_name": "yield", "coefficients": _quadratic_2f_coeffs(), "factor_names": FACTOR_NAMES_2F},
            {"response_name": "cost", "coefficients": _linear_2f_coeffs(), "factor_names": FACTOR_NAMES_2F},
        ]
        goals = [
            {"response": "yield", "goal": "maximize", "low": 30.0, "high": 50.0},
            {"response": "cost", "goal": "minimize", "low": 10.0, "high": 40.0},
        ]
        result = optimize_responses(models, goals=goals, method="pareto_front", n_pareto_points=9)["pareto_front"]
        assert "status" not in result
        assert len(result["front"]) >= 2

    def test_pareto_front_still_refuses_a_single_response(self) -> None:
        """One response has nothing to trade off against."""
        model = {
            "response_name": "yield",
            "coefficients": _quadratic_2f_coeffs(),
            "factor_names": FACTOR_NAMES_2F,
        }
        goals = [{"response": "yield", "goal": "maximize", "low": 30.0, "high": 50.0}]
        with pytest.raises(ValueError, match="at least two responses"):
            optimize_responses([model], goals=goals, method="pareto_front")


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    """Verify optimize_responses routes to the correct method."""

    def test_a_model_that_is_not_a_dict_is_rejected(self) -> None:
        """A string where a model dict belongs is named by position and type."""
        with pytest.raises(TypeError, match=r"^fitted_models\[0\] must be a dict; got str\.$"):
            optimize_responses(["y ~ A + B"], method="stationary_point")  # type: ignore[list-item]

    def test_factors_without_a_range_stay_in_coded_units(self) -> None:
        """Only the factors given a range are converted; the others are reported as coded values."""
        model = {"response_name": "yield", "coefficients": _quadratic_2f_coeffs(), "factor_names": FACTOR_NAMES_2F}
        result = optimize_responses([model], method="stationary_point", factor_ranges={"A": {"low": 150, "high": 200}})
        coded = result["stationary_point"]["stationary_point_coded"]
        actual = result["stationary_point"]["stationary_point_actual"]
        assert actual["B"] == coded["B"]
        assert actual["A"] == pytest.approx(175.0 + 25.0 * coded["A"])

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

    @pytest.mark.parametrize(
        "method", ["stationary_point", "canonical_analysis", "steepest_ascent", "steepest_descent", "ridge_analysis"]
    )
    def test_single_response_method_refuses_several_models(self, method: str) -> None:
        """These methods analysed fitted_models[0] and silently ignored the rest."""
        models = [
            {"response_name": name, "coefficients": _quadratic_2f_coeffs(), "factor_names": FACTOR_NAMES_2F}
            for name in ("y1", "y2")
        ]
        with pytest.raises(ValueError, match=f"method='{method}' analyses one response; got 2 models"):
            optimize_responses(models, method=method)

    @pytest.mark.parametrize("missing", ["factor_names", "coefficients"])
    def test_missing_model_key_is_named(self, missing: str) -> None:
        """A model without factor_names used to fail with KeyError('factor_names')."""
        model = {"response_name": "y", "coefficients": _quadratic_2f_coeffs(), "factor_names": FACTOR_NAMES_2F}
        del model[missing]
        with pytest.raises(ValueError, match=f"fitted_models\\[0\\] has no '{missing}'"):
            optimize_responses([model], method="stationary_point")

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
                "fitted_models": [
                    {
                        "response_name": "yield",
                        "coefficients": _quadratic_2f_coeffs(),
                        "factor_names": FACTOR_NAMES_2F,
                    }
                ],
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
                "fitted_models": [
                    {
                        "coefficients": _linear_2f_coeffs(),
                        "factor_names": FACTOR_NAMES_2F,
                    }
                ],
                "method": "desirability",
            },
        )
        assert "error" in result

    def test_tool_registered(self) -> None:
        """optimize_responses appears in the experiments tool registry."""
        specs = get_tool_specs(category="experiments")
        names = [s["name"] for s in specs]
        assert "optimize_responses" in names
