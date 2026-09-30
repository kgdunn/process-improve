"""Tests for D-optimal designs over a constrained factor region."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Constraint, Factor, generate_design
from process_improve.experiments.designs_constrained import (
    MAX_EXPRESSION_LENGTH,
    _Region,
    build_candidates,
    constrained_d_optimal,
    fedorov_exchange,
    model_matrix,
    parse_constraint,
)
from process_improve.experiments.designs_optimal import _n_model_parameters

# The worked example: temperature and dosing time share a heat budget.
TEMP = Factor(name="T", low=100, high=150, units="degC")
DOSE = Factor(name="D", low=20, high=60, units="min")
HEAT = Constraint(expression="3*T + 5*D <= 600")


def _g(expression: str, **values: float) -> list[float]:
    """Evaluate each inequality of ``expression`` at one point."""
    env = {k: np.array([v]) for k, v in values.items()}
    return [float(g(env)[0]) for g in parse_constraint(expression, set(values))]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseConstraint:
    def test_less_equal_is_lhs_minus_rhs(self) -> None:
        assert _g("3*T + 5*D <= 600", T=100, D=20) == [400 - 600]

    def test_greater_equal_flips_sign(self) -> None:
        assert _g("T >= 120", T=100) == [20.0]

    def test_chained_comparison_gives_two_inequalities(self) -> None:
        assert _g("400 <= 3*T + 5*D <= 600", T=100, D=40) == [400 - 500, 500 - 600]

    def test_functions_and_powers(self) -> None:
        assert _g("sqrt(T**2 + D**2) <= 5", T=3, D=4) == pytest.approx([0.0])

    def test_unknown_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown name 'Q'"):
            parse_constraint("Q <= 1", {"T"})

    @pytest.mark.parametrize("expression", ["T == 1", "T != 1"])
    def test_equality_is_rejected(self, expression: str) -> None:
        with pytest.raises(ValueError, match="only <, <=, >, >="):
            parse_constraint(expression, {"T"})

    @pytest.mark.parametrize(
        "expression",
        ["__import__('os').system('true') <= 1", "T.real <= 1", "(lambda: 1)() <= T", "[T][0] <= 1", "True <= T"],
    )
    def test_code_outside_the_arithmetic_grammar_is_rejected(self, expression: str) -> None:
        """Only numbers, factor names, arithmetic and a few functions are evaluated."""
        with pytest.raises(ValueError, match="Unsupported syntax"):
            parse_constraint(expression, {"T"})

    def test_not_a_comparison(self) -> None:
        with pytest.raises(ValueError, match="must be an inequality"):
            parse_constraint("T + 1", {"T"})

    def test_syntax_error(self) -> None:
        with pytest.raises(ValueError, match="not a valid expression"):
            parse_constraint("T <=", {"T"})

    def test_length_limit(self) -> None:
        with pytest.raises(ValueError, match="longer than"):
            parse_constraint("T" + " + T" * MAX_EXPRESSION_LENGTH + " <= 1", {"T"})


# ---------------------------------------------------------------------------
# Candidate set and model matrix
# ---------------------------------------------------------------------------


class TestCandidates:
    def test_boundary_points_lie_on_the_constraint(self) -> None:
        """Bisection adds points on 3T + 5D = 600, and every candidate is feasible."""
        region = _Region([TEMP, DOSE], [], parse_constraint(HEAT.expression, {"T", "D"}))
        coded, _cats, counts = build_candidates(region)
        actual = region.actual(coded)
        heat = 3 * actual["T"] + 5 * actual["D"]
        assert heat.max() == pytest.approx(600)
        assert (heat <= 600 + 1e-6).all()
        assert counts["n_boundary_points"] > 0
        # The box corner where T and D are both high is cut off; the vertex on the T = 150 edge is found.
        assert np.isclose(actual["T"], 150).any()
        assert np.isclose(actual["D"][np.isclose(actual["T"], 150)], 30).any()

    @pytest.mark.parametrize("model_type", ["main_effects", "interactions", "quadratic"])
    def test_model_matrix_width_matches_parameter_count(self, model_type: str) -> None:
        cat = Factor(name="C", type="categorical", levels=["a", "b", "c"])
        region = _Region([TEMP, DOSE], [cat], [])
        coded, cats, _ = build_candidates(region, n_levels=3)
        f = model_matrix(region, coded, cats, model_type)
        assert f.shape[1] == _n_model_parameters([TEMP, DOSE, cat], model_type)
        assert np.linalg.matrix_rank(f) == f.shape[1]

    def test_grid_too_large_is_refused(self) -> None:
        region = _Region([Factor(name=f"X{i}", low=0, high=1) for i in range(12)], [], [])
        with pytest.raises(ValueError, match="exceeds"):
            build_candidates(region)


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------


def test_exchange_beats_random_subsets() -> None:
    """The Fedorov exchange finds a larger det(X'X) than 2000 random designs of the same size."""
    region = _Region([TEMP, DOSE], [], parse_constraint(HEAT.expression, {"T", "D"}))
    coded, cats, _ = build_candidates(region)
    f = model_matrix(region, coded, cats, "quadratic")
    rng = np.random.default_rng(0)
    _rows, logdet = fedorov_exchange(f, 10, np.empty((0, f.shape[1])), rng)
    random_best = max(
        np.linalg.slogdet(x.T @ x)[1] for x in (f[rng.choice(len(f), 10, replace=False)] for _ in range(2000))
    )
    assert logdet >= random_best - 1e-9


# ---------------------------------------------------------------------------
# End to end through generate_design
# ---------------------------------------------------------------------------


class TestGenerateDesign:
    def test_worked_example(self) -> None:
        result = generate_design([TEMP, DOSE], budget=10, constraints=[HEAT], model_type="quadratic")
        assert result.design_type == "d_optimal"  # chosen automatically because of the constraint
        assert result.n_runs == 10
        assert result.metadata["constraints_enforced"] is True
        heat = 3 * result.design_actual["T"] + 5 * result.design_actual["D"]
        assert (heat <= 600 + 1e-6).all()
        assert heat.max() == pytest.approx(600)

    def test_reproducible_with_seed(self) -> None:
        a = generate_design([TEMP, DOSE], budget=8, constraints=[HEAT], random_seed=3)
        b = generate_design([TEMP, DOSE], budget=8, constraints=[HEAT], random_seed=3)
        pd.testing.assert_frame_equal(a.design_actual, b.design_actual)

    def test_nonlinear_constraints_categorical_and_fixed_run(self) -> None:
        factors = [Factor(name=n, low=0, high=10) for n in "ABC"]
        factors.append(Factor(name="Cat", type="categorical", levels=["x", "y", "z"]))
        constraints = [
            Constraint(expression="A + B <= 15"),
            Constraint(expression="2 <= C - A/2"),
            Constraint(expression="A*B <= 40", type="nonlinear"),
        ]
        fixed = pd.DataFrame({"A": [0.0], "B": [0.0], "C": [0.0], "Cat": ["y"]})
        result = generate_design(
            factors, design_type="d_optimal", budget=24, model_type="quadratic", constraints=constraints,
            fixed_runs=fixed,
        )  # fmt: skip
        d = result.design_actual
        assert result.n_runs == 24
        assert result.metadata["n_fixed_runs"] == 1
        assert ((d.A + d.B <= 15 + 1e-6) & (d.C - d.A / 2 >= 2 - 1e-6) & (d.A * d.B <= 40 + 1e-6)).all()
        assert set(d.Cat) == {"x", "y", "z"}
        assert ((d.A == 5) & (d.B == 5) & (d.C == 5) & (d.Cat == "y")).any()  # the fixed run is kept

    def test_other_design_types_flag_constraints_as_not_enforced(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = generate_design([TEMP, DOSE], design_type="full_factorial", constraints=[HEAT])
        assert result.metadata["constraints_enforced"] is False
        assert "does not enforce constraints" in caplog.text

    def test_hard_to_change_is_recorded_as_ignored(self) -> None:
        result = generate_design([TEMP, DOSE], budget=8, constraints=[HEAT], hard_to_change=["T"])
        assert result.metadata["hard_to_change_ignored"] == ["T"]


class TestErrors:
    def test_conflicting_constraints(self) -> None:
        with pytest.raises(ValueError, match="No point in the factor box"):
            constrained_d_optimal([TEMP, DOSE], 8, [Constraint(expression="T >= 200")])

    def test_region_too_small_for_the_model(self) -> None:
        """A region that is a thin sliver along T = 100 cannot estimate a T effect."""
        with pytest.raises(ValueError, match="cannot support"):
            constrained_d_optimal([TEMP, DOSE], 8, [Constraint(expression="T <= 100")], model_type="main_effects")

    def test_mixture_factors_are_refused(self) -> None:
        mix = [Factor(name="x1", type="mixture"), Factor(name="x2", type="mixture")]
        with pytest.raises(ValueError, match="mixture"):
            constrained_d_optimal(mix, 4, [Constraint(expression="x1 <= 0.5")])
