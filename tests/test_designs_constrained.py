"""Tests for D-optimal designs over a constrained factor region."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Constraint, Factor, evaluate_design, generate_design
from process_improve.experiments.designs_constrained import (
    MAX_CANDIDATES,
    MAX_EXPRESSION_LENGTH,
    ConstrainedOptions,
    Criterion,
    _phi_p,
    _Region,
    build_candidates,
    constrained_optimal_design,
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

    def test_grid_too_large_to_list_is_sampled(self) -> None:
        region = _Region([Factor(name=f"X{i}", low=0, high=1) for i in range(12)], [], [])
        coded, _cats, counts = build_candidates(region)
        assert counts["grid_sampled"]
        assert len(coded) <= MAX_CANDIDATES
        assert set(np.unique(coded)) == {-1.0, 0.0, 1.0}  # still points of the 3-level grid


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
        with pytest.raises(ValueError, match="No candidate point satisfies"):
            constrained_optimal_design([TEMP, DOSE], 8, [Constraint(expression="T >= 200")])

    def test_region_too_small_for_the_model(self) -> None:
        """A region that is a thin sliver along T = 100 cannot estimate a T effect."""
        with pytest.raises(ValueError, match="cannot support"):
            constrained_optimal_design(
                [TEMP, DOSE], 8, [Constraint(expression="T <= 100")], ConstrainedOptions(model_type="main_effects")
            )

    def test_mixture_factors_are_refused(self) -> None:
        mix = [Factor(name="x1", type="mixture"), Factor(name="x2", type="mixture")]
        with pytest.raises(ValueError, match="mixture"):
            constrained_optimal_design(mix, 4, [Constraint(expression="x1 <= 0.5")])


# ---------------------------------------------------------------------------
# I- and A-optimality in the candidate exchange
# ---------------------------------------------------------------------------


class TestCriteria:
    @pytest.mark.parametrize("name", ["d_optimal", "a_optimal", "i_optimal"])
    def test_swap_gains_match_recomputing_the_criterion(self, name: str) -> None:
        """The closed-form gain of each swap equals the change found by rebuilding X'X from scratch."""
        rng = np.random.default_rng(0)
        f_cand, rows = rng.normal(size=(30, 5)), rng.choice(30, 10, replace=False)
        region_rows = rng.normal(size=(200, 5))
        criterion = {"d_optimal": Criterion.d(), "a_optimal": Criterion.a(5), "i_optimal": Criterion.i(region_rows)}[
            name
        ]
        x = f_cand[rows]
        gains = criterion.swap_gains(np.linalg.inv(x.T @ x), x, f_cand)
        for i, j in [(0, 3), (4, 17), (9, 29)]:
            swapped = x.copy()
            swapped[i] = f_cand[j]
            if name == "d_optimal":  # the D gain is the determinant ratio minus one
                expected = np.linalg.det(swapped.T @ swapped) / np.linalg.det(x.T @ x) - 1
            else:
                expected = criterion.value(swapped.T @ swapped) - criterion.value(x.T @ x)
            assert gains[i, j] == pytest.approx(expected, rel=1e-9, abs=1e-12)

    def test_each_criterion_wins_on_its_own_measure(self) -> None:
        region = _Region([TEMP, DOSE], [], parse_constraint(HEAT.expression, {"T", "D"}))
        coded, cats, _ = build_candidates(region)
        f = model_matrix(region, coded, cats, "quadratic")
        no_fixed = np.empty((0, f.shape[1]))
        i_crit = Criterion.i(f)
        d_rows, _ = fedorov_exchange(f, 10, no_fixed, np.random.default_rng(0), Criterion.d())
        i_rows, _ = fedorov_exchange(f, 10, no_fixed, np.random.default_rng(0), i_crit)
        info = {k: f[r].T @ f[r] for k, r in {"d": d_rows, "i": i_rows}.items()}
        assert i_crit.value(info["i"]) >= i_crit.value(info["d"]) - 1e-9
        assert Criterion.d().value(info["d"]) >= Criterion.d().value(info["i"]) - 1e-9

    def test_i_optimal_constrained_design_has_the_better_i_efficiency(self) -> None:
        """The I-optimal design minimises the region-average variance that evaluate_design reports."""
        common = {"budget": 10, "constraints": [HEAT], "model_type": "quadratic"}
        d_design = generate_design([TEMP, DOSE], design_type="d_optimal", **common)
        i_design = generate_design([TEMP, DOSE], design_type="i_optimal", **common)
        heat = 3 * i_design.design_actual["T"] + 5 * i_design.design_actual["D"]
        assert (heat <= 600 + 1e-6).all()
        assert i_design.metadata["optimality_criterion"] == "i_optimal"
        kwargs = {"model": "quadratic", "metric": "i_efficiency", "n_samples": 20_000}
        assert evaluate_design(i_design, **kwargs)["i_efficiency"] > evaluate_design(d_design, **kwargs)["i_efficiency"]

    def test_i_optimal_mixture(self) -> None:
        mix = [
            Factor(name="x1", type="mixture", low=0.1, high=0.5),
            Factor(name="x2", type="mixture", low=0.1, high=0.7),
            Factor(name="x3", type="mixture", low=0.05, high=0.3),
        ]
        result = generate_design(mix, design_type="i_optimal", budget=10)
        assert result.metadata["method"] == "i_optimal_extreme_vertices"
        np.testing.assert_allclose(result.design_actual[["x1", "x2", "x3"]].sum(axis=1), 1.0)

    @pytest.mark.parametrize("design_type", ["d_optimal", "i_optimal", "a_optimal"])
    def test_random_seed_reproduces_the_design(self, design_type: str) -> None:
        kwargs = {"design_type": design_type, "budget": 9, "constraints": [HEAT], "random_seed": 5}
        first = generate_design([TEMP, DOSE], **kwargs)
        second = generate_design([TEMP, DOSE], **kwargs)
        pd.testing.assert_frame_equal(first.design_actual, second.design_actual)

    def test_unknown_criterion(self) -> None:
        with pytest.raises(ValueError, match="Unknown criterion"):
            constrained_optimal_design([TEMP, DOSE], 8, [], ConstrainedOptions(criterion="g_optimal"))


class TestEOptimal:
    @pytest.mark.parametrize(("phase", "measure"), [(0, "phi_p"), (1, "lambda_min")])
    def test_best_swap_gain_is_exact(self, phase: int, measure: str) -> None:
        """The swap's reported gain is the true change in phi_p (climbing) or lambda_min (polishing)."""
        rng = np.random.default_rng(3)
        f_cand, rows = rng.normal(size=(40, 6)), rng.choice(40, 12, replace=False)
        x = f_cand[rows]
        i, j, gain = Criterion.e().best_swap(x.T @ x, x, f_cand, phase)
        swapped = x.copy()
        swapped[i] = f_cand[j]

        def score(design: np.ndarray) -> float:
            eigenvalues = np.linalg.eigvalsh(design.T @ design)
            return float(_phi_p(eigenvalues)) if measure == "phi_p" else float(eigenvalues[0])

        assert gain == pytest.approx(score(swapped) - score(x))
        assert gain > 0

    def test_each_of_d_a_e_wins_on_its_own_measure(self) -> None:
        common = {"budget": 10, "constraints": [HEAT], "model_type": "quadratic"}
        metrics = ["d_efficiency", "a_optimality", "e_optimality"]
        scores = {
            name: evaluate_design(
                generate_design([TEMP, DOSE], design_type=name, **common), model="quadratic", metric=metrics
            )
            for name in ("d_optimal", "a_optimal", "e_optimal")
        }
        assert max(scores, key=lambda n: scores[n]["d_efficiency"]) == "d_optimal"
        assert min(scores, key=lambda n: scores[n]["a_optimality"]) == "a_optimal"
        assert max(scores, key=lambda n: scores[n]["e_optimality"]) == "e_optimal"

    def test_metadata_matches_evaluate_design(self) -> None:
        result = generate_design([TEMP, DOSE], design_type="e_optimal", budget=10, model_type="quadratic")
        assert result.metadata["backend"] == "candidate_exchange"  # pyoptex has no E metric
        reported = evaluate_design(result, model="quadratic", metric="e_optimality")["e_optimality"]
        assert result.metadata["min_eigenvalue"] == pytest.approx(reported)

    def test_e_optimal_mixture(self) -> None:
        mix = [Factor(name=n, type="mixture", low=0.1, high=0.6) for n in ("x1", "x2", "x3")]
        result = generate_design(mix, design_type="e_optimal", budget=9)
        assert result.metadata["method"] == "e_optimal_extreme_vertices"
        assert result.metadata["min_eigenvalue"] > 0
