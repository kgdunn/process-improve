"""Tests for D-optimal designs over a constrained factor region."""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Constraint, Factor, evaluate_design, generate_design
from process_improve.experiments.augment import augment_design
from process_improve.experiments.designs_constrained import (
    _CHUNK_ROWS,
    _GREEDY_SLACK,
    _MAX_STARTS,
    _N_STARTS,
    MAX_CANDIDATES,
    MAX_EXPRESSION_LENGTH,
    CandidatePool,
    ConstrainedOptions,
    Criterion,
    PolishLattice,
    _candidate_cap,
    _ExchangeState,
    _greedy_start,
    _n_starts,
    _phi_p,
    _quadratic_forms,
    _Region,
    build_candidates,
    constrained_optimal_design,
    fedorov_exchange,
    model_matrix,
    parse_constraint,
    polish_runs,
    select_runs,
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
        a = generate_design([TEMP, DOSE], budget=8, constraints=[HEAT], random_state=3)
        b = generate_design([TEMP, DOSE], budget=8, constraints=[HEAT], random_state=3)
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

    def test_i_optimal_constrained_design_has_the_lower_average_prediction_variance(self) -> None:
        """The I-optimal design minimises the region-average variance that evaluate_design reports."""
        common = {"budget": 10, "constraints": [HEAT], "model_type": "quadratic"}
        d_design = generate_design([TEMP, DOSE], design_type="d_optimal", **common)
        i_design = generate_design([TEMP, DOSE], design_type="i_optimal", **common)
        heat = 3 * i_design.design_actual["T"] + 5 * i_design.design_actual["D"]
        assert (heat <= 600 + 1e-6).all()
        assert i_design.metadata["optimality_criterion"] == "i_optimal"
        kwargs = {"model": "quadratic", "metric": "average_prediction_variance", "n_samples": 20_000}
        key = "average_prediction_variance"
        assert evaluate_design(i_design, **kwargs)[key] < evaluate_design(d_design, **kwargs)[key]

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
        kwargs = {"design_type": design_type, "budget": 9, "constraints": [HEAT], "random_state": 5}
        first = generate_design([TEMP, DOSE], **kwargs)
        second = generate_design([TEMP, DOSE], **kwargs)
        pd.testing.assert_frame_equal(first.design_actual, second.design_actual)

    def test_unknown_criterion(self) -> None:
        with pytest.raises(ValueError, match="Unknown criterion"):
            constrained_optimal_design([TEMP, DOSE], 8, [], ConstrainedOptions(criterion="t_optimal"))


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


# ---------------------------------------------------------------------------
# Row-wise exchange, greedy start, grid cap and polish
# ---------------------------------------------------------------------------


def _criteria(n_parameters: int, rng: np.random.Generator) -> dict[str, Criterion]:
    return {
        "d_optimal": Criterion.d(),
        "a_optimal": Criterion.a(n_parameters),
        "i_optimal": Criterion.i(rng.normal(size=(200, n_parameters))),
    }


def _cube(k: int) -> list[Factor]:
    return [Factor(name=f"X{i}", low=-1, high=1) for i in range(k)]


class TestRowExchange:
    @pytest.mark.parametrize("name", ["d_optimal", "a_optimal", "i_optimal"])
    def test_incremental_updates_match_brute_force_scoring(self, name: str) -> None:
        """After swaps made by rank-one updates, every swap scores as swap_gains does from a fresh inverse."""
        rng = np.random.default_rng(2)
        f_cand, f_fixed = rng.normal(size=(80, 6)), rng.normal(size=(3, 6))
        criterion = _criteria(6, rng)[name]
        state = _ExchangeState(f_cand, f_fixed, rng.choice(80, 12, replace=False), criterion)
        block = np.arange(12)
        terms = state.terms(block)
        for row, candidate in [(1, 40), (4, 7), (8, 63)]:
            steps = state.swap(row, candidate, terms, row)
            rest = f_cand[state.rows[block[row + 1 :]]]
            for step in steps or []:
                terms.follow(rest, step, row + 1)

        x = np.vstack([f_fixed, f_cand[state.rows]])
        expected = criterion.swap_gains(np.linalg.inv(x.T @ x), f_cand[state.rows], f_cand)
        best, gain = state.best_swaps(state.terms(block), slice(0, 12), block)
        np.testing.assert_allclose(gain, expected.max(axis=1), rtol=1e-8, atol=1e-10)
        np.testing.assert_array_equal(best, expected.argmax(axis=1))
        # The rows after the last swap followed it by rank-one updates; they match a fresh score too.
        fresh = state.terms(block)
        np.testing.assert_allclose(terms.d_ij[9:], fresh.d_ij[9:], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(state.variance, np.einsum("ij,jk,ik->i", f_cand, np.linalg.inv(x.T @ x), f_cand))

    @pytest.mark.parametrize("name", ["d_optimal", "a_optimal", "i_optimal"])
    def test_exchange_ends_where_no_single_swap_helps(self, name: str) -> None:
        """The row-wise climb stops where no swap of one run for one candidate helps, checked by brute force."""
        region = _Region([TEMP, DOSE], [], parse_constraint(HEAT.expression, {"T", "D"}))
        coded, cats, _ = build_candidates(region)
        f = model_matrix(region, coded, cats, "quadratic")
        criterion = _criteria(f.shape[1], np.random.default_rng(0))[name]
        if name == "i_optimal":
            criterion = Criterion.i(f)
        rows, _value = fedorov_exchange(f, 10, np.empty((0, f.shape[1])), np.random.default_rng(1), criterion)
        x = f[rows]
        assert criterion.swap_gains(np.linalg.inv(x.T @ x), x, f).max() <= 1e-9

    def test_greedy_start_takes_a_candidate_near_the_largest_variance_each_step(self) -> None:
        """Replayed with a fresh inverse at every step, each pick is within the slack of the largest variance."""
        f = model_matrix(
            _Region(_cube(4), [], []),
            build_candidates(_Region(_cube(4), [], []))[0],
            np.empty((625, 0), dtype=int),
            "quadratic",
        )
        f_fixed = f[:3]
        rows = _greedy_start(f, f_fixed, 30, np.random.default_rng(4))
        info = f_fixed.T @ f_fixed + 1e-6 * np.eye(f.shape[1]) + np.outer(f[rows[0]], f[rows[0]])
        for row in rows[1:]:
            variance = np.einsum("ij,jk,ik->i", f, np.linalg.inv(info), f)
            assert variance[row] >= (1 - _GREEDY_SLACK) * variance.max() * (1 - 1e-9)
            info += np.outer(f[row], f[row])

    def test_quadratic_forms_in_blocks(self) -> None:
        rng = np.random.default_rng(0)
        rows, matrix = rng.normal(size=(_CHUNK_ROWS + 123, 4)), rng.normal(size=(4, 4))
        np.testing.assert_allclose(_quadratic_forms(rows, matrix), np.einsum("ij,jk,ik->i", rows, matrix, rows))

    def test_small_problems_get_more_starts(self) -> None:
        small, large = np.zeros((100, 10)), np.zeros((50_000, 66))
        assert _n_starts(Criterion.d(), small, 12) == _MAX_STARTS
        assert _n_starts(Criterion.d(), large, 55) == _N_STARTS
        assert _n_starts(Criterion.e(), small, 12) == _N_STARTS  # E scores every swap at each step


class TestCandidateCap:
    def test_seven_factors_no_longer_list_the_5_level_grid(self) -> None:
        region = _Region(_cube(7), [], [])
        for criterion in ("d_optimal", "i_optimal"):
            coded, _cats, counts = build_candidates(region, model_type="quadratic", criterion=criterion)
            assert counts["n_levels"] == 3
            assert len(coded) <= _candidate_cap(region, "quadratic")

    def test_sample_for_d_leans_to_the_extremes(self) -> None:
        """A sampled 3-level grid for D has most points with at most one factor at its middle; for I, few do."""
        region = _Region(_cube(10), [], [])
        n_middle = {}
        for criterion in ("d_optimal", "i_optimal"):
            coded, _cats, counts = build_candidates(region, model_type="quadratic", criterion=criterion)
            assert counts["grid_sampled"]
            assert len(coded) <= _candidate_cap(region, "quadratic")
            n_middle[criterion] = np.mean((coded == 0).sum(axis=1) <= 1)
        assert n_middle["d_optimal"] > 0.5
        assert n_middle["i_optimal"] < 0.2

    def test_explicit_levels_keep_the_old_limit(self) -> None:
        coded, _cats, counts = build_candidates(_Region(_cube(7), [], []), n_levels=5)
        assert counts["n_levels"] == 5
        assert not counts["grid_sampled"]
        assert len(coded) == 5**7


class TestPolish:
    @pytest.mark.parametrize("name", ["d_optimal", "i_optimal"])
    def test_polish_only_improves_and_stays_on_the_lattice_and_in_the_region(self, name: str) -> None:
        region = _Region([TEMP, DOSE], [], parse_constraint(HEAT.expression, {"T", "D"}))
        coded, cats, _ = build_candidates(region, n_levels=3)
        f = model_matrix(region, coded, cats, "quadratic")
        criterion = Criterion.d() if name == "d_optimal" else Criterion.i(f)
        no_fixed = np.empty((0, f.shape[1]))
        rows, before = fedorov_exchange(f, 8, no_fixed, np.random.default_rng(0), criterion)
        lattice = PolishLattice(
            5,
            lambda c, k: model_matrix(region, c, k, "quadratic"),
            lambda c: region.slack(c) <= 1e-9,
        )
        polished, _cats, n_moves = polish_runs(coded[rows], cats[rows], no_fixed, criterion, lattice)
        x = model_matrix(region, polished, cats[rows], "quadratic")
        assert criterion.value(x.T @ x) >= before - 1e-12
        assert (region.slack(polished) <= 1e-9).all()
        moved = ~np.isclose(polished, coded[rows]).all(axis=1)
        assert moved.sum() <= n_moves
        on_lattice = np.isclose(polished[:, :, None], np.linspace(-1, 1, 5)).any(axis=2)
        assert on_lattice[moved].any(axis=1).all()  # a moved run has a coordinate on the 5-level lattice

    def test_select_runs_polishes_every_start_and_reports_runs_not_rows(self) -> None:
        region = _Region(_cube(3), [], [])
        coded, cats, _ = build_candidates(region, n_levels=3)
        f = model_matrix(region, coded, cats, "quadratic")
        lattice = PolishLattice(5, lambda c, k: model_matrix(region, c, k, "quadratic"))
        no_fixed = np.empty((0, f.shape[1]))
        rows, runs, _cats, value = select_runs(
            CandidatePool(coded, cats, f, lattice), 12, no_fixed, np.random.default_rng(0), Criterion.d()
        )
        _plain_rows, plain_value = fedorov_exchange(f, 12, no_fixed, np.random.default_rng(0), Criterion.d())
        assert rows is None
        assert runs.shape == (12, 3)
        assert value >= plain_value - 1e-12  # the polish keeps or improves the best start

    def test_constrained_design_records_the_polish(self) -> None:
        values, meta = constrained_optimal_design(
            _cube(6), 34, [], ConstrainedOptions(model_type="quadratic", criterion="d_optimal"), random_state=0
        )
        assert meta["n_levels"] == 3  # 5 levels would be 15,625 points, over the cap of 5,600
        assert meta["polish_levels"] == 5
        assert np.isclose(values[:, :, None], np.linspace(-1, 1, 5)).any(axis=2).all()

    def test_supplied_candidates_are_not_polished(self) -> None:
        grid = pd.DataFrame([(t, d) for t in (100, 125, 150) for d in (20, 40, 60)], columns=["T", "D"], dtype=float)
        _values, meta = constrained_optimal_design(
            [TEMP, DOSE], 8, [], ConstrainedOptions(model_type="quadratic", candidates=grid), random_state=0
        )
        assert "polish_levels" not in meta
        assert sum(meta["selected_candidates"].values()) == 8

    def test_augmentation_with_a_formula_is_polished_through_patsy(self) -> None:
        names = [f"X{i}" for i in range(6)]
        cube = np.array(list(itertools.product([-1, 1], repeat=6)), dtype=float)
        base = pd.DataFrame(cube[::4][:16], columns=names)
        formula = " + ".join(names) + " + I(X0 ** 2) + X0:X1"
        result = augment_design(base, "add_runs_optimal", n_additional_runs=6, target_model=formula, random_state=0)
        new = pd.DataFrame(result["new_runs"])[names].to_numpy()
        assert new.shape == (6, 6)
        assert np.isclose(new[:, :, None], np.linspace(-1, 1, 5)).any(axis=2).all()


# ---------------------------------------------------------------------------
# Design quality pins: the exchange may get faster, never worse
# ---------------------------------------------------------------------------


def _quality(coded: np.ndarray, factors: list[Factor], constraint: str | None = None) -> tuple[float, float]:
    """``log det(X'X) / p`` and the I-criterion ``trace((X'X)^-1 W)`` of a quadratic design in coded units.

    ``W`` is the moment matrix over a fixed uniform sample of the (constrained) region,
    drawn independently of the design's own random state.
    """
    region = _Region(factors, [], [])
    no_cats = np.empty((len(coded), 0), dtype=int)
    x = model_matrix(region, coded, no_cats, "quadratic")
    points = np.random.default_rng(12345).uniform(-1, 1, size=(200_000, len(factors)))
    if constraint is not None:
        inequalities = parse_constraint(constraint, {f.name for f in factors})
        env = region.actual(points)
        points = points[np.all([g(env) <= 0 for g in inequalities], axis=0)]
    points = points[:50_000]
    w = model_matrix(region, points, np.empty((len(points), 0), dtype=int), "quadratic")
    info = x.T @ x
    return float(np.linalg.slogdet(info)[1]) / x.shape[1], float(np.trace(np.linalg.solve(info, w.T @ w / len(w))))


class TestQualityPins:
    """D-efficiency (``log det / p``) and I trace measured on the exchange before it was made row-wise (1.97.0).

    Each value is a threshold the design must meet or beat, not an equality: a faster
    exchange may pick different runs, but not a worse design for its own criterion.
    Before the change the k = 10 augmentation took 18 s and the k = 7 I-optimal design
    36 s single-threaded; the cases still over 2 s are marked slow.
    """

    @pytest.mark.parametrize(
        ("k", "log_det_per_p"),
        [
            (6, 3.341461),
            pytest.param(8, 3.169435, marks=pytest.mark.slow),
            pytest.param(10, 3.331094, marks=pytest.mark.slow),
        ],
    )
    def test_screening_design_augmented_for_a_quadratic_model(self, k: int, log_det_per_p: float) -> None:
        """The reproducer of the slow ``add_runs_optimal`` report: 16 screening runs plus 40 for a quadratic model."""
        names = [f"X{i}" for i in range(k)]
        cube = np.array(list(itertools.product([-1, 1], repeat=k)), dtype=float)
        base = pd.DataFrame(cube[:: 2 ** (k - 4)][:16], columns=names)
        result = augment_design(
            base, "add_runs_optimal", n_additional_runs=40, target_model="quadratic", random_state=0
        )
        coded = pd.DataFrame(result["augmented_design"])[names].to_numpy(dtype=float)
        d_value, _ = _quality(coded, [Factor(name=n, low=-1, high=1) for n in names])
        assert d_value >= log_det_per_p - 1e-6

    @pytest.mark.parametrize(
        ("criterion", "k", "pinned"),
        [
            ("d_optimal", 3, 1.870472),
            ("d_optimal", 5, 2.467193),
            ("d_optimal", 7, 2.936326),
            ("i_optimal", 3, 0.346559),
            ("i_optimal", 5, 0.390857),
            pytest.param("i_optimal", 7, 0.433964, marks=pytest.mark.slow),
        ],
    )
    def test_constrained_quadratic_design(self, criterion: str, k: int, pinned: float) -> None:
        """A corner cut off the box by ``X0 + X1 <= 15``; quadratic model, six runs more than coefficients."""
        factors = [Factor(name=f"X{i}", low=0, high=10) for i in range(k)]
        constraint = "X0 + X1 <= 15"
        budget = 1 + 2 * k + k * (k - 1) // 2 + 6
        values, _meta = constrained_optimal_design(
            factors,
            budget,
            [Constraint(expression=constraint)],
            ConstrainedOptions(model_type="quadratic", criterion=criterion),
            random_state=0,
        )
        d_value, i_value = _quality(values.astype(float), factors, constraint)
        if criterion == "d_optimal":
            assert d_value >= pinned - 1e-6
        else:
            assert i_value <= pinned + 1e-6
