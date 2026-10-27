"""Edge cases of constrained regions, their rules on categorical factors, and the optimal designs over them."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy.stats import qmc

from process_improve.experiments import Constraint, DesignRegion, Factor, generate_design
from process_improve.experiments.designs_constrained import (
    MAX_CANDIDATES,
    ConstrainedOptions,
    Criterion,
    _Region,
    build_candidates,
    constrained_optimal_design,
    fedorov_exchange,
    region_with_rules,
)
from process_improve.experiments.factor import DesignResult

TEMP = Factor(name="T", low=150, high=200)
PRESSURE = Factor(name="P", low=1, high=5)
CATALYST = Factor(name="catalyst", type="categorical", levels=["A", "B", "C"])
SOLVENT = Factor(name="solvent", type="categorical", levels=["X", "Y"])
FACTORS = [TEMP, PRESSURE, CATALYST, SOLVENT]
NO_B_WITH_Y = "not (catalyst == 'B' and solvent == 'Y')"
B_UP_TO_180 = "not (catalyst == 'B' and T > 180)"
UNIT_SQUARE = [Factor(name="A", low=-1, high=1), Factor(name="B", low=-1, high=1)]


def _eight_level_factors(n: int) -> list[Factor]:
    """Return ``n`` categorical factors of 8 levels each, which no rule names."""
    return [Factor(name=f"C{i}", type="categorical", levels=[str(j) for j in range(8)]) for i in range(n)]


def _near_diagonal_candidates(offset: float) -> pd.DataFrame:
    """Return 9 settings on the line B = A, and one setting ``offset`` off it: nearly collinear main effects."""
    grid = np.linspace(-1.0, 1.0, 9)
    candidates = pd.DataFrame({"A": grid, "B": grid})
    candidates.loc[len(candidates)] = [0.0, offset]
    return candidates


def _main_effect_rows(points: np.ndarray) -> np.ndarray:
    """Return the rows ``[1, A, B]`` of a main-effects model at coded ``points``."""
    return np.column_stack([np.ones(len(points)), points])


# ---------------------------------------------------------------------------
# Parsing rules on categorical factors
# ---------------------------------------------------------------------------


def test_rules_naming_more_than_100000_combinations_of_levels_are_refused() -> None:
    """Two factors of 400 levels give 400 * 400 = 160000 combinations, more than can be checked."""
    many = [Factor(name=name, type="categorical", levels=[str(i) for i in range(400)]) for name in ("D", "E")]
    with pytest.raises(ValueError, match=r"\['D', 'E'\] with 160000 combinations of levels, more than the 100000"):
        DesignRegion([TEMP, *many], ["not (D == '0' and E == '0')"])


def test_a_rule_of_500_characters_is_read_and_one_of_501_is_refused() -> None:
    """The length limit on an expression also holds for a rule, and 500 characters is the last length accepted."""
    head = "not (catalyst == 'B' and T > 180"
    at_limit = head + " " * (500 - len(head) - 1) + ")"
    assert len(at_limit) == 500
    region = DesignRegion(FACTORS, [at_limit])
    assert region.allowed_combinations()["limits"].tolist() == [[], ["T <= 180"], []]
    with pytest.raises(ValueError, match="Constraint expression is longer than 500 characters"):
        DesignRegion(FACTORS, [head + " " * (501 - len(head) - 1) + ")"])


@pytest.mark.parametrize("rule", ["catalyst in ()", "not (catalyst in ('A', 'B', 'C'))"])
def test_a_rule_that_allows_no_level_can_never_hold(rule: str) -> None:
    """An empty list of levels, or the negation of every level, leaves the clause nothing that can hold."""
    with pytest.raises(ValueError, match=r"can never hold: it names no level of its factors"):
        DesignRegion(FACTORS, [rule])


def test_an_and_of_more_than_64_comparisons_is_refused() -> None:
    """A chain of 65 comparisons is an AND of 65 clauses, more than the 64 allowed."""
    chain = " < ".join(["T", *(str(150 + i) for i in range(65))])  # T < 150 < 151 < ... : 65 comparisons
    with pytest.raises(ValueError, match="expands into 65 clauses, more than 64"):
        DesignRegion(FACTORS, [f"catalyst != 'C' and {chain}"])


@pytest.mark.parametrize("rule", ["catalyst in 'AB'", "catalyst == True", "catalyst == None"])
def test_a_level_must_be_a_number_or_text_in_a_list(rule: str) -> None:
    """``in`` needs a list of levels, and a level is a number or text: not a bool, not None."""
    with pytest.raises(ValueError, match=r"compare the categorical factor 'catalyst' with its levels"):
        DesignRegion(FACTORS, [rule])


def test_a_constraint_that_does_not_parse_is_only_warned_about_by_a_design_that_ignores_constraints(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The refusal of unenforced rules cannot read the expression, so it leaves the warning to say it is ignored."""
    with caplog.at_level(logging.WARNING):
        result = generate_design([TEMP, CATALYST], design_type="full_factorial", constraints=["T <="])
    assert result.metadata["constraints_enforced"] is False
    assert "does not enforce constraints" in caplog.text


# ---------------------------------------------------------------------------
# DesignRegion with and without rules
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ruled() -> DesignRegion:
    return DesignRegion(FACTORS, [NO_B_WITH_Y, B_UP_TO_180])


def test_feasible_needs_levels_for_every_factor_the_rules_name(ruled: DesignRegion) -> None:
    """``labels`` without the solvent column cannot say whether a run's combination is allowed."""
    with pytest.raises(ValueError, match=r"labels has no levels for the categorical factor\(s\) \['solvent'\]"):
        ruled.feasible(np.zeros((2, 2)), labels={"catalyst": ["A", "B"]})


def test_feasible_needs_one_level_per_point(ruled: DesignRegion) -> None:
    """Three levels for two points cannot be matched to the points."""
    with pytest.raises(ValueError, match=r"labels must give one level per point \(2\)"):
        ruled.feasible(np.zeros((2, 2)), labels={"catalyst": ["A", "B", "C"], "solvent": ["X", "X", "X"]})


def test_broken_limits_lists_a_rule_once_when_two_of_its_clauses_fail() -> None:
    """``B with Y, or B above 180`` is two clauses of one rule; a run breaking both names the rule once."""
    rule = "not (catalyst == 'B' and (solvent == 'Y' or T > 180))"
    region = DesignRegion(FACTORS, [rule])
    x = np.array([[0.8, 0.0], [0.8, 0.0], [-1.0, 0.0]])  # T = 195, 195, 150
    labels = {"catalyst": ["B", "B", "B"], "solvent": ["Y", "X", "Y"]}
    assert region.broken_limits(x, labels=labels) == [[rule], [rule], [rule]]
    assert region.feasible(x, labels=labels).tolist() == [False, False, False]


def test_a_region_without_rules_lists_no_combinations() -> None:
    """An ordinary limit is no rule: the table of allowed combinations is empty but keeps its limits column."""
    table = DesignRegion(FACTORS, ["3*T + 5*P <= 1000"]).allowed_combinations()
    assert table.empty
    assert list(table.columns) == ["limits"]


def test_given_without_rules_is_refused() -> None:
    """Without rules there is one region of settings, so there is no combination of levels to choose."""
    with pytest.raises(ValueError, match="This region has no rules on categorical factors; use it as it is"):
        DesignRegion(FACTORS).given(catalyst="A", solvent="X")


def test_given_needs_a_level_of_each_factor_the_rules_name(ruled: DesignRegion) -> None:
    """The rules name catalyst and solvent, so a combination needs both."""
    with pytest.raises(ValueError, match=r"needs a level for each factor the rules name; missing \['solvent'\]"):
        ruled.given(catalyst="A")


def test_given_refuses_a_level_the_factor_does_not_have(ruled: DesignRegion) -> None:
    """A level outside the factor's levels is named in the error, with the levels it has."""
    with pytest.raises(ValueError, match=r"'catalyst' has no level 'D'; its levels are \['A', 'B', 'C'\]"):
        ruled.given(catalyst="D", solvent="X")


def test_sample_frame_without_rules_draws_every_level_and_settings_inside_the_limit() -> None:
    """With no rules the levels are drawn independently, each equally likely, and the settings by :meth:`sample`."""
    region = DesignRegion(FACTORS, ["3*T + 5*P <= 600"])
    frame = region.sample_frame(12_000, np.random.default_rng(3))
    assert list(frame.columns) == ["T", "P", "catalyst", "solvent"]
    assert np.allclose(frame["catalyst"].value_counts(normalize=True).sort_index(), 1 / 3, atol=0.02)
    assert np.allclose(frame["solvent"].value_counts(normalize=True).sort_index(), 1 / 2, atol=0.02)
    actual_t, actual_p = 175 + 25 * frame["T"], 3 + 2 * frame["P"]
    assert (3 * actual_t + 5 * actual_p <= 600 + 1e-6).all()
    # The limit cuts 20 of the box's 200 (degC bar): the band 4 <= P <= 5 keeps 50 - 7.5 = 42.5 of the 180 left.
    assert (actual_p > 4).mean() == pytest.approx(42.5 / 180, abs=0.015)


def test_support_frame_without_rules_is_refused() -> None:
    """Without rules the support points need no levels beside them: support_points() gives them."""
    with pytest.raises(ValueError, match="no rules on categorical factors; use support_points"):
        DesignRegion(FACTORS).support_frame()


def test_support_frame_refuses_more_than_100000_rows() -> None:
    """81 grid points of a 4-factor box times 1599 allowed combinations is 129519 rows, more than 100000."""
    numeric = [Factor(name=name, low=0, high=1) for name in ("a", "b", "c", "d")]
    big = [Factor(name=name, type="categorical", levels=[str(i) for i in range(40)]) for name in ("D", "E")]
    region = DesignRegion([*numeric, *big], ["not (D == '0' and E == '0')"])
    n_points = len(region.given(D="1", E="1").support_points())
    assert n_points == 3**4
    assert n_points * (40 * 40 - 1) > 100_000
    with pytest.raises(ValueError, match="The rules give too many support points to list"):
        region.support_frame()


def test_an_infeasible_mixture_has_no_seed_points_and_cannot_be_sampled() -> None:
    """x1 >= 0.8 and x2 >= 0.8 leave no blend; the vertex search raises, so the seeds are empty."""
    region = DesignRegion([Factor(name=f"x{i}", type="mixture") for i in (1, 2, 3)], ["x1 >= 0.8", "x2 >= 0.8"])
    assert region.seed_points().shape == (0, 3)
    with pytest.raises(ValueError, match="No point inside the region was found"):
        region.sample(10, np.random.default_rng(0))


def test_a_region_of_categorical_factors_only_has_an_empty_centroid() -> None:
    """With no numeric factor there is nothing to set, and every (empty) setting is allowed."""
    assert DesignRegion([CATALYST, SOLVENT]).centroid() == {}


# ---------------------------------------------------------------------------
# Uniform sampling of thin regions
# ---------------------------------------------------------------------------


def test_a_region_of_one_point_is_sampled_at_that_point() -> None:
    """A >= 1 and B >= 1 leave the corner (1, 1) alone: the chains have no spread to shape their directions by."""
    region = DesignRegion(UNIT_SQUARE, ["A >= 1", "B >= 1"])
    x = region.sample(5, np.random.default_rng(0))
    # To rounding, not exactly: on macOS some coordinates land one unit in the last place below 1.
    np.testing.assert_allclose(x, np.ones((5, 2)), rtol=0, atol=1e-12)
    assert region.feasible(x).all()


def test_a_curved_limit_met_on_a_line_only_keeps_the_chains_on_it() -> None:
    """``abs(A)**0.01 <= 0`` holds only at A = 0 exactly; every step off that line misses, so the chains stay."""
    region = DesignRegion(UNIT_SQUARE, [Constraint(expression="abs(A)**0.01 <= 0", type="nonlinear")])
    x = region.sample(10, np.random.default_rng(0))
    assert x.shape == (10, 2)
    np.testing.assert_array_equal(x[:, 0], 0.0)
    assert region.feasible(x).all()
    assert np.ptp(x[:, 1]) > 0.1  # the chains start at different points along the line


# ---------------------------------------------------------------------------
# The candidate region of the design engine
# ---------------------------------------------------------------------------


def test_a_region_without_rules_draws_each_combination_of_levels_equally_often() -> None:
    """Without rules every combination is allowed, none has a limit, and each of the 6 is drawn a sixth of the time."""
    region = region_with_rules(FACTORS, [])
    cats = region.draw_levels(24_000, np.random.default_rng(0))
    assert region.labels(cats) == {}
    assert region.allowed(cats).all()
    assert region.applies(cats).shape == (24_000, 0)
    shares = pd.Series(list(map(tuple, cats))).value_counts(normalize=True)
    assert len(shares) == 3 * 2
    assert np.allclose(shares, 1 / 6, atol=0.01)


def test_a_factor_the_rules_do_not_name_keeps_its_levels_equally_likely() -> None:
    """A limit on catalyst C leaves the solvent free: each solvent is drawn half the time, whatever the catalyst."""
    region = region_with_rules(FACTORS, [Constraint(expression="catalyst != 'C' or T <= 170")])
    assert region.rules is not None
    assert region.rules.factors == ["catalyst"]
    cats = region.draw_levels(24_000, np.random.default_rng(1))
    shares = pd.Series(list(map(tuple, cats))).value_counts(normalize=True)
    assert len(shares) == 3 * 2
    assert np.allclose(shares, 1 / 6, atol=0.01)


def test_an_i_optimal_design_runs_both_levels_of_a_factor_the_rules_do_not_name() -> None:
    """The moment matrix draws the solvent freely beside the catalyst the rule names, and the design runs both."""
    result = generate_design(
        [TEMP, PRESSURE, Factor(name="catalyst", type="categorical", levels=["A", "B"]), SOLVENT],
        design_type="i_optimal",
        budget=12,
        constraints=[B_UP_TO_180],
        random_state=0,
    )
    design = result.design_actual
    assert set(design["solvent"]) == {"X", "Y"}
    assert (design.loc[design["catalyst"] == "B", "T"] <= 180 + 1e-9).all()


def test_fewer_than_two_grid_levels_are_refused() -> None:
    """A grid of one level per factor cannot hold a run at both ends of a range."""
    with pytest.raises(ValueError, match="n_levels must be at least 2"):
        constrained_optimal_design([TEMP, PRESSURE], 6, [], ConstrainedOptions(n_levels=1))


def test_a_sampled_grid_on_an_even_number_of_levels_adds_no_centre_points() -> None:
    """4 levels on 9 factors is 262144 points: the grid is the 2**15 Sobol points rounded down onto it, and no more."""
    factors = [Factor(name=f"X{i}", low=-1, high=1) for i in range(9)]
    coded, _cats, counts = build_candidates(region_with_rules(factors, []), n_levels=4, model_type="main_effects")
    unit = qmc.Sobol(d=9, scramble=False).random_base2(15)
    cells = np.unique(np.minimum((unit * 4).astype(int), 3), axis=0)
    assert counts["grid_sampled"]
    assert counts["n_grid_points"] == len(cells)
    np.testing.assert_allclose(np.unique(coded.round(12)), [-1.0, -1 / 3, 1 / 3, 1.0])


def _limited_on_b(n_eight_level: int) -> _Region:
    """Return the engine's region of T, P, catalyst and ``n_eight_level`` other factors, T <= 180 under catalyst B."""
    factors = [TEMP, PRESSURE, CATALYST, *_eight_level_factors(n_eight_level)]
    return region_with_rules(factors, [Constraint(expression=B_UP_TO_180)])


def test_vertices_are_left_out_when_the_combinations_of_levels_are_too_many() -> None:
    """With a limit on catalyst B, 3 * 8**6 = 786432 combinations of levels is more than the candidate set takes."""
    assert MAX_CANDIDATES < 3 * 8**6
    _coded, _cats, counts = build_candidates(_limited_on_b(6), model_type="main_effects")
    assert counts["n_vertex_points"] == 0


def test_a_group_of_combinations_whose_vertices_are_too_many_is_left_out() -> None:
    """Under T <= 180 the box has 4 vertices; paired with the 8**5 combinations that have catalyst B, too many."""
    _coded, _cats, counts = build_candidates(_limited_on_b(1), model_type="main_effects")
    assert counts["n_vertex_points"] == 4 * 8  # (-1, -1), (-1, 1), (0.2, -1), (0.2, 1) with catalyst B
    assert 4 * 8**5 > MAX_CANDIDATES >= 3 * 8**5
    _coded, _cats, counts = build_candidates(_limited_on_b(5), model_type="main_effects")
    assert counts["n_vertex_points"] == 0


def test_vertices_shared_by_too_many_combinations_of_levels_are_left_out() -> None:
    """Without rules each vertex is paired with every combination: 4 vertices * 8**5 combinations is too many."""
    limit = [Constraint(expression="T + 10*P <= 210")]
    _coded, _cats, counts = build_candidates(
        region_with_rules([TEMP, PRESSURE, *_eight_level_factors(1)], limit), model_type="main_effects"
    )
    assert counts["n_vertex_points"] == 4 * 8
    _coded, _cats, counts = build_candidates(
        region_with_rules([TEMP, PRESSURE, *_eight_level_factors(5)], limit), model_type="main_effects"
    )
    assert counts["n_vertex_points"] == 0


# ---------------------------------------------------------------------------
# Optimality criteria and the exchanges
# ---------------------------------------------------------------------------


def _swapped(info: np.ndarray, removed: np.ndarray, added: np.ndarray) -> np.ndarray:
    """Return the information after the model row ``removed`` is swapped for ``added``."""
    return info - np.outer(removed, removed) + np.outer(added, added)


#: Five settings on a line, for the model [1, x]: the information is singular unless two distinct ones are run.
LINE = np.column_stack([np.ones(5), np.linspace(-1.0, 1.0, 5)])
#: The same rows with the x column repeated: no design of them can estimate three coefficients.
LINE_REPEATED = np.column_stack([LINE, LINE[:, 1]])


@pytest.mark.parametrize("criterion", [Criterion.a(3), Criterion.g(LINE_REPEATED), Criterion.k()], ids=["A", "G", "K"])
def test_a_singular_design_scores_minus_infinity(criterion: Criterion) -> None:
    """Candidates of rank 2 cannot estimate 3 coefficients; the exchange still returns a design, scored -inf."""
    assert np.linalg.matrix_rank(LINE_REPEATED) == 2
    rows, value = fedorov_exchange(LINE_REPEATED, 4, np.empty((0, 3)), np.random.default_rng(0), criterion)
    assert len(rows) == 4
    assert value == -np.inf


def test_g_optimality_needs_the_region_rows() -> None:
    """A G criterion built without the region's model rows has no points to take the largest variance over."""
    with pytest.raises(ValueError, match="G-optimality needs the region's model rows"):
        fedorov_exchange(LINE, 2, np.empty((0, 2)), np.random.default_rng(0), Criterion("g_optimal"))


def test_a_saturated_g_optimal_design_takes_the_two_ends() -> None:
    """With 2 runs for 2 coefficients no run can be dropped; the ends give variance (1 + x**2) / 2, at most 1."""
    rows, value = fedorov_exchange(LINE, 2, np.empty((0, 2)), np.random.default_rng(0), Criterion.g(LINE))
    assert sorted(LINE[rows, 1]) == [-1.0, 1.0]
    x = LINE[rows]
    variance = np.einsum("ij,jk,ik->i", LINE, np.linalg.inv(x.T @ x), LINE)
    np.testing.assert_allclose(variance, (1 + LINE[:, 1] ** 2) / 2)
    assert value == pytest.approx(-1.0)


def test_the_g_polish_stops_after_20_swaps_and_reports_the_design_it_keeps() -> None:
    """A design the I-lambda rounds leave 21 swaps from the polish's optimum is polished for 20 of them."""
    rng = np.random.default_rng(35)  # 18 random settings and 134 region points for a quadratic in one factor
    rng.integers(15, 51)
    settings = np.sort(rng.uniform(-1.0, 1.0, 18))
    rng.integers(50, 300)
    points = rng.uniform(-1.0, 1.0, 134)
    candidates = np.column_stack([np.ones(18), settings, settings**2])
    region = np.column_stack([np.ones(134), points, points**2])
    rows, value = fedorov_exchange(candidates, 65, np.empty((0, 3)), np.random.default_rng(0), Criterion.g(region))
    x = candidates[rows]
    variance = np.einsum("ij,jk,ik->i", region, np.linalg.inv(x.T @ x), region)
    assert len(rows) == 65
    assert value == pytest.approx(-variance.max(), rel=1e-9)


def test_the_best_d_swap_is_the_one_that_raises_the_determinant_most() -> None:
    """The gain is det(M - a a' + b b') / det(M) - 1, checked against every swap by numpy's determinant."""
    rng = np.random.default_rng(4)
    f_cand = rng.standard_normal((12, 3))
    f_design = f_cand[[0, 1, 2, 3, 4]]
    info = f_design.T @ f_design
    i, j, gain = Criterion.d().best_swap(info, f_design, f_cand)
    ratios = np.array(
        [[np.linalg.det(_swapped(info, a, b)) / np.linalg.det(info) - 1 for b in f_cand] for a in f_design]
    )
    assert gain == pytest.approx(ratios.max(), rel=1e-9)
    assert (i, j) == np.unravel_index(np.argmax(ratios), ratios.shape)


def test_the_best_a_swap_is_the_one_that_lowers_the_trace_most() -> None:
    """The gain is trace(M^-1) - trace((M - a a' + b b')^-1), checked against every swap by numpy's inverse."""
    rng = np.random.default_rng(5)
    f_cand = rng.standard_normal((12, 3))
    f_design = f_cand[[0, 1, 2, 3, 4]]
    info = f_design.T @ f_design
    i, j, gain = Criterion.a(3).best_swap(info, f_design, f_cand)
    trace = np.trace(np.linalg.inv(info))
    falls = np.array([[trace - np.trace(np.linalg.inv(_swapped(info, a, b))) for b in f_cand] for a in f_design])
    assert gain == pytest.approx(falls.max(), rel=1e-9)
    assert (i, j) == np.unravel_index(np.argmax(falls), falls.shape)


def test_the_e_polish_takes_no_swap_when_every_swap_lowers_the_smallest_eigenvalue() -> None:
    """Swapping either unit run for (0.1, 0.1) drops lambda_min of I from 1 to below 0.01, so no swap is taken."""
    info, f_design, f_cand = np.eye(2), np.eye(2), np.array([[0.1, 0.1]])
    smallest = [np.linalg.eigvalsh(_swapped(info, a, f_cand[0]))[0] for a in f_design]
    assert max(smallest) < 0.01
    assert Criterion.e().best_swap(info, f_design, f_cand, phase=1) == (0, 0, 0.0)


# ---------------------------------------------------------------------------
# Optimal designs from nearly collinear candidates
# ---------------------------------------------------------------------------


def _best_by_brute_force(
    candidates: pd.DataFrame, n_runs: int, score: Callable[[np.ndarray], float]
) -> tuple[float, tuple[int, ...]]:
    """Return the lowest ``score(X'X)`` over every multiset of ``n_runs`` candidates, with its rows."""
    rows = _main_effect_rows(candidates.to_numpy(dtype=float))
    best = (np.inf, ())
    for combo in itertools.combinations_with_replacement(range(len(rows)), n_runs):
        x = rows[list(combo)]
        info = x.T @ x
        if np.linalg.matrix_rank(info) == rows.shape[1]:
            best = min(best, (float(score(info)), combo))
    return best


def _chosen(result: DesignResult) -> np.ndarray:
    """Return the coded settings of a design's runs."""
    return result.design[["A", "B"]].to_numpy(dtype=float)


@pytest.mark.slow  # 50 starts of the exchange, some cycling for 100 passes on rounding noise: about 2 s
def test_an_a_optimal_design_from_nearly_collinear_candidates_has_the_least_trace() -> None:
    """The setting 1e-4 off the line B = A carries the B effect; the trace is the brute-force least, to rounding."""
    # The best designs differ in the ninth figure, below the rounding of a trace near 1.7e8: only the value is pinned.
    candidates = _near_diagonal_candidates(1e-4)
    result = generate_design(
        UNIT_SQUARE, design_type="a_optimal", budget=5, candidates=candidates, model_type="main_effects", random_state=0
    )
    x = _main_effect_rows(_chosen(result))
    trace = float(np.trace(np.linalg.inv(x.T @ x)))
    assert result.metadata["trace_criterion"] == pytest.approx(trace, rel=1e-6)
    best, _rows = _best_by_brute_force(candidates, 5, lambda info: np.trace(np.linalg.inv(info)))
    assert trace == pytest.approx(best, rel=1e-6)
    assert (_chosen(result)[:, 1] == 1e-4).any()  # without it, B - A is not estimable


def test_a_k_optimal_design_from_nearly_collinear_candidates_has_the_least_condition_number() -> None:
    """The condition number is about 8.3e8 whatever is run; the design's is the brute-force minimum's, to rounding."""
    candidates = _near_diagonal_candidates(1e-4)
    result = generate_design(
        UNIT_SQUARE, design_type="k_optimal", budget=5, candidates=candidates, model_type="main_effects", random_state=0
    )
    x = _main_effect_rows(_chosen(result))
    eigenvalues = np.linalg.eigvalsh(x.T @ x)
    assert result.metadata["condition_number"] == pytest.approx(eigenvalues[-1] / eigenvalues[0], rel=1e-6)
    best, _rows = _best_by_brute_force(candidates, 5, np.linalg.cond)
    assert eigenvalues[-1] / eigenvalues[0] == pytest.approx(best, rel=1e-6)
    assert (_chosen(result)[:, 1] == 1e-4).any()


def test_candidates_too_nearly_collinear_for_any_design_are_refused() -> None:
    """Off the line by 1e-8 the candidates have full rank, but no design the exchange can find estimates B."""
    candidates = _near_diagonal_candidates(1e-8)
    assert np.linalg.matrix_rank(_main_effect_rows(candidates.to_numpy(dtype=float))) == 3
    with pytest.raises(
        ValueError, match=r"No design of 5 runs from these candidates can estimate the 'main_effects' model \(3"
    ):
        generate_design(
            UNIT_SQUARE,
            design_type="d_optimal",
            budget=5,
            candidates=candidates,
            model_type="main_effects",
            random_state=0,
        )


def test_categorical_factors_tied_by_the_rules_lose_their_aliased_columns(caplog: pytest.LogCaptureFixture) -> None:
    """Allowing only (A, X) and (B, Y) makes the solvent column the catalyst column: 3 of 8 quadratic columns go."""
    # The solvent main effect and T:solvent repeat catalyst and T:catalyst, and catalyst:solvent is constant.
    catalyst = Factor(name="catalyst", type="categorical", levels=["A", "B"])
    rules = ["not (catalyst == 'A' and solvent == 'Y')", "not (catalyst == 'B' and solvent == 'X')"]
    with caplog.at_level(logging.WARNING):
        result = generate_design(
            [TEMP, catalyst, SOLVENT],
            design_type="d_optimal",
            budget=10,
            model_type="quadratic",
            constraints=rules,
            random_state=0,
        )
    assert result.metadata["inestimable_columns"] == {"solvent": 1, "T:solvent": 1, "catalyst:solvent": 1}
    # Independently: [1, t, c, s, t c, t s, c s, t^2] over t in {-1, 0, 1} with s = c = +/-1 has rank 5 of 8.
    t, c = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 1.0])
    t, c = t.ravel(), c.ravel()
    columns = np.column_stack([np.ones_like(t), t, c, c, t * c, t * c, c * c, t**2])
    assert columns.shape[1] - np.linalg.matrix_rank(columns) == 3
    assert "the solvent term with 0 of its 1 column(s) estimable;" in caplog.text
    assert "the T:solvent term with 0 of its 1 column(s) estimable;" in caplog.text
    assert "no allowed run has catalyst = 'A' with solvent = 'Y'" in caplog.text
