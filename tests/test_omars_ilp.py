"""Tests for the ILP-based OMARS design generator (generate_omars)."""

from __future__ import annotations

import itertools
import math
import subprocess
import sys
import textwrap
import threading
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from process_improve.experiments import Factor, analyze_omars, generate_design
from process_improve.experiments import designs_omars_ilp as omars_ilp
from process_improve.experiments.designs_omars import is_omars

# A generous wall-clock cap that never binds in these tests: the node budget,
# not the clock, decides where each solve stops, so results are deterministic.
_SOLVER = {"time_limit": 30, "msg": False}


def _factors(k: int) -> list[Factor]:
    return [Factor(name=chr(65 + i), low=-1, high=1) for i in range(k)]


def _coded(result) -> np.ndarray:
    names = result.factor_names
    return result.design[names].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_exposed_in_experiments_namespace() -> None:
    from process_improve.experiments import generate_omars as exported
    from process_improve.experiments.designs_omars_ilp import generate_omars

    assert exported is generate_omars


@pytest.mark.parametrize("k", [3, pytest.param(4, marks=pytest.mark.slow)])  # k=4 enumerates ~250k designs
def test_generated_design_is_omars(k: int) -> None:
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(k), solver_options=_SOLVER)
    assert is_omars(_coded(result))
    assert result.metadata["omars_verified"] is True
    assert result.metadata["family"] == "omars_ilp"


@pytest.mark.parametrize("k", [3, pytest.param(4, marks=pytest.mark.slow)])  # k=4 enumerates ~250k designs
def test_design_supports_analyze_omars(k: int) -> None:
    """The headline reason the generator exists: the design must leave error df."""
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(k), solver_options=_SOLVER)
    names = result.factor_names
    design = result.design[names]
    x = design.to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    y = 5 * x[:, 0] + 4 * x[:, 1] + 3 * (x[:, 0] * x[:, 1]) + 3 * (x[:, 0] ** 2) + rng.normal(0, 0.3, x.shape[0])

    analysis = analyze_omars(design, y)
    assert analysis.success is True
    assert analysis.initial_error_df >= 1
    assert result.metadata["expected_error_df"] >= 1


def test_exact_run_size_is_respected() -> None:
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), n_runs=15, solver_options=_SOLVER)
    assert result.metadata["n_runs_selected"] == 15
    assert _coded(result).shape[0] == 15
    assert is_omars(_coded(result))


def test_run_size_below_parameters_raises() -> None:
    from process_improve.experiments import generate_omars

    # k=3: full second-order model has 1 + 6 + 3 = 10 params; n_runs must exceed it.
    with pytest.raises(ValueError, match="error degrees of freedom"):
        generate_omars(_factors(3), n_runs=10, solver_options=_SOLVER)


def test_incompatible_run_size_parity_raises() -> None:
    from process_improve.experiments import generate_omars

    # n_runs - center_runs is the foldover part (h half-runs plus their mirrors),
    # so it must be a positive even number.
    with pytest.raises(ValueError, match="n_runs - center_runs must be a positive even number"):
        generate_omars(_factors(3), n_runs=16, solver_options=_SOLVER)
    with pytest.raises(ValueError, match="n_runs - center_runs must be a positive even number"):
        generate_omars(_factors(3), n_runs=15, center_runs=2, solver_options=_SOLVER)


def test_too_few_factors_raises() -> None:
    from process_improve.experiments import generate_omars

    with pytest.raises(ValueError, match="at least 3 factors"):
        generate_omars(_factors(2), solver_options=_SOLVER)


def test_unknown_selection_criterion_raises() -> None:
    from process_improve.experiments import generate_omars

    with pytest.raises(ValueError, match="selection_criterion"):
        generate_omars(_factors(3), selection_criterion="best", solver_options=_SOLVER)


def test_run_size_range_is_searched() -> None:
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), n_runs_range=(13, 17), solver_options=_SOLVER)
    n = result.metadata["n_runs_selected"]
    assert 13 <= n <= 17
    assert is_omars(_coded(result))


def test_extra_center_runs_are_added() -> None:
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), center_runs=3, solver_options=_SOLVER)
    coded = _coded(result)
    n_center = int(np.sum(np.all(coded == 0, axis=1)))
    assert n_center == 3
    assert is_omars(coded)


def test_sparsity_counts_the_extra_center_runs() -> None:
    """The sparsity pair was computed on the foldover alone, without the extra centre runs."""
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), center_runs=3, random_seed=3, solver_options=_SOLVER)
    coded = _coded(result)
    n_me0, n_ie0 = result.metadata["sparsity"]
    assert n_me0 == int(np.min(np.sum(coded == 0, axis=0)))
    interactions = [coded[:, i] * coded[:, j] for i in range(3) for j in range(i + 1, 3)]
    assert n_ie0 == min(int(np.sum(column == 0)) for column in interactions)


def test_n_restarts_is_honoured_and_checked() -> None:
    """n_restarts below 6 was silently raised to 6, and a negative value was accepted."""
    from process_improve.experiments import generate_omars

    report = generate_omars(_factors(3), n_restarts=0, solver_options=_SOLVER).metadata["omars_search"]
    assert report.n_restarts == 0
    report = generate_omars(_factors(3), n_restarts=0, max_candidates=6, solver_options=_SOLVER).metadata[
        "omars_search"
    ]
    assert report.n_restarts == 6
    with pytest.raises(ValueError, match="n_restarts must be >= 0"):
        generate_omars(_factors(3), n_restarts=-5, solver_options=_SOLVER)


@pytest.mark.parametrize(
    ("n_runs", "center_runs"),
    [(17, 3), (16, 2), (13, 1), (19, 5)],
)
def test_center_runs_count_towards_n_runs(n_runs: int, center_runs: int) -> None:
    """n_runs is the total run count of the returned design, centre runs included (#496)."""
    from process_improve.experiments import generate_omars

    result = generate_omars(
        _factors(3), n_runs=n_runs, center_runs=center_runs, model="main_quadratic", solver_options=_SOLVER
    )
    coded = _coded(result)
    assert coded.shape[0] == n_runs
    assert int(np.sum(np.all(coded == 0, axis=1))) == center_runs
    assert result.metadata["n_runs_selected"] == n_runs
    assert is_omars(coded)


def test_center_runs_below_one_raises() -> None:
    from process_improve.experiments import generate_omars

    with pytest.raises(ValueError, match="center_runs"):
        generate_omars(_factors(3), center_runs=0, solver_options=_SOLVER)


def test_factor_count_above_cap_raises() -> None:
    from process_improve.config import settings
    from process_improve.experiments import generate_omars

    original = settings.max_factors_combinatorial
    settings.max_factors_combinatorial = 4
    try:
        with pytest.raises(ValueError, match="combinatorial cap"):
            generate_omars(_factors(5), solver_options=_SOLVER)
    finally:
        settings.max_factors_combinatorial = original


def test_reproducible_for_fixed_seed() -> None:
    from process_improve.experiments import generate_omars

    a = generate_omars(_factors(3), random_state=7, solver_options=_SOLVER)
    b = generate_omars(_factors(3), random_state=7, solver_options=_SOLVER)
    np.testing.assert_array_equal(_coded(a), _coded(b))


def test_selection_criteria_all_yield_valid_omars() -> None:
    from process_improve.experiments import generate_omars

    for criterion in ("dominance", "d_efficiency", "min_second_order_correlation"):
        result = generate_omars(_factors(3), selection_criterion=criterion, solver_options=_SOLVER)
        assert is_omars(_coded(result))


def test_satisfice_thresholds_are_applied() -> None:
    from process_improve.experiments import generate_omars

    # A permissive ceiling keeps the candidates; the winner must honour it.
    result = generate_omars(
        _factors(3),
        satisfice={"max_second_order_correlation": 0.8},
        solver_options=_SOLVER,
    )
    assert is_omars(_coded(result))
    assert result.metadata["satisfice"] == {"max_second_order_correlation": 0.8}
    assert result.metadata["max_second_order_correlation"] <= 0.8


def test_satisfice_unreachable_threshold_raises() -> None:
    from process_improve.experiments import generate_omars

    with pytest.raises(ValueError, match="satisfice thresholds"):
        generate_omars(_factors(3), satisfice={"d_efficiency": 999.0}, solver_options=_SOLVER)


def test_satisfice_unknown_key_raises() -> None:
    from process_improve.experiments import generate_omars

    with pytest.raises(ValueError, match="satisfice keys"):
        generate_omars(_factors(3), satisfice={"g_efficiency": 50.0}, solver_options=_SOLVER)


# ---------------------------------------------------------------------------
# Quality-driven multistart search
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_multistart_reaches_catalogue_quality() -> None:
    """The randomized multistart finds a high-D-efficiency 25-run, 5-factor OMARS design.

    A pure feasibility search (the old behaviour) topped out near D-efficiency 37
    for this cell, while the enumerated OMARS catalogue reaches roughly 39 to 40.6.
    The multistart must clear a catalogue-competitive bar; this is the regression
    guard for the old feasibility-only ceiling.
    """
    from process_improve.experiments import generate_omars

    result = generate_omars(
        _factors(5),
        n_runs=25,
        model="main_quadratic",
        selection_criterion="d_efficiency",
        n_restarts=40,
        solver_options=_SOLVER,
    )
    assert is_omars(_coded(result))
    assert result.metadata["d_efficiency"] >= 39.0
    report = result.metadata["omars_search"]
    assert report.n_restarts == 40
    assert report.search_mode == "multistart"
    # The search retains many distinct designs, not the handful the old no-good cuts found.
    assert report.feasible_designs > 1


@pytest.mark.slow
def test_more_restarts_is_never_worse() -> None:
    """Adding restarts can only match or improve the selected design's quality.

    Both calls run the same deterministic baseline feasibility solve first, so the
    multistart's candidate set is a superset and its D-efficiency cannot be lower.
    """
    from process_improve.experiments import generate_omars

    baseline = generate_omars(
        _factors(5),
        n_runs=25,
        model="main_quadratic",
        selection_criterion="d_efficiency",
        n_restarts=0,
        solver_options=_SOLVER,
    )
    searched = generate_omars(
        _factors(5),
        n_runs=25,
        model="main_quadratic",
        selection_criterion="d_efficiency",
        n_restarts=40,
        solver_options=_SOLVER,
    )
    assert searched.metadata["d_efficiency"] >= baseline.metadata["d_efficiency"]
    assert searched.metadata["d_efficiency"] >= 39.0  # and it clears the catalogue-competitive bar


@pytest.mark.slow
def test_multistart_is_deterministic_for_seed() -> None:
    """The multistart path reproduces the same design for a fixed seed (k=5)."""
    from process_improve.experiments import generate_omars

    a = generate_omars(
        _factors(5), n_runs=25, model="main_quadratic", n_restarts=20, random_state=1, solver_options=_SOLVER
    )
    b = generate_omars(
        _factors(5), n_runs=25, model="main_quadratic", n_restarts=20, random_state=1, solver_options=_SOLVER
    )
    np.testing.assert_array_equal(_coded(a), _coded(b))


def test_legacy_max_candidates_sets_restart_floor() -> None:
    """max_candidates is retained as a floor on the effective restart budget."""
    from process_improve.experiments import generate_omars

    result = generate_omars(
        _factors(4), n_runs=13, model="main_quadratic", n_restarts=1, max_candidates=30, solver_options=_SOLVER
    )
    assert is_omars(_coded(result))
    assert result.metadata["omars_search"].n_restarts == 30


# ---------------------------------------------------------------------------
# Reduced-model sizing (model="main_quadratic")
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_main_quadratic_builds_sub_full_model_design() -> None:
    """A four-factor OMARS can be sized for the main-effects-plus-quadratics model.

    The full second-order model has 1 + 8 + 6 = 15 parameters, so it needs at
    least 17 runs; the main-quadratic model has only 1 + 8 = 9, so a thirteen-run
    design (like the one used in the book) is feasible.
    """
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(4), n_runs=13, model="main_quadratic", solver_options=_SOLVER)
    assert result.metadata["n_runs_selected"] == 13
    assert result.metadata["sizing_model"] == "main_quadratic"
    assert result.metadata["model_params"] == 9
    # The full-model parameter count is still reported for reference, and the
    # design still leaves error df for the model it was sized for.
    assert result.metadata["full_second_order_params"] == 15
    assert result.metadata["expected_error_df"] == 13 - 9
    assert is_omars(_coded(result))


@pytest.mark.slow
def test_main_quadratic_auto_size_is_smaller_than_full() -> None:
    from process_improve.experiments import generate_omars

    reduced = generate_omars(_factors(4), model="main_quadratic", solver_options=_SOLVER)
    full = generate_omars(_factors(4), model="full_second_order", solver_options=_SOLVER)
    assert reduced.n_runs < full.n_runs
    assert reduced.metadata["sizing_model"] == "main_quadratic"
    assert full.metadata["sizing_model"] == "full_second_order"


def test_main_quadratic_run_size_floor() -> None:
    """The reduced model still needs error df: 13 runs is fine, 9 is not."""
    from process_improve.experiments import generate_omars

    # k=4 main-quadratic has 9 parameters, so n_runs must exceed 9.
    with pytest.raises(ValueError, match="main_quadratic model has 9 parameters"):
        generate_omars(_factors(4), n_runs=9, model="main_quadratic", solver_options=_SOLVER)


def test_unknown_model_raises() -> None:
    from process_improve.experiments import generate_omars

    with pytest.raises(ValueError, match="model must be one of"):
        generate_omars(_factors(4), model="cubic", solver_options=_SOLVER)


def test_default_model_is_full_second_order() -> None:
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), solver_options=_SOLVER)
    assert result.metadata["sizing_model"] == "full_second_order"


@pytest.mark.slow
def test_a_optimal_selects_minimum_coefficient_variance() -> None:
    """The a_optimal criterion returns the lowest trace((X'X)^-1) design enumerated.

    This is the criterion that reproduces the precision-optimal four-factor OMARS
    member used as the book's running example (lower average prediction variance,
    which is why that design sits below the DSD on the FDS plot).
    """
    import numpy as np

    from process_improve.experiments import generate_omars
    from process_improve.experiments.designs_omars_ilp import _a_optimality

    result = generate_omars(
        _factors(4),
        n_runs=13,
        model="main_quadratic",
        selection_criterion="a_optimal",
        max_candidates=40,
        solver_options=_SOLVER,
    )
    coded = _coded(result)
    assert is_omars(coded)
    # The reported A-optimality matches a direct recomputation, and it is the known
    # optimum (A = 2.52) for the 13-run four-factor main-quadratic OMARS family.
    assert result.metadata["a_optimality"] == pytest.approx(_a_optimality(coded, "main_quadratic"))
    assert result.metadata["a_optimality"] == pytest.approx(2.517, abs=0.01)
    assert np.isclose(result.metadata["max_second_order_correlation"], 0.570, atol=0.005)


# ---------------------------------------------------------------------------
# Integration and dependency gating
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_registry_integration() -> None:
    result = generate_design(_factors(4), design_type="omars_ilp", budget=21)
    assert is_omars(result.design[result.factor_names].to_numpy(dtype=float))
    assert result.metadata["n_runs_selected"] == 21


@pytest.mark.slow
def test_omars_with_budget_reaches_ilp() -> None:
    """design_type="omars" with a budget routes to the ILP enumerator."""
    result = generate_design(_factors(4), design_type="omars", budget=21, n_center_points=0)
    assert result.metadata["family"] == "omars_ilp"
    assert result.metadata["n_runs_selected"] == 21
    assert is_omars(result.design[result.factor_names].to_numpy(dtype=float))


def test_omars_without_budget_is_minimal_foldover() -> None:
    """design_type="omars" with no budget keeps the minimal conference-foldover member."""
    result = generate_design(_factors(4), design_type="omars", n_center_points=0)
    assert result.metadata["family"] == "conference_foldover"
    assert result.n_runs == 9  # the minimal four-factor member (the DSD)


# ---------------------------------------------------------------------------
# The HiGHS solve (solve_omars_ilp)
# ---------------------------------------------------------------------------

# A main-effect-orthogonal half-design of 15 runs from _half_pool(5) whose
# foldover is a valid OMARS design but cannot fit the full second-order model
# (rank 16 of 21).  It is the selection a plain feasibility solve returns at
# this size on HiGHS 1.12, frozen here so the tests below do not depend on the
# solver version.
_RANK_DEFICIENT_K5 = [0, 1, 2, 3, 8, 26, 35, 53, 62, 71, 79, 80, 81, 98, 116]


class _MilpSpy:
    """Stand-in for ``designs_omars_ilp.milp``: records each call, then solves for real.

    *override* maps a call number (from 0) to a function that rewrites the real
    result, so a test can present the code with any status HiGHS can return.
    """

    def __init__(self, real_milp, override=None) -> None:
        self.real_milp = real_milp
        self.override = override or {}
        self.options: list[dict] = []

    def __call__(self, c, **kwargs) -> OptimizeResult:
        call = len(self.options)
        self.options.append(dict(kwargs["options"]))
        # Record what the module sent, but solve without "threads": HiGHS keeps one
        # thread pool per calling thread, and another test in this process (a
        # default linprog, say) may already have started it multi-threaded, which
        # would make a real threads=1 solve fail with "Not Set".
        options = {key: value for key, value in kwargs["options"].items() if key != "threads"}
        result = self.real_milp(c, **{**kwargs, "options": options})
        if call in self.override:
            result = self.override[call](result)
        return result


def _spy(monkeypatch: pytest.MonkeyPatch, override=None) -> _MilpSpy:
    # Start each spied test from clean per-thread and per-fork HiGHS bookkeeping.
    monkeypatch.setattr(omars_ilp, "_thread_state", threading.local())
    monkeypatch.setitem(omars_ilp._fork_state, "in_child", False)
    spy = _MilpSpy(omars_ilp.milp, override)
    monkeypatch.setattr(omars_ilp, "milp", spy)
    return spy


def _as(status: int, message: str, *, keep_x: bool = True):
    """Rewrite a real milp result to report *status* and *message* (optionally without x)."""

    def rewrite(result: OptimizeResult) -> OptimizeResult:
        return OptimizeResult(x=result.x if keep_x else None, status=status, message=message)

    return rewrite


def test_solve_returns_a_verified_foldover() -> None:
    design, status, chosen = omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=3, solver_options=_SOLVER)
    assert status == "Optimal"
    assert len(chosen) == 3
    assert design.shape == (7, 3)
    assert is_omars(design)


def test_seeded_objective_reproduces_the_selection() -> None:
    pool = omars_ilp._half_pool(4)
    objective = np.random.default_rng(7).standard_normal(len(pool))
    first = omars_ilp.solve_omars_ilp(pool, n_half=10, objective=objective, solver_options=_SOLVER)
    second = omars_ilp.solve_omars_ilp(pool, n_half=10, objective=objective, solver_options=_SOLVER)
    assert first[2] == second[2]
    np.testing.assert_array_equal(first[0], second[0])


def test_infeasible_size_returns_no_design() -> None:
    """Fourteen half-runs cannot be chosen from a 13-run pool of binaries."""
    assert omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=14) == (None, "Infeasible", [])


def test_solve_requires_a_size() -> None:
    with pytest.raises(ValueError, match="either n_half or half_bounds"):
        omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3))


def test_half_bounds_window_is_respected() -> None:
    _, _, chosen = omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), half_bounds=(5, 7), minimize_size=True)
    assert len(chosen) == 5


def test_no_good_cut_excludes_a_previous_selection() -> None:
    pool = omars_ilp._half_pool(3)
    _, _, first = omars_ilp.solve_omars_ilp(pool, n_half=4, solver_options=_SOLVER)
    design, _, second = omars_ilp.solve_omars_ilp(pool, n_half=4, exclude_solutions=[first], solver_options=_SOLVER)
    assert sorted(second) != sorted(first)
    assert is_omars(design)


def test_options_reach_highs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every solve is single-threaded; only objective solves carry the node budget."""
    spy = _spy(monkeypatch)
    pool = omars_ilp._half_pool(3)
    objective = np.random.default_rng(0).standard_normal(len(pool))
    user_options = {"time_limit": 0.5, "msg": False}
    omars_ilp.solve_omars_ilp(pool, half_bounds=(6, 8), minimize_size=True, solver_options=user_options)
    omars_ilp.solve_omars_ilp(pool, n_half=6, solver_options=user_options)
    omars_ilp.solve_omars_ilp(pool, n_half=6, objective=objective, solver_options=user_options)
    omars_ilp.solve_omars_ilp(pool, n_half=6, objective=objective, solver_options={"node_limit": 7, "msg": True})
    omars_ilp.solve_omars_ilp(pool, n_half=6, objective=objective, solver_options={"node_limit": None})
    minimize, feasibility, default_nodes, custom_nodes, unlimited = spy.options
    assert all(options["threads"] == 1 for options in spy.options)
    # A fractional limit reaches HiGHS unchanged; it used to be truncated to 0.
    assert minimize["time_limit"] == 0.5
    assert "node_limit" not in minimize
    assert "node_limit" not in feasibility
    assert default_nodes["node_limit"] == 100
    assert custom_nodes["node_limit"] == 7
    assert custom_nodes["disp"] is True
    assert minimize["disp"] is False
    assert "node_limit" not in unlimited
    # The randomized-objective solves skip the costly root heuristics and strong
    # branching; the minimise-size and feasibility solves keep the HiGHS defaults.
    for options in (default_nodes, custom_nodes, unlimited):
        assert options["mip_heuristic_run_rins"] is False
        assert options["mip_heuristic_run_rens"] is False
        assert options["mip_pscost_minreliable"] == 0
    for options in (minimize, feasibility):
        assert not set(options) & set(omars_ilp._RANDOM_OBJECTIVE_HIGHS_OPTIONS)
    # milp pops keys out of the options dict it receives; the caller's dict is untouched.
    assert user_options == {"time_limit": 0.5, "msg": False}


@pytest.mark.parametrize(
    ("status", "message", "keep_x", "label"),
    [
        (4, "The HiGHS status code was not recognized. (HiGHS Status 16: Solution limit reached)", True, "Node limit"),
        (1, "Iteration limit reached. (HiGHS Status 14: Iteration limit reached)", True, "Node limit"),
        (1, "Time limit reached. (HiGHS Status 13: Time limit reached)", True, "Time limit"),
        (1, "Time limit reached. (HiGHS Status 13: model_status is Time limit reached)", False, "Time limit"),
        (2, "The problem is infeasible. (HiGHS Status 8: model_status is Infeasible)", False, "Infeasible"),
        (4, "The HiGHS status code was not recognized. (HiGHS Status 15: Unknown)", False, "Not Solved"),
        (0, "Optimization terminated successfully.", True, "Optimal"),
        (4, "Something unexpected.", False, "Not Solved"),
    ],
)
def test_status_labels(monkeypatch: pytest.MonkeyPatch, status: int, message: str, keep_x: bool, label: str) -> None:
    """A design is returned whenever HiGHS hands one back, whatever stopped the solve."""
    _spy(monkeypatch, {0: _as(status, message, keep_x=keep_x)})
    pool = omars_ilp._half_pool(4)
    objective = np.random.default_rng(3).standard_normal(len(pool))
    design, solver_status, chosen = omars_ilp.solve_omars_ilp(pool, n_half=10, objective=objective)
    assert solver_status == label
    if keep_x:
        assert is_omars(design)
        assert len(chosen) == 10
    else:
        assert (design, chosen) == (None, [])


@pytest.fixture
def fresh_highs_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the module's per-thread and per-fork HiGHS bookkeeping from other tests."""
    monkeypatch.setattr(omars_ilp, "_thread_state", threading.local())
    monkeypatch.setitem(omars_ilp._fork_state, "in_child", False)


_NOT_SET = _as(4, "(HiGHS Status 0: Not Set)", keep_x=False)


@pytest.mark.usefixtures("fresh_highs_state")
def test_scheduler_conflict_retries_without_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """HiGHS refuses threads=1 with "Not Set" when this thread already runs it multi-threaded.

    The refused attempt is made once; later solves on the thread go straight to the
    existing pool.
    """
    spy = _spy(monkeypatch, {0: _NOT_SET})
    design, status, _ = omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=3)
    assert status == "Optimal"
    assert is_omars(design)
    omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=3)
    assert [options.get("threads") for options in spy.options] == [1, None, None]


@pytest.mark.usefixtures("fresh_highs_state")
def test_forked_child_resets_the_inherited_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """In a fork child the inherited pool has no threads; it is reset, never solved on."""
    spy = _spy(monkeypatch, {0: _NOT_SET})
    monkeypatch.setitem(omars_ilp._fork_state, "in_child", True)
    monkeypatch.setattr(omars_ilp, "_reset_highs_scheduler", lambda: True)
    _, status, _ = omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=3)
    assert status == "Optimal"
    assert [options.get("threads") for options in spy.options] == [1, 1]


@pytest.mark.usefixtures("fresh_highs_state")
def test_forked_child_without_a_reset_does_not_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _spy(monkeypatch, {0: _NOT_SET})
    monkeypatch.setitem(omars_ilp._fork_state, "in_child", True)
    monkeypatch.setattr(omars_ilp, "_reset_highs_scheduler", lambda: False)
    assert omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=3) == (None, "Not Solved", [])
    assert len(spy.options) == 1


def test_scheduler_reset_is_available() -> None:
    """The private SciPy hook the fork-child path relies on exists in the tested SciPy releases."""
    assert omars_ilp._reset_highs_scheduler() is True


def test_selection_violating_its_constraints_is_a_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    def all_runs(result: OptimizeResult) -> OptimizeResult:
        return OptimizeResult(x=np.ones_like(result.x), status=0, message="(HiGHS Status 7: Optimal)")

    _spy(monkeypatch, {0: all_runs})
    with pytest.raises(RuntimeError, match="internal: the HiGHS selection violates its constraints"):
        omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=3)


def test_single_factor_pool_has_no_orthogonality_rows() -> None:
    design, status, chosen = omars_ilp.solve_omars_ilp(omars_ilp._half_pool(1), n_half=1)
    assert (status, chosen) == ("Optimal", [0])
    np.testing.assert_array_equal(design, [[1.0], [-1.0], [0.0]])


def test_non_orthogonal_selection_is_a_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = omars_ilp._half_pool(3)
    row = int(np.flatnonzero((pool == [1, 1, 1]).all(axis=1))[0])

    def correlated(result: OptimizeResult) -> OptimizeResult:
        x = np.zeros_like(result.x)
        x[row] = 1.0
        return OptimizeResult(x=x, status=0, message="(HiGHS Status 7: Optimal)")

    _spy(monkeypatch, {0: correlated})
    with pytest.raises(RuntimeError, match="the main effects are not orthogonal"):
        omars_ilp.solve_omars_ilp(pool, half_bounds=(1, 6))


def test_every_factor_leaves_its_middle_level() -> None:
    """A factor at 0 in every half-run is trivially orthogonal but not three-level; the ILP forbids it."""
    pool = omars_ilp._half_pool(3)
    for seed in range(20):
        objective = np.random.default_rng(seed).standard_normal(len(pool))
        design, _, _ = omars_ilp.solve_omars_ilp(pool, n_half=3, objective=objective, solver_options=_SOLVER)
        assert np.all(np.abs(design).sum(axis=0) > 0)
        assert is_omars(design)


def test_selection_with_an_unvaried_factor_is_a_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact check also catches a factor that never leaves its middle level."""
    pool = omars_ilp._half_pool(3)
    # (1, 0, 0) and (0, 1, 0): orthogonal, but the third factor never moves.
    rows = [int(np.flatnonzero((pool == run).all(axis=1))[0]) for run in ([1, 0, 0], [0, 1, 0])]

    def degenerate(result: OptimizeResult) -> OptimizeResult:
        x = np.zeros_like(result.x)
        x[rows] = 1.0
        return OptimizeResult(x=x, status=0, message="(HiGHS Status 7: Optimal)")

    _spy(monkeypatch, {0: degenerate})
    with pytest.raises(RuntimeError, match="a factor never leaves its middle level"):
        omars_ilp.solve_omars_ilp(pool, half_bounds=(2, 6))


def test_empty_selection_is_infeasible() -> None:
    """Zero half-runs leave every factor at its middle level, which the coverage rows forbid."""
    assert omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=0) == (None, "Infeasible", [])


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ([("time_limit", 30)], TypeError, "solver_options must be a dict"),
        ({"solver": "cbc"}, ValueError, r"accepts only the keys \['msg', 'time_limit', 'node_limit'\]"),
        ({"time_limit": True}, TypeError, "time_limit'] must be a number of seconds"),
        ({"time_limit": "30"}, TypeError, "time_limit'] must be a number of seconds"),
        ({"time_limit": 0}, ValueError, "time_limit'] must be positive"),
        ({"time_limit": -1.0}, ValueError, "time_limit'] must be positive"),
        ({"time_limit": math.nan}, ValueError, "time_limit'] must be positive"),
        ({"node_limit": 100.0}, TypeError, "node_limit'] must be an integer or None"),
        ({"node_limit": True}, TypeError, "node_limit'] must be an integer or None"),
        ({"node_limit": 0}, ValueError, "node_limit'] must be between 1 and 2147483647"),
        ({"node_limit": 2**31}, ValueError, "node_limit'] must be between 1 and 2147483647"),
    ],
)
def test_solver_options_are_validated(options, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=3, solver_options=options)


def test_solver_options_accept_numpy_and_unbounded_values() -> None:
    settings = omars_ilp._solver_settings({"time_limit": math.inf, "node_limit": np.int64(5), "msg": False})
    assert settings == omars_ilp._SolverSettings(msg=False, time_limit=math.inf, node_limit=5)
    assert omars_ilp._solver_settings(None).node_limit == 100


def test_solver_options_are_validated_before_an_exhaustive_search() -> None:
    """A pinned size takes the solver-free exhaustive path, which still checks the options."""
    from process_improve.experiments import generate_omars

    with pytest.raises(ValueError, match="accepts only the keys"):
        generate_omars(_factors(3), n_runs=15, solver_options={"timeLimit": 30})


def test_no_warning_escapes_a_multistart() -> None:
    """SciPy warns about the "threads" option it forwards; the module silences exactly that."""
    from process_improve.experiments import generate_omars

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = generate_omars(
            _factors(5), n_runs=25, model="main_quadratic", n_restarts=2, max_candidates=0, solver_options=_SOLVER
        )
    assert result.metadata["search_mode"] == "multistart"
    assert result.metadata["solver"] == "highs"


def _run_isolated(code: str) -> str:
    """Run *code* in a fresh interpreter, so HiGHS's process-wide state cannot leak between tests."""
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and test-controlled code
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=300, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@pytest.mark.slow
def test_omars_solves_after_highs_ran_multi_threaded() -> None:
    """An earlier default-threads HiGHS solve in the process must not break generate_omars."""
    out = _run_isolated(
        """
        from scipy.optimize import linprog
        from process_improve.experiments import Factor, generate_omars
        linprog(c=[1, 1], A_ub=[[-1, -1]], b_ub=[-1], method="highs")
        factors = [Factor(name=c, low=-1, high=1) for c in "ABCDE"]
        result = generate_omars(factors, n_runs=25, model="main_quadratic", n_restarts=2, max_candidates=0)
        print(result.metadata["n_runs_selected"], result.metadata["omars_verified"])
        """
    )
    assert out.split() == ["25", "True"]


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="the fork start method is only the default on Linux")
def test_forked_child_can_solve_after_the_parent() -> None:
    """A multi-threaded HiGHS solve in a parent makes a forked child hang; ours runs on one thread."""
    out = _run_isolated(
        """
        import multiprocessing as mp
        import numpy as np
        from process_improve.experiments import designs_omars_ilp as omars_ilp

        pool = omars_ilp._half_pool(5)

        def solve(queue):
            objective = np.random.default_rng(1).standard_normal(len(pool))
            queue.put(omars_ilp.solve_omars_ilp(pool, n_half=15, objective=objective)[1])

        if __name__ == "__main__":
            omars_ilp.solve_omars_ilp(pool, n_half=15, objective=np.random.default_rng(0).standard_normal(len(pool)))
            context = mp.get_context("fork")
            queue = context.Queue()
            child = context.Process(target=solve, args=(queue,))
            child.start()
            child.join(120)
            hung = child.is_alive()
            if hung:
                child.kill()
            print("hung" if hung else queue.get())
        """
    )
    assert out.split() != ["hung"]
    assert out.strip() in {"Optimal", "Node limit"}


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="the fork start method is only the default on Linux")
def test_forked_child_solves_after_parent_ran_highs_multi_threaded() -> None:
    """Regression: the child used to retry on the inherited pool and hang."""
    out = _run_isolated(
        """
        import multiprocessing as mp
        import numpy as np
        from scipy.optimize import linprog
        from process_improve.experiments import designs_omars_ilp as omars_ilp

        pool = omars_ilp._half_pool(5)
        objective = np.random.default_rng(1).standard_normal(len(pool))

        def solve(queue):
            queue.put(omars_ilp.solve_omars_ilp(pool, n_half=15, objective=objective)[1])

        if __name__ == "__main__":
            linprog(c=[1, 1], A_ub=[[-1, -1]], b_ub=[-1], method="highs")
            omars_ilp.solve_omars_ilp(pool, n_half=15, objective=objective)
            context = mp.get_context("fork")
            queue = context.Queue()
            child = context.Process(target=solve, args=(queue,))
            child.start()
            child.join(120)
            hung = child.is_alive()
            if hung:
                child.kill()
            print("hung" if hung else queue.get())
        """
    )
    assert out.strip() in {"Optimal", "Node limit"}


@pytest.mark.slow
def test_omars_does_not_need_pulp() -> None:
    """The generator runs with pulp unimportable: HiGHS ships with SciPy."""
    out = _run_isolated(
        """
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "pulp" or name.startswith("pulp."):
                    raise ImportError("pulp is blocked in this test")

        sys.meta_path.insert(0, Blocker())
        from process_improve.experiments import Factor, generate_omars
        result = generate_omars([Factor(name=c, low=-1, high=1) for c in "ABC"])
        print(result.metadata["solver"], "pulp" in sys.modules)
        """
    )
    assert out.split() == ["highs", "False"]


# ---------------------------------------------------------------------------
# Search diagnostics, estimability and error messages
# ---------------------------------------------------------------------------


def test_search_report_records_diagnostics() -> None:
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), solver_options=_SOLVER)
    report = result.metadata["omars_search"]
    assert report.n_factors == 3
    assert report.half_pool_size == (3**3 - 1) // 2
    assert report.ilp_iterations >= 1
    assert report.feasible_designs >= 1
    assert report.total_solve_seconds >= 0.0
    # Three factors at the automatic size fall within the exhaustive regime.
    assert report.search_mode == "exhaustive"
    assert report.enumerated_designs == report.feasible_designs
    assert result.metadata["search_mode"] == "exhaustive"
    assert result.metadata["solver"] == "highs"
    assert result.metadata["solver_status"] == "Enumerated"
    # The minimise-size solve proved its size, and the enumeration set aside the
    # designs that cannot fit the model.
    assert report.size_proven_minimal is True
    assert 0 < report.rank_deficient_designs < report.enumerated_designs
    assert (report.node_limit, report.time_limit) == (100, 30.0)
    assert report.time_limited_solves == 0
    assert report.run_sizes_searched == 1


def test_pinned_size_is_not_reported_as_minimal() -> None:
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), n_runs=15, solver_options=_SOLVER)
    assert result.metadata["omars_search"].size_proven_minimal is None


def test_multistart_is_deterministic_and_estimable() -> None:
    """A short multistart: the same seed gives the same design, and the winner fits its model."""
    from process_improve.experiments import generate_omars

    def run():
        return generate_omars(
            _factors(5), n_runs=25, model="main_quadratic", n_restarts=3, max_candidates=0, solver_options=_SOLVER
        )

    first, second = run(), run()
    np.testing.assert_array_equal(_coded(first), _coded(second))
    assert first.metadata["search_mode"] == "multistart"
    assert first.metadata["model_rank"] == first.metadata["model_params"]
    report = first.metadata["omars_search"]
    assert report.feasible_designs >= 1
    assert report.size_proven_minimal is None


def test_limit_stopped_solves_are_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Node-limited solves are routine; time-limited ones mean the result may vary by machine."""
    from process_improve.experiments import generate_omars

    node = _as(4, "(HiGHS Status 16: Solution limit reached)")
    clock = _as(1, "(HiGHS Status 13: Time limit reached)")
    _spy(monkeypatch, {1: node, 2: clock, 3: node})
    result = generate_omars(
        _factors(5), n_runs=25, model="main_quadratic", n_restarts=3, max_candidates=0, solver_options=_SOLVER
    )
    report = result.metadata["omars_search"]
    assert (report.node_limited_solves, report.time_limited_solves) == (2, 1)


def test_unproven_minimum_size_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from process_improve.experiments import generate_omars

    _spy(monkeypatch, {0: _as(1, "(HiGHS Status 13: Time limit reached)")})
    result = generate_omars(_factors(3), solver_options=_SOLVER)
    report = result.metadata["omars_search"]
    assert report.size_proven_minimal is False
    assert report.time_limited_solves == 1


def _always_returns(monkeypatch: pytest.MonkeyPatch, result: tuple) -> None:
    """Make every solve in the search return *result*."""
    monkeypatch.setattr(omars_ilp, "solve_omars_ilp", lambda *_args, **_kwargs: result)


def test_rank_deficient_designs_never_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid OMARS design that cannot fit the model is set aside, and the error says why."""
    from process_improve.experiments import generate_omars

    pool = omars_ilp._half_pool(5)
    design = omars_ilp._foldover(pool[_RANK_DEFICIENT_K5])
    assert is_omars(design)
    assert omars_ilp._model_rank(design) < omars_ilp._full_second_order_params(5)
    _always_returns(monkeypatch, (design, "Optimal", _RANK_DEFICIENT_K5))
    with pytest.raises(ValueError, match=r"1 OMARS design\(s\) were found at n_runs=31, but the full_second_order"):
        generate_omars(_factors(5), n_runs=31, n_restarts=2, max_candidates=0)


def test_automatic_size_moves_up_when_nothing_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the size chosen automatically, a rank-deficient smallest size is not the answer.

    Every solve at 31 runs returns the rank-deficient design; the search moves
    to 33 runs, where the real solver finds designs that fit the model.
    """
    from process_improve.experiments import generate_omars

    real_solve = omars_ilp.solve_omars_ilp
    pool = omars_ilp._half_pool(5)
    stuck = (omars_ilp._foldover(pool[_RANK_DEFICIENT_K5]), "Optimal", _RANK_DEFICIENT_K5)

    def solve(half_pool, **kwargs):
        if kwargs.get("minimize_size") or kwargs.get("n_half") == len(_RANK_DEFICIENT_K5):
            return stuck
        return real_solve(half_pool, **kwargs)

    monkeypatch.setattr(omars_ilp, "solve_omars_ilp", solve)
    result = generate_omars(_factors(5), n_restarts=2, max_candidates=0, solver_options=_SOLVER)
    report = result.metadata["omars_search"]
    assert result.metadata["n_runs_selected"] == 33
    assert result.metadata["model_rank"] == result.metadata["model_params"] == 21
    assert report.run_sizes_searched == 2
    assert report.rank_deficient_designs >= 1
    # The minimum was proven at 31 runs, but the design has 33.
    assert report.size_proven_minimal is False


def _singular_at(monkeypatch: pytest.MonkeyPatch, n_half: int | None) -> None:
    """Score every enumerated design of *n_half* half-runs (all of them if None) as rank-deficient."""
    real_score = omars_ilp._score_count_vectors

    def score(count_matrix, *args, **kwargs):
        d_eff, a_opt, max_corr = real_score(count_matrix, *args, **kwargs)
        if n_half is None or int(count_matrix[0].sum()) == n_half:
            d_eff = np.zeros_like(d_eff)
        return d_eff, a_opt, max_corr

    monkeypatch.setattr(omars_ilp, "_score_count_vectors", score)


def test_exhaustive_search_moves_up_when_nothing_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    from process_improve.experiments import generate_omars

    _singular_at(monkeypatch, 6)  # three factors start at 6 half-runs (13 runs)
    result = generate_omars(_factors(3), solver_options=_SOLVER)
    report = result.metadata["omars_search"]
    assert result.metadata["n_runs_selected"] == 15
    assert report.search_mode == "exhaustive"
    assert report.run_sizes_searched == 2
    assert report.rank_deficient_designs > 0


def test_exhaustive_rank_deficiency_at_a_pinned_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete enumeration saw every design, so the advice is more runs, not more restarts."""
    from process_improve.experiments import generate_omars

    _singular_at(monkeypatch, None)
    with pytest.raises(ValueError, match=r"at n_runs=15, but the full_second_order model .* Ask for more runs\.$"):
        generate_omars(_factors(3), n_runs=15)


def test_rank_deficient_everywhere_names_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from process_improve.experiments import generate_omars

    pool = omars_ilp._half_pool(5)
    _always_returns(monkeypatch, (omars_ilp._foldover(pool[_RANK_DEFICIENT_K5]), "Optimal", _RANK_DEFICIENT_K5))
    with pytest.raises(ValueError, match=r"1 OMARS design\(s\) were found at 31 to 43 runs, but the full_second"):
        generate_omars(_factors(5), n_restarts=1, max_candidates=0)


@pytest.mark.slow
def test_exhaustive_search_skips_rank_deficient_designs() -> None:
    """Regression: four factors at the automatic size used to return a rank-14 design.

    Ranked by correlation first, the exhaustive winner was a design the full
    second-order model (15 parameters) cannot be fitted to, with D-efficiency 0.
    """
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(4), selection_criterion="min_second_order_correlation", solver_options=_SOLVER)
    assert result.metadata["model_rank"] == result.metadata["model_params"] == 15
    assert result.metadata["d_efficiency"] > 0
    assert result.metadata["omars_search"].rank_deficient_designs > 0


def test_no_design_in_auto_window_names_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from process_improve.experiments import generate_omars

    _always_returns(monkeypatch, (None, "Time limit", []))
    with pytest.raises(ValueError, match="at 13 to 25 runs") as raised:
        generate_omars(_factors(3))
    assert "n_runs_range=None" not in str(raised.value)
    assert "time_limit" in str(raised.value)


def test_no_design_at_pinned_size_names_the_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from process_improve.experiments import generate_omars

    _always_returns(monkeypatch, (None, "Node limit", []))
    with pytest.raises(ValueError, match="at n_runs=31 with center_runs=1: the last solve ended with status 'Node"):
        generate_omars(_factors(5), n_runs=31, n_restarts=2, max_candidates=0)


# ---------------------------------------------------------------------------
# Run-size windows and input validation
# ---------------------------------------------------------------------------


def test_window_beyond_the_distinct_pool_uses_repeated_half_runs() -> None:
    """Three factors have 13 distinct half-runs; 31 runs need 15, which only the enumeration can repeat."""
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), n_runs_range=(31, 37), solver_options=_SOLVER)
    assert result.metadata["n_runs_selected"] == 31
    assert result.metadata["search_mode"] == "exhaustive"
    assert result.metadata["omars_search"].size_proven_minimal is None
    assert is_omars(_coded(result))


def test_pinned_size_beyond_reach_is_refused() -> None:
    from process_improve.experiments import generate_omars

    with pytest.raises(ValueError, match=r"n_runs=39 needs 19 half-runs, but there are only 13 distinct"):
        generate_omars(_factors(3), n_runs=39)


@pytest.mark.parametrize(
    ("window", "match"),
    [((25, 13), r"min <= max, got \(25, 13\)"), ((5, 9), r"n_runs_range=\(5, 9\) ends below 13 runs")],
)
def test_unusable_run_window_is_refused(window: tuple[int, int], match: str) -> None:
    from process_improve.experiments import generate_omars

    with pytest.raises(ValueError, match=match):
        generate_omars(_factors(3), n_runs_range=window)


def test_rank_screen_does_not_depend_on_the_is_omars_tolerance() -> None:
    """A loose ``tol`` used to mark full-rank enumerated designs as singular and push the size up."""
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), tol=0.05, solver_options=_SOLVER)
    assert result.metadata["n_runs_selected"] == 13
    assert result.metadata["omars_search"].run_sizes_searched == 1


def test_enumeration_counts_only_designs_where_every_factor_varies() -> None:
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), solver_options=_SOLVER)
    pool = omars_ilp._half_pool(3)
    counts, overflow = omars_ilp._enumerate_feasible_counts(pool, 6, omars_ilp._ENUM_MAX_LEAVES)
    covered = int((counts @ np.abs(pool) > 0).all(axis=1).sum())
    assert not overflow
    assert covered < counts.shape[0]
    assert result.metadata["omars_search"].enumerated_designs == covered


def test_infeasible_status_gets_matching_advice(monkeypatch: pytest.MonkeyPatch) -> None:
    from process_improve.experiments import generate_omars

    _always_returns(monkeypatch, (None, "Infeasible", []))
    with pytest.raises(ValueError, match="No selection of distinct half-runs satisfies") as raised:
        generate_omars(_factors(3))
    assert "time_limit" not in str(raised.value)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"msg": "no"}, TypeError, r"solver_options\['msg'\] must be True or False"),
        ({"msg": 1}, TypeError, r"solver_options\['msg'\] must be True or False"),
    ],
)
def test_msg_must_be_a_bool(options, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=3, solver_options=options)


def test_huge_time_limit_means_no_limit() -> None:
    assert omars_ilp._solver_settings({"time_limit": 10**400}).time_limit == math.inf


@pytest.mark.parametrize("excluded", [[], [-1], [99], [1.7], [[0, 1]]])
def test_exclude_solutions_are_validated(excluded) -> None:
    with pytest.raises(ValueError, match="exclude_solutions entries must be non-empty lists"):
        omars_ilp.solve_omars_ilp(omars_ilp._half_pool(3), n_half=3, exclude_solutions=[excluded])


def test_generate_omars_rejects_categorical_factor() -> None:
    """A categorical factor gets a clear error, not a downstream TypeError.

    OMARS is built from three-level quantitative contrasts, so a categorical
    factor is out of scope; the message names the offending factor and points to
    the optimal-design path for mixed-level studies.
    """
    from process_improve.experiments import generate_omars

    factors = [
        Factor(name="supplier", type="categorical", levels=["A", "B", "C"]),
        Factor(name="temp", low=-1, high=1),
        Factor(name="time", low=-1, high=1),
    ]
    with pytest.raises(ValueError, match="OMARS designs require continuous factors"):
        generate_omars(factors, solver_options=_SOLVER)


# ---------------------------------------------------------------------------
# Search speed-ups: estimability cuts, symmetry dedupe, plateau, local search
# ---------------------------------------------------------------------------


def _main_effects_orthogonal(half: np.ndarray) -> bool:
    gram = half.T @ half
    return bool(np.all(gram == np.diag(np.diag(gram))))


@pytest.mark.parametrize("model", ["full_second_order", "main_quadratic"])
def test_estimability_cuts_exclude_the_deficient_selection_and_keep_estimable_ones(model: str) -> None:
    """Every cut is violated by the rank-deficient selection it came from, and met by estimable designs."""
    pool = omars_ilp._half_pool(5)
    features = omars_ilp._even_features(pool, model)
    deficient = list(_RANK_DEFICIENT_K5)
    rank = np.linalg.matrix_rank(features[deficient])
    cuts = omars_ilp._estimability_cuts(features, deficient)
    if rank == features.shape[1]:
        assert cuts == []
        return
    assert len(cuts) == features.shape[1] - rank
    for cut in cuts:
        assert not set(cut) & set(deficient)
    estimable = generate_omars_selection(5, n_runs=33, model=model)
    assert np.linalg.matrix_rank(features[estimable]) == features.shape[1]
    for cut in [*cuts, *omars_ilp._distinct_feature_cuts(features, 5)]:
        assert set(cut) & set(estimable)


def generate_omars_selection(k: int, *, n_runs: int, model: str) -> list[int]:
    """Half-pool rows of a generated multistart design."""
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(k), n_runs=n_runs, model=model, n_restarts=1, solver_options=_SOLVER)
    pool = omars_ilp._half_pool(k)
    coded = _coded(result)
    half = coded[np.abs(coded).sum(axis=1) > 0]
    rows = {tuple(r): i for i, r in enumerate(pool)}
    return sorted({rows[tuple(r)] for r in half if tuple(r) in rows})


def test_full_rank_selection_gives_no_estimability_cut() -> None:
    pool = omars_ilp._half_pool(4)
    features = omars_ilp._even_features(pool, "main_quadratic")
    assert omars_ilp._estimability_cuts(features, list(range(len(pool)))) == []


def test_cover_cut_is_enforced_and_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = omars_ilp._half_pool(3)
    _, _, first = omars_ilp.solve_omars_ilp(pool, n_half=4, solver_options=_SOLVER)
    others = [r for r in range(len(pool)) if r not in first]
    design, _, chosen = omars_ilp.solve_omars_ilp(pool, n_half=4, require_any=[others], solver_options=_SOLVER)
    assert design is not None
    assert set(chosen) & set(others)
    with pytest.raises(ValueError, match="require_any entries must be non-empty"):
        omars_ilp.solve_omars_ilp(pool, n_half=4, require_any=[[]])
    with pytest.raises(RuntimeError, match="none of the required rows"):
        omars_ilp._check_selection(pool, first, (4, 4), None, [others])


def _equivalent_forms(design: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Return the design with its runs shuffled, its factors permuted, and some factors sign-flipped."""
    forms = []
    for _ in range(5):
        signs = rng.choice([-1.0, 1.0], size=design.shape[1])
        forms.append(design[rng.permutation(design.shape[0])][:, rng.permutation(design.shape[1])] * signs)
    return forms


def test_canonical_key_is_shared_by_equivalent_designs() -> None:
    from process_improve.experiments import generate_omars

    rng = np.random.default_rng(7)
    design = omars_ilp._foldover(omars_ilp._half_pool(5)[_RANK_DEFICIENT_K5])
    key = omars_ilp._canonical_key(design)
    assert all(omars_ilp._canonical_key(form) == key for form in _equivalent_forms(design, rng))
    other = _coded(generate_omars(_factors(5), n_runs=31, n_restarts=3, solver_options=_SOLVER))
    assert omars_ilp._canonical_key(other) != key


def test_equal_canonical_keys_mean_equivalent_designs() -> None:
    """The key encodes a transformed copy of the design, which scores the same as the design."""
    rng = np.random.default_rng(3)
    pool = omars_ilp._half_pool(4)
    for _ in range(20):
        design = omars_ilp._foldover(pool[rng.choice(len(pool), size=8, replace=False)])
        codes = np.frombuffer(omars_ilp._canonical_key(design), dtype=np.int64)
        # Balanced-ternary run codes; shifting by sum(3**j) makes the digits 0, 1, 2.
        decoded = (((codes[:, None] + 40) // 3 ** np.arange(3, -1, -1)) % 3 - 1).astype(float)
        assert decoded.shape == design.shape
        for metric in (omars_ilp._d_efficiency, omars_ilp._a_optimality):
            assert metric(decoded, "main_quadratic") == pytest.approx(metric(design, "main_quadratic"))
        assert omars_ilp._max_second_order_correlation_metric(decoded) == pytest.approx(
            omars_ilp._max_second_order_correlation_metric(design)
        )


def _candidate(d: float, corr: float, a: float = 1.0) -> omars_ilp._Candidate:
    return omars_ilp._Candidate(np.zeros((1, 3)), 17, [], d, a, corr, "Optimal")


@pytest.mark.parametrize("criterion", ["dominance", "d_efficiency", "a_optimal"])
def test_infinite_correlation_ranks_last(criterion: str) -> None:
    """A design with a constant second-order column wins only when nothing else was found."""
    degenerate = _candidate(40.0, math.inf, a=0.5)
    finite = _candidate(30.0, 0.6)
    assert omars_ilp._select([degenerate, finite], criterion) is finite
    assert omars_ilp._select([degenerate], criterion) is degenerate


def test_exhaustive_winner_ranks_infinite_correlation_last() -> None:
    d_eff = np.array([40.0, 30.0])
    a_opt = np.array([0.5, 1.0])
    max_corr = np.array([np.inf, 0.6])
    for criterion in ("dominance", "d_efficiency", "a_optimal"):
        assert omars_ilp._pick_exhaustive_winner(d_eff, a_opt, max_corr, criterion) == 1


def test_improves_front() -> None:
    retained = [_candidate(35.0, 0.6), _candidate(33.0, 0.5)]
    assert omars_ilp._improves_front(_candidate(36.0, 0.7), retained)
    assert omars_ilp._improves_front(_candidate(34.0, 0.4), retained)
    assert not omars_ilp._improves_front(_candidate(34.0, 0.6), retained)
    assert not omars_ilp._improves_front(_candidate(35.0, 0.6), retained)


def test_swap_index_finds_exactly_the_orthogonal_swaps() -> None:
    """The hashed look-ups return every one-, two- and three-run swap that keeps orthogonality, and no other."""
    pool = omars_ilp._half_pool(4)
    _, _, selection = omars_ilp.solve_omars_ilp(pool, n_half=8, solver_options=_SOLVER)
    outside = [r for r in range(len(pool)) if r not in selection]

    def brute_force(sizes: tuple[int, ...]) -> set[tuple[int, ...]]:
        found = set()
        for size in sizes:
            for out in itertools.combinations(selection, size):
                for into in itertools.combinations(outside, size):
                    move = sorted([r for r in selection if r not in out] + list(into))
                    if _main_effects_orthogonal(pool[move]):
                        found.add(tuple(move))
        return found

    index = omars_ilp._SwapIndex(pool)
    swaps = index.swaps(selection)
    assert swaps == sorted(swaps)
    assert set(map(tuple, swaps)) == brute_force((1, 2))
    assert set(map(tuple, index.triple_swaps(selection))) == brute_force((3,))


def test_budget_search_uses_the_plateau_rule() -> None:
    """The run-budget path stops at a plateau; generate_omars keeps its exact restart budget."""
    result = generate_design(_factors(5), design_type="omars_ilp", budget=21, random_state=42)
    report = result.metadata["omars_search"]
    assert report.plateau == omars_ilp._PLATEAU
    assert report.n_restarts == omars_ilp._PLATEAU_MAX_RESTARTS
    assert report.ilp_iterations < 2 + omars_ilp._PLATEAU_MAX_RESTARTS
    assert is_omars(_coded(result))
    assert result.metadata["model_rank"] == result.metadata["model_params"]
    from process_improve.experiments import generate_omars

    plain = generate_omars(_factors(5), n_runs=25, model="main_quadratic", n_restarts=2, solver_options=_SOLVER)
    assert plain.metadata["omars_search"].plateau is None


@pytest.mark.slow
def test_budget_search_is_deterministic() -> None:
    first = generate_design(_factors(6), design_type="omars_ilp", budget=17, random_state=5)
    second = generate_design(_factors(6), design_type="omars_ilp", budget=17, random_state=5)
    np.testing.assert_array_equal(_coded(first), _coded(second))
