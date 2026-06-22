"""Tests for the ILP-based OMARS design generator (generate_omars)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from process_improve.experiments import Factor, analyze_omars, generate_design
from process_improve.experiments.designs_omars import is_omars

_HAS_PULP = importlib.util.find_spec("pulp") is not None
pytestmark = pytest.mark.skipif(not _HAS_PULP, reason="pulp (the 'ilp' extra) is not installed")

# Keep the solver fast and deterministic across the suite.
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


@pytest.mark.parametrize("k", [3, 4])
def test_generated_design_is_omars(k: int) -> None:
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(k), solver_options=_SOLVER)
    assert is_omars(_coded(result))
    assert result.metadata["omars_verified"] is True
    assert result.metadata["family"] == "omars_ilp"


@pytest.mark.parametrize("k", [3, 4])
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
    assert is_omars(_coded(result))


def test_run_size_below_parameters_raises() -> None:
    from process_improve.experiments import generate_omars

    # k=3: full second-order model has 1 + 6 + 3 = 10 params; n_runs must exceed it.
    with pytest.raises(ValueError, match="error degrees of freedom"):
        generate_omars(_factors(3), n_runs=10, solver_options=_SOLVER)


def test_even_run_size_raises() -> None:
    from process_improve.experiments import generate_omars

    # Foldover designs have an odd run count (2*h + 1).
    with pytest.raises(ValueError, match="must be odd"):
        generate_omars(_factors(3), n_runs=16, solver_options=_SOLVER)


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

    a = generate_omars(_factors(3), random_seed=7, solver_options=_SOLVER)
    b = generate_omars(_factors(3), random_seed=7, solver_options=_SOLVER)
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
    # The search retains many distinct designs, not the handful the old no-good cuts found.
    assert report.feasible_designs > 1


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


def test_multistart_is_deterministic_for_seed() -> None:
    """The multistart path reproduces the same design for a fixed seed (k=5)."""
    from process_improve.experiments import generate_omars

    a = generate_omars(
        _factors(5), n_runs=25, model="main_quadratic", n_restarts=20, random_seed=1, solver_options=_SOLVER
    )
    b = generate_omars(
        _factors(5), n_runs=25, model="main_quadratic", n_restarts=20, random_seed=1, solver_options=_SOLVER
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


def test_registry_integration() -> None:
    result = generate_design(_factors(4), design_type="omars_ilp", budget=21)
    assert is_omars(result.design[result.factor_names].to_numpy(dtype=float))
    assert result.metadata["n_runs_selected"] == 21


def test_omars_with_budget_reaches_ilp() -> None:
    """design_type="omars" with a budget routes to the ILP enumerator."""
    result = generate_design(_factors(4), design_type="omars", budget=21, center_points=0)
    assert result.metadata["family"] == "omars_ilp"
    assert result.metadata["n_runs_selected"] == 21
    assert is_omars(result.design[result.factor_names].to_numpy(dtype=float))


def test_omars_without_budget_is_minimal_foldover() -> None:
    """design_type="omars" with no budget keeps the minimal conference-foldover member."""
    result = generate_design(_factors(4), design_type="omars", center_points=0)
    assert result.metadata["family"] == "conference_foldover"
    assert result.n_runs == 9  # the minimal four-factor member (the DSD)


def test_missing_solver_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from process_improve.experiments import designs_omars_ilp as module

    monkeypatch.setattr(module, "_PULP_AVAILABLE", False)
    pool = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ImportError, match="ilp"):
        module.solve_omars_ilp(pool, n_half=1)


def test_search_report_records_diagnostics() -> None:
    from process_improve.experiments import generate_omars

    result = generate_omars(_factors(3), solver_options=_SOLVER)
    report = result.metadata["omars_search"]
    assert report.n_factors == 3
    assert report.half_pool_size == (3**3 - 1) // 2
    assert report.ilp_iterations >= 1
    assert report.feasible_designs >= 1
    assert report.total_solve_seconds >= 0.0
