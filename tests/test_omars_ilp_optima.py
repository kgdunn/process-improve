"""Regression tests pinning generate_omars selection criteria to exhaustive optima.

The pinned values are the exhaustively computed optima reported in issues #497
(A-optimality), #498 (D) and #499 (maximum second-order correlation), verified
there by two independent enumerators.  Every cell here falls inside the
exhaustive-search regime of the generator, so the returned design must match
the optimum exactly; a regression to a heuristic (or a broken enumeration)
shows up as a worse metric.

These cells run on the enumeration path with a pinned ``n_runs``, which needs
no ILP solve, so the tests do not require pulp or a working CBC binary.
"""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import Factor, generate_omars


def _factors(k: int) -> list[Factor]:
    return [Factor(name=chr(65 + i), low=-1, high=1) for i in range(k)]


def _design(result) -> np.ndarray:
    return result.design[result.factor_names].to_numpy(dtype=float)


def _main_quadratic_matrix(design: np.ndarray) -> np.ndarray:
    k = design.shape[1]
    columns = [np.ones(design.shape[0])]
    columns += [design[:, i] for i in range(k)]
    columns += [design[:, i] ** 2 for i in range(k)]
    return np.column_stack(columns)


def _a_value(design: np.ndarray) -> float:
    """Compute A = trace((X'X)^-1) / p for the main-effects-and-quadratics model, as in #497."""
    model_matrix = _main_quadratic_matrix(design)
    gram = model_matrix.T @ model_matrix
    return float(np.trace(np.linalg.inv(gram)) / gram.shape[0])


def _d_value(design: np.ndarray) -> float:
    """Compute D = det(X'X)^(1/p) for the main-effects-and-quadratics model, as in #498."""
    model_matrix = _main_quadratic_matrix(design)
    gram = model_matrix.T @ model_matrix
    return float(np.exp(np.linalg.slogdet(gram)[1] / gram.shape[0]))


def _max_abs_second_order_correlation(design: np.ndarray) -> float:
    """Compute max |r| over all second-order columns, as in #499 (no constant columns expected)."""
    k = design.shape[1]
    columns = [design[:, i] * design[:, i] for i in range(k)]
    columns += [design[:, i] * design[:, j] for i in range(k) for j in range(i + 1, k)]
    second_order = np.column_stack(columns)
    corr = np.abs(np.corrcoef(second_order, rowvar=False))
    np.fill_diagonal(corr, 0.0)
    return float(corr.max())


def _generate(k: int, n_runs: int, center_runs: int, criterion: str):
    result = generate_omars(
        _factors(k),
        n_runs=n_runs,
        center_runs=center_runs,
        model="main_quadratic",
        selection_criterion=criterion,
    )
    design = _design(result)
    assert design.shape[0] == n_runs
    assert int(np.sum(np.all(design == 0, axis=1))) == center_runs
    assert result.metadata["search_mode"] == "exhaustive"
    return design


# Exhaustive optima from the table in issue #497.
_A_OPTIMA = [
    (3, 13, 1, 0.273810),
    (3, 15, 1, 0.217262),
    (3, 17, 1, 0.198649),
    (3, 19, 1, 0.174286),
    (3, 21, 1, 0.152015),
    (3, 15, 3, 0.217262),
    (3, 17, 3, 0.183929),
    (4, 13, 1, 0.279630),
    (4, 15, 1, 0.227273),
    (4, 17, 1, 0.187831),
    (4, 17, 3, 0.187831),
]

_A_OPTIMA_SLOW = [
    (4, 19, 1, 0.174411),
    (4, 21, 1, 0.155556),
]


@pytest.mark.parametrize(("k", "n_runs", "center_runs", "optimum"), _A_OPTIMA)
def test_a_optimal_matches_exhaustive_optimum(k: int, n_runs: int, center_runs: int, optimum: float) -> None:
    design = _generate(k, n_runs, center_runs, "a_optimal")
    assert _a_value(design) == pytest.approx(optimum, abs=5e-6)


@pytest.mark.slow
@pytest.mark.parametrize(("k", "n_runs", "center_runs", "optimum"), _A_OPTIMA_SLOW)
def test_a_optimal_matches_exhaustive_optimum_large(k: int, n_runs: int, center_runs: int, optimum: float) -> None:
    design = _generate(k, n_runs, center_runs, "a_optimal")
    assert _a_value(design) == pytest.approx(optimum, abs=5e-6)


# Exhaustive optima from the table in issue #498.
_D_OPTIMA = [
    (3, 17, 7.401203),
    (3, 21, 9.077880),
    (3, 23, 10.010817),
    (3, 25, 10.868684),
    (4, 15, 6.090681),
    (4, 17, 7.132557),
]

_D_OPTIMA_SLOW = [
    (4, 19, 8.058584),
    (4, 21, 8.741520),
]


@pytest.mark.parametrize(("k", "n_runs", "optimum"), _D_OPTIMA)
def test_d_efficiency_matches_exhaustive_optimum(k: int, n_runs: int, optimum: float) -> None:
    design = _generate(k, n_runs, 1, "d_efficiency")
    assert _d_value(design) == pytest.approx(optimum, abs=5e-6)


@pytest.mark.slow
@pytest.mark.parametrize(("k", "n_runs", "optimum"), _D_OPTIMA_SLOW)
def test_d_efficiency_matches_exhaustive_optimum_large(k: int, n_runs: int, optimum: float) -> None:
    design = _generate(k, n_runs, 1, "d_efficiency")
    assert _d_value(design) == pytest.approx(optimum, abs=5e-6)


# Exhaustive optima from the table in issue #499.
_CORRELATION_OPTIMA = [
    (3, 17, 0.367315),
    (3, 21, 0.050000),
    (3, 25, 0.161165),
    (4, 13, 0.500000),
    (4, 15, 0.500000),
    (4, 17, 0.367315),
]

_CORRELATION_OPTIMA_SLOW = [
    (4, 19, 0.457143),
    (4, 21, 0.311805),
]


@pytest.mark.parametrize(("k", "n_runs", "optimum"), _CORRELATION_OPTIMA)
def test_min_second_order_correlation_matches_exhaustive_optimum(k: int, n_runs: int, optimum: float) -> None:
    design = _generate(k, n_runs, 1, "min_second_order_correlation")
    assert _max_abs_second_order_correlation(design) == pytest.approx(optimum, abs=5e-6)


@pytest.mark.slow
@pytest.mark.parametrize(("k", "n_runs", "optimum"), _CORRELATION_OPTIMA_SLOW)
def test_min_second_order_correlation_matches_exhaustive_optimum_large(k: int, n_runs: int, optimum: float) -> None:
    design = _generate(k, n_runs, 1, "min_second_order_correlation")
    assert _max_abs_second_order_correlation(design) == pytest.approx(optimum, abs=5e-6)


def test_dominance_shares_the_exact_d_axis() -> None:
    """The default criterion picks its winner from the exact Pareto front (#498)."""
    design = _generate(3, 17, 1, "dominance")
    assert _d_value(design) == pytest.approx(7.401203, abs=5e-6)
