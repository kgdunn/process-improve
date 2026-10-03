"""Categorical factors in effect (sum-to-zero) coding, on the design side.

``evaluate_design`` and the built-in optimal-design exchange code a categorical factor
the same way. The coding-dependent metrics (A, E, K, VIF, power) are checked against an
explicit numpy model matrix, and the coding-invariant ones (prediction variance, G,
FDS, degrees of freedom) are pinned to the values measured before the change.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, evaluate_design, generate_design


def _fixed_design() -> pd.DataFrame:
    """Return a 2 x 2 x 3 factorial in A, B and a three-level categorical C, plus three unbalancing runs."""
    rows = [(a, b, c) for a, b, c in itertools.product([-1.0, 1.0], [-1.0, 1.0], ["lo", "mid", "hi"])]
    rows += [(1.0, -1.0, "lo"), (-1.0, 1.0, "hi"), (0.0, 0.0, "mid")]
    return pd.DataFrame(rows, columns=["A", "B", "C"])


def _effect_columns(labels: list) -> np.ndarray:
    """Effect coding of one categorical column, written out independently of the library.

    Levels in sorted order. Two levels: the first is -1 and the second +1. Three or more:
    level ``j`` is +1 in column ``j`` and the last level is -1 in every column.
    """
    levels = sorted(set(labels))
    index = np.array([levels.index(v) for v in labels])
    if len(levels) == 2:
        return np.where(index == 0, -1.0, 1.0)[:, None]
    return np.vstack([np.eye(len(levels) - 1), -np.ones((1, len(levels) - 1))])[index]


def _effect_model_matrix(design: pd.DataFrame) -> np.ndarray:
    """Intercept, A, B, C and the two-factor interactions, with C effect-coded."""
    a = design["A"].to_numpy(dtype=float)
    b = design["B"].to_numpy(dtype=float)
    c = _effect_columns(list(design["C"]))
    return np.column_stack([np.ones(len(a)), a, b, c, a * b, a[:, None] * c, b[:, None] * c])


def _factors(levels: list[str]) -> list[Factor]:
    return [
        Factor(name="A", low=0, high=1),
        Factor(name="B", low=0, high=1),
        Factor(name="C", type="categorical", levels=levels),
    ]


# ---------------------------------------------------------------------------
# Metrics that do not depend on the coding keep their values
# ---------------------------------------------------------------------------


class TestCodingInvariantMetrics:
    def test_prediction_variance_metrics_and_degrees_of_freedom_are_pinned(self) -> None:
        out = evaluate_design(
            _fixed_design(),
            model="interactions",
            metric=["average_prediction_variance", "g_efficiency", "fds", "degrees_of_freedom"],
            n_samples=5000,
        )
        assert out["average_prediction_variance"] == pytest.approx(0.3692822142386848, rel=1e-9)
        assert out["g_efficiency"] == pytest.approx(81.93384223918574, rel=1e-9)
        assert out["fds"]["max_prediction_variance"] == pytest.approx(0.813664596273292, rel=1e-9)
        assert out["degrees_of_freedom"] == {
            "model": 9,
            "residual": 5,
            "total": 14,
            "pure_error": 2,
            "lack_of_fit": 3,
        }


# ---------------------------------------------------------------------------
# Treatment coding: the values evaluate_design gave before effect coding
# ---------------------------------------------------------------------------


class TestTreatmentCoding:
    def test_treatment_coding_values_are_pinned(self) -> None:
        out = evaluate_design(
            _fixed_design(),
            model="interactions",
            metric=["d_efficiency", "a_optimality", "e_optimality", "condition_number", "vif", "power"],
            effect_size=1.0,
        )
        assert out["d_efficiency"] == pytest.approx(34.35028327680051, rel=1e-9)
        assert out["a_optimality"] == pytest.approx(3.3692546583850933, rel=1e-9)
        assert out["e_optimality"] == pytest.approx(1.0717967697244946, rel=1e-9)
        assert out["condition_number"] == pytest.approx(4.42277022639572, rel=1e-9)
        assert out["vif"]["C[T.lo]"] == pytest.approx(1.4285714285714288, rel=1e-9)
        assert out["vif"]["A:C[T.mid]"] == pytest.approx(1.863354037267081, rel=1e-9)
        assert out["power"]["C[T.lo]"] == pytest.approx(0.23836843380200268, rel=1e-9)
        assert out["power"]["A"] == pytest.approx(0.4143624547309154, rel=1e-9)


# ---------------------------------------------------------------------------
# Quality of the built-in exchange's designs, judged in effect coding
# ---------------------------------------------------------------------------

#: Five-seed means (seeds 0-4, 16 runs, interactions model) measured on the treatment-coded
#: exchange, each design scored in effect coding. Effect coding changes which designs the
#: A-, E- and K-criteria prefer, so the new designs must be no worse on these numbers.
#: "lo mid hi" is declared out of sorted order, so the reference level is not the last declared.
_EXCHANGE_BASELINE = {
    ("lo mid hi", "a_optimal"): 1.0625,
    ("lo mid hi", "e_optimal"): 5.071796769724491,
    ("lo mid hi", "k_optimal"): 3.797273352909202,
    ("lo mid hi", "d_optimal"): 23.958672110309426,
    ("x y", "a_optimal"): 0.4375,
    ("x y", "e_optimal"): 16.0,
    ("x y", "k_optimal"): 1.0,
    ("x y", "d_optimal"): 19.408121055678468,
}

#: The prediction-variance criteria do not depend on the coding (same seeds and size).
_INVARIANT_BASELINE = {
    ("lo mid hi", "i_optimal"): ("trace_criterion", 0.345413968034522),
    ("lo mid hi", "g_optimal"): ("max_prediction_variance", 0.8045112781954888),
    ("x y", "i_optimal"): ("trace_criterion", 0.2153264006629351),
    ("x y", "g_optimal"): ("max_prediction_variance", 0.4375),
}

_SEEDS = range(5)


def _exchange_designs(levels: str, criterion: str) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            generate_design(_factors(levels.split()), design_type=criterion, budget=16, random_state=seed)
            for seed in _SEEDS
        ]


def _effect_score(design: pd.DataFrame, criterion: str) -> float:
    x = _effect_model_matrix(design)
    info = x.T @ x
    eig = np.linalg.eigvalsh(info)
    return {
        "a_optimal": float(np.trace(np.linalg.inv(info))),
        "e_optimal": float(eig.min()),
        "k_optimal": float(eig.max() / eig.min()),
        "d_optimal": float(np.linalg.slogdet(info)[1]),
    }[criterion]


@pytest.mark.parametrize(("levels", "criterion"), list(_EXCHANGE_BASELINE))
def test_exchange_designs_are_no_worse_in_effect_coding(levels: str, criterion: str) -> None:
    designs = _exchange_designs(levels, criterion)
    mean = float(np.mean([_effect_score(r.design[["A", "B", "C"]], criterion) for r in designs]))
    baseline = _EXCHANGE_BASELINE[(levels, criterion)]
    if criterion in ("a_optimal", "k_optimal"):
        assert mean <= baseline * (1 + 1e-9)
    else:
        assert mean >= baseline * (1 - 1e-9)


@pytest.mark.parametrize(("levels", "criterion"), list(_INVARIANT_BASELINE))
def test_exchange_prediction_variance_criteria_are_no_worse(levels: str, criterion: str) -> None:
    key, baseline = _INVARIANT_BASELINE[(levels, criterion)]
    mean = float(np.mean([r.metadata[key] for r in _exchange_designs(levels, criterion)]))
    assert mean <= baseline * (1 + 1e-9)
