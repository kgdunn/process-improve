"""Box and Hunter's (1957) rules for the centre runs of a rotatable central composite design (#788 A11).

Their table, reprinted by Montgomery (Table 11.11) and Myers, Montgomery and Anderson-Cook, gives
the centre runs for uniform precision and for orthogonality. These tests hold the rules to that
table and, independently, to what each rule means: the centred squared columns orthogonal, or the
prediction variance at the centre equal to that one unit away.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from process_improve.experiments import Factor, generate_design
from process_improve.experiments.designs_response_surface import ccd_center_points

# (factors, cube runs, uniform precision, orthogonal): Box and Hunter's table.
TABLE = [(2, 4, 5, 8), (3, 8, 6, 9), (4, 16, 7, 12), (5, 16, 6, 10), (5, 32, 10, 17), (6, 32, 9, 15), (6, 64, 15, 24)]


def composite(k: int, n_center_points: int | str, **options):
    factors = [Factor(name=name, low=-1, high=1) for name in "ABCDEFG"[:k]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        design = generate_design(factors, design_type="ccd", n_center_points=n_center_points, random_state=0, **options)
    return design, design.design[list(design.factor_names)].to_numpy(dtype=float)


def quadratic(x: np.ndarray) -> np.ndarray:
    pairs = itertools.combinations(range(x.shape[1]), 2)
    return np.column_stack([np.ones(len(x)), x, *[x[:, i] * x[:, j] for i, j in pairs], x**2])


def prediction_variance(runs: np.ndarray, point: np.ndarray) -> float:
    """N f'(X'X)^-1 f, the scaled prediction variance of the full quadratic model at one point."""
    model, f = quadratic(runs), quadratic(point[None, :])[0]
    return float(len(runs) * f @ np.linalg.solve(model.T @ model, f))


@pytest.mark.parametrize(("k", "cube_runs", "uniform", "orthogonal"), TABLE)
def test_the_rules_give_box_and_hunter_s_table(k, cube_runs, uniform, orthogonal):
    assert ccd_center_points("uniform_precision", cube_runs, k) == uniform
    assert ccd_center_points("orthogonal", cube_runs, k) == orthogonal


@pytest.mark.parametrize(("k", "cube"), [(2, "full"), (3, "full"), (4, "full"), (5, "fractional")])
def test_the_orthogonal_rule_makes_the_squared_columns_orthogonal(k, cube):
    """Exactly where the rule's number is whole (all but three factors); else as nearly as a whole run can."""

    def correlation(n_center: int) -> float:
        squares = composite(k, n_center, cube=cube, alpha="rotatable")[1] ** 2
        squares -= squares.mean(axis=0)
        return float(squares[:, 0] @ squares[:, 1] / (squares[:, 0] @ squares[:, 0]))

    _, runs = composite(k, "orthogonal", cube=cube)
    n_center = int(np.sum(~runs.any(axis=1)))
    assert n_center == ccd_center_points("orthogonal", 2 ** (k - (cube == "fractional")), k)
    gaps = [abs(correlation(n)) for n in (n_center - 1, n_center, n_center + 1)]
    assert gaps[1] == min(gaps)
    if k != 3:
        assert gaps[1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(("k", "cube"), [(2, "full"), (3, "full"), (4, "full"), (5, "fractional")])
def test_the_uniform_precision_rule_balances_the_centre_and_one_unit_away(k, cube):
    """The variance at the centre and one root-mean-square unit away are nearest equal at the rule's number."""

    def gap(n_center: int) -> float:
        runs = composite(k, n_center, cube=cube, alpha="rotatable")[1]
        unit = np.sqrt(np.mean(runs[:, 0] ** 2))
        return abs(prediction_variance(runs, np.zeros(k)) - prediction_variance(runs, unit * np.eye(k)[0]))

    _, runs = composite(k, "uniform_precision", cube=cube)
    n_center = int(np.sum(~runs.any(axis=1)))
    gaps = [gap(n) for n in (n_center - 1, n_center, n_center + 1)]
    assert gaps[1] == min(gaps)


def test_a_rule_makes_the_design_rotatable():
    design, runs = composite(4, "uniform_precision")
    assert design.alpha == pytest.approx(16**0.25)
    assert int(np.sum(~runs.any(axis=1))) == 7
    assert len(runs) == 16 + 8 + 7


def test_replicated_cube_and_axial_runs_need_more_centre_runs():
    assert ccd_center_points("orthogonal", 16, 4, n_replicates=2) == 24
    _, runs = composite(4, "orthogonal", n_replicates=2)
    assert int(np.sum(~runs.any(axis=1))) == 24


@pytest.mark.parametrize(
    ("design_type", "options", "message"),
    [
        ("full_factorial", {}, "names a rule for a central composite design"),
        ("ccd", {"alpha": "face_centered"}, "rule for a rotatable CCD in one block"),
        ("ccd", {"n_blocks": 2}, "rule for a rotatable CCD in one block"),
    ],
)
def test_a_rule_where_it_does_not_hold_is_refused(design_type, options, message):
    factors = [Factor(name=name, low=-1, high=1) for name in "ABC"]
    with pytest.raises(ValueError, match=message):
        generate_design(factors, design_type=design_type, n_center_points="orthogonal", **options)


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="rule must be one of"):
        ccd_center_points("rotatable", 16, 4)
