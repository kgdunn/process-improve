"""A design evaluated on its projections: the projections that estimate a model, and the metrics' averages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, evaluate_design, evaluate_projections, generate_design
from process_improve.experiments.projections import MOST_PROJECTIONS, projection_formula


def dsd(k: int, n_center_points: int = 3) -> pd.DataFrame:
    """Return the library's definitive screening design in k factors, its factor columns only."""
    factors = [Factor(name="ABCDEFGHJ"[i], low=-1, high=1) for i in range(k)]
    made = generate_design(factors, design_type="dsd", n_center_points=n_center_points)
    return made.design[list(made.factor_names)].astype(float)


def test_every_three_factor_projection_of_a_six_factor_dsd_fits_the_full_quadratic() -> None:
    found = evaluate_projections(dsd(6), 3, model="quadratic")
    assert (found["n_projections"], found["n_estimable"], found["fraction_estimable"]) == (20, 20, 1.0)


def test_the_average_of_a_metric_is_over_the_estimable_projections() -> None:
    design = dsd(6)
    found = evaluate_projections(design, 3, model="quadratic", metric=["d_efficiency", "g_efficiency"])
    one = next(p for p in found["projections"] if p["factors"] == ["A", "B", "C"])
    direct = evaluate_design(design[["A", "B", "C"]], model=projection_formula(design, ["A", "B", "C"], "quadratic"))
    assert one["metrics"]["d_efficiency"] == pytest.approx(direct["d_efficiency"])
    assert found["mean"]["d_efficiency"] == pytest.approx(
        np.mean([p["metrics"]["d_efficiency"] for p in found["projections"]])
    )
    assert "max_prediction_variance" in found["mean"]


def test_a_two_level_factor_has_no_square_in_a_projection() -> None:
    frame = pd.DataFrame({"Q": [-1.0, 0.0, 1.0, 0.0], "C": [-1.0, 1.0, -1.0, 1.0]})
    assert projection_formula(frame, ["Q", "C"], "quadratic") == "(Q + C) ** 2 + I(Q ** 2)"
    assert projection_formula(frame, ["Q"], "interactions") == "Q"


def test_projections_drawn_from_groups_of_factors() -> None:
    """Any two of the three-level factors with any one of the two-level ones: 3 x 2 projections."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.choice([-1.0, 0.0, 1.0], size=(30, 3)), columns=["Q1", "Q2", "Q3"])
    frame[["C1", "C2"]] = rng.choice([-1.0, 1.0], size=(30, 2))
    groups = {"quantitative": ["Q1", "Q2", "Q3"], "categorical": ["C1", "C2"]}
    found = evaluate_projections(frame, {"quantitative": 2, "categorical": 1}, model="quadratic", groups=groups)
    assert found["n_projections"] == 6
    assert all(len(p["factors"]) == 3 for p in found["projections"])


def test_a_projection_that_cannot_fit_the_model_is_counted_out() -> None:
    """A 2^3 cannot fit squares: as a two-level design its quadratic projections drop them, so all fit."""
    cube = pd.DataFrame(
        np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T, columns=list("ABC"), dtype=float
    )
    assert evaluate_projections(cube, 2, model="quadratic")["n_estimable"] == 3
    half = cube[cube.prod(axis=1) == 1]
    assert evaluate_projections(half, 3, model="interactions")["n_estimable"] == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"size": 2, "model": "cubic"}, "model must be one of"),
        ({"size": {"a": 1}}, "needs the groups"),
        ({"size": {"a": 1}, "groups": {"b": ["A"]}}, "not in groups"),
    ],
)
def test_bad_requests_are_refused(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_projections(dsd(4), **kwargs)


def test_too_many_projections_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("process_improve.experiments.projections.MOST_PROJECTIONS", 3)
    with pytest.raises(ValueError, match="projections is more than"):
        evaluate_projections(dsd(4), 2)
    assert MOST_PROJECTIONS == 50_000
