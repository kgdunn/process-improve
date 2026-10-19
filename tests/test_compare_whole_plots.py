"""compare_designs on a split-plot design: its runs share whole plots, so the measures use X'V^-1 X (#790 C2)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, compare_designs, evaluate_design, generate_design
from process_improve.experiments.compare import _expand, _model_exponents


def _split_plot() -> pd.DataFrame:
    """Return a 2^3 factorial twice over, A hard to change: four whole plots, each at one level of A."""
    rows = [(plot, a, b, c) for plot, a in enumerate([-1, 1, -1, 1], start=1) for b in (-1, 1) for c in (-1, 1)]
    return pd.DataFrame(rows, columns=["WholePlot", "A", "B", "C"])


def _gls(runs: pd.DataFrame, model: str, ratio: float) -> np.ndarray:
    """Return X'V^-1 X with V = I + ratio Z Z', built out in full."""
    names = [c for c in runs.columns if c != "WholePlot"]
    x = _expand(runs[names].to_numpy(dtype=float), _model_exponents(len(names), model))
    z = pd.get_dummies(runs["WholePlot"]).to_numpy(dtype=float)
    return x.T @ np.linalg.solve(np.eye(len(runs)) + ratio * z @ z.T, x)


def test_a_split_plot_design_is_scored_on_x_v_inverse_x():
    runs = _split_plot()
    runs.attrs["whole_plot_variance_ratio"] = 1.0
    found = compare_designs({"split-plot": runs}, model="interactions")
    values = found.values["split-plot [n=16]"]
    information = _gls(runs, "interactions", 1.0)
    p = information.shape[0]
    assert values["d"] == pytest.approx(np.linalg.det(information) ** (1 / p))
    assert values["a"] == pytest.approx(np.trace(np.linalg.inv(information)) / p)
    assert values["e"] == pytest.approx(np.linalg.eigvalsh(information).min())
    assert values["residual_df"] == 16 - p, "the run-level residual degrees of freedom"
    [note] = [n for n in found.notes["split-plot [n=16]"] if "whole plots" in n]
    assert note.startswith("Its runs share 4 whole plots, with a whole-plot variance 1 times the run variance")


def test_the_whole_plot_factor_loses_precision():
    """A is set once per whole plot, so its coefficient varies more; B, balanced within each whole plot, does not."""
    runs = _split_plot()
    independent = compare_designs({"runs": runs}, model="main_effects").values["runs [n=16]"]
    runs.attrs["whole_plot_variance_ratio"] = 1.0
    split = compare_designs({"runs": runs}, model="main_effects").values["runs [n=16]"]
    inverse = np.linalg.inv(_gls(runs, "main_effects", 1.0))
    assert inverse[1, 1] > 1 / 16, "A, against the 1/16 of independent runs"
    assert inverse[2, 2] == pytest.approx(1 / 16), "B"
    assert split["power_main"] < independent["power_main"], "the power is the least precise main effect's"


def test_a_ratio_of_zero_scores_the_runs_as_independent():
    runs = _split_plot()
    plain = compare_designs({"runs": runs}, model="interactions").values["runs [n=16]"]
    runs.attrs["whole_plot_variance_ratio"] = 0.0
    zero = compare_designs({"runs": runs}, model="interactions").values["runs [n=16]"]
    for key in ("a", "i", "e", "g", "d", "power_main", "power_interaction"):
        assert zero[key] == pytest.approx(plain[key]), key


def test_whole_plots_with_no_recorded_ratio_are_noted():
    found = compare_designs({"runs": _split_plot()}, model="main_effects")
    assert any("records no whole-plot variance ratio" in note for note in found.notes["runs [n=16]"])


@pytest.mark.parametrize("ratio", [-0.5, float("nan"), float("inf"), "1"])
def test_a_ratio_that_is_not_a_variance_ratio_is_refused(ratio):
    runs = _split_plot()
    runs.attrs["whole_plot_variance_ratio"] = ratio
    with pytest.raises(ValueError, match="whole-plot variance ratio must be a finite number"):
        compare_designs({"runs": runs}, model="main_effects")


def test_a_split_plot_design_from_generate_design_is_scored_with_its_own_ratio():
    """generate_design records the ratio it made the design for; compare_designs and evaluate_design agree."""
    pytest.importorskip("pyoptex")
    factors = [Factor(name=n, low=-1, high=1) for n in "ABCD"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pyoptex's own progress output
        design = generate_design(factors, "d_optimal", budget=16, hard_to_change=["A"], random_state=1)
    ratio = design.metadata["whole_plot_variance_ratio"]
    values = compare_designs({"split-plot": design}, model="main_effects").values["split-plot [n=16]"]
    evaluated = evaluate_design(
        design, model="main_effects", random_effects={"WholePlot": ratio}, metric="a_optimality"
    )
    assert values["a"] * 5 == pytest.approx(evaluated["a_optimality"]), "A/p against the trace of (X'V^-1 X)^-1"
