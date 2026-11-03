"""Curve responses in a designed experiment: FPCA, a model of each kept score, and bands on predicted curves."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import FunctionalFit, analyze_functional_experiment
from process_improve.experiments.reference_catalogue import reference_experiments

T = np.linspace(0, 1, 30)
PLATEAU, SLOPE = np.sin(np.pi * T), np.cos(np.pi * T)


def _truth(a: float, b: float) -> np.ndarray:
    """Return the mean curve at factor settings (a, b): two shapes whose sizes the factors set, as the model has it."""
    return (2 + 0.8 * a + 0.3 * a * b) * PLATEAU + 0.4 * b * SLOPE


def _design(replicates: int = 2) -> pd.DataFrame:
    return pd.DataFrame(list(itertools.product([-1, 0, 1], [-1, 0, 1])) * replicates, columns=["A", "B"])


def _curves(design: pd.DataFrame, seed: int, noise: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [_truth(a, b) + rng.normal(0, noise, T.size) for a, b in design[["A", "B"]].to_numpy()]
    return pd.DataFrame(rows, columns=T, index=design.index)


@pytest.fixture(scope="module")
def fitted() -> FunctionalFit:
    design = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return analyze_functional_experiment(design, _curves(design, seed=1), model="interactions")


def test_the_two_shapes_are_kept_and_each_score_is_analysed(fitted):
    assert fitted.formula == "(A + B) ** 2"
    assert fitted.components["kept"].tolist() == [True, True]
    assert fitted.components["reason"].tolist() == ["variance", "variance"]
    assert list(fitted.scores) == ["FPC 1", "FPC 2"]
    anova = fitted.scores["FPC 1"]["anova_table"]
    assert {"A", "B"} <= {row["source"] for row in anova}
    covariance = fitted.residual_covariance
    assert covariance.shape == (2, 2)
    assert covariance.to_numpy() == pytest.approx(covariance.to_numpy().T)


def test_a_small_shape_that_a_factor_drives_is_kept_though_it_carries_little_variance():
    """A third factor moves a narrow bump worth 0.1% to 0.3% of the variance: past the 99% the variance rule keeps."""
    design = pd.DataFrame(list(itertools.product([-1, 1], [-1, 1], [-1, 1])) * 2, columns=["A", "B", "C"])
    bump = np.exp(-0.5 * ((T - 0.3) / 0.05) ** 2)
    for hidden, seed in ((0.15, 3), (0.0, 3)):
        rng = np.random.default_rng(seed)
        rows = [
            (2 + 0.8 * a) * PLATEAU + 0.3 * b * SLOPE + hidden * c * bump + rng.normal(0, 0.06, T.size)
            for a, b, c in design.to_numpy()
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = analyze_functional_experiment(design, pd.DataFrame(rows, columns=T), model="main_effects")
        tested = fit.components["reason"] == "factor test"
        if hidden:
            assert tested.sum() == 1
            assert fit.components.loc[tested, "variance_ratio"].iloc[0] < 0.01
            assert len(fit.scores) == 3
        else:
            assert not tested.any()


def test_the_prediction_carries_pointwise_and_wider_simultaneous_bands(fitted):
    new = pd.DataFrame({"A": [0.5, -1.0], "B": [-0.5, 1.0]}, index=["p", "q"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bands = fitted.predict(new, n_bootstrap=60, random_state=0)
    for key in ("curve", "standard_error", "pointwise_low", "pointwise_high", "simultaneous_low", "simultaneous_high"):
        assert bands[key].shape == (2, T.size)
        assert list(bands[key].index) == ["p", "q"]
    assert (bands["pointwise_low"] <= bands["curve"]).all().all()
    simultaneous = bands["simultaneous_high"] - bands["simultaneous_low"]
    assert (simultaneous >= bands["pointwise_high"] - bands["pointwise_low"]).all().all()
    assert np.abs(bands["curve"].loc["p"].to_numpy() - _truth(0.5, -0.5)).max() < 0.05
    assert bands["level"] == 0.95
    assert bands["n_bootstrap"] == 60


def test_the_same_seed_gives_the_same_bands(fitted):
    new = pd.DataFrame({"A": [0.0], "B": [0.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first, again = (fitted.predict(new, n_bootstrap=20, random_state=3) for _ in range(2))
        other = fitted.predict(new, n_bootstrap=20, random_state=4)
    assert first["standard_error"].equals(again["standard_error"])
    assert not first["standard_error"].equals(other["standard_error"])


def test_runs_of_one_whole_plot_share_one_bootstrap_sign():
    design = _design().assign(WholePlot=np.repeat(np.arange(6), 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = analyze_functional_experiment(design, _curves(design, seed=2), model="interactions")
        bands = fit.predict(design[["A", "B"]].iloc[:1], n_bootstrap=20, random_state=0)
    assert fit._groups.tolist() == np.repeat(np.arange(6), 3).tolist()
    assert "split_plot" in fit.scores["FPC 1"]
    assert bands["curve"].shape == (1, T.size)


def test_blocks_enter_the_model_and_a_curve_is_predicted_for_the_average_block():
    design = _design().assign(Block=np.repeat([1, 2], 9))
    curves = _curves(design, seed=7) + np.where(design["Block"].to_numpy() == 2, 0.1, 0.0)[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = analyze_functional_experiment(design, curves, model="interactions")
        bands = fit.predict(pd.DataFrame({"A": [0.5], "B": [-0.5]}), n_bootstrap=20, random_state=0)
    assert fit.formula.startswith("Block1 + ")
    assert np.abs(bands["curve"].iloc[0].to_numpy() - (_truth(0.5, -0.5) + 0.05)).max() < 0.05
    with pytest.raises(ValueError, match="average over the blocks"):
        fit.predict(pd.DataFrame({"A": [0.5], "B": [-0.5], "Block": [1]}))


def test_a_model_whose_terms_the_design_does_not_hold_is_refused():
    """A factor named I is fitted by analyze_experiment under a stand-in name, which the design's columns lack."""
    design = _design().rename(columns={"A": "I"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="cannot be built from the design"):
            analyze_functional_experiment(design, _curves(_design(), seed=8), model="quadratic")


def test_curves_with_missing_points_are_analysed():
    design = _design()
    curves = _curves(design, seed=5)
    curves.iloc[::3, 5:9] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = analyze_functional_experiment(design, curves, model="interactions")
        bands = fit.predict(pd.DataFrame({"A": [0.0], "B": [0.0]}), n_bootstrap=20, random_state=0)
    assert np.isfinite(bands["curve"].to_numpy()).all()


def test_runs_that_do_not_match_or_leave_no_residual_are_refused():
    design = _design(replicates=1)
    curves = _curves(design, seed=6)
    with pytest.raises(ValueError, match="same runs"):
        analyze_functional_experiment(design, curves.iloc[::-1], model="interactions")
    corners = [0, 2, 6, 8]  # the four corners: (A + B) ** 2 has four terms, so no residual is left
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no residual degrees of freedom"):
            analyze_functional_experiment(design.iloc[corners], curves.iloc[corners], model="interactions")


@pytest.mark.parametrize(
    ("new", "options", "message"),
    [
        (pd.DataFrame({"A": [0.0]}), {}, "cannot be read by the model"),
        (pd.DataFrame({"A": [0.0], "B": [0.0]}), {"n_bootstrap": 1}, "at least 2"),
        (pd.DataFrame({"A": [0.0], "B": [0.0]}), {"level": 1.0}, "between 0 and 1"),
    ],
)
def test_a_prediction_that_cannot_be_made_is_refused(fitted, new, options, message):
    with pytest.raises(ValueError, match=message):
        fitted.predict(new, **options)


@pytest.mark.slow
def test_the_bands_cover_the_true_curve_close_to_their_level():
    """Twelve simulated experiments, three new points each, with a correct model: 95% bands, measured coverage."""
    design, new = _design(), pd.DataFrame({"A": [0.5, -1.0, 0.0], "B": [-0.5, 1.0, 0.0]})
    exact = np.vstack([_truth(a, b) for a, b in new.to_numpy()])
    pointwise, simultaneous = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for replicate in range(12):
            fit = analyze_functional_experiment(design, _curves(design, seed=100 + replicate), model="interactions")
            bands = fit.predict(new, n_bootstrap=300, random_state=replicate)
            low, high = bands["pointwise_low"].to_numpy(), bands["pointwise_high"].to_numpy()
            pointwise.append(((exact >= low) & (exact <= high)).mean())
            low, high = bands["simultaneous_low"].to_numpy(), bands["simultaneous_high"].to_numpy()
            simultaneous.append(np.all((exact >= low) & (exact <= high), axis=1).mean())
    # The smoothing's bias near the ends keeps both a little under 95%; with knots clamped at the ends, the
    # simultaneous coverage was 13%.
    assert np.mean(pointwise) > 0.8
    assert np.mean(simultaneous) > 0.7


@pytest.mark.dataset
def test_the_methoxylation_scores_follow_the_sources_quadratic_model():
    """McMullen and Jurica's 15 reactions, under the source's stepwise model: its first score is fitted closely.

    The source keeps one component, at 90% of the variance; the 99% rule keeps more, the first among them.
    """
    entry = reference_experiments("mcmullen-jurica-2024")[0]
    runs = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    curve = next(r["curve"] for r in entry["responses"] if "curve" in r)
    scored = runs.dropna(subset=["fpc_score"]).set_index("Experiment")
    samples = pd.concat(
        pd.DataFrame({"run": runs["Experiment"], "time": runs[t], "lcap": runs[y]})
        for t, y in zip(curve["argument"]["columns"], curve["columns"], strict=True)
    ).dropna()
    wide = samples.pivot_table(index="run", columns="time", values="lcap").loc[scored.index]
    model = (
        "Temperature + CuBr + DMF + NaOMe + Temperature:CuBr + Temperature:NaOMe + CuBr:NaOMe"
        " + I(Temperature ** 2) + I(CuBr ** 2) + I(DMF ** 2)"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = analyze_functional_experiment(scored[["Temperature", "CuBr", "DMF", "NaOMe"]], wide, model=model)
    assert fit.components["variance_ratio"].iloc[0] > 0.9
    assert fit.components["kept"].iloc[0]
    assert fit.scores["FPC 1"]["model_summary"]["r_squared"] > 0.99


def test_the_settings_that_reproduce_a_reachable_target_are_found(fitted):
    found = fitted.match_target(_truth(0.4, -0.6), random_state=0)
    assert found["settings"].to_dict() == pytest.approx({"A": 0.4, "B": -0.6}, abs=0.05)
    assert found["distance"] < 0.02
    assert found["scores"].to_numpy() == pytest.approx(found["target_scores"].to_numpy(), abs=1e-3)
    assert found["curve"].shape == (T.size,)


def test_a_factor_can_be_held_or_kept_within_a_range(fitted):
    held = fitted.match_target(_truth(0.4, -0.6), bounds={"B": 0.5}, random_state=0)
    assert held["settings"]["B"] == 0.5
    assert held["distance"] > fitted.match_target(_truth(0.4, -0.6), random_state=0)["distance"]
    ranged = fitted.match_target(_truth(0.9, 0.9), bounds={"A": (-0.5, 0.2)}, random_state=0)
    assert -0.5 <= ranged["settings"]["A"] <= 0.2


def test_a_categorical_factor_is_held_at_the_level_given():
    design = _design().assign(Line=np.tile(["north", "south"], 9))
    curves = _curves(design, seed=9) + np.where(design["Line"].to_numpy() == "south", 0.2, 0.0)[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = analyze_functional_experiment(design, curves, model="A + B + A:B + Line")
    found = fit.match_target(_truth(0.4, -0.6) + 0.2, bounds={"Line": "south"}, random_state=0)
    assert found["settings"]["Line"] == "south"
    assert found["settings"][["A", "B"]].astype(float).to_dict() == pytest.approx({"A": 0.4, "B": -0.6}, abs=0.1)
    with pytest.raises(ValueError, match="needs a level"):
        fit.match_target(_truth(0.0, 0.0))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"bounds": {"Z": (0.0, 1.0)}}, "does not use"),
        ({"bounds": {"A": 0.0, "B": 0.0}}, "nothing to search"),
        ({"n_starts": 0}, "at least 1"),
    ],
)
def test_a_target_search_that_cannot_run_is_refused(fitted, options, message):
    with pytest.raises(ValueError, match=message):
        fitted.match_target(_truth(0.0, 0.0), **options)
