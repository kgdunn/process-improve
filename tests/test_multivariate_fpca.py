"""Functional PCA: P-spline smoothing of each curve, then PCA in the L2 metric of the basis."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from process_improve.experiments import datasets
from process_improve.experiments.reference_catalogue import reference_experiments
from process_improve.multivariate import FPCA, PCA
from process_improve.multivariate.methods import SpecificationWarning


def _colour_curves(n_runs: int = 40, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Curves built as the book's colour case is: an amplitude times a plateau shape, plus a drift times a late ramp."""
    rng = np.random.default_rng(seed)
    t = np.arange(10.0)
    ref = 1 - np.exp(-t / 2)
    ref /= ref.max()
    tail = np.clip((t - 4) / 5, 0, None)
    tail /= tail.max()
    amplitude, drift = rng.uniform(0.5, 2.5, n_runs), rng.uniform(-0.1, 0.35, n_runs)
    curves = amplitude[:, None] * (ref + drift[:, None] * tail) + rng.normal(0, 0.03, (n_runs, t.size))
    return pd.DataFrame(curves, columns=t), ref, tail


def _smooth_curves(n_runs: int = 30, n_points: int = 201, seed: int = 2) -> pd.DataFrame:
    """Noise-free curves, each a random mix of three smooth shapes on [0, 1]."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_points)
    shapes = np.vstack([np.sin(np.pi * t), np.cos(2 * np.pi * t), t**2])
    return pd.DataFrame(rng.normal(0, [3.0, 1.5, 0.7], (n_runs, 3)) @ shapes, columns=t)


def test_two_components_span_the_two_shapes_the_curves_were_built_from():
    curves, ref, tail = _colour_curves()
    model = FPCA(n_components=2).fit(curves)
    assert model.explained_variance_ratio_.sum() > 0.99
    span = np.column_stack([np.ones(10), model.mean_function_, model.eigenfunctions_])
    for shape in (ref, tail):
        fitted = span @ np.linalg.lstsq(span, shape, rcond=None)[0]
        assert np.abs(fitted - shape).max() < 0.08  # the ramp's kink at t = 4 is the one point smoothed off


def test_on_a_fine_grid_the_components_are_those_of_pca_on_the_points_weighted_by_the_trapezoid_rule():
    """The L2 metric the Gram matrix gives is what the trapezoid rule approximates on the points."""
    curves = _smooth_curves()
    t = curves.columns.to_numpy(dtype=float)
    weight = np.gradient(t)  # the trapezoid rule's weights on an even grid: half at the two ends
    weight[[0, -1]] = (t[1] - t[0]) / 2
    model = FPCA(n_components=3, n_basis=40, smoothing=1e-9).fit(curves)
    discrete = PCA(n_components=3).fit((curves - curves.mean()) * np.sqrt(weight))
    assert model.explained_variance_ == pytest.approx(discrete.explained_variance_, rel=1e-3)
    for a in range(3):
        function = model.eigenfunctions_.iloc[:, a].to_numpy()
        vector = discrete.loadings_.iloc[:, a].to_numpy() / np.sqrt(weight)
        assert abs(np.corrcoef(function, vector)[0, 1]) > 0.9999


def test_the_components_do_not_depend_on_how_many_b_splines_carry_them():
    curves = _smooth_curves(n_points=61)
    coarse = FPCA(n_components=2, n_basis=15, smoothing=1e-6).fit(curves)
    fine = FPCA(n_components=2, n_basis=35, smoothing=1e-6).fit(curves)
    assert coarse.explained_variance_ == pytest.approx(fine.explained_variance_, rel=1e-3)
    for a in range(2):
        assert abs(np.corrcoef(coarse.eigenfunctions_.iloc[:, a], fine.eigenfunctions_.iloc[:, a])[0, 1]) > 0.9999


def test_the_eigenfunctions_are_orthonormal_as_functions_and_the_scores_uncorrelated():
    curves = _smooth_curves()
    model = FPCA(n_components=3).fit(curves)
    t, phi = curves.columns.to_numpy(dtype=float), model.eigenfunctions_.to_numpy()
    inner = np.array([[np.trapezoid(phi[:, i] * phi[:, j], t) for j in range(3)] for i in range(3)])
    assert inner == pytest.approx(np.eye(3), abs=1e-3)
    scores = model.scores_.to_numpy()
    assert scores.mean(axis=0) == pytest.approx(np.zeros(3), abs=1e-9)
    assert np.corrcoef(scores.T)[np.triu_indices(3, 1)] == pytest.approx(np.zeros(3), abs=1e-9)
    assert list(model.scores_.columns) == ["FPC 1", "FPC 2", "FPC 3"]
    assert model.scores_.index.equals(curves.index)


def test_transform_gives_the_fitted_scores_and_inverse_transform_rebuilds_the_curves():
    curves = _smooth_curves()
    model = FPCA(n_components=3, smoothing=1e-9).fit(curves)
    assert model.transform(curves).to_numpy() == pytest.approx(model.scores_.to_numpy(), abs=1e-8)
    rebuilt = model.inverse_transform(model.scores_)
    assert rebuilt.to_numpy() == pytest.approx(curves.to_numpy(), abs=1e-4)  # three shapes, three components
    assert rebuilt.index.equals(curves.index)


def test_a_derivative_is_read_from_the_b_splines():
    t = np.linspace(0, 2, 41)
    curves = pd.DataFrame([a * t**2 for a in (1.0, 1.5, 2.0, 3.0)], columns=t)
    model = FPCA(n_components=1, smoothing=1e-9).fit(curves)
    rates = model.inverse_transform(model.scores_, grid=[0.5, 1.0, 1.5], derivative=1)
    assert rates.to_numpy() == pytest.approx(np.outer([1.0, 1.5, 2.0, 3.0], 2 * np.array([0.5, 1.0, 1.5])), rel=1e-4)


def test_curves_with_missing_points_score_as_they_do_complete():
    curves, _, _ = _colour_curves(seed=3)
    rng = np.random.default_rng(4)
    holed = curves.mask(rng.random(curves.shape) < 0.2)
    holed.iloc[:, 0] = curves.iloc[:, 0]  # keep each curve's start, so no curve loses both ends
    full, partial = FPCA(n_components=2).fit(curves), FPCA(n_components=2).fit(holed)
    assert abs(np.corrcoef(full.scores_.iloc[:, 0], partial.scores_.iloc[:, 0])[0, 1]) > 0.99
    assert partial.transform(holed).to_numpy() == pytest.approx(partial.scores_.to_numpy(), abs=1e-8)


def test_the_given_smoothing_is_used_and_a_chosen_one_lies_inside_the_search():
    curves, _, _ = _colour_curves()
    assert FPCA(smoothing=0.5).fit(curves).smoothing_ == 0.5
    chosen = FPCA().fit(curves)
    basis = chosen._basis
    scale = np.trace(basis.T @ basis) / np.trace(chosen._penalty)
    assert scale * 1e-8 < chosen.smoothing_ < scale * 1e4


def test_a_request_for_more_components_than_the_curves_hold_is_clamped():
    curves, _, _ = _colour_curves(n_runs=5)
    with pytest.warns(SpecificationWarning, match="at most 4"):
        model = FPCA(n_components=9).fit(curves)
    assert model.n_components_ == 4
    assert FPCA().fit(curves).n_components_ == 4  # one fewer than the curves, once they are centred
    assert FPCA(n_components=1).fit(curves).eigenfunctions_.shape == (10, 1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rows": slice(0, 1)}, "at least two curves"),
        ({"grid": np.arange(9.0)}, "has 9 points"),
        ({"grid": np.r_[0.0, 2.0, 1.0, np.arange(3.0, 10.0)]}, "strictly increasing"),
        ({"columns": list("abcdefghij")}, "Give the grid"),
        ({"cell": np.inf}, "infinite"),
        ({"blank_row": True}, "fewer than 2 measured points"),
        ({"model": {"n_basis": 3}}, "n_basis must be at least 4"),
        ({"model": {"smoothing": 0.0}}, "smoothing must be positive"),
        ({"model": {"n_components": 0}}, "n_components must be >= 1; got 0"),
    ],
)
def test_bad_input_is_refused_with_a_reason(change, message):
    curves, _, _ = _colour_curves(n_runs=6)
    grid = change.get("grid")
    if "rows" in change:
        curves = curves.iloc[change["rows"]]
    if "columns" in change:
        curves.columns = change["columns"]
    if "cell" in change:
        curves.iloc[0, 3] = change["cell"]
    if change.get("blank_row"):
        curves.iloc[2, 1:] = np.nan
    with pytest.raises(ValueError, match=message):
        FPCA(**change.get("model", {})).fit(curves, grid=grid)


def test_scores_for_the_wrong_number_of_components_or_a_grid_outside_the_fit_are_refused():
    curves, _, _ = _colour_curves()
    model = FPCA(n_components=2).fit(curves)
    with pytest.raises(ValueError, match="Expected 2 scores"):
        model.inverse_transform(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="within the fitted range"):
        model.inverse_transform(np.zeros((1, 2)), grid=[-1.0, 5.0])


def test_the_model_clones_and_pickles_like_any_estimator():
    curves, _, _ = _colour_curves()
    model = FPCA(n_components=2, penalty_order=3).fit(curves)
    assert clone(model).get_params() == model.get_params()
    assert not model.scores_.empty  # fills the cache of frames, which a pickle leaves out
    restored = pickle.loads(pickle.dumps(model))  # noqa: S301 - a model this test just pickled
    assert "_frame_cache" not in restored.__dict__
    assert restored.scores_.equals(model.scores_)


@pytest.mark.dataset
def test_the_methoxylation_scores_match_the_sources_up_to_its_time_unit():
    """McMullen and Jurica's 15 reactions, sampled at their own times: one component, as the source finds.

    The source's PACE scores correlate with these almost perfectly and are sqrt(24) times larger, because it
    integrates over hours and the catalogue holds the time coded as hours / 24.
    """
    entry = reference_experiments("mcmullen-jurica-2024")[0]
    runs = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    curve = next(r["curve"] for r in entry["responses"] if "curve" in r)
    samples = pd.concat(
        pd.DataFrame({"run": runs["Experiment"], "time": runs[t], "lcap": runs[y]})
        for t, y in zip(curve["argument"]["columns"], curve["columns"], strict=True)
    ).dropna()
    scored = runs.dropna(subset=["fpc_score"]).set_index("Experiment")["fpc_score"]
    wide = samples.pivot_table(index="run", columns="time", values="lcap").loc[scored.index]
    model = FPCA().fit(wide)
    assert model.explained_variance_ratio_.iloc[0] > 0.9  # the source keeps one component at 90.717%
    ours = model.scores_.iloc[:, 0]
    assert abs(np.corrcoef(ours, scored)[0, 1]) > 0.99
    assert scored.std() / ours.std() == pytest.approx(np.sqrt(24), rel=0.05)


@pytest.mark.dataset
def test_the_biscuit_spectra_need_few_components():
    spectra = datasets.biscuit().iloc[:, 6:]
    model = FPCA(n_components=6).fit(spectra)
    assert model.n_basis_ == 40
    assert model.explained_variance_ratio_.iloc[:2].sum() > 0.98
    assert model.pca_.hotellings_t2_.shape == (72, 6)
