"""#1124: a component with no spread adds 0 to Hotelling's T² and blocks neither TSR nor a T² covariance inverse.

A component holds no spread when its training score standard deviation is at most 1.5e-8 times the
largest (``components_with_spread``), as for the last components of a rank-deficient fit. Each test
checks the model with such components against the same model fitted without them: the components
with spread are the same in both, so their projection and their T² must agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.batch._batch_monitor import BatchMonitor
from process_improve.batch._batch_pca import BatchPCA
from process_improve.batch._batch_pls import BatchPLS
from process_improve.multivariate._common import components_with_spread
from process_improve.multivariate.methods import PCA, PLS, MCUVScaler


def _rank_three_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ten rows of eight columns that span three dimensions, scaled, and one row with two cells missing."""
    rng = np.random.default_rng(1)
    raw = pd.DataFrame(rng.normal(size=(10, 3)) @ rng.normal(size=(3, 8)), columns=list("abcdefgh"))
    scaled = MCUVScaler().fit_transform(raw)
    new = scaled.iloc[[2, 7]] + 0.3
    new.iloc[0, [1, 4]] = np.nan
    return scaled, new


def _small_units_pls_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Columns that vary by 1e-3 in two patterns and by 1e-10 in a third, as #1120's batch test does.

    The third pattern is above PLS's floor, which is relative to the size of X (#1070), so a third
    component is fitted to it, and its score sd is below 1.5e-8: no spread, while every sd is below 1.
    """
    rng = np.random.default_rng(0)
    a, b, c = rng.standard_normal((3, 30))
    cols = np.arange(6)
    x = 1e-3 * (np.outer(a, np.sin(cols)) + np.outer(b, np.cos(cols))) + 1e-10 * np.outer(c, cols**2 - 6.0)
    y = 2 * a + b + 0.1 * rng.standard_normal(30)
    return pd.DataFrame(x - x.mean(axis=0)), pd.DataFrame({"y": y - y.mean()})


def test_the_rule_matches_the_one_t2_has_used_since_1026() -> None:
    """At most 1.5e-8 times the largest sd, or 1.5e-8 itself while every sd is below 1, is no spread."""
    assert components_with_spread(np.array([2.0, 1.0, 2.0 * 1.4e-8])).tolist() == [True, True, False]
    assert components_with_spread(np.array([2.0, 1.0, 2.0 * 1.6e-8])).tolist() == [True, True, True]
    assert components_with_spread(np.array([1e-3, 1e-9])).tolist() == [True, False]


def test_a_rank_deficient_pca_projects_missing_cells_by_tsr() -> None:
    """The two components with no spread score 0 and add 0; the other three agree with a three-component PCA."""
    scaled, new = _rank_three_rows()
    model = PCA(n_components=5).fit(scaled)
    assert components_with_spread(model.scaling_factor_for_scores_.to_numpy()).tolist() == [True] * 3 + [False] * 2
    smaller = PCA(n_components=3).fit(scaled)

    projected, expected = model.project(new), smaller.project(new)

    np.testing.assert_allclose(projected.scores.iloc[0, 3:], 0.0, atol=0.0)
    np.testing.assert_allclose(projected.scores.iloc[:, :3], expected.scores, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(projected.hotellings_t2, expected.hotellings_t2, rtol=1e-8)
    # By hand: the sum of the squared standardised scores of the three components with spread.
    by_hand = ((projected.scores.iloc[:, :3] / model.scaling_factor_for_scores_.iloc[:3]) ** 2).sum(axis=1)
    np.testing.assert_allclose(projected.hotellings_t2, by_hand, rtol=1e-12)
    # The row with missing cells is rebuilt from the components with spread only.
    assert projected.spe.iloc[0] == pytest.approx(expected.spe.iloc[0], rel=1e-8)


def test_a_rank_deficient_pls_projects_missing_cells_by_tsr() -> None:
    """A PLS whose third component has no spread projects by TSR as the two-component PLS does."""
    x, y = _small_units_pls_data()
    model = PLS(n_components=3, scale=False).fit(x, y)
    assert components_with_spread(model.scaling_factor_for_scores_.to_numpy()).tolist() == [True, True, False]
    smaller = PLS(n_components=2, scale=False).fit(x, y)
    new = x.iloc[[3, 11]].copy()
    new.iloc[0, 2] = np.nan

    projected, expected = model.project(new), smaller.project(new)

    assert projected.scores.iloc[0, 2] == 0.0
    np.testing.assert_allclose(projected.scores.iloc[:, :2], expected.scores, rtol=1e-6)
    np.testing.assert_allclose(projected.hotellings_t2, expected.hotellings_t2, rtol=1e-6)


def _rank_two_batches() -> dict[int, pd.DataFrame]:
    """Twelve batches of two tags whose unfolded rows span two dimensions."""
    rng = np.random.default_rng(1)
    time = np.arange(8)
    batches = {}
    for i in range(12):
        a, b = rng.standard_normal(2)
        batches[i] = pd.DataFrame({"u": a * np.sin(time / 3.0) + b, "r": a * np.cos(time / 2.0) - b})
    return batches


def test_a_rank_deficient_batch_model_projects_and_monitors_by_tsr() -> None:
    """``predict_online`` and ``BatchMonitor`` run; T² is that of the batch model without the empty component."""
    batches = _rank_two_batches()
    model = BatchPCA(n_components=3).fit(batches)
    assert components_with_spread(model.scaling_factor_for_scores_.to_numpy()).tolist() == [True, True, False]
    smaller = BatchPCA(n_components=2).fit(batches)

    online, expected = model.predict_online(batches[4], upto_k=3), smaller.predict_online(batches[4], upto_k=3)
    assert online.scores.iloc[2] == 0.0
    np.testing.assert_allclose(online.hotellings_t2, expected.hotellings_t2, rtol=1e-8)

    # The third path of #1124: the per-sample score covariance BatchMonitor inverts.
    monitor = BatchMonitor(model).fit(batches)
    reference = BatchMonitor(smaller).fit(batches)
    traced = monitor.monitor(batches[4])
    np.testing.assert_allclose(traced.hotellings_t2, reference.monitor(batches[4]).hotellings_t2, rtol=1e-6)
    # By hand at the third sample: t' S^-1 t over the two components with spread.
    scores = traced.scores.to_numpy()[2, :2]
    covariance = monitor.score_covariance_over_time_[2][:2, :2]
    assert traced.hotellings_t2[2] == pytest.approx(scores @ np.linalg.solve(covariance, scores), rel=1e-9)


def _rank_deficient_batch_pls(n_components: int) -> tuple[BatchPLS, dict[str, pd.DataFrame]]:
    """#1120's batch PLS: tags that vary by 1e-3, and a third pattern at 1e-9 that has no spread."""
    rng = np.random.default_rng(0)
    time = np.arange(10)
    batches, quality = {}, []
    for i in range(40):
        a, b, c = rng.standard_normal(3)
        u = 1e-3 * a * np.sin(time / 3.0) + 1e-9 * c * np.cos(time / 2.0)
        r = 1e-3 * (b * np.cos(time / 4.0) + 0.5 * a * np.sin(time / 3.0))
        batches[f"b{i}"] = pd.DataFrame({"u": u, "r": r})
        quality.append(2.0 * a + b + 0.3 * rng.standard_normal())
    y = pd.DataFrame({"q": quality}, index=list(batches))
    return BatchPLS(n_components=n_components, scale=False).fit(batches, y), batches


def test_the_mid_course_t2_paths_leave_out_a_component_with_no_spread() -> None:
    """``MidCourseCorrector.predict`` and ``midcourse_correction(score_covariance=...)`` invert over two components.

    Both took the inverse of the whole score covariance, which a component with no spread makes
    singular. ``predict`` gives the T² of the two-component model; ``midcourse_correction``'s T² is
    t' S^-1 t over the first two components, by hand.
    """
    pytest.importorskip("osqp")
    from process_improve.batch.control import MidCourseCorrector, midcourse_correction

    model, batches = _rank_deficient_batch_pls(3)
    assert components_with_spread(np.sqrt(model.explained_variance_)).tolist() == [True, True, False]
    smaller, _ = _rank_deficient_batch_pls(2)
    nominal = pd.DataFrame({"u": np.zeros(model.n_timesteps_)})
    k = 5
    so_far = batches["b3"].iloc[:k]
    predicted = MidCourseCorrector(model, nominal, mv_tags=["u"], y_target=0.5, dead_band=0.0).predict(so_far)
    expected = MidCourseCorrector(smaller, nominal, mv_tags=["u"], y_target=0.5, dead_band=0.0).predict(so_far)
    assert np.isfinite(predicted.t2)
    assert predicted.t2 == pytest.approx(expected.t2, rel=1e-6)

    covariance = MidCourseCorrector(model, nominal, mv_tags=["u"], y_target=0.5).limits_at(k).score_covariance
    observed = pd.Series({(tag, s): float(so_far.iloc[s][tag]) for s in range(k) for tag in model.tag_names_})
    corrected = midcourse_correction(
        model,
        observed=observed,
        free_columns=[("u", s) for s in range(k, model.n_timesteps_)],
        y_target=0.5,
        weights={"target": 1.0, "movement": 0.1},
        score_covariance=covariance,
    )
    scores = corrected.scores.to_numpy()[:2]
    assert np.isfinite(corrected.t2)
    assert corrected.t2 == pytest.approx(scores @ np.linalg.solve(covariance[:2, :2], scores), rel=1e-9)
