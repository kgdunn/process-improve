"""Tests for compare_cv_criteria and pseudo_validation_set (``multivariate/_cv_criteria.py``)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f as f_dist
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, RepeatedKFold

from process_improve._random import check_random_state
from process_improve.multivariate import PLS, MCUVScaler, compare_cv_criteria, pseudo_validation_set
from process_improve.multivariate._cv_criteria import (
    CV_ANOVA_DF_PER_COMPONENT,
    _cv_anova_pvalues,
    _fit_folds,
    _heldout_pass,
    _jackknife_angle,
    _partition_splits,
    _procrustes_scores,
)
from process_improve.multivariate._pls import _vandervoet_randomization

LDPE = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate" / "LDPE" / "LDPE.csv"


def _two_component_data(n: int = 60, k: int = 10, m: int = 1, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """X and Y sharing two latent variables, plus noise."""
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((n, 2))
    X = pd.DataFrame(scores @ rng.standard_normal((2, k)) + 0.3 * rng.standard_normal((n, k)))
    X.columns = [f"x{i}" for i in range(k)]
    Y = pd.DataFrame(scores @ rng.standard_normal((2, m)) + 0.3 * rng.standard_normal((n, m)))
    Y.columns = [f"y{i}" for i in range(m)]
    return X, Y


def _folds(X: pd.DataFrame, Y: pd.DataFrame, n_components: int, cv: object = 7, scope: str = "local") -> list:
    splits = _partition_splits(cv, X, Y, rng=np.random.default_rng(0), random_state=0)
    return _fit_folds(PLS, X, Y, splits, n_components, {}, scope=scope)


@pytest.fixture(scope="module")
def two_component_result() -> object:
    X, Y = _two_component_data()
    return compare_cv_criteria(
        X, Y.iloc[:, 0], max_components=6, random_state=0, n_permutations=199, n_cv_permutations=99
    )


# --- The identity between PRESS and the slope ratio -------------------------------------


@pytest.mark.parametrize("m", [1, 3])
def test_press_drop_equals_slope_ratio_identity(m: int) -> None:
    """PRESS[a-1] - PRESS[a] == (2 s_a - 1) * w_a, exactly, for one or several responses."""
    X, Y = _two_component_data(m=m)
    held = _heldout_pass(_folds(X, Y, 5), Y.to_numpy(), 5)
    press = np.r_[held.press_baseline, held.press_y.sum(axis=1)]
    np.testing.assert_allclose(-np.diff(press), (2 * held.slope_ratio - 1) * held.slope_weight, rtol=1e-10)


def test_press_falls_exactly_when_slope_ratio_exceeds_one_half(two_component_result: object) -> None:
    press = np.r_[two_component_result.press_baseline, two_component_result.press.to_numpy()]
    slope_ratio = two_component_result.table["slope_ratio"].to_numpy()
    assert np.array_equal(-np.diff(press) > 0, slope_ratio > 0.5)


def test_slope_ratio_is_one_in_sample() -> None:
    """Scoring the training rows themselves gives s_a = 1: the fitted slope is the in-sample slope."""
    X, Y = _two_component_data()
    splits = [(np.arange(len(X)), np.arange(len(X)))]
    folds = _fit_folds(PLS, X, Y, splits, 4, {}, scope="local")
    held = _heldout_pass(folds, Y.to_numpy(), 4)
    np.testing.assert_allclose(held.slope_ratio, 1.0, rtol=1e-8)


# --- Consistency with the existing predictive machinery ---------------------------------


@pytest.mark.parametrize("m", [1, 3])
def test_q2_matches_select_n_components(m: int) -> None:
    X, Y = _two_component_data(m=m)
    result = compare_cv_criteria(X, Y, max_components=5, cv=7, random_state=0, n_permutations=19, n_cv_permutations=19)
    reference = PLS.select_n_components(X, Y, max_components=5, cv=7, n_repeats=1, random_state=0)
    # Same folds, same fits. With several responses NIPALS iterates to a tolerance of
    # sqrt(eps), so the last digits can differ between two runs of the same fit.
    rtol = 1e-10 if m == 1 else 1e-6
    np.testing.assert_allclose(result.table["q2y"], reference.r2y_validated["scaled_total"], rtol=rtol)
    np.testing.assert_allclose(result.press, reference.press, rtol=rtol)
    np.testing.assert_allclose(result.table["q2y_se"], reference.q2_se, rtol=rtol)


def test_van_der_voet_pvalues_match_direct_call() -> None:
    X, Y = _two_component_data()
    result = compare_cv_criteria(X, Y, max_components=5, random_state=3, n_permutations=99, n_cv_permutations=0)
    rng_vdv, _ = check_random_state(3).spawn(2)
    held = _heldout_pass(_folds(X, Y, 5, cv=KFold(7, shuffle=True, random_state=3)), Y.to_numpy(), 5)
    rmsecv = np.sqrt(held.press_y.sum(axis=1) / len(X))
    _, p_values = _vandervoet_randomization(
        held.per_obs_sse, total_rmsecv=rmsecv, n_permutations=99, alpha=0.05, random_state=rng_vdv
    )
    np.testing.assert_allclose(result.table["vdv_p"], p_values)


def test_cv_anova_is_a_function_of_q2() -> None:
    q2 = np.array([0.8, 0.5, -0.1, 0.0])
    p_values = _cv_anova_pvalues(q2, N=40, M=1)
    df_reg = CV_ANOVA_DF_PER_COMPONENT * 1
    expected = f_dist.sf(0.8 / 0.2 * (40 - 1 - df_reg) / df_reg, df_reg, 40 - 1 - df_reg)
    assert p_values[0] == pytest.approx(expected)
    assert p_values[2] == 1.0
    assert p_values[3] == 1.0
    assert np.all(np.isnan(_cv_anova_pvalues(q2, N=40, M=2)))
    # Degrees of freedom exhausted: undefined, not zero.
    assert np.isnan(_cv_anova_pvalues(np.array([0.5] * 5), N=10, M=1)[-1])


# --- Procrustes cross-validation ---------------------------------------------------------


@pytest.mark.parametrize("scope", ["global", "local"])
def test_pseudo_validation_set_reproduces_scores_and_spe(scope: str) -> None:
    """The full-data model gives X_pv the D-scaled fold scores and the fold SPE, exactly."""
    X, Y = _two_component_data()
    pv = pseudo_validation_set(X, Y, n_components=3, scope=scope, random_state=0)
    diagnostics = pv.global_model.diagnose(pv.X_pv)
    np.testing.assert_allclose(diagnostics.scores.to_numpy(), pv.scores.to_numpy(), atol=1e-10)
    np.testing.assert_allclose(diagnostics.spe.to_numpy(), pv.local_spe.to_numpy(), atol=1e-10)


def test_local_spe_is_each_fold_models_own_spe() -> None:
    """An independent check of the local SPE: each fold model's own diagnose() on its held-out rows."""
    X, Y = _two_component_data()
    pv = pseudo_validation_set(X, Y, n_components=3, scope="local", random_state=0)
    for train, test in pv.cv_splits:
        scaler = MCUVScaler().fit(X.iloc[train])
        model = PLS(n_components=3).fit(scaler.transform(X.iloc[train]), MCUVScaler().fit_transform(Y.iloc[train]))
        spe = model.diagnose(scaler.transform(X.iloc[test])).spe.to_numpy()
        np.testing.assert_allclose(pv.local_spe.iloc[test].to_numpy(), spe, atol=1e-10)


def test_pseudo_validation_predictions_equal_fold_predictions_global_scope() -> None:
    """The published Procrustean rule: one response, global scope, predictions carry over exactly."""
    X, Y = _two_component_data()
    pv = pseudo_validation_set(X, Y, n_components=2, scope="global", random_state=0)
    x_scaler, y_scaler = MCUVScaler().fit(X), MCUVScaler().fit(Y)
    Xs, Ys = x_scaler.transform(X), y_scaler.transform(Y)
    fold_predictions = np.zeros(len(X))
    for train, test in pv.cv_splits:
        model = PLS(n_components=2, scale=False, warn_on_uncentred=False).fit(Xs.iloc[train], Ys.iloc[train])
        fold_predictions[test] = model.predict(Xs.iloc[test]).to_numpy().ravel()
    fold_predictions = fold_predictions * y_scaler.scale_.iloc[0] + y_scaler.center_.iloc[0]
    np.testing.assert_allclose(pv.global_model.predict(pv.X_pv).to_numpy().ravel(), fold_predictions, atol=1e-10)


def test_fold_sign_flip_changes_nothing() -> None:
    """A fold flips t and c together, so the score-space criteria and Procrustes scores are sign-free."""
    X, Y = _two_component_data()
    folds = _folds(X, Y, 4)
    model = PLS(n_components=4).fit(X, Y)
    before = _heldout_pass(folds, Y.to_numpy(), 4)
    pv_before = _procrustes_scores(model, folds)
    for attribute in ("weights", "direct_weights", "x_loadings", "y_loadings"):
        getattr(folds[2], attribute)[:, 1] *= -1.0
    after = _heldout_pass(folds, Y.to_numpy(), 4)
    pv_after = _procrustes_scores(model, folds)
    np.testing.assert_allclose(after.r_cv, before.r_cv, rtol=1e-12)
    np.testing.assert_allclose(after.slope_ratio, before.slope_ratio, rtol=1e-12)
    for left, right in zip(pv_before, pv_after, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-12)


def test_jackknife_angle_scales_the_tangent_by_sqrt_folds_minus_one() -> None:
    angles = np.full((5, 1), 10.0)
    expected = np.degrees(np.arctan(2.0 * np.tan(np.radians(10.0))))
    assert _jackknife_angle(angles, 5)[0] == pytest.approx(expected)
    assert _jackknife_angle(np.full((3, 1), 90.0), 3)[0] == pytest.approx(90.0)


# --- Behaviour on data with known structure ----------------------------------------------


def test_structural_rules_find_two_components(two_component_result: object) -> None:
    picks = two_component_result.recommendations["n_components"]
    assert picks["covariance_permutation"] == 2
    assert picks["subspace_stability"] == 2
    assert picks["score_correlation"] >= 1
    assert picks["pv_spe_alarm"] >= 2  # the monitoring limits hold at least for the real components
    table = two_component_result.table
    assert table.loc[1, "r_cv_p"] < 0.05
    assert table.loc[3, "angle_subspace_deg"] > 30 > table.loc[2, "angle_subspace_deg"]


def test_y_related_rules_find_nothing_in_noise() -> None:
    X, _ = _two_component_data()
    y = pd.Series(np.random.default_rng(3).standard_normal(len(X)), name="noise")
    result = compare_cv_criteria(X, y, max_components=4, random_state=0, n_permutations=199, n_cv_permutations=99)
    assert result.recommendations.loc["covariance_permutation", "n_components"] == 0
    assert result.recommendations.loc["score_correlation", "n_components"] == 0


@pytest.mark.slow
def test_false_positive_rates_under_the_null() -> None:
    """With Y unrelated to X, the two tests of a Y relationship fire at about their nominal 5% rate.

    The normal approximation for the held-out correlation fires about three times as
    often on this X, whose two strong latent variables widen the null distribution.
    """
    X, _ = _two_component_data()
    fired = {"score_correlation": 0, "covariance_permutation": 0}
    n_datasets = 30
    for seed in range(n_datasets):
        y = pd.Series(np.random.default_rng(100 + seed).standard_normal(len(X)))
        picks = compare_cv_criteria(
            X, y, max_components=2, random_state=seed, n_permutations=99, n_cv_permutations=39
        ).recommendations["n_components"]
        for rule in fired:
            fired[rule] += int(picks[rule] > 0)
    assert fired["score_correlation"] / n_datasets <= 0.2
    assert fired["covariance_permutation"] / n_datasets <= 0.2


def test_angles_do_not_depend_on_the_number_of_folds() -> None:
    """Jackknife scaling makes 7-fold and leave-one-out angles comparable (raw ones differ ~3x)."""
    X, Y = _two_component_data()
    seven = compare_cv_criteria(
        X, Y, max_components=3, cv=7, random_state=0, n_permutations=9, n_cv_permutations=0
    ).table
    loo = compare_cv_criteria(X, Y, max_components=3, cv=LeaveOneOut(), n_permutations=9, n_cv_permutations=0).table
    np.testing.assert_allclose(seven["angle_subspace_deg"], loo["angle_subspace_deg"], rtol=0.5)


# --- API contract --------------------------------------------------------------------------


def test_table_and_recommendations_shape(two_component_result: object) -> None:
    table = two_component_result.table
    assert list(table.index) == [1, 2, 3, 4, 5, 6]
    assert table.index.name == "n_components"
    assert {"q2y", "r_cv", "r_cv_p", "slope_ratio", "cov_perm_p", "angle_subspace_deg", "pv_spe_alarm_rate"} <= set(
        table
    )
    recommendations = two_component_result.recommendations
    assert list(recommendations.columns) == ["n_components", "rule", "question"]
    assert recommendations["n_components"].between(0, 6).all()
    assert len(two_component_result.cv_splits) == 7


def test_reproducible_with_int_and_generator() -> None:
    X, Y = _two_component_data(n=40)
    first = compare_cv_criteria(X, Y, max_components=3, random_state=5, n_permutations=49, n_cv_permutations=19)
    second = compare_cv_criteria(X, Y, max_components=3, random_state=5, n_permutations=49, n_cv_permutations=19)
    pd.testing.assert_frame_equal(first.table, second.table)
    from_generator = [
        compare_cv_criteria(
            X, Y, max_components=3, random_state=np.random.default_rng(11), n_permutations=49, n_cv_permutations=19
        ).table
        for _ in range(2)
    ]
    pd.testing.assert_frame_equal(*from_generator)


def test_accepts_arrays_series_and_group_splitter() -> None:
    X, Y = _two_component_data(n=42)
    groups = np.repeat(np.arange(6), 7)
    splits = list(GroupKFold(n_splits=6).split(X, Y, groups))
    result = compare_cv_criteria(
        X.to_numpy(), Y.to_numpy().ravel(), max_components=3, cv=splits, n_permutations=9, n_cv_permutations=19
    )
    assert result.table.shape[0] == 3
    series_result = compare_cv_criteria(
        X, Y.iloc[:, 0], max_components=2, random_state=0, n_permutations=9, n_cv_permutations=19
    )
    assert series_result.table["cv_anova_p"].notna().all()  # one response: CV-ANOVA is defined


def test_multiple_responses_leave_cv_anova_undefined() -> None:
    X, Y = _two_component_data(m=3)
    result = compare_cv_criteria(X, Y, max_components=3, random_state=0, n_permutations=19, n_cv_permutations=19)
    assert result.table["cv_anova_p"].isna().all()
    assert result.table["cov_perm_p"].between(0, 1).all()


def test_max_components_is_capped() -> None:
    X, Y = _two_component_data(n=21, k=4)
    result = compare_cv_criteria(X, Y, max_components=100, cv=3, random_state=0, n_permutations=9, n_cv_permutations=19)
    assert result.table.shape[0] == 4  # min(14 training rows - 1, 4 features)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cv": 1}, "cv must be >= 2"),
        ({"cv": RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)}, "exactly once"),
        ({"alpha": 0.0}, "alpha"),
        ({"conf_level": 1.5}, "conf_level"),
        ({"n_permutations": 0}, "n_permutations"),
        ({"n_cv_permutations": -1}, "n_cv_permutations"),
        ({"angle_threshold": 120.0}, "angle_threshold"),
        ({"max_components": 0}, "max_components"),
    ],
)
def test_invalid_settings_raise(kwargs: dict, message: str) -> None:
    X, Y = _two_component_data(n=30)
    with pytest.raises(ValueError, match=message):
        compare_cv_criteria(X, Y, **kwargs)


def test_missing_values_raise() -> None:
    X, Y = _two_component_data(n=30)
    X.iloc[3, 2] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        compare_cv_criteria(X, Y)
    with pytest.raises(ValueError, match="missing values"):
        pseudo_validation_set(X, Y, n_components=2)


def test_pseudo_validation_set_rejects_bad_arguments() -> None:
    X, Y = _two_component_data(n=30)
    with pytest.raises(ValueError, match="scope"):
        pseudo_validation_set(X, Y, n_components=2, scope="elsewhere")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_components"):
        pseudo_validation_set(X, Y, n_components=50)


def test_classmethods_fit_with_the_subclass() -> None:
    class TaggedPLS(PLS):
        pass

    X, Y = _two_component_data(n=40)
    result = TaggedPLS.compare_cv_criteria(
        X, Y, max_components=3, random_state=0, n_permutations=19, n_cv_permutations=19
    )
    assert type(result.global_model) is TaggedPLS
    direct = compare_cv_criteria(X, Y, max_components=3, random_state=0, n_permutations=19, n_cv_permutations=19)
    pd.testing.assert_frame_equal(result.table, direct.table)
    pv = TaggedPLS.pseudo_validation_set(X, Y, n_components=2, random_state=0)
    assert type(pv.global_model) is TaggedPLS
    assert pv.X_pv.shape == X.shape
    assert list(pv.X_pv.columns) == list(X.columns)


@pytest.mark.dataset
def test_ldpe_runs_and_recommends_within_range() -> None:
    values = pd.read_csv(LDPE, index_col=0)
    X, Y = values.iloc[:, :14], values.iloc[:, 14:]
    result = compare_cv_criteria(X, Y, max_components=6, random_state=0, n_permutations=199, n_cv_permutations=49)
    assert result.table.shape == (6, 17)
    assert result.recommendations["n_components"].between(0, 6).all()
    assert result.table["q2y"].iloc[0] > 0  # LDPE has real signal in the first component
    assert result.recommendations.loc["covariance_permutation", "n_components"] >= 1


def test_cv_criteria_plot_draws_four_panels(two_component_result: object) -> None:
    go = pytest.importorskip("plotly.graph_objects")
    from process_improve.multivariate.plots import cv_criteria_plot

    fig = cv_criteria_plot(two_component_result)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 13  # 9 series and bands, 4 reference lines
    assert {trace.legend for trace in fig.data} == {"legend", "legend2", "legend3", "legend4"}
    assert len(fig.layout.shapes) >= 4  # one dashed line per distinct recommendation
    with pytest.raises(ValueError, match="compare_cv_criteria"):
        cv_criteria_plot(object())
