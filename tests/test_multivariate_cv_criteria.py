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
from process_improve.multivariate._common import _vandervoet_randomization
from process_improve.multivariate._cv_criteria import (
    CV_ANOVA_DF_PER_COMPONENT,
    _cv_anova_pvalues,
    _fit_folds,
    _heldout_pass,
    _jackknife_angle,
    _leading_count,
    _nipals_scores,
    _partition_splits,
    _procrustes_scores,
    _score_correlation_null,
    _swapped_pairs,
)
from process_improve.multivariate._limits import spe_calculation
from process_improve.simulation import LatentStructure

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


#: Two latent variables drive y (LatentStructure uses Hadamard loadings, so the count is exact).
TWO_LATENT = LatentStructure(x_sd=[3.0, 1.0], y_coefficients=[1.0, 1.0])


def _sample(process: LatentStructure, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    drawn = process.sample(n, random_state=seed)
    return drawn.X, drawn.Y


def _folds(X: pd.DataFrame, Y: pd.DataFrame, n_components: int, cv: object = 7, scope: str = "local") -> list:
    splits = _partition_splits(cv, X, Y, rng=np.random.default_rng(0), random_state=0)
    return _fit_folds(lambda: PLS(n_components=n_components), X, Y, splits, scope=scope)


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
    folds = _fit_folds(lambda: PLS(n_components=4), X, Y, splits, scope="local")
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
    for attribute in ("scores", "weights", "direct_weights", "x_loadings", "y_loadings"):
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


def test_every_rule_recovers_two_latent_variables() -> None:
    """Two latent variables drive y, and every rule finds exactly two components."""
    X, Y = _sample(TWO_LATENT, 60, seed=0)
    result = compare_cv_criteria(X, Y, max_components=5, random_state=0, n_permutations=199, n_cv_permutations=99)
    assert set(result.recommendations["n_components"]) == {2}


def test_a_swapping_pair_keeps_its_stable_span() -> None:
    """Two equally strong components swap between folds: each direction is unstable, their span is not."""
    X, Y = _sample(LatentStructure(x_sd=[1.0, 1.0], y_coefficients=np.eye(2)), 60, seed=0)
    result = compare_cv_criteria(X, Y, max_components=4, random_state=0, n_permutations=99, n_cv_permutations=19)
    table = result.table
    assert table.loc[1, "angle_subspace_deg"] > result.angle_threshold > table.loc[2, "angle_subspace_deg"]
    assert result.recommendations.loc["subspace_stability", "n_components"] == 2


def test_leading_count_carries_a_swapped_pair() -> None:
    passes = np.array([False, True, False, True])
    assert _leading_count(passes) == 0
    swaps = _swapped_pairs(np.array([50.0, 10.0, 60.0, 20.0]), 30.0)
    np.testing.assert_array_equal(swaps, [True, False, True, False])
    assert _leading_count(passes, swaps) == 4
    assert _leading_count(np.array([True, False, False]), _swapped_pairs(np.array([5.0, 50.0, 60.0]), 30.0)) == 1


@pytest.mark.slow
@pytest.mark.parametrize(("n", "gaps"), [(30, 0.0), (60, 0.2)])
def test_spe_limit_refitted_to_held_out_rows_holds_on_new_rows(n: int, gaps: float) -> None:
    """The training-residual SPE limit is too tight for new rows; the pseudo-validation limit is not.

    Fresh, complete rows from the same simulated process are the ground truth: a limit
    at 95% should flag about 5% of them. With gaps in the training data, the training
    and held-out SPE are summed over fewer cells, which makes the model's own limit
    tighter still; the refitted limit scales each held-out row to its complete-row value.
    """
    X, Y = _sample(TWO_LATENT, n, seed=4)
    if gaps:
        X, Y = _with_gaps(X, Y, fraction=gaps)
    X_new, _ = _sample(TWO_LATENT, 4000, seed=1004)
    result = compare_cv_criteria(X, Y, max_components=2, random_state=0, n_permutations=19, n_cv_permutations=0)
    spe_new = result.global_model.diagnose(X_new).spe.to_numpy()
    limits = result.spe_limits.loc[2]
    assert np.mean(spe_new > limits["full_data"]) > 0.1
    assert np.mean(spe_new > limits["pseudo_validation"]) < 0.075
    assert result.table.loc[2, "pv_spe_limit_ratio"] == pytest.approx(limits["pseudo_validation"] / limits["full_data"])


@pytest.mark.parametrize("gaps", [0.0, 0.1])
def test_spe_limits_are_fitted_to_training_and_held_out_spe(gaps: float) -> None:
    """The refitted limit is fitted to the held-out SPE, each row scaled to K cells when it has gaps."""
    X, Y = _two_component_data()
    if gaps:
        X, Y = _with_gaps(X, Y, fraction=gaps)
    result = compare_cv_criteria(X, Y, max_components=3, random_state=0, n_permutations=9, n_cv_permutations=0)
    pv = pseudo_validation_set(X, Y, n_components=3, scope="local", random_state=0)
    to_complete = np.sqrt(X.shape[1] / X.notna().sum(axis=1).to_numpy())
    limits = result.spe_limits.loc[3]
    assert limits["full_data"] == pytest.approx(spe_calculation(result.global_model.spe_.iloc[:, 2].to_numpy()))
    assert limits["pseudo_validation"] == pytest.approx(spe_calculation(pv.local_spe.to_numpy() * to_complete))


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
        ({"scale": False}, "scale=False"),
    ],
)
def test_invalid_settings_raise(kwargs: dict, message: str) -> None:
    X, Y = _two_component_data(n=30)
    with pytest.raises(ValueError, match=message):
        compare_cv_criteria(X, Y, **kwargs)


# --- Missing values ------------------------------------------------------------------------


def _with_gaps(X: pd.DataFrame, Y: pd.DataFrame, fraction: float = 0.15, seed: int = 7) -> tuple:
    """Delete a random ``fraction`` of the cells of X and of Y (missing completely at random)."""
    rng = np.random.default_rng(seed)
    return X.mask(rng.random(X.shape) < fraction), Y.mask(rng.random(Y.shape) < fraction)


def test_nipals_scores_equal_direct_weights_on_complete_data() -> None:
    """With complete training data, P'W is unit upper triangular and NIPALS scoring is x @ W*."""
    X, Y = _two_component_data()
    model = PLS(n_components=4).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))
    x = MCUVScaler().fit_transform(X).to_numpy()
    scores = _nipals_scores(x, model.x_weights_.to_numpy(), model.x_loadings_.to_numpy())
    np.testing.assert_allclose(scores, x @ model.direct_weights_.to_numpy(), atol=1e-10)


def test_nipals_scores_reproduce_the_training_scores_with_gaps() -> None:
    """With gaps, x @ W* is not how the model scored its rows; the NIPALS step reproduces the fit exactly."""
    X, Y = _with_gaps(*_two_component_data())
    x = MCUVScaler().fit_transform(X)
    model = PLS(n_components=3).fit(x, MCUVScaler().fit_transform(Y))
    scores = _nipals_scores(x.to_numpy(), model.x_weights_.to_numpy(), model.x_loadings_.to_numpy())
    np.testing.assert_allclose(scores, model.scores_.to_numpy(), atol=1e-10)
    complete = x.notna().all(axis=1).to_numpy()
    direct = x.to_numpy()[complete] @ model.direct_weights_.to_numpy()
    assert np.abs(direct - model.scores_.to_numpy()[complete]).max() > 1e-3  # P'W is no longer triangular


@pytest.mark.parametrize("gaps", [False, True])
def test_kernel_null_statistic_matches_the_fold_models(gaps: bool) -> None:
    """The permutation test's observed statistic is r_cv itself for complete data, and close to it with gaps."""
    X, Y = _two_component_data()
    if gaps:
        X, Y = _with_gaps(X, Y, fraction=0.05)
    folds = _folds(X, Y, 3)
    splits = [(fold.train, fold.test) for fold in folds]
    held = _heldout_pass(folds, Y.to_numpy(), 3)
    observed, null = _score_correlation_null(X.to_numpy(), Y.to_numpy(), splits, (5, 3), np.random.default_rng(0))
    assert null.shape == (5, 3)
    np.testing.assert_allclose(observed[:2], held.r_cv[:2], atol=0.02 if gaps else 1e-8)


@pytest.mark.parametrize("m", [1, 3])
def test_identity_and_in_sample_slope_hold_with_gaps(m: int) -> None:
    """Missing X and Y cells: the PRESS identity stays exact, and s_a = 1 on the training rows."""
    X, Y = _with_gaps(*_two_component_data(m=m))
    held = _heldout_pass(_folds(X, Y, 4), Y.to_numpy(), 4)
    press = np.r_[held.press_baseline, held.press_y.sum(axis=1)]
    np.testing.assert_allclose(-np.diff(press), (2 * held.slope_ratio - 1) * held.slope_weight, rtol=1e-10)

    everything = [(np.arange(len(X)), np.arange(len(X)))]
    in_sample = _heldout_pass(_fit_folds(lambda: PLS(n_components=4), X, Y, everything, scope="local"), Y.to_numpy(), 4)
    np.testing.assert_allclose(in_sample.slope_ratio, 1.0, rtol=1e-8)


def test_compare_cv_criteria_runs_with_gaps_and_a_row_without_y() -> None:
    X, Y = _with_gaps(*_two_component_data())
    Y.iloc[5] = np.nan  # this row helps fit X and is left out of every Y criterion
    result = compare_cv_criteria(X, Y, max_components=4, random_state=0, n_permutations=99, n_cv_permutations=39)
    table = result.table
    assert np.isfinite(table.drop(columns="cv_anova_p").to_numpy()).all()
    assert result.recommendations.loc["covariance_permutation", "n_components"] == 2
    assert table.loc[1, "q2y"] > 0.5


def test_pseudo_validation_set_keeps_the_missing_pattern() -> None:
    """X_pv has the gaps of X; its complete rows keep the exact scores and SPE of the construction."""
    X, Y = _with_gaps(*_two_component_data(), fraction=0.05)
    pv = pseudo_validation_set(X, Y, n_components=2, scope="global", random_state=0)
    assert pv.X_pv.isna().equals(X.isna())
    complete = X.notna().all(axis=1)
    diagnostics = pv.global_model.diagnose(pv.X_pv[complete])
    np.testing.assert_allclose(diagnostics.scores.to_numpy(), pv.scores[complete].to_numpy(), atol=1e-10)
    np.testing.assert_allclose(diagnostics.spe.to_numpy(), pv.local_spe[complete].to_numpy(), atol=1e-10)


@pytest.mark.parametrize(
    ("where", "message"),
    [
        ("x_row", "every X value missing"),
        ("x_column", "Columns of X"),
        ("y_column", "Columns of Y"),
        ("sparse", "fold"),
    ],
)
def test_unusable_gaps_raise(where: str, message: str) -> None:
    X, Y = _two_component_data(n=30)
    if where == "x_row":
        X.iloc[3] = np.nan
    elif where == "x_column":
        X["x2"] = np.nan
    elif where == "y_column":
        Y["y0"] = np.nan
    else:
        X.iloc[2:, 0] = np.nan  # two observed values: some training fold keeps fewer than two
    with pytest.raises(ValueError, match=message):
        compare_cv_criteria(X, Y)
    with pytest.raises(ValueError, match=message):
        pseudo_validation_set(X, Y, n_components=2)


def test_pseudo_validation_set_rejects_bad_arguments() -> None:
    X, Y = _two_component_data(n=30)
    with pytest.raises(ValueError, match="scope"):
        pseudo_validation_set(X, Y, n_components=2, scope="elsewhere")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_components"):
        pseudo_validation_set(X, Y, n_components=50)
    with pytest.raises(ValueError, match="scale=False"):
        pseudo_validation_set(X, Y, n_components=2, scale=False)


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
    assert result.table.shape == (6, 18)
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
    assert len(fig.layout.shapes) >= 3  # one dotted line per distinct recommendation in panels 1 to 3
    assert any("SPE limit refitted" in annotation.text for annotation in fig.layout.annotations)
    with pytest.raises(ValueError, match="compare_cv_criteria"):
        cv_criteria_plot(object())
