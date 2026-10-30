"""Tests for compare_cv_criteria and pseudo_validation_set (``multivariate/_cv_criteria.py``)."""

from __future__ import annotations

import copy
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm
from scipy.stats import f as f_dist
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, RepeatedKFold, StratifiedKFold
from sklearn.utils import Bunch

import process_improve.multivariate._cv_criteria as cv_criteria
from process_improve._random import check_random_state
from process_improve.multivariate import (
    PLS,
    MCUVScaler,
    SpecificationWarning,
    compare_cv_criteria,
    pseudo_validation_set,
)
from process_improve.multivariate._common import _vandervoet_randomization
from process_improve.multivariate._cv_criteria import (
    CV_ANOVA_DF_PER_COMPONENT,
    _as_frames,
    _autoscale_parameters,
    _check_fold_coverage,
    _CompareSettings,
    _component_cap,
    _correlation_about_origin,
    _covariance_permutation_pvalues,
    _cv_anova_pvalues,
    _fit_folds,
    _Fold,
    _heldout_pass,
    _jackknife_angle,
    _leading_count,
    _monitoring_criteria,
    _nipals_scores,
    _partition_splits,
    _permutation_summary,
    _procrustes_scores,
    _recommendations,
    _score_correlation_null,
    _swapped_pairs,
    _weight_angles,
)
from process_improve.multivariate._limits import hotellings_t2_limit, spe_calculation
from process_improve.multivariate.methods import PRM
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


def _folds(X: pd.DataFrame, Y: pd.DataFrame, n_components: int, cv: object = 7) -> list:
    splits = _partition_splits(cv, X, Y, rng=np.random.default_rng(0), random_state=0)
    return _fit_folds(lambda: PLS(n_components=n_components), X, Y, splits)


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
    folds = _fit_folds(lambda: PLS(n_components=4), X, Y, splits)
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


def test_van_der_voet_counts_a_component_with_no_finite_difference_as_indistinguishable() -> None:
    """A count whose residuals were never compared on any row gets p = 1, so parsimony picks it."""
    per_obs_sse = np.array([[np.nan, np.nan, np.nan], [1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    recommended, p_values = _vandervoet_randomization(
        per_obs_sse, total_rmsecv=np.array([np.nan, 1.0, 1.2]), n_permutations=19, random_state=0
    )
    assert p_values[0] == 1.0
    assert recommended == 1


def test_van_der_voet_keeps_the_reference_when_no_count_qualifies() -> None:
    """If no p-value exceeds alpha (only possible at alpha = 1) the reference model is recommended."""
    per_obs_sse = np.array([[3.0, 4.0, 5.0], [1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    recommended, p_values = _vandervoet_randomization(
        per_obs_sse, total_rmsecv=np.array([1.5, 1.0, 1.2]), n_permutations=19, alpha=1.0, random_state=0
    )
    assert p_values[1] == 1.0  # the reference, compared with itself
    assert recommended == 2


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


def test_cv_anova_calls_a_perfect_prediction_significant() -> None:
    """A Q2 of 1 leaves no residual for the F-ratio: its p-value is 0, not a division by zero."""
    assert _cv_anova_pvalues(np.array([1.0, 0.5]), N=40, M=1)[0] == 0.0


def test_permutation_summary_leaves_a_component_without_evidence_undefined() -> None:
    """No observed statistic, or no finite null draw, gives a NaN p-value and threshold, not a number."""
    null = np.array([[0.1, np.nan, 0.2], [0.3, np.nan, 0.1], [0.2, np.nan, 0.4]])
    observed = np.array([0.25, 0.5, np.nan])
    p_values, thresholds = _permutation_summary(null, observed, alpha=0.05)
    assert p_values[0] == pytest.approx((1 + 1) / (3 + 1))  # one of three null draws beats 0.25
    assert np.isnan(p_values[1:]).all()
    assert np.isnan(thresholds[1:]).all()


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
        flipped = getattr(folds[2], attribute).copy()  # pandas 3's to_numpy() is a read-only view
        flipped[:, 1] *= -1.0
        setattr(folds[2], attribute, flipped)
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


@pytest.mark.parametrize(
    ("rows_x", "rows_y", "message"),
    [
        (30, 29, r"^X and Y must have the same number of rows; got 30 and 29\.$"),
        (2, 2, r"^No components can be evaluated; the data or the folds are too small\.$"),
    ],
    ids=["y-one-row-short", "two-rows-in-two-folds"],
)
def test_data_that_cannot_be_cross_validated_is_refused(rows_x: int, rows_y: int, message: str) -> None:
    """X and Y must pair up row for row, and each fold must keep enough rows to fit a component."""
    X, Y = _two_component_data(n=30)
    with pytest.raises(ValueError, match=message):
        compare_cv_criteria(X.iloc[:rows_x], Y.iloc[:rows_y], cv=2)


# --- The fold models' own space, and the T2 rule (#1029) -----------------------------------


def _robust_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two latent variables, with a few rows that pull the means away from PRM's robust centre.

    PRM's reweighting on these rows is sensitive: inputs that differ in their last digits
    can take 44 passes rather than 39 and land 0.26 apart. The blocks are returned as the
    module reads them, as float copies, so that a fold PRM fitted here and the module's own
    start from the same numbers, summed in the same order, on every numpy and pandas.
    """
    X, Y = _two_component_data(n=40, k=6)
    X.iloc[:5] += 4.0
    Y.iloc[:3] += 3.0
    return X.astype(float), Y.astype(float)


def test_prm_held_out_rows_are_predicted_as_each_fold_model_predicts_them() -> None:
    """A PRM fold model centres and scales its autoscaled rows again, at its robust centre (#1029).

    The reference is each fold's PRM predicting its held-out rows, after the fold's own
    autoscaling, back in Y's units. The held-out rows were read in the fold's autoscaled
    space instead of the PRM's, for a PRESS of 72.67 against the fold models' 73.11.
    """
    X, Y = _robust_data()
    cv = KFold(5, shuffle=True, random_state=0)
    result = PRM.compare_cv_criteria(X, Y, max_components=2, cv=cv, n_permutations=9, n_cv_permutations=0)
    press = 0.0
    for train, test in cv.split(X):
        sx, sy = MCUVScaler().fit(X.iloc[train]), MCUVScaler().fit(Y.iloc[train])
        fold = PRM(n_components=2).fit(sx.transform(X.iloc[train]), sy.transform(Y.iloc[train]))
        predicted = sy.inverse_transform(fold.predict(sx.transform(X.iloc[test])))
        press += float(((Y.iloc[test].to_numpy() - predicted.to_numpy()) ** 2).sum())
    assert result.press.iloc[-1] == pytest.approx(press, rel=1e-10)


@pytest.mark.parametrize("scope", ["global", "local"])
def test_prm_pseudo_validation_set_reproduces_scores_and_spe(scope: str) -> None:
    """The full-data PRM gives X_pv the D-scaled fold scores and the fold SPE, exactly, in its own space (#1029).

    X_pv was mapped back to the data's units by an autoscaler fitted to X, not by the PRM's
    robust scaler, and the fold models' held-out rows were not centred at their robust
    centres: the scores the full-data model gave X_pv were off by up to 5.6.
    """
    X, Y = _robust_data()
    pv = PRM.pseudo_validation_set(X, Y, n_components=2, scope=scope, random_state=0)
    diagnostics = pv.global_model.diagnose(pv.X_pv)
    np.testing.assert_allclose(diagnostics.scores.to_numpy(), pv.scores.to_numpy(), atol=1e-10)
    np.testing.assert_allclose(diagnostics.spe.to_numpy(), pv.local_spe.to_numpy(), atol=1e-10)
    if scope == "local":  # each fold PRM's own SPE of its held-out rows, after the fold's autoscaling
        for train, test in pv.cv_splits:
            sx = MCUVScaler().fit(X.iloc[train])
            fold = PRM(n_components=2).fit(sx.transform(X.iloc[train]), MCUVScaler().fit_transform(Y.iloc[train]))
            spe = fold.diagnose(sx.transform(X.iloc[test])).spe.to_numpy()
            np.testing.assert_allclose(pv.local_spe.iloc[test].to_numpy(), spe, atol=1e-10)


def test_pseudo_validation_t2_leaves_out_a_component_with_no_spread() -> None:
    """The held-out T2 counts only the components the full-data model's T2 counts (#1029).

    Thirty rows on a plane in units of 1e12, one of them nudged 1e3 off it, fitted unscaled:
    the third component's score sd, 181, is below 1.5e-8 of the first's, 2.8e12, so the
    model's own T2 leaves it out. The held-out T2 divided by it all the same, and the nudged
    row, which its fold model never saw, got a T2 of 1057 at three components, an alarm.
    Its T2 is ``t_1^2 / s_1^2 + t_2^2 / s_2^2`` of the pseudo-validation scores, by hand.
    """
    rng = np.random.default_rng(0)
    plane = rng.standard_normal((30, 2)) @ rng.standard_normal((2, 4))
    plane = (plane - plane.mean(axis=0)) * 1e12
    plane[0] += 1e3 * np.linalg.svd(plane)[2][2]  # a direction off the plane
    X = pd.DataFrame(plane - plane.mean(axis=0), columns=list("abcd"))
    noise = 0.1 * np.random.default_rng(1).standard_normal(30)
    y = pd.DataFrame({"y": X.to_numpy() @ [1e-12, 2e-12, 0.0, 0.0] + noise})
    y -= y.mean()

    def fit(rows: np.ndarray) -> PLS:
        with pytest.warns(SpecificationWarning, match="nothing left in Y that X can explain"):
            return PLS(n_components=3, scale=False, warn_on_uncentred=False).fit(X.iloc[rows], y.iloc[rows])

    model = fit(np.arange(30))
    sd = model.scaling_factor_for_scores_.to_numpy()
    assert sd[2] < 1.5e-8 * sd[0]
    folds = []
    for train, test in KFold(5).split(X):
        fold = fit(train)
        x_test = X.iloc[test].to_numpy()
        weights, x_loadings = fold.x_weights_.to_numpy(), fold.x_loadings_.to_numpy()
        folds.append(
            _Fold(
                train=train,
                test=test,
                x_test=x_test,
                scores=_nipals_scores(x_test, weights, x_loadings),
                y_centre=np.zeros(1),
                y_scale=np.ones(1),
                weights=weights,
                direct_weights=fold.direct_weights_.to_numpy(),
                x_loadings=x_loadings,
                y_loadings=fold.y_loadings_.to_numpy(),
            )
        )
    pv_scores = _procrustes_scores(model, folds)[0]
    assert (pv_scores[0, 2] / sd[2]) ** 2 > 1000.0  # what the third component added
    t2 = np.cumsum((pv_scores[:, :2] / sd[:2]) ** 2, axis=1)
    t2 = np.c_[t2, t2[:, 1]]
    limits = [hotellings_t2_limit(conf_level=0.95, n_components=a, n_rows=30) for a in (1, 2, 3)]
    alarms = _monitoring_criteria(model, folds, conf_level=0.95, alpha=0.05).t2_alarm
    np.testing.assert_allclose(alarms, [np.mean(t2[:, a] > limits[a]) for a in range(3)])
    assert alarms[2] == 0.0


@pytest.mark.parametrize("scale", [False, 0, None, np.False_])
def test_a_false_scale_is_refused(scale: object) -> None:
    """Any false ``scale`` fits the full-data model unscaled, against autoscaled folds: each is refused (#1029).

    The check was ``scale is False``, which ``0``, ``None`` and ``np.False_`` passed.
    """
    X, Y = _two_component_data(n=30)
    message = (
        f"^scale={re.escape(repr(scale))} is not supported: each training fold is autoscaled, and the model "
        r"fitted to all rows must live in the same space for the angles and the Procrustes step to compare "
        r"them\. Pass X and Y on their raw scale\.$"
    )
    with pytest.raises(ValueError, match=message):
        compare_cv_criteria(X, Y, scale=scale)
    with pytest.raises(ValueError, match=message):
        pseudo_validation_set(X, Y, n_components=2, scale=scale)


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
    in_sample = _heldout_pass(_fit_folds(lambda: PLS(n_components=4), X, Y, everything), Y.to_numpy(), 4)
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


@pytest.fixture(scope="module")
def noise_result() -> object:
    """compare_cv_criteria on a y unrelated to X: the Y-related rules recommend 0 components."""
    X, _ = _two_component_data()
    y = pd.Series(np.random.default_rng(3).standard_normal(len(X)), name="noise")
    return compare_cv_criteria(X, y, max_components=2, random_state=0, n_permutations=39, n_cv_permutations=19)


def test_cv_criteria_plot_names_but_does_not_draw_a_recommendation_of_zero(noise_result: object) -> None:
    """A rule that recommends no components is captioned in its panel, with no line at 0."""
    pytest.importorskip("plotly.graph_objects")
    from process_improve.multivariate.plots import cv_criteria_plot

    picks = noise_result.recommendations["n_components"]
    assert picks[["score_correlation", "covariance_permutation", "subspace_stability"]].eq(0).all()
    fig = cv_criteria_plot(noise_result)
    captions = [annotation.text for annotation in fig.layout.annotations]
    assert "Recommended: 0 (held-out r, covariance)" in captions
    assert "Recommended: 0 (subspace)" in captions
    # Only panel 1's rules recommend a component, so theirs is the only line drawn.
    assert [shape.x0 for shape in fig.layout.shapes] == [picks["q2_max"]]


def test_cv_criteria_plot_has_no_spe_caption_without_a_finite_limit_ratio(noise_result: object) -> None:
    """Panel 4's caption quotes the refitted SPE limit ratio, so with no finite ratio there is none."""
    pytest.importorskip("plotly.graph_objects")
    from process_improve.multivariate.plots import cv_criteria_plot

    without_ratio = copy.copy(noise_result)
    without_ratio.table = noise_result.table.assign(pv_spe_limit_ratio=np.nan)
    with_ratio = [annotation.text for annotation in cv_criteria_plot(noise_result).layout.annotations]
    without = [annotation.text for annotation in cv_criteria_plot(without_ratio).layout.annotations]
    assert any(text.startswith("SPE limit refitted") for text in with_ratio)
    assert not any(text.startswith("SPE limit refitted") for text in without)


# --- compare_cv_criteria and pseudo_validation_set written out by hand (#1029) --------------


def _observed_mean_sd(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each column's mean and ``ddof=1`` standard deviation over its observed cells; 1 for a constant column."""
    columns = [column[~np.isnan(column)] for column in block.T]
    sds = np.array([np.std(c, ddof=1) for c in columns])
    return np.array([np.mean(c) for c in columns]), np.where(sds > 0, sds, 1.0)


def _nipals_row_scores(row: np.ndarray, weights: np.ndarray, x_loadings: np.ndarray) -> np.ndarray:
    """Return a row's scores as NIPALS scores a training row: its observed cells regressed on each weight in turn."""
    seen = ~np.isnan(row)
    residual = row[seen]
    scores = np.zeros(weights.shape[1])
    for a in range(weights.shape[1]):
        w = weights[seen, a]
        scores[a] = residual @ w / (w @ w) if w @ w > 0 else 0.0
        residual = residual - scores[a] * x_loadings[seen, a]
    return scores


def _folds_by_hand(X: pd.DataFrame, Y: pd.DataFrame, splits: list, n_components: int) -> list[dict]:
    """Each training fold autoscaled over its observed cells and fitted; its held-out rows scored by hand."""
    x, y = X.to_numpy(dtype=float), Y.to_numpy(dtype=float)
    folds = []
    for train, test in splits:
        x_centre, x_scale = _observed_mean_sd(x[train])
        y_centre, y_scale = _observed_mean_sd(y[train])
        model = PLS(n_components=n_components, scale=False, warn_on_uncentred=False).fit(
            (x[train] - x_centre) / x_scale, (y[train] - y_centre) / y_scale
        )
        weights, x_loadings = model.x_weights_.to_numpy(), model.x_loadings_.to_numpy()
        x_test = (x[test] - x_centre) / x_scale
        folds.append(
            {
                "train": train,
                "test": test,
                "x_test": x_test,
                "scores": np.array([_nipals_row_scores(row, weights, x_loadings) for row in x_test]),
                "y_centre": y_centre,
                "y_scale": y_scale,
                "weights": weights,
                "direct_weights": model.direct_weights_.to_numpy(),
                "x_loadings": x_loadings,
                "y_loadings": model.y_loadings_.to_numpy(),
            }
        )
    return folds


def _held_out_by_hand(folds: list[dict], y: np.ndarray, n_components: int) -> dict:
    """PRESS, the slope ratio and the pooled held-out t-u correlation, row by row, over the observed Y cells."""
    A, (N, M), G = n_components, y.shape, len(folds)
    out = {
        "press_y": np.zeros((A, M)),
        "per_obs_sse": np.full((A, N), np.nan),
        "per_fold_press": np.zeros((A, G)),
        "per_fold_rmse": np.zeros((A, G)),
        "press_baseline": 0.0,
    }
    numerator, weight, tu, tt, uu = (np.zeros(A) for _ in range(5))
    for k, fold in enumerate(folds):
        cells = 0
        for row, i in enumerate(fold["test"]):
            seen = ~np.isnan(y[i])
            cells += int(seen.sum())
            residual = y[i, seen] - fold["y_centre"][seen]
            out["press_baseline"] += float(residual @ residual)
            for a in range(A):
                t = fold["scores"][row, a]
                c = fold["y_loadings"][seen, a]
                c_units = c * fold["y_scale"][seen]
                numerator[a] += t * (residual @ c_units)
                weight[a] += t**2 * (c_units @ c_units)
                if seen.any():
                    u = (residual / fold["y_scale"][seen]) @ c / (c @ c) if c @ c > 0 else 0.0
                    tu[a], tt[a], uu[a] = tu[a] + t * u, tt[a] + t * t, uu[a] + u * u
                residual = residual - t * c_units
                out["press_y"][a, seen] += residual**2
                out["per_fold_press"][a, k] += residual @ residual
                if seen.any():
                    out["per_obs_sse"][a, i] = residual @ residual
        out["per_fold_rmse"][:, k] = np.sqrt(out["per_fold_press"][:, k] / max(cells, 1))
    out["slope_ratio"] = np.where(weight > 0, numerator / np.where(weight > 0, weight, 1.0), np.nan)
    out["slope_weight"] = weight
    out["r_cv"] = tu / np.sqrt(tt * uu)
    return out


def _largest_singular_value(matrix: np.ndarray) -> float:
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def _covariance_pvalues_by_hand(
    x: np.ndarray, y: np.ndarray, model: PLS, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Freedman-Lane: the largest singular value of X_a'Y_a against row permutations of Y_a, drawn 256 at a time."""
    x_seen, y_seen = ~np.isnan(x), ~np.isnan(y)
    x_a, y_a = np.where(x_seen, x, 0.0), np.where(y_seen, y, 0.0)
    scores, x_loadings, y_loadings = (
        model.scores_.to_numpy(),
        model.x_loadings_.to_numpy(),
        model.y_loadings_.to_numpy(),
    )
    p_values = []
    for a in range(scores.shape[1]):
        statistic = _largest_singular_value(x_a.T @ y_a)
        exceed, done = 0, 0
        while done < n_permutations:
            batch = min(256, n_permutations - done)
            for order in rng.permuted(np.tile(np.arange(len(x)), (batch, 1)), axis=1):
                exceed += _largest_singular_value(x_a.T @ y_a[order]) >= statistic
            done += batch
        p_values.append((exceed + 1) / (n_permutations + 1))
        t = scores[:, [a]]
        x_a = np.where(x_seen, x_a - t @ x_loadings[:, [a]].T, 0.0)
        y_a = np.where(y_seen, y_a - t @ y_loadings[:, [a]].T, 0.0)
    return np.array(p_values)


def _score_correlation_by_hand(x: np.ndarray, y: np.ndarray, splits: list, n_components: int) -> np.ndarray:
    """Return the held-out t-u correlation with each training fold's missing cells at its mean (0, autoscaled).

    The training blocks are then complete, and the kernel algorithm the module uses gives the
    NIPALS model, so the fold models here are PLS fitted to the mean-filled blocks.
    """
    A = n_components
    tu, tt, uu = np.zeros(A), np.zeros(A), np.zeros(A)
    for train, test in splits:
        x_centre, x_scale = _observed_mean_sd(x[train])
        y_centre, y_scale = _observed_mean_sd(y[train])
        model = PLS(n_components=A, scale=False, warn_on_uncentred=False).fit(
            np.nan_to_num((x[train] - x_centre) / x_scale), np.nan_to_num((y[train] - y_centre) / y_scale)
        )
        weights, x_loadings, y_loadings = (
            model.x_weights_.to_numpy(),
            model.x_loadings_.to_numpy(),
            model.y_loadings_.to_numpy(),
        )
        for i in test:
            t_row = _nipals_row_scores((x[i] - x_centre) / x_scale, weights, x_loadings)
            seen = ~np.isnan(y[i])
            if not seen.any():
                continue
            residual = (y[i, seen] - y_centre[seen]) / y_scale[seen]
            for a in range(A):
                c = y_loadings[seen, a]
                u = residual @ c / (c @ c) if c @ c > 0 else 0.0
                tu[a], tt[a], uu[a] = tu[a] + t_row[a] * u, tt[a] + t_row[a] ** 2, uu[a] + u * u
                residual = residual - t_row[a] * c
    return tu / np.sqrt(tt * uu)


def _jackknife_by_hand(angles: np.ndarray) -> np.ndarray:
    """``atan(sqrt(G - 1) rms(tan theta_k))`` over the G folds (rows), in degrees."""
    tangent = np.tan(np.radians(angles))
    return np.degrees(np.arctan(np.sqrt(len(angles) - 1) * np.sqrt(np.mean(tangent**2, axis=0))))


def _largest_principal_angle(first: np.ndarray, second: np.ndarray) -> float:
    """Return the largest principal angle between two spans, in degrees, from the SVD of orthonormal bases."""
    cosines = np.linalg.svd(np.linalg.qr(first)[0].T @ np.linalg.qr(second)[0], compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(cosines.min(), 0.0, 1.0))))


def _angles_by_hand(global_weights: np.ndarray, folds: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    A = global_weights.shape[1]
    component = np.array(
        [
            [
                np.degrees(
                    np.arccos(
                        min(
                            1.0,
                            abs(fold["weights"][:, a] @ global_weights[:, a])
                            / (np.linalg.norm(fold["weights"][:, a]) * np.linalg.norm(global_weights[:, a])),
                        )
                    )
                )
                for a in range(A)
            ]
            for fold in folds
        ]
    )
    subspace = np.array(
        [
            [_largest_principal_angle(global_weights[:, : a + 1], fold["weights"][:, : a + 1]) for a in range(A)]
            for fold in folds
        ]
    )
    return _jackknife_by_hand(component), _jackknife_by_hand(subspace)


def _procrustes_by_hand(model: PLS, folds: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each fold model signed as the full one, its held-out scores times D, and their local squared SPE."""
    direct, y_loadings = model.direct_weights_.to_numpy(), model.y_loadings_.to_numpy()
    N, A = sum(len(fold["test"]) for fold in folds), direct.shape[1]
    scores, squared_spe, ratios = np.zeros((N, A)), np.zeros((N, A)), np.zeros((len(folds), A))
    for k, fold in enumerate(folds):
        signs = np.array([-1.0 if fold["direct_weights"][:, a] @ direct[:, a] < 0 else 1.0 for a in range(A)])
        ratios[k] = [
            signs[a] * fold["y_loadings"][:, a] @ y_loadings[:, a] / (y_loadings[:, a] @ y_loadings[:, a])
            for a in range(A)
        ]
        scores[fold["test"]] = fold["scores"] * signs * ratios[k]
        for a in range(A):
            residual = fold["x_test"] - fold["scores"][:, : a + 1] @ fold["x_loadings"][:, : a + 1].T
            squared_spe[fold["test"], a] = [np.sum(r[~np.isnan(r)] ** 2) for r in residual]
    return scores, squared_spe, ratios


def _compare_by_hand(X: pd.DataFrame, Y: pd.DataFrame, splits: list, settings: dict) -> dict:
    """Return compare_cv_criteria written out: its table and its other fields.

    ``settings`` holds ``n_components``, the ``seed``, the two counts of permutations, ``alpha``
    and ``conf_level``.
    """
    A, x, y = settings["n_components"], X.to_numpy(dtype=float), Y.to_numpy(dtype=float)
    n_permutations, n_cv_permutations = settings["n_permutations"], settings["n_cv_permutations"]
    alpha, conf_level = settings["alpha"], settings["conf_level"]
    N, K = x.shape
    rng_vdv, rng_perm, rng_null = np.random.default_rng(settings["seed"]).spawn(3)
    folds = _folds_by_hand(X, Y, splits, A)
    held = _held_out_by_hand(folds, y, A)
    model = PLS(n_components=A).fit(X, Y)
    seen = ~np.isnan(y)
    # Prediction, over the observed Y cells.
    tss = np.array([np.sum((c[~np.isnan(c)] - np.mean(c[~np.isnan(c)])) ** 2) for c in y.T])
    q2_per_target = 1.0 - held["press_y"] / tss
    q2_total = 1.0 - held["press_y"].sum(axis=1) / tss.sum()
    G = len(splits)
    rmsecv = np.sqrt(held["press_y"].sum(axis=1) / seen.sum())
    vdv_recommended, vdv_p = _vandervoet_randomization(
        held["per_obs_sse"], total_rmsecv=rmsecv, n_permutations=n_permutations, alpha=alpha, random_state=rng_vdv
    )
    cv_anova = np.full(A, np.nan)
    if Y.shape[1] == 1:
        for a in range(1, A + 1):
            df_res = seen.sum() - 1 - 2 * a
            if df_res > 0:
                cv_anova[a - 1] = f_dist.sf(q2_total[a - 1] / (1 - q2_total[a - 1]) * df_res / (2 * a), 2 * a, df_res)
    # The held-out t-u correlation against its permutation null.
    rows_with_y = seen.any(axis=1)
    if n_cv_permutations:
        null = np.array(
            [_score_correlation_by_hand(x, y[rng_null.permutation(N)], splits, A) for _ in range(n_cv_permutations)]
        )
        observed = _score_correlation_by_hand(x, y, splits, A)
        r_cv_p = (np.sum(null >= observed, axis=0) + 1) / (n_cv_permutations + 1)
        r_cv_threshold = np.quantile(null, 1 - alpha, axis=0, method="higher")
    else:
        r_cv_p = norm.sf(held["r_cv"] * np.sqrt(rows_with_y.sum()))
        r_cv_threshold = np.full(A, norm.ppf(1 - alpha) / np.sqrt(rows_with_y.sum()))
    x_mean, x_sd = _observed_mean_sd(x)
    y_mean, y_sd = _observed_mean_sd(y)
    cov_perm_p = _covariance_pvalues_by_hand((x - x_mean) / x_sd, (y - y_mean) / y_sd, model, n_permutations, rng_perm)
    angle_component, angle_subspace = _angles_by_hand(model.x_weights_.to_numpy(), folds)
    # Monitoring, on the pseudo-validation rows.
    pv_scores, squared_spe, ratios = _procrustes_by_hand(model, folds)
    sd = model.scaling_factor_for_scores_.to_numpy()
    t2 = np.cumsum((pv_scores / sd) ** 2, axis=1)
    to_complete = K / (~np.isnan(x)).sum(axis=1)
    full_limits = np.array([spe_calculation(model.spe_.iloc[:, a].to_numpy(), conf_level=conf_level) for a in range(A)])
    pv_limits = np.array(
        [spe_calculation(np.sqrt(squared_spe[:, a] * to_complete), conf_level=conf_level) for a in range(A)]
    )
    t, u = model.scores_.to_numpy()[rows_with_y], model.y_scores_.to_numpy()[rows_with_y]
    table = pd.DataFrame(
        {
            "r2y": model.r2_cumulative_.to_numpy(),
            "q2y": np.nanmean(np.where(tss > 0, q2_per_target, np.nan), axis=1),
            "q2y_se": np.std(held["per_fold_press"], axis=1, ddof=1) / np.sqrt(G) * G / tss.sum(),
            "vdv_p": vdv_p,
            "cv_anova_p": cv_anova,
            "r_train": np.sum(t * u, axis=0) / np.sqrt(np.sum(t * t, axis=0) * np.sum(u * u, axis=0)),
            "r_cv": held["r_cv"],
            "r_cv_threshold": r_cv_threshold,
            "r_cv_p": r_cv_p,
            "slope_ratio": held["slope_ratio"],
            "cov_perm_p": cov_perm_p,
            "angle_component_deg": angle_component,
            "angle_subspace_deg": angle_subspace,
            "d_ratio_median": np.median(ratios, axis=0),
            "d_ratio_min": ratios.min(axis=0),
            "pv_spe_alarm_rate": np.mean(np.sqrt(squared_spe) > full_limits, axis=0),
            "pv_t2_alarm_rate": [
                np.mean(t2[:, a] > hotellings_t2_limit(conf_level=conf_level, n_components=a + 1, n_rows=N))
                for a in range(A)
            ],
            "pv_spe_limit_ratio": pv_limits / full_limits,
        },
        index=pd.Index(range(1, A + 1), name="n_components"),
    )
    return {
        "table": table,
        "press": held["press_y"].sum(axis=1),
        "press_baseline": held["press_baseline"],
        "d_ratios": ratios,
        "spe_limits": np.c_[full_limits, pv_limits],
        "rmsecv": rmsecv,
        "se_rmsecv": np.std(held["per_fold_rmse"], axis=1, ddof=1) / np.sqrt(G),
        "alarm_rate_upper": binom.ppf(1 - alpha, N, 1 - conf_level) / N,
        "per_obs_sse": held["per_obs_sse"],
        "vdv_recommended": vdv_recommended,
    }


def _global_folds_by_hand(X: pd.DataFrame, Y: pd.DataFrame, splits: list, n_components: int) -> list[dict]:
    """Fold models fitted, unscaled and uncentred, to the training rows of the fully autoscaled blocks."""
    x, y = X.to_numpy(dtype=float), Y.to_numpy(dtype=float)
    (x_mean, x_sd), (y_mean, y_sd) = _observed_mean_sd(x), _observed_mean_sd(y)
    xs, ys = (x - x_mean) / x_sd, (y - y_mean) / y_sd
    folds = []
    for train, test in splits:
        model = PLS(n_components=n_components, scale=False, warn_on_uncentred=False).fit(xs[train], ys[train])
        weights, x_loadings = model.x_weights_.to_numpy(), model.x_loadings_.to_numpy()
        folds.append(
            {
                "train": train,
                "test": test,
                "x_test": xs[test],
                "scores": np.array([_nipals_row_scores(row, weights, x_loadings) for row in xs[test]]),
                "y_centre": y_mean,
                "y_scale": y_sd,
                "weights": weights,
                "direct_weights": model.direct_weights_.to_numpy(),
                "x_loadings": x_loadings,
                "y_loadings": model.y_loadings_.to_numpy(),
            }
        )
    return folds


def _pseudo_validation_by_hand(X: pd.DataFrame, Y: pd.DataFrame, splits: list, n_components: int, scope: str) -> dict:
    """Return X_pv = T_pv P' + E, seeded with 0: E mixes each fold's held-out rows at random, at the fold's SPE."""
    A = n_components
    (rng_pcv,) = np.random.default_rng(0).spawn(1)
    model = PLS(n_components=A).fit(X, Y)
    folds = (_global_folds_by_hand if scope == "global" else _folds_by_hand)(X, Y, splits, A)
    scores, squared_spe, ratios = _procrustes_by_hand(model, folds)
    direct, x_loadings = model.direct_weights_.to_numpy(), model.x_loadings_.to_numpy()
    x_pv = np.zeros(X.shape)
    for fold in folds:
        test = fold["test"]
        mixed = rng_pcv.standard_normal((len(test), len(test))) @ np.nan_to_num(fold["x_test"])
        # Each column scaled to unit length, and each residual row to the fold's SPE; zeros stay zeros.
        norms = np.linalg.norm(mixed, axis=0)
        mixed = np.divide(mixed, norms, out=np.zeros_like(mixed), where=norms > 0)
        residual = mixed - mixed @ direct @ x_loadings.T
        target, norms = np.sqrt(squared_spe[test, A - 1]), np.linalg.norm(residual, axis=1)
        factor = np.divide(target, norms, out=np.zeros_like(target), where=norms > 0)
        x_pv[test] = scores[test] @ x_loadings.T + residual * factor[:, np.newaxis]
    x_pv[X.isna().to_numpy()] = np.nan
    x_mean, x_sd = _observed_mean_sd(X.to_numpy(dtype=float))
    return {
        "X_pv": x_pv * x_sd + x_mean,
        "scores": scores,
        "local_spe": np.sqrt(squared_spe[:, A - 1]),
        "d_ratios": ratios,
    }


def _leading_by_hand(passes: np.ndarray, angles: np.ndarray | None = None, threshold: float = 30.0) -> int:
    """Leading components that pass; with ``angles``, a first unstable span whose pair is stable counts as two."""
    a = 0
    while a < len(passes):
        if passes[a]:
            a += 1
        elif angles is not None and a + 1 < len(passes) and angles[a] >= threshold > angles[a + 1]:
            a += 2
        else:
            break
    return a


def _recommended_by_hand(hand: dict, alpha: float = 0.05, threshold: float = 30.0) -> dict:
    table, rmsecv = hand["table"], hand["rmsecv"]
    best = int(np.argmin(rmsecv))
    return {
        "q2_max": int(np.argmax(table["q2y"].to_numpy())) + 1,
        "q2_1se": int(np.flatnonzero(rmsecv <= rmsecv[best] + hand["se_rmsecv"][best])[0]) + 1,
        "van_der_voet": hand["vdv_recommended"],
        "score_correlation": _leading_by_hand(table["r_cv_p"].to_numpy() < alpha),
        "covariance_permutation": _leading_by_hand(table["cov_perm_p"].to_numpy() < alpha),
        "subspace_stability": _leading_by_hand(
            table["angle_subspace_deg"].to_numpy() < threshold, table["angle_subspace_deg"].to_numpy(), threshold
        ),
    }


def _assert_near(actual: object, expected: object, rtol: float = 1e-7) -> None:
    """Agreement to ``rtol`` of each value, with an absolute floor of ``rtol`` times the largest one.

    The module's fold models and these differ by rounding in the scaling, and NIPALS with
    several responses iterates to a tolerance, so two routes agree to about 1e-8.
    """
    expected = np.asarray(expected, dtype=float)
    floor = rtol * (float(np.nanmax(np.abs(expected))) if np.isfinite(expected).any() else 0.0)
    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=rtol, atol=floor)


def _criteria_case(case: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return a data set for the by-hand comparisons, and its count of score-correlation permutations."""
    if case == "complete":
        return (*_two_component_data(n=40, k=6), 1)  # one permutation is enough to take the permutation route
    if case == "a constant and a small target":  # no Q2 for the constant, and a TSS below 1 for the other
        X, Y = _two_component_data(n=40, k=6)
        return X, pd.DataFrame({"small": 0.01 * Y["y0"], "constant": 5.0}), 0
    X, Y = _with_gaps(*_two_component_data(n=40, k=6, m=3 if case == "gaps, three targets" else 1), fraction=0.1)
    Y.iloc[5] = np.nan  # no response: it helps fit X and is in no Y criterion
    X.index = Y.index = [f"run{i}" for i in range(40)]
    return X, Y, 9 if case == "gaps, three targets" else 0


@pytest.mark.parametrize(
    "case", ["complete", "gaps, three targets", "gaps, normal approximation", "a constant and a small target"]
)
def test_compare_cv_criteria_by_hand(case: str) -> None:
    """Every number compare_cv_criteria reports, against the procedure written out row by row.

    Folds autoscaled over their observed cells; held-out rows scored as NIPALS scores its
    rows; PRESS, the slope ratio and the held-out t-u correlation over the observed Y cells;
    Q2 and its SE, RMSECV, CV-ANOVA; the score-correlation null with mean-filled training
    cells; the Freedman-Lane covariance test, drawn 256 permutations at a time; jackknifed
    weight angles; Procrustes D ratios, alarm rates and SPE limits; and the recommendations.
    """
    X, Y, n_cv = _criteria_case(case)
    cv = KFold(5, shuffle=True, random_state=0)
    # One case away from the default levels and threshold, so each must reach its criterion.
    levels = {"alpha": 0.1, "conf_level": 0.9, "angle_threshold": 20.0} if case == "complete" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a row without a response divides 0 by 0, silently
        result = compare_cv_criteria(
            X, Y, max_components=3, cv=cv, random_state=4, n_permutations=300, n_cv_permutations=n_cv, **levels
        )
    levels = {"alpha": 0.05, "conf_level": 0.95, "angle_threshold": 30.0, **levels}
    settings = {"n_components": 3, "seed": 4, "n_permutations": 300, "n_cv_permutations": n_cv, **levels}
    hand = _compare_by_hand(X, Y, list(cv.split(X)), settings)
    assert list(result.table.columns) == list(hand["table"].columns)
    pd.testing.assert_index_equal(result.table.index, pd.RangeIndex(1, 4, name="n_components"), exact=True)
    assert (result.table.dtypes == np.float64).all()
    for column in hand["table"]:
        _assert_near(result.table[column], hand["table"][column], rtol=1e-6)
    index = pd.RangeIndex(1, 4, name="n_components")
    _assert_near(result.press, hand["press"], rtol=1e-6)
    pd.testing.assert_index_equal(result.press.index, index, exact=True)
    assert result.press.name == "PRESS"
    assert result.press_baseline == pytest.approx(hand["press_baseline"], rel=1e-12)
    _assert_near(result.d_ratios, hand["d_ratios"], rtol=1e-6)
    assert list(result.d_ratios.index) == ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
    pd.testing.assert_index_equal(result.d_ratios.columns, index, exact=True)
    _assert_near(result.spe_limits, hand["spe_limits"], rtol=1e-6)
    pd.testing.assert_index_equal(result.spe_limits.index, index, exact=True)
    assert list(result.spe_limits.columns) == ["full_data", "pseudo_validation"]
    assert result.alarm_rate_upper == hand["alarm_rate_upper"]
    assert (result.alpha, result.conf_level, result.angle_threshold) == tuple(levels.values())
    for (train, test), (train_by_hand, test_by_hand) in zip(result.cv_splits, cv.split(X), strict=True):
        np.testing.assert_array_equal(train, train_by_hand)
        np.testing.assert_array_equal(test, test_by_hand)
    assert type(result.global_model) is PLS
    assert result.global_model.n_components_ == 3
    _assert_near(result.global_model.beta_coefficients_, PLS(n_components=3).fit(X, Y).beta_coefficients_, rtol=1e-6)
    recommended = _recommended_by_hand(hand, levels["alpha"], levels["angle_threshold"])
    assert result.recommendations["n_components"].to_dict() == recommended


@pytest.mark.parametrize("scope", ["global", "local"])
@pytest.mark.parametrize("case", ["complete", "gaps, three targets"])
def test_pseudo_validation_set_by_hand(scope: str, case: str) -> None:
    """The pseudo-validation set and everything returned with it, against the construction written out."""
    X, Y, _ = _criteria_case(case)
    cv = KFold(5, shuffle=True, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pv = pseudo_validation_set(X, Y, n_components=2, cv=cv, scope=scope, random_state=0)
    hand = _pseudo_validation_by_hand(X, Y, list(cv.split(X)), 2, scope)
    index = pd.RangeIndex(1, 3, name="n_components")
    _assert_near(pv.X_pv, hand["X_pv"], rtol=1e-6)
    pd.testing.assert_index_equal(pv.X_pv.index, X.index)
    pd.testing.assert_index_equal(pv.X_pv.columns, X.columns)
    _assert_near(pv.scores, hand["scores"], rtol=1e-6)
    pd.testing.assert_index_equal(pv.scores.index, X.index)
    pd.testing.assert_index_equal(pv.scores.columns, index, exact=True)
    _assert_near(pv.local_spe, hand["local_spe"], rtol=1e-6)
    pd.testing.assert_index_equal(pv.local_spe.index, X.index)
    assert pv.local_spe.name == "SPE"
    _assert_near(pv.d_ratios, hand["d_ratios"], rtol=1e-6)
    assert list(pv.d_ratios.index) == ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
    pd.testing.assert_index_equal(pv.d_ratios.columns, index, exact=True)
    pd.testing.assert_frame_equal(pv.Y_pv, Y)
    assert pv.Y_pv is not Y
    for (train, test), (train_by_hand, test_by_hand) in zip(pv.cv_splits, cv.split(X), strict=True):
        np.testing.assert_array_equal(train, train_by_hand)
        np.testing.assert_array_equal(test, test_by_hand)
    assert type(pv.global_model) is PLS
    assert pv.global_model.n_components_ == 2
    assert pv.scope == scope


# --- Each step's edges, by hand (#1029) ------------------------------------------------------


def test_inputs_are_read_as_float_frames_labelled_as_x() -> None:
    """Integers become floats; arrays are numbered from 0; a 1-D response array takes X's row labels."""
    X = pd.DataFrame(np.arange(12).reshape(6, 2), index=list("abcdef"), columns=["p", "q"])
    X_df, Y_df = _as_frames(X, np.arange(6))
    pd.testing.assert_frame_equal(X_df, X.astype(float))
    pd.testing.assert_frame_equal(Y_df, pd.DataFrame({0: np.arange(6.0)}, index=list("abcdef")))
    X_df, Y_df = _as_frames(np.arange(12).reshape(6, 2), np.arange(12).reshape(6, 2))
    pd.testing.assert_frame_equal(X_df, pd.DataFrame(np.arange(12.0).reshape(6, 2)))
    pd.testing.assert_frame_equal(Y_df, pd.DataFrame(np.arange(12.0).reshape(6, 2)))
    _, Y_df = _as_frames(X, pd.Series(np.arange(6), index=list("abcdef"), name="y"))
    pd.testing.assert_frame_equal(Y_df, pd.DataFrame({"y": np.arange(6.0)}, index=list("abcdef")))
    _, Y_df = _as_frames(X, pd.DataFrame({"y": np.arange(6)}, index=list("abcdef")))
    pd.testing.assert_frame_equal(Y_df, pd.DataFrame({"y": np.arange(6.0)}, index=list("abcdef")))


def test_rows_and_columns_with_nothing_observed_are_refused_naming_five() -> None:
    """A row with no X, or a column of X or Y with no value, leaves nothing to work with; five are named."""
    X = pd.DataFrame(np.ones((10, 7)), columns=[f"c{i}" for i in range(7)])
    Y = pd.DataFrame({f"y{i}": np.arange(10.0) for i in range(7)})
    rows = X.copy()
    rows.iloc[:6] = np.nan
    with pytest.raises(
        ValueError, match=r"^Rows with every X value missing cannot be scored; remove them: \[0, 1, 2, 3, 4\]\.$"
    ):
        _as_frames(rows, Y)
    columns = X.copy()
    columns.iloc[:, :6] = np.nan
    with pytest.raises(
        ValueError, match=r"^Columns of X with every value missing: \['c0', 'c1', 'c2', 'c3', 'c4'\]\.$"
    ):
        _as_frames(columns, Y)
    Y.iloc[:, 1:] = np.nan
    with pytest.raises(
        ValueError, match=r"^Columns of Y with every value missing: \['y1', 'y2', 'y3', 'y4', 'y5'\]\.$"
    ):
        _as_frames(X, Y)


def test_a_training_fold_needs_two_observed_values_in_every_column() -> None:
    """Two observed values in a column of a training fold are enough; one is not, and five columns are named."""
    X = pd.DataFrame(np.arange(28.0).reshape(4, 7), columns=[f"c{i}" for i in range(7)])
    X.iloc[2:, 1:] = np.nan
    Y = pd.DataFrame({"y": [1.0, 2.0, 3.0, np.nan]})
    _check_fold_coverage(X, Y, [(np.array([0, 1, 2]), np.array([3]))])  # c1 to c6 and y: two each
    with pytest.raises(
        ValueError,
        match=r"^Training fold 2 has fewer than two observed values in X column\(s\) "
        r"\['c1', 'c2', 'c3', 'c4', 'c5'\]; use fewer folds or impute\.$",
    ):
        _check_fold_coverage(X, Y, [(np.array([0, 1, 2]), np.array([3])), (np.array([1, 2, 3]), np.array([0]))])
    with pytest.raises(
        ValueError, match=r"^Training fold 1 has fewer than two observed values in Y column\(s\) \['y'\]; use fewer"
    ):
        _check_fold_coverage(X.iloc[:, :1], Y, [(np.array([2, 3]), np.array([0, 1]))])


def test_an_integer_cv_is_a_shuffled_kfold_seeded_as_given_or_from_the_generator() -> None:
    """An integer seed goes to KFold unchanged; otherwise KFold gets a seed drawn below 2**31 from the generator."""
    X, Y = _two_component_data(n=12)

    def same(splits: list, expected: object) -> None:
        for (train, test), (train_by_hand, test_by_hand) in zip(splits, expected, strict=True):
            np.testing.assert_array_equal(train, train_by_hand)
            np.testing.assert_array_equal(test, test_by_hand)

    for seed in (7, np.int64(7)):
        splits = _partition_splits(4, X, Y, rng=np.random.default_rng(0), random_state=seed)
        same(splits, KFold(4, shuffle=True, random_state=7).split(X))
    drawn = int(np.random.default_rng(3).integers(2**31))
    for random_state in (None, np.random.default_rng(3)):
        splits = _partition_splits(np.int64(2), X, Y, rng=np.random.default_rng(3), random_state=random_state)
        same(splits, KFold(2, shuffle=True, random_state=drawn).split(X))
    with pytest.raises(ValueError, match=r"^cv must be >= 2 when given as an int; got 1\.$"):
        _partition_splits(1, X, Y, rng=np.random.default_rng(0), random_state=0)
    # A bool is not a count of folds: it goes to sklearn, which refuses KFold(n_splits=True).
    with pytest.raises(ValueError, match="k-fold cross-validation requires at least one train/test split"):
        _partition_splits(True, X, Y, rng=np.random.default_rng(0), random_state=0)


def test_a_splitter_must_hold_out_every_row_exactly_once() -> None:
    """A row held out twice or never is refused, in full, and so is a splitter with no split at all."""
    X, Y = _two_component_data(n=6)
    message = (
        r"^The cross-validation splitter must hold out every row exactly once \(a partition, such as KFold or "
        r"LeaveOneOut\)\. Repeated or random splitters \(RepeatedKFold, ShuffleSplit\) are not supported, because "
        r"the pooled held-out scores and the pseudo-validation set need one held-out copy of each row\.$"
    )
    rng = np.random.default_rng(0)
    twice = [(np.arange(3, 6), np.arange(3)), (np.arange(2), np.arange(2, 6))]
    never = [(np.arange(3, 6), np.arange(3)), (np.arange(3), np.arange(3, 5))]
    for cv in (twice, never):
        with pytest.raises(ValueError, match=message):
            _partition_splits(cv, X, Y, rng=rng, random_state=0)
    with pytest.raises(ValueError, match=message):
        _partition_splits([], X.iloc[:0], Y.iloc[:0], rng=rng, random_state=0)
    in_sample = [(np.arange(6), np.arange(6))]
    assert len(_partition_splits(in_sample, X, Y, rng=rng, random_state=0)) == 1


def test_component_cap_by_hand() -> None:
    """One fewer than the smallest training fold, at most the features, at most what is asked for."""
    splits = [(np.arange(5), np.arange(5, 8)), (np.arange(7), np.arange(7, 8))]
    assert _component_cap(splits, 10, None) == 4
    assert _component_cap(splits, 3, None) == 3
    assert [_component_cap(splits, 10, requested) for requested in (1, 2, 100)] == [1, 2, 4]
    with pytest.raises(ValueError, match=r"^max_components must be >= 1; got 0\.$"):
        _component_cap(splits, 10, 0)
    with pytest.raises(ValueError, match=r"^No components can be evaluated; the data or the folds are too small\.$"):
        _component_cap([(np.arange(1), np.arange(1, 3))], 10, None)


def test_nipals_scores_skip_a_component_with_no_observed_weight() -> None:
    """A row observed only where w_a is zero has no regression on it: its score is 0, silently, and the next is."""
    weights = np.array([[1.0, 0.0], [0.0, 0.6], [0.0, 0.8]])
    x_loadings = np.array([[1.0, 0.0], [0.5, 0.6], [0.0, 0.8]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = _nipals_scores(np.array([[np.nan, 3.0, 1.0], [2.0, 1.0, np.nan]]), weights, x_loadings)
    # Row 2: t_1 = 2, leaving 1 - 2 * 0.5 = 0 in column 2, so t_2 = 0 * 0.6 / 0.36 = 0.
    np.testing.assert_allclose(scores, [[0.0, (3.0 * 0.6 + 1.0 * 0.8) / 1.0], [2.0, 0.0]], atol=1e-15)


def test_correlation_about_origin_by_hand() -> None:
    assert _correlation_about_origin(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.0)
    assert _correlation_about_origin(np.array([1.0, -1.0]), np.array([1.0, 1.0])) == 0.0
    assert _correlation_about_origin(np.array([1.0, 1.0]), np.array([3.0, 1.0])) == pytest.approx(4.0 / np.sqrt(20.0))
    assert np.isnan(_correlation_about_origin(np.zeros(2), np.array([1.0, 2.0])))
    assert _correlation_about_origin(np.array([0.5]), np.array([-0.5])) == -1.0  # a product of lengths below 1


def _fold_by_hand(test: list, scores: list, y_loadings: list, centre: float = 1.0, scale: float = 2.0) -> _Fold:
    """Return a fold of one response whose held-out scores and Y loadings are given; X does not enter the pass."""
    scores_array = np.array(scores, dtype=float)
    return _Fold(
        train=np.array([], dtype=int),
        test=np.array(test),
        x_test=np.zeros((len(test), 1)),
        scores=scores_array,
        y_centre=np.array([centre]),
        y_scale=np.array([scale]),
        weights=np.zeros((1, scores_array.shape[1])),
        direct_weights=np.zeros((1, scores_array.shape[1])),
        x_loadings=np.zeros((1, scores_array.shape[1])),
        y_loadings=np.array([y_loadings], dtype=float),
    )


def test_held_out_pass_on_folds_built_by_hand() -> None:
    """One held-out cell, a fold with no observed response, and a component with a zero Y loading.

    Row 0, y = 4, scaled loading 0.5 (1 in units): residual 3, then 3 - 2 * 1 = 1; the slope
    numerator is 2 * 3 * 1 = 6 over a weight of 4, s = 1.5; u = (3 / 2) * 0.5 / 0.25 = 3. The
    second component's loading is 0: no weight, so no slope ratio, u = 0 and no correlation.
    Rows 1 and 2 have no response: no residual, an RMSE of 0 over their fold, no SSE.
    """
    folds = [_fold_by_hand([0], [[2.0, 1.0]], [0.5, 0.0]), _fold_by_hand([1, 2], [[1.0, 1.0], [3.0, 2.0]], [0.5, 0.0])]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no weight: 0 / 0, silently
        held = _heldout_pass(folds, np.array([[4.0], [np.nan], [np.nan]]), 2)
    np.testing.assert_array_equal(held.press_y, [[1.0], [1.0]])
    np.testing.assert_array_equal(held.per_obs_sse, [[1.0, np.nan, np.nan], [1.0, np.nan, np.nan]])
    np.testing.assert_array_equal(held.per_fold_press, [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(held.per_fold_rmse, [[1.0, 0.0], [1.0, 0.0]])
    assert held.press_baseline == 9.0
    np.testing.assert_array_equal(held.slope_ratio, [1.5, np.nan])
    np.testing.assert_array_equal(held.slope_weight, [4.0, 0.0])
    np.testing.assert_array_equal(held.r_cv, [1.0, np.nan])


def test_held_out_u_of_a_row_whose_responses_have_no_loading_is_zero() -> None:
    """Row 0 has only the second response, which the component does not load: its u is 0, not 0 / 0.

    Row 1 has the first: u = ((3 - 1) / 2) * 0.5 / 0.25 = 2, with t = 1. Pooled about the
    origin, t = (2, 1) and u = (0, 2) correlate 2 / sqrt(5 * 4).
    """
    fold = _Fold(
        train=np.array([], dtype=int),
        test=np.array([0, 1]),
        x_test=np.zeros((2, 1)),
        scores=np.array([[2.0], [1.0]]),
        y_centre=np.array([1.0, 1.0]),
        y_scale=np.array([2.0, 2.0]),
        weights=np.zeros((1, 1)),
        direct_weights=np.zeros((1, 1)),
        x_loadings=np.zeros((1, 1)),
        y_loadings=np.array([[0.5], [0.0]]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        held = _heldout_pass([fold], np.array([[np.nan, 5.0], [3.0, np.nan]]), 1)
    assert held.r_cv[0] == pytest.approx(2.0 / np.sqrt(20.0))


def test_autoscale_parameters_by_hand() -> None:
    """Means and ddof=1 sds over observed cells; no spread, one value, or none, scales by 1, and nothing warns."""
    values = np.array(
        [
            [1.0, 5.0, np.nan, 3.0, 2.0**-510],
            [3.0, 5.0, np.nan, np.nan, 0.0],
            [5.0, 5.0, np.nan, np.nan, 2.0**-511],
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        centre, scale = _autoscale_parameters(values)
    np.testing.assert_array_equal(centre, [3.0, 5.0, np.nan, 3.0, 2.0**-511])
    # The last column's sd is exactly 2**-511, the square root of the smallest normal float: no spread.
    np.testing.assert_array_equal(scale, [2.0, 1.0, 1.0, 1.0, 1.0])
    _, scale = _autoscale_parameters(values[:, [4]] * 2.0)
    assert scale[0] == 2.0**-510


def test_permutation_summary_counts_every_permutation() -> None:
    """The p-value's denominator counts every permutation, a NaN draw too; the threshold is a finite draw."""
    null = np.array([[0.1, 0.3], [np.nan, 0.1], [0.4, 0.4]])
    p_values, thresholds = _permutation_summary(null, np.array([0.3, 0.3]), alpha=0.05)
    np.testing.assert_array_equal(p_values, [(1 + 1) / (3 + 1), (2 + 1) / (3 + 1)])  # a tie counts against it
    np.testing.assert_array_equal(thresholds, [0.4, 0.4])
    _, thresholds = _permutation_summary(np.array([[0.1], [0.2], [0.3], [0.4]]), np.array([0.0]), alpha=0.5)
    assert thresholds[0] == 0.3  # the quantile at 0.5 of four draws, rounded up to a draw


def test_jackknife_angle_at_its_bounds() -> None:
    """A fold weight at right angles gives just under 90 degrees; a small angle is not clipped; one fold scales by 1."""
    right_angle = _jackknife_angle(np.full((3, 1), 90.0), 3)[0]
    assert 90.0 - 1e-6 < right_angle < 90.0
    small = _jackknife_angle(np.array([[0.5], [0.25]]), 2)[0]
    assert small == pytest.approx(np.degrees(np.arctan(np.sqrt(np.mean(np.tan(np.radians([0.5, 0.25])) ** 2)))))
    assert _jackknife_angle(np.array([[10.0, 20.0]]), 1) == pytest.approx([10.0, 20.0])


def test_weight_angles_of_a_fold_equal_to_the_full_model_are_zero() -> None:
    """A unit vector's product with itself can round above 1; its angle is 0, not arccos's NaN."""
    rng = np.random.default_rng(0)
    weights = next(
        w for w in (rng.standard_normal((5, 1)) for _ in range(1000)) if np.sum((w / np.linalg.norm(w)) ** 2) > 1.0
    )
    fold = _fold_by_hand([0], [[1.0]], [1.0])
    fold.weights = weights
    component, subspace = _weight_angles(weights, [fold, fold])
    np.testing.assert_array_equal(component, [0.0])
    assert subspace[0] == pytest.approx(0.0, abs=1e-6)


def test_cv_anova_by_hand_at_its_edges() -> None:
    """F = Q2 / (1 - Q2) df_res / df_reg; a Q2 at or below 0 has p = 1, one of 1 has p = 0, silently; NaN is skipped."""
    q2 = np.array([0.5, 1.0, 0.0, -0.5, np.nan, 0.3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p_values = _cv_anova_pvalues(q2, N=40, M=1)
    expected = [f_dist.sf(1.0 * 37 / 2, 2, 37), 0.0, 1.0, 1.0, np.nan, f_dist.sf(0.3 / 0.7 * 27 / 12, 12, 27)]
    np.testing.assert_allclose(p_values, expected, rtol=1e-12)
    assert np.isnan(_cv_anova_pvalues(np.array([0.5, 1.0]), N=5, M=1)[1])  # no residual degree of freedom


def test_cv_anova_at_the_last_residual_degree_of_freedom() -> None:
    """With N - 1 - 2a = 1 the F test has one residual degree of freedom; at 0 it has none."""
    p_values = _cv_anova_pvalues(np.array([0.5, 0.5]), N=6, M=1)
    np.testing.assert_allclose(p_values, [f_dist.sf(1.0 * 3 / 2, 2, 3), f_dist.sf(1.0 * 1 / 4, 4, 1)], rtol=1e-12)
    assert np.isnan(_cv_anova_pvalues(np.array([0.5, 0.5]), N=5, M=1)[1])
    assert np.isnan(_cv_anova_pvalues(np.array([0.5, -0.1]), N=5, M=1)[1])  # not the 1 of a Q2 at or below 0
    assert np.isnan(_cv_anova_pvalues(np.array([np.nan]), N=40, M=1)[0])


def test_settings_at_their_bounds_and_their_refusals_in_full() -> None:
    _CompareSettings(n_permutations=1, n_cv_permutations=0, alpha=0.5, conf_level=0.5, angle_threshold=90.0).validate()
    _CompareSettings(angle_threshold=0.5).validate()
    refusals = [
        ({"n_permutations": 0}, r"^n_permutations must be >= 1; got 0\.$"),
        ({"n_cv_permutations": -1}, r"^n_cv_permutations must be >= 0; got -1\.$"),
        ({"alpha": 0.0}, r"^alpha must lie in \(0, 1\); got 0\.0\.$"),
        ({"alpha": 1.0}, r"^alpha must lie in \(0, 1\); got 1\.0\.$"),
        ({"conf_level": 1.0}, r"^conf_level must lie in \(0, 1\); got 1\.0\.$"),
        ({"angle_threshold": 0.0}, r"^angle_threshold must lie in \(0, 90\] degrees; got 0\.0\.$"),
        ({"angle_threshold": 90.5}, r"^angle_threshold must lie in \(0, 90\] degrees; got 90\.5\.$"),
    ]
    for settings, message in refusals:
        with pytest.raises(ValueError, match=message):
            _CompareSettings(**settings).validate()


def test_a_component_without_a_full_data_y_loading_has_no_d_ratio_or_held_out_t2() -> None:
    """A zero full-data Y loading gives D = 0 / 0: no D ratio, so no pseudo-validation T2 from that component on.

    The SPE needs no D ratio, so its alarm rate stays. Nothing warns.
    """
    X, Y = _two_component_data(n=30)
    folds = _folds(X, Y, 2, cv=KFold(5))
    model = PLS(n_components=2).fit(X, Y)
    model.y_loadings_.iloc[:, 1] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        monitoring = _monitoring_criteria(model, folds, conf_level=0.95, alpha=0.05)
    assert np.isfinite(monitoring.d_ratios[:, 0]).all()
    assert np.isnan(monitoring.d_ratios[:, 1]).all()
    assert np.isfinite(monitoring.t2_alarm[0])
    assert np.isnan(monitoring.t2_alarm[1])
    assert np.isfinite(monitoring.spe_alarm).all()


def test_in_sample_criteria_from_a_single_fold_of_every_row() -> None:
    """A splitter whose one fold trains and tests on every row: the slope ratio is 1, and no spread across folds.

    One fold has no standard error across folds (NaN, without a warning), and its jackknife
    scales by sqrt(1), not sqrt(0).
    """
    X, Y = _two_component_data(n=30)
    every_row = [(np.arange(30), np.arange(30))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compare_cv_criteria(X, Y, max_components=2, cv=every_row, n_permutations=9, n_cv_permutations=0)
    np.testing.assert_allclose(result.table["slope_ratio"], 1.0, rtol=1e-8)
    assert result.table["q2y_se"].isna().all()
    np.testing.assert_allclose(result.table["q2y"], result.table["r2y"], rtol=1e-8)


def test_an_autoscaled_x_is_warned_about_in_the_words_of_this_function() -> None:
    """Each fold re-standardises X, so an X that arrives autoscaled loses the scaling it came in."""
    X, Y = _two_component_data(n=30)
    with pytest.warns(SpecificationWarning, match=r"re-standardises inside every CV fold.* comparison of their Q2 "):
        compare_cv_criteria(MCUVScaler().fit_transform(X), Y, max_components=1, n_permutations=9, n_cv_permutations=0)


def test_recommendations_by_rule_in_full() -> None:
    """Each rule's count from a table built by hand, its wording and its question; a p-value of alpha fails."""
    table = pd.DataFrame(
        {
            "q2y": [0.2, 0.5, 0.4],
            "r_cv_p": [0.01, 0.05, 0.01],
            "cov_perm_p": [0.01, 0.05, 0.2],
            "angle_subspace_deg": [40.0, 10.0, 30.0],
        },
        index=pd.Index([1, 2, 3], name="n_components"),
    )
    predictive = Bunch(rmsecv=np.array([1.0, 0.8, 0.85]), se_rmsecv=np.array([0.1, 0.1, 0.1]), vdv_recommended=2)
    settings = _CompareSettings(n_permutations=99, alpha=0.05, angle_threshold=30.0)
    recommendations = _recommendations(table, predictive, settings)
    assert recommendations.index.name == "criterion"
    assert list(recommendations.columns) == ["n_components", "rule", "question"]
    expected = {
        "q2_max": (2, "Largest cumulative Q2.", "Does the model predict Y?"),
        "q2_1se": (
            2,
            "Fewest components whose RMSECV is within one standard error of the minimum.",
            "Does the model predict Y?",
        ),
        "van_der_voet": (
            2,
            "Fewest components whose predictions are not worse than the minimum-PRESS model (p > 0.05).",
            "Is the gain over fewer components real?",
        ),
        "score_correlation": (
            1,
            "Leading components whose held-out t-u correlation beats the Y-permutation null (p < 0.05).",
            "Does each component's inner relation hold on new rows?",
        ),
        "covariance_permutation": (
            1,
            "Leading components whose covariance beats 99 row permutations (p < 0.05).",
            "Is each component's covariance larger than chance?",
        ),
        "subspace_stability": (
            2,
            (
                "Leading components whose jackknife-scaled subspace angle is below 30.0 degrees; a pair of "
                "components that swap inside a stable span counts as two."
            ),
            "Does the model's latent space survive a change of rows?",
        ),
    }
    assert list(recommendations.index) == list(expected)
    for name, row in expected.items():
        assert tuple(recommendations.loc[name]) == row


def test_swapped_pairs_at_the_threshold() -> None:
    """An angle of exactly the threshold is not below it: unstable, and a stable next one makes a pair."""
    np.testing.assert_array_equal(_swapped_pairs(np.array([30.0, 29.0, 30.0]), 30.0), [True, False, False])
    np.testing.assert_array_equal(_swapped_pairs(np.array([40.0, 30.0]), 30.0), [False, False])


def test_every_fit_gets_the_settings_and_the_component_count() -> None:
    """Each fold model and the model fitted to all rows are this class, with the settings passed on.

    In global scope the fold models are fitted unscaled to rows that are not centred in the
    fold, so they are told not to warn about it.
    """
    fits = []

    class Recorded(PLS):
        def fit(self, X: pd.DataFrame, Y: pd.DataFrame, sample_weight: np.ndarray | None = None) -> Recorded:
            fits.append(self.get_params())
            return super().fit(X, Y, sample_weight=sample_weight)

    X, Y = _two_component_data(n=30)
    settings = {"max_iter": 7, "tol": 1e-5}
    Recorded.compare_cv_criteria(X, Y, max_components=2, cv=KFold(3), n_permutations=9, n_cv_permutations=0, **settings)
    assert [(p["n_components"], p["max_iter"], p["tol"], p["scale"]) for p in fits] == [(2, 7, 1e-5, True)] * 4
    for scope, fold_scale in (("global", False), ("local", True)):
        fits.clear()
        Recorded.pseudo_validation_set(X, Y, n_components=2, cv=KFold(3), scope=scope, **settings)
        assert [(p["n_components"], p["max_iter"], p["tol"]) for p in fits] == [(2, 7, 1e-5)] * 4
        assert [(p["scale"], p["warn_on_uncentred"]) for p in fits] == [(True, True)] + [(fold_scale, fold_scale)] * 3


def test_pseudo_validation_component_count_runs_from_one_to_the_cap() -> None:
    """One component, and as many as the smallest training fold minus one, are accepted; one more is refused."""
    X, Y = _two_component_data(n=12, k=20)
    cv = KFold(3)  # training folds of 8 rows: 7 components
    assert pseudo_validation_set(X, Y, n_components=1, cv=cv, random_state=0).scores.shape == (12, 1)
    assert pseudo_validation_set(X, Y, n_components=7, cv=cv, random_state=0).scores.shape == (12, 7)
    for count in (0, 8):
        with pytest.raises(ValueError, match=rf"^n_components must lie in \[1, 7\] for these folds; got {count}\.$"):
            pseudo_validation_set(X, Y, n_components=count, cv=cv, random_state=0)
    with pytest.raises(ValueError, match=r"^scope must be 'global' or 'local'; got 'elsewhere'\.$"):
        pseudo_validation_set(X, Y, n_components=1, scope="elsewhere")  # type: ignore[arg-type]


def test_pseudo_validation_set_by_hand_one_row_at_a_time_with_a_row_at_the_centre() -> None:
    """Leave-one-out, global scope: one row's mixing has short columns, and the row at the centre has none.

    Ten integer rows in five pairs, a and -a, and a row of zeros: the zeros are the mean,
    exactly, so that row is 0 once scaled. Held out alone, it mixes into a column of zeros
    and a residual of zeros, which stay zero rather than 0 / 0, and its X_pv row is the mean.
    """
    rng = np.random.default_rng(5)
    half = rng.integers(-9, 10, size=(5, 3)).astype(float)
    X = pd.DataFrame(np.vstack([half, -half, np.zeros((1, 3))]), columns=["a", "b", "c"])
    Y = pd.DataFrame({"y": X.to_numpy() @ [1.0, -2.0, 0.5] + np.r_[rng.standard_normal(10), 0.0]})
    cv = LeaveOneOut()
    pv = pseudo_validation_set(X, Y, n_components=2, cv=cv, scope="global", random_state=0)
    hand = _pseudo_validation_by_hand(X, Y, list(cv.split(X)), 2, "global")
    _assert_near(pv.X_pv, hand["X_pv"], rtol=1e-6)
    np.testing.assert_array_equal(pv.X_pv.iloc[10], [0.0, 0.0, 0.0])
    assert pv.local_spe.iloc[10] == 0.0


def test_a_splitter_that_reads_y_is_given_y() -> None:
    """A stratified splitter splits by the response, in both functions."""
    X, _ = _two_component_data(n=30)
    Y = pd.DataFrame({"y": np.tile([0.0, 1.0, 1.0], 10)})
    cv = StratifiedKFold(3)
    expected = list(cv.split(X, Y))
    result = compare_cv_criteria(X, Y, max_components=1, cv=cv, n_permutations=9, n_cv_permutations=0)
    pv = pseudo_validation_set(X, Y, n_components=1, cv=cv, random_state=0)
    for splits in (result.cv_splits, pv.cv_splits):
        for (train, test), (train_by_hand, test_by_hand) in zip(splits, expected, strict=True):
            np.testing.assert_array_equal(train, train_by_hand)
            np.testing.assert_array_equal(test, test_by_hand)


@pytest.mark.parametrize("m", [1, 2])
def test_covariance_permutation_pvalues_by_hand_on_noise_with_gaps(m: int) -> None:
    """Y unrelated to X, so every component's p-value is moderate; a missing cell adds nothing to X'Y."""
    X, _ = _with_gaps(*_two_component_data(n=30), fraction=0.1)
    Y = pd.DataFrame(np.random.default_rng(8).standard_normal((30, m))).mask(
        np.random.default_rng(9).random((30, m)) < 0.1
    )
    with warnings.catch_warnings():  # NIPALS may not settle on noise; both sides read the same model
        warnings.simplefilter("ignore", SpecificationWarning)
        model = PLS(n_components=2).fit(X, Y)
    (x_mean, x_sd), (y_mean, y_sd) = _observed_mean_sd(X.to_numpy()), _observed_mean_sd(Y.to_numpy())
    x, y = (X.to_numpy() - x_mean) / x_sd, (Y.to_numpy() - y_mean) / y_sd
    p_values = _covariance_permutation_pvalues(x, y, model, 300, np.random.default_rng(1))
    expected = _covariance_pvalues_by_hand(x, y, model, 300, np.random.default_rng(1))
    np.testing.assert_array_equal(p_values, expected)
    assert (expected > 0.05).all()


def test_covariance_permutation_counts_a_draw_that_ties_the_statistic() -> None:
    """With four rows, a draw often leaves Y in place: its statistic is Y's own, and counts against it."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 2))
    x -= x.mean(axis=0)
    y = x @ np.array([[1.0], [0.5]]) + 0.1 * rng.standard_normal((4, 1))
    y -= y.mean(axis=0)
    model = PLS(n_components=1, scale=False, warn_on_uncentred=False).fit(x, y)
    p_value = _covariance_permutation_pvalues(x, y, model, 200, np.random.default_rng(1))[0]
    orders = np.random.default_rng(1).permuted(np.tile(np.arange(4), (200, 1)), axis=1)  # one batch
    statistic = _largest_singular_value(x.T @ y)
    null = np.array([_largest_singular_value(x.T @ y[order]) for order in orders])
    ties = int(np.sum(null == statistic))
    assert ties > 0
    assert p_value == (np.sum(null >= statistic) + 1) / 201
    assert p_value > (np.sum(null > statistic) + 1) / 201


def test_score_correlation_of_a_response_without_spread_is_undefined() -> None:
    """A constant response has u = 0 everywhere: its held-out correlation is 0 / 0, NaN, without a warning."""
    X, _ = _two_component_data(n=12)
    every_row = [(np.arange(12), np.arange(12))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        observed, null = _score_correlation_null(
            X.to_numpy(), np.full((12, 1), 3.0), every_row, (2, 2), np.random.default_rng(0)
        )
    assert np.isnan(observed).all()
    assert null.shape == (2, 2)
    assert np.isnan(null).all()


def test_the_one_se_rule_is_handed_rmsecv_and_its_standard_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """RMSECV over the observed Y cells, and the sample standard error of the folds' RMSE, by hand."""
    calls = []

    def stand_in(rule: str, **kwargs: object) -> int:
        calls.append((rule, kwargs))
        return 1

    monkeypatch.setattr(cv_criteria, "_select_n_components", stand_in)
    X, Y = _with_gaps(*_two_component_data(n=40, k=6), fraction=0.1)
    cv = KFold(5, shuffle=True, random_state=0)
    compare_cv_criteria(X, Y, max_components=3, cv=cv, n_permutations=9, n_cv_permutations=0)
    held = _held_out_by_hand(_folds_by_hand(X, Y, list(cv.split(X)), 3), Y.to_numpy(), 3)
    [(rule, kwargs)] = calls
    assert rule == "1se"
    _assert_near(kwargs["mean_error"], np.sqrt(held["press_y"].sum(axis=1) / Y.notna().to_numpy().sum()), rtol=1e-6)
    _assert_near(kwargs["se_error"], np.std(held["per_fold_rmse"], axis=1, ddof=1) / np.sqrt(5), rtol=1e-6)
