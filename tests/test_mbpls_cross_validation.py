"""Cross-validating a multi-block PLS by holding out rows.

The scheme is the same one :meth:`PLS.select_n_components` uses, and it is sound
here for the same reason: a held-out row's super score comes from its X-blocks
alone, so its Y never contributes to its own prediction. Having several X-blocks
does not touch that argument.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold

from process_improve.multivariate._common import Q2_MIN_INCREMENT, _recommend_n_components_q2
from process_improve.multivariate.methods import MBPLS, SpecificationWarning


def _two_block_data(seed: int = 0, n: int = 60, rank: int = 2) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Two X-blocks and a Y block driven by the same `rank` latent variables."""
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((n, rank)) * np.array([5.0, 3.0, 1.5])[:rank]
    blocks = {
        "wide": pd.DataFrame(scores @ rng.standard_normal((rank, 8)) + rng.standard_normal((n, 8)) * 0.4),
        "narrow": pd.DataFrame(scores @ rng.standard_normal((rank, 3)) + rng.standard_normal((n, 3)) * 0.4),
    }
    y = pd.DataFrame(scores @ rng.standard_normal((rank, 2)) + rng.standard_normal((n, 2)) * 0.4)
    return blocks, y


def test_returns_the_documented_bunch() -> None:
    """Every field the docstring promises, with the shapes it promises."""
    blocks, y = _two_block_data()
    out = MBPLS.select_n_components(blocks, y, max_components=4, cv=5, n_repeats=2, random_state=0)

    assert list(out.rmsecv.index) == [1, 2, 3, 4]
    assert out.rmsecv.index.name == "n_components"
    assert out.se_rmsecv.shape == (4,)
    assert out.per_fold_rmsecv.shape == (4, 10)  # five folds, twice over
    assert list(out.r2y_validated.columns) == [*y.columns, "total", "scaled_total"]
    assert out.cv_predictions.shape == y.shape
    assert out.selection_rule == "1se"
    assert 1 <= out.n_components <= 4


def test_scaled_total_weights_every_target_equally() -> None:
    """``scaled_total`` is the equal-weight pooling; ``total`` is the raw-scale one.

    Stretching one target by a constant leaves every per-target value alone, and
    leaves ``scaled_total`` alone with them, while ``total`` follows the stretched
    target. That is the whole difference between the two columns.
    """
    blocks, y = _two_block_data()
    stretched = y.copy()
    stretched[1] = stretched[1] * 500.0

    plain = MBPLS.select_n_components(blocks, y, max_components=2, cv=5, n_repeats=1, random_state=0)
    wide = MBPLS.select_n_components(blocks, stretched, max_components=2, cv=5, n_repeats=1, random_state=0)

    # NIPALS is iterative, so rescaling a target moves the fit in the last few
    # digits; 1e-6 is far below the effect being tested.
    np.testing.assert_allclose(plain.r2y_validated[[0, 1]].to_numpy(), wide.r2y_validated[[0, 1]].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(
        wide.r2y_validated["scaled_total"].to_numpy(),
        wide.r2y_validated[[0, 1]].mean(axis=1).to_numpy(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        plain.r2y_validated["scaled_total"].to_numpy(), wide.r2y_validated["scaled_total"].to_numpy(), rtol=1e-6
    )
    # ``total`` is not invariant: it now tracks the stretched target alone, and
    # has moved away from the value it took on the unstretched Y.
    np.testing.assert_allclose(wide.r2y_validated["total"].to_numpy(), wide.r2y_validated[1].to_numpy(), atol=1e-4)
    assert abs(wide.r2y_validated["total"].iloc[0] - plain.r2y_validated["total"].iloc[0]) > 0.05


def test_holding_out_rows_costs_something() -> None:
    """The defining property: a held-out row is predicted worse than a fitted one.

    If the held-out Y were reaching its own prediction, the cross-validated value
    could match or beat the fit. It must not.

    The baseline is the fitted model's own predictions on the original Y scale,
    which is what the validated value is built from. It is deliberately not
    ``r2_y_cumulative_``: that is computed on the scaled Y, where every target
    carries equal weight, so the two are on different footings and differ by a
    wide margin on data whose targets have unequal spread.
    """
    blocks, y = _two_block_data()
    out = MBPLS.select_n_components(blocks, y, max_components=3, cv=5, n_repeats=2, random_state=0)
    values = y.to_numpy(dtype=float)
    tss = float(((values - values.mean(axis=0)) ** 2).sum())

    for a in (1, 2, 3):
        fitted = MBPLS(n_components=a).fit(blocks, y)
        predicted = np.asarray(fitted.diagnose(blocks).predictions, dtype=float)
        fitted_r2 = 1.0 - float(((values - predicted) ** 2).sum()) / tss
        validated = float(out.r2y_validated["total"].loc[a])
        assert validated < fitted_r2, f"{a} components: validated {validated} beat the fit {fitted_r2}"
        assert validated > 0, "the blocks do carry signal, so the model must beat the mean"


def test_finds_the_rank_the_blocks_carry() -> None:
    """Two latent variables generated the data, and the curve should say so."""
    blocks, y = _two_block_data(rank=2)
    out = MBPLS.select_n_components(blocks, y, max_components=5, cv=5, n_repeats=3, random_state=0)
    gains = np.diff([0.0, *out.r2y_validated["total"].tolist()])

    assert out.n_components <= 3
    assert gains[0] > 0.3, "the first component carries most of it"
    assert gains[2] < 0.02, "a third component adds nothing the blocks support"


def test_repeatable_and_seed_sensitive() -> None:
    """The same seed gives the same curve; a different one moves it."""
    blocks, y = _two_block_data()
    kwargs = {"max_components": 3, "cv": 5, "n_repeats": 2}
    first = MBPLS.select_n_components(blocks, y, random_state=0, **kwargs)
    again = MBPLS.select_n_components(blocks, y, random_state=0, **kwargs)
    other = MBPLS.select_n_components(blocks, y, random_state=7, **kwargs)

    assert np.allclose(first.rmsecv, again.rmsecv)
    assert not np.allclose(first.per_fold_rmsecv.to_numpy(), other.per_fold_rmsecv.to_numpy())


def test_accepts_a_splitter_and_ignores_n_repeats() -> None:
    """A pre-built splitter is used as given, as it is for PLS."""
    blocks, y = _two_block_data()
    out = MBPLS.select_n_components(blocks, y, max_components=2, cv=KFold(4), n_repeats=9, random_state=0)

    assert out.per_fold_rmsecv.shape == (2, 4), "four folds, not four times nine"


def test_component_count_capped_by_the_smallest_fold() -> None:
    """More components than a training fold can support are not evaluated."""
    blocks, y = _two_block_data(n=12)
    out = MBPLS.select_n_components(blocks, y, max_components=50, cv=4, n_repeats=1, random_state=0)

    smallest_train = 12 - 12 // 4
    assert len(out.rmsecv) <= smallest_train - 1


@pytest.mark.parametrize(
    ("blocks", "error", "message"),
    [
        ([], TypeError, r"dict of DataFrames.*got list"),
        ({}, ValueError, r"^X must hold at least one block\.$"),
        ({"a": np.zeros((10, 3))}, TypeError, r"'a' must be a pandas DataFrame; got ndarray"),
    ],
)
def test_rejects_malformed_blocks(blocks: object, error: type[Exception], message: str) -> None:
    """A wrong type raises TypeError and a wrong value ValueError, each naming what arrived."""
    _, y = _two_block_data(n=10)
    with pytest.raises(error, match=message):
        MBPLS.select_n_components(blocks, y, max_components=1, cv=2)


def test_rejects_a_component_count_below_one() -> None:
    """Asking for no components is the caller's mistake, and the message names it.

    The cap is applied afterwards with ``max(1, ...)``, so a fold too small to support
    more components still supports one. Zero can only come from the argument.
    """
    blocks, y = _two_block_data(n=20)
    with pytest.raises(ValueError, match=r"max_components must be at least 1; got 0"):
        MBPLS.select_n_components(blocks, y, max_components=0, cv=2)


def test_rejects_a_block_of_the_wrong_length() -> None:
    """A block that does not line up with y is named, with both row counts."""
    blocks, y = _two_block_data(n=20)
    blocks["narrow"] = blocks["narrow"].iloc[:15]
    with pytest.raises(ValueError, match=r"'narrow' has 15 rows; y has 20"):
        MBPLS.select_n_components(blocks, y, max_components=1, cv=2)


class _Folds:
    """A splitter that holds out the rows it is given: a row may be held out twice, or never."""

    def __init__(self, tests: list[list[int]]) -> None:
        self.tests = [np.asarray(test) for test in tests]

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:  # noqa: ARG002
        return len(self.tests)

    def split(self, X: pd.DataFrame, y: object = None, groups: object = None):  # noqa: ARG002
        for test in self.tests:
            yield np.setdiff1d(np.arange(len(X)), test), test


def _cross_validated_by_hand(
    blocks: dict[str, pd.DataFrame], y: pd.DataFrame, folds: list, n_components: int, **kwargs: object
) -> dict:
    """Fit one model per fold and component count on the training rows, and score the held-out ones.

    PRESS and each fold's RMSE are over the observed cells of Y; each row's reference sum of
    squares about the column mean counts as often as the row was held out; a held-out
    prediction is the mean over the folds that held the row out.
    """
    values = y.to_numpy(dtype=float)
    press = np.zeros((n_components, y.shape[1]))
    per_fold = np.zeros((n_components, len(folds)))
    tested = np.zeros(len(y))
    sums = np.zeros((n_components, *values.shape))
    for fold, (train, test) in enumerate(folds):
        tested[test] += 1
        for a in range(n_components):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SpecificationWarning)
                model = MBPLS(n_components=a + 1, **kwargs).fit(
                    {name: frame.iloc[train] for name, frame in blocks.items()}, y.iloc[train]
                )
            predicted = model.predict({name: frame.iloc[test] for name, frame in blocks.items()}).to_numpy()
            residual = values[test] - predicted
            press[a] += np.nansum(residual**2, axis=0)
            per_fold[a, fold] = np.sqrt(np.nanmean(residual**2))
            sums[a][test] += predicted
    tss = np.sum(tested[:, np.newaxis] * (values - np.nanmean(values, axis=0)) ** 2, axis=0, where=~np.isnan(values))
    with np.errstate(invalid="ignore"):
        held_out = sums / tested[:, np.newaxis]
    return {
        "press": press,
        "per_fold": per_fold,
        "rmsecv": np.sqrt(press.sum(axis=1) / (tested.sum() * y.shape[1])),
        "se": np.std(per_fold, axis=1, ddof=1) / np.sqrt(len(folds)) if len(folds) > 1 else np.nan,
        "r2": 1 - press / tss,
        "total": 1 - press.sum(axis=1) / tss.sum(),
        "held_out": held_out,
    }


def _small_two_blocks(n: int = 12) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Two blocks and two small targets, one cell missing, in rows whose first two share a label."""
    rng = np.random.default_rng(2)
    latent = rng.standard_normal((n, 2))
    index = ["s0", "s0", *(f"s{i}" for i in range(1, n - 1))]
    blocks = {
        "wide": pd.DataFrame(latent @ rng.standard_normal((2, 4)) + 0.3 * rng.standard_normal((n, 4)), index=index),
        "narrow": pd.DataFrame(latent @ rng.standard_normal((2, 2)) + 0.3 * rng.standard_normal((n, 2)), index=index),
    }
    y = pd.DataFrame(
        0.05 * (latent @ rng.standard_normal((2, 2)) + 0.5 * rng.standard_normal((n, 2))),
        index=index,
        columns=["u", "v"],
    )
    y.iloc[5, 1] = np.nan
    return blocks, y


@pytest.mark.parametrize("selection_rule", ["min", "q2_increment"])
def test_the_curve_matches_one_model_per_fold_and_component_count_by_hand(selection_rule: str) -> None:
    """Every field of the result against the cross-validation written out, rows held out twice or never.

    Row 0 is held out twice, and rows 10 and 11 never: each row's reference sum of squares
    counts as often as the row was held out, and the rows never held out have no held-out
    prediction. The targets are small, so every reference sum of squares is below 1.

    Two rows share the label s0. A fold's Y was taken by label, so it held both s0 rows for
    each one, no longer lined up with the fold's blocks, and the call failed with a
    ValueError; rows are now taken, and their held-out predictions gathered, by position
    (#1028).
    """
    blocks, y = _small_two_blocks()
    tests = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 9]]
    folds = list(_Folds(tests).split(y))
    out = MBPLS.select_n_components(blocks, y, cv=_Folds(tests), selection_rule=selection_rule, max_iter=3)
    by_hand = _cross_validated_by_hand(blocks, y, folds, 6, max_iter=3)
    index = pd.Index(range(1, 7), name="n_components")
    assert np.all(np.nansum((y - y.mean()) ** 2) < 1)

    pd.testing.assert_series_equal(out.rmsecv, pd.Series(by_hand["rmsecv"], index=index, name="RMSECV"), rtol=1e-12)
    pd.testing.assert_series_equal(
        out.se_rmsecv, pd.Series(by_hand["se"], index=index, name="SE of RMSECV"), rtol=1e-12
    )
    expected_folds = pd.DataFrame(by_hand["per_fold"], index=index, columns=["fold_1", "fold_2", "fold_3", "fold_4"])
    pd.testing.assert_frame_equal(out.per_fold_rmsecv, expected_folds, rtol=1e-12)
    pd.testing.assert_series_equal(
        out.press, pd.Series(by_hand["press"].sum(axis=1), index=index, name="PRESS"), rtol=1e-12
    )
    r2 = np.column_stack([by_hand["r2"], by_hand["total"], by_hand["r2"].mean(axis=1)])
    expected_r2 = pd.DataFrame(r2, index=index, columns=["u", "v", "total", "scaled_total"])
    pd.testing.assert_frame_equal(out.r2y_validated, expected_r2, rtol=1e-12)
    if selection_rule == "min":
        recommended = int(np.argmin(by_hand["rmsecv"])) + 1
    else:
        recommended = _recommend_n_components_q2(by_hand["total"], min_increment=Q2_MIN_INCREMENT)
    assert (out.n_components, out.selection_rule, out.n_splits) == (recommended, selection_rule, 4)
    held_out = pd.DataFrame(by_hand["held_out"][recommended - 1], index=y.index, columns=y.columns)
    pd.testing.assert_frame_equal(out.cv_predictions, held_out, rtol=1e-12)
    assert out.cv_predictions.iloc[10:].isna().all().all()


def test_an_integer_cv_is_a_shuffled_repeated_k_fold_and_any_splitter_is_used_as_given() -> None:
    """``cv=k`` splits k ways, ten times over by default, as ``RepeatedKFold`` (one repeat: ``KFold``) does.

    The defaults are five folds, ten times over. A splitter is used as given, and its count of
    splits is asked of it: ``LeaveOneOut`` needs the rows to give one.
    """
    blocks, y = _small_two_blocks()

    def curve(**kwargs: object) -> np.ndarray:
        return MBPLS.select_n_components(blocks, y, max_components=2, **kwargs).per_fold_rmsecv.to_numpy()

    np.testing.assert_array_equal(
        curve(cv=3, n_repeats=2, random_state=0),
        curve(cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)),
    )
    np.testing.assert_array_equal(
        curve(cv=3, n_repeats=1, random_state=4), curve(cv=KFold(3, shuffle=True, random_state=4))
    )
    np.testing.assert_array_equal(
        curve(cv=4, random_state=1), curve(cv=RepeatedKFold(n_splits=4, n_repeats=10, random_state=1))
    )
    default = MBPLS.select_n_components(blocks, y, max_components=1, random_state=1)
    assert default.n_splits == 5
    np.testing.assert_array_equal(
        default.per_fold_rmsecv.to_numpy(), curve(cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=1))[:1]
    )
    assert MBPLS.select_n_components(blocks, y, max_components=1, cv=3, n_repeats=1).n_splits == 3
    leave_one_out = MBPLS.select_n_components(blocks, y, max_components=1, cv=LeaveOneOut())
    assert leave_one_out.n_splits == 12
    assert leave_one_out.per_fold_rmsecv.shape == (1, 12)


def test_one_held_out_cell_gives_its_own_error_and_no_spread() -> None:
    """One fold holding out one row of one target: RMSECV is that row's error, and one fold has no SE."""
    blocks, y = _small_two_blocks()
    out = MBPLS.select_n_components(blocks, y[["u"]], max_components=1, cv=_Folds([[4]]))
    by_hand = _cross_validated_by_hand(blocks, y[["u"]], list(_Folds([[4]]).split(y)), 1)
    np.testing.assert_allclose(out.rmsecv, np.sqrt(by_hand["press"][0]), rtol=1e-12)
    assert np.isnan(out.se_rmsecv.iloc[0])
    assert out.n_components == 1


def test_the_component_count_is_capped_by_the_smallest_fold_and_the_width() -> None:
    """Six rows in three folds train on four, which support three components; a block one wide supports one."""
    blocks, y = _small_two_blocks(n=6)
    assert len(MBPLS.select_n_components(blocks, y, cv=3, n_repeats=1).rmsecv) == 3
    assert len(MBPLS.select_n_components(blocks, y, max_components=1, cv=3, n_repeats=1).rmsecv) == 1
    narrow = {"narrow": blocks["narrow"][[0]]}
    assert len(MBPLS.select_n_components(narrow, y, cv=3, n_repeats=1).rmsecv) == 1


def test_targets_without_spread_about_the_held_out_rows_have_no_r2_and_no_warning() -> None:
    """A target with no spread in the rows held out has no validated R2, and the total neither when all have none.

    ``v`` is constant. ``u`` is 0 in the ten rows held out, and 1 and -1 in the two never held
    out, so its mean is 0 and its reference sum of squares 0, although the fits see it vary.
    """
    blocks, y = _small_two_blocks()
    flat = y.fillna(0.0).assign(v=1.0)
    gaps = pd.DataFrame({"u": [0.0] * 10 + [1.0, -1.0]}, index=y.index)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = MBPLS.select_n_components(blocks, flat, max_components=1, cv=3, n_repeats=1, random_state=0)
        held_out = MBPLS.select_n_components(blocks, gaps, max_components=1, cv=_Folds([range(5), range(5, 10)]))
    assert np.isnan(out.r2y_validated.loc[1, "v"])
    assert np.isfinite(out.r2y_validated.loc[1, ["u", "total"]]).all()
    assert held_out.press.iloc[0] > 0
    assert held_out.r2y_validated.loc[1].isna().all()


def test_an_object_y_cross_validates_as_its_float_copy() -> None:
    """A Y of object dtype, as a frame read with mixed columns has, gives the curve of its float copy."""
    blocks, y = _small_two_blocks()
    y = y.fillna(0.0)
    kwargs = {"max_components": 2, "cv": 3, "n_repeats": 1, "random_state": 0}
    as_float = MBPLS.select_n_components(blocks, y, **kwargs)
    as_object = MBPLS.select_n_components(blocks, y.astype(object), **kwargs)
    pd.testing.assert_series_equal(as_object.rmsecv, as_float.rmsecv)
    pd.testing.assert_frame_equal(as_object.r2y_validated, as_float.r2y_validated)


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered:RuntimeWarning", "ignore:Mean of empty slice:RuntimeWarning"
)
def test_only_the_fits_specification_warnings_are_silenced() -> None:
    """The fold fits' convergence warnings are dropped; a numerical warning from the data still reaches the caller."""
    blocks, y = _small_two_blocks()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MBPLS.select_n_components(blocks, y, max_components=2, cv=3, n_repeats=1, max_iter=1)
    blocks["wide"].iloc[4, 0] = 1e155  # its square overflows
    with pytest.warns(RuntimeWarning, match="overflow"):
        MBPLS.select_n_components(blocks, y, max_components=1, cv=3, n_repeats=1)
