"""A row or a column with nothing observed, treated alike across the multivariate family (#1098).

PCA, PLS and TPLS fit a row with every value missing: it scores 0 and has no weight. MBPCA and
MBPLS refused it. A column with every value missing was refused by MBPCA and MBPLS, fitted
without a word by PCA and PLS, and named in a warning by TPLS (#982). Now every estimator fits
the row as PCA does, and names the column in TPLS's words and leaves it out: the fit is the fit
without the column, and a value a new row has in it changes nothing.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import MBPCA, MBPLS, PCA, PLS, MCUVScaler


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Twelve rows of five X columns and two Y columns from two latent variables, one X cell missing.

    The missing cell sends every fit down its masked path, with or without an empty column, so a
    fit that leaves a column out can be compared with the fit of the data without it.
    """
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((12, 2))
    index = [f"r{i}" for i in range(12)]
    x = pd.DataFrame(
        latent @ rng.standard_normal((2, 5)) + 0.2 * rng.standard_normal((12, 5)), index=index, columns=list("abcde")
    )
    y = pd.DataFrame(
        latent @ rng.standard_normal((2, 2)) + 0.2 * rng.standard_normal((12, 2)), index=index, columns=["y1", "y2"]
    )
    x.iloc[1, 0] = np.nan
    return x, y


def _blocks(x: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the five X columns into two blocks."""
    return {"p": x[["a", "b"]], "q": x[["c", "d", "e"]]}


def _fit_naming(fit: Callable[[], object], label: str, column: str) -> object:
    """Fit, and check the one warning the fit gives is the family's, naming the column and its block."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit()
    message = (
        f"Block {label} has columns with no observed values, which carry no information and are excluded from the "
        f"model: '{column}'. Drop them, or check the data for a sensor that was never logged."
    )
    assert [(w.category, str(w.message)) for w in caught] == [(UserWarning, message)]
    return model


def _close(actual: pd.DataFrame | pd.Series, expected: pd.DataFrame | pd.Series) -> None:
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("value", [np.nan, 5.0])
def test_pca_names_a_column_with_nothing_observed_and_leaves_it_out(value: float) -> None:
    """PCA used to fit it without a word, and any NaN a new row had in it made every score of the row NaN."""
    x, _ = _data()
    scaled = MCUVScaler().fit_transform(x.assign(c=np.nan))
    model = _fit_naming(lambda: PCA(n_components=2, algorithm="nipals").fit(scaled), "X", "c")
    without = PCA(n_components=2, algorithm="nipals").fit(scaled.drop(columns="c"))

    assert model.n_features_in_ == 5
    np.testing.assert_array_equal(model.feature_names_in_, list("abcde"))
    np.testing.assert_array_equal(model.loadings_.loc["c"], [0.0, 0.0])
    _close(model.loadings_.drop(index="c"), without.loadings_)
    _close(model.scores_, without.scores_)
    _close(model.spe_, without.spe_)

    new = scaled.iloc[2:6].assign(c=value)
    result, expected = model.diagnose(new), without.diagnose(new.drop(columns="c"))
    _close(result.scores, expected.scores)
    _close(result.spe, expected.spe)
    _close(result.hotellings_t2, expected.hotellings_t2)
    _close(model.transform(new), expected.scores)
    _close(model.project(new).spe, without.project(new.drop(columns="c")).spe)

    del model._unobserved_columns  # a model pickled before #1098 left no column out
    _close(model.transform(new.assign(c=0.0)), expected.scores)


@pytest.mark.parametrize("md_method", ["nipals", "tsr"])
@pytest.mark.parametrize("value", [np.nan, 5.0])
def test_pls_names_a_column_of_x_with_nothing_observed_and_leaves_it_out(md_method: str, value: float) -> None:
    """On every md_method: it was fitted without a word, and a NaN in a new row made its predictions NaN."""
    x, y = _data()
    settings = {"md_method": md_method}
    model = _fit_naming(
        lambda: PLS(n_components=2, missing_data_settings=settings).fit(x.assign(c=np.nan), y), "X", "c"
    )
    without = PLS(n_components=2, missing_data_settings=settings).fit(x.drop(columns="c"), y)

    assert model.n_features_in_ == 5
    np.testing.assert_array_equal(model.feature_names_in_, list("abcde"))
    for weights in (model.x_weights_, model.x_loadings_, model.direct_weights_, model.beta_coefficients_):
        np.testing.assert_array_equal(weights.loc["c"], [0.0, 0.0])
    _close(model.x_weights_.drop(index="c"), without.x_weights_)
    _close(model.scores_, without.scores_)
    _close(model.predictions_, without.predictions_)

    new = x.iloc[2:6].assign(c=value)
    result, expected = model.diagnose(new), without.diagnose(new.drop(columns="c"))
    for key in ("scores", "spe", "hotellings_t2", "y_hat"):
        _close(result[key], expected[key])
    _close(model.predict(new), expected.y_hat)
    _close(model.transform(new), expected.scores)

    del model._unobserved_columns  # a model pickled before #1098 left no column out
    _close(model.transform(new.assign(c=model._x_scaler.center_["c"])), expected.scores)


def test_pls_names_a_response_with_nothing_observed_and_does_not_predict_it() -> None:
    """It was fitted without a word, predicted as 0, and its RMSE warned of a mean of an empty slice."""
    x, y = _data()
    model = _fit_naming(lambda: PLS(n_components=2).fit(x, y.assign(y2=np.nan)), "Y", "y2")
    without = PLS(n_components=2).fit(x, y[["y1"]])

    assert model.n_targets_ == 2
    np.testing.assert_array_equal(model.y_loadings_.loc["y2"], [0.0, 0.0])
    _close(model.scores_, without.scores_)
    _close(model.predictions_[["y1"]], without.predictions_)
    assert model.predictions_["y2"].isna().all()
    assert model.rmse_.loc["y2"].isna().all()

    new = x.iloc[2:6]
    for y_hat in (model.diagnose(new).y_hat, model.predict(new), model.project(new).y_hat):
        _close(y_hat[["y1"]], without.predict(new))
        assert y_hat["y2"].isna().all()

    del model._unobserved_targets  # a model pickled before #1098 left no response out
    assert model.predict(new)["y2"].notna().all()


@pytest.mark.parametrize("value", [np.nan, 5.0])
def test_mbpca_names_a_column_with_nothing_observed_and_leaves_it_out(value: float) -> None:
    """MBPCA used to refuse the fit."""
    x, _ = _data()
    blocks = _blocks(x.assign(c=np.nan))
    model = _fit_naming(lambda: MBPCA(n_components=2).fit(blocks), 'X["q"]', "c")
    without = MBPCA(n_components=2).fit({"p": blocks["p"], "q": blocks["q"].drop(columns="c")})

    assert model.n_features_in_ == 5
    np.testing.assert_array_equal(model.feature_names_in_, list("abcde"))
    assert model.block_informative_widths_ == {"p": 2, "q": 2}
    np.testing.assert_array_equal(model.block_loadings_["q"].loc["c"], [0.0, 0.0])
    _close(model.block_loadings_["q"].drop(index="c"), without.block_loadings_["q"])
    _close(model.super_loadings_, without.super_loadings_)
    _close(model.super_scores_, without.super_scores_)

    new = {name: frame.iloc[2:6] for name, frame in _blocks(x.assign(c=value)).items()}
    result = model.diagnose(new)
    expected = without.diagnose({"p": new["p"], "q": new["q"].drop(columns="c")})
    _close(result.super_scores, expected.super_scores)
    _close(result.hotellings_t2, expected.hotellings_t2)
    for name in ("p", "q"):
        _close(result.block_spe[name], expected.block_spe[name])
    _close(model.transform(new), expected.super_scores)

    # A row with nothing in the block but the left-out column has nothing observed in the block,
    # so no block score or SPE there, not a score and an SPE of 0 (#1076). It is scored from p.
    only_c = {"p": new["p"], "q": new["q"].assign(d=np.nan, e=np.nan, c=1.0)}
    result = model.diagnose(only_c)
    assert result.block_spe["q"].isna().all()
    assert result.block_scores["q"].isna().all(axis=None)
    assert result.super_scores.notna().all(axis=None)
    _close(result.super_scores, without.transform({"p": new["p"], "q": only_c["q"].drop(columns="c")}))

    del model._unobserved_columns  # a model pickled before #1098 left no column out
    complete = {"p": new["p"], "q": new["q"].assign(c=model.preproc_["q"].center_["c"])}
    _close(model.transform(complete), expected.super_scores)


@pytest.mark.parametrize("value", [np.nan, 5.0])
def test_mbpls_names_a_column_of_x_with_nothing_observed_and_leaves_it_out(value: float) -> None:
    """MBPLS used to refuse the fit; a new row's value in the column is read as missing."""
    x, y = _data()
    blocks = _blocks(x.assign(c=np.nan))
    model = _fit_naming(lambda: MBPLS(n_components=2).fit(blocks, y), 'X["q"]', "c")
    without = MBPLS(n_components=2).fit({"p": blocks["p"], "q": blocks["q"].drop(columns="c")}, y)

    assert model.n_features_in_ == 5
    np.testing.assert_array_equal(model.feature_names_in_, list("abcde"))
    for frame in (model.block_weights_["q"], model.block_loadings_["q"]):
        np.testing.assert_array_equal(frame.loc["c"], [0.0, 0.0])
    _close(model.block_weights_["q"].drop(index="c"), without.block_weights_["q"])
    _close(model.super_scores_, without.super_scores_)
    _close(model.predictions_, without.predictions_)

    new = {name: frame.iloc[2:6] for name, frame in _blocks(x.assign(c=value)).items()}
    result = model.diagnose(new)
    expected = without.diagnose({"p": new["p"], "q": new["q"].drop(columns="c")})
    for key in ("super_scores", "predictions", "hotellings_t2"):
        _close(result[key], expected[key])
    for name in ("p", "q"):
        _close(result.block_spe[name], expected.block_spe[name])
    _close(model.predict(new), expected.predictions)

    # A row with nothing in the block but the left-out column has nothing observed in the block.
    only_c = {"p": new["p"], "q": new["q"].assign(d=np.nan, e=np.nan, c=1.0)}
    assert model.diagnose(only_c).block_spe["q"].isna().all()

    del model._unobserved_columns  # a model pickled before #1098 left no column out
    complete = {"p": new["p"], "q": new["q"].assign(c=model.preproc_["q"].center_["c"])}
    _close(model.transform(complete), expected.super_scores)


def test_mbpls_names_a_response_with_nothing_observed_and_does_not_predict_it() -> None:
    """MBPLS used to refuse the fit."""
    x, y = _data()
    blocks = _blocks(x)
    model = _fit_naming(lambda: MBPLS(n_components=2).fit(blocks, y.assign(y2=np.nan)), "Y", "y2")
    without = MBPLS(n_components=2).fit(blocks, y[["y1"]])

    np.testing.assert_array_equal(model.super_y_loadings_.loc["y2"], [0.0, 0.0])
    _close(model.super_scores_, without.super_scores_)
    _close(model.predictions_[["y1"]], without.predictions_)
    assert model.predictions_["y2"].isna().all()

    new = {name: frame.iloc[2:6] for name, frame in blocks.items()}
    for predictions in (model.diagnose(new).predictions, model.predict(new)):
        _close(predictions[["y1"]], without.predict(new))
        assert predictions["y2"].isna().all()

    del model._unobserved_targets  # a model pickled before #1098 left no response out
    assert model.predict(new)["y2"].notna().all()


def test_mbpca_fits_a_row_with_nothing_observed_as_pca_does() -> None:
    """The row scores 0 and has no weight: every other row, and every loading, is as fitted without it.

    MBPCA used to refuse it. It has no block score and no block SPE, as in MBPLS. Scored as a
    new row it has no score, SPE or T², as PCA.diagnose gives it.
    """
    x, _ = _data()
    gap = x.copy()
    gap.loc["r4"] = np.nan
    blocks = _blocks(gap)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = MBPCA(n_components=2).fit(blocks)
        pca = PCA(n_components=2, algorithm="nipals").fit(MCUVScaler().fit_transform(gap))
    without = MBPCA(n_components=2).fit({name: frame.drop(index="r4") for name, frame in blocks.items()})

    np.testing.assert_array_equal(pca.scores_.loc["r4"], [0.0, 0.0])
    np.testing.assert_array_equal(model.super_scores_.loc["r4"], [0.0, 0.0])
    np.testing.assert_array_equal(model.super_hotellings_t2_.loc["r4"], [0.0, 0.0])
    _close(model.super_scores_.drop(index="r4"), without.super_scores_)
    _close(model.super_loadings_, without.super_loadings_)
    for name in blocks:
        _close(model.block_loadings_[name], without.block_loadings_[name])
        _close(model.block_scores_[name].drop(index="r4"), without.block_scores_[name])
        assert model.block_scores_[name].loc["r4"].isna().all()
        assert model.block_spe_[name].loc["r4"].isna().all()

    result = model.diagnose({name: frame.loc[["r4"]] for name, frame in blocks.items()})
    single = pca.diagnose(MCUVScaler().fit(x).transform(gap.loc[["r4"]]))
    for values in (result.super_scores, result.hotellings_t2, *result.block_spe.values(), single.scores, single.spe):
        assert values.isna().all(axis=None)


def test_mbpls_fits_a_row_with_nothing_observed_in_x_as_pls_does() -> None:
    """The row scores 0, has no weight, and is predicted at Y's mean, as PLS fits it.

    MBPLS used to refuse it. Its Y is the mean of the other rows', so the only change to Y's
    preprocessing is its scale, which no score depends on: every other row's score, and every
    weight, is as fitted without the row. Scored as a new row it has no score, SPE, T² or
    prediction, as PLS.diagnose gives it.
    """
    x, y = _data()
    gap = x.copy()
    gap.loc["r4"] = np.nan
    blocks = _blocks(gap)
    gap_y = y.copy()
    gap_y.loc["r4"] = y.drop(index="r4").mean()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = MBPLS(n_components=2).fit(blocks, gap_y)
        pls = PLS(n_components=2).fit(gap, gap_y)
    without = MBPLS(n_components=2).fit({name: frame.drop(index="r4") for name, frame in blocks.items()}, y.drop("r4"))

    np.testing.assert_array_equal(pls.scores_.loc["r4"], [0.0, 0.0])
    np.testing.assert_array_equal(model.super_scores_.loc["r4"], [0.0, 0.0])
    np.testing.assert_array_equal(model.super_hotellings_t2_.loc["r4"], [0.0, 0.0])
    _close(model.predictions_.loc["r4"], gap_y.loc["r4"])
    _close(pls.predictions_.loc["r4"], gap_y.loc["r4"])
    _close(model.super_scores_.drop(index="r4"), without.super_scores_)
    _close(model.super_weights_, without.super_weights_)
    _close(model.predictions_.drop(index="r4"), without.predictions_)
    for name in blocks:
        _close(model.block_weights_[name], without.block_weights_[name])
        assert model.block_scores_[name].loc["r4"].isna().all()
        assert model.block_spe_[name].loc["r4"].isna().all()

    result = model.diagnose({name: frame.loc[["r4"]] for name, frame in blocks.items()})
    single = pls.diagnose(gap.loc[["r4"]])
    for values in (result.super_scores, result.predictions, result.hotellings_t2, *result.block_spe.values()):
        assert values.isna().all(axis=None)
    for values in (single.scores, single.spe, single.y_hat):
        assert values.isna().all(axis=None)


def test_mbpls_fits_a_row_with_no_observed_response_as_pls_does() -> None:
    """The row's Y-score is 0, so it has no weight in the block weights; its X is still scored.

    MBPLS used to refuse it. A one-block MBPLS fit of autoscaled data is PLS, a row without Y
    included, with its super scores divided by the square root of the block's width.
    """
    x, y = _data()
    gap_y = y.copy()
    gap_y.loc["r4"] = np.nan
    scaled = MCUVScaler().fit_transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = MBPLS(n_components=2).fit({"all": scaled}, gap_y)
        pls = PLS(n_components=2).fit(scaled, gap_y)

    np.testing.assert_array_equal(model.super_y_scores_.loc["r4"], [0.0, 0.0])
    np.testing.assert_array_equal(pls.y_scores_.loc["r4"], [0.0, 0.0])
    # Two NIPALS loops, each stopped at a relative change of epsqrt
    np.testing.assert_allclose(model.super_scores_ * np.sqrt(5), pls.scores_, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(model.predictions_, pls.predictions_, rtol=1e-7, atol=1e-8)
    assert model.predictions_.loc["r4"].notna().all()


@pytest.mark.parametrize("method", ["scp", "tsr"])
@pytest.mark.parametrize("value", [np.nan, 5.0])
@pytest.mark.parametrize("estimator", [PCA, PLS])
def test_contributions_give_a_column_left_out_nothing(estimator: type, value: float, method: str) -> None:
    """A column the fit left out contributes 0, and the rest are those of the model without it (#1114).

    The contribution functions read the column as it was given, not at 0 as ``transform`` reads it:
    a value there was a whole SPE contribution, and a NaN sent every row down the missing-data path
    and gave the column NaN. Row r1 has a missing cell in column a, which stays NaN.
    """
    x, y = _data()
    scaled = MCUVScaler().fit_transform(x.assign(c=np.nan))
    target = () if estimator is PCA else (y,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the column with nothing observed, named in the fit
        model = estimator(n_components=2).fit(scaled, *target)
    without = estimator(n_components=2).fit(scaled.drop(columns="c"), *target)

    new = scaled.assign(c=value)
    for name in ("t2_contributions", "spe_contributions", "score_contributions"):
        result = getattr(model, name)(new, method=method)
        expected = getattr(without, name)(new.drop(columns="c"), method=method)
        assert (result["c"] == 0.0).all()
        _close(result.drop(columns="c"), expected)
    group = model.group_contributions(new.drop(index="r1"), group=["r2", "r3"], reference=["r4"])
    assert group["c"] == 0.0
    _close(group.drop("c"), without.group_contributions(new.drop(index="r1", columns="c"), ["r2", "r3"], ["r4"]))

    del model._unobserved_columns  # a model pickled before #1098 left no column out: it reads the column as given
    assert (model.spe_contributions(new.assign(c=1.0), method=method)["c"] == 1.0).all()


@pytest.mark.parametrize("value", [np.nan, 5.0])
@pytest.mark.parametrize("estimator", [MBPCA, MBPLS])
def test_multiblock_contributions_give_a_column_left_out_nothing(estimator: type, value: float) -> None:
    """As for PCA and PLS: the column contributes 0, the rest as without it; MBPLS gave it NaN (#1114)."""
    x, y = _data()
    blocks = _blocks(x.assign(c=np.nan))
    target = () if estimator is MBPCA else (y,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the column with nothing observed, named in the fit
        model = estimator(n_components=2).fit(blocks, *target)
    without = estimator(n_components=2).fit({"p": blocks["p"], "q": blocks["q"].drop(columns="c")}, *target)

    new = {name: frame.iloc[2:6] for name, frame in _blocks(x.assign(c=value)).items()}
    new_without = {"p": new["p"], "q": new["q"].drop(columns="c")}
    for name in ("spe_contributions", "score_contributions"):
        result, expected = getattr(model, name)(new), getattr(without, name)(new_without)
        assert (result["q"]["c"] == 0.0).all()
        _close(result["q"].drop(columns="c"), expected["q"])
        _close(result["p"], expected["p"])
    group = model.group_contributions(new, group=["r2", "r3"])
    assert group["q"]["c"] == 0.0
    _close(group["q"].drop("c"), without.group_contributions(new_without, group=["r2", "r3"])["q"])

    del model._unobserved_columns  # a model pickled before #1098 left no column out: it reads the column as given
    given = {"p": new["p"], "q": new["q"].assign(c=5.0)}
    assert (model.spe_contributions(given)["q"]["c"] > 0.0).all()
    _close(model.score_contributions(given)["q"].drop(columns="c"), without.score_contributions(new_without)["q"])


def test_pls_invert_sets_no_value_for_a_column_left_out() -> None:
    """The model says nothing about the column, so x_new has NaN there, as predict has for a response left out.

    ``invert`` gave it 0 on X's original scale, a value no row ever had (#1114).
    """
    x, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the column with nothing observed, named in the fit
        model = PLS(n_components=2).fit(x.assign(c=np.nan), y)
    without = PLS(n_components=2).fit(x.drop(columns="c"), y)

    result, expected = model.invert([1.0, -0.5]), without.invert([1.0, -0.5])
    assert np.isnan(result.x_new["c"])
    _close(result.x_new.drop("c"), expected.x_new)
    _close(result.scores, expected.scores)
    _close(result.y_hat, expected.y_hat)
    _close(model.predict(result.x_new.to_frame().T), expected.y_hat.to_frame().T)

    del model._unobserved_columns  # a model pickled before #1098 left no column out
    assert model.invert([1.0, -0.5]).x_new.notna().all()


@pytest.mark.parametrize("desired", [{"y1": 1.0}, {"y1": 1.0, "y2": 3.0}])
def test_pls_invert_does_not_target_a_response_left_out(desired: dict) -> None:
    """A response left out needs no target, and is not predicted at x_new: NaN, as predict gives it.

    ``invert`` refused a target without a value for it, and predicted it at 0 whatever was asked (#1114).
    """
    x, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the response with nothing observed, named in the fit
        model = PLS(n_components=2).fit(x, y.assign(y2=np.nan))
    without = PLS(n_components=2).fit(x, y[["y1"]])

    result, expected = model.invert(desired), without.invert(1.0)
    assert np.isnan(result.y_hat["y2"])
    _close(result.y_hat[["y1"]], expected.y_hat)
    _close(result.x_new, expected.x_new)
    _close(result.scores, expected.scores)
    assert result.null_space_dimension == expected.null_space_dimension == 1
    with pytest.raises(ValueError, match="y_desired is missing value"):
        model.invert({"y2": 3.0})

    del model._unobserved_targets  # a model pickled before #1098 left no response out
    assert model.invert({"y1": 1.0, "y2": 3.0}).y_hat.notna().all()
