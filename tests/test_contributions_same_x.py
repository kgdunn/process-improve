"""Contributions take the same ``X`` as the model's ``predict`` / ``transform`` / ``diagnose`` (#1125).

A model that centres and scales ``X`` itself (``PLS(scale=True)``, the default, and ``PLSDA`` and
``PRM``, which inherit it) applies that preprocessing before it decomposes a row, so the training
data, in the units the model was fitted on, reproduces the stored ``hotellings_t2_``, ``spe_`` and
``scores_``. A model that scales nothing (``PCA``, ``PLS(scale=False)``) is given data that is
already scaled, and nothing changes for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate import PCA, PLS, MCUVScaler
from process_improve.multivariate._diagnostics import (
    group_contributions,
    meda,
    omeda,
    score_contributions,
    spe_contributions,
    t2_contributions,
)
from process_improve.multivariate.methods import MBPCA, MBPLS, PLSDA, PRM, AdaptivePCA, AdaptivePLS

N_COMPONENTS = 2


@pytest.fixture(scope="module")
def raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thirty rows on two latent directions, in units far from 0 and 1, and one response.

    Each column is offset by 50 and spread over a different unit, so a model's centring and
    scaling changes every value: a contribution that skipped it would be far off.
    """
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((30, 2))
    values = latent @ rng.standard_normal((2, 5)) + 0.2 * rng.standard_normal((30, 5))
    X = pd.DataFrame(
        50.0 + values * [1.0, 10.0, 100.0, 0.1, 5.0],
        columns=list("abcde"),
        index=[f"r{i}" for i in range(30)],
    )
    y = pd.DataFrame({"y": 20.0 + 3.0 * latent[:, 0] + 0.1 * rng.standard_normal(30)}, index=X.index)
    return X, y


def _fitted(name: str, X: pd.DataFrame, y: pd.DataFrame) -> tuple[PCA | PLS, pd.DataFrame]:
    """Return the fitted model and the X it was fitted on, which its predict / transform take."""
    scaled = MCUVScaler().fit_transform(X)
    if name == "PCA":  # scales nothing itself, so it is fitted on scaled data
        return PCA(n_components=N_COMPONENTS).fit(scaled), scaled
    if name == "PLS, scale=False":
        return PLS(n_components=N_COMPONENTS, scale=False).fit(scaled, MCUVScaler().fit_transform(y)), scaled
    if name == "PLSDA":
        return PLSDA(n_components=N_COMPONENTS).fit(X, np.where(y["y"] > y["y"].median(), "high", "low")), X
    if name == "PRM":
        return PRM(n_components=N_COMPONENTS).fit(X, y), X
    return PLS(n_components=N_COMPONENTS).fit(X, y), X


SINGLE_BLOCK = ["PCA", "PLS, scale=True", "PLS, scale=False", "PLSDA", "PRM"]


def _t2(diagnosed: object) -> pd.Series:
    """Return the full model's T2 from ``diagnose``, where PCA gives it cumulative, one column per component."""
    t2 = diagnosed.hotellings_t2  # type: ignore[attr-defined]
    return t2.iloc[:, -1] if isinstance(t2, pd.DataFrame) else t2


@pytest.mark.parametrize("name", SINGLE_BLOCK)
def test_the_training_data_reproduces_the_stored_diagnostics(name: str, raw: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The data the model was fitted on, in the same units, sums to the stored T2, SPE and scores."""
    model, X = _fitted(name, *raw)
    np.testing.assert_allclose(t2_contributions(model, X).sum(axis=1), model.hotellings_t2_.iloc[:, -1], rtol=1e-9)
    np.testing.assert_allclose(
        (spe_contributions(model, X) ** 2).sum(axis=1), model.spe_.iloc[:, -1] ** 2, rtol=1e-9, atol=1e-20
    )
    for component in range(1, N_COMPONENTS + 1):
        np.testing.assert_allclose(
            score_contributions(model, X, component=component).sum(axis=1),
            model.scores_.iloc[:, component - 1],
            rtol=1e-9,
            atol=1e-12,
        )


@pytest.mark.parametrize("name", SINGLE_BLOCK)
def test_a_new_row_gives_what_diagnose_gives(name: str, raw: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """New rows in the fitted units decompose the T2, SPE and scores that ``diagnose`` reports for them."""
    model, X = _fitted(name, *raw)
    moves = np.array([[1, 0, 0, 0, 0], [0, -1, 0, 1, 0], [0, 0, 0, 0, 2], [0, 0, 0, 0, 0]])
    new = X.iloc[:4] + 3.0 * X.std().to_numpy() * moves
    diagnosed = model.diagnose(new)
    np.testing.assert_allclose(t2_contributions(model, new).sum(axis=1), _t2(diagnosed), rtol=1e-9)
    residuals = spe_contributions(model, new)
    np.testing.assert_allclose((residuals**2).sum(axis=1), diagnosed.spe**2, rtol=1e-9)
    if "residuals" in diagnosed:  # PLS reports the residual of every variable
        np.testing.assert_allclose(residuals, diagnosed.residuals, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        score_contributions(model, new, component=2).sum(axis=1), diagnosed.scores.iloc[:, 1], rtol=1e-9
    )


def test_scaling_inside_the_model_or_by_hand_gives_the_same_contributions(
    raw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A ``scale=True`` model given raw rows equals a ``scale=False`` model given the same rows scaled by hand.

    The two models are the same model. Every function that decomposes a row of ``X`` gives the same
    answer from the two, on rows with a missing cell too, which take the missing-data path.
    """
    X, y = raw
    scaler_x, scaler_y = MCUVScaler().fit(X), MCUVScaler().fit(y)
    inside = PLS(n_components=N_COMPONENTS).fit(X, y)
    by_hand = PLS(n_components=N_COMPONENTS, scale=False).fit(scaler_x.transform(X), scaler_y.transform(y))
    gappy = X.copy()
    gappy.iloc[3, 1] = np.nan
    groups = {"group": X.index[:5], "reference": X.index[5:15]}
    calls = {
        "t2_contributions": t2_contributions,
        "spe_contributions": spe_contributions,
        "score_contributions": lambda model, data: score_contributions(model, data, component=2),
        "group_contributions": lambda model, data: group_contributions(model, data, **groups),
        "meda": meda,
        "omeda": lambda model, data: omeda(model, data, **groups),
    }
    for name, call in calls.items():
        for data in (X, gappy):
            expected = call(by_hand, scaler_x.transform(data))
            np.testing.assert_allclose(call(inside, data), expected, rtol=1e-8, atol=1e-10, err_msg=name)


@pytest.fixture(scope="module")
def blocks(raw: tuple[pd.DataFrame, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Split the raw columns into two blocks."""
    X, _ = raw
    return {"first": X[["a", "b"]], "second": X[["c", "d", "e"]]}


@pytest.mark.parametrize("kind", ["MBPCA", "MBPLS"])
def test_multiblock_models_take_raw_blocks(
    kind: str, blocks: dict[str, pd.DataFrame], raw: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """MBPCA and MBPLS always centre and scale each block, and their contributions take the raw blocks.

    The training blocks reproduce the stored block SPE and super scores, and new rows reproduce
    what ``diagnose`` reports for them.
    """
    model = MBPCA(n_components=N_COMPONENTS).fit(blocks) if kind == "MBPCA" else MBPLS(N_COMPONENTS).fit(blocks, raw[1])
    new = {name: block.iloc[:4] * 1.1 for name, block in blocks.items()}
    diagnosed = model.diagnose(new)
    for data, spe, scores in (
        (blocks, {name: frame.iloc[:, -1] for name, frame in model.block_spe_.items()}, model.super_scores_),
        (new, diagnosed.block_spe, diagnosed.super_scores),
    ):
        for name, squared in model.spe_contributions(data).items():  # both return the squared residuals
            np.testing.assert_allclose(squared.sum(axis=1), spe[name] ** 2, rtol=1e-8, atol=1e-12)
        for component in range(1, N_COMPONENTS + 1):
            total = sum(frame.sum(axis=1) for frame in model.score_contributions(data, component=component).values())
            np.testing.assert_allclose(total, scores[component], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("kind", ["AdaptivePCA", "AdaptivePLS"])
def test_adaptive_models_take_the_rows_transform_takes(kind: str, raw: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """An adaptive model centres and scales a row with its current centre and spread, for contributions too."""
    X, y = raw
    model = AdaptivePCA(N_COMPONENTS).fit(X) if kind == "AdaptivePCA" else AdaptivePLS(N_COMPONENTS).fit(X, y)
    model.partial_fit(X.iloc[:5] + 1.0, None if kind == "AdaptivePCA" else y.iloc[:5])  # the centre moves
    new = X.iloc[5:9]
    scores = model.transform(new)
    np.testing.assert_allclose(score_contributions(model, new, component=2).sum(axis=1), scores.iloc[:, 1], rtol=1e-9)
    t2 = ((scores / model.scaling_factor_for_scores_.to_numpy()) ** 2).sum(axis=1)
    np.testing.assert_allclose(t2_contributions(model, new).sum(axis=1), t2, rtol=1e-9)


def test_a_training_row_with_nothing_observed_is_decomposed_as_fit_scored_it(
    raw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """``fit`` places a row with no observed value at the centre; so do the contributions and ``project``.

    Its scores, T2 and SPE are 0, and so are the sums of its contributions, which are NaN at
    every cell, as at any missing cell. Before #1125 the contributions and ``project`` refused it.
    """
    X, y = raw
    gappy = X.copy()
    gappy.loc["r4"] = np.nan
    model = PLS(n_components=N_COMPONENTS).fit(gappy, y)
    assert (model.scores_.loc["r4"] == 0.0).all()
    assert model.hotellings_t2_.loc["r4"].iloc[-1] == model.spe_.loc["r4"].iloc[-1] == 0.0
    for frame in (
        t2_contributions(model, gappy),
        spe_contributions(model, gappy),
        score_contributions(model, gappy, component=1),
    ):
        assert frame.loc["r4"].isna().all()
        assert frame.loc["r4"].sum() == 0.0
    projected = model.project(gappy)
    assert (projected.scores.loc["r4"] == 0.0).all()
    np.testing.assert_allclose(projected.y_hat.loc["r4"], model.predictions_.loc["r4"], rtol=1e-12)
