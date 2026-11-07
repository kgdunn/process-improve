"""One rule for a new row with a missing cell, across the multivariate family (#1144, items 1 and 4).

Every estimator scores such a row from the cells it has: ``PCA`` and ``PLS`` by trimmed score
regression, the estimator ``project`` uses by default, and ``MBPCA``, ``MBPLS`` and ``TPLS`` by
the masked regressions of their own fits. ``PCA`` and ``PLS`` gave NaN, and ``PLS`` a T² of 0.
The contributions of the row then add up to the score, T² and SPE the model gives it.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate.methods import (
    MBPCA,
    MBPLS,
    PCA,
    PLS,
    PLSDA,
    PRM,
    TPLS,
    DataFrameDict,
    MCUVScaler,
)

GAPPY = ["r3", "r10", "r25"]  # the rows with a missing cell, as new rows
COMPLETE = ["r0", "r1", "r2", "r4", "r5"]


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forty complete rows of six columns on two latent directions, and one response."""
    rng = np.random.default_rng(7)
    latent = rng.standard_normal((40, 2))
    index = [f"r{i}" for i in range(40)]
    X = pd.DataFrame(
        latent @ rng.standard_normal((2, 6)) + 0.2 * rng.standard_normal((40, 6)), index=index, columns=list("abcdef")
    )
    y = pd.DataFrame(latent @ [[1.0], [-0.5]] + 0.1 * rng.standard_normal((40, 1)), index=index, columns=["y"])
    return X, y


def _with_gaps(X: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of ``X``, with one missing cell in each of three rows, in different columns and blocks."""
    gappy = X.copy()
    gappy.loc["r3", "b"] = np.nan
    gappy.loc["r10", "e"] = np.nan
    gappy.loc["r25", ["a", "f"]] = np.nan
    return gappy


def _blocks(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"b1": X[["a", "b", "c"]], "b2": X[["d", "e", "f"]]}


def _tpls_data() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    """Return a D matrix and its F, Z and Y blocks: 40 blends of 4 materials, one group "G"."""
    rng = np.random.default_rng(0)
    index = [f"r{i}" for i in range(40)]
    materials = [f"mat{j}" for j in range(4)]
    d_matrix = {"G": pd.DataFrame(rng.normal(size=(4, 3)), index=materials, columns=list("pqr"))}
    blocks = {
        "F": {"G": pd.DataFrame(rng.random((40, 4)), index=index, columns=materials)},
        "Z": {"Z1": pd.DataFrame(rng.normal(size=(40, 3)), index=index, columns=["z1", "z2", "z3"])},
        "Y": {"Y1": pd.DataFrame({"y": rng.normal(size=40)}, index=index)},
    }
    return d_matrix, blocks


# Each case fits a model on complete data and returns a function that diagnoses new rows of ``X``
# (with the gaps of _with_gaps, or without) and returns their scores, T², SPE and prediction.
Diagnosed = dict[str, np.ndarray]


def _pca_case() -> Callable[[bool], Diagnosed]:
    X, _ = _data()
    scaled = MCUVScaler().fit_transform(X)
    model = PCA(n_components=2).fit(scaled)

    def diagnosed(gaps: bool) -> Diagnosed:
        result = model.diagnose(_with_gaps(scaled) if gaps else scaled)
        return {
            "scores": result.scores.to_numpy(),
            "t2": result.cumulative_hotellings_t2.iloc[:, -1].to_numpy(),
            "spe": result.spe.to_numpy(),
        }

    return diagnosed


def _pls_case() -> Callable[[bool], Diagnosed]:
    X, y = _data()
    model = PLS(n_components=2).fit(X, y)

    def diagnosed(gaps: bool) -> Diagnosed:
        result = model.diagnose(_with_gaps(X) if gaps else X)
        return {
            "scores": result.scores.to_numpy(),
            "t2": result.cumulative_hotellings_t2.iloc[:, -1].to_numpy(),
            "spe": result.spe.to_numpy(),
            "y_hat": result.y_hat.to_numpy(),
        }

    return diagnosed


def _multiblock_case(estimator: type) -> Callable[[], Callable[[bool], Diagnosed]]:
    def case() -> Callable[[bool], Diagnosed]:
        X, y = _data()
        model = estimator(n_components=2)
        model = model.fit(_blocks(X)) if estimator is MBPCA else model.fit(_blocks(X), y)

        def diagnosed(gaps: bool) -> Diagnosed:
            result = model.diagnose(_blocks(_with_gaps(X) if gaps else X))
            out = {
                "scores": result.super_scores.to_numpy(),
                "t2": result.cumulative_hotellings_t2.iloc[:, -1].to_numpy(),
                "spe": np.column_stack([spe.to_numpy() for spe in result.block_spe.values()]),
            }
            if estimator is MBPLS:
                out["y_hat"] = result.predictions.to_numpy()
            return out

        return diagnosed

    return case


def _tpls_case() -> Callable[[bool], Diagnosed]:
    d_matrix, blocks = _tpls_data()
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(copy.deepcopy(blocks)))

    def diagnosed(gaps: bool) -> Diagnosed:
        new = copy.deepcopy(blocks)
        if gaps:
            new["Z"]["Z1"].loc["r3", "z2"] = np.nan
            new["F"]["G"].loc["r10", "mat1"] = np.nan
            new["Z"]["Z1"].loc["r25", "z1"] = np.nan
            new["F"]["G"].loc["r25", "mat3"] = np.nan
        result = model.diagnose(DataFrameDict(new))
        return {
            "scores": result.t_scores_super.to_numpy(),
            "t2": result.cumulative_hotellings_t2.iloc[:, -1].to_numpy(),
            "spe": np.column_stack(
                [frame.iloc[:, -1].to_numpy() for group in result.spe.values() for frame in group.values()]
            ),
            "y_hat": result.hat["Y1"].to_numpy(),
        }

    return diagnosed


FAMILY = {
    "PCA": _pca_case,
    "PLS": _pls_case,
    "MBPCA": _multiblock_case(MBPCA),
    "MBPLS": _multiblock_case(MBPLS),
    "TPLS": _tpls_case,
}
ROWS = {name: [int(row[1:]) for row in GAPPY] for name in FAMILY}


@pytest.mark.parametrize("name", list(FAMILY))
def test_a_new_row_with_a_missing_cell_gets_finite_scores_t2_spe_and_prediction(name: str) -> None:
    """Every estimator of the family scores the row from the cells it has; PCA and PLS gave NaN."""
    diagnosed = FAMILY[name]()(True)
    gappy = ROWS[name]
    for key, values in diagnosed.items():
        assert np.isfinite(values[gappy]).all(), f"{name} {key} is not finite for a row with a missing cell"


@pytest.mark.parametrize("name", list(FAMILY))
def test_a_missing_cell_leaves_the_complete_rows_bit_for_bit_as_they_were(name: str) -> None:
    """A gap in one row changes no number of another row, so a complete row keeps the complete-data path."""
    diagnose = FAMILY[name]()
    with_gaps, without = diagnose(True), diagnose(False)
    complete = np.setdiff1d(np.arange(40), ROWS[name])
    for key in with_gaps:
        np.testing.assert_array_equal(with_gaps[key][complete], without[key][complete], err_msg=f"{name} {key}")


@pytest.fixture(scope="module")
def pca_and_rows() -> tuple[PCA, pd.DataFrame]:
    X, _ = _data()
    scaled = MCUVScaler().fit_transform(X)
    return PCA(n_components=2).fit(scaled), _with_gaps(scaled)


@pytest.fixture(scope="module")
def pls_and_rows() -> tuple[PLS, pd.DataFrame]:
    X, y = _data()
    return PLS(n_components=2).fit(X, y), _with_gaps(X)


def test_pca_scores_a_row_with_a_missing_cell_as_project_does_by_default(
    pca_and_rows: tuple[PCA, pd.DataFrame],
) -> None:
    """``transform`` and ``diagnose`` give the scores, T² and SPE of ``project()``'s default estimator, TSR."""
    model, rows = pca_and_rows
    projected, result = model.project(rows), model.diagnose(rows)
    pd.testing.assert_frame_equal(result.scores, projected.scores, rtol=1e-12, atol=1e-14)
    pd.testing.assert_frame_equal(model.transform(rows), result.scores)
    np.testing.assert_allclose(result.cumulative_hotellings_t2.iloc[:, -1], projected.hotellings_t2, rtol=1e-12)
    np.testing.assert_allclose(result.spe, projected.spe, rtol=1e-12)


def test_pls_scores_and_predicts_a_row_with_a_missing_cell_as_project_does_by_default(
    pls_and_rows: tuple[PLS, pd.DataFrame],
) -> None:
    """``transform``, ``diagnose`` and ``predict`` give the TSR scores and prediction of ``project()``.

    The residuals of such a row are NaN at its missing cells, and SPE is their length over the others.
    """
    model, rows = pls_and_rows
    projected, result = model.project(rows), model.diagnose(rows)
    pd.testing.assert_frame_equal(result.scores, projected.scores, rtol=1e-12, atol=1e-14)
    pd.testing.assert_frame_equal(model.transform(rows), result.scores)
    pd.testing.assert_frame_equal(result.y_hat, projected.y_hat, rtol=1e-12)
    pd.testing.assert_frame_equal(model.predict(rows), result.y_hat)
    np.testing.assert_allclose(result.cumulative_hotellings_t2.iloc[:, -1], projected.hotellings_t2, rtol=1e-12)
    np.testing.assert_allclose(result.spe, projected.spe, rtol=1e-12)
    assert result.residuals.isna().equals(rows.isna())
    np.testing.assert_allclose(np.sqrt((result.residuals**2).sum(axis=1)), result.spe, rtol=1e-12)


def test_a_row_with_nothing_observed_is_still_nan(
    pca_and_rows: tuple[PCA, pd.DataFrame], pls_and_rows: tuple[PLS, pd.DataFrame]
) -> None:
    """Nothing observed says nothing about the row: NaN, as before, and never the 0 ``project()`` places it at."""
    for model, rows in (pca_and_rows, pls_and_rows):
        empty = rows.copy()
        empty.loc["r0"] = np.nan
        result = model.diagnose(empty)
        assert result.scores.loc["r0"].isna().all()
        assert result.cumulative_hotellings_t2.loc["r0"].isna().all()
        assert np.isnan(result.spe.loc["r0"])
        assert model.transform(empty).loc["r0"].isna().all()
    assert pls_and_rows[0].predict(empty).loc["r0"].isna().all()


@pytest.mark.parametrize("fixture", ["pca_and_rows", "pls_and_rows"])
def test_a_row_too_sparse_to_estimate_is_nan_with_a_warning(fixture: str, request: pytest.FixtureRequest) -> None:
    """One observed cell cannot pin two scores: that row is NaN and named in a warning, the others are scored.

    ``project()`` refuses such a row with ``LinAlgError``; a ridge above 0 estimates it.
    """
    model, rows = request.getfixturevalue(fixture)
    sparse = rows.copy()
    sparse.loc["r1", ["b", "c", "d", "e", "f"]] = np.nan
    with pytest.warns(SpecificationWarning, match=r"^1 row with missing cells has too few observed values"):
        result = model.diagnose(sparse)
    assert result.scores.loc["r1"].isna().all()
    assert np.isnan(result.spe.loc["r1"])
    assert np.isfinite(result.scores.loc[GAPPY].to_numpy()).all()
    with pytest.raises(np.linalg.LinAlgError):
        model.project(sparse.loc[["r1"]])
    assert np.isfinite(model.project(sparse.loc[["r1"]], ridge=0.1).scores.to_numpy()).all()


def test_the_estimators_that_inherit_from_pls_follow() -> None:
    """``PLSDA`` and ``PRM`` predict a row with a missing cell; PLSDA gave it the first class, from NaN."""
    X, y = _data()
    rows = _with_gaps(X)
    labels = np.where(y["y"] > 0, "high", "low")
    plsda = PLSDA(n_components=2).fit(X, labels)
    assert np.isfinite(plsda.decision_function(rows).loc[GAPPY].to_numpy()).all()
    np.testing.assert_allclose(plsda.decision_function(rows), plsda.project(rows).y_hat, rtol=1e-12)
    assert np.isfinite(PRM(n_components=2).fit(X, y).predict(rows).loc[GAPPY].to_numpy()).all()


# The estimators with contribution methods. TPLS has none.
Totals = list[tuple[str, np.ndarray, np.ndarray]]


def _single_block_totals(model: PCA | PLS, rows: pd.DataFrame) -> Totals:
    """Each contribution's row sum, beside the number it splits, as ``diagnose`` gives it."""
    result = model.diagnose(rows)
    t2 = result.cumulative_hotellings_t2.iloc[:, -1].to_numpy()
    totals = [
        ("T2", model.t2_contributions(rows).sum(axis=1).to_numpy(), t2),
        ("SPE", (model.spe_contributions(rows) ** 2).sum(axis=1).to_numpy(), result.spe.to_numpy() ** 2),
    ]
    for a in (1, 2):
        parts = model.score_contributions(rows, component=a).sum(axis=1).to_numpy()
        totals.append((f"t{a}", parts, result.scores.iloc[:, a - 1].to_numpy()))
        shift = model.group_contributions(rows, group=GAPPY[:2], reference=COMPLETE, component=a).sum()
        scores = result.scores.iloc[:, a - 1]
        totals.append(
            (f"group t{a}", np.array([shift]), np.array([scores[GAPPY[:2]].mean() - scores[COMPLETE].mean()]))
        )
    return totals


def _multiblock_totals(model: MBPCA | MBPLS, rows: dict[str, pd.DataFrame]) -> Totals:
    """As _single_block_totals, summed over the blocks; these models have no T² contributions."""
    result = model.diagnose(rows)
    squared = model.spe_contributions(rows)
    totals = [
        (f"SPE[{name}]", squared[name].sum(axis=1).to_numpy(), result.block_spe[name].to_numpy() ** 2)
        for name in model.block_names_
    ]
    for a in (1, 2):
        parts = sum(frame.sum(axis=1) for frame in model.score_contributions(rows, component=a).values())
        totals.append((f"t{a}", parts.to_numpy(), result.super_scores.iloc[:, a - 1].to_numpy()))
        shift = model.group_contributions(rows, group=GAPPY[:2], reference=COMPLETE, component=a)
        scores = result.super_scores.iloc[:, a - 1]
        totals.append(
            (
                f"group t{a}",
                np.array([sum(part.sum() for part in shift.values())]),
                np.array([scores[GAPPY[:2]].mean() - scores[COMPLETE].mean()]),
            )
        )
    return totals


def _contribution_totals(name: str) -> Totals:
    X, y = _data()
    if name == "PCA":
        scaled = MCUVScaler().fit_transform(X)
        return _single_block_totals(PCA(n_components=2).fit(scaled), _with_gaps(scaled))
    if name == "PLS":
        return _single_block_totals(PLS(n_components=2).fit(X, y), _with_gaps(X))
    if name == "MBPCA":
        return _multiblock_totals(MBPCA(n_components=2).fit(_blocks(X)), _blocks(_with_gaps(X)))
    return _multiblock_totals(MBPLS(n_components=2).fit(_blocks(X), y), _blocks(_with_gaps(X)))


@pytest.mark.parametrize("name", ["PCA", "PLS", "MBPCA", "MBPLS"])
def test_contributions_of_a_row_with_a_missing_cell_add_up_to_what_the_model_gives_it(name: str) -> None:
    """Score and T² contributions add up to the row's score and T², SPE contributions to SPE² over its cells.

    Whatever estimator scored the row: TSR for PCA and PLS, the masked regressions of the fit for
    MBPCA and MBPLS, whose contributions used to divide by the loadings' length over every cell
    rather than over the row's own (#1141). A missing cell contributes nothing, so the group form,
    the mean of the rows' contributions, adds up to the mean scores.
    """
    for label, parts, whole in _contribution_totals(name):
        assert np.isfinite(whole).all(), f"{name} {label}"
        np.testing.assert_allclose(parts, whole, rtol=1e-10, atol=1e-12, err_msg=f"{name} {label}")


def _select(name: str, X: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    """Run ``select_n_components`` of the estimator ``name`` with three folds and return its main curve."""
    settings = {"max_components": 3, "cv": 3, "random_state": 0}
    if name == "PCA":
        return PCA.select_n_components(X, n_repeats=1, **settings).press
    if name == "PLSDA":
        classes = np.where(y["y"] > y["y"].median(), "high", "low")
        return PLSDA.select_n_components(X, classes, n_repeats=1, **settings).rmsecv["total"]
    if name == "MBPLS":
        return MBPLS.select_n_components(_blocks(X), y, n_repeats=1, **settings).rmsecv
    if name == "PLS.nested_cv":
        return PLS.nested_cv(X, y, outer_cv=3, inner_cv=3, max_components=3, random_state=0).rmsep
    return PLS.select_n_components(X, y, n_repeats=1, **settings).rmsecv["total"]


@pytest.mark.parametrize("name", ["PCA", "PLS", "PLSDA", "MBPLS", "PLS.nested_cv"])
def test_select_n_components_reads_a_pandas_missing_value_as_missing(name: str) -> None:
    """A nullable column's pd.NA is a missing cell to ``select_n_components``, as it is to ``fit`` (#1144).

    PCA raised ``TypeError`` on converting it to a float, and PLS (and PLSDA through it, and
    ``PLS.nested_cv``) on testing it for NaN, where ``fit`` took it. The curve is the one of the
    same data with NaN.
    """
    X, y = _data()
    gappy = _with_gaps(X)
    nullable = gappy.astype("Float64")
    assert nullable.isna().equals(gappy.isna())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # whatever the selection warns of, it warns of for both
        expected = _select(name, gappy, y)
        got = _select(name, nullable, y.astype("Float64"))
    pd.testing.assert_series_equal(got, expected, rtol=1e-12)
