"""The multivariate family answers the same input the same way, and names the block at fault (#1114).

Each test is parametrised over the estimators, or the tools, its item applies to, so a member that
drifts from the others fails here.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate._impute import impute_low_rank
from process_improve.multivariate._limits import spe_calculation
from process_improve.multivariate.methods import (
    ASCA,
    FAMD,
    GPA,
    MBPCA,
    MBPLS,
    OPLS,
    PCA,
    PLS,
    PLSDA,
    PRM,
    TPLS,
    AdaptivePCA,
    AdaptivePLS,
    DataFrameDict,
    NotEnoughVarianceError,
    compare_cv_criteria,
    pseudo_validation_set,
    randomization_test_mbpls,
)

N_ROWS = 30


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thirty rows of five columns on two latent directions, and one response."""
    rng = np.random.default_rng(3)
    latent = rng.standard_normal((N_ROWS, 2))
    X = pd.DataFrame(
        latent @ rng.standard_normal((2, 5)) + 0.3 * rng.standard_normal((N_ROWS, 5)), columns=list("abcde")
    )
    y = pd.DataFrame({"y": latent @ [1.0, -0.5] + 0.1 * rng.standard_normal(N_ROWS)})
    return X, y


def _blocks(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"b1": X[["a", "b"]], "b2": X[["c", "d", "e"]]}


def _labels(y: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(y["y"].to_numpy() > 0, "hi", "lo"))


def _tpls_inputs(X: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], DataFrameDict]:
    """TPLS's ``d_matrix`` and blocks, with X as the one Z block."""
    rng = np.random.default_rng(0)
    materials = [f"mat{j}" for j in range(4)]
    d_matrix = {"G": pd.DataFrame(rng.normal(size=(4, 3)), index=materials, columns=list("pqr"))}
    blocks = {
        "F": {"G": pd.DataFrame(rng.random((N_ROWS, 4)), index=X.index, columns=materials)},
        "Z": {"Z1": X},
        "Y": {"Y1": pd.DataFrame({"y": rng.normal(size=N_ROWS)}, index=X.index)},
    }
    return d_matrix, DataFrameDict(blocks)


# --- The response block is checked as X is ----------------------------------------------------------

#: Each fits a regression of Y on X, X split into two blocks for MBPLS.
REGRESSIONS: dict[str, Callable[[pd.DataFrame, pd.DataFrame], object]] = {
    "PLS": lambda X, Y: PLS(n_components=2).fit(X, Y),
    "PRM": lambda X, Y: PRM(n_components=2).fit(X, Y),
    "OPLS": lambda X, Y: OPLS(n_orthogonal_components=1).fit(X, Y),
    "MBPLS": lambda X, Y: MBPLS(n_components=2).fit(_blocks(X), Y),
    "AdaptivePLS": lambda X, Y: AdaptivePLS(n_components=2).fit(X, Y),
}


@pytest.mark.parametrize("estimator", list(REGRESSIONS))
def test_an_infinite_response_is_named_as_y(estimator: str) -> None:
    """PLS, PRM and MBPLS blamed X: the scaler was the first to meet Y, and named its input X (#1114)."""
    X, y = _data()
    y.iloc[3, 0] = np.inf
    with pytest.raises(ValueError, match=r"^Input Y contains infinity or a value too large for dtype\('float64'\)\.$"):
        REGRESSIONS[estimator](X, y)


@pytest.mark.parametrize("n_y_rows", [20, 40])
@pytest.mark.parametrize("estimator", list(REGRESSIONS))
def test_a_response_of_another_length_gets_the_family_error(estimator: str, n_y_rows: int) -> None:
    """PRM's robust centring met a Y of another length as an IndexError; MBPLS had its own wording (#1114)."""
    X, y = _data()
    y = pd.DataFrame({"y": np.resize(y["y"].to_numpy(), n_y_rows)})
    message = rf"^The X and Y arrays must have the same number of rows: X has {N_ROWS} and Y has {n_y_rows}\.$"
    with pytest.raises(ValueError, match=message):
        REGRESSIONS[estimator](X, y)


@pytest.mark.parametrize("estimator", list(REGRESSIONS))
def test_an_x_with_no_variance_is_refused_before_any_warning(estimator: str) -> None:
    """OPLS divided by the zero norm of X'y first, and its RuntimeWarning came before the error (#1114)."""
    X, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NotEnoughVarianceError, match=r"^There is no variance"):
            REGRESSIONS[estimator](X * 0.0, y)


def test_opls_refuses_an_x_with_no_covariance_with_y() -> None:
    """An X that varies but is orthogonal to y gave OPLS a NaN model, after a RuntimeWarning (#1114)."""
    X = pd.DataFrame({"a": [1.0, -1.0, 1.0, -1.0], "b": [1.0, 1.0, -1.0, -1.0]})
    y = pd.DataFrame({"y": [1.0, -1.0, -1.0, 1.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NotEnoughVarianceError, match=r"^There is nothing in X to explain Y by"):
            OPLS(n_orthogonal_components=0).fit(X, y)


# --- A block with no columns -------------------------------------------------------------------------

#: Each fits a single-block estimator on X (and Y, or labels).
SINGLE_BLOCK: dict[str, Callable[[pd.DataFrame, pd.DataFrame], object]] = {
    "PCA": lambda X, _: PCA(n_components=1).fit(X),
    "PLS": lambda X, Y: PLS(n_components=1).fit(X, Y),
    "PLSDA": lambda X, Y: PLSDA(n_components=1).fit(X, _labels(Y)),
    "PRM": lambda X, Y: PRM(n_components=1).fit(X, Y),
    "OPLS": lambda X, Y: OPLS(n_orthogonal_components=0).fit(X, Y),
    "AdaptivePCA": lambda X, _: AdaptivePCA(n_components=1).fit(X),
    "AdaptivePLS": lambda X, Y: AdaptivePLS(n_components=1).fit(X, Y),
    "ASCA": lambda X, _: ASCA(n_components=1).fit(X, pd.DataFrame({"f": ["a", "b"] * (N_ROWS // 2)})),
    "FAMD": lambda X, _: FAMD(n_components=1).fit(X),
}


@pytest.mark.parametrize("estimator", list(SINGLE_BLOCK))
def test_an_x_with_no_columns_is_refused_by_name(estimator: str) -> None:
    """Numpy's "at least one array or dtype is required" named neither the block nor the cause (#1077)."""
    X, y = _data()
    with pytest.raises(ValueError, match=r"^Block X has no columns; a block needs at least one\.$"):
        SINGLE_BLOCK[estimator](X[[]], y)


@pytest.mark.parametrize("estimator", ["PLS", "PRM", "AdaptivePLS"])
def test_a_y_with_no_columns_is_refused_by_name(estimator: str) -> None:
    """The response block, the same way, as MBPLS and TPLS name theirs (#1077)."""
    X, y = _data()
    with pytest.raises(ValueError, match=r"^Block Y has no columns; a block needs at least one\.$"):
        SINGLE_BLOCK[estimator](X, y[[]])


# --- A constant column -------------------------------------------------------------------------------

#: Each fits an estimator on X, whose column "c" is constant, and names the block that holds "c".
CONSTANT_C: dict[str, tuple[Callable[[pd.DataFrame, pd.DataFrame], object], str]] = {
    "PCA": (lambda X, _: PCA(n_components=2).fit(X), "X"),
    "PLS": (lambda X, Y: PLS(n_components=2).fit(X, Y), "X"),
    "PLSDA": (lambda X, Y: PLSDA(n_components=2).fit(X, _labels(Y)), "X"),
    "PRM": (lambda X, Y: PRM(n_components=2).fit(X, Y), "X"),
    "OPLS": (lambda X, Y: OPLS(n_orthogonal_components=1).fit(X, Y), "X"),
    "MBPCA": (lambda X, _: MBPCA(n_components=2).fit(_blocks(X)), 'X["b2"]'),
    "MBPLS": (lambda X, Y: MBPLS(n_components=2).fit(_blocks(X), Y), 'X["b2"]'),
    "TPLS": (lambda X, _: TPLS(n_components=2, d_matrix=_tpls_inputs(X)[0]).fit(_tpls_inputs(X)[1]), 'Z["Z1"]'),
}


@pytest.mark.parametrize("estimator", list(CONSTANT_C))
def test_a_constant_column_gets_one_warning_across_the_family(estimator: str) -> None:
    """TPLS named a column with no variance; PCA, PLS, MBPCA and MBPLS fitted it without a word (#1114)."""
    X, y = _data()
    X["c"] = 3.0
    fit, label = CONSTANT_C[estimator]
    message = (
        rf"^Block {re.escape(label)} has columns with no variance, which carry no information: 'c'\. "
        r"Drop them, or check the data for a stuck sensor\.$"
    )
    with pytest.warns(UserWarning, match=message):
        fit(X, y)


@pytest.mark.parametrize("estimator", ["PLS", "MBPLS"])
def test_a_constant_response_is_named_too(estimator: str) -> None:
    """A response with no variance carries nothing for X to explain, as TPLS says of its Y blocks."""
    X, y = _data()
    Y = y.assign(stuck=1.0)
    with pytest.warns(UserWarning, match=r"^Block Y has columns with no variance, which carry no information: 'stuck'"):
        REGRESSIONS[estimator](X, Y)


@pytest.mark.parametrize("estimator", list(CONSTANT_C))
def test_columns_that_vary_get_no_such_warning(estimator: str) -> None:
    X, y = _data()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*columns with no variance")
        CONSTANT_C[estimator][0](X, y)


# --- A training row with nothing observed ------------------------------------------------------------


def _spe_of_rows(estimator: str, X: pd.DataFrame, y: pd.DataFrame) -> list[np.ndarray]:
    """Fit on X and return the training SPE, one array of shape (n_rows, n_columns) per block."""
    if estimator == "PCA":
        return [PCA(n_components=2).fit(X).spe_.to_numpy()]
    if estimator == "PLS":
        return [PLS(n_components=2).fit(X, y).spe_.to_numpy()]
    if estimator == "MBPCA":
        return [frame.to_numpy() for frame in MBPCA(n_components=2).fit(_blocks(X)).block_spe_.values()]
    if estimator == "MBPLS":
        return [frame.to_numpy() for frame in MBPLS(n_components=2).fit(_blocks(X), y).block_spe_.values()]
    d_matrix, blocks = _tpls_inputs(X)
    blocks["F"]["G"].iloc[5] = np.nan
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(blocks)
    return [model.spe["Z"]["Z1"].to_numpy(), model.spe["F"]["G"].to_numpy()]


@pytest.mark.parametrize("estimator", ["PCA", "PLS", "MBPCA", "MBPLS", "TPLS"])
def test_a_training_row_with_nothing_observed_has_no_spe(estimator: str) -> None:
    """PCA, PLS and TPLS gave it an SPE of 0, MBPCA and MBPLS NaN; NaN, as ``diagnose`` gives such a row (#1148)."""
    X, y = _data()
    X.iloc[5] = np.nan
    for spe in _spe_of_rows(estimator, X, y):
        others = np.delete(spe, 5, axis=0)
        assert np.isnan(spe[5]).all()
        assert np.isfinite(others).all()


@pytest.mark.parametrize("estimator", ["PCA", "PLS"])
def test_the_spe_limit_leaves_that_row_out(estimator: str) -> None:
    """The limit is over the rows that have an SPE: the same as a fit whose row is left out of the limit."""
    X, y = _data()
    X.iloc[5] = np.nan
    model = PCA(n_components=2).fit(X) if estimator == "PCA" else PLS(n_components=2).fit(X, y)
    others = model.spe_.iloc[:, -1].drop(5).to_numpy()
    assert model.spe_limit(0.95) == pytest.approx(spe_calculation(others, 0.95))


# --- The cross-validation tools take an all-missing row and column, as fit does ----------------------

#: Each runs a cross-validation tool on X and y and returns its pooled PRESS, one value per component count.
CV_TOOLS: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = {
    "PLS.select_n_components": lambda X, y: PLS.select_n_components(
        X, y, max_components=2, n_repeats=1, random_state=0
    ).press.to_numpy(),
    "MBPLS.select_n_components": lambda X, y: MBPLS.select_n_components(
        _blocks(X), y, max_components=2, n_repeats=1, random_state=0
    ).press.to_numpy(),
    "PCA.select_n_components": lambda X, _: PCA.select_n_components(X, max_components=2).press.to_numpy(),
    "PLS.cross_validate": lambda X, y: np.atleast_1d(
        PLS(n_components=2).fit(X, y).cross_validate(X, y, cv=5, random_state=0, show_progress=False).press
    ),
    "compare_cv_criteria": lambda X, y: compare_cv_criteria(
        X, y, max_components=2, n_permutations=9, n_cv_permutations=9, random_state=0
    ).press.to_numpy(),
    "pseudo_validation_set": lambda X, y: (
        pseudo_validation_set(X, y, n_components=2, random_state=0).local_spe.dropna().to_numpy()
    ),
}


@pytest.mark.parametrize("tool", list(CV_TOOLS))
def test_cross_validation_takes_a_row_with_nothing_observed(tool: str) -> None:
    """``fit`` takes such a row; ``select_n_components`` raised RuntimeError and the CV criteria refused it (#1114)."""
    X, y = _data()
    X.iloc[5] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        press = CV_TOOLS[tool](X, y)
    assert np.isfinite(press).all()


@pytest.mark.parametrize("tool", list(CV_TOOLS))
def test_cross_validation_takes_a_column_with_nothing_observed(tool: str) -> None:
    """``fit`` leaves such a column out with a warning; the CV criteria refused it; select warned of an empty mean."""
    X, y = _data()
    X["c"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.filterwarnings("ignore", message=r"Block X.* has columns with no observed values")
        press = CV_TOOLS[tool](X, y)
    assert np.isfinite(press).all()


@pytest.mark.parametrize("tool", ["PLS.select_n_components", "MBPLS.select_n_components", "compare_cv_criteria"])
def test_the_default_component_count_leaves_out_a_column_with_nothing_observed(tool: str) -> None:
    """Such a column adds no component, so the default count of components is capped by the others.

    Capped by every column, the folds were asked for more components than they have, and refused.
    """
    X, y = _data()
    X["c"] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Block X.* has columns with no observed values")
        if tool == "PLS.select_n_components":
            press = PLS.select_n_components(X, y, n_repeats=1, random_state=0).press
        elif tool == "MBPLS.select_n_components":
            press = MBPLS.select_n_components(_blocks(X), y, n_repeats=1, random_state=0).press
        else:
            press = compare_cv_criteria(X, y, n_permutations=9, n_cv_permutations=9, random_state=0).press
    assert len(press) == 4


def _held_out_q2(tool: str, X: pd.DataFrame, y: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Run ``tool`` and return its held-out predictions, its PRESS and its Q2, at one component count."""
    if tool == "PLS.cross_validate":
        result = PLS(n_components=2).fit(X, y).cross_validate(X, y, cv=5, random_state=0, show_progress=False)
        return result.y_hat_cv.to_numpy().ravel(), result.press, float(result.q_squared.iloc[0])
    if tool == "PLS.select_n_components":
        result = PLS.select_n_components(X, y, max_components=2, n_repeats=1, random_state=0, selection_rule="min")
    else:
        result = MBPLS.select_n_components(
            _blocks(X), y, max_components=2, n_repeats=1, random_state=0, selection_rule="min"
        )
    a = result.n_components
    return result.cv_predictions.to_numpy().ravel(), result.press[a], result.r2y_validated.loc[a, "total"]


@pytest.mark.parametrize("tool", ["PLS.select_n_components", "MBPLS.select_n_components", "PLS.cross_validate"])
def test_a_row_with_nothing_observed_is_left_out_of_press_and_q2(tool: str) -> None:
    """Held out, such a row has no prediction, as ``predict`` gives it none (#1148), so Q2 is over the other rows.

    ``select_n_components`` raised RuntimeError; the other two left it out of PRESS but kept it in Q2's reference.
    """
    X, y = _data()
    X.iloc[5] = np.nan
    predicted, press, q2 = _held_out_q2(tool, X, y)
    others = np.delete(predicted, 5)
    observed = np.delete(y["y"].to_numpy(), 5)
    assert np.isnan(predicted[5])
    assert np.isfinite(others).all()
    by_hand = float(np.sum((observed - others) ** 2))
    assert press == pytest.approx(by_hand)
    assert q2 == pytest.approx(1 - by_hand / float(np.sum((observed - observed.mean()) ** 2)))


# --- A count that is not a whole number --------------------------------------------------------------

#: Each calls a function with the named count set to a value with a fractional part.
COUNTS: dict[str, Callable[[pd.DataFrame, pd.DataFrame, float], object]] = {
    "PLS.select_n_components(max_components)": lambda X, y, v: PLS.select_n_components(
        X, y, max_components=v, n_repeats=1
    ),
    "PLS.select_n_components(n_repeats)": lambda X, y, v: PLS.select_n_components(X, y, max_components=2, n_repeats=v),
    "PLS.select_n_components(n_permutations)": lambda X, y, v: PLS.select_n_components(
        X, y, max_components=2, n_repeats=1, selection_rule="randomization", n_permutations=v
    ),
    "PCA.select_n_components(max_components)": lambda X, _, v: PCA.select_n_components(X, max_components=v),
    "PCA.select_n_components(n_repeats)": lambda X, _, v: PCA.select_n_components(X, max_components=2, n_repeats=v),
    "MBPLS.select_n_components(max_components)": lambda X, y, v: MBPLS.select_n_components(
        _blocks(X), y, max_components=v, n_repeats=1
    ),
    "MBPLS.select_n_components(n_repeats)": lambda X, y, v: MBPLS.select_n_components(
        _blocks(X), y, max_components=2, n_repeats=v
    ),
    "compare_cv_criteria(max_components)": lambda X, y, v: compare_cv_criteria(X, y, max_components=v),
    "compare_cv_criteria(n_permutations)": lambda X, y, v: compare_cv_criteria(X, y, n_permutations=v),
    "compare_cv_criteria(n_cv_permutations)": lambda X, y, v: compare_cv_criteria(X, y, n_cv_permutations=v),
    "pseudo_validation_set(n_components)": lambda X, y, v: pseudo_validation_set(X, y, n_components=v),
    "PLS.nested_cv(max_components)": lambda X, y, v: PLS.nested_cv(X, y, max_components=v),
    "PLS.nested_cv(n_permutations)": lambda X, y, v: PLS.nested_cv(X, y, n_permutations=v),
    "PLSDA.permutation_test(n_permutations)": lambda X, y, v: (
        PLSDA(n_components=1).fit(X, _labels(y)).permutation_test(X, _labels(y), n_permutations=v)
    ),
    "randomization_test_mbpls(n_permutations)": lambda X, y, v: randomization_test_mbpls(
        MBPLS(n_components=1).fit(_blocks(X), y), _blocks(X), y, n_permutations=v
    ),
    "ASCA.permutation_test(n_permutations)": lambda X, _, v: (
        ASCA(n_components=1).fit(X, pd.DataFrame({"f": ["a", "b"] * (N_ROWS // 2)})).permutation_test(n_permutations=v)
    ),
    "ASCA.vasca(n_permutations)": lambda X, _, v: (
        ASCA(n_components=1).fit(X, pd.DataFrame({"f": ["a", "b"] * (N_ROWS // 2)})).vasca("f", n_permutations=v)
    ),
    "GPA.consensus_test(n_permutations)": lambda X, _, v: (
        GPA().fit({"one": X, "two": X + 0.1}).consensus_test(n_permutations=v)
    ),
    "impute_low_rank(n_components)": lambda X, _, v: impute_low_rank(X.mask(X > 1.5), n_components=v),
}


@pytest.mark.parametrize("call", list(COUNTS))
def test_a_count_that_is_not_whole_is_refused(call: str) -> None:
    """2.5 was read as 2 and 1.5 as 1, without a word, or failed in a slice with a TypeError (#1114)."""
    X, y = _data()
    name = call.rsplit("(", maxsplit=1)[-1].rstrip(")")
    with pytest.raises(ValueError, match=rf"^{name} must be a whole number; got 2\.5\.$"):
        COUNTS[call](X, y, 2.5)
