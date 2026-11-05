"""Input a method cannot use is refused with a message naming the argument, its value and what is allowed (#1077).

Each of these used to fail deep inside numpy, pandas or Python, with a message about array shapes
or ``zip`` that said nothing about which argument was wrong.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import MBPCA, MBPLS, PLS, PLSDA, TPLS, DataFrameDict


@pytest.fixture
def xy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thirty rows, four features, one response."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
    y = pd.DataFrame({"y": X.to_numpy() @ [1.0, -0.5, 0.3, 0.0] + rng.normal(0, 0.1, 30)})
    return X, y


def test_prediction_interval_refuses_a_bootstrap_result(xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A bootstrap holds no rows out, so its result has no cross-validated RMSE to build the interval on."""
    X, y = xy
    model = PLS(n_components=2).fit(X, y)
    bootstrap = model.cross_validate(X, y, n_bootstrap=20, random_state=0, show_progress=False)
    message = (
        r"^cv_result is a 'bootstrap' result, which has no rmse_cv: a bootstrap holds no rows out\. Pass the "
        r"result of cross_validate with n_bootstrap=0, which runs the jackknife \(cv='loo'\) or K-fold \(cv=K\)\.$"
    )
    with pytest.raises(ValueError, match=message):
        model.prediction_interval(X.iloc[:3], cv_result=bootstrap)


def test_prediction_interval_still_takes_a_jackknife_result(xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The check refuses only a result without ``rmse_cv``: the jackknife's interval uses it, as before."""
    X, y = xy
    model = PLS(n_components=2).fit(X, y)
    jackknife = model.cross_validate(X, y, show_progress=False)
    interval = model.prediction_interval(X.iloc[:3], cv_result=jackknife)
    assert (interval.upper.to_numpy() > interval.lower.to_numpy()).all()


@pytest.mark.parametrize("n_rows", [0, 2])
def test_invert_refuses_other_than_one_row(xy: tuple[pd.DataFrame, pd.DataFrame], n_rows: int) -> None:
    """``invert`` finds the inputs for one target; two rows broadcast against the features and failed in numpy."""
    X, y = xy
    model = PLS(n_components=2).fit(X, y)
    message = (
        rf"^y_desired has {n_rows} rows; invert finds the inputs for one target at a time: pass one row, or "
        r"call invert once per row\.$"
    )
    with pytest.raises(ValueError, match=message):
        model.invert(pd.DataFrame({"y": [0.5, 1.0][:n_rows]}))


def test_invert_names_only_the_missing_targets(xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The message named every target, the ones given as well as the one missing (#1144 item 11)."""
    X, y = xy
    Y = y.assign(z=y["y"] * 2 + np.random.default_rng(1).normal(0, 0.1, len(y)))
    model = PLS(n_components=2).fit(X, Y)
    with pytest.raises(ValueError, match=r"^y_desired is missing value\(s\) for target\(s\): \['z'\]\.$"):
        model.invert({"y": 0.5})


@pytest.mark.parametrize("method", ["confusion", "score", "roc_auc"])
def test_plsda_refuses_labels_of_another_length(xy: tuple[pd.DataFrame, pd.DataFrame], method: str) -> None:
    """X and its labels are paired row by row, so a different count is refused with both counts, as in ``fit``."""
    X, y = xy
    labels = pd.Series(np.where(y["y"] > 0, "hi", "lo"))
    model = PLSDA(n_components=2).fit(X, labels)
    with pytest.raises(ValueError, match=r"^X has 30 rows but y has 20 labels\.$"):
        getattr(model, method)(X, labels.iloc[:20])


def _tpls_blocks(empty: str) -> tuple[dict[str, pd.DataFrame], DataFrameDict]:
    """TPLS's ``d_matrix`` and blocks, with the named group's block left without columns."""
    rng = np.random.default_rng(0)
    blends = [f"blend{i}" for i in range(20)]
    materials = [f"mat{j}" for j in range(4)]
    d_matrix = {"G": pd.DataFrame(rng.normal(size=(4, 3)), index=materials, columns=list("pqr"))}
    blocks = {
        "F": {"G": pd.DataFrame(rng.random((20, 4)), index=blends, columns=materials)},
        "Z": {"Z1": pd.DataFrame(rng.normal(size=(20, 2)), index=blends, columns=["z1", "z2"])},
        "Y": {"Y1": pd.DataFrame({"y": rng.normal(size=20)}, index=blends)},
    }
    if empty == 'd_matrix["G"]':
        d_matrix["G"] = d_matrix["G"][[]]
    else:
        block, group = empty.split('["')
        blocks[block][group.rstrip('"]')] = pd.DataFrame(index=blends)
    return d_matrix, DataFrameDict(blocks)


@pytest.mark.parametrize(
    ("estimator", "label"),
    [
        ("MBPCA", 'X["b2"]'),
        ("MBPLS", 'X["b2"]'),
        ("MBPLS", "y"),
        ("TPLS", 'Z["Z1"]'),
        ("TPLS", 'Y["Y1"]'),
        ("TPLS", 'd_matrix["G"]'),
    ],
)
def test_a_block_with_no_columns_is_refused_by_name(
    xy: tuple[pd.DataFrame, pd.DataFrame], estimator: str, label: str
) -> None:
    """numpy's "at least one array or dtype is required" named neither the block nor the cause."""
    X, y = xy
    blocks = {"b1": X[["a", "b"]], "b2": X[[]] if label == 'X["b2"]' else X[["c", "d"]]}
    fits = {
        "MBPCA": lambda: MBPCA(n_components=2).fit(blocks),
        "MBPLS": lambda: MBPLS(n_components=2).fit(blocks, y[[]] if label == "y" else y),
        "TPLS": lambda: TPLS(n_components=2, d_matrix=_tpls_blocks(label)[0]).fit(_tpls_blocks(label)[1]),
    }
    with pytest.raises(ValueError, match=rf"^Block {re.escape(label)} has no columns; a block needs at least one\.$"):
        fits[estimator]()
