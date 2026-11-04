"""#983: every ``diagnose`` returns Hotelling's T² in one shape, ``cumulative_hotellings_t2``.

The field is the cumulative DataFrame of the fitted T² attribute: one row per diagnosed row, one
column per component, column ``a`` holding the T² over the first ``a`` components. The old field,
``hotellings_t2``, was that DataFrame from PCA and TPLS and its last column, a Series, from PLS,
MBPLS and MBPCA. It still returns what it did, with a ``DeprecationWarning`` when it is read.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import process_improve.batch as batch_package
import process_improve.multivariate.methods as multivariate_package
from process_improve.batch._batch_pca import BatchPCA
from process_improve.multivariate.methods import MBPCA, MBPLS, PCA, PLS, PLSDA, PRM, TPLS, DataFrameDict, MCUVScaler

RNG = np.random.default_rng(983)
LATENT = RNG.standard_normal((30, 2))
X = pd.DataFrame(LATENT @ RNG.standard_normal((2, 6)) + 0.1 * RNG.standard_normal((30, 6)), columns=list("abcdef"))
X.index = [f"row {i}" for i in range(30)]
Y = pd.DataFrame({"y": LATENT @ [1.0, -0.5] + 0.1 * RNG.standard_normal(30)}, index=X.index)
BLOCKS = {"A": X[["a", "b", "c"]], "B": X[["d", "e", "f"]]}
TPLS_DATA = Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate" / "tpls-pyphi"


def _line_of(marker: str) -> int:
    """Return the number of the line of this file that ends with ``marker``: the read a warning must point at."""
    lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    return next(number for number, line in enumerate(lines, start=1) if line.endswith(marker))


def _fitted(name: str) -> tuple[object, object, pd.DataFrame]:
    """Return a fitted ``name``, the rows it was fitted to as its ``diagnose`` takes them, and its fitted T²."""
    if name == "PCA":
        scaled = MCUVScaler().fit_transform(X)
        model = PCA(n_components=2).fit(scaled)
        return model, scaled, model.hotellings_t2_
    if name in {"PLS", "PRM"}:
        model = (PLS if name == "PLS" else PRM)(n_components=2).fit(X, Y)
        return model, X, model.hotellings_t2_
    if name == "PLSDA":
        model = PLSDA(n_components=2).fit(X, np.where(Y["y"] > 0, "high", "low"))
        return model, X, model.hotellings_t2_
    if name in {"MBPLS", "MBPCA"}:
        model = MBPLS(n_components=2).fit(BLOCKS, Y) if name == "MBPLS" else MBPCA(n_components=2).fit(BLOCKS)
        return model, BLOCKS, model.super_hotellings_t2_
    if name == "TPLS":

        def read(file: str) -> pd.DataFrame:
            return pd.read_csv(TPLS_DATA / file, index_col=0).astype("float64")

        groups = {f"Group {i}": i for i in range(1, 6)}
        z = {"Conditions": read("process_conditions.csv")}
        f = {group: read(f"formulas_Group{i}.csv") for group, i in groups.items()}
        d = {group: read(f"properties_Group{i}.csv") for group, i in groups.items()}
        y = {"Quality": read("quality_indicators.csv")}
        model = TPLS(n_components=2, d_matrix=d).fit(DataFrameDict({"Z": z, "F": f, "Y": y}))
        return model, DataFrameDict({"Z": z, "F": f}), model.hotellings_t2
    time = np.arange(6)
    batches = {
        f"b{i}": pd.DataFrame({"u": a * np.sin(time) + 0.1 * c, "r": b * np.cos(time) - 0.2 * c})
        for i, (a, b, c) in enumerate(RNG.standard_normal((15, 3)))
    }
    model = BatchPCA(n_components=2).fit(batches)
    return model, batches, model.hotellings_t2_


#: Every estimator with a ``diagnose``, and what its ``hotellings_t2`` held before #983.
OLD_SHAPE = {
    "PCA": "frame",
    "TPLS": "frame",
    "BatchPCA": "frame",
    "PLS": "series",
    "PLSDA": "series",
    "PRM": "series",
    "MBPLS": "series",
    "MBPCA": "series",
}


@pytest.fixture(scope="module", params=sorted(OLD_SHAPE))
def case(request: pytest.FixtureRequest) -> tuple:
    """Return the estimator's name, the model, its fitted rows and its fitted T²."""
    return (request.param, *_fitted(request.param))


def test_every_estimator_with_a_diagnose_is_checked_here() -> None:
    """A new estimator with a ``diagnose`` fails here until it is added, so its T² shape cannot drift."""
    found = {
        name
        for package in (multivariate_package, batch_package)
        for name, member in vars(package).items()
        if inspect.isclass(member) and callable(getattr(member, "diagnose", None))
    }
    assert found == set(OLD_SHAPE)


def test_cumulative_hotellings_t2_is_the_fitted_frame_for_the_fitted_rows(case: tuple) -> None:
    """One DataFrame per estimator, rows by components, with the fitted attribute's labels and values."""
    _, model, rows, fitted = case
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # reading the new field does not warn
        t2 = model.diagnose(rows).cumulative_hotellings_t2
    assert isinstance(t2, pd.DataFrame)
    assert t2.shape == fitted.shape
    assert list(t2.columns) == list(fitted.columns) == list(range(1, fitted.shape[1] + 1))
    assert list(t2.index) == list(fitted.index)
    np.testing.assert_allclose(t2.to_numpy(), fitted.to_numpy(), rtol=1e-7, atol=1e-10)


def test_the_old_field_warns_once_on_read_and_returns_what_it_did(case: tuple) -> None:
    """``hotellings_t2`` still holds its old shape; reading it, by name, warns once at the reader's line."""
    name, model, rows, _ = case
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.diagnose(rows)
        assert set(dict(result)) >= {"hotellings_t2", "cumulative_hotellings_t2"}  # reading the whole does not warn
        old = result.hotellings_t2  # the deprecated read
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    (warning,) = deprecations
    package = "batch" if name == "BatchPCA" else "multivariate"
    assert str(warning.message).startswith(
        f"process_improve.{package}.{name}.diagnose(...).hotellings_t2 is deprecated since 2.4.0 and will be removed "
        "in 3.0; use .cumulative_hotellings_t2 instead"
    )
    assert (warning.filename, warning.lineno) == (__file__, _line_of("# the deprecated read"))

    cumulative = result.cumulative_hotellings_t2
    if OLD_SHAPE[name] == "frame":
        pd.testing.assert_frame_equal(old, cumulative)
    else:
        assert isinstance(old, pd.Series)
        np.testing.assert_allclose(old.to_numpy(), cumulative.iloc[:, -1].to_numpy(), rtol=1e-12)
        assert list(old.index) == list(cumulative.index)
    with pytest.warns(DeprecationWarning, match="cumulative_hotellings_t2"):
        result["hotellings_t2"]
    with pytest.warns(DeprecationWarning, match="cumulative_hotellings_t2"):
        result.get("hotellings_t2")
