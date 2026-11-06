"""Feature-name consistency between fit and transform/predict (issue #195).

Before #195, PCA / PLS only checked the *number* of columns passed to
``transform`` / ``predict``. A correctly-shaped frame whose columns were
reordered or renamed was projected positionally (PCA) or silently label-aligned
to all-``NaN`` (PLS ``X @ direct_weights_``), producing wrong results with no
error. These tests pin the corrected behaviour:

* reordered columns are realigned to the training order (same result);
* renamed / wrong columns raise ``ValueError``;
* unnamed (ndarray) input is taken positionally and still works - in particular
  PLS no longer returns all-``NaN``.

A fitted model keeps its own copy of the caller's row and column labels: renaming the
caller's axes after the fit does not rename the model's frames.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate._base import _LazyFrame
from process_improve.multivariate.methods import MBPCA, MBPLS, OPLS, PCA, PLS, TPLS, DataFrameDict, MCUVScaler


@pytest.fixture
def xy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a small, well-conditioned scaled (X, Y) pair with named columns."""
    rng = np.random.default_rng(42)
    n = 30
    X = pd.DataFrame(
        rng.standard_normal((n, 4)),
        columns=["temp", "pressure", "flow", "ph"],
    )
    beta = np.array([[1.5], [-2.0], [0.5], [0.0]])
    Y = pd.DataFrame(X.values @ beta + 0.1 * rng.standard_normal((n, 1)), columns=["yield"])
    X_scaled = MCUVScaler().fit_transform(X)
    Y_scaled = MCUVScaler().fit_transform(Y)
    return X_scaled, Y_scaled


class TestPCAFeatureNames:
    def test_reordered_columns_are_realigned(self, xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, _ = xy
        pca = PCA(n_components=2).fit(X)
        reordered = X[X.columns[::-1]]
        base = pca.transform(X)
        got = pca.transform(reordered)
        # Realigned to training order -> identical scores despite the input order.
        np.testing.assert_allclose(got.to_numpy(), base.to_numpy(), rtol=1e-10, atol=1e-10)

    def test_renamed_column_raises(self, xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, _ = xy
        pca = PCA(n_components=2).fit(X)
        bad = X.rename(columns={"flow": "FLOW_TYPO"})
        with pytest.raises(ValueError, match=r"Feature names .* do not match"):
            pca.transform(bad)
        with pytest.raises(ValueError, match=r"Feature names .* do not match"):
            pca.diagnose(bad)

    def test_ndarray_input_still_works(self, xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, _ = xy
        pca = PCA(n_components=2).fit(X)
        base = pca.transform(X)
        got = pca.transform(X.to_numpy())
        np.testing.assert_allclose(got.to_numpy(), base.to_numpy(), rtol=1e-10, atol=1e-10)

    def test_wrong_column_count_raises(self, xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, _ = xy
        pca = PCA(n_components=2).fit(X)
        with pytest.raises(ValueError, match="columns"):
            pca.transform(X.iloc[:, :3])


class TestPLSFeatureNames:
    def test_reordered_columns_are_realigned(self, xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, Y = xy
        pls = PLS(n_components=2).fit(X, Y)
        reordered = X[X.columns[::-1]]
        base = pls.diagnose(X)
        got = pls.diagnose(reordered)
        np.testing.assert_allclose(got.scores.to_numpy(), base.scores.to_numpy(), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(got.y_hat.to_numpy(), base.y_hat.to_numpy(), rtol=1e-10, atol=1e-10)

    def test_renamed_column_raises(self, xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, Y = xy
        pls = PLS(n_components=2).fit(X, Y)
        bad = X.rename(columns={"temp": "temperature"})
        with pytest.raises(ValueError, match=r"Feature names .* do not match"):
            pls.predict(bad)
        with pytest.raises(ValueError, match=r"Feature names .* do not match"):
            pls.transform(bad)

    def test_ndarray_transform_is_not_all_nan(self, xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        # Regression: PLS.transform on an ndarray used to label-align integer
        # columns against the named direct_weights_ index and return all-NaN.
        X, Y = xy
        pls = PLS(n_components=2).fit(X, Y)
        base = pls.transform(X)
        got = pls.transform(X.to_numpy())
        assert not np.isnan(got.to_numpy()).any()
        np.testing.assert_allclose(got.to_numpy(), base.to_numpy(), rtol=1e-10, atol=1e-10)

    def test_ndarray_predict_works(self, xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, Y = xy
        pls = PLS(n_components=2).fit(X, Y)
        # predict() returns a y_hat DataFrame directly.
        base = pls.predict(X)
        got = pls.predict(X.to_numpy())
        assert not np.isnan(got.to_numpy()).any()
        np.testing.assert_allclose(got.to_numpy(), base.to_numpy(), rtol=1e-10, atol=1e-10)

    def test_transform_wrong_column_count_raises(self, xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        # PLS.transform previously had no count check at all.
        X, Y = xy
        pls = PLS(n_components=2).fit(X, Y)
        with pytest.raises(ValueError, match="columns"):
            pls.transform(X.iloc[:, :3])


def _named(rows: int, columns: list[str], rng: np.random.Generator, index: str, axis: str) -> pd.DataFrame:
    """Return a frame of random numbers whose row and column axes carry a name."""
    frame = pd.DataFrame(rng.normal(size=(rows, len(columns))), columns=columns, index=[f"r{i}" for i in range(rows)])
    return frame.rename_axis(index=index, columns=axis)


def _fit_pca(rng: np.random.Generator) -> tuple[object, list[pd.DataFrame]]:
    x = _named(20, list("abcd"), rng, "run", "variable")
    return PCA(n_components=2).fit(x), [x]


def _fit_pls(rng: np.random.Generator) -> tuple[object, list[pd.DataFrame]]:
    x, y = _named(20, list("abcd"), rng, "run", "variable"), _named(20, ["y1", "y2"], rng, "run", "response")
    return PLS(n_components=2).fit(x, y), [x, y]


def _fit_opls(rng: np.random.Generator) -> tuple[object, list[pd.DataFrame]]:
    x, y = _named(20, list("abcd"), rng, "run", "variable"), _named(20, ["y"], rng, "run", "response")
    return OPLS(n_orthogonal_components=1).fit(x, y), [x, y]


def _fit_mbpls(rng: np.random.Generator) -> tuple[object, list[pd.DataFrame]]:
    blocks = {"a": _named(20, ["a1", "a2"], rng, "run", "variable"), "b": _named(20, ["b1", "b2"], rng, "run", "var")}
    y = _named(20, ["y1", "y2"], rng, "run", "response")
    return MBPLS(n_components=2).fit(blocks, y), [*blocks.values(), y]


def _fit_mbpca(rng: np.random.Generator) -> tuple[object, list[pd.DataFrame]]:
    blocks = {"a": _named(20, ["a1", "a2"], rng, "run", "variable"), "b": _named(20, ["b1", "b2"], rng, "run", "var")}
    return MBPCA(n_components=2).fit(blocks), list(blocks.values())


def _fit_tpls(rng: np.random.Generator) -> tuple[object, list[pd.DataFrame]]:
    d_matrix = {"g": _named(5, ["p1", "p2", "p3"], rng, "material", "property")}
    f = pd.DataFrame(rng.dirichlet(np.ones(5), size=20), columns=d_matrix["g"].index.copy())
    f.index = pd.Index([f"r{i}" for i in range(20)], name="blend")
    blocks = {
        "F": {"g": f},
        "Z": {"Z": _named(20, ["z1", "z2"], rng, "blend", "condition")},
        "Y": {"Y": _named(20, ["y1", "y2"], rng, "blend", "quality")},
    }
    callers = [f, blocks["Z"]["Z"], blocks["Y"]["Y"], d_matrix["g"]]
    return TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(blocks)), callers


def _names_of(value: object) -> list | None:
    """Return the axis names of an index, series or frame, and None for anything else."""
    if isinstance(value, pd.Index):
        return list(value.names)
    if isinstance(value, pd.Series):
        return list(value.index.names)
    if isinstance(value, pd.DataFrame):
        return [*value.index.names, *value.columns.names]
    return None


def _collect(value: object, where: str, found: dict[str, list]) -> None:
    """Record the axis names of ``value``, or of what it holds: in dicts, lists and the package's own objects."""
    names = _names_of(value)
    if names is not None:
        found[where] = names
        return
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list | tuple):
        items = list(enumerate(value))
    elif type(value).__module__.startswith("process_improve") and hasattr(value, "__dict__"):
        items = list(vars(value).items())  # the scalers a model keeps
    else:
        return
    for key, item in items:
        _collect(item, f"{where}[{key!r}]", found)


def _axis_names(model: object) -> dict[str, list]:
    """Return the axis names of every frame, series and index a fitted model holds, by where it holds them.

    The lazily built frames are built. The constructor's parameters are left out: scikit-learn
    keeps them as given, so TPLS's ``d_matrix`` is the caller's own dict.
    """
    found: dict[str, list] = {}
    parameters = set(model.get_params())  # type: ignore[attr-defined]
    for key, value in vars(model).items():
        if key not in parameters and key != "_frame_cache":
            _collect(value, key, found)
    for owner in type(model).__mro__:
        for key, value in vars(owner).items():
            if isinstance(value, _LazyFrame):
                _collect(getattr(model, key), key, found)
    return found


@pytest.mark.parametrize("fit", [_fit_pca, _fit_pls, _fit_opls, _fit_mbpls, _fit_mbpca, _fit_tpls])
def test_renaming_the_callers_axes_after_the_fit_leaves_the_model_as_it_was(
    fit: Callable[[np.random.Generator], tuple[object, list[pd.DataFrame]]],
) -> None:
    """The model copies the labels of the caller's rows and columns: they are not the same objects.

    pandas passes an index on by reference, so a frame built on the caller's index holds that
    very object. The model's frames then took any name given to the caller's axes after the fit:
    a score plot's axis title changed with no new fit.
    """
    model, callers = fit(np.random.default_rng(3))
    before = _axis_names(model)
    assert before
    for frame in callers:
        frame.index.name = "renamed"
        frame.columns.name = "renamed"
    assert _axis_names(model) == before
