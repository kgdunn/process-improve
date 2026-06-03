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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import PCA, PLS, MCUVScaler


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
            pca.predict(bad)

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
        base = pls.predict(X)
        got = pls.predict(reordered)
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
        base = pls.predict(X)
        got = pls.predict(X.to_numpy())
        assert not np.isnan(got.y_hat.to_numpy()).any()
        np.testing.assert_allclose(got.y_hat.to_numpy(), base.y_hat.to_numpy(), rtol=1e-10, atol=1e-10)

    def test_transform_wrong_column_count_raises(self, xy: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        # PLS.transform previously had no count check at all.
        X, Y = xy
        pls = PLS(n_components=2).fit(X, Y)
        with pytest.raises(ValueError, match="columns"):
            pls.transform(X.iloc[:, :3])
