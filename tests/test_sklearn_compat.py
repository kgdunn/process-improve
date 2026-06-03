"""ENG-07 (#289): sklearn API compatibility without inheriting a concrete sklearn estimator.

The multivariate estimators keep the lightweight sklearn mixins (``BaseEstimator``,
``TransformerMixin``, ``RegressorMixin``) for API compatibility - ``get_params`` /
``set_params`` / ``clone`` / Pipeline support - but must NOT inherit a concrete sklearn
estimator (such as ``sklearn.cross_decomposition.PLSRegression``) whose private attribute
layout would couple the package to a specific sklearn version and break on a major bump.

These tests:
  * lock in the decoupling (no concrete sklearn estimator in any MRO; mixins retained);
  * validate the sklearn estimator API (clone / get_params / set_params);
  * cross-check numerical consistency against a variety of sklearn multivariate models
    (``sklearn.decomposition.PCA`` and ``sklearn.cross_decomposition.PLSRegression``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSSVD, PLSCanonical, PLSRegression
from sklearn.decomposition import PCA as SKLearnPCA  # noqa: N811 - aliased to avoid collision with our PCA
from sklearn.preprocessing import StandardScaler

from process_improve.multivariate.methods import MBPCA, MBPLS, PCA, PLS, TPLS, MCUVScaler

# Concrete sklearn estimators we explicitly refuse to inherit from (their private,
# version-specific attribute layout is exactly what ENG-07 decouples us from).
_CONCRETE_SKLEARN_ESTIMATORS = (PLSRegression, PLSCanonical, PLSSVD, SKLearnPCA)


@pytest.mark.parametrize("estimator_cls", [PCA, PLS, TPLS, MBPLS, MBPCA])
def test_estimators_do_not_inherit_concrete_sklearn(estimator_cls: type) -> None:
    """No estimator inherits a concrete sklearn estimator, but all keep BaseEstimator."""
    mro = estimator_cls.__mro__
    for concrete in _CONCRETE_SKLEARN_ESTIMATORS:
        assert concrete not in mro, f"{estimator_cls.__name__} must not inherit {concrete.__name__}"
    assert BaseEstimator in mro, f"{estimator_cls.__name__} should keep the sklearn BaseEstimator mixin"


def test_estimators_keep_expected_sklearn_mixins() -> None:
    """The mixins that provide the documented sklearn API are retained."""
    assert TransformerMixin in PCA.__mro__
    assert RegressorMixin in PLS.__mro__
    assert TransformerMixin in PLS.__mro__
    assert RegressorMixin in MBPLS.__mro__
    assert TransformerMixin in MBPCA.__mro__


def test_pca_clone_and_params_round_trip() -> None:
    """PCA supports the sklearn estimator API: clone, get_params, set_params."""
    est = PCA(n_components=3, algorithm="svd")
    assert clone(est).get_params() == est.get_params()
    rebuilt = PCA(n_components=1).set_params(n_components=3, algorithm="svd")
    assert rebuilt.get_params() == est.get_params()


def test_pls_clone_and_params_round_trip() -> None:
    """PLS supports the sklearn estimator API: clone, get_params, set_params."""
    est = PLS(n_components=2, scale=False)
    assert clone(est).get_params() == est.get_params()


def test_pca_matches_sklearn_pca() -> None:
    """Our PCA (SVD path) matches sklearn.decomposition.PCA on the same scaled data."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(40, 6)), columns=[f"x{i}" for i in range(6)])
    x_scaled = MCUVScaler().fit_transform(X)

    ours = PCA(n_components=3, algorithm="svd").fit(x_scaled)
    ref = SKLearnPCA(n_components=3, svd_solver="full").fit(x_scaled.values)

    # Scores and loadings agree up to a per-component sign flip.
    assert np.abs(ours.scores_.values) == pytest.approx(np.abs(ref.transform(x_scaled.values)), abs=1e-6)
    assert np.abs(ours.loadings_.values) == pytest.approx(np.abs(ref.components_.T), abs=1e-6)
    assert np.asarray(ours.explained_variance_) == pytest.approx(ref.explained_variance_, abs=1e-6)
    assert float(ours.r2_cumulative_.iloc[-1]) == pytest.approx(
        float(ref.explained_variance_ratio_[:3].sum()), abs=1e-6
    )


def test_pls_matches_plsregression() -> None:
    """Our PLS matches sklearn.cross_decomposition.PLSRegression (single response)."""
    rng = np.random.default_rng(42)
    n_samples, n_features, n_components = 50, 6, 3
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=[f"x{i}" for i in range(n_features)])
    beta = rng.normal(size=(n_features, 1))
    Y = pd.DataFrame(X.values @ beta + rng.normal(scale=0.2, size=(n_samples, 1)), columns=["y"])

    # Both fed the identically scaled X; centre Y up front (PLSRegression centres Y internally).
    x_scaled = pd.DataFrame(StandardScaler().fit_transform(X.values), columns=X.columns)
    y_scaled = pd.DataFrame(StandardScaler().fit_transform(Y.values), columns=["y"])

    ours = PLS(n_components=n_components, scale=False).fit(x_scaled, y_scaled)
    ref = PLSRegression(n_components=n_components, scale=False).fit(x_scaled.values, y_scaled.values)

    assert np.abs(ours.scores_.values) == pytest.approx(np.abs(ref.x_scores_), abs=1e-6)
    assert np.abs(ours.x_loadings_.values) == pytest.approx(np.abs(ref.x_loadings_), abs=1e-6)
    assert np.abs(ours.x_weights_.values) == pytest.approx(np.abs(ref.x_weights_), abs=1e-6)
    assert np.abs(ours.beta_coefficients_.values) == pytest.approx(np.abs(ref.coef_.T), abs=1e-6)
