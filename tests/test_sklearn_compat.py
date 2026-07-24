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

from process_improve.multivariate.methods import MBPCA, MBPLS, OPLS, PCA, PLS, TPLS, MCUVScaler

# Concrete sklearn estimators we explicitly refuse to inherit from (their private,
# version-specific attribute layout is exactly what ENG-07 decouples us from).
_CONCRETE_SKLEARN_ESTIMATORS = (PLSRegression, PLSCanonical, PLSSVD, SKLearnPCA)


@pytest.mark.parametrize("estimator_cls", [PCA, PLS, TPLS, MBPLS, MBPCA, OPLS])
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
    assert RegressorMixin in OPLS.__mro__
    assert TransformerMixin in OPLS.__mro__


def test_opls_clone_and_params_round_trip() -> None:
    """OPLS supports the sklearn estimator API: clone, get_params, set_params."""
    est = OPLS(n_orthogonal_components=2, scale=False)
    assert clone(est).get_params() == est.get_params()
    rebuilt = OPLS(n_orthogonal_components=1).set_params(n_orthogonal_components=2, scale=False)
    assert rebuilt.get_params() == est.get_params()


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


# ---------------------------------------------------------------------------
# sklearn-interop verification, audit set
# (#397 TransformedTargetRegressor, #399 make_column_transformer,
#  #398 HalvingGridSearchCV / HalvingRandomSearchCV).
# These are "drop into a more demanding sklearn composition and check that
# it works end-to-end" tests: each issue posited that something would
# probably surface; if nothing does, the test locks in the working state.
# ---------------------------------------------------------------------------


def _synthetic_xy(n_samples: int = 60, n_features: int = 6, n_factors: int = 3, seed: int = 0):
    """Latent-factor X and a Y that's a noisy linear combination of those factors.

    Used by the interop tests below; large enough that PLS recovers the
    structure and small enough that GridSearchCV / HalvingGridSearchCV
    finish quickly.
    """
    from process_improve.multivariate.methods import MCUVScaler

    rng = np.random.default_rng(seed)
    T = rng.standard_normal((n_samples, n_factors))
    P = rng.standard_normal((n_factors, n_features))
    X = pd.DataFrame(T @ P + 0.05 * rng.standard_normal((n_samples, n_features)),
                     columns=[f"x{i}" for i in range(n_features)])
    Y = pd.DataFrame(T @ rng.standard_normal((n_factors, 1)) + 0.1 * rng.standard_normal((n_samples, 1)),
                     columns=["y"])
    assert MCUVScaler  # used elsewhere in this module; silence linter
    return X, Y


def test_transformed_target_regressor_with_mcuvscaler_and_pls() -> None:
    """#397: TransformedTargetRegressor over Pipeline([MCUVScaler, PLS]) round-trips Y scale.

    The outer ``transformer=MCUVScaler()`` scales Y for the inner regressor at fit
    time and inverse-transforms the prediction back to the original Y scale.
    """
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline

    from process_improve.multivariate.methods import PLS, MCUVScaler

    X, Y = _synthetic_xy(seed=1)
    # TransformedTargetRegressor fits the *target* transformer on Y, then
    # forwards scaled Y to the inner regressor; predictions come back on
    # the original Y scale via inverse_transform.
    pipe = Pipeline([("sc", MCUVScaler()), ("pls", PLS(n_components=2))])
    model = TransformedTargetRegressor(regressor=pipe, transformer=MCUVScaler())
    model.fit(X, Y.values.ravel())  # sklearn's check_X_y rejects DataFrame y here
    y_pred = model.predict(X)
    assert y_pred.shape == (len(X),) or y_pred.shape == (len(X), 1)

    # Sanity: predictions live on the original Y scale, not the scaled space.
    # A trivial sanity check: the prediction mean should be close to Y mean,
    # not to zero (which is what a scaled prediction would centre on).
    y_pred = np.asarray(y_pred).ravel()
    assert abs(y_pred.mean() - Y.values.ravel().mean()) < 0.5
    # And the magnitude should match Y's scale, not the unit-variance scale.
    assert 0.3 * Y.values.std() < y_pred.std() < 3.0 * Y.values.std()


def test_make_column_transformer_with_mcuvscaler_and_pls() -> None:
    """#399: make_column_transformer((MCUVScaler, numeric_cols), (OneHotEncoder, cat_cols)) → PLS fits and predicts."""
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from process_improve.multivariate.methods import PLS, MCUVScaler

    rng = np.random.default_rng(2)
    n = 60
    X = pd.DataFrame({
        "temp": rng.standard_normal(n),
        "pressure": rng.standard_normal(n),
        "flow": rng.standard_normal(n),
        "batch_type": rng.choice(["A", "B", "C"], size=n),
    })
    # Y depends on the numeric columns + a categorical-dependent offset.
    cat_offset = X["batch_type"].map({"A": 0.0, "B": 1.0, "C": -0.5}).to_numpy()
    Y = pd.DataFrame(
        0.5 * X["temp"] + 0.3 * X["pressure"] - 0.2 * X["flow"] + cat_offset
        + 0.05 * rng.standard_normal(n),
        columns=["y"],
    )

    ct = make_column_transformer(
        (MCUVScaler(), ["temp", "pressure", "flow"]),
        (OneHotEncoder(sparse_output=False), ["batch_type"]),
        remainder="drop",
    )
    pipe = Pipeline([("ct", ct), ("pls", PLS(n_components=2))])
    pipe.fit(X, Y.values.ravel())
    y_pred = pipe.predict(X)
    assert np.asarray(y_pred).shape[0] == n

    # ColumnTransformer + downstream PLS preserve feature-name introspection
    # via get_feature_names_out (added in #391/#405).
    feature_names = ct.get_feature_names_out()
    # Three scaled numeric columns + 3 one-hot columns for batch_type:
    assert len(feature_names) == 6


def test_halving_grid_search_cv_with_mcuvscaler_and_pls() -> None:
    """#398: HalvingGridSearchCV over Pipeline([MCUVScaler, PLS]) finishes and picks a config.

    Successive halving stresses estimator + scheduler interop in a way plain
    GridSearchCV does not (resource dispatch, repeated re-fits on shrinking subsets).
    """
    # HalvingGridSearchCV / HalvingRandomSearchCV are still experimental in sklearn.
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
    from sklearn.pipeline import Pipeline

    from process_improve.multivariate.methods import PLS, MCUVScaler

    X, Y = _synthetic_xy(n_samples=180, n_features=8, n_factors=3, seed=3)
    pipe = Pipeline([("sc", MCUVScaler()), ("pls", PLS(n_components=2))])

    # Keep the grid narrow so the smallest halving resource still has
    # enough samples to fit ``n_components`` cleanly; otherwise sklearn
    # warns about non-finite scores for the extreme combinations. The
    # interop check is whether the search completes and picks a best
    # configuration, not which exact value wins.
    grid = HalvingGridSearchCV(
        pipe,
        {"pls__n_components": [1, 2, 3]},
        cv=3,
        factor=2,
        resource="n_samples",
        random_state=0,
    )
    grid.fit(X, Y.values.ravel())
    # Halving search runs to completion and picks a positive component count.
    assert grid.best_params_["pls__n_components"] >= 1
    assert grid.best_score_ > 0.3  # synthetic data with real latent factors

    # Random analogue too: confirms the search dispatches resource budgets correctly.
    rng_search = HalvingRandomSearchCV(
        pipe,
        {"pls__n_components": [1, 2, 3]},
        n_candidates=3,
        cv=3,
        factor=2,
        resource="n_samples",
        random_state=0,
    )
    rng_search.fit(X, Y.values.ravel())
    assert rng_search.best_params_["pls__n_components"] >= 1
