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
from scipy import sparse
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
    X = pd.DataFrame(
        T @ P + 0.05 * rng.standard_normal((n_samples, n_features)), columns=[f"x{i}" for i in range(n_features)]
    )
    Y = pd.DataFrame(T @ rng.standard_normal((n_factors, 1)) + 0.1 * rng.standard_normal((n_samples, 1)), columns=["y"])
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
    X = pd.DataFrame(
        {
            "temp": rng.standard_normal(n),
            "pressure": rng.standard_normal(n),
            "flow": rng.standard_normal(n),
            "batch_type": rng.choice(["A", "B", "C"], size=n),
        }
    )
    # Y depends on the numeric columns + a categorical-dependent offset.
    cat_offset = X["batch_type"].map({"A": 0.0, "B": 1.0, "C": -0.5}).to_numpy()
    Y = pd.DataFrame(
        0.5 * X["temp"] + 0.3 * X["pressure"] - 0.2 * X["flow"] + cat_offset + 0.05 * rng.standard_normal(n),
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


def test_column_transformer_sparse_output_is_rejected_with_the_remedy() -> None:
    """#399: the default OneHotEncoder makes the whole ColumnTransformer output sparse.

    This is the case the issue predicted and the test above dodges by passing
    `sparse_output=False`. `ColumnTransformer` flips its *entire* concatenated output to
    sparse once the result is more than `sparse_threshold` (default 0.3) zeros, which a
    one-hot block with a dozen levels easily is. NIPALS centres and scales every column,
    so there is no sparse path to take, and the message has to name the knob that avoids
    the round trip rather than sklearn's generic advice to call `.toarray()`.
    """
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from process_improve.multivariate.methods import PCA, PLS, MCUVScaler

    rng = np.random.default_rng(4)
    n = 60
    X = pd.DataFrame(
        {
            "temp": rng.standard_normal(n),
            "pressure": rng.standard_normal(n),
            "batch_type": rng.choice([f"L{i}" for i in range(12)], size=n),
        }
    )
    Y = pd.DataFrame(0.5 * X["temp"] + 0.3 * X["pressure"], columns=["y"])

    ct = make_column_transformer(
        (MCUVScaler(), ["temp", "pressure"]),
        (OneHotEncoder(), ["batch_type"]),  # sparse by default
    )
    assert sparse.issparse(ct.fit_transform(X)), "fixture no longer exercises the sparse path"

    expected = r"PLS does not accept sparse input.*sparse_threshold=0.*sparse_output=False"
    with pytest.raises(TypeError, match=expected):
        Pipeline([("ct", ct), ("pls", PLS(n_components=2))]).fit(X, Y)

    # The same door on the other two estimators a ColumnTransformer output can arrive at.
    dense_sparse = sparse.csr_matrix(rng.standard_normal((20, 4)))
    with pytest.raises(TypeError, match=r"PCA does not accept sparse input"):
        PCA(n_components=2).fit(dense_sparse)
    with pytest.raises(TypeError, match=r"MCUVScaler does not accept sparse input"):
        MCUVScaler().fit(dense_sparse)

    # And the remedy the message names actually works.
    ct_dense = make_column_transformer(
        (MCUVScaler(), ["temp", "pressure"]),
        (OneHotEncoder(), ["batch_type"]),
        sparse_threshold=0,
    )
    pipe = Pipeline([("ct", ct_dense), ("pls", PLS(n_components=2))]).fit(X, Y)
    assert np.isfinite(np.asarray(pipe.predict(X))).all()


def test_column_transformer_set_output_pandas_carries_names_into_pls() -> None:
    """#399: `set_output(transform="pandas")` puts the transformed names on the loadings.

    The issue expected `get_feature_names_out` to matter here, and this is the shape it
    takes in practice: with the default ndarray output PLS can only label its loadings
    0..K-1, because that is all it is given. Asking the ColumnTransformer for a DataFrame
    carries `get_feature_names_out` through to `x_loadings_.index`, so a loading is
    readable as "the one-hot column for batch_type == B" rather than "column 4".
    """
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from process_improve.multivariate.methods import PLS, MCUVScaler

    rng = np.random.default_rng(5)
    n = 60
    X = pd.DataFrame(
        {
            "temp": rng.standard_normal(n),
            "pressure": rng.standard_normal(n),
            "batch_type": rng.choice(["A", "B", "C"], size=n),
        }
    )
    Y = pd.DataFrame(0.5 * X["temp"] + 0.3 * X["pressure"], columns=["y"])

    def _fit(ct):
        return Pipeline([("ct", ct), ("pls", PLS(n_components=2))]).fit(X, Y).named_steps["pls"]

    steps = ((MCUVScaler(), ["temp", "pressure"]), (OneHotEncoder(sparse_output=False), ["batch_type"]))
    named = _fit(make_column_transformer(*steps).set_output(transform="pandas"))
    positional = _fit(make_column_transformer(*steps))

    expected = [
        "mcuvscaler__temp",
        "mcuvscaler__pressure",
        "onehotencoder__batch_type_A",
        "onehotencoder__batch_type_B",
        "onehotencoder__batch_type_C",
    ]
    assert list(named.feature_names_in_) == expected
    assert list(named.x_loadings_.index) == expected
    # Without set_output the numbers are identical, only the labels are lost.
    assert list(positional.x_loadings_.index) == list(range(len(expected)))
    np.testing.assert_allclose(named.x_loadings_.to_numpy(), positional.x_loadings_.to_numpy())


@pytest.mark.integration
@pytest.mark.slow
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


@pytest.mark.integration
@pytest.mark.slow
def test_halving_search_cv_with_a_pipeline_aware_budget() -> None:
    """#398: halving can spend a *pipeline parameter* as its resource, not just samples.

    The test above uses `resource="n_samples"`, the default, where the budget is rows and
    the estimator never sees it. The issue also asks for "a Pipeline-aware budget", which
    is the case that actually stresses what it worried about: sklearn sets the resource as
    a hyperparameter on every candidate, so `pls__n_components` is written into the PLS
    step through `set_params` at each rung and must survive `clone`. Here the search
    starts every candidate at one component and doubles, while separately grid-searching
    `pls__scale`, so both mechanisms are exercised at once.
    """
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
    from sklearn.pipeline import Pipeline

    from process_improve.multivariate.methods import PLS, MCUVScaler

    X, Y = _synthetic_xy(n_samples=180, n_features=8, n_factors=3, seed=3)
    pipe = Pipeline([("sc", MCUVScaler()), ("pls", PLS(n_components=2))])
    budget = {"resource": "pls__n_components", "max_resources": 4, "min_resources": 1}

    grid = HalvingGridSearchCV(pipe, {"pls__scale": [True, False]}, cv=3, factor=2, random_state=0, **budget)
    grid.fit(X, Y.values.ravel())
    # The resource really was spent on components: the rungs double from min_resources,
    # and the winning configuration carries the component count the last rung reached.
    assert list(grid.n_resources_) == [1, 2]
    assert grid.best_params_["pls__n_components"] == 2
    assert grid.best_score_ > 0.3  # synthetic data with real latent factors

    rng_search = HalvingRandomSearchCV(
        pipe, {"pls__scale": [True, False]}, n_candidates=2, cv=3, factor=2, random_state=0, **budget
    )
    rng_search.fit(X, Y.values.ravel())
    assert list(rng_search.n_resources_) == [1, 2]
    assert rng_search.best_params_["pls__n_components"] == 2

    # The template's constructor parameter is untouched by the search: `clone` gave each candidate
    # its own estimator, so the template still holds what it was built with (#505).
    assert pipe.named_steps["pls"].n_components == 2


# ---------------------------------------------------------------------------
# #505: fit() must not mutate constructor parameters (clone contract)
# ---------------------------------------------------------------------------


def _small_xy(n_samples: int = 12, n_features: int = 4, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)))
    beta = rng.normal(size=(n_features, 1))
    Y = pd.DataFrame(X.values @ beta + 0.05 * rng.normal(size=(n_samples, 1)))
    return X, Y


def test_pca_fit_leaves_constructor_params_untouched() -> None:
    """After fit(), get_params() returns exactly what was passed to __init__ (#505)."""
    X, _ = _small_xy()
    model = PCA(n_components=None).fit(X)
    assert model.get_params()["n_components"] is None
    assert model.n_components is None
    assert model.n_components_ == min(X.shape)

    with pytest.warns(UserWarning, match="requested number of components"):
        clamped = PCA(n_components=99).fit(X)
    assert clamped.get_params()["n_components"] == 99
    assert clamped.n_components_ == min(X.shape)


def test_pls_fit_leaves_constructor_params_untouched() -> None:
    """PLS keeps n_components and missing_data_settings as the user set them (#505)."""
    X, Y = _small_xy()
    X_missing = X.copy()
    X_missing.iloc[0, 0] = np.nan
    requested = {"md_max_iter": 250}
    model = PLS(n_components=2, missing_data_settings=dict(requested)).fit(X_missing, Y)
    assert model.get_params()["n_components"] == 2
    assert model.get_params()["missing_data_settings"] == requested
    assert model.missing_data_settings == requested
    assert model.n_components_ == 2

    with pytest.warns(UserWarning, match="requested number of components"):
        clamped = PLS(n_components=99).fit(X, Y)
    assert clamped.get_params()["n_components"] == 99
    assert clamped.n_components_ == min(X.shape)


def test_clone_reproduces_requested_configuration_after_fit() -> None:
    """clone() of a fitted estimator carries the request, not the resolved value (#505)."""
    X, Y = _small_xy()
    with pytest.warns(UserWarning, match="requested number of components"):
        parent = PLS(n_components=99).fit(X, Y)
    cloned = clone(parent)
    assert cloned.get_params()["n_components"] == 99

    pca_parent = PCA(n_components=None).fit(X)
    assert clone(pca_parent).get_params()["n_components"] is None


def test_refit_resolves_component_count_per_dataset() -> None:
    """Fitting the same instance twice on different shapes re-derives the clamp (#505)."""
    rng = np.random.default_rng(1)
    model = PCA(n_components=None)
    model.fit(pd.DataFrame(rng.normal(size=(10, 3))))
    assert model.n_components_ == 3
    model.fit(pd.DataFrame(rng.normal(size=(10, 6))))
    assert model.n_components_ == 6
    assert model.n_components is None


def test_cross_validate_submodels_use_requested_n_components() -> None:
    """cross_validate() resamples fit the user's requested configuration (#505)."""
    X, Y = _small_xy(n_samples=20)
    model = PLS(n_components=2).fit(X, Y)
    seen: list[int | None] = []
    original_fit = PLS.fit

    def spy_fit(self: PLS, *args: object, **kwargs: object) -> PLS:
        seen.append(self.get_params()["n_components"])
        return original_fit(self, *args, **kwargs)

    PLS.fit = spy_fit  # type: ignore[method-assign]
    try:
        model.cross_validate(X, Y, cv=3, random_state=0, show_progress=False)
    finally:
        PLS.fit = original_fit  # type: ignore[method-assign]
    assert seen, "cross_validate() fitted no sub-models"
    assert all(value == 2 for value in seen)
