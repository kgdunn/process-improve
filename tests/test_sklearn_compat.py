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

import importlib
import typing
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSSVD, PLSCanonical, PLSRegression
from sklearn.decomposition import PCA as SKLearnPCA  # noqa: N811 - aliased to avoid collision with our PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils._param_validation import InvalidParameterError, validate_parameter_constraints

from process_improve.batch import (
    BatchFeatureExtractor,
    BatchMonitor,
    BatchPCA,
    BatchPLS,
    DTWAligner,
    ResampleAligner,
)
from process_improve.multivariate.methods import (
    ASCA,
    CA,
    FAMD,
    FPCA,
    GPA,
    MBPCA,
    MBPLS,
    MCA,
    MFA,
    OPLS,
    PCA,
    PLS,
    PLSDA,
    PRM,
    TPLS,
    AdaptivePCA,
    AdaptivePLS,
    DataFrameDict,
    MCUVScaler,
)

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
    requested = {"md_method": "nipals"}
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


# ---------------------------------------------------------------------------
# #1071: fit checks the ranges, so set_params and clone cannot skip the checks
# ---------------------------------------------------------------------------


def _range_check_data() -> dict[str, typing.Any]:
    """Small data sets for every estimator of the family: X and y, two X-blocks, and TPLS's blocks."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
    y = pd.DataFrame({"y": X.to_numpy() @ [1.0, -0.5, 0.3, 0.0] + rng.normal(0, 0.1, 30)})
    blends = [f"blend{i}" for i in range(20)]
    materials = [f"mat{j}" for j in range(4)]
    return {
        "X": X,
        "y": y,
        "labels": np.where(y["y"] > 0, "high", "low"),
        "blocks": {"b1": X[["a", "b"]], "b2": X[["c", "d"]]},
        "d_matrix": {"G": pd.DataFrame(rng.normal(size=(4, 3)), index=materials, columns=list("pqr"))},
        "tpls_blocks": {
            "F": {"G": pd.DataFrame(rng.random((20, 4)), index=blends, columns=materials)},
            "Z": {"Z1": pd.DataFrame(rng.normal(size=(20, 2)), index=blends, columns=["z1", "z2"])},
            "Y": {"Y1": pd.DataFrame({"y": rng.normal(size=20)}, index=blends)},
        },
    }


# Every estimator that takes ``n_components``, ``max_iter`` or ``tol``: how to build a valid one, and how to fit it.
_FAMILY: dict[str, tuple[Callable[[dict], BaseEstimator], Callable[[BaseEstimator, dict], object]]] = {
    "PCA": (lambda _: PCA(n_components=2), lambda model, data: model.fit(data["X"])),
    "PLS": (lambda _: PLS(n_components=2), lambda model, data: model.fit(data["X"], data["y"])),
    "PLSDA": (lambda _: PLSDA(n_components=2), lambda model, data: model.fit(data["X"], data["labels"])),
    "PRM": (lambda _: PRM(n_components=2), lambda model, data: model.fit(data["X"], data["y"])),
    "MBPCA": (lambda _: MBPCA(n_components=2), lambda model, data: model.fit(data["blocks"])),
    "MBPLS": (lambda _: MBPLS(n_components=2), lambda model, data: model.fit(data["blocks"], data["y"])),
    "TPLS": (
        lambda data: TPLS(n_components=2, d_matrix=data["d_matrix"]),
        lambda model, data: model.fit(DataFrameDict(data["tpls_blocks"])),
    ),
    "AdaptivePCA": (lambda _: AdaptivePCA(n_components=2), lambda model, data: model.fit(data["X"])),
    "AdaptivePLS": (lambda _: AdaptivePLS(n_components=2), lambda model, data: model.fit(data["X"], data["y"])),
}

_OUT_OF_RANGE = [
    ("n_components", 0, r"^n_components must be >= 1; got 0\.$"),
    ("n_components", -1, r"^n_components must be >= 1; got -1\.$"),
    ("max_iter", 0, r"^max_iter must be >= 1; got 0\.$"),
    ("tol", 0.0, r"^Tolerance must exceed machine precision; got 0\.0\.$"),
    ("tol", 10.0, r"^Tolerance should not be too large; got 10\.0\.$"),
]


@pytest.mark.parametrize("route", ["set_params", "clone"])
@pytest.mark.parametrize(
    ("name", "param", "value", "message"),
    [
        pytest.param(name, param, value, message, id=f"{name}-{param}={value}")
        for name, (make, _) in _FAMILY.items()
        for param, value, message in _OUT_OF_RANGE
        if param in make(_range_check_data()).get_params()
    ],
)
def test_fit_refuses_an_out_of_range_setting_however_it_was_set(
    route: str, name: str, param: str, value: float, message: str
) -> None:
    """``set_params`` and ``clone`` do not pass through ``__init__``'s checks, so ``fit`` makes them (#1071).

    Every estimator gives the same message. PCA checks ``tol`` and ``max_iter`` on the SVD path too,
    which these data take: an out-of-range value is refused whichever algorithm would read it.
    """
    data = _range_check_data()
    make, fit = _FAMILY[name]
    model = make(data).set_params(**{param: value})
    if route == "clone":
        model = clone(model)
    with pytest.raises(ValueError, match=message):
        fit(model, data)


@pytest.mark.parametrize("name", list(_FAMILY))
def test_the_constructor_stores_an_out_of_range_setting_unchanged(name: str) -> None:
    """``__init__`` stores its arguments without checking them, as sklearn requires (#1071)."""
    valid = _FAMILY[name][0](_range_check_data())
    requested = {
        param: value
        for param, value in {"n_components": 0, "max_iter": 0, "tol": 0.0}.items()
        if param in valid.get_params()
    }
    model = type(valid)(**{**valid.get_params(), **requested})
    assert {param: model.get_params()[param] for param in requested} == requested


# ---------------------------------------------------------------------------
# #1127: a numpy scalar is a number, and only fit sets state
# ---------------------------------------------------------------------------

# A count from ``np.argmax``, or one of a grid of ``np.arange`` values, is a numpy scalar, not a Python one.
_NUMPY_SCALARS = [("n_components", np.int64(2)), ("max_iter", np.int32(500)), ("tol", np.float32(1e-6))]


@pytest.mark.parametrize(
    ("name", "param", "value"),
    [
        pytest.param(name, param, value, id=f"{name}-{param}={type(value).__name__}")
        for name, (make, _) in _FAMILY.items()
        for param, value in _NUMPY_SCALARS
        if param in make(_range_check_data()).get_params()
    ],
)
def test_fit_takes_a_numpy_scalar_and_leaves_every_setting_as_given(name: str, param: str, value: object) -> None:
    """A numpy scalar fits as its Python value does, and ``fit`` changes no setting, as ``clone`` needs (#1127)."""
    data = _range_check_data()
    make, fit = _FAMILY[name]
    model = make(data).set_params(**{param: value})
    requested = model.get_params(deep=False)
    fit(model, data)
    assert model.n_components_ == 2
    assert all(model.get_params(deep=False)[key] is setting for key, setting in requested.items())


def _all_subclasses(cls: type) -> set[type]:
    return {grandchild for child in cls.__subclasses__() for grandchild in (child, *_all_subclasses(child))}


# Every estimator of the package that declares the constraints sklearn checks its settings against. The walk
# over BaseEstimator's subclasses finds a class only once its module is imported, so load the batch module.
importlib.import_module("process_improve.batch")
_CONSTRAINED = sorted(
    (
        cls
        for cls in _all_subclasses(BaseEstimator)
        if cls.__module__.startswith("process_improve.") and "_parameter_constraints" in vars(cls)
    ),
    key=lambda cls: cls.__name__,
)


@pytest.mark.parametrize("estimator_cls", _CONSTRAINED, ids=lambda cls: cls.__name__)
def test_every_declared_constraint_takes_the_numpy_scalar_of_a_python_number_it_takes(estimator_cls: type) -> None:
    """A bare ``int`` or ``float`` in the constraints refused ``np.int64(2)`` and ``np.float32(0.5)`` (#1127)."""
    for param, constraint in estimator_cls._parameter_constraints.items():
        for python_value, numpy_values in [(2, [np.int32(2), np.int64(2)]), (0.5, [np.float32(0.5)])]:
            try:
                validate_parameter_constraints({param: constraint}, {param: python_value}, estimator_cls.__name__)
            except InvalidParameterError:
                continue
            for numpy_value in numpy_values:
                validate_parameter_constraints({param: constraint}, {param: numpy_value}, estimator_cls.__name__)


def test_the_walk_finds_the_constrained_estimators() -> None:
    """The test above is not vacuous: it reaches the multivariate and the batch estimators."""
    assert {"PCA", "MBPCA", "MBPLS", "TPLS", "BatchPCA", "BatchPLS", "BatchMonitor"} <= {
        c.__name__ for c in _CONSTRAINED
    }


@pytest.mark.parametrize("name", list(_FAMILY))
def test_the_constructor_stores_its_arguments_and_nothing_else(name: str) -> None:
    """``__init__`` sets no public attribute but its arguments, as sklearn requires: no fitted state (#1127)."""
    model = _FAMILY[name][0](_range_check_data())
    assert {attr for attr in vars(model) if not attr.startswith("_")} == set(model.get_params(deep=False))


# ---------------------------------------------------------------------------
# #1144: one rule for every estimator's settings, checked by its fit
# ---------------------------------------------------------------------------


def _every_estimator_data() -> dict[str, typing.Any]:
    """``_range_check_data``, plus what the categorical, functional, ANOVA, Procrustes and batch estimators fit."""
    data = _range_check_data()
    rng = np.random.default_rng(1)
    categories = pd.DataFrame({name: rng.choice(list(levels), 30) for name, levels in [("c1", "abc"), ("c2", "xyz")]})
    batches = {i: pd.DataFrame({"T": rng.normal(size=8) + np.arange(8), "P": rng.normal(size=8)}) for i in range(12)}
    data.update(
        {
            "table": pd.DataFrame(rng.integers(1, 20, size=(5, 4)), columns=list("pqrs")),
            "categories": categories,
            "mixed": pd.concat([data["X"], categories], axis=1),
            "curves": pd.DataFrame(np.sin(np.linspace(0, 3, 25)) * rng.normal(1, 0.3, (30, 1))),
            "design": pd.DataFrame({"f": np.repeat(["lo", "hi"], 15), "g": np.tile(["u", "v", "w"], 10)}),
            "configurations": {"one": data["X"].iloc[:8], "two": data["X"].iloc[:8] * 2 + 1},
            "batches": batches,
            "batch_quality": pd.DataFrame({"q": rng.normal(size=12)}),
        }
    )
    return data


# Every public estimator that takes a count or declares constraints: how to build a valid one, and how to fit it.
_EVERY_ESTIMATOR: dict[str, tuple[Callable[[dict], BaseEstimator], Callable[[BaseEstimator, dict], object]]] = {
    **_FAMILY,
    "OPLS": (lambda _: OPLS(n_orthogonal_components=1), lambda model, data: model.fit(data["X"], data["y"])),
    "CA": (lambda _: CA(n_components=2), lambda model, data: model.fit(data["table"])),
    "MCA": (lambda _: MCA(n_components=2), lambda model, data: model.fit(data["categories"])),
    "FAMD": (lambda _: FAMD(n_components=2), lambda model, data: model.fit(data["mixed"])),
    "MFA": (
        lambda _: MFA(groups={"g1": ["a", "b"], "g2": ["c", "d"]}, n_components=2),
        lambda model, data: model.fit(data["X"]),
    ),
    "FPCA": (lambda _: FPCA(n_components=2, n_basis=10), lambda model, data: model.fit(data["curves"])),
    "ASCA": (lambda _: ASCA(n_components=2), lambda model, data: model.fit(data["X"], data["design"])),
    "GPA": (lambda _: GPA(), lambda model, data: model.fit(data["configurations"])),
    "BatchPCA": (lambda _: BatchPCA(n_components=2), lambda model, data: model.fit(data["batches"])),
    "BatchPLS": (
        lambda _: BatchPLS(n_components=2),
        lambda model, data: model.fit(data["batches"], data["batch_quality"]),
    ),
    "BatchMonitor": (
        lambda data: BatchMonitor(BatchPCA(n_components=2).fit(data["batches"])),
        lambda model, data: model.fit(data["batches"]),
    ),
    "ResampleAligner": (lambda _: ResampleAligner(reference_batch=0), lambda model, data: model.fit(data["batches"])),
    "DTWAligner": (
        lambda _: DTWAligner(reference_batch=0, settings={"show_progress": False}),
        lambda model, data: model.fit(data["batches"]),
    ),
    "BatchFeatureExtractor": (lambda _: BatchFeatureExtractor(), lambda model, data: model.fit(data["batches"])),
}

# The settings that count something: components, rounds of an iteration, rows of a window, or B-splines.
_COUNTS = (
    "n_components",
    "n_orthogonal_components",
    "max_iter",
    "max_weight_iter",
    "spe_limit_window",
    "n_basis",
    "degree",
    "penalty_order",
)


def _count_settings() -> list:
    """One case per estimator and count setting it takes."""
    data = _every_estimator_data()
    return [
        pytest.param(name, param, id=f"{name}-{param}")
        for name, (make, _) in _EVERY_ESTIMATOR.items()
        for param in _COUNTS
        if param in make(data).get_params(deep=False)
    ]


def test_the_registry_holds_every_estimator_that_counts_or_declares_constraints() -> None:
    """A new estimator with a count or constraints joins the tests below, or this fails (#1144)."""
    expected = {
        cls.__name__
        for cls in _all_subclasses(BaseEstimator)
        if cls.__module__.startswith("process_improve.")
        and not cls.__name__.startswith("_")
        and (hasattr(cls, "_parameter_constraints") or set(_COUNTS) & set(cls._get_param_names()))
    }
    assert expected == set(_EVERY_ESTIMATOR)


@pytest.mark.parametrize(("name", "param"), _count_settings())
def test_a_count_that_is_not_a_whole_number_is_refused(name: str, param: str) -> None:
    """``n_components=2.5`` fitted two components without a word outside PCA and the multi-block family (#1144)."""
    data = _every_estimator_data()
    make, fit = _EVERY_ESTIMATOR[name]
    model = make(data).set_params(**{param: 2.5})
    message = rf"^The '{param}' parameter of {name} must be an instance of 'int'( or None)?\. Got 2\.5 instead\.$"
    with pytest.raises(InvalidParameterError, match=message):
        fit(model, data)


@pytest.mark.parametrize(("name", "param"), _count_settings())
def test_a_count_that_is_a_numpy_integer_is_taken(name: str, param: str) -> None:
    """A count from ``np.argmax`` or ``np.arange`` is a whole number too, and fits (#1140)."""
    data = _every_estimator_data()
    make, fit = _EVERY_ESTIMATOR[name]
    model = make(data)
    fit(model.set_params(**{param: np.int64(model.get_params()[param])}), data)


def _declared_settings() -> list:
    """
    One case per estimator and setting it declares a constraint for.

    A setting declared ``"no_validation"``, as sklearn spells "any value", has no constraint to
    break: the aligners' ``reference_batch`` may be any key, and fit refuses one that names no batch.
    """
    data = _every_estimator_data()
    return [
        pytest.param(name, param, id=f"{name}-{param}")
        for name, (make, _) in _EVERY_ESTIMATOR.items()
        for param, constraint in getattr(make(data), "_parameter_constraints", {}).items()
        if constraint != "no_validation"
    ]


@pytest.mark.parametrize(("name", "param"), _declared_settings())
def test_fit_checks_every_declared_constraint(name: str, param: str) -> None:
    """A declaration nothing checks is no check: the batch estimators declared theirs and never read them (#1144)."""
    data = _every_estimator_data()
    make, fit = _EVERY_ESTIMATOR[name]
    model = make(data).set_params(**{param: object()})
    with pytest.raises(InvalidParameterError, match=rf"^The '{param}' parameter of {name} must be "):
        fit(model, data)


# The counts a method takes as an argument, not a setting: each refuses 0 in the settings' words.
_COUNT_ARGUMENTS: dict[str, Callable[[dict], object]] = {
    "max_weight_iter": lambda data: PRM(n_components=2, max_weight_iter=0).fit(data["X"], data["y"]),
    "max_components": lambda data: MBPLS.select_n_components(data["blocks"], data["y"], max_components=0),
    "n_permutations": lambda data: GPA().fit(data["configurations"]).consensus_test(n_permutations=0),
}


@pytest.mark.parametrize("name", list(_COUNT_ARGUMENTS))
def test_a_count_below_one_is_refused_in_the_family_wording(name: str) -> None:
    """These three said "at least 1" where every setting says ">= 1" (#1144)."""
    with pytest.raises(ValueError, match=rf"^{name} must be >= 1; got 0\.$"):
        _COUNT_ARGUMENTS[name](_every_estimator_data())
