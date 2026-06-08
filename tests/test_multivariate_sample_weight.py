"""Weighted PLS fit via ``sample_weight`` (#394).

Locks in the implementation of row-weighted NIPALS in :class:`PLS` via the
``sqrt(w)``-rescale identity. The tests cover the four numerical
properties that have to hold for any reasonable weighted-PLS contract:

1. ``sample_weight = np.ones(N)`` reproduces the unweighted fit exactly.
2. Zero weights collapse to a fit on the surviving rows (i.e. weights of
   ``0`` are equivalent to dropping those rows).
3. Under heteroscedastic noise, the optimal ``sample_weight = 1/sigma_i**2``
   recovers a beta closer to the true generator than the unweighted fit.
4. sklearn's ``Pipeline`` fit-params routing
   (``pipe.fit(X, y, pls__sample_weight=w)``) produces the same fit as a
   direct ``PLS().fit(X_scaled, y_scaled, sample_weight=w)``.

Plus three validation tests for non-negative / finite / length checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from process_improve.multivariate.methods import PLS, MCUVScaler


def _latent_xy(n_samples: int, n_features: int, n_factors: int, seed: int):
    """Two-factor synthetic X / single-target Y with mild noise."""
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((n_samples, n_factors))
    P = rng.standard_normal((n_factors, n_features))
    gamma = rng.standard_normal((n_factors, 1))
    X = pd.DataFrame(T @ P + 0.05 * rng.standard_normal((n_samples, n_features)),
                     columns=[f"x{i}" for i in range(n_features)])
    Y = pd.DataFrame(T @ gamma + 0.05 * rng.standard_normal((n_samples, 1)), columns=["y"])
    return X, Y


def test_sample_weight_ones_reproduces_unweighted_fit() -> None:
    """``sample_weight = ones(N)`` is the identity: every fitted attribute matches the unweighted fit."""
    X, Y = _latent_xy(50, 6, 3, seed=0)
    Xs, Ys = MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y)
    unweighted = PLS(n_components=3).fit(Xs, Ys)
    weighted = PLS(n_components=3).fit(Xs, Ys, sample_weight=np.ones(len(X)))

    for attr in (
        "beta_coefficients_",
        "x_loadings_",
        "x_weights_",
        "direct_weights_",
        "predictions_",
    ):
        np.testing.assert_allclose(
            getattr(weighted, attr).values, getattr(unweighted, attr).values, atol=1e-10,
            err_msg=f"weighted vs unweighted differ on {attr}",
        )
    np.testing.assert_allclose(weighted.scores_.values, unweighted.scores_.values, atol=1e-10)
    np.testing.assert_allclose(weighted.r2_cumulative_.values, unweighted.r2_cumulative_.values, atol=1e-10)


def test_sample_weight_half_zero_equals_fit_on_remaining_half() -> None:
    """Zero weights effectively drop those rows: beta matches a fit on the surviving rows."""
    X, Y = _latent_xy(60, 5, 2, seed=7)
    Xs, Ys = MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y)
    # First half kept, second half excluded.
    w = np.ones(len(X))
    w[len(X) // 2:] = 0.0

    weighted = PLS(n_components=2).fit(Xs, Ys, sample_weight=w)
    subset = PLS(n_components=2).fit(Xs.iloc[: len(X) // 2], Ys.iloc[: len(X) // 2])
    np.testing.assert_allclose(
        weighted.beta_coefficients_.values, subset.beta_coefficients_.values, atol=1e-9,
        err_msg="half-zero weights should fit only the surviving rows",
    )
    # Loadings and weights also collapse cleanly (up to sign).
    np.testing.assert_allclose(
        np.abs(weighted.x_loadings_.values), np.abs(subset.x_loadings_.values), atol=1e-9,
    )


def test_sample_weight_recovers_true_beta_under_heteroscedastic_noise() -> None:
    """Optimal weights down-weight noisy rows: beta is closer to the low-noise-only fit than the unweighted fit."""
    # Comparison is against an "oracle" beta fit on the clean half of the
    # data: with correctly specified weights, weighted PLS on the full
    # heteroscedastic dataset should approach this oracle, while
    # unweighted PLS is dragged by the noisy half.
    rng = np.random.default_rng(42)
    N, K, n_factors = 200, 6, 3
    T = rng.standard_normal((N, n_factors))
    P_true = rng.standard_normal((n_factors, K))
    gamma_true = rng.standard_normal((n_factors, 1))
    sigma = np.where(np.arange(N) < N // 2, 5.0, 0.1)
    Y_clean = T @ gamma_true
    Y_noisy = Y_clean + sigma[:, None] * rng.standard_normal((N, 1))

    X = pd.DataFrame(T @ P_true + 0.05 * rng.standard_normal((N, K)),
                     columns=[f"x{i}" for i in range(K)])
    Y = pd.DataFrame(Y_noisy, columns=["y"])

    sc_x = MCUVScaler().fit(X)
    sc_y = MCUVScaler().fit(Y)
    Xs, Ys = sc_x.transform(X), sc_y.transform(Y)

    # Oracle: fit on the clean half only.
    clean = np.arange(N) >= N // 2
    oracle = PLS(n_components=3).fit(Xs.loc[clean], Ys.loc[clean])
    beta_oracle = oracle.beta_coefficients_.values

    unweighted = PLS(n_components=3).fit(Xs, Ys)
    weighted = PLS(n_components=3).fit(Xs, Ys, sample_weight=1.0 / sigma**2)

    err_unweighted = np.linalg.norm(unweighted.beta_coefficients_.values - beta_oracle)
    err_weighted = np.linalg.norm(weighted.beta_coefficients_.values - beta_oracle)
    # Weighted should be meaningfully closer to the clean-data oracle than
    # the unweighted full-data fit.
    assert err_weighted < 0.5 * err_unweighted, (
        f"weighted err {err_weighted:.3f} not < 0.5 * unweighted err {err_unweighted:.3f}"
    )


def test_pipeline_fit_params_routes_sample_weight_to_pls() -> None:
    """``Pipeline(...).fit(X, y, pls__sample_weight=w)`` reaches the inner PLS.

    Compares the Pipeline's inner PLS (fed scaled X, raw Y by the Pipeline
    machinery) against a direct PLS fed the same (scaled X, raw Y, w).
    Y isn't scaled in the Pipeline because we don't have a Y-scaling step;
    the comparison uses raw Y on both sides for parity.
    """
    X, Y = _latent_xy(60, 5, 3, seed=11)
    w = np.linspace(0.1, 5.0, num=len(X))

    sc = MCUVScaler().fit(X)
    Xs = sc.transform(X)
    direct = PLS(n_components=2).fit(Xs, Y, sample_weight=w)

    pipe = Pipeline([("sc", MCUVScaler()), ("pls", PLS(n_components=2))])
    pipe.fit(X, Y, pls__sample_weight=w)

    inner = pipe.named_steps["pls"]
    np.testing.assert_allclose(inner.beta_coefficients_.values, direct.beta_coefficients_.values, atol=1e-10)


def test_sample_weight_rejects_negative() -> None:
    X, Y = _latent_xy(30, 4, 2, seed=1)
    Xs, Ys = MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y)
    w = np.ones(len(X))
    w[3] = -0.5
    with pytest.raises(ValueError, match=r"non-negative"):
        PLS(n_components=2).fit(Xs, Ys, sample_weight=w)


def test_sample_weight_rejects_nan_or_inf() -> None:
    X, Y = _latent_xy(30, 4, 2, seed=1)
    Xs, Ys = MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y)
    w = np.ones(len(X))
    w[2] = np.nan
    with pytest.raises(ValueError, match=r"finite"):
        PLS(n_components=2).fit(Xs, Ys, sample_weight=w)


def test_sample_weight_rejects_wrong_length() -> None:
    X, Y = _latent_xy(30, 4, 2, seed=1)
    Xs, Ys = MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y)
    with pytest.raises(ValueError, match=r"sample_weight has"):
        PLS(n_components=2).fit(Xs, Ys, sample_weight=np.ones(len(X) - 1))


def test_cross_validate_threads_sample_weight_per_fold() -> None:
    """``PLS.cross_validate(X, Y, sample_weight=w)`` subsets ``w`` by training fold.

    Validation: weights of the wrong length raise; ``sample_weight=ones(N)``
    reproduces the unweighted CV (the per-fold subset of ones is still
    ones, so each refit is unweighted).
    """
    X, Y = _latent_xy(40, 5, 2, seed=3)
    Xs, Ys = MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y)

    model = PLS(n_components=2).fit(Xs, Ys)

    # Length validation reaches into cross_validate too.
    with pytest.raises(ValueError, match=r"sample_weight has"):
        model.cross_validate(Xs, Ys, cv=5, random_state=0, show_progress=False,
                             sample_weight=np.ones(len(X) - 1))

    # Equivalence: ones-weighted CV matches the unweighted CV (per-fold
    # subset of ones is still ones, so each refit is unweighted).
    cv_unweighted = model.cross_validate(Xs, Ys, cv=5, random_state=0, show_progress=False)
    cv_weighted = model.cross_validate(
        Xs, Ys, cv=5, random_state=0, show_progress=False, sample_weight=np.ones(len(X)),
    )
    np.testing.assert_allclose(cv_unweighted.beta_mean.values, cv_weighted.beta_mean.values, atol=1e-10)
    np.testing.assert_allclose(cv_unweighted.q_squared.values, cv_weighted.q_squared.values, atol=1e-10)
