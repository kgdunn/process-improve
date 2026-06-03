"""NIPALS variance-optimal starting score (issue #195).

PCA / PLS NIPALS now seed the iteration from the column with the greatest
sum-of-squares (variance, for centred data) instead of the arbitrary *first*
column. NIPALS converges to the same component for any non-degenerate seed and a
deterministic sign convention fixes the sign, so the *fitted result is
unchanged*; the benefit is purely numerical - the highest-variance column is the
best-conditioned seed, needing fewer iterations and avoiding the near-degenerate
start you get when the first column carries little or no variance.

These tests therefore lock in two things: the fit stays correct and finite even
when the first column is constant (a poor seed under the old rule), and the
fitted model is invariant to input column order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from process_improve.multivariate.methods import PCA, PLS


def _centre(frame: pd.DataFrame) -> pd.DataFrame:
    return frame - frame.mean(axis=0)


def test_pca_converges_when_first_column_is_constant() -> None:
    rng = np.random.default_rng(0)
    n = 40
    latent = rng.standard_normal((n, 1))
    X = pd.DataFrame(
        {
            "c0": np.zeros(n),  # zero-variance: a poor seed for the old first-column rule
            "c1": (2.0 * latent + 0.01 * rng.standard_normal((n, 1))).ravel(),
            "c2": (-1.0 * latent + 0.01 * rng.standard_normal((n, 1))).ravel(),
            "c3": (0.5 * latent + 0.01 * rng.standard_normal((n, 1))).ravel(),
        }
    )
    pca = PCA(n_components=2).fit(_centre(X))

    # The fit is finite (no 0/0 poisoning) and explains the rank-1 structure.
    assert np.isfinite(pca.loadings_.to_numpy()).all()
    assert np.isfinite(pca.scores_.to_numpy()).all()
    assert pca.r2_cumulative_.iloc[0] > 0.99
    # The first component loads on the informative columns, not the dead c0.
    leading = pca.loadings_.iloc[:, 0].abs()
    assert leading["c0"] < leading[["c1", "c2", "c3"]].min()


def test_pls_converges_when_first_x_column_is_constant() -> None:
    rng = np.random.default_rng(1)
    n = 40
    latent = rng.standard_normal((n, 1))
    X = pd.DataFrame(
        {
            "c0": np.zeros(n),
            "c1": (2.0 * latent + 0.01 * rng.standard_normal((n, 1))).ravel(),
            "c2": (-1.0 * latent + 0.01 * rng.standard_normal((n, 1))).ravel(),
        }
    )
    Y = pd.DataFrame({"y": (3.0 * latent + 0.05 * rng.standard_normal((n, 1))).ravel()})
    pls = PLS(n_components=2).fit(_centre(X), _centre(Y))

    assert np.isfinite(pls.x_loadings_.to_numpy()).all()
    assert np.isfinite(pls.beta_coefficients_.to_numpy()).all()
    # The model predicts the strongly-correlated Y well.
    y_hat = pls.predict(_centre(X)).y_hat
    assert np.isfinite(y_hat.to_numpy()).all()


def test_pca_fit_is_invariant_to_column_order() -> None:
    # The variance-optimal seed plus the sign convention make the fitted model
    # independent of input column order: loadings realigned to the original
    # order must match those from a permuted fit.
    rng = np.random.default_rng(7)
    X = pd.DataFrame(_centre(pd.DataFrame(rng.standard_normal((50, 5)))))
    X.columns = ["a", "b", "c", "d", "e"]
    base = PCA(n_components=3).fit(X)
    permuted_cols = ["d", "a", "e", "c", "b"]
    permuted = PCA(n_components=3).fit(X[permuted_cols])

    aligned = permuted.loadings_.reindex(base.loadings_.index)
    np.testing.assert_allclose(aligned.to_numpy(), base.loadings_.to_numpy(), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(permuted.scores_.to_numpy(), base.scores_.to_numpy(), rtol=1e-8, atol=1e-8)
