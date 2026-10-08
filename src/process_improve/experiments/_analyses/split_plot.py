# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Split-plot analysis: REML with a whole-plot random effect, and Satterthwaite tests (#630).

A split-plot design changes its hard-to-change factors only between whole plots, so the
runs inside one whole plot share that whole plot's error. The model is

.. math::

    y = X\beta + Zu + e, \qquad u \sim N(0, \sigma^2_{wp} I), \qquad e \sim N(0, \sigma^2 I),

with :math:`Z` the run-to-whole-plot indicator matrix, so the runs' covariance is
:math:`V = \sigma^2 H(\eta)`, :math:`H(\eta) = I + \eta ZZ^\top` and
:math:`\eta = \sigma^2_{wp} / \sigma^2`. Ordinary least squares is the special case
:math:`\eta = 0`: it treats every run as independent, so the standard errors of the
whole-plot effects come out too small and those of the subplot effects too large.

With one variance component the REML likelihood, profiled over :math:`\sigma^2`, is a
function of :math:`\eta` alone, and :math:`H^{-1}` has a closed form whole plot by whole
plot, so the fit is a bounded one-dimensional search rather than a general mixed-model
optimiser. For a balanced design it reproduces the classical split-plot ANOVA exactly.

Tests use Satterthwaite's denominator degrees of freedom, from the REML expected
information of :math:`(\sigma^2_{wp}, \sigma^2)` and the exact derivatives of
:math:`\operatorname{cov}(\hat\beta)` with respect to them; a multi-degree-of-freedom
term combines its eigen-directions as Fai and Cornelius (1996) do, as lmerTest does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

#: Variance ratios searched before the refinement: zero, then 1e-8 to 1e8.
_ETA_GRID = np.concatenate([[0.0], np.logspace(-8, 8, 81)])


@dataclass(frozen=True)
class RemlFit:
    r"""A REML fit of the split-plot model; see the module docstring for the notation.

    Attributes
    ----------
    beta : np.ndarray of shape (p,)
        Generalised least-squares estimates at the REML variance components.
    cov_beta : np.ndarray of shape (p, p)
        Their covariance, :math:`(X^\top V^{-1} X)^{-1}`.
    sigma2_wp, sigma2 : float
        The whole-plot and the residual (subplot) variance.
    vc_cov : np.ndarray of shape (2, 2)
        Asymptotic covariance of :math:`(\hat\sigma^2_{wp}, \hat\sigma^2)`: the inverse
        of the REML expected information.
    dcov : np.ndarray of shape (2, p, p)
        Derivatives of ``cov_beta`` with respect to :math:`\sigma^2_{wp}` and :math:`\sigma^2`.
    """

    beta: np.ndarray
    cov_beta: np.ndarray
    sigma2_wp: float
    sigma2: float
    vc_cov: np.ndarray
    dcov: np.ndarray

    def satterthwaite_df(self, contrast: np.ndarray) -> float:
        """Denominator degrees of freedom of the t-test of ``contrast @ beta`` (Satterthwaite, 1946).

        ``2 v**2 / (g' A g)``, with ``v`` the variance of the estimate, ``g`` its gradient
        with respect to the two variance components, and ``A`` their asymptotic covariance.
        """
        variance = float(contrast @ self.cov_beta @ contrast)
        gradient = np.einsum("i,kij,j->k", contrast, self.dcov, contrast)
        spread = float(gradient @ self.vc_cov @ gradient)
        return 2.0 * variance**2 / spread if spread > 0 else np.inf

    def wald_test(self, contrasts: np.ndarray) -> tuple[float, float]:
        """F statistic and Satterthwaite denominator df for ``contrasts @ beta = 0`` (q rows).

        The q x q covariance of the estimates is split into independent directions; each
        gets its own Satterthwaite df, and they are combined by matching the mean of the F
        distribution (Fai and Cornelius, 1996; Kuznetsova, Brockhoff and Christensen, 2017).
        """
        variances, vectors = np.linalg.eigh(contrasts @ self.cov_beta @ contrasts.T)
        directions = vectors.T @ contrasts
        f_value = float(np.sum((directions @ self.beta) ** 2 / variances)) / len(variances)
        nus = np.array([self.satterthwaite_df(row) for row in directions])
        if len(nus) == 1 or np.allclose(nus, nus[0], rtol=1e-8):
            return f_value, float(np.mean(nus))
        if np.any(nus <= 2):
            return f_value, 2.0
        expected = float(np.sum(nus / (nus - 2.0)))
        return f_value, 2.0 * expected / (expected - len(nus))


def _gls(
    eta: float, y: np.ndarray, X: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generalised least squares at the variance ratio ``eta``.

    Returns ``(H^-1, X' H^-1 X, beta, residuals)``. Whole plot by whole plot,
    ``(I + eta J)^-1 = I - eta / (1 + eta n_i) J`` for a whole plot of ``n_i`` runs.
    """
    shrink = eta / (1.0 + eta * Z.sum(axis=0))
    h_inv = np.eye(len(y)) - (Z * shrink) @ Z.T
    xhx = X.T @ h_inv @ X
    beta = np.linalg.solve(xhx, X.T @ h_inv @ y)
    return h_inv, xhx, beta, y - X @ beta


def _deviance(eta: float, y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> float:
    """-2 x the REML log-likelihood profiled over ``sigma2``, up to a constant (Harville, 1977)."""
    h_inv, xhx, _, resid = _gls(eta, y, X, Z)
    n, p = X.shape
    return float(
        np.log1p(eta * Z.sum(axis=0)).sum() + np.linalg.slogdet(xhx)[1] + (n - p) * np.log(resid @ h_inv @ resid)
    )


def _score(eta: float, y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> float:
    """Return the derivative of :func:`_deviance` with respect to ``eta``.

    ``tr(Z' P Z) - (n - p) |Z' H^-1 r|^2 / (r' H^-1 r)``, with
    ``P = H^-1 - H^-1 X (X' H^-1 X)^-1 X' H^-1``: the first part from the two log
    determinants, the second from the profiled ``sigma2`` (``beta`` is at its optimum,
    so its own change does not enter).
    """
    h_inv, xhx, _, resid = _gls(eta, y, X, Z)
    h_inv_x = h_inv @ X
    proj = h_inv - h_inv_x @ np.linalg.solve(xhx, h_inv_x.T)
    n, p = X.shape
    z_resid = Z.T @ (h_inv @ resid)
    return float(np.sum(Z * (proj @ Z)) - (n - p) * (z_resid @ z_resid) / (resid @ h_inv @ resid))


def _reml_ratio(y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> float:
    """Return the REML estimate of ``eta``, zero allowed.

    A grid search finds the neighbourhood of the minimum deviance; the root of its
    derivative there is then found to machine precision. Minimising the deviance
    directly would stop at about the square root of machine precision, since the
    deviance is flat at its minimum.
    """
    deviances = [_deviance(eta, y, X, Z) for eta in _ETA_GRID]
    best = int(np.argmin(deviances))
    if best == 0 and _score(0.0, y, X, Z) >= 0:
        return 0.0  # The deviance rises from eta = 0: the whole-plot variance is estimated as zero.
    low, high = _ETA_GRID[max(best - 1, 0)], _ETA_GRID[min(best + 1, len(_ETA_GRID) - 1)]
    if _score(low, y, X, Z) < 0 < _score(high, y, X, Z):
        return float(optimize.brentq(_score, low, high, args=(y, X, Z), xtol=1e-300))
    return float(_ETA_GRID[best])  # Still falling at the end of the grid.


def _check_estimable(X: np.ndarray, Z: np.ndarray) -> None:
    """Raise unless the fixed effects and both variance components can be estimated.

    The whole-plot error has ``rank([X Z]) - rank(X)`` degrees of freedom and the subplot
    error ``n - rank([X Z])``; with either at zero, its variance cannot be separated.
    """
    n, p = X.shape
    rank_x = np.linalg.matrix_rank(X)
    if rank_x < p:
        raise ValueError(
            f"The model matrix has {p} columns but rank {rank_x}, so some terms are aliased and cannot be "
            "estimated. The split-plot analysis needs every term estimable: fit fewer terms."
        )
    rank_xz = np.linalg.matrix_rank(np.hstack([X, Z]))
    n_whole_plots = Z.shape[1]
    if rank_xz == rank_x:
        raise ValueError(
            f"No degrees of freedom are left for the whole-plot error: the model's whole-plot terms use all "
            f"{n_whole_plots} whole plots, so the whole-plot variance cannot be told apart from them. Use more "
            "whole plots, or fit fewer terms in the hard-to-change factors."
        )
    if rank_xz == n:
        raise ValueError(
            "No degrees of freedom are left for the subplot error: the whole plots and the model use up every "
            "run. Put more than one run in each whole plot, or fit fewer terms."
        )


def fit_reml(y: np.ndarray, X: np.ndarray, whole_plots: np.ndarray) -> RemlFit:
    """Fit the split-plot model by REML.

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        The response.
    X : np.ndarray of shape (n, p)
        The fixed-effects model matrix, of full column rank.
    whole_plots : np.ndarray of shape (n,)
        Each run's whole-plot label.

    Returns
    -------
    RemlFit

    Raises
    ------
    ValueError
        If a term is aliased, or either variance component has no degrees of freedom.
    """
    codes, labels = pd.factorize(whole_plots)
    Z = np.zeros((len(y), len(labels)))
    Z[np.arange(len(y)), codes] = 1.0
    _check_estimable(X, Z)

    eta = _reml_ratio(y, X, Z)
    h_inv, xhx, beta, resid = _gls(eta, y, X, Z)
    sigma2 = float(resid @ h_inv @ resid) / (len(y) - X.shape[1])
    cov_beta = sigma2 * np.linalg.inv(xhx)

    # REML expected information for (s2_wp, s2), with dV/ds2_wp = ZZ' and dV/ds2 = I:
    # I_kl = tr(P V_k P V_l) / 2, P = V^-1 - V^-1 X C X' V^-1 (Searle, Casella and McCulloch,
    # 1992, section 6.6). P is symmetric, so each trace is a squared Frobenius norm.
    v_inv_x_c = (h_inv / sigma2) @ X @ cov_beta
    proj = h_inv / sigma2 - v_inv_x_c @ X.T @ (h_inv / sigma2)
    proj_z = proj @ Z
    cross = float(np.sum(proj_z**2))
    info = 0.5 * np.array([[np.sum((Z.T @ proj_z) ** 2), cross], [cross, np.sum(proj**2)]])

    # d cov(beta) / d theta_k = C X' V^-1 V_k V^-1 X C.
    z_g = Z.T @ v_inv_x_c
    dcov = np.stack([z_g.T @ z_g, v_inv_x_c.T @ v_inv_x_c])
    return RemlFit(
        beta=beta,
        cov_beta=cov_beta,
        sigma2_wp=eta * sigma2,
        sigma2=sigma2,
        vc_cov=np.linalg.inv(info),
        dcov=dcov,
    )
