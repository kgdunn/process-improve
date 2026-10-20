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
With the expected information, the df equal Kenward and Roger's (1997, as pbkrtest
computes them); lmerTest's Satterthwaite df use the observed information instead, and
differ slightly in an unbalanced design.

Kenward and Roger's method also inflates the covariance of the estimates, for the
uncertainty in the variance components that :math:`(X^\top V^{-1} X)^{-1}` leaves out,
and scales the F statistic of a term with more than one degree of freedom. For a single
coefficient the scale is one and the df are Satterthwaite's, so only its standard error
changes. JMP computes them with the observed information; SAS and pbkrtest with the
expected. A random-block analysis is the same model, with the blocks as the whole plots.

A two-way split-plot (strip-plot) design has two crossed random effects instead of one: a
row effect, shared by the runs of one setting of the first stage's hard-to-change factors,
and a column effect, shared by the runs processed together in the second stage, so
:math:`V = \sigma^2_r Z_r Z_r^\top + \sigma^2_c Z_c Z_c^\top + \sigma^2 I`. With more than
one grouping the variance components are estimated by the general crossed-random-intercepts
REML of :func:`~process_improve.regression._variance_components.fit_variance_components`,
and the same Satterthwaite and Kenward-Roger machinery is applied to them
(:func:`fit_crossed_reml`).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from patsy import EvalEnvironment, EvalFactor, ModelDesc, Term, dmatrices, dmatrix
from patsy.builtins import C, Sum
from scipy import optimize, stats

from process_improve.regression._variance_components import fit_variance_components

from ._shared import is_block_term

#: The degrees-of-freedom methods :func:`run_split_plot` offers.
DF_METHODS = ("satterthwaite", "kenward_roger")

#: Variance ratios searched before the refinement: zero, then 1e-8 to 1e8.
_ETA_GRID = np.concatenate([[0.0], np.logspace(-8, 8, 81)])

#: Private names for patsy's ``C`` and ``Sum`` in a rewritten formula. A data column
#: shadows a name in a formula, and ``C`` is a common factor name in designed experiments.
_CODER, _SUM = "_pi_C", "_pi_Sum"
_SUM_CODED = re.compile(rf"{_CODER}\((\w+), {_SUM}\)")


class _WaldTests:
    """The t and F tests of a REML fit, from its estimates and their covariances.

    A subclass holds ``beta``, ``cov_beta``, ``cov_beta_kr``, ``dcov`` (the derivatives of
    ``cov_beta`` with respect to the variance parameters) and ``vc_cov`` (those parameters'
    asymptotic covariance).
    """

    beta: np.ndarray
    cov_beta: np.ndarray
    cov_beta_kr: np.ndarray
    dcov: np.ndarray
    vc_cov: np.ndarray

    def satterthwaite_df(self, contrast: np.ndarray) -> float:
        """Denominator degrees of freedom of the t-test of ``contrast @ beta`` (Satterthwaite, 1946).

        ``2 v**2 / (g' A g)``, with ``v`` the variance of the estimate, ``g`` its gradient
        with respect to the variance parameters, and ``A`` their asymptotic covariance.
        """
        variance = float(contrast @ self.cov_beta @ contrast)
        gradient = np.einsum("i,kij,j->k", contrast, self.dcov, contrast)
        return 2.0 * variance**2 / float(gradient @ self.vc_cov @ gradient)

    def wald_test(self, contrasts: np.ndarray, *, kenward_roger: bool = False) -> tuple[float, float]:
        """F statistic and Satterthwaite denominator df for ``contrasts @ beta = 0`` (q rows).

        The q x q covariance of the estimates is split into independent directions; each
        gets its own Satterthwaite df, and they are combined by matching the mean of the F
        distribution (Fai and Cornelius, 1996; Kuznetsova, Brockhoff and Christensen, 2017).
        With ``kenward_roger`` the test is Kenward and Roger's instead (see :meth:`kenward_roger_test`).
        """
        if kenward_roger:
            return self.kenward_roger_test(contrasts)
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

    def kenward_roger_test(self, contrasts: np.ndarray) -> tuple[float, float]:
        """Kenward and Roger's (1997) F statistic and denominator df for ``contrasts @ beta = 0`` (q rows).

        The Wald statistic on the adjusted covariance ``cov_beta_kr``, scaled by lambda so that
        its first two moments match those of an F distribution on (q, m) degrees of freedom.
        With ``D_i`` the derivatives in ``dcov`` and ``Theta = L' (L Phi L')^-1 L``,
        ``A1 = sum W_ij tr(Theta D_i) tr(Theta D_j)`` and ``A2 = sum W_ij tr(Theta D_i Theta D_j)``.
        For one row, lambda is one and m is Satterthwaite's df.
        """
        q = contrasts.shape[0]
        theta = contrasts.T @ np.linalg.solve(contrasts @ self.cov_beta @ contrasts.T, contrasts)
        traces = np.array([np.trace(theta @ d) for d in self.dcov])
        products = np.array([[np.trace(theta @ di @ theta @ dj) for dj in self.dcov] for di in self.dcov])
        a1 = float(traces @ self.vc_cov @ traces)
        a2 = float(np.sum(self.vc_cov * products))
        estimate = contrasts @ self.beta
        f_value = float(estimate @ np.linalg.solve(contrasts @ self.cov_beta_kr @ contrasts.T, estimate)) / q
        b = (a1 + 6.0 * a2) / (2.0 * q)
        g = ((q + 1.0) * a1 - (q + 4.0) * a2) / ((q + 2.0) * a2)
        denominator = 3.0 * q + 2.0 * (1.0 - g)
        c1, c2, c3 = g / denominator, (q - g) / denominator, (q + 2.0 - g) / denominator
        e_star = 1.0 / (1.0 - a2 / q)
        v_star = (2.0 / q) * (1.0 + c1 * b) / ((1.0 - c2 * b) ** 2 * (1.0 - c3 * b))
        rho = v_star / (2.0 * e_star**2)
        m = 4.0 + (q + 2.0) / (q * rho - 1.0)
        scale = m / (e_star * (m - 2.0))
        return scale * f_value, m


@dataclass(frozen=True)
class RemlFit(_WaldTests):
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
        of the REML expected information, or of the observed information when the fit
        asks for it.
    dcov : np.ndarray of shape (2, p, p)
        Derivatives of ``cov_beta`` with respect to :math:`\sigma^2_{wp}` and :math:`\sigma^2`.
    cov_beta_kr : np.ndarray of shape (p, p)
        Kenward and Roger's (1997) adjusted covariance,
        :math:`\Phi + 2\Phi \{\sum_{ij} W_{ij} (Q_{ij} - P_i \Phi P_j)\} \Phi`, with
        :math:`\Phi` = ``cov_beta``, :math:`W` = ``vc_cov``,
        :math:`P_i = X^\top V^{-1} V_i V^{-1} X` and
        :math:`Q_{ij} = X^\top V^{-1} V_i V^{-1} V_j V^{-1} X`.
    whole_plot_error_df, subplot_error_df : int
        Degrees of freedom of the two error strata: ``rank([X Z]) - rank(X)`` and
        ``n - rank([X Z])``. In a balanced design, whole-plot terms are tested on the first
        and subplot terms on the second.
    """

    beta: np.ndarray
    cov_beta: np.ndarray
    sigma2_wp: float
    sigma2: float
    vc_cov: np.ndarray
    dcov: np.ndarray
    whole_plot_error_df: int
    subplot_error_df: int
    cov_beta_kr: np.ndarray


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


def _check_estimable(X: np.ndarray, Z: np.ndarray) -> tuple[int, int]:
    """Return the whole-plot and subplot error df, raising unless every term and both variances are estimable.

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
    return int(rank_xz - rank_x), int(n - rank_xz)


def fit_reml(y: np.ndarray, X: np.ndarray, whole_plots: np.ndarray, information: str = "expected") -> RemlFit:
    """Fit the split-plot model by REML.

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        The response.
    X : np.ndarray of shape (n, p)
        The fixed-effects model matrix, of full column rank.
    whole_plots : np.ndarray of shape (n,)
        Each run's whole-plot label.
    information : {"expected", "observed"}
        The REML information whose inverse is the variance components' covariance, which
        the degrees of freedom and Kenward and Roger's adjustment use. The observed one falls
        back to the expected one where it is not positive definite.

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
    whole_plot_error_df, subplot_error_df = _check_estimable(X, Z)

    eta = _reml_ratio(y, X, Z)
    h_inv, xhx, beta, resid = _gls(eta, y, X, Z)
    sigma2 = float(resid @ h_inv @ resid) / (len(y) - X.shape[1])
    cov_beta = sigma2 * np.linalg.inv(xhx)

    # With C = cov(beta) and G = V^-1 X C, the variance parameters theta = (s2_wp, s2) have
    # dV/ds2_wp = ZZ' and dV/ds2 = I, so:
    #   d C / d theta_k = G' (dV/d theta_k) G
    #   REML expected information I_kl = tr(P V_k P V_l) / 2, with P = V^-1 - G X' V^-1
    #   (Searle, Casella and McCulloch, 1992, section 6.6). P is symmetric, so each trace
    #   is a squared Frobenius norm.
    v_inv = h_inv / sigma2
    g = v_inv @ X @ cov_beta
    z_g = Z.T @ g
    dcov = np.stack([z_g.T @ z_g, g.T @ g])

    proj = v_inv - g @ X.T @ v_inv
    proj_z = proj @ Z
    z_proj_z = Z.T @ proj_z
    cross = float(np.sum(proj_z**2))
    info = 0.5 * np.array([[np.sum(z_proj_z**2), cross], [cross, np.sum(proj**2)]])
    if information == "observed":
        # -d2 l_R / d theta_k d theta_l = -tr(P V_k P V_l) / 2 + y' P V_k P V_l P y, and P y = V^-1 r.
        u = v_inv @ resid
        z_u = Z.T @ u
        z_proj_u = Z.T @ (proj @ u)
        data = np.array([[z_u @ z_proj_z @ z_u, z_u @ z_proj_u], [z_u @ z_proj_u, u @ proj @ u]])
        observed = data - info
        if np.all(np.linalg.eigvalsh(observed) > 0):
            info = observed
    vc_cov = np.linalg.inv(info)

    # Kenward and Roger's adjustment, with A = V^-1 X: P_1 = A'ZZ'A, P_2 = A'A, and Q_ij the
    # same products with V^-1 between the two derivatives of V.
    a = v_inv @ X
    z_a = Z.T @ a
    p_terms = [z_a.T @ z_a, a.T @ a]
    z_v_a = Z.T @ v_inv @ a
    q12 = z_a.T @ z_v_a
    q_terms = [[z_a.T @ (Z.T @ v_inv @ Z) @ z_a, q12], [q12.T, a.T @ v_inv @ a]]
    middle = sum(
        vc_cov[i, j] * (q_terms[i][j] - p_terms[i] @ cov_beta @ p_terms[j]) for i in range(2) for j in range(2)
    )
    return RemlFit(
        beta=beta,
        cov_beta=cov_beta,
        sigma2_wp=eta * sigma2,
        sigma2=sigma2,
        vc_cov=vc_cov,
        dcov=dcov,
        whole_plot_error_df=whole_plot_error_df,
        subplot_error_df=subplot_error_df,
        cov_beta_kr=cov_beta + 2.0 * cov_beta @ middle @ cov_beta,
    )


@dataclass(frozen=True)
class CrossedFit(_WaldTests):
    r"""A REML fit with several crossed random effects, as of a two-way split-plot design.

    Attributes
    ----------
    beta, cov_beta, cov_beta_kr : np.ndarray
        As in :class:`RemlFit`; ``cov_beta_kr`` is Kenward and Roger's adjusted covariance over
        every variance parameter that is not estimated as zero.
    variances : np.ndarray of shape (K,)
        The random effects' variances, in the order of the groupings.
    sigma2 : float
        The residual variance.
    vc_cov : np.ndarray of shape (m, m)
        Asymptotic covariance of the variance parameters off the boundary (each nonzero
        variance, then ``sigma2``): the inverse of their expected or observed REML information.
    dcov : np.ndarray of shape (m, p, p)
        Derivatives of ``cov_beta`` with respect to the same parameters.
    blups : list of np.ndarray
        The best linear unbiased predictions of each grouping's random effects, its groups in
        the order of their first run.
    error_df : np.ndarray of shape (K + 1,)
        Each grouping's error degrees of freedom, ``rank([X Z_k]) - rank(X)``, then the
        residual's, ``n - rank([X Z_1 ... Z_K])``.
    """

    beta: np.ndarray
    cov_beta: np.ndarray
    cov_beta_kr: np.ndarray
    variances: np.ndarray
    sigma2: float
    vc_cov: np.ndarray
    dcov: np.ndarray
    blups: list[np.ndarray]
    error_df: np.ndarray


def _crossed_error_df(X: np.ndarray, indicators: list[np.ndarray]) -> np.ndarray:
    """Return each grouping's and the residual's error df, raising when one of them has none."""
    n, p = X.shape
    rank_x = np.linalg.matrix_rank(X)
    if rank_x < p:
        raise ValueError(
            f"The model matrix has {p} columns but rank {rank_x}, so some terms are aliased and cannot be "
            "estimated. The split-plot analysis needs every term estimable: fit fewer terms."
        )
    found = [int(np.linalg.matrix_rank(np.hstack([X, Z])) - rank_x) for Z in indicators]
    found.append(int(n - np.linalg.matrix_rank(np.hstack([X, *indicators]))))
    if min(found) == 0:
        raise ValueError(
            "No degrees of freedom are left for the error of every random effect: the model's terms in a "
            "grouping's factors use all its groups, or the groupings and the model use up every run. Use more "
            "groups (more rows or columns of a two-way split-plot design), or fit fewer terms."
        )
    return np.array(found)


def kenward_roger_covariance(
    cov_beta: np.ndarray, X: np.ndarray, v_inv: np.ndarray, derivatives: list[np.ndarray], vc_cov: np.ndarray
) -> np.ndarray:
    r"""Return Kenward and Roger's (1997) adjusted covariance of the estimates.

    :math:`\Phi + 2\Phi \{\sum_{ij} W_{ij} (Q_{ij} - P_i \Phi P_j)\} \Phi`, with
    :math:`\Phi` = ``cov_beta``, :math:`W` = ``vc_cov``, :math:`V_i` the ``derivatives`` of
    the runs' covariance with respect to the variance parameters,
    :math:`P_i = X^\top V^{-1} V_i V^{-1} X` and :math:`Q_{ij} = X^\top V^{-1} V_i V^{-1} V_j V^{-1} X`.
    The second derivatives of :math:`V` vanish, since it is linear in its parameters.
    """
    a = v_inv @ X
    p_terms = [a.T @ d @ a for d in derivatives]
    va = [v_inv @ d @ a for d in derivatives]
    m = len(derivatives)
    middle = sum(
        vc_cov[i, j] * ((derivatives[i] @ a).T @ va[j] - p_terms[i] @ cov_beta @ p_terms[j])
        for i in range(m)
        for j in range(m)
    )
    return cov_beta + 2.0 * cov_beta @ middle @ cov_beta


def fit_crossed_reml(
    y: np.ndarray, X: np.ndarray, groupings: list[np.ndarray], information: str = "expected"
) -> CrossedFit:
    """Fit a model with crossed random effects by REML, with the tests' degrees of freedom and adjustment.

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        The response.
    X : np.ndarray of shape (n, p)
        The fixed-effects model matrix, of full column rank.
    groupings : list of np.ndarray of shape (n,)
        Each random effect's group labels: the rows and the columns of a two-way split-plot design.
    information : {"expected", "observed"}
        The REML information whose inverse is the variance parameters' covariance.

    Returns
    -------
    CrossedFit

    Raises
    ------
    ValueError
        If a term is aliased, or a random effect or the residual has no degrees of freedom.
    """
    codes = [pd.factorize(labels)[0] for labels in groupings]
    indicators = [np.eye(int(c.max()) + 1)[c] for c in codes]
    error_df = _crossed_error_df(X, indicators)
    fit = fit_variance_components(y, X, codes)
    free = fit.variances > 0
    derivatives = [Z @ Z.T for Z, keep in zip(indicators, free, strict=True) if keep] + [np.eye(len(y))]
    v = fit.sigma2 * np.eye(len(y)) + sum(var * Z @ Z.T for var, Z in zip(fit.variances, indicators, strict=True))
    v_inv = np.linalg.inv(v)
    vc_cov = fit.vc_cov
    if information == "expected":
        # I_kl = tr(P V_k P V_l) / 2, with P = V^-1 - V^-1 X cov(beta) X' V^-1.
        g = v_inv @ X
        proj = v_inv - g @ fit.cov_beta @ g.T
        pv = [proj @ d for d in derivatives]
        vc_cov = np.linalg.inv(0.5 * np.array([[np.sum(a * b.T) for b in pv] for a in pv]))
    sizes = [Z.shape[1] for Z in indicators]
    return CrossedFit(
        beta=fit.beta,
        cov_beta=fit.cov_beta,
        cov_beta_kr=kenward_roger_covariance(fit.cov_beta, X, v_inv, derivatives, vc_cov),
        variances=fit.variances,
        sigma2=fit.sigma2,
        vc_cov=vc_cov,
        dcov=fit.dcov,
        blups=np.split(fit.blups, np.cumsum(sizes)[:-1]),
        error_df=error_df,
    )


def sum_coded(formula: str, categorical: Collection[str]) -> str:
    """Rewrite ``formula`` so that each categorical factor has sum-to-zero contrasts.

    The split-plot tests are marginal (Type III): each term adjusted for all the others.
    Under sum coding that is the usual test, a categorical main effect averaged over the
    levels of the factors it interacts with; under patsy's default treatment coding it
    would be the effect at one reference level only.
    """
    response, rhs = formula.split("~", 1)
    for name in categorical:
        rhs = re.sub(rf"(?<![\w.]){re.escape(name)}(?!\w)", f"{_CODER}({name}, {_SUM})", rhs)
    return f"{response}~{rhs}"


def sum_coding_env() -> EvalEnvironment:
    """Return the patsy namespace a :func:`sum_coded` formula is evaluated in."""
    return EvalEnvironment([{_CODER: C, _SUM: Sum}])


def _readable(label: str) -> str:
    """Undo :func:`sum_coded`'s private names in a term or column label: ``Coating[S.C1]``."""
    return _SUM_CODED.sub(r"\1", label)


def _sum_code(code: str) -> str:
    """Return a categorical factor's code with sum-to-zero contrasts: ``x`` and ``C(x)`` become ``_pi_C(x, _pi_Sum)``.

    A ``C(...)`` call keeps its data and its ``levels``, and loses any other contrast; one with
    ``Sum`` contrasts already is returned as it is.
    """
    call = ast.parse(code, mode="eval").body
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in {"C", _CODER} and call.args:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        contrast = call.args[1] if len(call.args) > 1 else keywords.get("contrast")
        helper = contrast.func if isinstance(contrast, ast.Call) else contrast
        if isinstance(helper, ast.Name) and helper.id in {"Sum", _SUM}:
            return code
        levels = call.args[2] if len(call.args) > 2 else keywords.get("levels")
        tail = [] if levels is None else [f"levels={ast.unparse(levels)}"]
        return f"{_CODER}({', '.join([ast.unparse(call.args[0]), _SUM, *tail])})"
    return f"{_CODER}({code}, {_SUM})"


def to_sum_coding(X: pd.DataFrame, df: pd.DataFrame) -> np.ndarray | None:
    """Return ``M`` such that ``M @ beta`` are the coefficients of ``X``'s model under sum-to-zero coding.

    The split-plot tests are marginal (Type III): each term adjusted for all the others. Under
    sum-to-zero contrasts that is the usual test, a categorical main effect averaged over the
    levels of the factors it interacts with. Under patsy's default treatment coding, the
    coefficients of a main effect beside its own interaction are its effect at one reference
    level of the other factor, so a test on them depends on how the factor is coded (#805).

    Recoding every categorical factor to sum-to-zero contrasts keeps each term's columns and
    their span, so the recoded matrix is ``X @ A`` for a square ``A``, and the same fitted
    values give ``beta = A beta_sum``: ``M`` is ``A^-1``. The terms are then tested on
    ``M @ beta``, and the coefficients stay in the coding asked for.

    Parameters
    ----------
    X : pd.DataFrame
        The fixed-effects model matrix, as patsy built it (it carries a ``design_info``).
    df : pd.DataFrame
        The runs ``X`` was built from.

    Returns
    -------
    np.ndarray of shape (p, p) or None
        ``M``, or None when every categorical factor is sum-coded already.

    Raises
    ------
    RuntimeError
        If the recoded matrix does not span ``X``'s columns, which would be a bug.
    """
    info = X.design_info
    recoded = {
        factor: EvalFactor(_sum_code(factor.name()))
        for factor, factor_info in info.factor_infos.items()
        if factor_info.type == "categorical"
    }
    recoded = {factor: new for factor, new in recoded.items() if new.name() != factor.name()}
    if not recoded:
        return None
    terms = [Term([recoded.get(factor, factor) for factor in term.factors]) for term in info.terms]
    summed = dmatrix(ModelDesc([], terms), df, eval_env=sum_coding_env(), NA_action="raise")
    columns = np.asarray(X, dtype=float)
    transform = np.linalg.lstsq(columns, np.asarray(summed, dtype=float), rcond=None)[0]
    scale = 1.0 + float(np.abs(columns).max())
    if list(summed.design_info.term_slices.values()) != list(info.term_slices.values()) or not np.allclose(
        columns @ transform, summed, rtol=0.0, atol=1e-9 * scale
    ):
        raise RuntimeError(
            "internal: the sum-coded model matrix does not span the columns of the model as given; this is a bug."
        )
    return np.linalg.inv(transform)


@dataclass(frozen=True)
class _Recoded(_WaldTests):
    """A fit's estimates under other contrasts, ``M @ beta``, with their covariances and derivatives."""

    beta: np.ndarray
    cov_beta: np.ndarray
    cov_beta_kr: np.ndarray
    dcov: np.ndarray
    vc_cov: np.ndarray


def _tested(fit: _WaldTests, to_sum: np.ndarray | None) -> _WaldTests:
    """Return ``fit`` with the coefficients its terms are tested on: under sum-to-zero coding (:func:`to_sum_coding`).

    The variance parameters do not depend on the coding, so their covariance is kept as it is.
    """
    if to_sum is None:
        return fit
    return _Recoded(
        beta=to_sum @ fit.beta,
        cov_beta=to_sum @ fit.cov_beta @ to_sum.T,
        cov_beta_kr=to_sum @ fit.cov_beta_kr @ to_sum.T,
        dcov=np.einsum("ij,kjl,ml->kim", to_sum, fit.dcov, to_sum),
        vc_cov=fit.vc_cov,
    )


def _constant_within(X: pd.DataFrame, whole_plots: np.ndarray) -> np.ndarray:
    """Whether each model column is constant inside every whole plot, so belongs to the whole-plot stratum."""
    grouped = X.groupby(whole_plots)
    spread = (grouped.max() - grouped.min()).abs().max()
    return (spread <= 1e-9 * (1.0 + X.abs().max())).to_numpy()


def _stratum(whole_plot_columns: np.ndarray | dict[str, np.ndarray], columns: list[int]) -> str:
    """Return the error stratum of ``columns``: the first grouping within whose groups they are constant.

    With one grouping, ``"whole_plot"`` or else ``"subplot"``; with several (a dict from each
    grouping's name to its columns' constancy), that grouping's name or else ``"residual"``.
    """
    if isinstance(whole_plot_columns, dict):
        return next((name for name, constant in whole_plot_columns.items() if constant[columns].all()), "residual")
    return "whole_plot" if bool(whole_plot_columns[columns].all()) else "subplot"


def _coefficients(
    fit: _WaldTests,
    X: pd.DataFrame,
    whole_plot_columns: np.ndarray | dict[str, np.ndarray],
    alpha: float,
    *,
    kenward_roger: bool = False,
) -> list[dict[str, Any]]:
    """One t-test per model column, on its own Satterthwaite df, with a confidence interval.

    With ``kenward_roger`` the standard error is from Kenward and Roger's adjusted covariance;
    the df are the same, since a single coefficient's Kenward-Roger df are Satterthwaite's.
    """
    cov = fit.cov_beta_kr if kenward_roger else fit.cov_beta
    records = []
    for j, column in enumerate(X.columns):
        unit = np.eye(X.shape[1])[j]
        estimate, std_error = float(fit.beta[j]), float(np.sqrt(cov[j, j]))
        dof = fit.satterthwaite_df(unit)
        half_width = float(stats.t.ppf(1.0 - alpha / 2.0, dof)) * std_error
        records.append(
            {
                "term": _readable(str(column)),
                "coefficient": estimate,
                "std_error": std_error,
                "df": dof,
                "t_value": estimate / std_error,
                "p_value": float(2.0 * stats.t.sf(abs(estimate / std_error), dof)),
                "ci_low": estimate - half_width,
                "ci_high": estimate + half_width,
                "stratum": _stratum(whole_plot_columns, [j]),
            }
        )
    return records


def _term_tests(
    fit: _WaldTests,
    X: pd.DataFrame,
    whole_plot_columns: np.ndarray | dict[str, np.ndarray],
    blocks: tuple[str, ...],
    *,
    kenward_roger: bool = False,
) -> list[dict[str, Any]]:
    """One F-test per model term, the block contrasts together as one ``"Block"`` term."""
    columns_of: dict[str, list[int]] = {}
    for term, columns in X.design_info.term_name_slices.items():
        if term != "Intercept":
            source = "Block" if is_block_term(term, blocks) else _readable(term)
            columns_of.setdefault(source, []).extend(range(columns.start, columns.stop))
    records = []
    for source, columns in columns_of.items():
        f_value, df_denominator = fit.wald_test(np.eye(X.shape[1])[columns], kenward_roger=kenward_roger)
        records.append(
            {
                "source": source,
                "df": len(columns),
                "df_denominator": df_denominator,
                "F": f_value,
                "p_value": float(stats.f.sf(f_value, len(columns), df_denominator)),
                "stratum": _stratum(whole_plot_columns, columns),
            }
        )
    return records


@dataclass(frozen=True)
class SplitPlotSpec:
    """How :func:`run_split_plot` analyses the runs.

    Attributes
    ----------
    whole_plot : str or tuple[str, ...]
        The column of the runs that labels each run's whole plot; or several columns, each
        labelling the groups of one of several crossed random effects, as the rows and the
        columns of a two-way split-plot (strip-plot) design do.
    alpha : float
        Significance level of the tests and the confidence intervals.
    blocks : tuple[str, ...]
        The block contrast columns in the formula, tested together as one ``"Block"`` term.
    df_method : {"satterthwaite", "kenward_roger"}
        ``"kenward_roger"`` adjusts the covariance of the estimates and scales the F tests
        as Kenward and Roger (1997) do; see the module docstring.
    information : {"expected", "observed"}
        The REML information the degrees of freedom are computed from (see :func:`fit_reml`).
    """

    whole_plot: str | tuple[str, ...]
    alpha: float
    blocks: tuple[str, ...] = ()
    df_method: str = "satterthwaite"
    information: str = "expected"


def run_split_plot(df: pd.DataFrame, formula: str, spec: SplitPlotSpec) -> dict[str, Any]:
    """Analyse a split-plot experiment by REML, with Satterthwaite's or Kenward and Roger's degrees of freedom.

    Parameters
    ----------
    df : pd.DataFrame
        The runs: the response, the factors and the whole-plot labels, with no missing values.
    formula : str
        The model, as a patsy formula. The coefficients are in the coding it gives each
        categorical factor; the tests are under sum-to-zero coding whatever it gives (see
        :func:`to_sum_coding`).
    spec : SplitPlotSpec
        The whole-plot column, the significance level, the block terms and the method.

    Returns
    -------
    dict[str, Any]
        ``{"split_plot": {...}}``, with ``"split_plot_note"`` when the whole-plot variance
        is estimated as zero. See :func:`~process_improve.experiments.analysis.analyze_experiment`.
    """
    whole_plot, alpha, blocks, information = spec.whole_plot, spec.alpha, spec.blocks, spec.information
    y, X = dmatrices(formula, df, eval_env=sum_coding_env(), NA_action="raise", return_type="dataframe")
    to_sum = to_sum_coding(X, df)
    if not isinstance(whole_plot, str):
        if len(whole_plot) > 1:
            return _run_crossed(df, y.to_numpy(dtype=float).ravel(), X, spec, to_sum)
        whole_plot = whole_plot[0]
    labels = df[whole_plot].to_numpy()  # by position: NA_action="raise" keeps every row, in order
    fit = fit_reml(y.to_numpy(dtype=float).ravel(), X.to_numpy(dtype=float), labels, information)
    whole_plot_columns = _constant_within(X, labels)
    kenward_roger = spec.df_method == "kenward_roger"
    tests = _term_tests(_tested(fit, to_sum), X, whole_plot_columns, blocks, kenward_roger=kenward_roger)
    factor_tests = [row for row in tests if row["source"] != "Block"]
    result: dict[str, Any] = {
        "split_plot": {
            "method": "REML",
            "df_method": "Kenward-Roger" if kenward_roger else "Satterthwaite",
            "information": information,
            "test_type": "III",
            "whole_plot_column": whole_plot,
            "n_whole_plots": len(pd.unique(labels)),
            "variance_components": {
                "whole_plot": fit.sigma2_wp,
                "residual": fit.sigma2,
                "eta": fit.sigma2_wp / fit.sigma2,
            },
            "error_df": {"whole_plot": fit.whole_plot_error_df, "subplot": fit.subplot_error_df},
            "blups": {whole_plot: _whole_plot_blups(fit, y.to_numpy(dtype=float).ravel(), X.to_numpy(float), labels)},
            "coefficients": _coefficients(fit, X, whole_plot_columns, alpha, kenward_roger=kenward_roger),
            "tests": tests,
            "significant_terms": [row["source"] for row in factor_tests if row["p_value"] < alpha],
            "not_significant_terms": [row["source"] for row in factor_tests if row["p_value"] >= alpha],
            "significance_level": alpha,
        }
    }
    if fit.sigma2_wp == 0.0:
        result["split_plot_note"] = (
            "The whole-plot variance is estimated as zero, so the estimates are those of ordinary least "
            "squares. The tests still allow for the uncertainty in that estimate, so whole-plot terms are "
            "judged on fewer degrees of freedom than ordinary least squares would give them."
        )
    return result


def _whole_plot_blups(fit: RemlFit, y: np.ndarray, X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return the best linear unbiased prediction of each whole plot's random effect, by its label.

    ``sigma2_wp Z' V^-1 (y - X beta)``: the whole plot's departure from the fixed effects, shrunk by
    how much of the variance lies between whole plots.
    """
    codes, names = pd.factorize(labels)
    Z = np.eye(len(names))[codes]
    eta = fit.sigma2_wp / fit.sigma2 if fit.sigma2 > 0 else 0.0
    h_inv = _gls(eta, y, X, Z)[0]
    predicted = eta * Z.T @ (h_inv @ (y - X @ fit.beta))
    return {str(name): float(value) for name, value in zip(names, predicted, strict=True)}


def _run_crossed(
    df: pd.DataFrame, y: np.ndarray, X: pd.DataFrame, spec: SplitPlotSpec, to_sum: np.ndarray | None
) -> dict[str, Any]:
    """Analyse runs with several crossed random effects, as :func:`run_split_plot` does with one.

    The result has one variance component, one error df and one set of BLUPs per grouping,
    named by its column, beside the residual's; a term's stratum is the first grouping within
    whose groups its columns are constant (``"residual"`` for none, as a row-by-column
    interaction of a two-way split-plot design).
    """
    names = tuple(spec.whole_plot)
    groupings = [df[name].to_numpy() for name in names]
    fit = fit_crossed_reml(y, X.to_numpy(dtype=float), groupings, spec.information)
    constant = {name: _constant_within(X, labels) for name, labels in zip(names, groupings, strict=True)}
    kenward_roger = spec.df_method == "kenward_roger"
    tests = _term_tests(_tested(fit, to_sum), X, constant, spec.blocks, kenward_roger=kenward_roger)
    factor_tests = [row for row in tests if row["source"] != "Block"]
    variances = dict(zip(names, map(float, fit.variances), strict=True))
    result: dict[str, Any] = {
        "split_plot": {
            "method": "REML",
            "df_method": "Kenward-Roger" if kenward_roger else "Satterthwaite",
            "information": spec.information,
            "test_type": "III",
            "whole_plot_column": list(names),
            "n_whole_plots": {name: len(pd.unique(labels)) for name, labels in zip(names, groupings, strict=True)},
            "variance_components": variances | {"residual": float(fit.sigma2)},
            "variance_ratios": {name: value / fit.sigma2 for name, value in variances.items()},
            "error_df": dict(zip([*names, "residual"], map(int, fit.error_df), strict=True)),
            "blups": {
                name: {str(level): float(v) for level, v in zip(pd.unique(labels), blups, strict=True)}
                for name, labels, blups in zip(names, groupings, fit.blups, strict=True)
            },
            "coefficients": _coefficients(fit, X, constant, spec.alpha, kenward_roger=kenward_roger),
            "tests": tests,
            "significant_terms": [row["source"] for row in factor_tests if row["p_value"] < spec.alpha],
            "not_significant_terms": [row["source"] for row in factor_tests if row["p_value"] >= spec.alpha],
            "significance_level": spec.alpha,
        }
    }
    zero = [name for name, value in variances.items() if value == 0.0]
    if zero:
        result["split_plot_note"] = (
            f"The variance of {', '.join(map(repr, zero))} is estimated as zero; the tests leave it out of the "
            "variance parameters whose uncertainty they allow for."
        )
    return result
