r"""(c) Kevin Dunn, 2010-2026. MIT License.

REML for a linear model with crossed random intercepts, and Satterthwaite F-tests.

The model is

.. math::

    y = X\beta + \sum_k Z_k u_k + e, \qquad u_k \sim N(0, \sigma^2_k I), \qquad e \sim N(0, \sigma^2 I),

with each :math:`Z_k` the 0/1 indicator matrix of one grouping (a panelist, a
panelist-by-product cell, a session). So :math:`V = \sigma^2 \tilde V(\psi)` with
:math:`\tilde V = I + \sum_k \psi_k Z_k Z_k^\top` and :math:`\psi_k = \sigma^2_k / \sigma^2`.

Profiled over :math:`\sigma^2`, the REML criterion (minus twice the restricted
log-likelihood) depends on :math:`\psi` alone:

.. math::

    d(\psi) = \log|\tilde V| + \log|X^\top \tilde V^{-1} X|
              + (n - p)\left(1 + \log \frac{2\pi\, y^\top \tilde P y}{n - p}\right),

with :math:`\tilde P = \tilde V^{-1} - \tilde V^{-1} X (X^\top \tilde V^{-1} X)^{-1} X^\top \tilde V^{-1}`.
This is the quantity lme4 reports as ``REMLcrit``, so differences of it are
lme4's likelihood-ratio statistics. Its gradient and Hessian in :math:`\psi`
have closed forms in :math:`Z^\top \tilde P Z` and :math:`Z^\top \tilde P y`. The fit
is a projected Newton search over ``psi >= 0`` with that exact Hessian, which ends
with the score zero to rounding. Unbounded, as JMP fits by default, a component may go
below zero wherever :math:`\tilde V` stays positive definite; in a balanced design the
estimates are then those of the analysis of variance, a negative one included.

Satterthwaite's denominator degrees of freedom use the observed information of
the variance parameters, as lmerTest does, and the exact derivatives of
:math:`\operatorname{cov}(\hat\beta)` with respect to them. A component estimated
at zero drops out of both, as it does in lmerTest's parametrisation, where the
derivatives with respect to it vanish on the boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

#: Relative norm below which a column counts as a combination of the ones before it,
#: as in R's ``qr(..., tol = 1e-7)``, which lme4 uses to drop aliased fixed effects.
ALIAS_TOLERANCE = 1e-7

#: The Newton search stops once its predicted fall in the criterion is below this.
_DECREMENT_TOLERANCE = 1e-20

#: A predicted fall below this, relative to the criterion, is lost in its rounding.
_RESOLVABLE = 1e-10

#: Iteration cap of the Newton search, and the shortest step its line search tries.
_MAX_ITERATIONS = 200
_MIN_STEP = 1e-10

#: Relative spread below which the eigenvalues of :func:`require_apart_from_residual` count as equal.
_EQUAL_SPREAD = 1e-10


@dataclass(frozen=True)
class VarianceComponentFit:
    r"""A REML fit of the variance-component model; see the module docstring for the notation.

    Attributes
    ----------
    beta : np.ndarray of shape (p,)
        Generalised least-squares estimates at the REML variance components.
    cov_beta : np.ndarray of shape (p, p)
        Their covariance, :math:`(X^\top V^{-1} X)^{-1}`.
    variances : np.ndarray of shape (K,)
        The random-effect variances :math:`\sigma^2_k`, in the order of ``groups``.
    sigma2 : float
        The residual variance.
    reml_criterion : float
        Minus twice the restricted log-likelihood at the estimates (lme4's ``REMLcrit``).
    vc_cov : np.ndarray of shape (m, m)
        Asymptotic covariance of the variance parameters off the boundary (each nonzero
        :math:`\sigma^2_k`, then :math:`\sigma^2`): the inverse of their observed information.
    dcov : np.ndarray of shape (m, p, p)
        Derivatives of ``cov_beta`` with respect to the same parameters.
    blups : np.ndarray of shape (q,)
        The best linear unbiased predictions of the random effects, group by group in the
        order of ``groups``, each group's levels in the order of their codes.
    fitted : np.ndarray of shape (n,)
        :math:`X\hat\beta + Z\hat u`, the fitted values with the random effects predicted.
    """

    beta: np.ndarray
    cov_beta: np.ndarray
    variances: np.ndarray
    sigma2: float
    reml_criterion: float
    vc_cov: np.ndarray
    dcov: np.ndarray
    blups: np.ndarray = field(default_factory=lambda: np.empty(0))
    fitted: np.ndarray = field(default_factory=lambda: np.empty(0))

    def satterthwaite_df(self, contrast: np.ndarray) -> float:
        """Denominator degrees of freedom of the t-test of ``contrast @ beta`` (Satterthwaite, 1946).

        ``2 v**2 / (g' A g)``, with ``v`` the variance of the estimate, ``g`` its gradient
        with respect to the variance parameters, and ``A`` their asymptotic covariance.
        """
        variance = float(contrast @ self.cov_beta @ contrast)
        gradient = np.einsum("i,kij,j->k", contrast, self.dcov, contrast)
        return 2.0 * variance**2 / float(gradient @ self.vc_cov @ gradient)

    def f_test(self, contrasts: np.ndarray) -> tuple[float, float]:
        """F statistic and Satterthwaite denominator df for ``contrasts @ beta = 0`` (q rows).

        The q x q covariance of the estimates is split into independent directions; each
        gets its own Satterthwaite df, and they are combined by matching the mean of the F
        distribution (Fai and Cornelius, 1996), as lmerTest does.
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


def independent_columns(X: np.ndarray, tol: float = ALIAS_TOLERANCE) -> np.ndarray:
    """Return a mask of the columns of ``X`` that are not combinations of earlier ones.

    A column is dropped when what is left of it after projecting out the columns kept
    before it is below ``tol`` of its norm: the choice R's pivoting QR, and so lme4,
    makes. With assessor-by-covariate columns whose sum is already in the model, the
    last assessor's column is the one dropped.
    """
    keep = np.zeros(X.shape[1], dtype=bool)
    basis = np.empty((X.shape[0], 0))
    for j, column in enumerate(X.T):
        norm = np.linalg.norm(column)
        residual = column - basis @ (basis.T @ column)
        residual -= basis @ (basis.T @ residual)  # second pass, for orthogonality
        if norm > 0 and np.linalg.norm(residual) > tol * norm:
            keep[j] = True
            basis = np.column_stack([basis, residual / np.linalg.norm(residual)])
    return keep


def type1_hypotheses(X: np.ndarray, assign: np.ndarray) -> dict[int, np.ndarray]:
    """Type I (sequential) hypothesis matrices, one per term, for a full-rank ``X``.

    Each term is tested adjusted for the terms before it and ignoring those after.
    The rows are the term's rows of the triangular factor of ``X'X``, scaled to a unit
    diagonal: lmerTest's forward Doolittle construction. ``assign`` gives each column's
    term number; negative numbers (the intercept) get no test.
    """
    triangular = np.linalg.qr(X, mode="r")
    normalised = triangular / np.diag(triangular)[:, None]
    return {int(term): normalised[assign == term] for term in np.unique(assign) if term >= 0}


def require_apart_from_residual(X: np.ndarray, groups: Sequence[np.ndarray], names: Sequence[str]) -> None:
    r"""Raise if a random intercept's variance cannot be told apart from the residual's.

    REML sees the data only through the residuals of the fixed effects, :math:`W y` with
    :math:`W = I - X (X^\top X)^{-1} X^\top`, whose covariance is
    :math:`\sigma^2 W + \sum_k \sigma^2_k W Z_k Z_k^\top W`. Where :math:`W Z_k Z_k^\top W = c W`
    with :math:`c > 0`, term :math:`k` and the residual enter it only through
    :math:`\sigma^2 + c\,\sigma^2_k`, so every split of that sum fits equally well. One reading at
    each level (:math:`Z_k Z_k^\top = I`) is the usual case, but the fixed terms can leave a term
    with more readings than levels in the same state. The condition holds exactly when the
    :math:`n - p` eigenvalues of :math:`W Z_k Z_k^\top W` on the range of :math:`W` are equal, that
    is when :math:`(n - p) \sum \lambda^2 = (\sum \lambda)^2`; both sums come from the small
    :math:`Z_k^\top W Z_k`, which has the same nonzero eigenvalues. A term that only uses up the
    residual degrees of freedom, :math:`\operatorname{rank}[X\ Z_k] = n`, is still told apart when
    those eigenvalues differ.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        The fixed-effect model matrix, of full column rank.
    groups : sequence of np.ndarray of shape (n,)
        One array of integer codes ``0 .. q_k - 1`` per random intercept.
    names : sequence of str
        The random intercepts' names, in the order of ``groups``, for the message.

    Raises
    ------
    ValueError
        Naming the first random intercept whose variance cannot be told apart from the residual's.
    """
    n, p = X.shape
    basis = np.linalg.qr(X)[0]
    for name, codes in zip(names, groups, strict=True):
        indicator = np.eye(int(np.max(codes)) + 1)[codes]
        projected = basis.T @ indicator
        inner = indicator.T @ indicator - projected.T @ projected  # Z_k' W Z_k
        total, squares = float(np.trace(inner)), float(np.sum(inner**2))
        # c = 0, a term the fixed effects take up whole, is a different fault: its variance is not estimable at all.
        if total > _EQUAL_SPREAD * n and (n - p) * squares - total**2 <= _EQUAL_SPREAD * (n - p) * squares:
            levels = indicator.shape[1]
            how = (
                f"has one reading at each of its {levels} levels"
                if levels == n
                else "changes the readings in the same way as the residual once the fixed terms are fitted"
            )
            raise ValueError(
                f"{name} {how}, so its variance cannot be told apart from the residual. "
                "Add repeat readings, or leave the term out."
            )


@dataclass(frozen=True)
class _State:
    r"""The profiled fit at one :math:`\psi`: what the score, Hessian and final fit need.

    ``p_tilde`` is :math:`\tilde P`, ``rss`` is :math:`y^\top \tilde P y`, ``zpz`` is
    :math:`Z^\top \tilde P Z` and ``u`` is :math:`Z^\top \tilde P y`.
    """

    criterion: float
    beta: np.ndarray
    xvx: np.ndarray
    v_inv_x: np.ndarray
    p_tilde: np.ndarray
    p_y: np.ndarray
    rss: float
    zpz: np.ndarray
    u: np.ndarray


class _Profile:
    """The profiled REML criterion of one data set, and its derivatives in ``psi``."""

    def __init__(self, y: np.ndarray, X: np.ndarray, groups: Sequence[np.ndarray]) -> None:
        self.y, self.X = y, X
        self.n, self.p = X.shape
        sizes = [int(codes.max()) + 1 for codes in groups]
        self.Z = np.hstack(
            [np.eye(size)[codes] for size, codes in zip(sizes, groups, strict=True)] or [np.empty((self.n, 0))]
        )
        starts = np.cumsum([0, *sizes])[:-1]
        self.blocks = [slice(start, start + size) for start, size in zip(starts, sizes, strict=True)]

    def _factor(self, psi: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return the Cholesky factor of ``I + sum_k psi_k Z_k Z_k'``."""
        expand = np.repeat(psi, [b.stop - b.start for b in self.blocks])
        return linalg.cho_factor(np.eye(self.n) + (self.Z * expand) @ self.Z.T)

    def _criterion(self, factor: tuple, xvx_factor: tuple, rss: float) -> float:
        """Return the REML criterion from the two Cholesky factors and ``y' P~ y``."""
        df_resid = self.n - self.p
        return float(
            2.0 * np.sum(np.log(np.diag(factor[0])))
            + 2.0 * np.sum(np.log(np.diag(xvx_factor[0])))
            + df_resid * (1.0 + np.log(2.0 * np.pi * rss / df_resid))
        )

    def criterion(self, psi: np.ndarray) -> float:
        """Return the REML criterion alone, for a line search: no inverse of ``V~`` is formed.

        Where ``V~`` is not positive definite, as an unbounded search can reach, it is infinite.
        """
        try:
            factor = self._factor(psi)
        except linalg.LinAlgError:
            return np.inf
        solved = linalg.cho_solve(factor, np.column_stack([self.X, self.y]))
        xvx_factor = linalg.cho_factor(self.X.T @ solved[:, :-1])
        xvy = self.X.T @ solved[:, -1]
        rss = float(self.y @ solved[:, -1] - xvy @ linalg.cho_solve(xvx_factor, xvy))
        return self._criterion(factor, xvx_factor, rss)

    def evaluate(self, psi: np.ndarray) -> _State:
        """Return the criterion and the pieces its derivatives and the final fit need."""
        factor = self._factor(psi)
        v_inv = linalg.cho_solve(factor, np.eye(self.n))
        v_inv_x = v_inv @ self.X
        xvx = self.X.T @ v_inv_x
        xvx_factor = linalg.cho_factor(xvx)
        p_tilde = v_inv - v_inv_x @ linalg.cho_solve(xvx_factor, v_inv_x.T)
        p_y = p_tilde @ self.y
        rss = float(self.y @ p_y)
        return _State(
            criterion=self._criterion(factor, xvx_factor, rss),
            beta=linalg.cho_solve(xvx_factor, v_inv_x.T @ self.y),
            xvx=xvx,
            v_inv_x=v_inv_x,
            p_tilde=p_tilde,
            p_y=p_y,
            rss=rss,
            zpz=self.Z.T @ p_tilde @ self.Z,
            u=self.Z.T @ p_y,
        )

    def gradient(self, state: _State) -> np.ndarray:
        """Return the score in ``psi``: ``tr(Z_k' P~ Z_k) - (n - p) |Z_k' P~ y|^2 / y' P~ y``."""
        df_resid = self.n - self.p
        return np.array(
            [np.trace(state.zpz[b, b]) - df_resid * float(state.u[b] @ state.u[b]) / state.rss for b in self.blocks]
        )

    def hessian(self, state: _State) -> np.ndarray:
        """Return the second derivatives of the criterion in ``psi``, in closed form.

        ``-|Z_k' P~ Z_l|^2 + (n - p) (2 u_k' Z_k' P~ Z_l u_l / r - |u_k|^2 |u_l|^2 / r^2)``,
        with ``u = Z' P~ y`` and ``r = y' P~ y``.
        """
        df_resid = self.n - self.p
        u, rss = state.u, state.rss
        size = np.array([float(u[b] @ u[b]) for b in self.blocks])
        hess = np.empty((len(self.blocks), len(self.blocks)))
        for k, bk in enumerate(self.blocks):
            for m, bm in enumerate(self.blocks):
                cross = state.zpz[bk, bm]
                hess[k, m] = -np.sum(cross**2) + df_resid * (
                    2.0 * float(u[bk] @ cross @ u[bm]) / rss - size[k] * size[m] / rss**2
                )
        return hess


def _minimise(profile: _Profile, psi: np.ndarray, bounded: bool = True) -> np.ndarray:
    """Minimise the criterion over ``psi >= 0`` by projected Newton steps with the exact Hessian.

    Unbounded, every component is free and a step stops short of where ``V~`` would cease to
    be positive definite (the criterion is infinite there, so the line search halves the step).

    Components on the boundary with a positive score stay there (the criterion rises
    into the interior); the rest are free. Where the Hessian of the free components is
    not positive definite, its eigenvalues are replaced by their absolute values, which
    keeps the step downhill (a modified Newton method). A step is halved until the
    criterion falls enough (Armijo), unless the predicted fall is too small for the
    criterion's rounding to show, where the quadratic model is the better judge and the
    full step is taken. The search ends when the Newton decrement, the predicted fall,
    is at rounding level, so the score is zero to rounding.
    """
    state = profile.evaluate(psi)
    for _ in range(_MAX_ITERATIONS):
        score = profile.gradient(state)
        free = np.ones(len(psi), dtype=bool) if not bounded else (psi > 0) | (score < 0)
        if not free.any():
            break
        values, vectors = np.linalg.eigh(profile.hessian(state)[np.ix_(free, free)])
        values = np.maximum(np.abs(values), 1e-10 * max(1.0, float(np.max(np.abs(values)))))
        direction = -(vectors / values) @ (vectors.T @ score[free])
        decrement = -float(score[free] @ direction)
        if decrement < _DECREMENT_TOLERANCE:
            break
        length, unresolved = 1.0, decrement < _RESOLVABLE * (1.0 + abs(state.criterion))
        while length > _MIN_STEP:
            trial = psi.copy()
            stepped = psi[free] + length * direction
            trial[free] = np.maximum(stepped, 0.0) if bounded else stepped
            if unresolved or profile.criterion(trial) <= state.criterion + 1e-4 * float(score @ (trial - psi)):
                break
            length /= 2.0
        else:
            break  # no downhill step left: the optimum to rounding
        psi = trial
        state = profile.evaluate(psi)
    return psi


def _information(profile: _Profile, state: _State, free: np.ndarray) -> tuple[np.ndarray, list[slice], np.ndarray]:
    r"""Return the observed information of the nonzero :math:`\sigma^2_k`, then :math:`\sigma^2`.

    With :math:`V_k = Z_k Z_k^\top`, the residual is one more block, :math:`V_e = I = Z_e Z_e^\top`.
    Its entries :math:`y^\top P V_k P V_l P y - \tfrac12 \operatorname{tr}(P V_k P V_l)` are
    then :math:`w_k^\top W_{kl} w_l / \sigma^6 - \tfrac12 |W_{kl}|^2 / \sigma^4`, with
    :math:`W = [Z, I]^\top \tilde P [Z, I]` and :math:`w = [Z, I]^\top \tilde P y`. Also
    returns the blocks of ``[Z, I]`` and the matrix itself, for the derivatives of the
    covariance.
    """
    columns = [np.arange(b.start, b.stop) for b, is_free in zip(profile.blocks, free, strict=True) if is_free]
    z_free = profile.Z[:, np.concatenate(columns)] if columns else np.empty((profile.n, 0))
    z_ext = np.hstack([z_free, np.eye(profile.n)])
    sizes = [len(c) for c in columns] + [profile.n]
    starts = np.cumsum([0, *sizes])[:-1]
    blocks = [slice(start, start + size) for start, size in zip(starts, sizes, strict=True)]
    w_matrix = z_ext.T @ state.p_tilde @ z_ext
    w = z_ext.T @ state.p_y
    sigma2 = state.rss / (profile.n - profile.p)
    info = np.empty((len(blocks), len(blocks)))
    for k, bk in enumerate(blocks):
        for m, bm in enumerate(blocks):
            cross = w_matrix[bk, bm]
            info[k, m] = float(w[bk] @ cross @ w[bm]) / sigma2**3 - 0.5 * np.sum(cross**2) / sigma2**2
    return info, blocks, z_ext


def fit_variance_components(
    y: np.ndarray,
    X: np.ndarray,
    groups: Sequence[np.ndarray],
    start: np.ndarray | None = None,
    bounded: bool = True,
) -> VarianceComponentFit:
    r"""Fit the variance-component model by REML.

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        The response.
    X : np.ndarray of shape (n, p)
        The fixed-effect model matrix, of full column rank (see :func:`independent_columns`).
    groups : sequence of np.ndarray of shape (n,)
        One array of integer codes ``0 .. q_k - 1`` per random intercept.
    start : np.ndarray of shape (K,), optional
        Starting variance ratios :math:`\psi`; ones (lme4's start) by default. A nearby
        fit, such as the model with one term more, shortens the search.
    bounded : bool
        Keep every variance component at or above zero, as lme4 does (the default). ``False``
        lets a component go negative, as JMP's unbounded estimates do; in a balanced design
        they are then the analysis-of-variance estimates.

    Returns
    -------
    VarianceComponentFit
        The estimates and what the F-tests need.

    Raises
    ------
    ValueError
        If ``X`` leaves no residual degrees of freedom.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"The model has {X.shape[1]} fixed-effect columns for {X.shape[0]} observations; none are left for error."
        )
    profile = _Profile(y, X, [np.asarray(codes) for codes in groups])
    psi = np.ones(len(groups)) if start is None else np.asarray(start, dtype=float)
    if len(groups):
        psi = _minimise(profile, psi, bounded=bounded)
    state = profile.evaluate(psi)
    sigma2 = state.rss / (profile.n - profile.p)
    cov_scaled = linalg.inv(state.xvx)
    observed, blocks, z_ext = _information(profile, state, psi != 0)
    # The random effects' predictions, psi_k Z_k' V~^-1 (y - X beta), with V~^-1 (y - X beta) = P~ y.
    blups = np.repeat(psi, [b.stop - b.start for b in profile.blocks]) * state.u
    # d cov(beta) / d sigma2_k = C X' V^-1 V_k V^-1 X C, with V^-1 X = V~^-1 X / sigma2.
    pieces = z_ext.T @ state.v_inv_x
    dcov = np.array([cov_scaled @ pieces[b].T @ pieces[b] @ cov_scaled for b in blocks])
    return VarianceComponentFit(
        beta=state.beta,
        cov_beta=sigma2 * cov_scaled,
        variances=psi * sigma2,
        sigma2=float(sigma2),
        reml_criterion=state.criterion,
        vc_cov=linalg.inv(observed),
        dcov=dcov,
        blups=blups,
        fitted=X @ state.beta + profile.Z @ blups,
    )
