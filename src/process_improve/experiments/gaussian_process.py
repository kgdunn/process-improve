# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Gaussian process models for computer experiments, fitted by maximum likelihood.

A computer model gives the same output every time it is run at the same inputs, so its
experiments are modelled by interpolation rather than regression. The usual model (ordinary
kriging; Sacks, Welch, Mitchell and Wynn, 1989) treats the output as a constant :math:`\mu` plus a
Gaussian process of variance :math:`\sigma^2` whose correlation between two runs falls with their
distance in each input:

.. math::

    R_{ij} = \exp\Big(-\sum_k \theta_k (x_{ik} - x_{jk})^2\Big).

This is the model and the parameterisation of Montgomery's Example 11.4 and of JMP's Gaussian
process platform: the inputs in their own units, :math:`\mu` and :math:`\sigma^2` by maximum
likelihood at each :math:`\theta`, and :math:`\theta` maximising the likelihood,

.. math::

    -2 \log L = n \log(2 \pi \hat\sigma^2) + \log|R| + n, \qquad
    \hat\mu = \frac{1^T R^{-1} y}{1^T R^{-1} 1}, \qquad
    \hat\sigma^2 = \frac{(y - 1\hat\mu)^T R^{-1} (y - 1\hat\mu)}{n}.

The prediction at :math:`x` is :math:`\hat\mu + r(x)^T w` with weights :math:`w = R^{-1}(y - 1\hat\mu)`,
which passes through every run, and its variance is
:math:`\hat\sigma^2 [1 - r^T R^{-1} r + (1 - 1^T R^{-1} r)^2 / (1^T R^{-1} 1)]`.

The sensitivity indices split the variance of the prediction over the box of the inputs into the
share each input explains alone (the main effect) and the share it is part of with the others
(the total effect), as Sobol's indices do. For this correlation the integrals over the box
separate into one integral per input, worked by Gauss-Legendre quadrature to machine precision.
They are arranged so that no two large terms cancel: a smooth output is fitted with a small
:math:`\theta`, and then the weights are large and of both signs, where the usual
:math:`E[f^2] - E[f]^2` loses its digits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import linalg, optimize
from threadpoolctl import threadpool_limits

from process_improve._random import check_random_state

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Random starts of the likelihood search, beside the start at the middle of the range.
_N_STARTS = 20

#: The search looks for theta_k between these multiples of 1 / range_k^2.
_THETA_BOUNDS = (1e-3, 1e5)

#: Added to the diagonal of R, as a share of it, so that the Cholesky factor exists at small theta.
_JITTER = 1e-10

#: Gauss-Legendre nodes per panel of the sensitivity integrals; a panel is one correlation length wide at most.
_NODES = 20


@dataclass(frozen=True, eq=False)
class GaussianProcessFit:
    r"""A Gaussian process model fitted by :func:`fit_gaussian_process`.

    Attributes
    ----------
    inputs : tuple[str, ...]
        The inputs, in the order of ``theta``.
    theta : numpy.ndarray
        The correlation parameter of each input, in the inputs' own units.
    mu : float
        The constant mean.
    sigma2 : float
        The process variance.
    minus_two_log_likelihood : float
        :math:`-2 \log L` at the estimates.
    weights : numpy.ndarray
        :math:`R^{-1}(y - 1\hat\mu)`, one per run: the prediction is :math:`\hat\mu + r(x)^T w`.
    runs : numpy.ndarray
        The inputs of the runs, one row each.
    """

    inputs: tuple[str, ...]
    theta: np.ndarray
    mu: float
    sigma2: float
    minus_two_log_likelihood: float
    weights: np.ndarray = field(repr=False)
    runs: np.ndarray = field(repr=False)
    _factor: np.ndarray = field(repr=False)  # the Cholesky factor of R

    def predict(self, points: pd.DataFrame) -> pd.DataFrame:
        """Predict at new points, with the standard error of each prediction.

        Parameters
        ----------
        points : pandas.DataFrame
            A column per input, in its own units.

        Returns
        -------
        pandas.DataFrame
            ``predicted`` and ``std_error``, one row per point; at a run the prediction is the
            run's output and the standard error is 0.
        """
        r = _correlation(_matrix(points, self.inputs), self.runs, self.theta)  # (m, n)
        ones = np.ones(len(self.runs))
        r_inv_r = linalg.cho_solve((self._factor, True), r.T)  # (n, m)
        r_inv_1 = linalg.cho_solve((self._factor, True), ones)
        explained = np.einsum("nm,nm->m", r.T, r_inv_r)
        correction = (1.0 - ones @ r_inv_r) ** 2 / float(ones @ r_inv_1)
        variance = np.clip(self.sigma2 * (1.0 - explained + correction), 0.0, None)
        return pd.DataFrame(
            {"predicted": self.mu + r @ self.weights, "std_error": np.sqrt(variance)}, index=points.index
        )

    def sensitivity(self, ranges: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
        r"""Return each input's main-effect and total sensitivity index over the box of the inputs.

        Parameters
        ----------
        ranges : dict or None
            Each input's range, ``{"x": (low, high)}``; by default the range of the runs.

        Returns
        -------
        pandas.DataFrame
            ``main_effect`` and ``total_effect``, one row per input: the shares of the
            prediction's variance over the box that the input explains alone, and that it
            explains alone or together with the others. For two inputs, ``total_effect -
            main_effect`` is the share of their interaction.

        Raises
        ------
        ValueError
            If a range names no input, or does not go from a low end to a higher one.

        Notes
        -----
        The prediction is :math:`\hat\mu + \sum_i w_i \prod_k c_{ik}(x_k)`, with
        :math:`c_{ik}(t) = \exp(-\theta_k (t - x_{ik})^2)`. Over the box, write :math:`s_k` for
        the vector of the averages of :math:`c_{ik}`, :math:`D_k` for their covariance matrix,
        and :math:`P_k = s_k s_k^T + D_k`. With products taken entry by entry, the main effect
        and the total effect of input :math:`j` are :math:`V_j / V` and :math:`T_j / V`, where

        .. math::

            V_j = (w \circ a_j)^T D_j (w \circ a_j), \quad a_j = \prod_{k \ne j} s_k, \qquad
            T_j = w^T \Big(D_j \circ \prod_{k \ne j} P_k\Big) w

        are the variance that input :math:`j` explains alone and the variance left, on average,
        when the other inputs are held, and the variance of the prediction is

        .. math::

            V = \sum_m (w \circ b_m)^T \Big(D_m \circ \prod_{k < m} P_k\Big) (w \circ b_m),
            \quad b_m = \prod_{k > m} s_k.
        """
        lows, highs = _box(self.runs, self.inputs, ranges)
        moments = [_moments(self.runs[:, k], self.theta[k], lows[k], highs[k]) for k in range(len(self.inputs))]
        means = np.array([mean for mean, _ in moments])  # (k, n)
        covariances = np.array([covariance for _, covariance in moments])  # (k, n, n)
        products = covariances + means[:, :, None] * means[:, None, :]

        def form(matrix: np.ndarray, scale: np.ndarray) -> float:
            """Return (w * scale)' matrix (w * scale)."""
            scaled = self.weights * scale
            return float(scaled @ matrix @ scaled)

        k, every = len(self.inputs), np.ones_like(self.weights)
        variance = sum(
            form(covariances[m] * np.prod(products[:m], axis=0), np.prod(means[m + 1 :], axis=0)) for m in range(k)
        )
        rows = {}
        for j, name in enumerate(self.inputs):
            others = np.arange(k) != j
            alone = form(covariances[j], np.prod(means[others], axis=0))
            left = form(covariances[j] * np.prod(products[others], axis=0), every)
            rows[name] = {"main_effect": alone / variance, "total_effect": left / variance}
        return pd.DataFrame.from_dict(rows, orient="index")


def fit_gaussian_process(
    data: pd.DataFrame,
    response: str,
    inputs: Sequence[str] | None = None,
    *,
    random_state: int | np.random.Generator | None = 0,
) -> GaussianProcessFit:
    r"""Fit a Gaussian process (ordinary kriging) model by maximum likelihood.

    Parameters
    ----------
    data : pandas.DataFrame
        The runs of a computer experiment: the inputs, in their own units, and the output.
    response : str
        The output's column.
    inputs : sequence of str or None
        The input columns; every other column by default.
    random_state : int, numpy.random.Generator or None
        Seeds the random starts of the likelihood search.

    Returns
    -------
    GaussianProcessFit
        The estimates, with ``predict`` and ``sensitivity``.

    Raises
    ------
    ValueError
        If there are fewer than three runs, a value is missing or not finite, an input or the
        output does not vary, or two runs share their inputs (the model passes through every
        run, so it cannot take two outputs at one setting).

    Examples
    --------
    A smooth function of one input, interpolated from six runs:

    >>> import numpy as np
    >>> import pandas as pd
    >>> runs = pd.DataFrame({"x": np.linspace(0, 1, 6)}).assign(y=lambda d: np.sin(3 * d["x"]))
    >>> fit = fit_gaussian_process(runs, "y")
    >>> predicted = fit.predict(pd.DataFrame({"x": [0.4]}))["predicted"].iloc[0]
    >>> print(f"{predicted:.3f} against sin(1.2) = {np.sin(1.2):.3f}")
    0.932 against sin(1.2) = 0.932
    """
    names = tuple(inputs) if inputs is not None else tuple(c for c in data.columns if c != response)
    x, y = _checked(data, response, names)
    squared = _squared_differences(x, x)
    spans = np.ptp(x, axis=0)
    lows, highs = np.log(_THETA_BOUNDS[0] / spans**2), np.log(_THETA_BOUNDS[1] / spans**2)
    rng = check_random_state(random_state)
    starts = [(lows + highs) / 2.0, *(rng.uniform(lows, highs) for _ in range(_N_STARTS))]
    bounds = list(zip(lows, highs, strict=True))
    # Many small dense factorisations: waking a BLAS thread pool for each costs more than the arithmetic.
    with threadpool_limits(limits=1, user_api="blas"):
        searches = [
            optimize.minimize(_criterion, start, args=(squared, y), jac=True, method="L-BFGS-B", bounds=bounds)
            for start in starts
        ]
    theta = np.exp(min(searches, key=lambda found: found.fun).x)
    factor, mu, sigma2, criterion = _profile(theta, squared, y)
    weights = linalg.cho_solve((factor, True), y - mu)
    return GaussianProcessFit(names, theta, mu, sigma2, criterion, weights, x, factor)


def _checked(data: pd.DataFrame, response: str, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return the inputs and the output as arrays, refusing data the model cannot interpolate."""
    x, y = _matrix(data, names), data[response].to_numpy(dtype=float)
    if len(y) < 3 or not names or not (np.isfinite(x).all() and np.isfinite(y).all()):
        msg = "A Gaussian process needs at least three runs and an input, with every value a finite number."
        raise ValueError(msg)
    fixed = [n for n, span in zip(names, np.ptp(x, axis=0), strict=True) if span == 0]
    if fixed:
        msg = f"Every input must vary over the runs; {fixed} do not."
        raise ValueError(msg)
    if np.ptp(y) == 0:
        msg = f"The output {response!r} is the same in every run; there is nothing to model."
        raise ValueError(msg)
    if len(np.unique(x, axis=0)) < len(x):
        msg = "Two runs share their inputs; a model that passes through every run cannot take two outputs there."
        raise ValueError(msg)
    return x, y


def _matrix(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    missing = [n for n in names if n not in frame.columns]
    if missing:
        msg = f"The inputs {missing} are not columns of the data."
        raise ValueError(msg)
    return frame[list(names)].to_numpy(dtype=float)


def _squared_differences(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (a_ik - b_jk)^2 for every pair of rows and every input, shaped (len(a), len(b), k)."""
    return (a[:, None, :] - b[None, :, :]) ** 2


def _correlation(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return exp(-sum_k theta_k (a_ik - b_jk)^2) for every pair of rows."""
    return np.exp(-_squared_differences(a, b) @ theta)


def _profile(theta: np.ndarray, squared: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Return the Cholesky factor of R, the estimates of mu and sigma^2, and -2 log L, at ``theta``."""
    n = len(y)
    factor = linalg.cholesky(np.exp(-squared @ theta) + _JITTER * np.eye(n), lower=True)
    ones = np.ones(n)
    r_inv_1 = linalg.cho_solve((factor, True), ones)
    mu = float(r_inv_1 @ y / (ones @ r_inv_1))
    residual = y - mu
    sigma2 = float(residual @ linalg.cho_solve((factor, True), residual) / n)
    log_det = 2.0 * float(np.sum(np.log(np.diag(factor))))
    return factor, mu, sigma2, n * np.log(2.0 * np.pi * sigma2) + log_det + n


def _criterion(log_theta: np.ndarray, squared: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Return -2 log L at theta = exp(log_theta), and its gradient in log theta.

    With mu and sigma^2 at their estimates, d(-2 log L) = tr(R^-1 dR) - w' dR w / sigma^2, and
    dR / d(log theta_k) = -theta_k (x_ik - x_jk)^2 R_ij.
    """
    theta = np.exp(log_theta)
    factor, mu, sigma2, value = _profile(theta, squared, y)
    r_inv = linalg.cho_solve((factor, True), np.eye(len(y)))
    weights = r_inv @ (y - mu)
    middle = (r_inv - np.outer(weights, weights) / sigma2) * np.exp(-squared @ theta)
    return value, -theta * np.einsum("ij,ijk->k", middle, squared)


def _box(
    runs: np.ndarray, inputs: tuple[str, ...], ranges: dict[str, tuple[float, float]] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the low and high end of each input's range: as given, or the range of the runs."""
    ranges = ranges or {}
    unknown = sorted(set(ranges) - set(inputs))
    if unknown:
        msg = f"The ranges {unknown} are not inputs of the model; its inputs are {list(inputs)}."
        raise ValueError(msg)
    lows, highs = runs.min(axis=0).copy(), runs.max(axis=0).copy()
    for j, name in enumerate(inputs):
        if name in ranges:
            lows[j], highs[j] = ranges[name]
    if np.any(highs <= lows):
        msg = "Each range must go from a low end to a higher one."
        raise ValueError(msg)
    return lows, highs


def _moments(runs: np.ndarray, theta: float, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the averages over [low, high] of c_i(t) = exp(-theta (t - x_i)^2), one per run, and their covariances.

    The integrals are by Gauss-Legendre quadrature on panels at most 1 / sqrt(theta) wide, which is
    exact to rounding for these Gaussians. The covariances are of c_i - 1, kept by ``expm1`` to
    full relative precision when theta is small and c_i close to 1.
    """
    panels = max(1, int(np.ceil((high - low) * np.sqrt(theta))))
    nodes, weights = np.polynomial.legendre.leggauss(_NODES)
    edges = np.linspace(low, high, panels + 1)
    half = np.diff(edges)[:, None] / 2.0
    t = (edges[:-1, None] + half * (nodes + 1.0)).ravel()
    share = (half * weights).ravel() / (high - low)  # sums to 1: the quadrature of an average
    shifted = np.expm1(-theta * (t[None, :] - runs[:, None]) ** 2)  # c_i(t) - 1
    offset = shifted @ share
    centred = shifted - offset[:, None]
    return 1.0 + offset, (centred * share) @ centred.T
