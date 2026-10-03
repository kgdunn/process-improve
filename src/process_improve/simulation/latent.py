r"""(c) Kevin Dunn, 2010-2026. MIT License.

Data with a known latent structure, for checking latent-variable methods.

:class:`LatentStructure` describes a linear process

.. math::

    \mathbf{X} = \mathbf{T}\mathbf{P}^\top + \mathbf{E}, \qquad
    \mathbf{Y} = \mathbf{T}\mathbf{B} + \mathbf{F},

in which the caller sets the number of latent variables, how strongly each one varies,
and which of them drive each response. :meth:`LatentStructure.sample` draws data from
it, optionally with cells missing completely at random. A second call with another seed
draws fresh rows from the same process, which is the ground truth a cross-validated
estimate can be checked against. The best that any linear model can do on new rows is
known in closed form (:meth:`LatentStructure.population_coefficients` and
:meth:`LatentStructure.population_r2`).

The loadings are columns of a Hadamard matrix scaled to unit length, so every entry is
:math:`\pm 1/\sqrt{K}`. Every X column then has the same variance, and autoscaling, which
PCA and PLS apply by default, is a uniform rescale that leaves the structure unchanged.
With loadings of unequal row norms, autoscaling makes the noise heteroscedastic, and a
PLS model of the autoscaled data then needs more components than there are latent
variables to reach the best linear predictor, so the number of latent variables the
process was built with would no longer be the right answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from sklearn.utils import Bunch

from .._random import check_random_state


@dataclass(frozen=True, eq=False)
class LatentStructure:
    """A linear latent-variable process whose structure is known.

    Parameters
    ----------
    x_sd : sequence of float
        Standard deviation of each latent variable. Its length is the number of latent
        variables in X.
    y_coefficients : array-like of shape (n_latent,) or (n_latent, n_targets)
        ``B``: how each latent variable drives each response. A zero row is variation in
        X that Y does not see (Y-orthogonal variation). A 1-D array means one response.
    n_features : int, default=16
        Number of X variables, ``K``: a power of two, larger than the number of latent
        variables.
    noise_sd : float, default=0.3
        Standard deviation of the noise added to every X cell.
    y_noise_sd : float, optional
        Standard deviation of the noise added to every Y cell; ``noise_sd`` when omitted.

    Examples
    --------
    Two latent variables drive ``y``; a PLS model should need two components.

    >>> process = LatentStructure(x_sd=[3.0, 1.0], y_coefficients=[1.0, 1.0])
    >>> train = process.sample(60, random_state=0)
    >>> fresh = process.sample(4000, random_state=1)  # new rows from the same process
    >>> process.n_relevant
    2
    """

    x_sd: tuple[float, ...]
    y_coefficients: np.ndarray
    n_features: int = 16
    noise_sd: float = 0.3
    y_noise_sd: float | None = None

    def __post_init__(self) -> None:
        """Coerce the inputs to arrays and check that the structure can be built."""
        x_sd = tuple(float(value) for value in np.atleast_1d(np.asarray(self.x_sd, dtype=float)))
        coefficients = np.asarray(self.y_coefficients, dtype=float)
        coefficients = coefficients.reshape(-1, 1) if coefficients.ndim == 1 else coefficients
        object.__setattr__(self, "x_sd", x_sd)
        object.__setattr__(self, "y_coefficients", coefficients)
        n_latent = len(x_sd)
        if n_latent == 0 or not all(np.isfinite(x_sd)) or min(x_sd) < 0:
            raise ValueError(f"x_sd must hold one or more finite, non-negative values; got {x_sd}.")
        if coefficients.ndim != 2 or coefficients.shape[0] != n_latent:
            raise ValueError(
                f"y_coefficients must have one row per latent variable ({n_latent}); got shape {coefficients.shape}."
            )
        K = int(self.n_features)
        if K < 2 or K & (K - 1) or n_latent >= K:  # a Hadamard order is a power of two; column 0 is unused
            raise ValueError(
                f"n_features must be a power of two larger than the number of latent variables ({n_latent}); got {K}."
            )
        noise = (self.noise_sd, self._y_noise_sd)
        if not all(np.isfinite(noise)) or min(noise) < 0:
            raise ValueError(f"noise_sd and y_noise_sd must be finite and non-negative; got {noise}.")

    @property
    def _y_noise_sd(self) -> float:
        return float(self.noise_sd if self.y_noise_sd is None else self.y_noise_sd)

    @property
    def n_latent(self) -> int:
        """Number of latent variables in X."""
        return len(self.x_sd)

    @property
    def n_targets(self) -> int:
        """Number of responses in Y."""
        return int(self.y_coefficients.shape[1])

    @property
    def n_relevant(self) -> int:
        """Number of latent variables that drive at least one response (a nonzero row of ``B``)."""
        return int(np.any(self.y_coefficients != 0, axis=1).sum())

    @property
    def loadings(self) -> pd.DataFrame:
        """The loadings ``P``: orthonormal columns with every entry equal to ``+-1/sqrt(K)``."""
        K = int(self.n_features)
        # Column 0 of a Sylvester Hadamard matrix is all ones; the latent variables use the next ones.
        values = hadamard(K)[:, 1 : self.n_latent + 1] / np.sqrt(K)
        return pd.DataFrame(values, index=self._feature_names, columns=self._latent_names)

    @property
    def _feature_names(self) -> list[str]:
        return [f"x{k}" for k in range(int(self.n_features))]

    @property
    def _target_names(self) -> list[str]:
        return [f"y{m}" for m in range(self.n_targets)]

    @property
    def _latent_names(self) -> list[str]:
        return [f"t{a + 1}" for a in range(self.n_latent)]

    def _shrinkage(self) -> np.ndarray:
        """``lambda / (lambda + sigma^2)`` per latent variable: how much of each one X reveals."""
        variances = np.square(self.x_sd)
        total = variances + self.noise_sd**2
        return np.divide(variances, total, out=np.zeros_like(variances), where=total > 0)

    def population_coefficients(self) -> pd.DataFrame:
        r"""Coefficients of the best linear predictor of Y from X, on an infinite sample.

        With :math:`\boldsymbol{\Sigma}_{XX} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^\top
        + \sigma^2\mathbf{I}` and orthonormal :math:`\mathbf{P}`, the regression
        :math:`\boldsymbol{\Sigma}_{XX}^{-1}\boldsymbol{\Sigma}_{XY}` reduces to
        :math:`\mathbf{P}\,\mathrm{diag}\{\lambda_j / (\lambda_j + \sigma^2)\}\,\mathbf{B}`:
        the noise in X shrinks each latent variable's contribution by the fraction of its
        variance that X reveals. Every variable has mean zero, so there is no intercept.

        Returns
        -------
        pd.DataFrame of shape (n_features, n_targets)
        """
        values = self.loadings.to_numpy() @ (self._shrinkage()[:, None] * self.y_coefficients)
        return pd.DataFrame(values, index=self._feature_names, columns=self._target_names)

    def population_r2(self) -> pd.Series:
        r"""Fraction of each response's variance that the best linear predictor explains on new rows.

        No linear model does better on new rows from this process, so this is the value a
        cross-validated :math:`Q^2` estimates at best (an estimate from a finite sample can
        land slightly above it). The predictable part is
        :math:`\sum_j b_{jm}^2\,\lambda_j^2 / (\lambda_j + \sigma^2)`, out of a total
        variance :math:`\sum_j b_{jm}^2\,\lambda_j + \sigma_y^2`.

        Returns
        -------
        pd.Series of length n_targets
        """
        variances = np.square(self.x_sd)
        explained = (variances * self._shrinkage()) @ np.square(self.y_coefficients)
        total = variances @ np.square(self.y_coefficients) + self._y_noise_sd**2
        ratio = np.divide(explained, total, out=np.zeros_like(explained), where=total > 0)
        return pd.Series(ratio, index=self._target_names, name="population_r2")

    def sample(
        self,
        n_samples: int,
        *,
        missing_fraction: float = 0.0,
        random_state: int | np.random.Generator | None = None,
    ) -> Bunch:
        """Draw ``n_samples`` rows from the process.

        The scores, the X noise and the Y noise are drawn in that order, and the missing
        cells after them, so the same ``random_state`` gives the same values with or
        without missing cells.

        Parameters
        ----------
        n_samples : int
            Number of rows, ``N``.
        missing_fraction : float, default=0.0
            Fraction of the X and of the Y cells deleted completely at random (set to
            NaN), in ``[0, 1)``. A row never loses every X cell: one is restored at random,
            so every row can still be scored.
        random_state : int, numpy.random.Generator or None, default=None
            Seed or generator; see ``process_improve._random.check_random_state``.

        Returns
        -------
        sklearn.utils.Bunch
            With ``X`` (DataFrame, ``N x K``), ``Y`` (DataFrame, ``N x M``) and ``scores``
            (DataFrame, ``N x n_latent``, the true latent values ``T``).
        """
        N = int(n_samples)
        if N < 1:
            raise ValueError(f"n_samples must be >= 1; got {n_samples}.")
        if not 0.0 <= missing_fraction < 1.0:
            raise ValueError(f"missing_fraction must lie in [0, 1); got {missing_fraction}.")
        rng = check_random_state(random_state)
        K, M = int(self.n_features), self.n_targets
        scores = rng.standard_normal((N, self.n_latent)) * np.asarray(self.x_sd)
        x = scores @ self.loadings.to_numpy().T + self.noise_sd * rng.standard_normal((N, K))
        y = scores @ self.y_coefficients + self._y_noise_sd * rng.standard_normal((N, M))
        if missing_fraction > 0:
            x_missing = rng.random((N, K)) < missing_fraction
            y_missing = rng.random((N, M)) < missing_fraction
            empty = np.flatnonzero(x_missing.all(axis=1))
            x_missing[empty, rng.integers(K, size=empty.size)] = False
            x[x_missing] = np.nan
            y[y_missing] = np.nan
        return Bunch(
            X=pd.DataFrame(x, columns=self._feature_names),
            Y=pd.DataFrame(y, columns=self._target_names),
            scores=pd.DataFrame(scores, columns=self._latent_names),
        )
