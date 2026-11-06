# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Functional principal component analysis (FPCA) of curves.

Each row of the data is one curve, measured at the points of a common grid; a point a
curve was not measured at is ``NaN``, so curves sampled at their own times share the
union of all the times as their grid. Every curve is smoothed by one penalised B-spline
(P-spline) smoother, and the principal components are found in the L2 metric of the
basis, so they are functions of the grid's argument (time, temperature, wavelength),
not of how the basis happens to be laid out. See ``docs/development/functional_doe.rst``
for how the scores serve a designed experiment.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from sklearn.base import BaseEstimator, TransformerMixin

from ._base import _LazyFrame
from ._common import SpecificationWarning, check_at_least_one, user_stacklevel
from ._impute import _group_by_pattern
from ._pca import PCA

#: The candidate smoothing penalties GCV searches, as multiples of trace(B'B) / trace(D'D), so the search is the
#: same whatever the curves' units or the number of points.
_GCV_GRID = 10.0 ** np.linspace(-8, 4, 49)


def _knots(low: float, high: float, n_basis: int, degree: int) -> np.ndarray:
    """Return equally spaced knots for ``n_basis`` B-splines of this degree on [low, high], extended past both ends.

    Knots repeated at the ends (clamped) make the edge B-splines narrower than the others, so a penalty on the
    differences of their coefficients pulls the fit at the ends more than in the middle: a bias in each end
    larger than the standard error of a predicted curve. Equal spacing that runs past the ends treats every
    B-spline alike (Eilers and Marx, Statistical Science, 1996).
    """
    step = (high - low) / (n_basis - degree)
    return low + step * np.arange(-degree, n_basis + 1)


def _gram(knots: np.ndarray, degree: int) -> np.ndarray:
    """Return W, the integral of the outer product of the B-splines: the L2 inner product of two curves is c1' W c2.

    Gauss-Legendre quadrature with degree + 1 nodes per knot interval integrates the product of two
    B-splines, a polynomial of degree 2 * degree on each interval, exactly.
    """
    nodes, weights = leggauss(degree + 1)
    edges = np.unique(knots[degree:-degree])  # the curves' domain only, not the knots beyond its ends
    mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
    points = (mid[:, None] + half[:, None] * nodes).ravel()
    weight = (half[:, None] * weights).ravel()
    basis = BSpline.design_matrix(points, knots, degree).toarray()
    gram = (basis.T * weight) @ basis
    return (gram + gram.T) / 2  # symmetric to rounding, so its eigendecomposition is real


def _symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """Return a symmetric positive-definite matrix raised to ``power``, through its eigendecomposition."""
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * values**power) @ vectors.T


def _patterns(missing: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group the curves by which points they observe; the curves of one pattern share one smoother."""
    complete = np.flatnonzero(~missing.any(axis=1))
    groups = _group_by_pattern(missing)
    return ([(np.ones(missing.shape[1], dtype=bool), complete)] if complete.size else []) + groups


class FPCA(TransformerMixin, BaseEstimator):
    """Functional principal component analysis of curves measured on a grid.

    Each curve is smoothed by a cubic penalised B-spline (P-spline): ``n_basis`` B-splines on
    equally spaced knots that run past the ends of the grid, with a penalty on the
    ``penalty_order``-th differences of their coefficients. One penalty is shared by every curve, so
    the smoother is one linear map and cannot be mistaken for a difference between curves. The
    principal components of the smoothed curves are then found in the L2 metric of the basis: the
    coefficients are weighted by the
    square root of the Gram matrix W (the integral of the products of the B-splines), so the
    eigenfunctions are orthonormal as functions and do not depend on where the knots are placed
    (Ramsay and Silverman, *Functional Data Analysis*, 2005, section 8.4).

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components. ``None`` fits as many as the curves support: one fewer than the
        number of curves, or ``n_basis``, whichever is smaller. A larger request is clamped to that,
        with a :class:`SpecificationWarning`.
    n_basis : int or None, default=None
        Number of B-splines. ``None`` takes the number of grid points plus two, at most 40: the
        penalty, not the number of B-splines, sets how smooth the curves are.
    smoothing : float or None, default=None
        The penalty's weight. ``None`` chooses it by generalised cross-validation, pooled over all
        the curves.
    degree : int, default=3
        Degree of the B-splines (3 is cubic).
    penalty_order : int, default=2
        Order of the coefficient differences penalised. Raise it to 3 when rates (first derivatives)
        are what is wanted, so the rates are penalised for curvature, not the curves.

    Attributes
    ----------
    n_components_ : int
        The number of components fitted.
    grid_ : np.ndarray of shape (n_points,)
        The grid the curves are measured on.
    n_basis_ : int
        The number of B-splines used.
    knots_ : np.ndarray
        The knots of the B-splines.
    smoothing_ : float
        The penalty's weight, as given or as chosen by cross-validation.
    mean_function_ : pd.Series
        The mean curve on ``grid_``.
    eigenfunctions_ : pd.DataFrame of shape (n_points, n_components)
        The eigenfunctions (functional principal components) on ``grid_``: orthonormal in L2, so
        the integral of each one squared is 1.
    scores_ : pd.DataFrame of shape (n_curves, n_components)
        Each curve's score on each component.
    explained_variance_ : np.ndarray of shape (n_components,)
        The variance of the scores of each component: the eigenvalues of the covariance of the
        smoothed curves.
    explained_variance_ratio_ : pd.Series of length n_components
        The fraction of the smoothed curves' variance each component explains.
    pca_ : PCA
        The PCA of the weighted, centred coefficients the components come from. Its
        ``hotellings_t2_`` and ``spe_`` flag a curve unlike the others.
    n_features_in_ : int
        The number of grid points.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> t = np.linspace(0, 1, 25)
    >>> rng = np.random.default_rng(0)
    >>> sizes = rng.uniform(1, 2, 12)
    >>> curves = pd.DataFrame([a * np.sin(np.pi * t) + rng.normal(0, 0.01, t.size) for a in sizes], columns=t)
    >>> model = FPCA(n_components=2).fit(curves)
    >>> bool(model.explained_variance_ratio_.iloc[0] > 0.99)
    True
    """

    eigenfunctions_ = _LazyFrame("_eigenfunctions", index="_grid_index", columns="_component_names")
    scores_ = _LazyFrame("_scores", index="_sample_index", columns="_component_names")

    def __init__(
        self,
        n_components: int | None = None,
        *,
        n_basis: int | None = None,
        smoothing: float | None = None,
        degree: int = 3,
        penalty_order: int = 2,
    ) -> None:
        self.n_components = n_components
        self.n_basis = n_basis
        self.smoothing = smoothing
        self.degree = degree
        self.penalty_order = penalty_order

    def __getstate__(self) -> dict:
        """Leave the lazily built frames out of a pickle; they are rebuilt from the arrays on access."""
        state = dict(super().__getstate__())
        state.pop("_frame_cache", None)
        return state

    def fit(self, X: pd.DataFrame | np.ndarray, y: object = None, *, grid: np.ndarray | None = None) -> FPCA:  # noqa: ARG002
        """Smooth the curves and find their principal components.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_curves, n_points)
            One curve per row, one grid point per column; ``NaN`` where a curve was not measured.
        y : None
            Ignored; present for the scikit-learn API.
        grid : np.ndarray of shape (n_points,), optional
            The grid, strictly increasing. Taken from the columns of a DataFrame when they are
            numbers (or text that reads as numbers) and ``grid`` is not given.

        Returns
        -------
        FPCA
            The fitted model.
        """
        values, self._sample_index, self.grid_ = self._read(X, grid)
        n_curves, n_points = values.shape
        if n_curves < 2:
            raise ValueError(f"FPCA needs at least two curves; got {n_curves}.")
        self.n_features_in_ = n_points
        self._set_basis(n_points)
        self.smoothing_ = self._choose_smoothing(values) if self.smoothing is None else float(self.smoothing)
        if self.smoothing_ <= 0:
            raise ValueError(f"smoothing must be positive; got {self.smoothing_}.")
        coefficients = self._coefficients(values, self.smoothing_)
        self._mean_coef = coefficients.mean(axis=0)
        self.n_components_ = self._resolve_components(n_curves)
        self._component_names = pd.Index([f"FPC {a + 1}" for a in range(self.n_components_)])
        weighted = pd.DataFrame((coefficients - self._mean_coef) @ self._root_gram, index=self._sample_index)
        self.pca_ = PCA(n_components=self.n_components_, algorithm="svd").fit(weighted)
        loadings = self.pca_.loadings_.to_numpy()
        self._eigen_coef = self._inverse_root_gram @ loadings
        self._scores = self.pca_.scores_.to_numpy()
        self.explained_variance_ = np.asarray(self.pca_.explained_variance_, dtype=float)
        self.explained_variance_ratio_ = pd.Series(
            self.pca_.r2_per_component_.to_numpy(), index=self._component_names, name="explained_variance_ratio"
        )
        self._grid_index = pd.Index(self.grid_, name="grid")
        self._eigenfunctions = self._basis @ self._eigen_coef
        self.mean_function_ = pd.Series(self._basis @ self._mean_coef, index=self._grid_index, name="mean")
        self.__dict__.pop("_frame_cache", None)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return the scores of new curves, measured on the grid the model was fitted to.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_curves, n_points)
            One curve per row, on ``grid_``; ``NaN`` where a curve was not measured.

        Returns
        -------
        pd.DataFrame of shape (n_curves, n_components)
            The scores.
        """
        values, index, _ = self._read(X, self.grid_)
        weighted = (self._coefficients(values, self.smoothing_) - self._mean_coef) @ self._root_gram
        return pd.DataFrame(weighted @ self.pca_.loadings_.to_numpy(), index=index, columns=self._component_names)

    def inverse_transform(
        self, scores: pd.DataFrame | np.ndarray, *, grid: np.ndarray | None = None, derivative: int = 0
    ) -> pd.DataFrame:
        """Return the curves that these scores describe: the mean plus each score times its eigenfunction.

        Parameters
        ----------
        scores : pd.DataFrame or np.ndarray of shape (n_curves, n_components)
            Scores on the fitted components.
        grid : np.ndarray, optional
            Where to evaluate the curves, within the fitted grid's range. ``grid_`` when not given.
        derivative : int, default=0
            Order of the derivative to return: 1 gives each curve's rate of change, read from the
            B-splines exactly, not by differencing.

        Returns
        -------
        pd.DataFrame of shape (n_curves, len(grid))
            One curve per row.
        """
        values = np.atleast_2d(np.asarray(scores, dtype=float))
        if values.shape[1] != self.n_components_:
            raise ValueError(f"Expected {self.n_components_} scores per curve; got {values.shape[1]}.")
        at = self.grid_ if grid is None else np.asarray(grid, dtype=float)
        if at.size and (at.min() < self.grid_[0] or at.max() > self.grid_[-1]):
            raise ValueError(f"grid must lie within the fitted range [{self.grid_[0]}, {self.grid_[-1]}].")
        coefficients = self._mean_coef + values @ self._eigen_coef.T
        curves = BSpline(self.knots_, coefficients.T, self.degree)(at, nu=derivative)
        index = scores.index if isinstance(scores, pd.DataFrame) else None
        return pd.DataFrame(curves.T, index=index, columns=at)

    # ----------------------------------------------------------------------------------------- helpers
    @staticmethod
    def _read(X: pd.DataFrame | np.ndarray, grid: np.ndarray | None) -> tuple[np.ndarray, pd.Index, np.ndarray]:
        """Return the curves as floats, their index, and the grid, checked."""
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.atleast_2d(np.asarray(X, dtype=float)))
        values = frame.to_numpy(dtype=float)
        if grid is None:
            try:
                grid = np.asarray([float(c) for c in frame.columns])
            except (TypeError, ValueError):
                raise ValueError("Give the grid: the columns are not numbers, so they cannot be the grid.") from None
        at = np.asarray(grid, dtype=float).ravel()
        if at.size != values.shape[1]:
            raise ValueError(f"The grid has {at.size} points but the curves have {values.shape[1]} columns.")
        if at.size < 2 or not np.all(np.isfinite(at)) or np.any(np.diff(at) <= 0):
            raise ValueError("The grid must be at least two finite points, strictly increasing.")
        if np.any(np.isinf(values)):
            raise ValueError("The curves hold an infinite value; mark a missing point with NaN.")
        return values, frame.index, at

    def _set_basis(self, n_points: int) -> None:
        """Build the B-splines on the grid, their Gram matrix and its square roots, and the penalty."""
        self.n_basis_ = min(n_points + 2, 40) if self.n_basis is None else int(self.n_basis)
        if self.n_basis_ < max(self.degree + 1, self.penalty_order + 1):
            raise ValueError(f"n_basis must be at least {max(self.degree + 1, self.penalty_order + 1)}.")
        self.knots_ = _knots(self.grid_[0], self.grid_[-1], self.n_basis_, self.degree)
        self._basis = BSpline.design_matrix(self.grid_, self.knots_, self.degree).toarray()
        difference = np.diff(np.eye(self.n_basis_), n=self.penalty_order, axis=0)
        self._penalty = difference.T @ difference
        gram = _gram(self.knots_, self.degree)
        self._root_gram = _symmetric_power(gram, 0.5)
        self._inverse_root_gram = _symmetric_power(gram, -0.5)

    def _coefficients(self, values: np.ndarray, smoothing: float) -> np.ndarray:
        """Return each curve's B-spline coefficients: its penalised least-squares fit to the points it observes."""
        coefficients = np.empty((values.shape[0], self.n_basis_))
        for observed, rows in self._measured(values):
            basis = self._basis[observed]
            system = basis.T @ basis + smoothing * self._penalty
            coefficients[rows] = np.linalg.solve(system, basis.T @ values[np.ix_(rows, observed)].T).T
        return coefficients

    def _choose_smoothing(self, values: np.ndarray) -> float:
        """Return the penalty weight that minimises the generalised cross-validation score, pooled over the curves."""
        patterns = self._measured(values)
        scale = np.trace(self._basis.T @ self._basis) / max(np.trace(self._penalty), np.finfo(float).tiny)
        n_observed = int(np.isfinite(values).sum())
        best, best_score = scale, np.inf
        for weight in scale * _GCV_GRID:
            residual, used = 0.0, 0.0
            for observed, rows in patterns:
                basis = self._basis[observed]
                hat = basis @ np.linalg.solve(basis.T @ basis + weight * self._penalty, basis.T)
                points = values[np.ix_(rows, observed)]
                residual += float(np.sum((points - points @ hat.T) ** 2))
                used += rows.size * float(np.trace(hat))
            score = n_observed * residual / (n_observed - used) ** 2 if n_observed > used else np.inf
            if score < best_score:
                best, best_score = weight, score
        return float(best)

    def _measured(self, values: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return the curves grouped by the points they observe, once each curve has enough points to fit.

        The penalty leaves polynomials of degree below ``penalty_order`` unpenalised, so a curve
        needs at least that many points for its fit to be determined.
        """
        missing = np.isnan(values)
        too_few = np.flatnonzero((~missing).sum(axis=1) < self.penalty_order)
        if too_few.size:
            raise ValueError(
                f"Curve {too_few[0]} has fewer than {self.penalty_order} measured points, so its fit is not determined."
            )
        return _patterns(missing)

    def _resolve_components(self, n_curves: int) -> int:
        """Return the number of components, clamped to what the centred curves can support."""
        most = min(n_curves - 1, self.n_basis_)
        if self.n_components is None:
            return most
        check_at_least_one("n_components", self.n_components)
        wanted = int(self.n_components)
        if wanted > most:
            warnings.warn(
                f"{wanted} components were asked for, but {n_curves} curves on {self.n_basis_} B-splines support "
                f"at most {most}; fitting {most}.",
                SpecificationWarning,
                stacklevel=user_stacklevel(),
            )
        return min(wanted, most)
