# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Variance dispersion: how a design's prediction variance spreads over spheres about its centre.

The scaled prediction variance of a design of :math:`N` runs at a point :math:`x` (in coded units) is

.. math::

    v(x) = N\,\mathrm{Var}(\hat y(x))/\sigma^2 = N\,f(x)^\top (X^\top X)^{-1} f(x),

with :math:`f(x)` the model's terms at :math:`x` and :math:`X` the model matrix of the runs. A
variance dispersion graph plots its minimum, average and maximum over the sphere of each radius
:math:`r` about the centre (Myers, Montgomery and Anderson-Cook; Montgomery, section 11.4). A
design is *rotatable* when :math:`v` is the same at every point of each sphere, so that the
minimum and the maximum meet: a rotatable central composite design, or a Box-Behnken design in
four or seven factors.

The model's terms are polynomials in the factors, so :math:`v` is one too. It is written once in
monomials, which makes the sphere's average exact: the average of a monomial over the unit sphere
in :math:`k` dimensions is

.. math::

    \mathrm{E}\left[\prod_i u_i^{e_i}\right] = \frac{\prod_i (e_i - 1)!!}{k (k + 2) \cdots (k + |e| - 2)}

when every :math:`e_i` is even, and 0 otherwise. The minimum and the maximum are searched for:
the best of a seeded sample of directions, refined by a local search on the sphere.

Examples
--------
A Box-Behnken design in four factors is rotatable, and one in three is not:

>>> from process_improve.experiments import Factor, generate_design, variance_dispersion
>>> def box_behnken(k):
...     factors = [Factor(name=f"x{i + 1}", low=-1, high=1) for i in range(k)]
...     return generate_design(factors, design_type="box_behnken", n_center_points=3)
>>> variance_dispersion(box_behnken(4))["rotatable"], variance_dispersion(box_behnken(3))["rotatable"]
(True, False)
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from patsy import DesignInfo, build_design_matrices
from scipy import optimize

from process_improve._random import check_random_state
from process_improve.experiments.evaluate import _CODED_LIMIT_FLOOR, _build_model_matrix, _separate_run_columns
from process_improve.experiments.factor import DesignResult

#: The highest degree of a model, in the factors, whose prediction variance is written in monomials.
MAX_DEGREE = 3

#: The minimum and the maximum on a sphere meet within this fraction of its average on every
#: sphere of a rotatable design.
ROTATABLE_TOLERANCE = 1e-8

#: The best directions of the sample from which the minimum and the maximum are refined.
N_STARTS = 4


def variance_dispersion(
    design_matrix: pd.DataFrame | DesignResult,
    model: str | None = "quadratic",
    radii: int | Sequence[float] = 21,
    n_directions: int = 2000,
    random_state: int | np.random.Generator | None = 42,
) -> dict[str, Any]:
    r"""Return the minimum, average and maximum scaled prediction variance on spheres about the centre.

    Parameters
    ----------
    design_matrix : pandas.DataFrame or DesignResult
        The design, in coded units, its factors all numeric. ``RunOrder`` and ``Block`` columns are
        set aside, as :func:`evaluate_design` does when the model does not name them.
    model : str or None
        ``"main_effects"``, ``"interactions"``, ``"quadratic"`` (the default, as response-surface
        designs are compared), or a formula; None means ``"interactions"``. Its terms must be
        polynomials of degree 3 or less in the factors.
    radii : int or Sequence[float]
        The radii of the spheres: a number of them, evenly spaced from 0 to the largest distance of
        a run from the centre, or the radii themselves.
    n_directions : int
        The directions sampled on each sphere, the same on every sphere, before the minimum and the
        maximum are refined.
    random_state : int, numpy.random.Generator or None
        Seeds the directions; see :doc:`/development/reproducibility`.

    Returns
    -------
    dict[str, Any]
        - ``radius``: the radii;
        - ``minimum``, ``mean`` and ``maximum``: the scaled prediction variance on each sphere; the
          mean is the exact average over the sphere;
        - ``spread``: the maximum less the minimum, 0 on every sphere of a rotatable design;
        - ``rotatable``: True when the spread is within ``ROTATABLE_TOLERANCE`` of the mean on every
          sphere;
        - ``design_radius``: the largest distance of a run from the centre;
        - ``n_runs`` and ``n_terms``: :math:`N`, and the model's columns.

    Raises
    ------
    ValueError
        If a factor is not numeric, the design cannot estimate the model, the model is not a
        polynomial of degree 3 or less in the factors, a radius is negative or not a finite
        number, or ``n_directions`` is not a whole number of at least 1.
    """
    design_df, names = _numeric_design(design_matrix, model)
    if isinstance(n_directions, bool) or not isinstance(n_directions, (int, np.integer)) or n_directions < 1:
        msg = f"n_directions must be a whole number of at least 1; got {n_directions!r}."
        raise ValueError(msg)
    X, _, design_info = _build_model_matrix(design_df, model, names)
    information = X.T @ X
    if np.linalg.matrix_rank(information) < X.shape[1]:
        msg = f"The design's {len(X)} runs cannot estimate the model's {X.shape[1]} terms: X'X is singular."
        raise ValueError(msg)
    exponents, quadratic = _monomial_form(design_info, names, len(X) * np.linalg.inv(information))
    design_radius = float(np.linalg.norm(design_df[names].to_numpy(dtype=float), axis=1).max())
    spheres = _radii(radii, design_radius)

    rng = check_random_state(random_state)
    directions = rng.normal(size=(n_directions, len(names)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    mean = _sphere_mean(exponents, quadratic)
    minimum, maximum = zip(*(_extremes(exponents, quadratic, directions, r) for r in spheres), strict=True)
    spread = np.array(maximum) - np.array(minimum)
    averages = [mean(r) for r in spheres]
    return {
        "radius": spheres.tolist(),
        "minimum": list(minimum),
        "mean": averages,
        "maximum": list(maximum),
        "spread": spread.tolist(),
        "rotatable": bool(np.all(spread <= ROTATABLE_TOLERANCE * np.array(averages))),
        "design_radius": design_radius,
        "n_runs": len(X),
        "n_terms": X.shape[1],
    }


def _numeric_design(design_matrix: pd.DataFrame | DesignResult, model: str | None) -> tuple[pd.DataFrame, list[str]]:
    """Return the design's factor columns and their names, refusing a factor that is not numeric."""
    if isinstance(design_matrix, DesignResult):
        design_df, names = pd.DataFrame(design_matrix.design), list(design_matrix.factor_names)
    else:
        design_df = pd.DataFrame(design_matrix)
        names = [str(column) for column in design_df.columns]
    design_df, names, _ = _separate_run_columns(design_df, names, model)
    labels = [name for name in names if not pd.api.types.is_numeric_dtype(design_df[name])]
    if labels:
        msg = f"Spheres about the centre need numeric factors; {labels} are categorical."
        raise ValueError(msg)
    settings = np.abs(design_df[names].to_numpy(dtype=float))
    if not np.all(np.isfinite(settings)):
        msg = "Each factor setting must be a finite number; the design has a missing or infinite one."
        raise ValueError(msg)
    if settings.max() > max(_CODED_LIMIT_FLOOR, 2 ** (len(names) / 4), np.sqrt(len(names))):
        warnings.warn(
            f"variance_dispersion expects factor settings in coded units, centred on 0, but a setting reaches "
            f"|x| = {settings.max():g}. In actual units the spheres are about the wrong centre; pass the "
            "DesignResult, or code the columns first.",
            category=UserWarning,
            stacklevel=3,
        )
    return design_df, names


def _radii(radii: int | Sequence[float], design_radius: float) -> np.ndarray:
    """Return the radii: ``radii`` of them evenly spaced from 0 to the design's radius, or those given."""
    if isinstance(radii, (int, np.integer)) and not isinstance(radii, bool):
        if radii < 1:
            msg = f"radii, as a number of spheres, must be at least 1; got {radii}."
            raise ValueError(msg)
        return np.linspace(0.0, design_radius, int(radii))
    spheres = np.asarray(radii, dtype=float).ravel()
    if spheres.size == 0 or not np.all(np.isfinite(spheres)) or np.any(spheres < 0):
        msg = f"radii must be one or more finite numbers, 0 or more; got {radii!r}."
        raise ValueError(msg)
    return spheres


def _exponents(k: int, degree: int) -> np.ndarray:
    """Return the exponents of every monomial in ``k`` factors of total degree up to ``degree``, a row each."""
    return np.array(
        [
            np.bincount(np.array(factors, dtype=int), minlength=k)
            for total in range(degree + 1)
            for factors in itertools.combinations_with_replacement(range(k), total)
        ],
        dtype=int,
    )


def _monomials(points: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """Return each monomial at each point, a row per point."""
    return np.prod(points[:, None, :] ** exponents[None, :, :], axis=2)


def _monomial_form(
    design_info: DesignInfo, names: list[str], scaled_inverse: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Write the scaled prediction variance as m(x)' Q m(x), with m(x) the monomials of the lowest degree that fits.

    The model's columns are found as combinations of the monomials by least squares at points
    spread over the region; they are polynomials, so the fit is exact once the degree is high
    enough. Returns the monomials' exponents and ``Q``.
    """
    for degree in range(1, MAX_DEGREE + 1):
        exponents = _exponents(len(names), degree)
        points = np.random.default_rng(degree).uniform(-2.0, 2.0, size=(2 * len(exponents) + 10, len(names)))
        terms = np.asarray(build_design_matrices([design_info], pd.DataFrame(points, columns=names))[0])
        basis = _monomials(points, exponents)
        weights = np.linalg.lstsq(basis, terms, rcond=None)[0]
        if np.max(np.abs(basis @ weights - terms)) <= 1e-9 * max(1.0, float(np.max(np.abs(terms)))):
            return exponents, weights @ scaled_inverse @ weights.T
    msg = f"The model's terms are not polynomials of degree {MAX_DEGREE} or less in the factors {names}."
    raise ValueError(msg)


def _sphere_moments(exponents: np.ndarray) -> np.ndarray:
    """Return the average of each monomial (a row of exponents) over the unit sphere in as many dimensions."""
    k = exponents.shape[1]
    double_factorial = np.ones(int(exponents.max()) + 1)  # (n - 1)!! at each even n
    for n in range(2, len(double_factorial)):
        double_factorial[n] = double_factorial[n - 2] * (n - 1)
    half = exponents.sum(axis=1) // 2
    denominators = np.cumprod([1.0] + [k + 2.0 * j for j in range(int(half.max()))])
    moments = np.prod(double_factorial[exponents], axis=1) / denominators[half]
    return np.where(np.any(exponents % 2, axis=1), 0.0, moments)


def _sphere_mean(exponents: np.ndarray, quadratic: np.ndarray) -> Callable[[float], float]:
    """Return the exact average of m(x)' Q m(x) over the sphere of radius r, as a function of r."""
    pairs = (exponents[:, None, :] + exponents[None, :, :]).reshape(-1, exponents.shape[1])
    weights = quadratic.reshape(-1) * _sphere_moments(pairs)
    degrees = pairs.sum(axis=1)

    def mean(radius: float) -> float:
        return float(np.sum(weights * radius**degrees))

    return mean


def _extremes(
    exponents: np.ndarray, quadratic: np.ndarray, directions: np.ndarray, radius: float
) -> tuple[float, float]:
    """Return the minimum and the maximum of m(x)' Q m(x) on the sphere of ``radius``.

    The sampled directions give the starting points, and a local search refines the best of them.
    """

    def variance(points: np.ndarray) -> np.ndarray:
        values = _monomials(points, exponents)
        return np.einsum("ij,jk,ik->i", values, quadratic, values)

    sampled = variance(radius * directions)
    if radius == 0:
        return float(sampled[0]), float(sampled[0])
    lowest = _search(variance, directions[np.argsort(sampled)[:N_STARTS]], radius, 1.0)
    highest = -_search(variance, directions[np.argsort(-sampled)[:N_STARTS]], radius, -1.0)
    return min(lowest, float(sampled.min())), max(highest, float(sampled.max()))


def _search(variance: Callable[[np.ndarray], np.ndarray], starts: np.ndarray, radius: float, sign: float) -> float:
    """Return the least of ``sign`` times the variance that a local search from each start finds on the sphere.

    The search is over a direction ``u`` in all of space, put on the sphere as ``radius * u / |u|``,
    so it needs no constraint.
    """

    def objective(u: np.ndarray) -> float:
        return float(sign * variance(radius * u[None, :] / np.linalg.norm(u))[0])

    return float(min(optimize.minimize(objective, start, method="BFGS").fun for start in starts))
