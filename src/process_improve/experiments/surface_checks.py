# (c) Kevin Dunn, 2010-2026. MIT License.

"""Checks on a fitted response surface before it is read (#797).

A second-order surface fitted to a composite design is read for its optimum. Three questions come
first: does the quadratic fail along an axis, how large is the noise the surface is read against,
and how well is its stationary point located?

- :func:`third_order_contrasts` compares, for each factor, the slope its axial runs give with the
  slope its factorial runs give, which agree under a second-order model.
- :func:`pooled_pure_error` pools the replicated runs of every design of one investigation, each
  set of replicates compared with its own mean.
- :func:`stationary_point_region` gives the confidence region of the stationary point: the
  settings at which the fitted surface's slope does not differ significantly from zero.

Whether the fitted values change by enough to read at all is ``analyze_experiment``'s
``model_summary["adequate_precision"]``, with a note in ``model_note`` when it is below 4.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from process_improve.experiments._analyses._shared import BLOCK_COL
from process_improve.experiments._analyses.lack_of_fit import pure_error
from process_improve.experiments.optimization import _FLAT_EIGENVALUE_TOL, _eigen_by_size, _surface_shape


def third_order_contrasts(
    design: pd.DataFrame,
    response_col: str,
    factor_cols: Sequence[str],
    sigma: float | None = None,
) -> dict[str, Any]:
    """Return, for each factor of a composite design, the slope its axial runs give less the slope its cube gives.

    A composite design measures each factor's slope twice: from its axial runs, at plus and minus
    alpha with the other factors at their centres, and from the runs of its factorial cube. Under
    a second-order model the two agree. Their difference estimates (alpha^2 - 1) b_iii less the sum
    of b_ijj over the other factors j, which is b_iii - b_ijj for two factors at alpha = sqrt(2), as
    Box, Hunter and Hunter (15.20, 15.21) write it. A contrast many standard errors from zero says
    the quadratic fails along that axis, which the lack-of-fit F test, pooling every axis, does not.
    Each slope is a difference within one set of runs, so a block effect between the cube and the
    axial runs cancels.

    Parameters
    ----------
    design : DataFrame
        The runs: the factors in coded units (-1 and +1 at the cube's levels), and the response.
    response_col : str
        The response column.
    factor_cols : sequence of str
        The factors, two or more.
    sigma : float or None
        The standard deviation of one run, when it is known beforehand (Box, Hunter and Hunter
        take 1.5). By default it is the pure error of the replicated runs, each set compared
        with its own mean, within blocks when the design has a ``Block`` column.

    Returns
    -------
    dict
        ``contrasts``: for each factor, its ``axial_slope``, ``factorial_slope``, their difference
        ``contrast``, and the ``std_error`` of that difference. ``sigma`` is the standard deviation
        used, and ``sigma_df`` its degrees of freedom when it comes from the runs (``None`` for a
        given ``sigma``).

    Raises
    ------
    ValueError
        If there are fewer than two factors, no runs of a factorial cube, a factor without an
        axial run on each side of the centre, or no replicated runs and no ``sigma``.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments import third_order_contrasts
    >>> a = 2 ** 0.5
    >>> runs = pd.DataFrame(
    ...     [(-1, -1, 78.8), (1, -1, 84.5), (-1, 1, 91.2), (1, 1, 77.4), (-a, 0, 83.3), (a, 0, 81.2),
    ...      (0, -a, 81.2), (0, a, 79.5), (0, 0, 87.0), (0, 0, 86.0)],
    ...     columns=["time", "temperature", "y"],
    ... )
    >>> out = third_order_contrasts(runs, "y", ["time", "temperature"], sigma=1.5)
    >>> {name: round(c["contrast"], 2) for name, c in out["contrasts"].items()}
    {'time': 1.28, 'temperature': -1.93}
    >>> round(out["contrasts"]["time"]["std_error"], 2)
    1.06
    """
    names = list(factor_cols)
    if len(names) < 2:
        raise ValueError(f"A composite design has two factors or more; got {names}.")
    x = design[names].to_numpy(dtype=float)
    y = design[response_col].to_numpy(dtype=float)
    cube = np.all(np.isclose(np.abs(x), 1.0), axis=1)
    if cube.sum() < 2:
        raise ValueError("The design has no factorial cube: no two runs with every factor at -1 or +1.")
    scale, scale_df = _run_sigma(design, response_col, names, sigma)
    contrasts = {}
    for i, name in enumerate(names):
        alone = ~np.isclose(x[:, i], 0.0) & np.all(np.isclose(np.delete(x, i, axis=1), 0.0), axis=1)
        plus, minus = alone & (x[:, i] > 0), alone & (x[:, i] < 0)
        if not (plus.any() and minus.any()):
            raise ValueError(f"{name!r} needs an axial run on each side of the centre, the other factors at 0.")
        reach = x[plus, i].mean() - x[minus, i].mean()
        axial = (y[plus].mean() - y[minus].mean()) / reach
        factorial = float(y[cube] @ x[cube, i] / cube.sum())
        variance = (1 / plus.sum() + 1 / minus.sum()) / reach**2 + 1 / cube.sum()
        contrasts[name] = {
            "axial_slope": float(axial),
            "factorial_slope": factorial,
            "contrast": float(axial - factorial),
            "std_error": scale * math.sqrt(variance),
        }
    return {"contrasts": contrasts, "sigma": scale, "sigma_df": scale_df}


def pooled_pure_error(designs: Sequence[pd.DataFrame], response_col: str, factor_cols: Sequence[str]) -> dict[str, Any]:
    """Return the pure error pooled over the designs of one investigation (#797 R3).

    Each design's replicated runs are compared with their own mean, and the sums of squares and
    the degrees of freedom are added over the designs. Runs of two designs at the same coded
    setting are not replicates of each other, since the design's centre moves from one to the
    next. Box, Hunter and Hunter (15.16, 15.24) pool the centre runs of three designs this way. A
    blocked design whose blocks may differ goes in as one design per block.

    Parameters
    ----------
    designs : sequence of DataFrame
        The designs, each with its factor columns and the response.
    response_col : str
        The response column.
    factor_cols : sequence of str
        The factors whose settings say which runs are replicates.

    Returns
    -------
    dict
        ``ss_pure_error``, ``df_pure_error``, ``ms_pure_error`` (the pooled variance) and
        ``std_dev``, its square root, with ``per_design``, each design's ``ss`` and ``df``.

    Raises
    ------
    ValueError
        If no design is given, or none has replicated runs.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments import pooled_pure_error
    >>> first = pd.DataFrame({"x": [0, 0, 0], "y": [60.3, 64.3, 62.3]})
    >>> later = pd.DataFrame({"x": [0, 0], "y": [89.7, 86.8]})
    >>> round(pooled_pure_error([first, later], "y", ["x"])["ms_pure_error"], 2)
    4.07
    """
    if not designs:
        raise ValueError("Give at least one design whose replicated runs to pool.")
    parts = [pure_error(design, response_col, list(factor_cols)) for design in designs]
    ss, df = sum(ss for ss, _ in parts), sum(df for _, df in parts)
    if df == 0:
        raise ValueError("No design has replicated runs, so there is no pure error to pool.")
    return {
        "ss_pure_error": ss,
        "df_pure_error": df,
        "ms_pure_error": ss / df,
        "std_dev": math.sqrt(ss / df),
        "per_design": [{"ss": ss, "df": df} for ss, df in parts],
    }


def stationary_point_region(
    design: pd.DataFrame,
    response_col: str,
    factor_cols: Sequence[str],
    sigma: float | None = None,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """Return the confidence region of the stationary point of the second-order surface fitted to the runs (#797 R5).

    At the true stationary point the true surface's slope in every direction, its gradient
    b + 2Bx, is zero. The fitted surface's gradient at any setting x is linear in the estimated
    coefficients, so its covariance follows from theirs, and an F test on k and n - p degrees of
    freedom asks whether it differs from zero there. The settings at which it does not, at level
    ``significance_level``, form a 1 - ``significance_level`` confidence region for the stationary
    point (Box and Hunter, 1954, "A confidence region for the solution of a set of simultaneous
    equations with an application to experimental design", *Biometrika* 41, 190-199). The region
    need not be an ellipse around the estimate, nor bounded: along a ridge it runs out of the
    experiment. With one factor it is Fieller's interval for -b1 / (2 b11).

    The full second-order model is fitted to the factors as they are given, with a fixed effect
    for each block when the design has a ``Block`` column; the region does not depend on how the
    factors are coded. Beside it, ``std_error`` gives the delta-method standard error of each
    coordinate of the estimated stationary point x_s = -B^-1 b / 2, whose derivative with respect
    to the coefficients is -(2B)^-1 times the gradient's: a summary that holds near x_s when the
    region is small and nearly elliptical.

    Parameters
    ----------
    design : DataFrame
        The runs: the factors and the response, and optionally a ``Block`` column.
    response_col : str
        The response column.
    factor_cols : sequence of str
        The factors of the surface, one or more.
    sigma : float or None
        The standard deviation of one run, when it is known beforehand. The test is then on the
        chi-square distribution with k degrees of freedom, scaled to the F statistic's units. By
        default it is the residual standard deviation of the fit.
    significance_level : float
        Default 0.05, for a 95% region.

    Returns
    -------
    dict
        ``stationary_point``, the estimate in the units of the factors as given (``None`` when the
        fitted surface has none: a rising ridge), ``classification`` (``"maximum"``,
        ``"minimum"``, ``"saddle_point"``, ``"stationary_ridge"`` or ``"rising_ridge"``, as
        :func:`~process_improve.experiments.optimize_responses` reads the eigenvalues of B), and
        ``std_error``, each coordinate's delta-method standard error (``None`` for a ridge, along
        which the point is not determined). ``statistic`` is a function of settings returning the
        F statistic of the gradient there, and ``contains`` one returning whether each setting is
        in the region, where the statistic is at most ``critical_value``. Both take a DataFrame or
        a mapping of the factors' names to values or arrays (a grid from ``numpy.meshgrid``, say),
        or an array whose last axis holds the factors in order, and return an array of the shape
        of the settings. ``confidence_level`` is 1 - ``significance_level``; ``sigma`` is the
        standard deviation used, and ``sigma_df`` its degrees of freedom when it comes from the
        runs (``None`` for a given ``sigma``).

    Raises
    ------
    ValueError
        If there is no factor, a run has a missing value, the runs cannot estimate every term of
        the second-order model, there is no residual and no ``sigma``, or ``sigma`` or
        ``significance_level`` is out of range.

    Examples
    --------
    Box, Hunter and Hunter's composite design (Table 15.4), at alpha = sqrt(2). The fitted
    surface is a maximum whose smaller eigenvalue is 3% of the larger, nearly a ridge, so its
    stationary point lies far outside the runs and is poorly located: the centre of the design
    is inside the 95% region, while (1, 0) is not.

    >>> import pandas as pd
    >>> from process_improve.experiments import stationary_point_region
    >>> a = 2 ** 0.5
    >>> runs = pd.DataFrame(
    ...     [(-1, -1, 78.8), (1, -1, 84.5), (-1, 1, 91.2), (1, 1, 77.4), (0, 0, 89.7), (0, 0, 86.8),
    ...      (-a, 0, 83.3), (a, 0, 81.2), (0, -a, 81.2), (0, a, 79.5), (0, 0, 87.0), (0, 0, 86.0)],
    ...     columns=["time", "temperature", "y"],
    ... )
    >>> region = stationary_point_region(runs, "y", ["time", "temperature"])
    >>> region["classification"]
    'maximum'
    >>> {name: round(value, 1) for name, value in region["stationary_point"].items()}
    {'time': -3.7, 'temperature': 3.0}
    >>> {name: round(value) for name, value in region["std_error"].items()}
    {'time': 22, 'temperature': 18}
    >>> region["contains"]({"time": [0.0, 1.0], "temperature": [0.0, 0.0]}).tolist()
    [True, False]
    """
    names = list(factor_cols)
    k = len(names)
    if not names:
        raise ValueError("Name at least one factor of the surface.")
    if not 0 < significance_level < 1:
        raise ValueError(f"significance_level must be between 0 and 1; got {significance_level!r}.")
    sigma = None if sigma is None else _positive(sigma)
    blocked = BLOCK_COL in design.columns
    used = [*names, response_col, *([BLOCK_COL] if blocked else [])]
    if design[used].isna().to_numpy().any():
        raise ValueError(f"A run has a missing value in {used}; leave it out, or fill it in.")
    x = design[names].to_numpy(dtype=float)
    pairs = [(i, j) for i in range(k) for j in range(i, k)]
    model = np.column_stack([np.ones(len(x)), x, *(x[:, i] * x[:, j] for i, j in pairs)])
    surface_terms = model.shape[1]
    if blocked:  # one column for each block after the first; none for a single block
        model = np.column_stack([model, pd.get_dummies(design[BLOCK_COL], drop_first=True, dtype=float)])
    if np.linalg.matrix_rank(model) < model.shape[1]:
        raise ValueError(
            f"The runs cannot estimate every term of the second-order model in {names}: it has {model.shape[1]} "
            f"terms and {len(x)} runs, and each factor needs three levels or more."
        )
    unscaled = np.linalg.inv(model.T @ model)
    y = design[response_col].to_numpy(dtype=float)
    coefficients = unscaled @ model.T @ y
    scale, scale_df = _region_scale(y - model @ coefficients, model.shape[1], sigma)
    level = 1 - significance_level
    # with a known sigma the statistic times k is chi-square on k df; put its critical value on the F scale
    critical = float(stats.chi2.ppf(level, k) / k if scale_df is None else stats.f.ppf(level, k, scale_df))
    theta = coefficients[:surface_terms]
    covariance = scale**2 * unscaled[:surface_terms, :surface_terms]

    def statistic(points: Any) -> np.ndarray:  # noqa: ANN401 - a DataFrame, mapping or array of settings
        z = _settings(points, names)
        rows = _gradient_rows(z.reshape(-1, k), pairs)
        gradient = rows @ theta
        spread = rows @ covariance @ rows.transpose(0, 2, 1)
        found = np.einsum("mi,mi->m", gradient, np.linalg.solve(spread, gradient[..., None])[..., 0])
        return (found / k).reshape(z.shape[:-1])

    point, shape = _stationary_point(theta, pairs, k)
    return {
        "stationary_point": None if point is None else dict(zip(names, map(float, point), strict=True)),
        "classification": shape,
        "std_error": _delta_std_errors(point, theta, covariance, pairs, names),
        "statistic": statistic,
        "contains": lambda points: statistic(points) <= critical,
        "critical_value": critical,
        "confidence_level": level,
        "sigma": scale,
        "sigma_df": scale_df,
    }


def _region_scale(residual: np.ndarray, n_terms: int, sigma: float | None) -> tuple[float, int | None]:
    """Return the standard deviation of one run, given or from the residual, and its degrees of freedom."""
    if sigma is not None:
        return sigma, None
    df = len(residual) - n_terms
    if df == 0:
        raise ValueError("The fit leaves no residual to estimate the noise from: add runs, or give sigma.")
    return math.sqrt(float(residual @ residual) / df), df


def _settings(points: Any, names: list[str]) -> np.ndarray:  # noqa: ANN401 - a DataFrame, mapping or array
    """Return settings as an array whose last axis holds the factors in the order of ``names``."""
    if isinstance(points, (Mapping, pd.DataFrame)):
        return np.stack(np.broadcast_arrays(*(np.asarray(points[name], dtype=float) for name in names)), axis=-1)
    z = np.asarray(points, dtype=float)
    if z.ndim == 0 or z.shape[-1] != len(names):
        raise ValueError(f"Give each setting's {len(names)} values in the last axis, in the order {names}.")
    return z


def _gradient_rows(z: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Return, at each setting, the matrix that turns the second-order coefficients into the surface's gradient.

    The coefficients are the intercept, the k linear terms, and the products x_i x_j (i <= j) in the order of
    ``pairs``; row l of each matrix is the derivative of the model's terms with respect to x_l.
    """
    m, k = z.shape
    rows = np.zeros((m, k, 1 + k + len(pairs)))
    rows[:, np.arange(k), 1 + np.arange(k)] = 1.0
    for column, (i, j) in enumerate(pairs, start=1 + k):
        rows[:, i, column] += z[:, j]
        rows[:, j, column] += z[:, i]  # twice x_i for a square, i == j
    return rows


def _quadratic_matrix(theta: np.ndarray, pairs: list[tuple[int, int]], k: int) -> np.ndarray:
    """Return the symmetric B of the surface b0 + b'x + x'Bx from its coefficients, as optimize_responses forms it."""
    quadratic = np.zeros((k, k))
    for coefficient, (i, j) in zip(theta[1 + k :], pairs, strict=True):
        quadratic[i, j] = quadratic[j, i] = coefficient if i == j else coefficient / 2
    return quadratic


def _stationary_point(theta: np.ndarray, pairs: list[tuple[int, int]], k: int) -> tuple[np.ndarray | None, str]:
    """Return the fitted surface's stationary point and its classification, read as optimize_responses reads them.

    Along a stationary ridge the point is the one nearest the centre; a rising ridge has none.
    """
    b, quadratic = theta[1 : 1 + k], _quadratic_matrix(theta, pairs, k)
    eigenvalues, eigenvectors = _eigen_by_size(quadratic)
    shape = _surface_shape(b, eigenvalues, eigenvectors)
    if shape.drifts:
        return None, shape.classification
    curved = eigenvectors[:, ~shape.flat]
    return curved @ (-0.5 * (curved.T @ b) / eigenvalues[~shape.flat]), shape.classification


def _delta_std_errors(
    point: np.ndarray | None,
    theta: np.ndarray,
    covariance: np.ndarray,
    pairs: list[tuple[int, int]],
    names: list[str],
) -> dict[str, float] | None:
    """Return the delta-method standard error of each coordinate of the stationary point; None along a ridge."""
    if point is None:
        return None
    quadratic = _quadratic_matrix(theta, pairs, len(names))
    eigenvalues = np.linalg.eigvalsh(quadratic)
    if np.abs(eigenvalues).min() <= _FLAT_EIGENVALUE_TOL * np.abs(eigenvalues).max():
        return None  # a ridge: the point moves freely along it
    jacobian = -np.linalg.solve(2 * quadratic, _gradient_rows(point[None, :], pairs)[0])
    return dict(zip(names, map(float, np.sqrt(np.diag(jacobian @ covariance @ jacobian.T))), strict=True))


def _positive(sigma: float) -> float:
    """Return ``sigma`` as a float, refusing anything but a positive finite number."""
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma must be a positive number; got {sigma!r}.")
    return float(sigma)


def _run_sigma(
    design: pd.DataFrame, response_col: str, names: list[str], sigma: float | None
) -> tuple[float, int | None]:
    """Return the standard deviation of one run, given or from the replicated runs, and its degrees of freedom."""
    if sigma is not None:
        return _positive(sigma), None
    groups = [*names, BLOCK_COL] if BLOCK_COL in design.columns else names
    ss, df = pure_error(design, response_col, groups)
    if df == 0:
        raise ValueError("The design has no replicated runs to estimate the noise from: give sigma.")
    return math.sqrt(ss / df), df
