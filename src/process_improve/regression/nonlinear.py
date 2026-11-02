# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Least squares for a model that is nonlinear in its parameters, with the usual inference.

The model :math:`y = f(x; \theta) + \varepsilon` is fitted by minimizing the residual sum of squares
:math:`S(\theta) = \sum_u (y_u - f(x_u; \theta))^2`. Inference rests on the linearization of
:math:`f` at the estimate :math:`\hat\theta`, the matrix :math:`J` of the derivatives of the fitted
values in the :math:`p` parameters (Box, Hunter and Hunter, section 14.6; Bates and Watts):

- the covariance of the estimates is :math:`s^2 (J^\top J)^{-1}`, with
  :math:`s^2 = S(\hat\theta)/(n - p)`, which gives standard errors, t values and intervals;
- the approximate joint confidence region for all the parameters is
  :math:`\{\theta : S(\theta) \le S(\hat\theta)\,[1 + \tfrac{p}{n - p} F_{p,\,n-p}]\}`. The
  region follows the contours of :math:`S` itself, so it keeps the curved shape of the valley
  that the linearized intervals miss.

Examples
--------
Box, Hunter and Hunter's biochemical oxygen demand against incubation time, with
:math:`\eta = \beta_1 (1 - e^{-\beta_2 x})`:

>>> import numpy as np
>>> days = np.array([1, 2, 3, 5, 7, 10])
>>> demand = np.array([109, 149, 149, 191, 213, 224])
>>> fit = nonlinear_regression(lambda x, b1, b2: b1 * (1 - np.exp(-b2 * x)), days, demand, {"b1": 200, "b2": 0.5})
>>> round(fit["estimates"]["b1"], 1), round(fit["estimates"]["b2"], 4), round(fit["sse"], 2)
(213.8, 0.5472, 1168.01)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats


def nonlinear_regression(
    model: Callable[..., Any],
    x: np.ndarray | pd.DataFrame | Sequence[float],
    y: np.ndarray | Sequence[float],
    start: Mapping[str, float],
    conf_level: float = 0.95,
) -> dict[str, Any]:
    r"""Fit ``y = model(x, *parameters)`` by least squares, and give the inference of the linearized model.

    Parameters
    ----------
    model : Callable[..., Any]
        Called as ``model(x, theta_1, ..., theta_p)``, with the parameters in the order of ``start``;
        returns the fitted value of each row of ``x``.
    x : numpy.ndarray, pandas.DataFrame or Sequence[float]
        The regressors, passed to ``model`` as they are.
    y : numpy.ndarray or Sequence[float]
        The response, one value per row of ``x``.
    start : Mapping[str, float]
        The parameters' names and their starting values, from which the search begins. A
        nonlinear model can have more than one minimum: start near the expected values.
    conf_level : float
        The confidence level of the intervals and of the joint region.

    Returns
    -------
    dict[str, Any]
        - ``estimates``, ``std_errors``, ``t_values``, ``p_values`` (two-sided, of each parameter
          against 0) and ``intervals`` (the estimate plus or minus :math:`t_{n-p}` standard errors),
          each by parameter name;
        - ``correlation``: the correlations of the estimates, a nested dict;
        - ``sse`` (:math:`S(\hat\theta)`), ``df_resid`` (:math:`n - p`), ``s2``, ``n``;
        - ``fitted`` and ``residuals``, in the order of the rows;
        - ``region_sse``: :math:`S(\hat\theta)\,[1 + \tfrac{p}{n - p} F_{p,\,n-p}]`, the contour of
          the residual sum of squares that bounds the joint confidence region.

    Raises
    ------
    ValueError
        If ``start`` names no parameter or gives one that is not a finite number, ``y`` has a value
        that is not a finite number, there are no more rows than parameters, ``model`` does not
        return a finite value for each row at the start, ``conf_level`` is not between 0 and 1,
        the search does not converge, or the estimates cannot be told apart (the derivatives of the
        fitted values in the parameters are linearly dependent at the estimate).
    """
    names, theta0, response = _prepare(start, y, conf_level)

    def residuals(theta: np.ndarray) -> np.ndarray:
        return response - _predict(model, x, theta, len(response))

    residuals(theta0)  # refuses a model that cannot be evaluated at the start
    found = optimize.least_squares(residuals, theta0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    if found.status <= 0:
        msg = f"The search did not converge from the start values {dict(start)}: {found.message}"
        raise ValueError(msg)
    theta = found.x
    jacobian = _jacobian(model, x, theta, len(response))
    n, p = len(response), len(theta)
    rest = residuals(theta)
    sse = float(rest @ rest)
    s2 = sse / (n - p)
    covariance = s2 * _inverse_information(jacobian, names)
    errors = np.sqrt(np.diag(covariance))
    t_values = np.divide(theta, errors, out=np.full(p, np.inf), where=errors > 0)
    half = stats.t.ppf((1 + conf_level) / 2, n - p) * errors
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    correlation = np.divide(covariance, scale, out=np.eye(p), where=scale > 0)
    return {
        "estimates": dict(zip(names, theta.tolist(), strict=True)),
        "std_errors": dict(zip(names, errors.tolist(), strict=True)),
        "t_values": dict(zip(names, t_values.tolist(), strict=True)),
        "p_values": dict(zip(names, (2 * stats.t.sf(np.abs(t_values), n - p)).tolist(), strict=True)),
        "intervals": {name: (v - h, v + h) for name, v, h in zip(names, theta.tolist(), half.tolist(), strict=True)},
        "correlation": {
            a: dict(zip(names, row.tolist(), strict=True)) for a, row in zip(names, correlation, strict=True)
        },
        "sse": sse,
        "df_resid": n - p,
        "s2": s2,
        "n": n,
        "fitted": (response - rest).tolist(),
        "residuals": rest.tolist(),
        "region_sse": sse * (1 + p / (n - p) * stats.f.ppf(conf_level, p, n - p)),
    }


def _prepare(
    start: Mapping[str, float], y: np.ndarray | Sequence[float], conf_level: float
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return the parameters' names, their starting values and the response, or refuse them."""
    if not isinstance(start, Mapping) or not start:
        msg = f"start must map each parameter's name to its starting value, one parameter at least; got {start!r}."
        raise ValueError(msg)
    names = [str(name) for name in start]
    theta0 = np.array([float(value) for value in start.values()])
    if not np.all(np.isfinite(theta0)):
        msg = f"Each starting value must be a finite number; got {dict(start)}."
        raise ValueError(msg)
    response = np.asarray(y, dtype=float)
    if response.ndim != 1 or not np.all(np.isfinite(response)):
        msg = "y must be a sequence of finite numbers, one per row of x."
        raise ValueError(msg)
    if len(response) <= len(names):
        msg = f"{len(response)} rows leave no residual degree of freedom for {len(names)} parameters."
        raise ValueError(msg)
    if not 0 < conf_level < 1:
        msg = f"conf_level must be between 0 and 1; got {conf_level!r}."
        raise ValueError(msg)
    return names, theta0, response


def _predict(model: Callable[..., Any], x: Any, theta: np.ndarray, n: int) -> np.ndarray:  # noqa: ANN401 - x is the caller's, passed through
    """Return the model's fitted values at ``theta``, refusing any that are missing or not finite."""
    fitted = np.asarray(model(x, *theta), dtype=float).ravel()
    if fitted.shape != (n,) or not np.all(np.isfinite(fitted)):
        msg = (
            f"model must return a finite value for each of the {n} rows; at the parameters {theta.tolist()} "
            f"it returned {fitted.size} value(s), {int(np.sum(~np.isfinite(fitted)))} of them not finite."
        )
        raise ValueError(msg)
    return fitted


def _jacobian(model: Callable[..., Any], x: Any, theta: np.ndarray, n: int) -> np.ndarray:  # noqa: ANN401 - x is the caller's, passed through
    """Return the derivatives of the fitted values in each parameter at ``theta``, by central differences.

    The step for each parameter is the cube root of the machine epsilon times its size (1 for a
    parameter at 0), the step that balances the truncation and the rounding of a central difference.
    """
    steps = np.cbrt(np.finfo(float).eps) * np.maximum(np.abs(theta), 1.0)
    columns = []
    for j, step in enumerate(steps):
        shift = np.zeros_like(theta)
        shift[j] = step
        columns.append((_predict(model, x, theta + shift, n) - _predict(model, x, theta - shift, n)) / (2 * step))
    return np.column_stack(columns)


def _inverse_information(jacobian: np.ndarray, names: list[str]) -> np.ndarray:
    r"""Return :math:`(J^\top J)^{-1}`, refusing a Jacobian whose columns are linearly dependent.

    The columns are scaled to unit length first, so that parameters on different scales are not
    taken for dependent ones. Central differences are accurate to about :math:`\epsilon^{2/3}`,
    so columns that depend on each other exactly leave a smallest singular value of about that
    size: below :math:`\sqrt{\epsilon}` of the largest they are taken to be dependent.
    """
    lengths = np.linalg.norm(jacobian, axis=0)
    _, singular, rotation = np.linalg.svd(jacobian / np.where(lengths > 0, lengths, 1.0), full_matrices=False)
    if np.any(lengths == 0) or singular[-1] <= np.sqrt(np.finfo(float).eps) * singular[0]:
        msg = (
            f"The parameters {names} cannot be told apart at the estimate: the derivatives of the fitted "
            "values in them are linearly dependent. Fix one of them, or rewrite the model."
        )
        raise ValueError(msg)
    return (rotation.T / singular**2) @ rotation / np.outer(lengths, lengths)
