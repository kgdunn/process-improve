# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Box and Tidwell's estimate of the power of a regressor.

A response that is curved in a regressor :math:`x` can be straight in a power of it, :math:`x^\alpha`,
with :math:`\ln x` in the place of :math:`\alpha = 0`. Box and Tidwell (*Technometrics*, 1962) estimate
the powers of the model

.. math::

    y = \beta_0 + \sum_j \beta_j x_j^{\alpha_j} + (\text{regressors that enter as they are}) + \varepsilon

with ordinary regressions only. Near a trial power :math:`\alpha`,

.. math::

    x^{\alpha'} \approx x^{\alpha} + (\alpha' - \alpha)\, x^{\alpha} \ln x,

so adding the *constructed variable* :math:`x^{\alpha}\ln x` to a regression on :math:`x^{\alpha}` gives
it the coefficient :math:`\gamma = \beta(\alpha' - \alpha)`. Each step:

1. regresses :math:`y` on the regressors at the trial powers, which gives each :math:`\hat\beta`;
2. adds the constructed variables, which gives each :math:`\hat\gamma`;
3. moves each power to :math:`\alpha + \hat\gamma / \hat\beta`.

From :math:`\alpha = 1`, the regressor as it is, the first step is Box and Tidwell's
:math:`\alpha_1 = 1 + \hat\gamma/\hat\beta`. Repeated, the steps are the Gauss-Newton method for
the least-squares estimate of the powers; here a step is halved whenever it would raise the
residual sum of squares. The standard error of each power comes from the same linearization at the
estimate, as for any nonlinear least-squares parameter.

The constructed variable's t value at :math:`\alpha = 1` tests whether the regressor needs a power
other than 1 at all: it is the score test of a straight-line relation in that regressor.

Examples
--------
A response that is straight in :math:`\sqrt{x}`:

>>> import numpy as np
>>> x = np.arange(1.0, 11.0)
>>> found = box_tidwell(x, 2 + 3 * np.sqrt(x))
>>> round(found["power"]["x"], 6), found["df_resid"]
(0.5, 7)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

#: Below this size a power's columns are summed from their series in it, exact to rounding there.
SERIES = 1e-6

#: The most times a step is halved in search of a lower residual sum of squares.
HALVINGS = 40

#: A regressor whose straight-line part of the fit is below this share of the spread of y has no
#: slope at its trial power: its power cannot be moved, whatever rounding leaves in its coefficient.
FLAT = float(np.sqrt(np.finfo(float).eps))


def box_tidwell(
    X: pd.DataFrame | np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    columns: Sequence[str] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict[str, Any]:
    r"""Estimate the power of one or more regressors by Box and Tidwell's method.

    Parameters
    ----------
    X : pandas.DataFrame, numpy.ndarray or Sequence[float]
        The regressors, a column each. A one-dimensional array is one regressor, named ``"x"``;
        the columns of a two-dimensional array are named ``"x1"``, ``"x2"``, and so on.
    y : numpy.ndarray or Sequence[float]
        The response.
    columns : Sequence[str] or None
        The regressors whose power is estimated; each must be positive and take three or more
        different values. None estimates a power for every column. The other columns enter the
        model as they are, beside the intercept.
    max_iter : int
        The most steps to take.
    tol : float
        The steps stop when no power moves by more than this.

    Returns
    -------
    dict[str, Any]
        - ``power``: the least-squares estimate of each power, by column name;
        - ``std_error``: its standard error, :math:`s` times the square root of the power's
          diagonal element of :math:`(J^\top J)^{-1}`, with :math:`J` the derivatives of the
          fitted values in every parameter at the estimate;
        - ``linearity_t`` and ``linearity_p_value``: the t value of each constructed variable in
          the regression with every power at 1, and its two-sided p value; a small p value says
          the relation is not a straight line in that regressor;
        - ``sse``, the residual sum of squares at the estimate, and ``sse_linear``, with every
          power at 1;
        - ``df_resid``, the residual degrees of freedom: the rows less the intercept, the
          regressors' coefficients and the powers; ``n``, the rows;
        - ``iterations``, the powers at the start (all 1) and after each step: the first step,
          unless it had to be halved, is Box and Tidwell's :math:`\alpha_1`.

    Raises
    ------
    ValueError
        If the data or the arguments cannot give a power: a name that is not a column, a column
        to transform with a value that is not positive or fewer than three different values, a
        value that is not a finite number, a response of another length or with no variation,
        too few rows to leave a residual degree of freedom, a regressor with no clear
        straight-line effect on the response, or powers that do not settle in ``max_iter`` steps.
        Powers that keep growing in size mean that no finite power fits best: the fit improves
        the more a regressor is reduced to its largest or smallest value.
    """
    frame, response, names = _prepare(X, y, columns)
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        msg = f"max_iter must be a whole number of at least 1; got {max_iter!r}."
        raise ValueError(msg)
    if not tol > 0:
        msg = f"tol must be a positive number; got {tol!r}."
        raise ValueError(msg)
    x = frame[names].to_numpy(dtype=float)
    as_is = [column for column in frame.columns if column not in names]
    linear = np.column_stack([np.ones(len(response)), frame[as_is].to_numpy(dtype=float)])
    df_resid = len(response) - linear.shape[1] - 2 * len(names)

    powers, history = _iterate(response, linear, x, max_iter, tol)
    t_values, p_values = _linearity(response, linear, x)
    return {
        "power": dict(zip(names, powers.tolist(), strict=True)),
        "std_error": dict(zip(names, _std_errors(response, linear, x, powers, df_resid).tolist(), strict=True)),
        "linearity_t": dict(zip(names, t_values.tolist(), strict=True)),
        "linearity_p_value": dict(zip(names, p_values.tolist(), strict=True)),
        "sse": _sse(response, linear, x, powers),
        "sse_linear": _sse(response, linear, x, np.ones(len(names))),
        "df_resid": df_resid,
        "n": len(response),
        "iterations": [dict(zip(names, step.tolist(), strict=True)) for step in history],
    }


def _prepare(
    X: pd.DataFrame | np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    columns: Sequence[str] | None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return the regressors as a frame, the response, and the names to transform, or refuse them."""
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        values = np.asarray(X, dtype=float)
        names = ["x"] if values.ndim == 1 else [f"x{j + 1}" for j in range(values.shape[1])]
        frame = pd.DataFrame(values.reshape(len(values), -1), columns=names)
    frame.columns = [str(column) for column in frame.columns]
    response = np.asarray(y, dtype=float)
    if response.ndim != 1 or len(response) != len(frame):
        msg = f"y must have one value per row of X ({len(frame)}); got shape {response.shape}."
        raise ValueError(msg)
    if not (np.all(np.isfinite(frame.to_numpy(dtype=float))) and np.all(np.isfinite(response))):
        msg = "Each value of X and y must be a finite number."
        raise ValueError(msg)
    if np.ptp(response) == 0:
        msg = "y has the same value in every row, so no regressor has an effect on it to transform."
        raise ValueError(msg)
    names = list(frame.columns) if columns is None else [str(name) for name in columns]
    if not names or len(set(names)) != len(names) or not set(names) <= set(frame.columns):
        msg = f"columns must name one or more different columns of X, {list(frame.columns)}; got {columns!r}."
        raise ValueError(msg)
    for name in names:
        values = frame[name].to_numpy(dtype=float)
        if np.any(values <= 0) or len(np.unique(values)) < 3:  # through two values, every power fits
            msg = (
                f"Column {name!r} must be positive, for its power and its logarithm, and take three or "
                "more different values: through two, every power fits equally well."
            )
            raise ValueError(msg)
    if len(response) - (len(frame.columns) + 1) - len(names) < 1:
        msg = (
            f"{len(response)} rows leave no residual degree of freedom for an intercept, "
            f"{len(frame.columns)} coefficients and {len(names)} powers."
        )
        raise ValueError(msg)
    return frame, response, names


def _power_columns(x: np.ndarray, powers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Return each regressor at its power, and the derivative of that column in the power.

    The column is :math:`(x^\alpha - 1)/\alpha`, :math:`\ln x` at :math:`\alpha = 0`. With an
    intercept in the model it fits exactly as :math:`x^\alpha` does, and it stays well conditioned
    as :math:`\alpha` nears 0, where :math:`x^\alpha` nears the intercept's column of ones.
    """
    columns, derivatives = [], []
    for log_x, power in zip(np.log(x).T, powers, strict=True):
        if abs(power) < SERIES:
            columns.append(log_x + power * log_x**2 / 2 + power**2 * log_x**3 / 6)
            derivatives.append(log_x**2 / 2 + power * log_x**3 / 3 + power**2 * log_x**4 / 8)
        else:
            z = power * log_x
            with np.errstate(over="ignore", invalid="ignore"):  # a trial power too large: _sse rejects it
                columns.append(np.expm1(z) / power)
                derivatives.append((z * np.exp(z) - np.expm1(z)) / power**2)
    return np.column_stack(columns), np.column_stack(derivatives)


def _fit(model: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the least-squares coefficients of ``y`` on the columns of ``model``, and the residuals."""
    coefficients = np.linalg.lstsq(model, y, rcond=None)[0]
    return coefficients, y - model @ coefficients


def _sse(y: np.ndarray, linear: np.ndarray, x: np.ndarray, powers: np.ndarray) -> float:
    """Return the residual sum of squares with the regressors at ``powers``; infinite where a column overflows."""
    columns = _power_columns(x, powers)[0]
    if not np.all(np.isfinite(columns)):
        return np.inf
    try:
        _, residuals = _fit(np.column_stack([linear, columns]), y)
    except np.linalg.LinAlgError:  # columns finite but too large for the decomposition: as good as infinite
        return np.inf
    total = float(residuals @ residuals)
    return total if np.isfinite(total) else np.inf


def _step(y: np.ndarray, linear: np.ndarray, x: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Return Box and Tidwell's step for each power, gamma / beta, from two regressions at the trial powers."""
    columns, derivatives = _power_columns(x, powers)
    model = np.column_stack([linear, columns])
    beta = _fit(model, y)[0][-len(powers) :]
    gamma = _fit(np.column_stack([model, derivatives]), y)[0][-len(powers) :]
    effect = np.abs(beta) * np.linalg.norm(columns - columns.mean(axis=0), axis=0)
    if np.any(effect <= FLAT * np.linalg.norm(y - y.mean())):
        msg = "A regressor has no straight-line effect on y at a trial power, so its power cannot be moved."
        raise ValueError(msg)
    return gamma / beta


def _iterate(
    y: np.ndarray, linear: np.ndarray, x: np.ndarray, max_iter: int, tol: float
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Step from every power at 1 until no power moves by more than ``tol``; return the powers and their path."""
    powers = np.ones(x.shape[1])
    history = [powers]
    sse = _sse(y, linear, x, powers)
    for _ in range(max_iter):
        step = _step(y, linear, x, powers)
        trial_sse = _sse(y, linear, x, powers + step)
        for _halving in range(HALVINGS):
            if trial_sse <= sse:
                break
            step = step / 2
            trial_sse = _sse(y, linear, x, powers + step)
        if trial_sse > sse:  # no part of the step lowers the sum of squares
            if np.max(np.abs(step)) <= tol:  # the powers are at its minimum, to rounding
                return powers, history
            msg = (
                f"The powers cannot be estimated: no step from {np.round(powers, 6).tolist()} lowers the residual "
                "sum of squares. A regressor with no clear straight-line effect on y has no power to estimate."
            )
            raise ValueError(msg)
        powers, sse = powers + step, trial_sse
        history.append(powers)
        if np.max(np.abs(step)) <= tol:
            return powers, history
    msg = (
        f"The powers did not settle in max_iter={max_iter} steps; the last were {np.round(powers, 6).tolist()}. "
        "Powers that keep growing in size mean no finite power fits best; otherwise pass a larger max_iter."
    )
    raise ValueError(msg)


def _std_errors(y: np.ndarray, linear: np.ndarray, x: np.ndarray, powers: np.ndarray, df_resid: int) -> np.ndarray:
    """Return the standard error of each power, from the derivatives of the fitted values at the estimate."""
    columns, derivatives = _power_columns(x, powers)
    model = np.column_stack([linear, columns])
    coefficients, residuals = _fit(model, y)
    jacobian = np.column_stack([model, derivatives * coefficients[-len(powers) :]])
    _, singular, rotation = np.linalg.svd(jacobian, full_matrices=False)
    if singular[-1] <= singular[0] * max(jacobian.shape) * np.finfo(float).eps:
        msg = "The powers cannot be estimated: their derivative columns are linearly dependent at the estimate."
        raise ValueError(msg)
    covariance = (rotation.T / singular**2) @ rotation * float(residuals @ residuals) / df_resid
    return np.sqrt(np.diag(covariance)[-len(powers) :])


def _linearity(y: np.ndarray, linear: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the t value and two-sided p value of each constructed variable, every power at 1."""
    columns, derivatives = _power_columns(x, np.ones(x.shape[1]))
    model = np.column_stack([linear, columns, derivatives])
    coefficients, residuals = _fit(model, y)
    df = len(y) - model.shape[1]
    covariance = np.linalg.pinv(model.T @ model) * float(residuals @ residuals) / df
    t_values = coefficients[-x.shape[1] :] / np.sqrt(np.diag(covariance)[-x.shape[1] :])
    return t_values, 2 * stats.t.sf(np.abs(t_values), df)
