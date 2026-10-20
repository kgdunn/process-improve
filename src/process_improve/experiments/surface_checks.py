# (c) Kevin Dunn, 2010-2026. MIT License.

"""Checks on a fitted response surface before it is read (#797).

A second-order surface fitted to a composite design is read for its optimum. Two questions come
first: does the quadratic fail along an axis, and how large is the noise the surface is read against?

- :func:`third_order_contrasts` compares, for each factor, the slope its axial runs give with the
  slope its factorial runs give, which agree under a second-order model.
- :func:`pooled_pure_error` pools the replicated runs of every design of one investigation, each
  set of replicates compared with its own mean.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from process_improve.experiments._analyses._shared import BLOCK_COL
from process_improve.experiments._analyses.lack_of_fit import pure_error


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


def _run_sigma(
    design: pd.DataFrame, response_col: str, names: list[str], sigma: float | None
) -> tuple[float, int | None]:
    """Return the standard deviation of one run, given or from the replicated runs, and its degrees of freedom."""
    if sigma is not None:
        if not (math.isfinite(sigma) and sigma > 0):
            raise ValueError(f"sigma must be a positive number; got {sigma!r}.")
        return float(sigma), None
    groups = [*names, BLOCK_COL] if BLOCK_COL in design.columns else names
    ss, df = pure_error(design, response_col, groups)
    if df == 0:
        raise ValueError("The design has no replicated runs to estimate the noise from: give sigma.")
    return math.sqrt(ss / df), df
