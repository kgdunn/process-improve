# (c) Kevin Dunn, 2010-2026. MIT License.
"""Curvature test: centre-point mean against factorial-point mean (ENG-02).

The single-degree-of-freedom test of Montgomery (*Design and Analysis of
Experiments*, section 6.8): ``SS_curvature = nF nC (yF - yC)**2 / (nF + nC)`` on one
degree of freedom, tested against **pure error** from the replicated runs (the centre
points, and any other replicated settings). The residual of a model fitted to every
run cannot serve as the denominator: it holds the curvature contrast itself, so the
two are not independent, and real curvature inflates the very error it is judged by.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from .lack_of_fit import pure_error


def _run_curvature_test(
    design_df: pd.DataFrame,
    response_col: str,
    factor_cols: list[str],
    alpha: float = 0.05,
    group_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Test for curvature by comparing the centre-point and factorial-point means against pure error.

    Parameters
    ----------
    design_df : pandas.DataFrame
        Factor settings and the response, in coded units.
    response_col : str
        Name of the response column.
    factor_cols : list[str]
        Factor column names. Centre points have every factor at 0, factorial points
        every factor at -1 or +1.
    alpha : float
        Significance level for the ``significant`` flag.
    group_cols : list[str] or None
        Columns that identify replicated runs for pure error; ``factor_cols`` when
        ``None`` (pass the block column too when the design is blocked).

    Returns
    -------
    dict[str, Any]
        ``{"curvature_test": {...}}`` with the two means, their ``difference`` (the factorial
        mean less the centre mean, ``yF - yC``, as Box, Hunter and Hunter and Montgomery write
        it: it estimates the sum of the pure quadratic coefficients, b11 + ... + bkk) and its
        ``std_error``, ``s * sqrt(1/nF + 1/nC)`` with ``s`` from pure error; the F statistic (and
        the equivalent t statistic, ``difference / std_error``) on ``(1, df_pure_error)`` degrees
        of freedom, the p-value and the ``significant`` flag, or an ``error``.
    """
    factors = design_df[factor_cols]
    y = design_df[response_col]

    is_center = (factors == 0).all(axis=1)
    n_center = int(is_center.sum())
    if n_center == 0:
        return {"curvature_test": {"error": "No center points in design."}}

    is_factorial = factors.abs().eq(1).all(axis=1)
    n_factorial = int(is_factorial.sum())
    if n_factorial == 0:
        return {"curvature_test": {"error": "No factorial points found."}}

    ss_pe, df_pe = pure_error(design_df, response_col, group_cols or factor_cols)
    if df_pe == 0 or ss_pe <= 0:
        return {
            "curvature_test": {
                "error": "No pure error: the test needs replicated runs, such as two or more center points."
            }
        }
    ms_pe = ss_pe / df_pe

    y_center_mean = float(y[is_center].mean())
    y_factorial_mean = float(y[is_factorial].mean())
    difference = y_factorial_mean - y_center_mean
    std_error = (ms_pe * (1.0 / n_factorial + 1.0 / n_center)) ** 0.5
    ss_curvature = n_factorial * n_center * difference**2 / (n_factorial + n_center)
    f_stat = ss_curvature / ms_pe
    p_value = float(stats.f.sf(f_stat, 1, df_pe))

    return {
        "curvature_test": {
            "center_point_mean": y_center_mean,
            "factorial_point_mean": y_factorial_mean,
            "difference": difference,
            "std_error": float(std_error),
            "ss_curvature": float(ss_curvature),
            "ms_pure_error": float(ms_pe),
            "df_pure_error": int(df_pe),
            "F_statistic": float(f_stat),
            "t_statistic": float(difference / std_error),
            "p_value": p_value,
            "significant": p_value < alpha,
            "significance_level": alpha,
            "n_center_points": n_center,
            "n_factorial_points": n_factorial,
        }
    }
