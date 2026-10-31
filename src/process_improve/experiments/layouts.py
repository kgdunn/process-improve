# (c) Kevin Dunn, 2010-2026. MIT License.
"""Analyses of blocked and two-way layouts that the classical sources print (#800).

- :func:`tukey_nonadditivity`: Tukey's one degree of freedom for nonadditivity, and the power of
  the response that would remove it;
- :func:`estimate_missing`: the least-squares estimates of the runs whose response is missing,
  and the approximate analysis of the table they complete.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from process_improve.experiments._lm import validate_contrast_levels, validate_formula_is_safe

__all__ = ["estimate_missing", "tukey_nonadditivity"]

#: What Tukey's sum of squares can be tested against.
_ERRORS = ("remainder", "pure")


def _fit(design_matrix: pd.DataFrame, formula: str) -> Any:  # noqa: ANN401 - a statsmodels results wrapper
    """Fit ``formula`` by least squares, after checking it is a safe formula over the data's columns."""
    if "~" not in formula:
        raise ValueError(f"The formula must name the response, as 'y ~ ...'; got {formula!r}.")
    validate_formula_is_safe(formula, design_matrix.columns, allow_transforms=True)
    validate_contrast_levels(formula, design_matrix)
    return smf.ols(formula, data=design_matrix).fit()


def tukey_nonadditivity(design_matrix: pd.DataFrame, formula: str, error: str = "remainder") -> dict[str, float]:
    r"""Return Tukey's one degree of freedom for nonadditivity, and the power of the response that would remove it.

    The additive model, such as ``"y ~ C(block) + C(treatment)"``, is fitted again with one more
    regressor, the square of its fitted values (Tukey 1949). The sum of squares that regressor
    takes is the part of the interaction that is proportional to the product of the row and
    column effects, :math:`D a_i b_j`, with :math:`D` twice the regressor's coefficient. A power
    :math:`1 - D \bar{y}` of the response would remove that part, so a significant test points
    to a transformation: near 0, a log; near -1, a reciprocal.

    Parameters
    ----------
    design_matrix : pd.DataFrame
        The runs: the factors and the response.
    formula : str
        The additive model, with the response on the left.
    error : {"remainder", "pure"}
        What the sum of squares is tested against: the residual of the model with the regressor
        (default), as for one run per cell; or the pure error between replicated runs of the
        same cell, as Box, Hunter and Hunter (1978) test the toxic-agents data.

    Returns
    -------
    dict[str, float]
        ``ss`` on ``df`` (1), its ``f`` and ``p_value``; ``ss_error``, ``df_error`` and
        ``ms_error``, the denominator's; ``remainder_ss`` and ``remainder_df``, the interaction
        left once ``ss`` is taken out (the denominator itself, with one run per cell);
        ``coefficient``, the regressor's (Box, Hunter and Hunter's theta); and ``power``.

    Raises
    ------
    ValueError
        If ``error`` is not one of the two, ``"pure"`` is asked of runs with no replicated cell,
        or the denominator has no degrees of freedom.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments import tukey_nonadditivity
    >>> impurity = pd.DataFrame({  # Montgomery (2013), Table 5.10
    ...     "temperature": [100] * 5 + [125] * 5 + [150] * 5,
    ...     "pressure": [25, 30, 35, 40, 45] * 3,
    ...     "y": [5, 4, 6, 3, 5, 3, 1, 4, 2, 3, 1, 1, 3, 1, 2],
    ... })
    >>> found = tukey_nonadditivity(impurity, "y ~ C(temperature) + C(pressure)")
    >>> round(found["ss"], 4), round(found["ss_error"], 4), found["df_error"], round(found["f"], 2)
    (0.0985, 1.9015, 7.0, 0.36)
    """
    if error not in _ERRORS:
        raise ValueError(f"error must be one of {_ERRORS}; got {error!r}.")
    additive = _fit(design_matrix, formula)
    exog, y = np.asarray(additive.model.exog, dtype=float), np.asarray(additive.model.endog, dtype=float)
    augmented = sm.OLS(y, np.column_stack([exog, np.asarray(additive.fittedvalues) ** 2])).fit()
    cells = np.unique(exog, axis=0, return_inverse=True)[1].reshape(-1)
    pure_ss = float(((y - (np.bincount(cells, y) / np.bincount(cells))[cells]) ** 2).sum())
    pure_df = len(y) - int(cells.max()) - 1
    if error == "pure" and pure_df < 1:
        raise ValueError("No cell has replicated runs, so there is no pure error; test against the remainder.")
    ss_error, df_error = (
        (pure_ss, float(pure_df)) if error == "pure" else (float(augmented.ssr), float(augmented.df_resid))
    )
    if df_error < 1:
        raise ValueError("The model with Tukey's regressor leaves no residual degrees of freedom.")
    ss, coefficient = float(additive.ssr - augmented.ssr), float(augmented.params[-1])
    f = ss / (ss_error / df_error)
    return {
        "ss": ss,
        "df": 1,
        "f": f,
        "p_value": float(stats.f.sf(f, 1, df_error)),
        "ss_error": ss_error,
        "df_error": df_error,
        "ms_error": ss_error / df_error,
        "remainder_ss": float(augmented.ssr) - pure_ss,
        "remainder_df": float(augmented.df_resid) - pure_df,
        "coefficient": coefficient,
        "power": 1 - 2 * coefficient * float(y.mean()),
    }


def estimate_missing(design_matrix: pd.DataFrame, formula: str, decimals: int | None = None) -> dict[str, Any]:
    """Return the least-squares estimates of the runs whose response is missing, and the table they complete.

    Each estimate is the model's prediction at the missing run, fitted to the runs that were
    observed: the value that leaves that run a residual of zero. For one missing value of a
    randomized block design it is Yates's formula; several are estimated together. The exact
    analysis is that of the observed runs alone (``analyze_experiment`` gives it). The
    approximate one, which the classical sources print, treats the completed table as complete,
    with the residual degrees of freedom reduced by one for each estimate; it overstates the
    effects a little.

    Parameters
    ----------
    design_matrix : pd.DataFrame
        The runs, the missing ones with their factor settings and no response.
    formula : str
        The model, with the response on the left, such as ``"y ~ C(block) + C(treatment)"``.
    decimals : int, optional
        Round the estimates to this many decimals before they complete the table, as a source
        does that analyses the estimates it prints.

    Returns
    -------
    dict[str, Any]
        ``estimates``, one row per missing run: its index, its factor settings and the estimate
        under the response's name; ``completed``, the runs with the estimates filled in; and
        ``approximate_anova``, the sequential analysis of the completed table, one row per term
        and the residual, with ``df``, ``sum_sq``, ``mean_sq``, ``f`` and ``p_value``.

    Raises
    ------
    ValueError
        If no response is missing, or the formula's response is not a column of the runs.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments import estimate_missing
    >>> runs = pd.DataFrame({
    ...     "block": ["I", "I", "I", "II", "II", "II"], "treatment": ["A", "B", "C"] * 2,
    ...     "y": [10.0, 12.0, 15.0, 11.0, None, 16.0],
    ... })
    >>> round(estimate_missing(runs, "y ~ C(block) + C(treatment)")["estimates"][0]["y"], 6)  # 12 + (27 - 25) / 2
    13.0
    """
    response = formula.split("~", 1)[0].strip()
    if response not in design_matrix.columns:
        raise ValueError(f"The formula's response {response!r} is not a column of the runs.")
    missing = design_matrix[response].isna().to_numpy()
    if not missing.any():
        raise ValueError(f"No run is missing its {response!r}.")
    estimates = _fit(design_matrix.loc[~missing], formula).predict(design_matrix.loc[missing])
    if decimals is not None:
        estimates = estimates.round(decimals)
    completed = design_matrix.copy()
    completed.loc[missing, response] = estimates.to_numpy()
    table = anova_lm(_fit(completed, formula), typ=1)
    table.loc["Residual", "df"] -= int(missing.sum())
    table["mean_sq"] = table["sum_sq"] / table["df"]
    table["F"] = table["mean_sq"] / table.loc["Residual", "mean_sq"]
    table["PR(>F)"] = stats.f.sf(table["F"], table["df"], table.loc["Residual", "df"])
    rows = [
        {"term": str(term), "df": float(row["df"]), "sum_sq": float(row["sum_sq"]), "mean_sq": float(row["mean_sq"])}
        | ({} if term == "Residual" else {"f": float(row["F"]), "p_value": float(row["PR(>F)"])})
        for term, row in table.iterrows()
    ]
    found = design_matrix.loc[missing].drop(columns=response)
    return {
        "estimates": [
            {"index": index, **settings, response: float(estimate)}
            for (index, settings), estimate in zip(found.to_dict("index").items(), estimates, strict=True)
        ],
        "completed": completed,
        "approximate_anova": rows,
    }
