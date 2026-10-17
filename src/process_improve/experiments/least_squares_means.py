# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Least-squares means of a factor's levels, and their Tukey, Dunnett or unadjusted comparisons.

When every level of a factor meets every level of the others equally often, the average of the
runs at a level is a fair summary of it. In an incomplete block design, a Youden square or an
unbalanced experiment it is not: a level that happened to fall in the better blocks looks better
than it is. The *least-squares mean* (adjusted mean, estimated marginal mean) is the fitted
model's prediction at that level, averaged with equal weight over the levels of every other
factor in the model, with each numeric covariate at its average. It is a linear combination
:math:`\ell^\top \hat\beta` of the coefficients, so its standard error is
:math:`\sqrt{\ell^\top \widehat{\operatorname{cov}}(\hat\beta)\, \ell}`, on the residual degrees of
freedom; so are the differences of two of them.

- Tukey's comparisons (``"tukey"``) test every pair against the studentized range of the
  number of levels, so the family of all pairs has the stated confidence (Tukey-Kramer).
- Dunnett's (``"dunnett"``) compare each level with one ``control``, against the largest of the
  correlated t statistics, with their correlations from the model.
- ``"none"`` gives each pair's plain t test and interval (Fisher's least significant difference),
  and the confidence the whole family of those intervals has, which is lower.

``at`` fixes other factors at one level (or a covariate at one value) instead of averaging over
them: the comparisons of one factor's levels within one level of another, on the error of the whole
model. Tukey's and the unadjusted comparisons also give *connecting letters*: levels that share a
letter do not differ significantly.

Examples
--------
>>> import pandas as pd
>>> from process_improve.experiments.least_squares_means import least_squares_means
>>> runs = pd.DataFrame({"t": list("AABBCC"), "b": list("121212"), "y": [1.0, 2.0, 3.0, 4.5, 2.0, 3.5]})
>>> found = least_squares_means(runs, "y", "t + b", "t")
>>> [round(m["mean"], 2) for m in found["means"]]
[1.5, 3.75, 2.75]
"""

from __future__ import annotations

import itertools
import re
import string
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrices
from scipy import optimize, stats

from process_improve.experiments.models import validate_formula_is_safe, validate_identifier_is_safe

COMPARISONS = ("tukey", "dunnett", "none")
_SEED = 20_260_101  # the multivariate t's quadrature is randomised: one seed makes Dunnett's numbers repeat


def _label(column: pd.Series, value: Any) -> str:  # noqa: ANN401 - a level as the caller gives it: a number or a name
    """Return a value as the label its column's levels take once they are text: 70 is "70.0" among floats."""
    return str(column.dtype.type(value)) if pd.api.types.is_numeric_dtype(column) else str(value)


def _grid(table: pd.DataFrame, used: list[str], factor: str, at: dict[str, Any]) -> pd.DataFrame:
    """Return the reference grid: each level of the factor with every level of the other categorical factors.

    A numeric covariate sits at its average; a factor or covariate in *at* sits at the value given.
    """
    others = [c for c in used if c != factor]
    categorical = [c for c in others if not pd.api.types.is_numeric_dtype(table[c])]
    for column in set(at) & set(categorical):
        if at[column] not in set(table[column]):
            raise ValueError(f"{column} has no level {at[column]!r}.")
    numeric = {c: float(at.get(c, table[c].mean())) for c in others if c not in categorical}
    levels = [list(pd.unique(table[factor]))]
    levels += [[at[c]] if c in at else list(pd.unique(table[c])) for c in categorical]
    rows = [dict(zip([factor, *categorical], combo, strict=True)) | numeric for combo in itertools.product(*levels)]
    return pd.DataFrame(rows)


def _letters(means: dict[str, float], different: list[tuple[str, str]]) -> dict[str, str]:
    """Return connecting letters: levels that share a letter do not differ significantly.

    Piepho's (2004) insert-and-absorb: start with every level in one group; for each significantly
    different pair, split each group that holds both levels into two, each without one of them,
    then drop any group held inside another. The group with the largest mean gets ``A``.
    """
    groups = [frozenset(means)]
    for a, b in different:
        split = {part for g in groups for part in ([g - {a}, g - {b}] if {a, b} <= g else [g])}
        groups = [g for g in split if not any(g < h for h in split)]
    groups.sort(key=lambda g: (-max(means[v] for v in g), -sum(means[v] for v in g) / len(g)))
    return {
        v: "".join(letter for letter, g in zip(string.ascii_uppercase, groups, strict=False) if v in g) for v in means
    }


def _dunnett(t_values: np.ndarray, correlation: np.ndarray, df: float, conflevel: float) -> tuple[np.ndarray, float]:
    """Return two-sided Dunnett p-values for the observed t's and the critical value, from the multivariate t."""
    k = len(t_values)
    joint = stats.multivariate_t(shape=correlation, df=df, seed=_SEED)

    def inside(d: float) -> float:
        return float(joint.cdf(np.full(k, d), lower_limit=np.full(k, -d), maxpts=200_000 * k))

    p_values = np.array([1.0 - inside(abs(t)) for t in t_values])
    critical = float(optimize.brentq(lambda d: inside(d) - conflevel, 1e-6, 50.0, xtol=1e-8))
    return np.clip(p_values, 0.0, 1.0), critical


def _compare(  # noqa: PLR0913 - the fitted model's pieces, and how the levels are compared
    levels: list[str],
    weights: np.ndarray,
    beta: np.ndarray,
    cov: np.ndarray,
    df: float,
    comparisons: str,
    control: str | None,
    conflevel: float,
) -> tuple[list[dict[str, Any]], float]:
    """Return one row per compared pair of levels, and the critical value the intervals use."""
    if comparisons == "dunnett":
        if control is not None and control not in levels:
            raise ValueError(f"The factor has no level {control!r}; its levels are {levels}.")
        reference = levels[0] if control is None else control
        pairs = [(level, reference) for level in levels if level != reference]
    else:
        pairs = [(levels[j], levels[i]) for i, j in itertools.combinations(range(len(levels)), 2)]
    index = {level: i for i, level in enumerate(levels)}
    contrasts = np.array([weights[index[a]] - weights[index[b]] for a, b in pairs])
    differences = contrasts @ beta
    covariance = contrasts @ cov @ contrasts.T
    std_errors = np.sqrt(np.diag(covariance))
    t_values = differences / std_errors
    if comparisons == "tukey":
        p_values = stats.studentized_range.sf(np.abs(t_values) * np.sqrt(2), len(levels), df)
        critical = float(stats.studentized_range.ppf(conflevel, len(levels), df)) / np.sqrt(2)
    elif comparisons == "dunnett":
        correlation = covariance / np.outer(std_errors, std_errors)
        p_values, critical = _dunnett(t_values, correlation, df, conflevel)
    else:
        p_values, critical = 2 * stats.t.sf(np.abs(t_values), df), float(stats.t.ppf(0.5 + conflevel / 2, df))
    rows = [
        {
            "level": str(a),
            "versus": str(b),
            "difference": float(d),
            "std_error": float(se),
            "t": float(t),
            "p_value": float(p),
            "ci_low": float(d - critical * se),
            "ci_high": float(d + critical * se),
            "half_width": float(critical * se),
        }
        for (a, b), d, se, t, p in zip(pairs, differences, std_errors, t_values, p_values, strict=True)
    ]
    return rows, float(critical)


def least_squares_means(  # noqa: PLR0913 - the data, the model, the factor, and how its levels are compared
    design_matrix: pd.DataFrame,
    response: str,
    model: str,
    factor: str,
    comparisons: str = "tukey",
    control: str | None = None,
    conflevel: float = 0.95,
    at: dict[str, Any] | None = None,
    categorical: list[str] | None = None,
) -> dict[str, Any]:
    r"""Return the least-squares mean of each level of a factor, and their comparisons.

    Parameters
    ----------
    design_matrix : DataFrame
        The runs: the factors and the response. A categorical factor is a column of labels.
    response : str
        The response column.
    model : str
        The right-hand side of the model, a plain formula over the columns, ``"treatment + block"``;
        it must hold *factor*.
    factor : str
        The factor whose levels are compared, taken as levels even when they are numbers.
    comparisons : {"tukey", "dunnett", "none"}
        All pairs by Tukey's studentized range, each level against *control* by Dunnett's
        multivariate t, or all pairs unadjusted.
    control : str or None
        For Dunnett's comparisons, the control level; the first level by default.
    conflevel : float
        The confidence of the intervals, and of the family for Tukey's and Dunnett's.
    at : dict[str, Any] or None
        Other factors held at one level, ``{"temperature": 70}``, and covariates held at one value,
        instead of averaged over: the comparisons are then within that level, on the residual of
        the whole model.
    categorical : list[str] or None
        Other numeric columns the model takes as levels, such as a temperature set at three values,
        rather than as a covariate with a slope. A column of labels is always taken as levels.

    Returns
    -------
    dict[str, Any]
        ``means``, for each level (in the order of the data) its ``mean``, ``std_error`` and
        interval on the residual degrees of freedom, and ``by_level``, the same keyed by level;
        ``comparisons``, one row per pair (``level`` less ``versus``) with the ``difference``,
        ``std_error``, ``t``, adjusted ``p_value`` and simultaneous interval (and its
        ``half_width``); ``pairs``, the same keyed ``"level - versus"``; the residual
        ``df``; the ``method``; the ``critical_value`` that multiplies each difference's standard
        error; ``individual_confidence``, the confidence each interval has alone, and
        ``family_confidence``, the confidence that every interval holds its difference (for
        ``"none"``, the studentized range's, exact when the levels are equally replicated). Tukey's
        also give the ``studentized_range`` point, the critical value times the square root of 2,
        and Tukey's and the unadjusted give ``letters``, each level's connecting letters at the
        significance level 1 - *conflevel*.
    """
    if comparisons not in COMPARISONS:
        raise ValueError(f"comparisons must be one of {COMPARISONS}; got {comparisons!r}.")
    for name in (response, factor):
        validate_identifier_is_safe(name)
    table = design_matrix.reset_index(drop=True)
    validate_formula_is_safe(f"{response} ~ {model}", table.columns)
    used = sorted(
        {c for c in table.columns if re.search(rf"\b{re.escape(str(c))}\b", model)}, key=list(table.columns).index
    )
    labels = [factor, *(categorical or [])]
    missing = [c for c in labels if c not in used]
    if missing:
        raise ValueError(f"The model {model!r} does not hold {missing}.")
    unknown = sorted(set(at or {}) - set(used) | {factor} & set(at or {}))
    if unknown:
        raise ValueError(f"`at` names {unknown}, which are not other factors of the model.")
    named = {c for c in used if c in labels or not pd.api.types.is_numeric_dtype(table[c])}
    at = {c: _label(table[c], v) if c in named else v for c, v in (at or {}).items()}
    control = None if control is None else _label(table[factor], control)
    table = table.astype(dict.fromkeys(labels, str))
    outcome, exog = dmatrices(f"{response} ~ {model}", table, return_type="dataframe")
    fit = sm.OLS(outcome, exog).fit()
    grid = _grid(table, used, factor, at)
    rows = np.asarray(build_design_matrices([exog.design_info], grid)[0], dtype=float)
    levels = [str(level) for level in pd.unique(table[factor])]
    weights = np.array([rows[(grid[factor] == level).to_numpy()].mean(axis=0) for level in levels])
    beta, cov, df = fit.params.to_numpy(), fit.cov_params().to_numpy(), float(fit.df_resid)
    means = weights @ beta
    t_crit = float(stats.t.ppf(0.5 + conflevel / 2, df))
    mean_rows = []
    for level, mean, w in zip(levels, means, weights, strict=True):
        se = float(np.sqrt(w @ cov @ w))
        mean_rows.append(
            {
                "level": str(level),
                "mean": float(mean),
                "std_error": se,
                "ci_low": float(mean - t_crit * se),
                "ci_high": float(mean + t_crit * se),
            }
        )

    comparison_rows, critical = _compare(levels, weights, beta, cov, df, comparisons, control, conflevel)
    k = len(levels)
    result: dict[str, Any] = {
        "means": mean_rows,
        "by_level": {row["level"]: row for row in mean_rows},
        "comparisons": comparison_rows,
        "pairs": {f"{row['level']} - {row['versus']}": row for row in comparison_rows},
        "df": df,
        "method": comparisons,
        "critical_value": float(critical),
        "individual_confidence": float(1 - 2 * stats.t.sf(critical, df)),
        "family_confidence": float(stats.studentized_range.cdf(critical * np.sqrt(2), k, df))
        if comparisons == "none"
        else conflevel,
    }
    if comparisons == "tukey":
        result["studentized_range"] = float(critical * np.sqrt(2))
    if comparisons != "dunnett":
        different = [(row["level"], row["versus"]) for row in comparison_rows if row["p_value"] < 1 - conflevel]
        result["letters"] = _letters(dict(zip(map(str, levels), map(float, means), strict=True)), different)
    return result
