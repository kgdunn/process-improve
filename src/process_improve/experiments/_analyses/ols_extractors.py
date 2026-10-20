# (c) Kevin Dunn, 2010-2026. MIT License.
"""Thin extractors over a fitted OLS result: ANOVA, effects, coefficients,
significance, and confidence intervals (ENG-02).

The ANOVA and significance take a fit with one column per alias chain
(:func:`~.aliasing.chain_reduced_fit`) and a ``labels`` map that renames each chain's
leader to the chain, so exactly aliased terms are tested once, as in the effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

from ._shared import is_block_term
from .aliasing import estimable_effects

_SATURATED_NOTE = (
    "Saturated model - no residual degrees of freedom, so no term can be tested. "
    "Use analysis_type='lenth_method' for an unreplicated two-level design, or fit fewer terms."
)


def _block_row(ols_result: RegressionResultsWrapper, blocks: tuple[str, ...]) -> dict[str, Any]:
    """One ANOVA row for all the block contrasts together, by the F-test that they are all zero."""
    names = [str(n) for n in ols_result.model.exog_names]
    present = [name for name in blocks if name in names]
    restrictions = np.zeros((len(present), len(names)))
    for row, name in enumerate(present):
        restrictions[row, names.index(name)] = 1.0
    test = ols_result.f_test(restrictions)
    f_value = float(np.squeeze(test.fvalue))
    mse = float(ols_result.mse_resid)
    return {
        "source": "Block",
        "df": len(present),
        "sum_sq": f_value * len(present) * mse,
        "mean_sq": f_value * mse,
        "F": f_value,
        "p_value": float(test.pvalue),
    }


def _least_squares(columns: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Return the residual sum of squares of ``y`` on ``columns``, and the rank of ``columns``."""
    if columns.shape[1] == 0:
        return float(y @ y), 0
    coefficients, _, rank, _ = np.linalg.lstsq(columns, y, rcond=None)
    residuals = y - columns @ coefficients
    return float(residuals @ residuals), int(rank)


def _term_columns(ols_result: RegressionResultsWrapper) -> dict[str, tuple[frozenset, np.ndarray]]:
    """Return each model term's factors and columns, by the term's name, the intercept's factors empty.

    The columns are found by name, so a fit with columns dropped (one per alias chain, as
    :func:`~.aliasing.chain_reduced_fit` leaves) keeps only the columns it has.
    """
    data = ols_result.model.data
    # The formula's patsy DesignInfo: statsmodels 0.15 holds it as model_spec, 0.14 (Pyodide's) as design_info.
    info = getattr(data, "model_spec", None) or data.design_info
    exog = np.asarray(ols_result.model.exog, dtype=float)
    position = {str(name): k for k, name in enumerate(ols_result.model.exog_names)}
    terms = {}
    for term in info.terms:
        found = [position[name] for name in info.column_names[info.slice(term)] if name in position]
        terms[term.name()] = (frozenset(term.factors), exog[:, found])
    return terms


def _type_ii_by_rank(ols_result: RegressionResultsWrapper, blocks: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    """Type II sums of squares of a model whose matrix lacks full rank, each on the degrees of freedom it adds.

    A factor nested in another, such as the times of spinning within each square of Davies's
    yarn experiment (#789 B7), has columns that lie in that other factor's. statsmodels' Type
    II table then counts every column as a degree of freedom, and its rows can exceed the
    total. Here each term's sum of squares is the drop in the residual sum of squares when it
    joins the terms that do not contain it, on the rise in rank it brings. A term that adds no
    rank cannot be tested, and a term that adds less than its columns is tested on what it adds.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The table, laid out as ``anova_lm``'s with a ``Residual`` row, and one sentence for each
        term that cannot be tested on all its columns.
    """
    y = np.asarray(ols_result.model.endog, dtype=float)
    rss, rank = _least_squares(np.asarray(ols_result.model.exog, dtype=float), y)
    df_resid = len(y) - rank
    mse = rss / df_resid
    terms = _term_columns(ols_result)
    rows, notes = {}, []
    for name, (factors, columns) in terms.items():
        if not factors or is_block_term(name, blocks):
            continue
        kept = [other for other_factors, other in terms.values() if not factors <= other_factors]
        reduced = np.hstack(kept) if kept else columns[:, :0]
        rss_reduced, rank_reduced = _least_squares(reduced, y)
        rss_added, rank_added = _least_squares(np.hstack([reduced, columns]), y)
        df, width = rank_added - rank_reduced, columns.shape[1]
        sum_sq = max(rss_reduced - rss_added, 0.0) if df > 0 else 0.0
        f_value = sum_sq / df / mse if df > 0 else np.nan
        rows[name] = (df, sum_sq, f_value, stats.f.sf(f_value, df, df_resid) if df > 0 else np.nan)
        if df == 0:
            notes.append(_untestable_note(name, terms))
        elif df < width:
            notes.append(
                f"{name} is tested on {df} of its {width} degrees of freedom: the rest lie in the columns of the "
                "terms it is adjusted for."
            )
    rows["Residual"] = (df_resid, rss, np.nan, np.nan)
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["df", "sum_sq", "F", "PR(>F)"])
    return table, notes


def _untestable_note(name: str, terms: dict[str, tuple[frozenset, np.ndarray]]) -> str:
    """Say why term ``name`` adds no degrees of freedom: which terms' columns span its own, and the strata to use."""
    factors, columns = terms[name]
    intercept = [other for other_factors, other in terms.values() if not other_factors]

    def spans(other: np.ndarray) -> bool:
        base = np.hstack([*intercept, other])
        return int(np.linalg.matrix_rank(np.hstack([base, columns]))) == int(np.linalg.matrix_rank(base))

    within = [
        other_name
        for other_name, (other_factors, other) in terms.items()
        if other_factors and other_name != name and not factors <= other_factors and spans(other)
    ]
    if not within:
        return (
            f"{name} cannot be tested: its columns lie in those of the other terms together, so adjusting it for them "
            "leaves it no degrees of freedom."
        )
    strata = " + ".join(f"{name}/{other}" for other in within)
    return (
        f"{name} cannot be tested: its columns lie in those of {' and '.join(within)}, as a factor's do when another "
        "is nested in it, so adjusting it for them leaves it no degrees of freedom. The other rows are adjusted for "
        f'it. multistratum_anova(..., strata="{strata}") gives the nested sums of squares.'
    )


def _run_anova(
    ols_result: RegressionResultsWrapper,
    anova_type: int = 2,
    labels: dict[str, str] | None = None,
    blocks: tuple[str, ...] = (),
) -> dict[str, Any]:
    """ANOVA table via statsmodels, with Type II sums of squares by default.

    Type II tests each term adjusted for every term that does not contain it, so a main
    effect is not adjusted for its own interactions. In an orthogonal design this equals
    the Type III (fully adjusted) test that the coefficient t-tests make; in a
    non-orthogonal one the two can differ, and the type used is reported under
    ``anova_type``. The ``blocks`` contrast columns are tested together, as one
    ``"Block"`` row with one degree of freedom fewer than the number of blocks.

    When the model matrix lacks full rank, as with a factor nested in another, the Type II
    rows are worked out on the rank each term adds (:func:`_type_ii_by_rank`), and
    ``anova_note`` names each term that cannot be tested on all its columns.
    """
    if ols_result.df_resid <= 0:
        return {"anova_table": [], "anova_note": _SATURATED_NOTE}
    labels = labels or {}
    notes: list[str] = []
    exog = np.asarray(ols_result.model.exog, dtype=float)
    if anova_type == 2 and np.linalg.matrix_rank(exog) < exog.shape[1]:
        table, notes = _type_ii_by_rank(ols_result, blocks)
    else:
        table = sm.stats.anova_lm(ols_result, typ=anova_type)
    records = []
    for idx, row in table.iterrows():
        if is_block_term(str(idx), blocks):
            continue
        df = float(row.get("df", 0))
        sum_sq = float(row.get("sum_sq", 0))
        records.append(
            {
                "source": labels.get(str(idx), str(idx)),
                "df": int(df),
                "sum_sq": sum_sq,
                "mean_sq": sum_sq / df if df > 0 else None,
                "F": float(row["F"]) if "F" in row and pd.notna(row.get("F")) else None,
                "p_value": (float(row["PR(>F)"]) if "PR(>F)" in row and pd.notna(row.get("PR(>F)")) else None),
            }
        )
    if any(name in ols_result.model.exog_names for name in blocks):
        records.insert(0, _block_row(ols_result, blocks))
    result = {"anova_table": records, "anova_type": "I" * anova_type if anova_type <= 3 else str(anova_type)}
    if notes:
        result["anova_note"] = " ".join(notes)
    return result


def _run_effects(
    ols_result: RegressionResultsWrapper,
    blocks: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Effects of the factors: each coefficient times the span of its coded model column.

    ``ols_result`` must be fitted on factors coded to -1/+1, as
    :func:`analyze_experiment` arranges. An effect is then the change in response
    from the lowest to the highest value of its model column: twice the coefficient
    for a main effect or interaction (Box, Hunter and Hunter, chapter 5), and the
    coefficient itself for a square, whose column runs from 0 to 1.

    Also returns ``effect_std_errors`` (the coefficient standard error times
    the same span) when residual degrees of freedom are available; consumers such
    as the Pareto plot use this to draw effect-level error bars.

    Exactly aliased terms are reported as one effect per alias chain, named
    ``"A:B + C:D"``, since the design determines only their sum (#16). The
    chains are listed under ``alias_chains``, and terms aliased with the
    intercept under ``confounded_with_mean``; both keys appear only when
    there is aliasing. The ``blocks`` contrasts, and chains they lead, are not
    factor effects and are left out.
    """
    estimable = estimable_effects(ols_result)
    keep = [label for label in estimable.coefficients.index if not is_block_term(label, blocks)]
    spans = estimable.spans[keep]
    result: dict[str, Any] = {"effects": (spans * estimable.coefficients[keep]).to_dict()}
    if estimable.std_errors is not None:
        result["effect_std_errors"] = {str(k): float(spans[k] * estimable.std_errors[k]) for k in keep}
    chains = {label: members for label, members in estimable.chains.items() if label in keep}
    if chains:
        result["alias_chains"] = chains
    if estimable.confounded_with_mean:
        result["confounded_with_mean"] = estimable.confounded_with_mean
    return result


def _standardised(ols_result: RegressionResultsWrapper) -> np.ndarray:
    """Each coefficient in standard deviations of the response per standard deviation of its model column.

    The coefficient times the standard deviation of its column of the model matrix, over the
    standard deviation of the response: unitless, so terms in different units compare. The
    intercept's column does not vary, so its standardised coefficient is zero. Not a number when
    the response does not vary.
    """
    columns = np.asarray(ols_result.model.exog, dtype=float)
    response = np.asarray(ols_result.model.endog, dtype=float)
    if len(response) < 2 or np.ptp(response) == 0:  # a spread needs two values that differ
        return np.full(columns.shape[1], np.nan)
    return np.asarray(ols_result.params, dtype=float) * columns.std(axis=0, ddof=1) / response.std(ddof=1)


def _run_coefficients(ols_result: RegressionResultsWrapper, alpha: float = 0.05) -> dict[str, Any]:
    """Coefficients with standard errors, t-values, p-values, CIs at level 1 - alpha and standardised coefficients."""
    ci = ols_result.conf_int(alpha=alpha)
    summary_df = pd.DataFrame(
        {
            "coefficient": ols_result.params,
            "std_error": ols_result.bse,
            "t_value": ols_result.tvalues,
            "p_value": ols_result.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "standardised_coefficient": _standardised(ols_result),
        }
    )
    records = []
    for name, row in summary_df.iterrows():
        records.append(
            {
                "term": str(name),
                "coefficient": float(row["coefficient"]),
                "std_error": float(row["std_error"]),
                "t_value": float(row["t_value"]),
                "p_value": float(row["p_value"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "standardised_coefficient": float(row["standardised_coefficient"]),
            }
        )
    return {"coefficients": records, "confidence_level": 1.0 - alpha}


def _run_significance(
    ols_result: RegressionResultsWrapper,
    alpha: float = 0.05,
    labels: dict[str, str] | None = None,
    blocks: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Split the terms into significant and not significant, and list those that cannot be tested.

    A term whose p-value is not a number (no residual degrees of freedom, or a term the
    design cannot estimate) is listed under ``not_estimable_terms`` rather than left out
    of both lists. The ``blocks`` contrasts are not listed.
    """
    labels = labels or {}
    pvals = ols_result.pvalues.drop("Intercept", errors="ignore")
    named = [(labels.get(str(n), str(n)), float(p)) for n, p in pvals.items() if not is_block_term(str(n), blocks)]
    result: dict[str, Any] = {
        "significant_terms": [n for n, p in named if p < alpha],
        "not_significant_terms": [n for n, p in named if p >= alpha],
        "not_estimable_terms": [n for n, p in named if not np.isfinite(p)],
        "significance_level": alpha,
    }
    if ols_result.df_resid <= 0:
        result["significance_note"] = _SATURATED_NOTE
    return result


def _run_confidence_intervals(ols_result: RegressionResultsWrapper, alpha: float = 0.05) -> dict[str, Any]:
    """Confidence intervals for coefficients."""
    ci = ols_result.conf_int(alpha=alpha)
    records = [
        {
            "term": str(name),
            "ci_low": float(ci.loc[name, 0]),
            "ci_high": float(ci.loc[name, 1]),
        }
        for name in ci.index
    ]
    return {"confidence_intervals": records, "confidence_level": 1.0 - alpha}
