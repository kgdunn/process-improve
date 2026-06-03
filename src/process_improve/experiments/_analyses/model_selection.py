# (c) Kevin Dunn, 2010-2026. MIT License.
"""Stepwise model selection using AIC/BIC (ENG-02)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import statsmodels.formula.api as smf


def _run_model_selection(  # noqa: C901
    design_df: pd.DataFrame,
    response_col: str,
    factor_cols: list[str],
    direction: str = "backward",
    criterion: str = "aic",
) -> dict[str, Any]:
    """Stepwise model selection using AIC/BIC.

    Custom forward/backward selection (~60 lines).
    """
    import itertools  # noqa: PLC0415

    all_terms = list(factor_cols)
    # Add 2FI terms
    for a, b in itertools.combinations(factor_cols, 2):
        all_terms.append(f"{a}:{b}")

    def _fit_and_score(terms: list[str]) -> tuple[float, Any]:
        formula = f"{response_col} ~ 1" if not terms else f"{response_col} ~ {' + '.join(terms)}"
        result = smf.ols(formula, data=design_df).fit()
        score = result.aic if criterion == "aic" else result.bic
        return score, result

    if direction == "backward":
        current_terms = list(all_terms)
        best_score, best_result = _fit_and_score(current_terms)

        improved = True
        while improved and len(current_terms) > 0:
            improved = False
            for term in list(current_terms):
                candidate = [t for t in current_terms if t != term]
                score, result = _fit_and_score(candidate)
                if score < best_score:
                    best_score = score
                    best_result = result
                    current_terms = candidate
                    improved = True
                    break
    else:
        # Forward selection
        current_terms: list[str] = []
        remaining = list(all_terms)
        best_score, best_result = _fit_and_score(current_terms)

        improved = True
        while improved and remaining:
            improved = False
            for term in list(remaining):
                candidate = [*current_terms, term]
                score, result = _fit_and_score(candidate)
                if score < best_score:
                    best_score = score
                    best_result = result
                    current_terms = candidate
                    remaining.remove(term)
                    improved = True
                    break

    if hasattr(best_result.model, "formula"):
        selected_formula = best_result.model.formula
    else:
        selected_formula = str(best_result.model.endog_names)

    return {
        "model_selection": {
            "selected_formula": selected_formula,
            "criterion": criterion,
            "criterion_value": float(best_score),
            "direction": direction,
            "r_squared": float(best_result.rsquared),
            "r_squared_adj": float(best_result.rsquared_adj),
            "n_terms": int(best_result.df_model),
        }
    }
