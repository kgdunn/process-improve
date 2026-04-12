# (c) Kevin Dunn, 2010-2026. MIT License.

"""Experiment analysis: fit models, ANOVA, diagnostics, residuals.

Provides :func:`analyze_experiment`, the main analytical workhorse for
designed experiments (Tool 3 in the DOE tool architecture).

Uses statsmodels and scipy for the heavy lifting, with thin custom code
for lack-of-fit, curvature test, Lenth's method, pred-R², adequate
precision, and confirmation run testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# ---------------------------------------------------------------------------
# Formula builder
# ---------------------------------------------------------------------------


def build_formula(
    response: str,
    factors: list[str],
    model: str | None = None,
) -> str:
    """Build a patsy/statsmodels formula string.

    Parameters
    ----------
    response : str
        Name of the response column.
    factors : list[str]
        Factor column names.
    model : str or None
        ``"main_effects"``, ``"interactions"``, ``"quadratic"``, or an
        explicit formula string.  *None* defaults to ``"interactions"``.

    Returns
    -------
    str
        A formula like ``"Y ~ A + B + A:B"``.
    """
    if model is None:
        model = "interactions"

    if "~" in str(model):
        return model

    joined = " + ".join(factors)

    if model == "main_effects":
        rhs = joined
    elif model == "interactions":
        rhs = f"({joined}) ** 2"
    elif model == "quadratic":
        squared = " + ".join(f"I({f} ** 2)" for f in factors)
        rhs = f"({joined}) ** 2 + {squared}"
    else:
        # Treat as raw RHS
        rhs = model

    return f"{response} ~ {rhs}"


# ---------------------------------------------------------------------------
# Fitted model container
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Container returned by :func:`analyze_experiment`.

    Holds the fitted OLS result and all requested analysis outputs.
    """

    ols_result: Any = None
    formula: str = ""
    results: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analysis-type implementations
# ---------------------------------------------------------------------------


def _run_anova(ols_result: Any, anova_type: int = 2) -> dict[str, Any]:
    """ANOVA table via statsmodels."""
    if ols_result.df_resid <= 0:
        return {"anova_table": [], "note": "Saturated model — no residual degrees of freedom for ANOVA."}
    table = sm.stats.anova_lm(ols_result, typ=anova_type)
    records = [
        {
            "source": str(idx),
            "df": int(row.get("df", 0)),
            "sum_sq": float(row.get("sum_sq", 0)),
            "mean_sq": float(row.get("mean_sq", 0)) if "mean_sq" in row else None,
            "F": float(row["F"]) if "F" in row and pd.notna(row.get("F")) else None,
            "p_value": (
                float(row["PR(>F)"]) if "PR(>F)" in row and pd.notna(row.get("PR(>F)")) else None
            ),
        }
        for idx, row in table.iterrows()
    ]
    return {"anova_table": records}


def _run_effects(ols_result: Any) -> dict[str, Any]:
    """Coefficient effects (2x the coefficient for coded ±1 factors)."""
    params = ols_result.params.drop("Intercept", errors="ignore")
    effects = (2.0 * params).to_dict()
    return {"effects": effects}


def _run_coefficients(ols_result: Any) -> dict[str, Any]:
    """Coefficients with standard errors, t-values, p-values, and CIs."""
    summary_df = pd.DataFrame({
        "coefficient": ols_result.params,
        "std_error": ols_result.bse,
        "t_value": ols_result.tvalues,
        "p_value": ols_result.pvalues,
        "ci_low": ols_result.conf_int()[0],
        "ci_high": ols_result.conf_int()[1],
    })
    records = []
    for name, row in summary_df.iterrows():
        records.append({
            "term": str(name),
            "coefficient": float(row["coefficient"]),
            "std_error": float(row["std_error"]),
            "t_value": float(row["t_value"]),
            "p_value": float(row["p_value"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        })
    return {"coefficients": records}


def _run_significance(ols_result: Any, alpha: float = 0.05) -> dict[str, Any]:
    """Identify significant and non-significant terms."""
    pvals = ols_result.pvalues.drop("Intercept", errors="ignore")
    significant = [str(n) for n, p in pvals.items() if p < alpha]
    not_significant = [str(n) for n, p in pvals.items() if p >= alpha]
    return {
        "significant_terms": significant,
        "not_significant_terms": not_significant,
        "significance_level": alpha,
    }


def _run_residual_diagnostics(ols_result: Any) -> dict[str, Any]:
    """Residual diagnostics: normality, independence, homoscedasticity."""
    residuals = ols_result.resid.values
    fitted = ols_result.fittedvalues.values

    # Shapiro-Wilk normality test
    n = len(residuals)
    if n >= 3:
        sw_stat, sw_p = stats.shapiro(residuals)
    else:
        sw_stat, sw_p = None, None

    # Durbin-Watson (independence)
    from statsmodels.stats.stattools import durbin_watson  # noqa: PLC0415

    dw = float(durbin_watson(residuals))

    # Breusch-Pagan (homoscedasticity)
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: PLC0415

        bp_stat, bp_p, _bp_f, _bp_fp = het_breuschpagan(residuals, ols_result.model.exog)
    except Exception:  # noqa: BLE001
        bp_stat, bp_p = None, None

    # Cook's distance
    influence = ols_result.get_influence()
    cooks_d = influence.cooks_distance[0]

    # Leverage
    leverage = influence.hat_matrix_diag

    return {
        "residual_diagnostics": {
            "shapiro_wilk": {
                "statistic": float(sw_stat) if sw_stat else None,
                "p_value": float(sw_p) if sw_p else None,
            },
            "durbin_watson": dw,
            "breusch_pagan": {
                "statistic": float(bp_stat) if bp_stat else None,
                "p_value": float(bp_p) if bp_p else None,
            },
            "cooks_distance": [float(c) for c in cooks_d],
            "leverage": [float(h) for h in leverage],
            "residuals": [float(r) for r in residuals],
            "fitted_values": [float(f) for f in fitted],
        }
    }


def _run_lack_of_fit(ols_result: Any, design_df: pd.DataFrame, response_col: str) -> dict[str, Any]:
    """Lack-of-fit F-test using pure error from replicated points.

    Separates residual SS into pure-error SS (from replicates) and
    lack-of-fit SS.  Custom implementation (~40 lines).
    """
    residuals = ols_result.resid
    y = design_df[response_col]

    # Identify replicate groups by rounding factor values
    factor_cols = [c for c in design_df.columns if c != response_col]
    if not factor_cols:
        return {"lack_of_fit": {"error": "No factor columns found."}}

    groups = design_df.groupby(factor_cols, sort=False)

    ss_pure_error = 0.0
    df_pure_error = 0

    for _name, group in groups:
        ni = len(group)
        if ni > 1:
            yi = y.loc[group.index]
            ss_pure_error += float(((yi - yi.mean()) ** 2).sum())
            df_pure_error += ni - 1

    if df_pure_error == 0:
        return {"lack_of_fit": {"error": "No replicated points — cannot test lack of fit."}}

    ss_residual = float((residuals**2).sum())
    df_residual = int(ols_result.df_resid)

    ss_lof = ss_residual - ss_pure_error
    df_lof = df_residual - df_pure_error

    if df_lof <= 0 or ss_pure_error <= 0:
        return {"lack_of_fit": {"error": "Insufficient degrees of freedom for lack-of-fit test."}}

    ms_lof = ss_lof / df_lof
    ms_pe = ss_pure_error / df_pure_error
    f_stat = ms_lof / ms_pe
    p_value = 1.0 - stats.f.cdf(f_stat, df_lof, df_pure_error)

    return {
        "lack_of_fit": {
            "ss_lack_of_fit": ss_lof,
            "df_lack_of_fit": df_lof,
            "ms_lack_of_fit": ms_lof,
            "ss_pure_error": ss_pure_error,
            "df_pure_error": df_pure_error,
            "ms_pure_error": ms_pe,
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }
    }


def _run_curvature_test(
    ols_result: Any,
    design_df: pd.DataFrame,
    response_col: str,
    factor_cols: list[str],
) -> dict[str, Any]:
    """Curvature test: compare center point mean vs factorial point mean.

    Custom implementation (~20 lines).
    """
    factors = design_df[factor_cols]
    y = design_df[response_col]

    # Center points: rows where ALL factors == 0
    is_center = (factors == 0).all(axis=1)
    n_center = int(is_center.sum())

    if n_center == 0:
        return {"curvature_test": {"error": "No center points in design."}}

    # Factorial points: rows where ALL factors are ±1
    is_factorial = factors.abs().eq(1).all(axis=1)
    n_factorial = int(is_factorial.sum())

    if n_factorial == 0:
        return {"curvature_test": {"error": "No factorial points found."}}

    y_center_mean = float(y[is_center].mean())
    y_factorial_mean = float(y[is_factorial].mean())
    difference = y_center_mean - y_factorial_mean

    mse = float(ols_result.mse_resid)
    se_diff = np.sqrt(mse * (1.0 / n_center + 1.0 / n_factorial))
    t_stat = difference / se_diff if se_diff > 0 else 0.0
    df_resid = int(ols_result.df_resid)
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df_resid))

    return {
        "curvature_test": {
            "center_point_mean": y_center_mean,
            "factorial_point_mean": y_factorial_mean,
            "difference": difference,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_center_points": n_center,
            "n_factorial_points": n_factorial,
        }
    }


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

    def _fit_and_score(terms: list[str]) -> tuple[float, Any]:  # noqa: ANN401
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


def _run_box_cox(design_df: pd.DataFrame, response_col: str) -> dict[str, Any]:
    """Box-Cox transformation using scipy."""
    from scipy.stats import boxcox  # noqa: PLC0415

    y = design_df[response_col].values.astype(float)

    if np.any(y <= 0):
        return {"box_cox": {"error": "Box-Cox requires all positive response values."}}

    y_transformed, lmbda = boxcox(y)

    return {
        "box_cox": {
            "lambda": float(lmbda),
            "transformed_values": [float(v) for v in y_transformed],
            "recommendation": (
                "log transform" if abs(lmbda) < 0.05
                else "square root" if abs(lmbda - 0.5) < 0.05
                else "inverse" if abs(lmbda - (-1.0)) < 0.05
                else f"power transform (lambda={lmbda:.3f})"
            ),
        }
    }


def _run_lenth_method(ols_result: Any) -> dict[str, Any]:
    """Lenth's method (PSE) for unreplicated factorials.

    Not available in mainstream Python libraries — custom (~30 lines).
    """
    params = ols_result.params.drop("Intercept", errors="ignore")
    effects = 2.0 * params.values  # coded ±1 → effect = 2 * coefficient
    abs_effects = np.abs(effects)

    # Step 1: initial median
    s0 = 1.5 * np.median(abs_effects)

    # Step 2: pseudo standard error — median of |effects| <= 2.5 * s0
    trimmed = abs_effects[abs_effects <= 2.5 * s0]
    pse = s0 if len(trimmed) == 0 else 1.5 * np.median(trimmed)

    # Margin of error and simultaneous margin of error
    m = len(effects)
    t_val = stats.t.ppf(1 - 0.025, df=m / 3)
    t_val_sim = stats.t.ppf(1 - 0.025 / m, df=m / 3)  # Bonferroni-like
    me = t_val * pse
    sme = t_val_sim * pse

    term_names = list(params.index)
    effect_list = [
        {
            "term": str(name),
            "effect": float(effects[i]),
            "active_ME": bool(abs(effects[i]) > me),
            "active_SME": bool(abs(effects[i]) > sme),
        }
        for i, name in enumerate(term_names)
    ]

    return {
        "lenth_method": {
            "PSE": float(pse),
            "ME": float(me),
            "SME": float(sme),
            "effects": effect_list,
        }
    }


def _run_confidence_intervals(ols_result: Any, alpha: float = 0.05) -> dict[str, Any]:
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


def _run_prediction(
    ols_result: Any,
    new_points: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Predictions with confidence and prediction intervals."""
    pred = ols_result.get_prediction(new_points)
    summary = pred.summary_frame(alpha=alpha)

    records = [
        {
            "predicted": float(row["mean"]),
            "ci_low": float(row["mean_ci_lower"]),
            "ci_high": float(row["mean_ci_upper"]),
            "pi_low": float(row["obs_ci_lower"]),
            "pi_high": float(row["obs_ci_upper"]),
        }
        for _i, row in summary.iterrows()
    ]
    return {"predictions": records}


def _run_confirmation_test(
    ols_result: Any,
    new_points: pd.DataFrame,
    observed: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare observed confirmation runs against predicted values with PI."""
    pred = ols_result.get_prediction(new_points)
    summary = pred.summary_frame(alpha=alpha)

    results = []
    for i, obs in enumerate(observed):
        row = summary.iloc[i]
        pi_low = float(row["obs_ci_lower"])
        pi_high = float(row["obs_ci_upper"])
        predicted = float(row["mean"])
        within_pi = pi_low <= obs <= pi_high
        results.append({
            "observed": obs,
            "predicted": predicted,
            "pi_low": pi_low,
            "pi_high": pi_high,
            "within_PI": within_pi,
        })

    all_pass = all(r["within_PI"] for r in results)
    return {
        "confirmation_test": {
            "results": results,
            "all_within_PI": all_pass,
            "confidence_level": 1.0 - alpha,
        }
    }


def _compute_pred_r_squared(ols_result: Any) -> float:
    """Predicted R² from PRESS residuals (~5 lines)."""
    influence = ols_result.get_influence()
    press_residuals = influence.resid_press
    ss_press = float(np.sum(press_residuals**2))
    ss_total = float(np.sum((ols_result.model.endog - ols_result.model.endog.mean()) ** 2))
    if ss_total == 0:
        return 0.0
    return 1.0 - ss_press / ss_total


def _compute_adequate_precision(ols_result: Any) -> float:
    """Adequate precision: signal-to-noise ratio (~10 lines)."""
    predicted = ols_result.fittedvalues.values
    mse = float(ols_result.mse_resid)
    n = len(predicted)
    p = ols_result.df_model + 1

    max_pred = float(np.max(predicted))
    min_pred = float(np.min(predicted))
    signal = max_pred - min_pred

    # Average variance of prediction
    avg_var = p * mse / n
    noise = np.sqrt(avg_var)

    if noise == 0:
        return float("inf")
    return signal / noise


# ---------------------------------------------------------------------------
# Analysis-type dispatch registry
# ---------------------------------------------------------------------------

_ANALYSIS_REGISTRY: dict[str, str] = {
    "anova": "anova",
    "effects": "effects",
    "coefficients": "coefficients",
    "significance": "significance",
    "residual_diagnostics": "residual_diagnostics",
    "lack_of_fit": "lack_of_fit",
    "curvature_test": "curvature_test",
    "model_selection": "model_selection",
    "box_cox": "box_cox",
    "lenth_method": "lenth_method",
    "confidence_intervals": "confidence_intervals",
    "prediction": "prediction",
    "confirmation_test": "confirmation_test",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_experiment(  # noqa: PLR0912, PLR0913, PLR0915, C901
    design_matrix: pd.DataFrame,
    responses: pd.DataFrame | pd.Series | None = None,
    model: str | None = None,
    analysis_type: str | list[str] = "anova",
    significance_level: float = 0.05,
    transform: str | None = None,
    coding: str = "coded",
    new_points: pd.DataFrame | None = None,
    observed_at_new: list[float] | None = None,
    response_column: str | None = None,
) -> dict[str, Any]:
    """Fit models, run ANOVA, compute effects, diagnose residuals.

    Parameters
    ----------
    design_matrix : DataFrame
        Factor settings per run.  May also contain the response column(s).
    responses : DataFrame, Series, or None
        Response column(s).  If *None*, ``response_column`` must name a
        column already present in *design_matrix*.
    model : str or None
        ``"main_effects"``, ``"interactions"``, ``"quadratic"``, an explicit
        formula, or *None* (defaults to ``"interactions"``).
    analysis_type : str or list[str]
        One or more of: ``"anova"``, ``"effects"``, ``"coefficients"``,
        ``"significance"``, ``"residual_diagnostics"``, ``"lack_of_fit"``,
        ``"curvature_test"``, ``"model_selection"``, ``"box_cox"``,
        ``"lenth_method"``, ``"confidence_intervals"``, ``"prediction"``,
        ``"confirmation_test"``.
    significance_level : float
        Default 0.05.
    transform : str or None
        ``"log"``, ``"sqrt"``, ``"inverse"``, ``"box_cox"``, or ``None``.
    coding : str
        ``"coded"`` or ``"actual"``.
    new_points : DataFrame or None
        For prediction or confirmation testing.
    observed_at_new : list[float] or None
        Observed values at *new_points* (for confirmation testing).
    response_column : str or None
        Name of the response column when it lives inside *design_matrix*.

    Returns
    -------
    dict[str, Any]
        Results keyed by analysis type.  Always includes ``"model_summary"``
        with R², adj-R², pred-R², and adequate precision.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments.analysis import analyze_experiment
    >>> df = pd.DataFrame({
    ...     "A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1],
    ...     "y": [28, 36, 18, 31],
    ... })
    >>> result = analyze_experiment(df, response_column="y", analysis_type="coefficients")
    >>> result["coefficients"][0]["term"]
    'Intercept'
    """
    # --- Assemble data -------------------------------------------------
    df = design_matrix.copy()

    if responses is not None:
        if isinstance(responses, pd.Series):
            responses = responses.to_frame()
        for col in responses.columns:
            df[col] = responses[col].values

    # Identify response column
    if response_column is None:
        # If responses was provided, use its first column
        if responses is not None:
            response_col = responses.columns[0] if isinstance(responses, pd.DataFrame) else responses.name
        else:
            raise ValueError("Must provide either 'responses' or 'response_column'.")
    else:
        response_col = response_column

    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found in data.")

    # Factor columns = everything except the response
    factor_cols = [c for c in df.columns if c != response_col]

    # --- Apply transform -----------------------------------------------
    if transform == "log":
        df[response_col] = np.log(df[response_col])
    elif transform == "sqrt":
        df[response_col] = np.sqrt(df[response_col])
    elif transform == "inverse":
        df[response_col] = 1.0 / df[response_col]
    elif transform == "box_cox":
        from scipy.stats import boxcox  # noqa: PLC0415

        vals = df[response_col].values.astype(float)
        if np.all(vals > 0):
            df[response_col], _ = boxcox(vals)

    # --- Build formula and fit -----------------------------------------
    formula = build_formula(response_col, factor_cols, model)
    ols_result = smf.ols(formula, data=df).fit()

    # --- Normalize analysis_type to list -------------------------------
    types = [analysis_type] if isinstance(analysis_type, str) else list(analysis_type)

    # Validate
    unknown = [t for t in types if t not in _ANALYSIS_REGISTRY]
    if unknown:
        available = sorted(_ANALYSIS_REGISTRY.keys())
        raise ValueError(f"Unknown analysis_type(s): {unknown}. Available: {available}")

    # --- Always include model summary ----------------------------------
    pred_r2 = _compute_pred_r_squared(ols_result)
    adeq_prec = _compute_adequate_precision(ols_result)

    results: dict[str, Any] = {
        "model_summary": {
            "formula": formula,
            "r_squared": float(ols_result.rsquared),
            "r_squared_adj": float(ols_result.rsquared_adj),
            "r_squared_pred": pred_r2,
            "adequate_precision": adeq_prec,
            "n_obs": int(ols_result.nobs),
            "df_model": int(ols_result.df_model),
            "df_residual": int(ols_result.df_resid),
            "mse_residual": float(ols_result.mse_resid),
        }
    }

    # --- Dispatch requested analyses -----------------------------------
    for t in types:
        if t == "anova":
            results.update(_run_anova(ols_result))
        elif t == "effects":
            results.update(_run_effects(ols_result))
        elif t == "coefficients":
            results.update(_run_coefficients(ols_result))
        elif t == "significance":
            results.update(_run_significance(ols_result, significance_level))
        elif t == "residual_diagnostics":
            results.update(_run_residual_diagnostics(ols_result))
        elif t == "lack_of_fit":
            results.update(_run_lack_of_fit(ols_result, df, response_col))
        elif t == "curvature_test":
            results.update(_run_curvature_test(ols_result, df, response_col, factor_cols))
        elif t == "model_selection":
            results.update(_run_model_selection(df, response_col, factor_cols))
        elif t == "box_cox":
            results.update(_run_box_cox(df, response_col))
        elif t == "lenth_method":
            results.update(_run_lenth_method(ols_result))
        elif t == "confidence_intervals":
            results.update(_run_confidence_intervals(ols_result, significance_level))
        elif t == "prediction":
            if new_points is None:
                results["prediction"] = {"error": "new_points is required for prediction."}
            else:
                results.update(_run_prediction(ols_result, new_points, significance_level))
        elif t == "confirmation_test":
            if new_points is None or observed_at_new is None:
                results["confirmation_test"] = {"error": "new_points and observed_at_new are required."}
            else:
                results.update(_run_confirmation_test(ols_result, new_points, observed_at_new, significance_level))

    return results
