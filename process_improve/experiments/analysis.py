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
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

# ENG-02: the per-analysis implementations now live in ``_analyses``; they are
# re-exported here so the ``analyze_experiment`` dispatcher (and external
# importers such as ``tests/test_sec09`` reaching for ``_run_residual_diagnostics``)
# keep working unchanged.
from process_improve.experiments._analyses._shared import (
    _compute_adequate_precision,
    _compute_pred_r_squared,
)
from process_improve.experiments._analyses.box_cox import _run_box_cox
from process_improve.experiments._analyses.curvature import _run_curvature_test
from process_improve.experiments._analyses.diagnostics import _run_residual_diagnostics
from process_improve.experiments._analyses.lack_of_fit import _run_lack_of_fit
from process_improve.experiments._analyses.lenth import _run_lenth_method
from process_improve.experiments._analyses.model_selection import _run_model_selection
from process_improve.experiments._analyses.ols_extractors import (
    _run_anova,
    _run_coefficients,
    _run_confidence_intervals,
    _run_effects,
    _run_significance,
)
from process_improve.experiments._analyses.prediction import _run_confirmation_test, _run_prediction
from process_improve.experiments.models import validate_formula_is_safe, validate_identifier_is_safe

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

    ols_result: RegressionResultsWrapper = None
    formula: str = ""
    results: dict[str, Any] = field(default_factory=dict)


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

    # User-supplied names (response_column / design_matrix dict keys) are
    # interpolated into the patsy formula, so reject anything that is not a
    # plain identifier before it can become an injection vector (SEC-14).
    validate_identifier_is_safe(response_col)

    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found in data.")

    # Factor columns = everything except the response
    factor_cols = [c for c in df.columns if c != response_col]
    for col in factor_cols:
        validate_identifier_is_safe(col)

    # --- Apply transform -----------------------------------------------
    if transform == "log":
        df[response_col] = np.log(df[response_col])
    elif transform == "sqrt":
        df[response_col] = np.sqrt(df[response_col])
    elif transform == "inverse":
        # SEC-26 (#275): a zero in the response column would produce
        # inf, then a downstream LinAlgError whose message would leak
        # via the tool wrapper. Reject up front with a clear message.
        if (df[response_col] == 0).any():
            raise ValueError(
                f"transform='inverse' is undefined when the response column "
                f"{response_col!r} contains zero. Remove the zero observations "
                "or pick a different transform."
            )
        df[response_col] = 1.0 / df[response_col]
    elif transform == "box_cox":
        from scipy.stats import boxcox  # noqa: PLC0415

        vals = df[response_col].values.astype(float)
        if np.all(vals > 0):
            df[response_col], _ = boxcox(vals)

    # --- Build formula and fit -----------------------------------------
    formula = build_formula(response_col, factor_cols, model)
    # Patsy evaluates formula terms as Python, so a custom ``model`` string is a
    # code-execution vector. Permit only a safe Wilkinson formula, optionally
    # with I()/Q() over data columns (the ``quadratic`` shorthand needs it).
    validate_formula_is_safe(formula, df.columns, allow_transforms=True)
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
