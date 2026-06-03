# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``fit_linear_model`` (ENG-02)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class FitLinearModelInput(BaseModel):
    """Input contract for ``fit_linear_model``."""

    model_config = ConfigDict(extra="forbid")

    formula: str = Field(
        ...,
        description=(
            "Model formula in Wilkinson notation, e.g. 'y ~ A*B*C'. "
            "The left-hand side is the response variable name, the right-hand "
            "side specifies the terms. '*' expands to main effects and all "
            "interactions; ':' denotes a specific interaction; '+' adds terms."
        ),
    )
    data: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "List of dictionaries, one per experimental run. Each dict must "
            "contain keys for every factor and the response variable referenced "
            "in the formula. Example: [{'A': -1, 'B': -1, 'y': 45.2}, ...]"
        ),
    )


@tool_spec(
    name="fit_linear_model",
    description=(
        "Fit a linear model to experimental data using a formula specification. "
        "The formula uses Wilkinson notation, e.g. 'y ~ A*B*C' for a full factorial model "
        "with all main effects and interactions. Use 'y ~ A + B' for main effects only, or "
        "'y ~ A*B' for main effects plus the A:B interaction. "
        "The data should be a list of dictionaries, where each dictionary represents one "
        "experimental run with column names as keys. Factor columns should contain -1/+1 "
        "coded values (or real values), and there must be a response column matching the "
        "left-hand side of the formula. "
        "Returns the fitted coefficients (name, estimate), R-squared, and a text summary."
    ),
    input_model=FitLinearModelInput,
    examples="""
    # "Fit a model y ~ A*B to my 2^2 factorial data"
        -> ``fit_linear_model(formula="y ~ A*B",
                data=[{"A":-1,"B":-1,"y":28}, {"A":1,"B":-1,"y":36},
                      {"A":-1,"B":1,"y":18}, {"A":1,"B":1,"y":31}])``

    # "Fit a main-effects-only model"
        -> ``fit_linear_model(formula="y ~ A + B + C", data=[...])``
    """,
    category="experiments",
)
def fit_linear_model(spec: FitLinearModelInput) -> dict[str, Any]:
    """Fit a linear model to experimental data."""
    try:
        from process_improve.config import settings  # noqa: PLC0415
        from process_improve.experiments.models import lm, validate_formula_is_safe  # noqa: PLC0415
        from process_improve.experiments.structures import Expt  # noqa: PLC0415

        if len(spec.data) > settings.max_matrix_rows:
            return {
                "error": (
                    f"data has {len(spec.data)} rows; the cap is "
                    f"settings.max_matrix_rows={settings.max_matrix_rows}."
                )
            }
        if len(spec.formula) > settings.max_formula_chars:
            return {
                "error": (
                    f"formula is {len(spec.formula)} chars; the cap is "
                    f"settings.max_formula_chars={settings.max_formula_chars}."
                )
            }

        df = pd.DataFrame(spec.data)

        # Patsy evaluates formula terms as Python expressions, so a formula from an
        # untrusted caller is a code-execution vector. Only allow a plain Wilkinson
        # formula over the columns actually present in the data.
        validate_formula_is_safe(spec.formula, df.columns)

        expt_data = Expt(df)
        expt_data.pi_title = None
        expt_data.pi_source = None
        expt_data.pi_units = None

        model = lm(spec.formula, expt_data)

        params = model.get_parameters(drop_intercept=True)
        if isinstance(params, pd.Series):
            coefficients = [
                {"name": str(name), "estimate": float(value)}
                for name, value in params.items()
            ]
        else:
            coefficients = params.to_dict(orient="records")

        r2 = float(model._OLS.rsquared)

        smry = model.summary(print_to_screen=False)
        summary_text = str(smry)

        return clean({
            "coefficients": coefficients,
            "r2": r2,
            "summary_text": summary_text,
        })
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool fit_linear_model failed")
        return {"error": str(e)}


_register("fit_linear_model")
