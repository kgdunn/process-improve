"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for robust regression.

Pydantic input contract (ENG-04 / ENG-10): each tool pairs its
``@tool_spec`` decorator with a ``BaseModel`` carrying
``ConfigDict(extra="forbid")``; the function receives the parsed
model as its single positional argument.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from process_improve.tool_spec import clean, get_tool_specs, tool_spec

_REGRESSION_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _REGRESSION_TOOL_NAMES.append(name)


# ---------------------------------------------------------------------------
# robust_regression
# ---------------------------------------------------------------------------


class RobustRegressionInput(BaseModel):
    """Input contract for ``robust_regression``."""

    model_config = ConfigDict(extra="forbid")

    x: list[float] = Field(
        ...,
        min_length=3,
        description="Independent variable (predictor) values.",
    )
    y: list[float] = Field(
        ...,
        min_length=3,
        description="Dependent variable (response) values. Must be same length as x.",
    )
    confidence_level: float = Field(
        0.95,
        gt=0,
        lt=1,
        description="Confidence level for intervals (default 0.95).",
    )
    fit_intercept: bool = Field(
        True,
        description="If true (default), fit an intercept term. If false, force through origin.",
    )


@tool_spec(
    name="robust_regression",
    description=(
        "Fit a robust simple linear regression between x and y variables using the repeated "
        "median slope estimator. Unlike ordinary least squares, the repeated median method is "
        "highly resistant to outliers - up to ~29% of the data can be contaminated without "
        "affecting the fit. "
        "Returns slope, intercept, R-squared, standard errors, prediction intervals, and "
        "fitted values. "
        "Use this when you suspect outliers in the data or want a more reliable fit."
    ),
    input_model=RobustRegressionInput,
    examples="""
    # "Fit a robust line to x=[1,2,3,4,5] and y=[2.1,4.0,5.9,8.1,10.0]"
        -> ``robust_regression(x=[1,2,3,4,5], y=[2.1,4.0,5.9,8.1,10.0])``

    # "Fit without intercept at 99% confidence"
        -> ``robust_regression(x=[...], y=[...], fit_intercept=false, confidence_level=0.99)``
    """,
    category="regression",
)
def robust_regression(spec: RobustRegressionInput) -> dict[str, Any]:
    """Fit a robust simple linear regression."""
    from process_improve.regression.methods import robust_regression as _robust_regression  # noqa: PLC0415

    try:
        x_arr = np.asarray(spec.x, dtype=float)
        y_arr = np.asarray(spec.y, dtype=float)

        result = _robust_regression(
            x_arr, y_arr,
            fit_intercept=spec.fit_intercept,
            conflevel=spec.confidence_level,
        )

        out: dict[str, Any] = {
            "slope": result["coefficients"][0],
            "intercept": result["intercept"],
            "r2": result["R2"],
            "standard_error": result["SE"],
            "slope_std_error": result["standard_errors"][0],
            "n": result["N"],
            "fitted_values": list(result["fitted_values"]),
            "residuals": list(result["residuals"]),
            "confidence_level": spec.confidence_level,
        }

        if result.get("conf_intervals") is not None:
            ci = result["conf_intervals"]
            if hasattr(ci, "tolist"):
                ci = ci.tolist()
            out["slope_confidence_interval"] = ci[0] if ci else None

        if result.get("pi_range") is not None and hasattr(result["pi_range"], "tolist"):
            pi = result["pi_range"].tolist()
            out["prediction_interval_x"] = [row[0] for row in pi]
            out["prediction_interval_lower"] = [row[1] for row in pi]
            out["prediction_interval_upper"] = [row[2] for row in pi]

        return clean(out)
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        return {"error": str(exc)}


_register("robust_regression")


# ---------------------------------------------------------------------------
# repeated_median
# ---------------------------------------------------------------------------


class RepeatedMedianInput(BaseModel):
    """Input contract for ``repeated_median``."""

    model_config = ConfigDict(extra="forbid")

    x: list[float] = Field(
        ...,
        min_length=3,
        description="Independent variable values.",
    )
    y: list[float] = Field(
        ...,
        min_length=3,
        description="Dependent variable values. Must be same length as x.",
    )


@tool_spec(
    name="repeated_median",
    description=(
        "Compute the repeated median slope between x and y vectors. "
        "This is a highly robust slope estimator - the median of medians of all pairwise slopes. "
        "It gives just the slope (no intercept, no full regression output). "
        "Use robust_regression if you need a complete regression analysis."
    ),
    input_model=RepeatedMedianInput,
    examples="""
    # "What is the robust slope between x=[1,2,3,4,5] and y=[2.1,4.0,5.9,8.1,10.0]?"
        -> ``repeated_median(x=[1,2,3,4,5], y=[2.1,4.0,5.9,8.1,10.0])``
    """,
    category="regression",
)
def repeated_median(spec: RepeatedMedianInput) -> dict[str, Any]:
    """Compute the repeated median slope."""
    from process_improve.regression.methods import repeated_median_slope  # noqa: PLC0415

    try:
        x_arr = np.asarray(spec.x, dtype=float)
        y_arr = np.asarray(spec.y, dtype=float)
        slope = float(repeated_median_slope(x_arr, y_arr))
        return clean({"slope": slope, "n": len(spec.x)})
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        return {"error": str(exc)}


_register("repeated_median")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_regression_tool_specs() -> list[dict]:
    """Return tool specs for all regression tools registered in this module."""
    return get_tool_specs(names=_REGRESSION_TOOL_NAMES)
