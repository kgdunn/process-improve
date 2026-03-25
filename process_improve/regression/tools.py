"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for robust regression.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from process_improve.tool_spec import clean, get_tool_specs, tool_spec

_REGRESSION_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _REGRESSION_TOOL_NAMES.append(name)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool_spec(
    name="robust_regression",
    description=(
        "Fit a robust simple linear regression between x and y variables using the repeated "
        "median slope estimator. Unlike ordinary least squares, the repeated median method is "
        "highly resistant to outliers — up to ~29% of the data can be contaminated without "
        "affecting the fit. "
        "Returns slope, intercept, R-squared, standard errors, prediction intervals, and "
        "fitted values. "
        "Use this when you suspect outliers in the data or want a more reliable fit."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Independent variable (predictor) values.",
                    "minItems": 3,
                },
                "y": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Dependent variable (response) values. Must be same length as x.",
                    "minItems": 3,
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence level for intervals (default 0.95).",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
                "fit_intercept": {
                    "type": "boolean",
                    "description": "If true (default), fit an intercept term. If false, force through origin.",
                },
            },
            "required": ["x", "y"],
        }
    },
    examples="""
    # "Fit a robust line to x=[1,2,3,4,5] and y=[2.1,4.0,5.9,8.1,10.0]"
        -> ``robust_regression(x=[1,2,3,4,5], y=[2.1,4.0,5.9,8.1,10.0])``

    # "Fit without intercept at 99% confidence"
        -> ``robust_regression(x=[...], y=[...], fit_intercept=false, confidence_level=0.99)``
    """,
    category="regression",
)
def robust_regression(
    *,
    x: list[float],
    y: list[float],
    confidence_level: float = 0.95,
    fit_intercept: bool = True,
) -> dict[str, Any]:
    """Fit a robust simple linear regression."""
    from process_improve.regression.methods import robust_regression as _robust_regression  # noqa: PLC0415

    try:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        result = _robust_regression(
            x_arr, y_arr,
            fit_intercept=fit_intercept,
            conflevel=confidence_level,
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
            "confidence_level": confidence_level,
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
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


_register("robust_regression")


@tool_spec(
    name="repeated_median",
    description=(
        "Compute the repeated median slope between x and y vectors. "
        "This is a highly robust slope estimator — the median of medians of all pairwise slopes. "
        "It gives just the slope (no intercept, no full regression output). "
        "Use robust_regression if you need a complete regression analysis."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Independent variable values.",
                    "minItems": 3,
                },
                "y": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Dependent variable values. Must be same length as x.",
                    "minItems": 3,
                },
            },
            "required": ["x", "y"],
        }
    },
    examples="""
    # "What is the robust slope between x=[1,2,3,4,5] and y=[2.1,4.0,5.9,8.1,10.0]?"
        -> ``repeated_median(x=[1,2,3,4,5], y=[2.1,4.0,5.9,8.1,10.0])``
    """,
    category="regression",
)
def repeated_median(
    *,
    x: list[float],
    y: list[float],
) -> dict[str, Any]:
    """Compute the repeated median slope."""
    from process_improve.regression.methods import repeated_median_slope  # noqa: PLC0415

    try:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        slope = float(repeated_median_slope(x_arr, y_arr))
        return clean({"slope": slope, "n": len(x)})
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


_register("repeated_median")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_regression_tool_specs() -> list[dict]:
    """Return tool specs for all regression tools registered in this module."""
    return get_tool_specs(names=_REGRESSION_TOOL_NAMES)
