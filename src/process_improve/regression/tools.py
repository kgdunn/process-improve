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
    n_boot: int | None = Field(
        None,
        ge=1,
        le=1_000,
        description=(
            "Bootstrap resamples behind the intervals. By default 999 up to 197 points, then fewer as the "
            "points grow (156 at 500 points), down to 39 from 1,000 points up; fewer resamples give rougher "
            "intervals. Above 1,000 points each resample still costs time in proportion to the square of the "
            "number of points. The output reports the number drawn."
        ),
    )
    random_state: int = Field(0, ge=0, le=2**31 - 1, description="Seed; the same seed reproduces the intervals.")


#: The fewest resamples the tool draws by default: 0.025 * (39 + 1) = 1, so each limit of a 95%
#: percentile interval rests on the most extreme of 39 resampled slopes. Fewer would make the
#: interval the range of a handful of slopes, or a single slope.
_MIN_DEFAULT_N_BOOT = 39


def _default_n_boot(n_points: int) -> int:
    """Return the resamples the tool draws by default for ``n_points`` points.

    Each resample costs time in proportion to ``n_points ** 2``, so a fixed 999 took the tool past its
    10 s timeout above about 800 points (#1095). Up to 1,000 points the default keeps the work to that
    of 39 resamples of 1,000 points; above, it draws 39 and the time grows as ``n_points ** 2``.
    The count depends only on ``n_points``, so a seed still reproduces the intervals.
    """
    # shortcut: sized for a machine 3 times slower than one measured at 12.9 ms per fit of 1,000 points
    # and 467 ms per fit of 5,000 (13 to 19 ns per pairwise slope). There, 39 resamples of 1,000 points
    # take about 1.5 s of the 10 s timeout, but 39 of 5,000 take about a minute, so a hosted call above
    # about 1,500 points can still time out. Upgrade to an O(N log N) repeated median, or to
    # least-squares intervals for large N, if hosted callers need more points.
    return min(999, max(_MIN_DEFAULT_N_BOOT, _MIN_DEFAULT_N_BOOT * 1_000**2 // n_points**2))


@tool_spec(
    name="robust_regression",
    description=(
        "Fit a robust simple linear regression between x and y variables using the repeated "
        "median slope estimator. Unlike ordinary least squares, the repeated median method is "
        "highly resistant to outliers - up to half of the data can be contaminated before "
        "the slope estimate breaks down (Siegel's repeated median has a 50% breakdown point). "
        "Returns slope, intercept, R-squared, standard errors, a confidence interval for the slope, "
        "prediction intervals, and fitted values. The slope interval and the prediction intervals "
        "are percentile-bootstrap intervals over n_boot resamples of the (x, y) pairs, each refitted "
        "with the repeated median, so like the slope they resist outliers. R-squared and the "
        "standard errors use least-squares formulas on the residuals around the robust line, so "
        "an outlier raises the standard errors and lowers R-squared even when it does not move "
        "the slope. Each resample costs time in proportion to the square of the number of points. "
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
    rng={"uses_rng": True, "seed_param": "random_state", "default_seed": 0},
)
def robust_regression(spec: RobustRegressionInput) -> dict[str, Any]:
    """Fit a robust simple linear regression."""
    from process_improve.regression.methods import robust_regression as _robust_regression  # noqa: PLC0415

    try:
        x_arr = np.asarray(spec.x, dtype=float)
        y_arr = np.asarray(spec.y, dtype=float)
        n_boot = _default_n_boot(len(spec.x)) if spec.n_boot is None else spec.n_boot

        result = _robust_regression(
            x_arr,
            y_arr,
            fit_intercept=spec.fit_intercept,
            conflevel=spec.confidence_level,
            n_boot=n_boot,
            random_state=spec.random_state,
            intervals="bootstrap",
        )
        if result["N"] is None:
            # The fit hands back its empty result when two or fewer complete pairs remain.
            return {"error": "robust_regression needs at least three complete (x, y) pairs to fit a line."}

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
            "n_boot": n_boot,
        }

        # A fit with three or more complete pairs always returns both as arrays.
        out["slope_confidence_interval"] = result["conf_intervals"].tolist()[0]
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
