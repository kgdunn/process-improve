"""(c) Kevin Dunn, 2010-2025. MIT License.

Agent-callable tool wrappers for robust univariate statistics.

Each function in this module is decorated with ``@tool_spec`` so it can be
passed directly to an LLM tool-use API (e.g. Anthropic ``tools=``).
The wrappers accept plain JSON-serialisable inputs (lists of numbers, strings,
booleans) and always return JSON-serialisable ``dict`` results.

Import all specs at once::

    from process_improve.univariate.tools import get_univariate_tool_specs
    # or get everything registered so far
    from process_improve.tool_spec import get_tool_specs

Dispatch a tool call returned by the model::

    from process_improve.tool_spec import execute_tool_call
    result = execute_tool_call(block.name, block.input)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from process_improve.tool_spec import _TOOL_REGISTRY, get_tool_specs, tool_spec
from process_improve.univariate.metrics import (
    Sn,
    median_abs_deviation,
    normality_check,
    outlier_detection_multiple,
    summary_stats,
    t_value,
    t_value_cdf,
    ttest_difference_calculate,
    ttest_paired_difference_calculate,
    within_between_standard_deviation,
)

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNIVARIATE_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _UNIVARIATE_TOOL_NAMES.append(name)


def _clean(value: Any) -> Any:
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool_spec(
    name="robust_summary_stats",
    description=(
        "Compute comprehensive summary statistics for a list of numeric values. "
        "Returns both classical (mean, std) and robust (median, Sn scale estimator) measures "
        "of location and spread, along with percentiles, IQR, and relative standard deviation. "
        "Use the 'robust' method (default) when data may contain outliers. "
        "Use 'classical' only when you are certain the data are clean and normally distributed."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numeric observations. Missing values should be omitted.",
                    "minItems": 2,
                },
                "method": {
                    "type": "string",
                    "enum": ["robust", "classical"],
                    "description": (
                        "Which method to use for the primary 'center' and 'spread' outputs. "
                        "'robust' uses the median and Sn scale estimator (default). "
                        "'classical' uses the mean and standard deviation."
                    ),
                },
            },
            "required": ["values"],
        }
    },
    examples="""
    # "Give me summary statistics for [10, 12, 11, 9, 100]"
        -> ``robust_summary_stats(values=[10, 12, 11, 9, 100])``

    # "Summarise this batch result using classical statistics: [5.1, 4.9, 5.0, 5.2]"
        -> ``robust_summary_stats(values=[5.1, 4.9, 5.0, 5.2], method="classical")``
    """,
)
def robust_summary_stats(*, values: list[float], method: str = "robust") -> dict:
    """Compute summary statistics; see tool spec for parameter details."""
    arr = np.asarray(values, dtype=float)
    result = summary_stats(arr, method=method)
    return _clean(result)


_register("robust_summary_stats")


@tool_spec(
    name="detect_outliers",
    description=(
        "Detect outliers in a list of numeric values using the Generalised ESD (Extreme Studentised "
        "Deviate) test. "
        "Returns the zero-based indices and values of any detected outliers. "
        "The robust variant (default) uses the median and MAD instead of mean and std, making the "
        "test itself resistant to the influence of the very outliers it is trying to detect. "
        "Set max_outliers_to_detect to at least the number of outliers you suspect; the algorithm "
        "will find the actual count up to that limit."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numeric observations to test.",
                    "minItems": 3,
                },
                "max_outliers_to_detect": {
                    "type": "integer",
                    "description": (
                        "Upper bound on the number of outliers to search for "
                        "(default 5, must be < len(values))."
                    ),
                    "minimum": 1,
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance level for each individual test (default 0.05).",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
                "robust_variant": {
                    "type": "boolean",
                    "description": (
                        "When true (default), use median and MAD instead of mean and std. "
                        "Recommended when outliers may already be present."
                    ),
                },
            },
            "required": ["values"],
        }
    },
    examples="""
    # "Are there outliers in [1, 2, 2, 3, 2, 100]?"
        -> ``detect_outliers(values=[1, 2, 2, 3, 2, 100])``

    # "Find up to 3 outliers at the 1% level in my data"
        -> ``detect_outliers(values=[...], max_outliers_to_detect=3, alpha=0.01)``

    # "Run a classical (non-robust) outlier test"
        -> ``detect_outliers(values=[...], robust_variant=False)``
    """,
)
def detect_outliers(
    *,
    values: list[float],
    max_outliers_to_detect: int = 5,
    alpha: float = 0.05,
    robust_variant: bool = True,
) -> dict:
    """Detect outliers using the Generalised ESD test; see tool spec for details."""
    arr = np.asarray(values, dtype=float)
    max_k = min(max_outliers_to_detect, len(arr) - 1)
    outlier_indices, details = outlier_detection_multiple(
        arr,
        algorithm="esd",
        max_outliers_detected=max_k,
        alpha=alpha,
        robust_variant=robust_variant,
    )
    outlier_values = [float(values[i]) for i in outlier_indices]
    return {
        "outlier_indices": list(outlier_indices),
        "outlier_values": outlier_values,
        "n_outliers_found": len(outlier_indices),
        "alpha": alpha,
        "robust_variant": robust_variant,
    }


_register("detect_outliers")


@tool_spec(
    name="robust_scale_sn",
    description=(
        "Compute the Sn robust scale estimator for a list of numeric values. "
        "Sn is an alternative to the standard deviation that is highly resistant to outliers "
        "and does not assume symmetry around the median, unlike MAD. "
        "For normally distributed data with no outliers, Sn ≈ std. "
        "A large Sn relative to the mean suggests high variability or outliers. "
        "Reference: Rousseeuw & Croux (1993)."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numeric observations.",
                    "minItems": 2,
                },
            },
            "required": ["values"],
        }
    },
    examples="""
    # "What is the robust spread of [5, 6, 5, 7, 5, 6, 100]?"
        -> ``robust_scale_sn(values=[5, 6, 5, 7, 5, 6, 100])``
    """,
)
def robust_scale_sn(*, values: list[float]) -> dict:
    """Compute Sn robust scale estimator; see tool spec for details."""
    arr = np.asarray(values, dtype=float)
    sn_value = float(Sn(arr))
    mean = float(np.nanmean(arr))
    rsd = (sn_value / mean) if mean != 0 else None
    return _clean({"sn": sn_value, "rsd": rsd, "n": int(np.sum(~np.isnan(arr)))})


_register("robust_scale_sn")


@tool_spec(
    name="median_absolute_deviation",
    description=(
        "Compute the Median Absolute Deviation (MAD) for a list of numeric values. "
        "MAD = median(|x_i - median(x)|). "
        "With scale='normal' (default) the result is normalised to be consistent with the "
        "standard deviation for normally distributed data (divide by ~0.6745). "
        "MAD is more robust than the standard deviation but assumes approximate symmetry. "
        "Prefer Sn (robust_scale_sn) when symmetry cannot be assumed."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numeric observations.",
                    "minItems": 2,
                },
                "scale": {
                    "type": "string",
                    "enum": ["normal", "raw"],
                    "description": (
                        "'normal' (default): normalise so MAD ≈ std for Gaussian data. "
                        "'raw': return the raw median of absolute deviations."
                    ),
                },
            },
            "required": ["values"],
        }
    },
    examples="""
    # "What is the MAD of [10, 11, 12, 10, 9, 10, 50]?"
        -> ``median_absolute_deviation(values=[10, 11, 12, 10, 9, 10, 50])``

    # "Give me the raw (un-normalised) MAD"
        -> ``median_absolute_deviation(values=[...], scale="raw")``
    """,
)
def median_absolute_deviation(*, values: list[float], scale: str = "normal") -> dict:
    """Compute Median Absolute Deviation; see tool spec for details."""
    arr = np.asarray(values, dtype=float)
    scale_arg = "normal" if scale == "normal" else 1.0
    mad_value = float(median_abs_deviation(arr, scale=scale_arg))
    return _clean({"mad": mad_value, "scale": scale, "n": int(np.sum(~np.isnan(arr)))})


_register("median_absolute_deviation")


@tool_spec(
    name="normality_test",
    description=(
        "Test whether a list of numeric values is consistent with a normal (Gaussian) distribution "
        "using the Shapiro-Wilk test. "
        "Returns the test statistic, p-value, and a plain-language interpretation. "
        "A p-value below the significance threshold (default 0.05) provides evidence AGAINST "
        "normality. Failure to reject does not prove normality — it merely means the data are "
        "not inconsistent with it. "
        "Use this to decide whether robust or classical statistics are more appropriate."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numeric observations (3 to ~5000 values).",
                    "minItems": 3,
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance level for the decision (default 0.05).",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
            },
            "required": ["values"],
        }
    },
    examples="""
    # "Is this data normally distributed? [10.1, 9.8, 10.2, 9.9, 10.0]"
        -> ``normality_test(values=[10.1, 9.8, 10.2, 9.9, 10.0])``

    # "Test for normality at the 1% level"
        -> ``normality_test(values=[...], alpha=0.01)``
    """,
)
def normality_test(*, values: list[float], alpha: float = 0.05) -> dict:
    """Shapiro-Wilk normality test; see tool spec for details."""
    from scipy.stats import shapiro

    arr = np.asarray(values, dtype=float)
    arr_clean = arr[~np.isnan(arr)]
    stat, p_value = shapiro(arr_clean)
    is_normal = bool(p_value >= alpha)
    return _clean(
        {
            "statistic": float(stat),
            "p_value": float(p_value),
            "alpha": alpha,
            "is_normal": is_normal,
            "interpretation": (
                f"At the {alpha:.0%} significance level, the data appear consistent with "
                "a normal distribution."
                if is_normal
                else f"At the {alpha:.0%} significance level, there is evidence that the data "
                "are NOT normally distributed (p={p_value:.4f}). Consider using robust methods."
            ),
        }
    )


_register("normality_test")


@tool_spec(
    name="confidence_interval",
    description=(
        "Calculate a confidence interval for the center of a list of numeric values. "
        "The 'robust' method (default) uses the median as center and MAD as spread estimate, "
        "making it resistant to outliers. "
        "The 'classical' method uses the mean and standard deviation. "
        "The interval is computed using the t-distribution to account for small samples."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numeric observations.",
                    "minItems": 3,
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence level between 0 and 1 (default 0.95 → 95% CI).",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
                "method": {
                    "type": "string",
                    "enum": ["robust", "classical"],
                    "description": (
                        "'robust' (default): use median ± t * MAD / √n. "
                        "'classical': use mean ± t * std / √n."
                    ),
                },
            },
            "required": ["values"],
        }
    },
    examples="""
    # "What is the 95% confidence interval for [10, 11, 12, 10, 9, 10]?"
        -> ``confidence_interval(values=[10, 11, 12, 10, 9, 10])``

    # "Give me a 99% classical confidence interval"
        -> ``confidence_interval(values=[...], confidence_level=0.99, method="classical")``
    """,
)
def confidence_interval(
    *,
    values: list[float],
    confidence_level: float = 0.95,
    method: str = "robust",
) -> dict:
    """Compute a confidence interval; see tool spec for details."""
    arr = np.asarray(values, dtype=float)
    arr_clean = arr[~np.isnan(arr)]
    n = len(arr_clean)
    if method == "robust":
        center = float(np.median(arr_clean))
        spread = float(median_abs_deviation(arr_clean))
    else:
        center = float(np.mean(arr_clean))
        spread = float(np.std(arr_clean, ddof=1))
    ct = float(t_value(1 - (1 - confidence_level) / 2, n - 1))
    margin = ct * spread / np.sqrt(n)
    return _clean(
        {
            "lower": center - margin,
            "center": center,
            "upper": center + margin,
            "margin_of_error": margin,
            "confidence_level": confidence_level,
            "method": method,
            "n": n,
        }
    )


_register("confidence_interval")


@tool_spec(
    name="ttest_two_samples",
    description=(
        "Perform an unpaired (independent samples) two-sided t-test to determine whether the "
        "means of two groups differ significantly. "
        "Returns the z/t statistic, p-value, confidence interval for the difference "
        "(group_b_mean − group_a_mean), and degrees of freedom. "
        "The groups must be independent (different subjects/items). "
        "Use ttest_paired_samples instead when each observation in group A is matched to one in B."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "group_a": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Observations for group A (the reference / baseline group).",
                    "minItems": 2,
                },
                "group_b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Observations for group B (the comparison group).",
                    "minItems": 2,
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence level for the interval (default 0.95).",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
            },
            "required": ["group_a", "group_b"],
        }
    },
    examples="""
    # "Is the average yield different between process A [102,98,100] and process B [110,112,108]?"
        -> ``ttest_two_samples(group_a=[102,98,100], group_b=[110,112,108])``

    # "t-test at 99% confidence level"
        -> ``ttest_two_samples(group_a=[...], group_b=[...], confidence_level=0.99)``
    """,
)
def ttest_two_samples(
    *,
    group_a: list[float],
    group_b: list[float],
    confidence_level: float = 0.95,
) -> dict:
    """Unpaired t-test for two independent samples; see tool spec for details."""
    a = pd.Series(np.asarray(group_a, dtype=float)).dropna()
    b = pd.Series(np.asarray(group_b, dtype=float)).dropna()
    result = ttest_difference_calculate(a, b, conflevel=confidence_level)
    result["confidence_level"] = confidence_level
    significant = bool(result["p value"] < (1 - confidence_level))
    result["significant"] = significant
    result["interpretation"] = (
        f"The difference (B − A = {result['Group B average'] - result['Group A average']:.4g}) "
        + (
            f"IS statistically significant (p={result['p value']:.4f} < {1-confidence_level:.2f})."
            if significant
            else f"is NOT statistically significant (p={result['p value']:.4f} ≥ {1-confidence_level:.2f})."
        )
    )
    return _clean(result)


_register("ttest_two_samples")


@tool_spec(
    name="ttest_paired_samples",
    description=(
        "Perform a paired (repeated-measures) two-sided t-test to determine whether the "
        "mean of the differences between matched pairs is significantly different from zero. "
        "The difference is defined as before − after (group_a − group_b). "
        "Each position i in group_a must correspond to the same subject/unit as position i in "
        "group_b (e.g. before/after measurements on the same individual). "
        "Use ttest_two_samples instead for independent groups."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "group_a": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Before / reference measurements (one per subject).",
                    "minItems": 2,
                },
                "group_b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "After / comparison measurements (same order as group_a).",
                    "minItems": 2,
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence level for the interval (default 0.95).",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
            },
            "required": ["group_a", "group_b"],
        }
    },
    examples="""
    # "Did the treatment improve scores? Before: [70,65,80], After: [75,70,82]"
        -> ``ttest_paired_samples(group_a=[70,65,80], group_b=[75,70,82])``
    """,
)
def ttest_paired_samples(
    *,
    group_a: list[float],
    group_b: list[float],
    confidence_level: float = 0.95,
) -> dict:
    """Paired t-test; see tool spec for details."""
    a = pd.Series(np.asarray(group_a, dtype=float))
    b = pd.Series(np.asarray(group_b, dtype=float))
    assert len(a) == len(b), "group_a and group_b must have the same length for a paired test."
    differences = a - b.values
    differences = differences.dropna()
    result = ttest_paired_difference_calculate(differences, conflevel=confidence_level)
    result["Group A average"] = float(a.mean())
    result["Group B average"] = float(b.mean())
    result["Group A number"] = int(a.count())
    result["Group B number"] = int(b.count())
    result["confidence_level"] = confidence_level
    significant = bool(result["p value"] < (1 - confidence_level))
    result["significant"] = significant
    result["interpretation"] = (
        f"The mean paired difference (A − B = {result['Differences mean']:.4g}) "
        + (
            f"IS statistically significant (p={result['p value']:.4f} < {1-confidence_level:.2f})."
            if significant
            else f"is NOT statistically significant (p={result['p value']:.4f} ≥ {1-confidence_level:.2f})."
        )
    )
    return _clean(result)


_register("ttest_paired_samples")


@tool_spec(
    name="within_between_variance",
    description=(
        "Decompose the total variability of a numeric variable into within-group and between-group "
        "components using one-way ANOVA variance decomposition. "
        "Useful for gauge R&R studies, reproducibility analysis, or any situation where you want "
        "to understand whether variation is mainly due to differences within repeated measurements "
        "of the same condition, or due to differences between conditions. "
        "Supply two parallel lists: 'values' (the measurements) and 'groups' (a group label for "
        "each measurement)."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Numeric measurement for each observation.",
                    "minItems": 3,
                },
                "groups": {
                    "type": "array",
                    "items": {},
                    "description": (
                        "Group label for each observation (same length as values). "
                        "Can be strings, integers, or any hashable type."
                    ),
                    "minItems": 3,
                },
            },
            "required": ["values", "groups"],
        }
    },
    examples="""
    # "Measurements on day 1: [101,102] and day 2: [94,95]. How much variation is within vs between days?"
        -> ``within_between_variance(values=[101,102,94,95], groups=[1,1,2,2])``

    # "Operator study: Alice measured [10.1,10.2,10.0], Bob measured [10.5,10.4,10.6]"
        -> ``within_between_variance(values=[10.1,10.2,10.0,10.5,10.4,10.6], groups=["Alice","Alice","Alice","Bob","Bob","Bob"])``
    """,
)
def within_between_variance(
    *,
    values: list[float],
    groups: list[Any],
) -> dict:
    """Within- and between-group variance decomposition; see tool spec for details."""
    assert len(values) == len(groups), "'values' and 'groups' must have the same length."
    df = pd.DataFrame({"measured": values, "repeat": groups})
    result = within_between_standard_deviation(df, measured="measured", repeat="repeat")
    return _clean(result)


_register("within_between_variance")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_univariate_tool_specs() -> list[dict]:
    """Return tool specs for all univariate tools registered in this module."""
    return get_tool_specs(names=_UNIVARIATE_TOOL_NAMES)
