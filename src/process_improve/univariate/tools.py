"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for robust univariate statistics.

Each function in this module is decorated with ``@tool_spec`` so it can be
passed directly to an LLM tool-use API (e.g. Anthropic ``tools=``).
The wrappers accept plain JSON-serialisable inputs (lists of numbers, strings,
booleans) and always return JSON-serialisable ``dict`` results.

Pydantic input contract (ENG-04 / ENG-10)
-----------------------------------------

Every tool below pairs its ``@tool_spec`` decorator with a pydantic
``BaseModel`` subclass that is the single source of truth for both the
function's call signature and the MCP JSON Schema. The model carries
``ConfigDict(extra="forbid")`` so unknown kwargs are rejected at
``execute_tool_call`` time -- closing the SEC-15 kwarg-injection vector
at the schema layer.

The function then receives the parsed model as its single positional
argument. Migration from the legacy ``input_schema={...}`` form was
done in PR #328; subsequent packages follow.

Import all specs at once::

    from process_improve.univariate.tools import get_univariate_tool_specs
    # or get everything registered so far
    from process_improve.tool_spec import get_tool_specs

Dispatch a tool call returned by the model::

    from process_improve.tool_spec import execute_tool_call
    result = execute_tool_call(block.name, block.input)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.tool_spec import clean, get_tool_specs, tool_spec
from process_improve.univariate.metrics import (
    Sn,
    detect_outliers_esd,
    median_absolute_deviation,
    summary_stats,
    t_value,
    ttest_independent,
    ttest_paired,
    variance_decomposition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNIVARIATE_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _UNIVARIATE_TOOL_NAMES.append(name)


# ---------------------------------------------------------------------------
# robust_summary_stats
# ---------------------------------------------------------------------------


class RobustSummaryStatsInput(BaseModel):
    """Input contract for ``robust_summary_stats``."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        ...,
        min_length=2,
        description="List of numeric observations. Missing values should be omitted.",
    )
    method: Literal["robust", "classical"] = Field(
        "robust",
        description=(
            "Which method to use for the primary 'center' and 'spread' outputs. "
            "'robust' uses the median and Sn scale estimator (default). "
            "'classical' uses the mean and standard deviation."
        ),
    )


@tool_spec(
    name="robust_summary_stats",
    description=(
        "Compute comprehensive summary statistics for a list of numeric values. "
        "Returns both classical (mean, std) and robust (median, Sn scale estimator) measures "
        "of location and spread, along with percentiles, IQR, and relative standard deviation. "
        "Use the 'robust' method (default) when data may contain outliers. "
        "Use 'classical' only when you are certain the data are clean and normally distributed."
    ),
    input_model=RobustSummaryStatsInput,
    examples="""
    # "Give me summary statistics for [10, 12, 11, 9, 100]"
        -> ``robust_summary_stats(values=[10, 12, 11, 9, 100])``

    # "Summarise this batch result using classical statistics: [5.1, 4.9, 5.0, 5.2]"
        -> ``robust_summary_stats(values=[5.1, 4.9, 5.0, 5.2], method="classical")``
    """,
    category="univariate",
)
def robust_summary_stats(spec: RobustSummaryStatsInput) -> dict:
    """Compute summary statistics; see tool spec for parameter details."""
    arr = np.asarray(spec.values, dtype=float)
    result = summary_stats(arr, method=spec.method)
    return clean(result)


_register("robust_summary_stats")


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------


class DetectOutliersInput(BaseModel):
    """Input contract for ``detect_outliers``."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        ...,
        min_length=3,
        description="List of numeric observations to test.",
    )
    max_outliers_to_detect: int = Field(
        5,
        ge=1,
        description=(
            "Upper bound on the number of outliers to search for "
            "(default 5, must be < len(values))."
        ),
    )
    alpha: float = Field(
        0.05,
        gt=0,
        lt=1,
        description="Significance level for each individual test (default 0.05).",
    )
    robust_variant: bool = Field(
        True,
        description=(
            "When true (default), use median and MAD instead of mean and std. "
            "Recommended when outliers may already be present."
        ),
    )


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
    input_model=DetectOutliersInput,
    examples="""
    # "Are there outliers in [1, 2, 2, 3, 2, 100]?"
        -> ``detect_outliers(values=[1, 2, 2, 3, 2, 100])``

    # "Find up to 3 outliers at the 1% level in my data"
        -> ``detect_outliers(values=[...], max_outliers_to_detect=3, alpha=0.01)``

    # "Run a classical (non-robust) outlier test"
        -> ``detect_outliers(values=[...], robust_variant=False)``
    """,
    category="univariate",
)
def detect_outliers(spec: DetectOutliersInput) -> dict:
    """Detect outliers using the Generalised ESD test; see tool spec for details."""
    arr = np.asarray(spec.values, dtype=float)
    max_k = min(spec.max_outliers_to_detect, len(arr) - 1)
    outlier_indices, _details = detect_outliers_esd(
        arr,
        algorithm="esd",
        max_outliers_detected=max_k,
        alpha=spec.alpha,
        robust_variant=spec.robust_variant,
    )
    outlier_values = [float(spec.values[i]) for i in outlier_indices]
    return {
        "outlier_indices": list(outlier_indices),
        "outlier_values": outlier_values,
        "n_outliers_found": len(outlier_indices),
        "alpha": spec.alpha,
        "robust_variant": spec.robust_variant,
    }


_register("detect_outliers")


# ---------------------------------------------------------------------------
# robust_scale_sn
# ---------------------------------------------------------------------------


class RobustScaleSnInput(BaseModel):
    """Input contract for ``robust_scale_sn``."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        ...,
        min_length=2,
        description="List of numeric observations.",
    )


@tool_spec(
    name="robust_scale_sn",
    description=(
        "Compute the Sn robust scale estimator for a list of numeric values. "
        "Sn is an alternative to the standard deviation that is highly resistant to outliers "
        "and does not assume symmetry around the median, unlike MAD. "
        "For normally distributed data with no outliers, Sn ~ std. "
        "A large Sn relative to the mean suggests high variability or outliers. "
        "Reference: Rousseeuw & Croux (1993)."
    ),
    input_model=RobustScaleSnInput,
    examples="""
    # "What is the robust spread of [5, 6, 5, 7, 5, 6, 100]?"
        -> ``robust_scale_sn(values=[5, 6, 5, 7, 5, 6, 100])``
    """,
    category="univariate",
)
def robust_scale_sn(spec: RobustScaleSnInput) -> dict:
    """Compute Sn robust scale estimator; see tool spec for details."""
    arr = np.asarray(spec.values, dtype=float)
    sn_value = float(Sn(arr))
    mean = float(np.nanmean(arr))
    rsd = (sn_value / mean) if mean != 0 else None
    return clean({"sn": sn_value, "rsd": rsd, "n": int(np.sum(~np.isnan(arr)))})


_register("robust_scale_sn")


# ---------------------------------------------------------------------------
# median_absolute_deviation
# ---------------------------------------------------------------------------


class MedianAbsoluteDeviationInput(BaseModel):
    """Input contract for ``median_absolute_deviation``."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        ...,
        min_length=2,
        description="List of numeric observations.",
    )
    scale: Literal["normal", "raw"] = Field(
        "normal",
        description=(
            "'normal' (default): normalise so MAD ~ std for Gaussian data. "
            "'raw': return the raw median of absolute deviations."
        ),
    )


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
    input_model=MedianAbsoluteDeviationInput,
    examples="""
    # "What is the MAD of [10, 11, 12, 10, 9, 10, 50]?"
        -> ``median_absolute_deviation(values=[10, 11, 12, 10, 9, 10, 50])``

    # "Give me the raw (un-normalised) MAD"
        -> ``median_absolute_deviation(values=[...], scale="raw")``
    """,
    category="univariate",
)
def median_absolute_deviation_tool(spec: MedianAbsoluteDeviationInput) -> dict:
    """Compute Median Absolute Deviation; see tool spec for details."""
    arr = np.asarray(spec.values, dtype=float)
    scale_arg: Any = "normal" if spec.scale == "normal" else 1.0
    mad_value = float(median_absolute_deviation(arr, scale=scale_arg))
    return clean({"mad": mad_value, "scale": spec.scale, "n": int(np.sum(~np.isnan(arr)))})


_register("median_absolute_deviation")


# ---------------------------------------------------------------------------
# normality_test
# ---------------------------------------------------------------------------


class NormalityTestInput(BaseModel):
    """Input contract for ``normality_test``."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        ...,
        min_length=3,
        description="List of numeric observations (3 to ~5000 values).",
    )
    alpha: float = Field(
        0.05,
        gt=0,
        lt=1,
        description="Significance level for the decision (default 0.05).",
    )


@tool_spec(
    name="normality_test",
    description=(
        "Test whether a list of numeric values is consistent with a normal (Gaussian) distribution "
        "using the Shapiro-Wilk test. "
        "Returns the test statistic, p-value, and a plain-language interpretation. "
        "A p-value below the significance threshold (default 0.05) provides evidence AGAINST "
        "normality. Failure to reject does not prove normality - it merely means the data are "
        "not inconsistent with it. "
        "Use this to decide whether robust or classical statistics are more appropriate."
    ),
    input_model=NormalityTestInput,
    examples="""
    # "Is this data normally distributed? [10.1, 9.8, 10.2, 9.9, 10.0]"
        -> ``normality_test(values=[10.1, 9.8, 10.2, 9.9, 10.0])``

    # "Test for normality at the 1% level"
        -> ``normality_test(values=[...], alpha=0.01)``
    """,
    category="univariate",
)
def normality_test(spec: NormalityTestInput) -> dict:
    """Shapiro-Wilk normality test; see tool spec for details."""
    from scipy.stats import shapiro  # noqa: PLC0415

    arr = np.asarray(spec.values, dtype=float)
    arr_clean = arr[~np.isnan(arr)]
    stat, p_value = shapiro(arr_clean)
    is_normal = bool(p_value >= spec.alpha)
    return clean(
        {
            "statistic": float(stat),
            "p_value": float(p_value),
            "alpha": spec.alpha,
            "is_normal": is_normal,
            "interpretation": (
                f"At the {spec.alpha:.0%} significance level, the data appear consistent with a normal distribution."
                if is_normal
                else f"At the {spec.alpha:.0%} significance level, there is evidence that the data "
                f"are NOT normally distributed (p={p_value:.4f}). Consider using robust methods."
            ),
        }
    )


_register("normality_test")


# ---------------------------------------------------------------------------
# confidence_interval
# ---------------------------------------------------------------------------


class ConfidenceIntervalInput(BaseModel):
    """Input contract for ``confidence_interval``."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        ...,
        min_length=3,
        description="List of numeric observations.",
    )
    confidence_level: float = Field(
        0.95,
        gt=0,
        lt=1,
        description="Confidence level between 0 and 1 (default 0.95 -> 95% CI).",
    )
    method: Literal["robust", "classical"] = Field(
        "robust",
        description=(
            "'robust' (default): use median +/- t * MAD / sqrt(n). "
            "'classical': use mean +/- t * std / sqrt(n)."
        ),
    )


@tool_spec(
    name="confidence_interval",
    description=(
        "Calculate a confidence interval for the center of a list of numeric values. "
        "The 'robust' method (default) uses the median as center and MAD as spread estimate, "
        "making it resistant to outliers. "
        "The 'classical' method uses the mean and standard deviation. "
        "The interval is computed using the t-distribution to account for small samples."
    ),
    input_model=ConfidenceIntervalInput,
    examples="""
    # "What is the 95% confidence interval for [10, 11, 12, 10, 9, 10]?"
        -> ``confidence_interval(values=[10, 11, 12, 10, 9, 10])``

    # "Give me a 99% classical confidence interval"
        -> ``confidence_interval(values=[...], confidence_level=0.99, method="classical")``
    """,
    category="univariate",
)
def confidence_interval_tool(spec: ConfidenceIntervalInput) -> dict:
    """Compute a confidence interval; see tool spec for details."""
    arr = np.asarray(spec.values, dtype=float)
    arr_clean = arr[~np.isnan(arr)]
    n = len(arr_clean)
    if spec.method == "robust":
        center = float(np.median(arr_clean))
        spread = float(median_absolute_deviation(arr_clean))
    else:
        center = float(np.mean(arr_clean))
        spread = float(np.std(arr_clean, ddof=1))
    ct = float(t_value(1 - (1 - spec.confidence_level) / 2, n - 1))
    margin = ct * spread / np.sqrt(n)
    return clean(
        {
            "lower": center - margin,
            "center": center,
            "upper": center + margin,
            "margin_of_error": margin,
            "confidence_level": spec.confidence_level,
            "method": spec.method,
            "n": n,
        }
    )


_register("confidence_interval")


# ---------------------------------------------------------------------------
# ttest_two_samples
# ---------------------------------------------------------------------------


class TtestTwoSamplesInput(BaseModel):
    """Input contract for ``ttest_two_samples``."""

    model_config = ConfigDict(extra="forbid")

    group_a: list[float] = Field(
        ...,
        min_length=2,
        description="Observations for group A (the reference / baseline group).",
    )
    group_b: list[float] = Field(
        ...,
        min_length=2,
        description="Observations for group B (the comparison group).",
    )
    confidence_level: float = Field(
        0.95,
        gt=0,
        lt=1,
        description="Confidence level for the interval (default 0.95).",
    )


@tool_spec(
    name="ttest_two_samples",
    description=(
        "Perform an unpaired (independent samples) two-sided t-test to determine whether the "
        "means of two groups differ significantly. "
        "Returns the t statistic, p-value, confidence interval for the difference "
        "(group_b_mean - group_a_mean), and degrees of freedom. "
        "The groups must be independent (different subjects/items). "
        "Use ttest_paired_samples instead when each observation in group A is matched to one in B."
    ),
    input_model=TtestTwoSamplesInput,
    examples="""
    # "Is the average yield different between process A [102,98,100] and process B [110,112,108]?"
        -> ``ttest_two_samples(group_a=[102,98,100], group_b=[110,112,108])``

    # "t-test at 99% confidence level"
        -> ``ttest_two_samples(group_a=[...], group_b=[...], confidence_level=0.99)``
    """,
    category="univariate",
)
def ttest_two_samples(spec: TtestTwoSamplesInput) -> dict:
    """Unpaired t-test for two independent samples; see tool spec for details."""
    a = pd.Series(np.asarray(spec.group_a, dtype=float)).dropna()
    b = pd.Series(np.asarray(spec.group_b, dtype=float)).dropna()
    raw = ttest_independent(a, b, conflevel=spec.confidence_level)

    # Remap keys to snake_case
    result = {
        "group_a_n": raw["Group A number"],
        "group_b_n": raw["Group B number"],
        "group_a_mean": raw["Group A average"],
        "group_b_mean": raw["Group B average"],
        "z_value": raw["z value"],
        "conf_int_lower": raw["ConfInt: Lo"],
        "conf_int_upper": raw["ConfInt: Hi"],
        "p_value": raw["p value"],
        "degrees_of_freedom": raw["Degrees of freedom"],
        "pooled_std": raw["Pooled standard deviation"],
        "confidence_level": spec.confidence_level,
    }
    significant = bool(result["p_value"] < (1 - spec.confidence_level))
    result["significant"] = significant
    diff = result["group_b_mean"] - result["group_a_mean"]
    result["interpretation"] = (
        f"The difference (B - A = {diff:.4g}) "
        + (
            f"IS statistically significant (p={result['p_value']:.4f} < {1 - spec.confidence_level:.2f})."
            if significant
            else f"is NOT statistically significant (p={result['p_value']:.4f} >= {1 - spec.confidence_level:.2f})."
        )
    )
    return clean(result)


_register("ttest_two_samples")


# ---------------------------------------------------------------------------
# ttest_paired_samples
# ---------------------------------------------------------------------------


class TtestPairedSamplesInput(BaseModel):
    """Input contract for ``ttest_paired_samples``."""

    model_config = ConfigDict(extra="forbid")

    group_a: list[float] = Field(
        ...,
        min_length=2,
        description="Before / reference measurements (one per subject).",
    )
    group_b: list[float] = Field(
        ...,
        min_length=2,
        description="After / comparison measurements (same order as group_a).",
    )
    confidence_level: float = Field(
        0.95,
        gt=0,
        lt=1,
        description="Confidence level for the interval (default 0.95).",
    )


@tool_spec(
    name="ttest_paired_samples",
    description=(
        "Perform a paired (repeated-measures) two-sided t-test to determine whether the "
        "mean of the differences between matched pairs is significantly different from zero. "
        "The difference is defined as before - after (group_a - group_b). "
        "Each position i in group_a must correspond to the same subject/unit as position i in "
        "group_b (e.g. before/after measurements on the same individual). "
        "Use ttest_two_samples instead for independent groups."
    ),
    input_model=TtestPairedSamplesInput,
    examples="""
    # "Did the new catalyst improve yields? Before [92, 95, 89] After [101, 99, 97]"
        -> ``ttest_paired_samples(group_a=[92,95,89], group_b=[101,99,97])``
    """,
    category="univariate",
)
def ttest_paired_samples(spec: TtestPairedSamplesInput) -> dict:
    """Paired t-test; see tool spec for details."""
    a = pd.Series(np.asarray(spec.group_a, dtype=float))
    b = pd.Series(np.asarray(spec.group_b, dtype=float))
    if len(a) != len(b):
        raise ValueError(
            f"group_a and group_b must have the same length for a paired test; "
            f"got len(group_a)={len(a)}, len(group_b)={len(b)}."
        )
    differences = a - b.values
    differences = differences.dropna()
    raw = ttest_paired(differences, conflevel=spec.confidence_level)
    result = {
        "n_pairs": len(differences),
        "differences_mean": raw["Differences mean"],
        "differences_std": raw["Standard deviation"],
        "z_value": raw["z value"],
        "conf_int_lower": raw["ConfInt: Lo"],
        "conf_int_upper": raw["ConfInt: Hi"],
        "p_value": raw["p value"],
        "degrees_of_freedom": raw["Degrees of freedom"],
        "confidence_level": spec.confidence_level,
    }
    significant = bool(result["p_value"] < (1 - spec.confidence_level))
    result["significant"] = significant
    result["interpretation"] = (
        f"The mean paired difference (A - B = {result['differences_mean']:.4g}) "
        + (
            f"IS statistically significant (p={result['p_value']:.4f} < {1 - spec.confidence_level:.2f})."
            if significant
            else f"is NOT statistically significant (p={result['p_value']:.4f} >= {1 - spec.confidence_level:.2f})."
        )
    )
    return clean(result)


_register("ttest_paired_samples")


# ---------------------------------------------------------------------------
# within_between_variance
# ---------------------------------------------------------------------------


class WithinBetweenVarianceInput(BaseModel):
    """Input contract for ``within_between_variance``.

    ``groups`` is intentionally typed as ``list[Any]`` so the LLM can
    pass strings, ints, or any hashable group label; pydantic only
    enforces the list shape.
    """

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        ...,
        min_length=3,
        description="Numeric measurement for each observation.",
    )
    groups: list[Any] = Field(
        ...,
        min_length=3,
        description=(
            "Group label for each observation (same length as values). "
            "Can be strings, integers, or any hashable type."
        ),
    )


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
    input_model=WithinBetweenVarianceInput,
    examples="""
    # "Measurements on day 1: [101,102] and day 2: [94,95]. How much variation is within vs between days?"
        -> ``within_between_variance(values=[101,102,94,95], groups=[1,1,2,2])``

    # "Operator study: Alice measured [10.1,10.2,10.0], Bob measured [10.5,10.4,10.6]"
        -> ``within_between_variance(values=[10.1,10.2,10.0,10.5,10.4,10.6],
                groups=["Alice","Alice","Alice","Bob","Bob","Bob"])``
    """,
    category="univariate",
)
def within_between_variance(spec: WithinBetweenVarianceInput) -> dict:
    """Within- and between-group variance decomposition; see tool spec for details."""
    if len(spec.values) != len(spec.groups):
        raise ValueError(
            f"'values' and 'groups' must have the same length; "
            f"got len(values)={len(spec.values)}, len(groups)={len(spec.groups)}."
        )
    df = pd.DataFrame({"measured": spec.values, "repeat": spec.groups})
    result = variance_decomposition(df, measured="measured", repeat="repeat")
    return clean(result)


_register("within_between_variance")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_univariate_tool_specs() -> list[dict]:
    """Return tool specs for all univariate tools registered in this module."""
    return get_tool_specs(names=_UNIVARIATE_TOOL_NAMES)
