"""(c) Kevin Dunn, 2010-2026. MIT License."""

from process_improve.univariate.metrics import (
    Sn,
    confidence_interval,
    detect_outliers_esd,
    median_absolute_deviation,
    summary_stats,
    t_value,
    t_value_cdf,
    test_normality,
    ttest_independent,
    ttest_independent_from_df,
    ttest_paired,
    ttest_paired_from_df,
    variance_decomposition,
)
from process_improve.univariate.tools import (
    confidence_interval_tool,
    detect_outliers,
    get_univariate_tool_specs,
    median_absolute_deviation_tool,
    normality_test,
    robust_scale_sn,
    robust_summary_stats,
    ttest_paired_samples,
    ttest_two_samples,
    within_between_variance,
)

__all__ = [
    # Core metrics
    "Sn",
    "confidence_interval",
    # Tool wrappers
    "confidence_interval_tool",
    "detect_outliers",
    "detect_outliers_esd",
    "get_univariate_tool_specs",
    "median_absolute_deviation",
    "median_absolute_deviation_tool",
    "normality_test",
    "robust_scale_sn",
    "robust_summary_stats",
    "summary_stats",
    "t_value",
    "t_value_cdf",
    "test_normality",
    "ttest_independent",
    "ttest_independent_from_df",
    "ttest_paired",
    "ttest_paired_from_df",
    "ttest_paired_samples",
    "ttest_two_samples",
    "variance_decomposition",
    "within_between_variance",
]
