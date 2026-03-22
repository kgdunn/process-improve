"""Batch process data analysis: alignment, feature extraction, and visualization."""

from process_improve.batch.data_input import (
    check_valid_batch_dict,
    dict_to_melted,
    dict_to_wide,
    melted_to_dict,
    melted_to_wide,
    wide_to_melted,
)
from process_improve.batch.features import (
    cross,
    f_area,
    f_count,
    f_crossing,
    f_elbow,
    f_iqr,
    f_last,
    f_mad,
    f_max,
    f_mean,
    f_median,
    f_min,
    f_robust_mad,
    f_rupture,
    f_slope,
    f_std,
    f_sum,
)
from process_improve.batch.preprocessing import (
    batch_dtw,
    determine_scaling,
    find_reference_batch,
    resample_to_reference,
)

__all__ = [
    # Preprocessing/alignment
    "batch_dtw",
    # Data input/conversion
    "check_valid_batch_dict",
    "cross",
    "determine_scaling",
    "dict_to_melted",
    "dict_to_wide",
    "f_area",
    "f_count",
    "f_crossing",
    "f_elbow",
    "f_iqr",
    "f_last",
    "f_mad",
    "f_max",
    # Feature extraction
    "f_mean",
    "f_median",
    "f_min",
    "f_robust_mad",
    "f_rupture",
    "f_slope",
    "f_std",
    "f_sum",
    "find_reference_batch",
    "melted_to_dict",
    "melted_to_wide",
    "resample_to_reference",
    "wide_to_melted",
]
