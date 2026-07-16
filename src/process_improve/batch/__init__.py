"""Batch process data analysis: alignment, feature extraction, modelling, and visualization."""

from process_improve.batch._batch_monitor import BatchMonitor
from process_improve.batch._batch_pca import BatchPCA
from process_improve.batch._batch_plots import (
    contribution_at_time_plot,
    online_monitoring_plot,
    time_varying_loading_plot,
)
from process_improve.batch._batch_pls import BatchPLS
from process_improve.batch._transformers import (
    BatchFeatureExtractor,
    BatchScaler,
    DTWAligner,
    ResampleAligner,
)
from process_improve.batch.data_input import (
    check_valid_batch_dict,
    dict_to_melted,
    dict_to_wide,
    melted_to_dict,
    melted_to_wide,
    wide_to_dict,
    wide_to_melted,
)
from process_improve.batch.datasets import (
    load_batch_fake_data,
    load_dryer,
    load_nylon,
)
from process_improve.batch.features import (
    cross,
    f_agemax,
    f_agemin,
    f_area,
    f_count,
    f_crossing,
    f_elbow,
    f_iqr,
    f_last,
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
    # Modelling and monitoring
    "BatchFeatureExtractor",
    "BatchMonitor",
    "BatchPCA",
    "BatchPLS",
    "BatchScaler",
    "DTWAligner",
    "ResampleAligner",
    # Preprocessing/alignment
    "batch_dtw",
    # Data input/conversion
    "check_valid_batch_dict",
    "contribution_at_time_plot",
    "cross",
    "determine_scaling",
    "dict_to_melted",
    "dict_to_wide",
    "f_agemax",
    "f_agemin",
    "f_area",
    "f_count",
    "f_crossing",
    "f_elbow",
    "f_iqr",
    "f_last",
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
    # Dataset loaders
    "load_batch_fake_data",
    "load_dryer",
    "load_nylon",
    "melted_to_dict",
    "melted_to_wide",
    "online_monitoring_plot",
    "resample_to_reference",
    "time_varying_loading_plot",
    "wide_to_dict",
    "wide_to_melted",
]
