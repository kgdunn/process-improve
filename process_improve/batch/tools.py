"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for batch process data analysis.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from process_improve.tool_spec import clean, get_tool_specs, tool_spec

_BATCH_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _BATCH_TOOL_NAMES.append(name)


# ---------------------------------------------------------------------------
# Feature name mapping
# ---------------------------------------------------------------------------

_FEATURE_MAP = {
    "mean": "f_mean",
    "median": "f_median",
    "std": "f_std",
    "iqr": "f_iqr",
    "mad": "f_mad",
    "robust_mad": "f_robust_mad",
    "sum": "f_sum",
    "min": "f_min",
    "max": "f_max",
    "last": "f_last",
    "count": "f_count",
}

_TIME_FEATURE_MAP = {
    "area": "f_area",
    "slope": "f_slope",
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool_spec(
    name="extract_batch_features",
    description=(
        "Extract summary features from batch process data. Given a table of time-series "
        "measurements from multiple batches, compute features like mean, std, slope, area, "
        "etc. for each batch and each measurement tag. "
        "The result is a feature matrix with one row per batch, suitable for multivariate "
        "analysis (e.g. PCA on batch data). "
        "Available features: 'mean', 'median', 'std', 'iqr', 'mad', 'robust_mad', 'sum', "
        "'min', 'max', 'last', 'count'. "
        "Time-dependent features (require time_column): 'area', 'slope'."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Batch data as a list of row-dicts. Each dict must contain the batch_column "
                        "and one or more value columns. Example: "
                        '[{"batch": "B1", "time": 0, "temp": 100}, {"batch": "B1", "time": 1, "temp": 105}, ...]'
                    ),
                    "minItems": 2,
                },
                "features": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "mean", "median", "std", "iqr", "mad", "robust_mad",
                            "sum", "min", "max", "last", "count", "area", "slope",
                        ],
                    },
                    "description": "List of features to extract (default: ['mean', 'std']).",
                },
                "value_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column names of the measurement tags to extract features for.",
                },
                "batch_column": {
                    "type": "string",
                    "description": "Column name identifying each batch (default: 'batch').",
                },
                "time_column": {
                    "type": "string",
                    "description": (
                        "Column name for the time/age axis. Required for 'area' and 'slope' features."
                    ),
                },
            },
            "required": ["data", "value_columns"],
        }
    },
    examples="""
    # "Extract mean and std for temperature and pressure from my batch data"
        -> ``extract_batch_features(
                data=[...],
                features=["mean", "std"],
                value_columns=["temperature", "pressure"],
                batch_column="batch_id"
            )``

    # "Get slope features for my reactor data"
        -> ``extract_batch_features(
                data=[...],
                features=["slope"],
                value_columns=["concentration"],
                batch_column="batch",
                time_column="time_minutes"
            )``
    """,
    category="batch",
)
def extract_batch_features(
    *,
    data: list[dict[str, Any]],
    value_columns: list[str],
    features: list[str] | None = None,
    batch_column: str = "batch",
    time_column: str | None = None,
) -> dict[str, Any]:
    """Extract summary features from batch process data."""
    from process_improve.batch import features as feat_mod  # noqa: PLC0415

    if features is None:
        features = ["mean", "std"]

    try:
        df = pd.DataFrame(data)

        results: list[pd.DataFrame] = []
        for feature_name in features:
            if feature_name in _FEATURE_MAP:
                func = getattr(feat_mod, _FEATURE_MAP[feature_name])
                result_df = func(df, tags=value_columns, batch_col=batch_column)
                results.append(result_df)
            elif feature_name in _TIME_FEATURE_MAP:
                if time_column is None:
                    return {"error": f"Feature '{feature_name}' requires time_column to be specified."}
                func = getattr(feat_mod, _TIME_FEATURE_MAP[feature_name])
                result_df = func(df, time_tag=time_column, tags=value_columns, batch_col=batch_column)
                results.append(result_df)
            else:
                available = sorted(list(_FEATURE_MAP) + list(_TIME_FEATURE_MAP))
                return {"error": f"Unknown feature: '{feature_name}'. Available: {available}"}

        if results:
            combined = pd.concat(results, axis=1)
            feature_matrix = combined.reset_index().to_dict(orient="records")
        else:
            feature_matrix = []

        return clean({
            "feature_matrix": feature_matrix,
            "n_batches": len(combined) if results else 0,
            "n_features": combined.shape[1] if results else 0,
            "features_extracted": features,
        })
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


_register("extract_batch_features")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_batch_tool_specs() -> list[dict]:
    """Return tool specs for all batch tools registered in this module."""
    return get_tool_specs(names=_BATCH_TOOL_NAMES)
