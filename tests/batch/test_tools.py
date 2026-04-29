"""Tests for the agent-callable wrappers in `process_improve.batch.tools`.

The wrappers in `batch/tools.py` were untested before this file existed. They
flatten a list of row-dicts into a DataFrame, call into `batch.features`, and
return a JSON-friendly result dict (or `{"error": ...}` on failure).
"""

from __future__ import annotations

import pytest

from process_improve.batch.tools import (
    _FEATURE_MAP,
    _TIME_FEATURE_MAP,
    extract_batch_features,
    get_batch_tool_specs,
)


@pytest.fixture
def two_batch_timeseries() -> list[dict]:
    """Two batches, each with a few timepoints and two measurement tags."""
    return [
        {"batch": "B1", "time": 0.0, "temp": 100.0, "press": 1.0},
        {"batch": "B1", "time": 1.0, "temp": 102.0, "press": 1.1},
        {"batch": "B1", "time": 2.0, "temp": 104.0, "press": 1.2},
        {"batch": "B1", "time": 3.0, "temp": 106.0, "press": 1.3},
        {"batch": "B2", "time": 0.0, "temp": 200.0, "press": 2.0},
        {"batch": "B2", "time": 1.0, "temp": 198.0, "press": 2.1},
        {"batch": "B2", "time": 2.0, "temp": 196.0, "press": 2.2},
        {"batch": "B2", "time": 3.0, "temp": 194.0, "press": 2.3},
    ]


def test_extract_batch_features_default_features(two_batch_timeseries: list[dict]) -> None:
    """Defaults extract mean and std; result has one row per batch and 4 feature columns."""
    result = extract_batch_features(
        data=two_batch_timeseries,
        value_columns=["temp", "press"],
    )

    assert "error" not in result
    assert result["features_extracted"] == ["mean", "std"]
    assert result["n_batches"] == 2
    # 2 features x 2 tags = 4 feature columns.
    assert result["n_features"] == 4
    assert len(result["feature_matrix"]) == 2

    # Check the mean row for B1: temp mean is 103.0, press mean is 1.15
    by_batch = {row["batch"]: row for row in result["feature_matrix"]}
    assert by_batch["B1"]["temp_mean"] == pytest.approx(103.0)
    assert by_batch["B1"]["press_mean"] == pytest.approx(1.15)


# f_mad and f_robust_mad have pre-existing implementation issues in
# `process_improve.batch.features` (mad: removed pandas API; robust_mad:
# stub raises). The wrapper still surfaces them as `{"error": ...}` rather
# than raising — that branch is covered by the unknown-feature and bad-data
# tests below — so they're excluded here to keep this test focused on the
# working dispatch path.
_WORKING_LOCATION_FEATURES = sorted(set(_FEATURE_MAP) - {"mad", "robust_mad"})


@pytest.mark.parametrize("feature_name", _WORKING_LOCATION_FEATURES)
def test_extract_batch_features_each_location_feature(
    two_batch_timeseries: list[dict], feature_name: str
) -> None:
    """Every (working) location-based feature in the documented mapping is dispatchable."""
    result = extract_batch_features(
        data=two_batch_timeseries,
        value_columns=["temp"],
        features=[feature_name],
    )

    assert "error" not in result
    assert result["features_extracted"] == [feature_name]
    assert result["n_batches"] == 2


@pytest.mark.parametrize("broken_feature", ["mad", "robust_mad"])
def test_extract_batch_features_known_broken_features_return_error(
    two_batch_timeseries: list[dict], broken_feature: str
) -> None:
    """Pre-existing wrapper limitation: 'mad' and 'robust_mad' surface as errors.

    Documents the current behavior so a future fix in `batch/features` is noticed
    here and these tests can be flipped to assert success.
    """
    result = extract_batch_features(
        data=two_batch_timeseries,
        value_columns=["temp"],
        features=[broken_feature],
    )
    assert "error" in result


def test_extract_batch_features_area_with_time_column(two_batch_timeseries: list[dict]) -> None:
    """Time-dependent feature 'area' computes correctly when a time_column is given."""
    result = extract_batch_features(
        data=two_batch_timeseries,
        value_columns=["temp"],
        features=["area"],
        time_column="time",
    )

    assert "error" not in result
    assert result["features_extracted"] == ["area"]
    assert result["n_batches"] == 2
    by_batch = {row["batch"]: row for row in result["feature_matrix"]}
    # Trapezoidal area for B1 temp: increases linearly 100 -> 106 over t=0..3, area = 309.0
    assert by_batch["B1"]["temp_area"] == pytest.approx(309.0)


def test_extract_batch_features_time_feature_without_time_column_returns_error(
    two_batch_timeseries: list[dict],
) -> None:
    """Time-dependent features without time_column yield an explicit error message."""
    result = extract_batch_features(
        data=two_batch_timeseries,
        value_columns=["temp"],
        features=["area"],
        # time_column intentionally omitted
    )

    assert "error" in result
    assert "time_column" in result["error"]


def test_extract_batch_features_unknown_feature_returns_error(
    two_batch_timeseries: list[dict],
) -> None:
    """An unknown feature name surfaces a helpful error listing the valid options."""
    result = extract_batch_features(
        data=two_batch_timeseries,
        value_columns=["temp"],
        features=["bogus"],
    )

    assert "error" in result
    assert "bogus" in result["error"]
    # The error should mention at least one of the valid feature names.
    assert "mean" in result["error"]


def test_extract_batch_features_bad_data_returns_error() -> None:
    """Garbage inputs surface as an error dict, not an exception."""
    result = extract_batch_features(
        data=[{"batch": "B1"}],  # no value column data at all
        value_columns=["nonexistent"],
    )

    assert "error" in result


def test_get_batch_tool_specs_lists_extract_batch_features() -> None:
    """The module-level convenience returns the registered tool spec."""
    specs = get_batch_tool_specs()
    names = {spec.get("name") for spec in specs}
    assert "extract_batch_features" in names


def test_feature_maps_documented_in_schema_match_implementations() -> None:
    """Sanity: every key in the maps must match a real `f_*` function (cheap import test)."""
    from process_improve.batch import features as feat_mod

    for short_name, long_name in {**_FEATURE_MAP, **_TIME_FEATURE_MAP}.items():
        assert hasattr(feat_mod, long_name), (
            f"Mapped feature '{short_name}' -> '{long_name}' but no such function exists."
        )
