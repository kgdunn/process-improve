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
    ExtractBatchFeaturesInput,
    get_batch_tool_specs,
)
from process_improve.batch.tools import extract_batch_features as _extract_batch_features


def extract_batch_features(**kwargs):
    """Test-local convenience wrapper: build the pydantic input from kwargs.

    Lets the existing kwarg-style test call sites work unchanged after the
    ENG-04 / ENG-10 migration to ``input_model=`` on @tool_spec.
    """
    return _extract_batch_features(ExtractBatchFeaturesInput(**kwargs))


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


@pytest.mark.parametrize("feature_name", sorted(_FEATURE_MAP))
def test_extract_batch_features_each_location_feature(
    two_batch_timeseries: list[dict], feature_name: str
) -> None:
    """Every location-based feature in the documented mapping is dispatchable."""
    result = extract_batch_features(
        data=two_batch_timeseries,
        value_columns=["temp"],
        features=[feature_name],
    )

    assert "error" not in result
    assert result["features_extracted"] == [feature_name]
    assert result["n_batches"] == 2


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
    """An unknown feature name is rejected by the pydantic Literal contract.

    Pydantic validates the ``features`` list against the documented set of
    keys before the function body ever runs; the resulting ValidationError
    (a ValueError subclass) mentions the bad value and the allowed
    alternatives.
    """
    with pytest.raises(ValueError, match="bogus"):
        extract_batch_features(
            data=two_batch_timeseries,
            value_columns=["temp"],
            features=["bogus"],
        )


def test_extract_batch_features_bad_data_returns_error() -> None:
    """Garbage inputs are rejected at the pydantic boundary or in-band as an error dict.

    The original single-row payload now fails the ``min_length=2`` constraint
    on ``data``; with two rows but a missing column, the underlying call
    raises a KeyError that surfaces as ``{"error": ...}``.
    """
    result = extract_batch_features(
        data=[{"batch": "B1"}, {"batch": "B2"}],
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
