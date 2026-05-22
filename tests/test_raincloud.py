"""Tests for the raincloud plot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.visualization.raincloud import raincloud


def test_raincloud_grouped_dataframe() -> None:
    """One violin trace is drawn per group, each combining cloud, box and rain."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "value": rng.normal(size=60),
        "reactor": ["A"] * 30 + ["B"] * 30,
    })
    fig = raincloud(df, value="value", group="reactor", title="Yield by reactor")

    assert len(fig.data) == 2
    assert all(trace.type == "violin" for trace in fig.data)
    assert {trace.name for trace in fig.data} == {"A", "B"}
    for trace in fig.data:
        assert trace.box.visible is True
        assert trace.points == "all"
        assert trace.side == "positive"
    assert fig.layout.title.text == "Yield by reactor"


def test_raincloud_series_is_single_ungrouped_cloud() -> None:
    """A bare Series produces a single raincloud."""
    fig = raincloud(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert len(fig.data) == 1
    assert fig.data[0].type == "violin"


def test_raincloud_vertical_orientation() -> None:
    """Vertical orientation puts the data on the y-axis."""
    fig = raincloud(pd.Series([1.0, 2.0, 3.0]), orientation="v")
    assert fig.data[0].y is not None
    assert fig.data[0].x is None


def test_raincloud_requires_value_for_dataframe() -> None:
    """A DataFrame input without `value` is rejected."""
    with pytest.raises(ValueError, match="value"):
        raincloud(pd.DataFrame({"a": [1.0, 2.0]}))


def test_raincloud_rejects_bad_orientation() -> None:
    """An unknown orientation is rejected."""
    with pytest.raises(ValueError, match="orientation"):
        raincloud(pd.Series([1.0, 2.0]), orientation="x")
