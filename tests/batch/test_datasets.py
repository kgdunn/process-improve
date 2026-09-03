"""Tests for the bundled batch dataset loaders."""

import pandas as pd

from process_improve.batch.data_input import check_valid_batch_dict
from process_improve.batch.datasets import load_batch_fake_data, load_dryer, load_nylon


def test_load_nylon() -> None:
    """Nylon loads as a valid batch dict: 57 batches of 10 numeric tags, no NaN."""
    batches = load_nylon()
    assert len(batches) == 57
    assert check_valid_batch_dict(batches, no_nan=True)
    first = next(iter(batches.values()))
    assert list(first.columns) == [f"Tag{i:02d}" for i in range(1, 11)]
    assert "batch_id" not in first.columns


def test_load_dryer() -> None:
    """Dryer loads as a valid batch dict with the documented tag names."""
    batches = load_dryer()
    assert len(batches) == 71
    assert check_valid_batch_dict(batches)
    first = next(iter(batches.values()))
    assert "DryerTemp" in first.columns
    assert "ClockTime" in first.columns
    assert "batch_id" not in first.columns


def test_load_batch_fake_data() -> None:
    """Synthetic data loads with only numeric columns (DateTime dropped)."""
    batches = load_batch_fake_data()
    assert len(batches) >= 2
    assert check_valid_batch_dict(batches)
    first = next(iter(batches.values()))
    assert "DateTime" not in first.columns
    assert "Batch" not in first.columns


def test_loaders_reset_row_index() -> None:
    """Each per-batch frame counts its own samples from zero."""
    for batches in (load_nylon(), load_dryer()):
        for batch in batches.values():
            assert isinstance(batch.index, pd.RangeIndex)
            assert batch.index[0] == 0
