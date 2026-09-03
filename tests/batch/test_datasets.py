"""Tests for the bundled and hosted batch dataset loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import pandas as pd
import pytest

from process_improve.batch.data_input import check_valid_batch_dict
from process_improve.batch.datasets import (
    load_batch_fake_data,
    load_dryer,
    load_dupont,
    load_fmc,
    load_nylon,
    load_sbr,
)

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


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


# ---------------------------------------------------------------------------
# Hosted case-study datasets (downloaded from openmv.net; skipped offline)
# ---------------------------------------------------------------------------


def _load_or_skip(loader: Callable[[], T]) -> T:
    """Call a remote loader, skipping the test when the download fails."""
    try:
        data = loader()
    except RuntimeError as exc:
        pytest.skip(f"Cannot download the dataset: {exc}")
    return data


@pytest.mark.dataset
def test_load_dupont() -> None:
    """DuPont loads as 55 aligned, complete batches of 100 samples and 10 tags."""
    batches = _load_or_skip(load_dupont)
    assert sorted(batches) == list(range(1, 56))
    assert check_valid_batch_dict(batches, no_nan=True)
    first = batches[1]
    assert first.shape == (100, 10)
    assert list(first.columns) == [
        "TempR-1", "TempR-2", "TempR-3", "Press-1", "Flow-1", "TempH-1", "TempC-1", "Press-2", "Press-3", "Flow-2",
    ]  # fmt: skip
    assert isinstance(first.index, pd.RangeIndex)


@pytest.mark.dataset
def test_load_fmc() -> None:
    """FMC loads as four blocks over the same 59 batches, with its missing cells kept."""
    fmc = _load_or_skip(load_fmc)
    assert set(fmc) >= {"X", "Y", "Zop", "Zchem", "batch_ids", "missing_chemistry"}
    assert len(fmc.X) == 59
    assert fmc.batch_ids[0] == 2
    assert fmc.batch_ids[-1] == 71
    assert check_valid_batch_dict(fmc.X)
    first = fmc.X[fmc.batch_ids[0]]
    assert first.shape == (325, 11)
    assert list(first.columns)[-1] == "ClockTime"
    for block in (fmc.Y, fmc.Zop, fmc.Zchem):
        assert list(block.index) == list(fmc.X)
    assert (fmc.Y.shape, fmc.Zop.shape, fmc.Zchem.shape) == ((59, 8), (59, 9), (59, 11))
    nan_counts = (
        int(pd.concat(fmc.X.values()).isna().sum().sum()),
        int(fmc.Zop.isna().sum().sum()),
        int(fmc.Zchem.isna().sum().sum()),
        int(fmc.Y.isna().sum().sum()),
    )
    assert nan_counts == (1410, 0, 134, 21)
    assert len(fmc.missing_chemistry) == 13
    assert set(fmc.missing_chemistry) <= set(fmc.batch_ids)


@pytest.mark.dataset
def test_load_sbr() -> None:
    """SBR loads as 53 complete batches of 200 samples and 9 tags, plus 5 quality attributes."""
    sbr = _load_or_skip(load_sbr)
    assert sorted(sbr.X) == list(range(1, 54))
    assert check_valid_batch_dict(sbr.X, no_nan=True)
    assert sbr.X[1].shape == (200, 9)
    assert set(sbr.trajectory_tags) <= set(sbr.X[1].columns)
    assert sbr.Y.shape == (53, 5)
    assert list(sbr.Y.columns) == ["Composition", "ParticleSize", "Branching", "CrossLinking", "Polydispersity"]
    assert list(sbr.Y.index) == list(sbr.X)
    assert sbr.fault_batches == [34, 37]
