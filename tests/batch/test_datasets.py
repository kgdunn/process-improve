"""Tests for the bundled and hosted batch dataset loaders."""

from __future__ import annotations

import pathlib

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
from tests._case_study_scripts import load_or_skip


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


@pytest.mark.dataset
@pytest.mark.slow
def test_load_dupont() -> None:
    """DuPont loads as 55 aligned, complete batches of 100 samples and 10 tags."""
    batches = load_or_skip(load_dupont)
    assert sorted(batches) == list(range(1, 56))
    assert check_valid_batch_dict(batches, no_nan=True)
    first = batches[1]
    assert first.shape == (100, 10)
    assert list(first.columns) == [
        "TempR-1", "TempR-2", "TempR-3", "Press-1", "Flow-1", "TempH-1", "TempC-1", "Press-2", "Press-3", "Flow-2",
    ]  # fmt: skip
    assert isinstance(first.index, pd.RangeIndex)


@pytest.mark.dataset
@pytest.mark.slow
def test_load_fmc() -> None:
    """FMC loads as four blocks over the same 59 batches, with its missing cells kept."""
    fmc = load_or_skip(load_fmc)
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
@pytest.mark.slow
def test_load_sbr() -> None:
    """SBR loads as 53 complete batches of 200 samples and 9 tags, plus 5 quality attributes."""
    sbr = load_or_skip(load_sbr)
    assert sorted(sbr.X) == list(range(1, 54))
    assert check_valid_batch_dict(sbr.X, no_nan=True)
    assert sbr.X[1].shape == (200, 9)
    assert set(sbr.trajectory_tags) <= set(sbr.X[1].columns)
    assert sbr.Y.shape == (53, 5)
    assert list(sbr.Y.columns) == ["Composition", "ParticleSize", "Branching", "CrossLinking", "Polydispersity"]
    assert list(sbr.Y.index) == list(sbr.X)
    assert sbr.fault_batches == [34, 37]


# ---------------------------------------------------------------------------
# The same loaders on local copies (file:// URLs), so their logic is tested offline
# ---------------------------------------------------------------------------


def _write_workbook(path: pathlib.Path, sheets: dict[str, pd.DataFrame]) -> str:
    pytest.importorskip("openpyxl")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return path.as_uri()


def test_load_dupont_from_a_local_copy(tmp_path: pathlib.Path) -> None:
    """A file:// URL reads a local copy; the time column is dropped and batches are keyed by id."""
    melted = pd.DataFrame(
        {
            "batch_id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "TempR-1": [0.5, 0.6, 0.7, 0.8],
            "Flow-2": [1.0, 1.1, 1.2, 1.3],
        }
    )
    path = tmp_path / "polymerization.csv"
    melted.to_csv(path, index=False)
    batches = load_dupont(url=path.as_uri())
    assert list(batches) == [1, 2]
    assert list(batches[1].columns) == ["TempR-1", "Flow-2"]
    assert batches[2].shape == (2, 2)
    assert batches[2].index.tolist() == [0, 1]


def test_load_fmc_from_a_local_copy(tmp_path: pathlib.Path) -> None:
    """The four sheets come back as a batch dictionary plus per-batch tables aligned to it."""
    url = _write_workbook(
        tmp_path / "batch-dryer.xlsx",
        {
            "Z_operations": pd.DataFrame({"batch_id": [7, 5], "Level1": [1.0, 2.0]}),
            "Z_chemistry": pd.DataFrame({"batch_id": [5, 7], "Z1": [0.1, None]}),
            "X_batch": pd.DataFrame(
                {"batch_id": [5, 5, 7, 7], "CTankLvl": [0.0, 1.0, 0.0, 2.0], "ClockTime": [1, 2, 1, 2]}
            ),
            "Y_quality": pd.DataFrame({"batch_id": [5, 7], "Y1": [10.0, 11.0]}),
        },
    )
    fmc = load_fmc(url=url)
    assert list(fmc.X) == [5, 7]
    assert fmc.batch_ids == [5, 7]
    assert list(fmc.X[5].columns) == ["CTankLvl", "ClockTime"]
    assert list(fmc.Zop.index) == [5, 7]  # reindexed into trajectory order
    assert fmc.Zop.loc[5, "Level1"] == 2.0
    assert int(fmc.Zchem.isna().sum().sum()) == 1
    assert list(fmc.Y.columns) == ["Y1"]
    assert len(fmc.missing_chemistry) == 13


def test_load_sbr_from_a_local_copy(tmp_path: pathlib.Path) -> None:
    """The two sheets come back as a batch dictionary (time dropped) and a quality table."""
    tags = ["StyreneFlow", "ButadieneFlow", "FeedTemp", "ReactorTemp", "CoolingTemp", "JacketTemp"]
    tags += ["LatexDensity", "Conversion", "EnergyReleased"]
    x = pd.DataFrame({"batch_id": [1, 1, 2, 2], "time": [1, 2, 1, 2]})
    for j, tag in enumerate(tags):
        x[tag] = [j, j + 0.5, j + 1.0, j + 1.5]
    y = pd.DataFrame({"batch_id": [1, 2]})
    for name in ("Composition", "ParticleSize", "Branching", "CrossLinking", "Polydispersity"):
        y[name] = [1.0, 2.0]
    sbr = load_sbr(url=_write_workbook(tmp_path / "sbr-batch-reactor.xlsx", {"X_batch": x, "Y_quality": y}))
    assert list(sbr.X) == [1, 2]
    assert list(sbr.X[1].columns) == tags
    assert "time" not in sbr.X[1].columns
    assert sbr.Y.shape == (2, 5)
    assert list(sbr.Y.index) == [1, 2]
    assert set(sbr.trajectory_tags) <= set(tags)
    assert sbr.fault_batches == [34, 37]
