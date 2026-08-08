"""Tests for the experiments dataset loaders.

``boilingpot()``, ``golf()``, ``pollutant()`` and ``solar()`` load from
CSVs bundled with the package; ``oildoe()`` and ``distillateflow()``
fetch from openmv.net. The bundled data were extracted from the ``.rda``
files in the companion R package, so the shapes and values below are the
R package's values.
"""

from __future__ import annotations

import urllib.error
from collections.abc import Callable

import pandas as pd
import pytest

from process_improve.experiments import datasets


def _load_or_skip(loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Call the network-backed loader, ``pytest.skip`` on any network error."""
    try:
        return loader()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        pytest.skip(f"could not fetch from openmv.net: {exc}")


def test_boilingpot_loads() -> None:
    """``boilingpot()`` returns the documented 11x4 factorial frame."""
    df = datasets.boilingpot()
    assert df.shape == (11, 4)
    assert set(df.columns) == {"A", "B", "C", "y"}


@pytest.mark.dataset
def test_oildoe_loads() -> None:
    """``oildoe()`` fetches the openmv.net file (skipped if offline)."""
    df = _load_or_skip(datasets.oildoe)
    assert df.shape == (19, 5)
    assert set(df.columns) == {"A", "B", "C", "D", "y"}


@pytest.mark.dataset
def test_distillateflow_loads() -> None:
    """``distillateflow()`` fetches the openmv.net file (skipped if offline)."""
    df = _load_or_skip(datasets.distillateflow)
    assert df.shape == (44640, 1)
    assert "Flow" in df.columns


def test_pollutant_loads() -> None:
    """``pollutant()`` returns the 8-run BHH2 water-treatment factorial."""
    df = datasets.pollutant()
    assert df.shape == (8, 4)
    assert list(df.columns) == ["C", "T", "S", "y"]
    # A 2^3 factorial in coded units.
    for factor in ("C", "T", "S"):
        assert set(df[factor]) == {-1, 1}
    assert df["y"].tolist() == [5, 30, 6, 33, 4, 3, 5, 4]


def test_golf_loads() -> None:
    """``golf()`` returns the 16-run full factorial, factors as labels."""
    df = datasets.golf()
    assert df.shape == (16, 5)
    assert list(df.columns) == ["H", "N", "C", "T", "y"]
    assert set(df["C"]) == {"Callaway", "Titleist"}
    assert set(df["T"]) == {"9:00", "14:00"}
    assert set(df["H"]) == {1, 3}
    assert set(df["N"]) == {1, 9}


def test_solar_loads() -> None:
    """``solar()`` returns the 2^4 factorial with its two responses."""
    df = datasets.solar()
    assert df.shape == (16, 6)
    assert list(df.columns) == ["A", "B", "C", "D", "y1", "y2"]
    for factor in ("A", "B", "C", "D"):
        assert set(df[factor]) == {-1, 1}
    assert df["y1"].iloc[0] == pytest.approx(43.5)
    assert df["y2"].iloc[-1] == pytest.approx(100.0)


@pytest.mark.parametrize("name", ["boilingpot", "golf", "pollutant", "solar"])
def test_data_dispatch_matches_direct_loader(name: str) -> None:
    """``data(name)`` returns exactly what the named loader returns."""
    pd.testing.assert_frame_equal(datasets.data(name), getattr(datasets, name)())


@pytest.mark.parametrize("alias", ["oil.doe", "oilDOE"])
def test_data_dispatch_accepts_the_oildoe_aliases(alias: str) -> None:
    """The R package documents two aliases for ``oildoe``; both resolve."""
    assert datasets._LOADERS[alias] is datasets.oildoe


def test_data_dispatch_rejects_an_unknown_name() -> None:
    """An unknown name raises, and the message lists what is available."""
    with pytest.raises(ValueError, match="Unknown dataset 'nope'"):
        datasets.data("nope")
