"""Tests for the experiments dataset loaders.

``boilingpot()`` loads from a CSV bundled with the package; ``oildoe()``
and ``distillateflow()`` fetch from openmv.net. The remaining functions
in ``process_improve.experiments.datasets`` are still docstring-only
stubs (they return ``None``); we exercise them so they keep counting
toward coverage.
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


def test_oildoe_loads() -> None:
    """``oildoe()`` fetches the openmv.net file (skipped if offline)."""
    df = _load_or_skip(datasets.oildoe)
    assert df.shape == (19, 5)
    assert set(df.columns) == {"A", "B", "C", "D", "y"}


def test_distillateflow_loads() -> None:
    """``distillateflow()`` fetches the openmv.net file (skipped if offline)."""
    df = _load_or_skip(datasets.distillateflow)
    assert df.shape == (44640, 1)
    assert "Flow" in df.columns


def test_pollutant_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.pollutant() is None


def test_golf_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.golf() is None


def test_solar_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.solar() is None


def test_data_dispatch_signature_typed() -> None:
    """``datasets.data`` is the planned dispatcher; verify its signature
    is preserved so we notice if it is later refactored away.
    """
    # The annotation is stored as a string under PEP 563
    # (``from __future__ import annotations``).
    assert datasets.data.__annotations__["return"] == "pd.DataFrame"
    assert datasets.data.__annotations__["dataset"] == "str"
