"""Smoke tests for the experiments dataset stubs.

The functions in ``process_improve.experiments.datasets`` are currently
docstring-only stubs (they return ``None``). They still count toward
coverage when called, so we exercise each one to keep the file from
sitting at 0% coverage.
"""

from __future__ import annotations

import pandas as pd

from process_improve.experiments import datasets


def test_distillateflow_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.distillateflow() is None


def test_pollutant_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.pollutant() is None


def test_oildoe_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.oildoe() is None


def test_golf_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.golf() is None


def test_boilingpot_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.boilingpot() is None


def test_solar_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.solar() is None


def test_data_dispatch_signature_typed() -> None:
    """``datasets.data`` is the planned dispatcher; verify its signature
    is preserved so we notice if it is later refactored away.
    """
    # The annotation is stored as a string under PEP 563 (`from __future__ import annotations`).
    assert datasets.data.__annotations__["return"] == "pd.DataFrame"
