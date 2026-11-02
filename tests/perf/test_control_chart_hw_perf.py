"""Deterministic cost-shape assertions for the Holt-Winters ``ControlChart`` fit.

The Holt-Winters recursion used to read ``chart.df`` about ten times per row
(``df["alpha_hat"][i - 1]`` and friends) and write every row back with ``df.loc``,
so one grid-search fit of N rows cost O(26 N) DataFrame operations of a millisecond
or more each. It now runs over numpy arrays and writes each column once per fit.

These tests pin that shape without timing anything (#511): the number of DataFrame
column lookups and writes during a fit must not grow with N. Losing it (a per-row
DataFrame access creeping back into the loop) fails them on any CI runner. Wall-clock
numbers come from ``scripts/benchmark_control_chart_hw.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.monitoring.control_charts import ControlChart


def _dataframe_item_accesses(monkeypatch: pytest.MonkeyPatch, n: int, **kwargs: float) -> int:
    """Count ``DataFrame.__getitem__`` / ``__setitem__`` calls during one fit of ``n`` rows."""
    calls: list[str] = []
    real_getitem = pd.DataFrame.__getitem__
    real_setitem = pd.DataFrame.__setitem__

    def counting_getitem(self: pd.DataFrame, key: object) -> object:
        calls.append("get")
        return real_getitem(self, key)

    def counting_setitem(self: pd.DataFrame, key: object, value: object) -> None:
        calls.append("set")
        real_setitem(self, key, value)

    y = 50.0 + 2.0 * np.random.default_rng(n).standard_normal(n)
    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "__getitem__", counting_getitem)
        patch.setattr(pd.DataFrame, "__setitem__", counting_setitem)
        ControlChart(variant="hw").calculate_limits(y, **kwargs)
    return len(calls)


@pytest.mark.parametrize(
    "kwargs",
    [pytest.param({}, id="grid-search"), pytest.param({"ld_1": 0.4, "ld_2": 0.7}, id="fixed-lambdas")],
)
def test_holt_winters_dataframe_access_does_not_grow_with_n(monkeypatch: pytest.MonkeyPatch, kwargs: dict) -> None:
    """A 10x longer series makes exactly as many DataFrame accesses: none of them is per row."""
    short = _dataframe_item_accesses(monkeypatch, 50, **kwargs)
    long = _dataframe_item_accesses(monkeypatch, 500, **kwargs)
    assert short == long, f"DataFrame accesses grew from {short} (N=50) to {long} (N=500): per-row access is back"
