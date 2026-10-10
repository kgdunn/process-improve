"""Tests for the text output of ``TPLS``: ``display_results`` and ``help``.

The display_results method (multivariate.methods.TPLS.display_results,
lines 2538-2604) was previously uncovered: the only reference in the
test suite lives inside ``manual_cross_validation`` which is itself
never invoked.
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd
import pytest

from process_improve.multivariate.methods import TPLS, DataFrameDict


@pytest.fixture
def tpls_pyphi_data() -> dict[str, dict[str, pd.DataFrame]]:
    """Load the pyphi TPLS example dataset."""
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate" / "tpls-pyphi"
    properties = {
        f"Group {i}": pd.read_csv(folder / f"properties_Group{i}.csv", sep=",", index_col=0, header=0).astype("float64")
        for i in range(1, 6)
    }
    formulas = {
        f"Group {i}": pd.read_csv(folder / f"formulas_Group{i}.csv", sep=",", index_col=0, header=0).astype("float64")
        for i in range(1, 6)
    }
    process_conditions = {
        "Conditions": pd.read_csv(folder / "process_conditions.csv", sep=",", index_col=0, header=0).astype("float64"),
    }
    quality_indicators = {
        "Quality": pd.read_csv(folder / "quality_indicators.csv", sep=",", index_col=0, header=0).astype("float64"),
    }
    return {"Z": process_conditions, "D": properties, "F": formulas, "Y": quality_indicators}


def test_tpls_display_results_cumulative(tpls_pyphi_data: dict) -> None:
    """display_results should return a multi-line string with cumulative R-squared."""
    n_components = 2
    d_matrix = tpls_pyphi_data.pop("D")
    model = TPLS(n_components=n_components, d_matrix=d_matrix)
    model.fit(DataFrameDict(tpls_pyphi_data))

    output = model.display_results(show_cumulative_stats=True)
    assert isinstance(output, str)
    # The header is rendered with the "sum(" prefix in cumulative mode.
    assert "sum(R2:" in output
    # Each LV row should appear in the output.
    for a in range(1, n_components + 1):
        assert f"LV {a}" in output


def test_tpls_display_results_per_component(tpls_pyphi_data: dict) -> None:
    """display_results should also render per-component (non-cumulative) stats."""
    d_matrix = tpls_pyphi_data.pop("D")
    model = TPLS(n_components=2, d_matrix=d_matrix)
    model.fit(DataFrameDict(tpls_pyphi_data))

    output = model.display_results(show_cumulative_stats=False)
    # Non-cumulative header omits the "sum(" prefix.
    assert "R2: D" in output
    assert "sum(R2:" not in output


def test_tpls_display_results_unfitted_raises(tpls_pyphi_data: dict) -> None:
    """Calling display_results before fit() should raise RuntimeError."""
    d_matrix = tpls_pyphi_data["D"]
    model = TPLS(n_components=2, d_matrix=d_matrix)

    with pytest.raises(RuntimeError, match="not fitted"):
        model.display_results()


def test_tpls_display_results_flags_a_component_stopped_by_max_iter(tpls_pyphi_data: dict) -> None:
    """A component that ran out of iterations, rather than converging, is marked in its row."""
    d_matrix = tpls_pyphi_data.pop("D")
    capped = TPLS(n_components=2, d_matrix=d_matrix, max_iter=1).fit(DataFrameDict(tpls_pyphi_data))
    converged = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(tpls_pyphi_data))

    def component_rows(model: TPLS) -> list[str]:
        return [line for line in model.display_results().splitlines() if re.match(r"LV \d", line)]

    assert [row.endswith("** (max iter reached)") for row in component_rows(capped)] == [True, True]
    assert not any("max iter reached" in row for row in component_rows(converged))


def test_tpls_help_returns_the_quick_reference(tpls_pyphi_data: dict) -> None:
    """help() needs no fit, and returns the quick reference as plain text."""
    text = TPLS(n_components=2, d_matrix=tpls_pyphi_data["D"]).help()
    assert text.startswith("Help for the TPLS Estimator.")
    for call in ("tpls.diagnose(X_new)", "tpls.display_results()", "tpls.help()", ".hotellings_t2_limit()"):
        assert call in text
