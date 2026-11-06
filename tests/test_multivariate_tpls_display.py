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


@pytest.mark.parametrize("with_z", [True, False], ids=["with-z", "without-z"])
def test_tpls_display_results_lines_up_its_columns(tpls_pyphi_data: dict, with_z: bool) -> None:
    """#1083: every row has its R2 columns and its "|" under the header's, with or without a Z block.

    Without a Z block the "-" placeholder was one character narrower than a number, so the F and
    Y columns, and the timing after them, sat one place to the left of their headers.
    """
    d_matrix = tpls_pyphi_data.pop("D")
    if not with_z:
        tpls_pyphi_data["Z"] = {}
    model = TPLS(n_components=2, d_matrix=d_matrix).fit(DataFrameDict(tpls_pyphi_data))
    lines = model.display_results().splitlines()
    header = next(line for line in lines if line.startswith("LV #"))
    rows = [line for line in lines if re.match(r"LV \d", line)]
    right_edges = [match.end() for match in re.finditer(r"\)", header)]  # each "sum(R2: X)" ends at ")"
    assert len(rows) == 2
    for row in rows:
        assert row.index("|") == header.index("|")
        assert all(row[edge - 1] != " " and row[edge] == " " for edge in right_edges[:-1]), row
    if not with_z:
        assert all(row[header.index("Z)") + 1] == "-" for row in rows)


def test_tpls_help_returns_the_quick_reference(tpls_pyphi_data: dict) -> None:
    """help() needs no fit, and returns the quick reference as plain text."""
    text = TPLS(n_components=2, d_matrix=tpls_pyphi_data["D"]).help()
    assert text.startswith("Help for the TPLS Estimator.")
    for call in ("tpls.diagnose(X_new)", "tpls.display_results()", "tpls.help()", ".hotellings_t2_limit()"):
        assert call in text


def test_tpls_help_keeps_its_columns(tpls_pyphi_data: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """help() is its docstring dedented, columns and blank lines as written.

    Removing every run of eight spaces, as it did, also removed the spaces that line up the
    columns, and ran ".spe" into its description. Under ``python -OO``, which drops
    docstrings, it is empty.
    """
    lines = TPLS(n_components=2, d_matrix=tpls_pyphi_data["D"]).help().splitlines()
    assert lines[:3] == ["Help for the TPLS Estimator.", "", "Data organization"]
    assert lines[7] == "Build model:                tpls = TPLS(n_components=2, d_matrix=d_matrix).fit(X)"
    spe = ".spe" + " " * 24 + "Squared prediction error for each block" + " " * 21 + "[dict of pd.DataFrames]"
    assert lines[18:21] == [spe, "", ""]
    vip = ".vip()                      Variable importance (VIP) for the D- and F-blocks          [dict]"
    assert lines[-1] == vip

    monkeypatch.setattr(TPLS.help, "__doc__", None)
    assert TPLS(n_components=2, d_matrix=tpls_pyphi_data["D"]).help() == ""
