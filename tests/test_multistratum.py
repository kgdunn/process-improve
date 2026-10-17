"""Multistratum analysis of variance: split-plot and strip-plot layouts, and their tests on each stratum's error.

The layouts are NIST's electroplating and implant-anneal examples (Engineering Statistics
Handbook, section 5.5.5), which print sources and degrees of freedom without data; the data are
Montgomery's tensile-strength split plot (Example 14.4) and catalyst design (Example 4.5).
"""

from __future__ import annotations

import itertools
import json

import pandas as pd
import pytest

from process_improve.experiments import multistratum_anova, reference_catalogue
from process_improve.experiments.multistratum import error_terms


def _crossed(names: str | list[str]) -> pd.DataFrame:
    """Every combination of two levels of each factor named: a layout without a response."""
    return pd.DataFrame(list(itertools.product(*[[0, 1]] * len(names))), columns=list(names))


def _montgomery(name: str) -> pd.DataFrame:
    path = reference_catalogue._CATALOGUE_DIR / "montgomery-2013.json"
    entry = next(e for e in json.loads(path.read_text(encoding="utf-8"))["experiments"] if e["name"] == name)
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


@pytest.mark.parametrize(
    ("strata", "expected"),
    [("rep/plot", ["rep", "rep:plot"]), ("row + column", ["row", "column"]), (["a", "a:b"], ["a", "a:b"])],
)
def test_error_terms_expand_nesting(strata, expected):
    assert error_terms(strata) == expected


def test_replicated_whole_plots_put_their_interactions_in_the_whole_plot_stratum():
    """NIST's first scenario: concentration set once per replicate, Rep*Conc its whole-plot error."""
    layout = multistratum_anova(_crossed(["rep", "S", "T", "A"]), "rep * S * T * A", "rep/S")
    assert layout["df"]["rep"] == {"rep": 1, "Residual": 0}
    assert layout["df"]["rep:S"] == {"S": 1, "rep:S": 1, "Residual": 0}
    assert len(layout["df"]["Within"]) == 12 + 1  # twelve subplot terms and an empty residual
    assert [s["units"] for s in layout["strata"]] == [2, 4, 16]


@pytest.mark.parametrize(
    ("units", "whole_plot", "within"),
    [("S:T", {"S": 1, "T": 1, "S:T": 1, "Residual": 0}, 8), ("S:T:rep", {"S": 1, "T": 1, "S:T": 1, "Residual": 4}, 4)],
)
def test_batches_of_strips(units, whole_plot, within):
    """NIST's second and third scenarios: four batches of four strips, or eight of two."""
    layout = multistratum_anova(_crossed(["S", "T", "rep", "A"]), "S * T * A", units)
    assert layout["df"][units] == whole_plot
    assert layout["df"]["Within"] == {"A": 1, "S:A": 1, "T:A": 1, "S:T:A": 1, "Residual": within}


def test_a_strip_plot_has_a_stratum_for_each_process_step():
    layout = multistratum_anova(_crossed("ABCDEF"), "A * B * C * D * E * F", "A:B:C + D:E:F")
    assert layout["df"]["A:B:C"]["A:B:C"] == layout["df"]["D:E:F"]["D:E:F"] == 1
    assert sum(layout["df"]["Within"].values()) == 64 - 1 - 7 - 7


def test_split_plot_tests_each_term_on_its_own_stratum():
    """Montgomery's Table 14.18: methods on the whole-plot error, temperatures on the subplot error."""
    runs = _montgomery("montgomery-tensile-strength-split-plot")
    found = multistratum_anova(runs, "method * temperature", "replicate/method", response="strength")
    table = found["table"].set_index(["stratum", "term"])
    assert table.loc[("replicate", "Residual"), "ss"] == pytest.approx(77.5556, abs=5e-5)
    assert table.loc[("replicate:method", "method"), "ss"] == pytest.approx(128.3889, abs=5e-5)
    assert table.loc[("replicate:method", "Residual"), "ss"] == pytest.approx(36.2778, abs=5e-5)
    assert table.loc[("replicate:method", "method"), "f"] == pytest.approx(64.1944 / 9.0694, abs=5e-4)
    assert table.loc[("Within", "temperature"), "ss"] == pytest.approx(434.0833, abs=5e-5)
    assert table.loc[("Within", "Residual"), "df"] == 18  # replicates x B and x AB, pooled: 20.67 + 50.83
    assert table.loc[("Within", "Residual"), "ss"] == pytest.approx(71.5, abs=5e-5)


def test_an_incomplete_block_design_estimates_treatments_in_two_strata():
    """The catalyst design: treatments between blocks and within them, the intrablock analysis the second."""
    runs = _montgomery("montgomery-catalyst-bibd")
    found = multistratum_anova(runs, "catalyst", "block", response="y")
    assert [row["term"] for row in found["strata"][0]["terms"]] == ["catalyst"]
    within = found["strata"][1]
    assert within["terms"][0]["ss"] == pytest.approx(22.75)  # the adjusted treatment sum of squares
    assert (within["residual"]["df"], within["residual"]["ss"]) == (5, pytest.approx(3.25))


def test_a_missing_column_is_refused():
    with pytest.raises(ValueError, match="no column 'plot'"):
        multistratum_anova(_crossed("AB"), "A * B", "plot")
