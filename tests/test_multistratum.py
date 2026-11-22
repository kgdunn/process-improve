"""Multistratum analysis of variance: split-plot and strip-plot layouts, and their tests on each stratum's error.

The layouts are NIST's electroplating and implant-anneal examples (Engineering Statistics
Handbook, section 5.5.5), which print sources and degrees of freedom without data; the data are
Montgomery's tensile-strength split plot (Example 14.4) and catalyst design (Example 4.5).
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import multistratum_anova, reference_catalogue
from process_improve.experiments.multistratum import error_terms
from process_improve.tool_spec import execute_tool_call


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


def test_a_stratum_the_terms_fit_exactly_has_no_f_test():
    """Its residual is rounding error only: 0, with no F, on every BLAS, where one gave 1e15 and another 0/0."""
    runs = pd.DataFrame(
        [
            {"rep": r, "M": m, "T": t, "y": r + 2 * m + t + 0.1 * ((r + m + t) % 3)}
            for r in (1, 2)
            for m in (0, 1)
            for t in (0, 1)
        ]
    )
    found = multistratum_anova(runs, "M * T", "rep/M", response="y")
    table = found["table"].set_index(["stratum", "term"])
    assert table.loc[("rep:M", "Residual"), "ss"] == 0.0
    assert np.isnan(table.loc[("rep:M", "M"), "f"])
    assert np.isnan(table.loc[("rep:M", "M"), "p_value"])
    assert table.loc[("Within", "T"), "f"] == pytest.approx(80.2222, abs=5e-4)


def test_an_error_term_that_names_every_run_leaves_the_within_stratum_empty():
    """Error(replicate/method/temperature): each sample is a unit, so that stratum holds Montgomery's subplots."""
    runs = _montgomery("montgomery-tensile-strength-split-plot")
    found = multistratum_anova(runs, "method * temperature", "replicate/method/temperature", response="strength")
    table = found["table"].set_index(["stratum", "term"])
    assert table.loc[("replicate:method:temperature", "temperature"), "ss"] == pytest.approx(434.0833, abs=5e-5)
    assert table.loc[("replicate:method:temperature", "Residual"), "ss"] == pytest.approx(71.5, abs=5e-5)
    assert found["df"]["Within"] == {"Residual": 0}
    assert "Within" not in table.index.get_level_values("stratum")


def test_the_catalogue_reads_a_term_of_a_stratum_from_the_runs_it_prints():
    """The catalog's check fits the experiment's own runs, and names the stratum a term is not in."""
    path = reference_catalogue._CATALOGUE_DIR / "montgomery-2013.json"
    entry = next(
        e
        for e in json.loads(path.read_text(encoding="utf-8"))["experiments"]
        if e["name"] == "montgomery-tensile-strength-split-plot"
    )
    check = {"model": "method * temperature", "strata": "replicate/method", "stratum": "replicate:method"}
    assert reference_catalogue._multistratum(entry, {**check, "term": "method", "key": "ss"}) == pytest.approx(
        128.3889, abs=5e-5
    )
    with pytest.raises(KeyError, match="No term 'temperature' in the stratum 'replicate:method'"):
        reference_catalogue._multistratum(entry, {**check, "term": "temperature", "key": "ss"})


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


# --------------------------------------------------------------------------- the tool an agent calls
def test_the_tool_gives_the_skeleton_and_the_tests_through_the_server_s_path():
    """``multistratum_anova`` as an agent calls it: Montgomery's Table 14.18 and, without a response, its skeleton."""
    runs = _montgomery("montgomery-tensile-strength-split-plot")
    asked = {"data": runs.to_dict("records"), "model": "method * temperature", "strata": "replicate/method"}
    skeleton = execute_tool_call("multistratum_anova", asked)
    assert skeleton["df"]["replicate:method"] == {"method": 2, "Residual": 4}
    assert skeleton["df"]["Within"] == {"temperature": 3, "method:temperature": 6, "Residual": 18}
    tested = execute_tool_call("multistratum_anova", {**asked, "response_column": "strength"})
    row = next(r for r in tested["table"] if (r["stratum"], r["term"]) == ("replicate:method", "method"))
    assert row["f"] == pytest.approx(64.1944 / 9.0694, abs=5e-4)


def test_the_tool_refuses_a_column_the_runs_do_not_have_in_a_sentence():
    asked = {"data": _crossed("AB").to_dict("records"), "model": "A * B", "strata": "rep/plot"}
    assert execute_tool_call("multistratum_anova", asked)["error"] in {
        f"The runs have no column {c!r}." for c in ("rep", "plot")
    }


def test_the_tool_refuses_more_runs_than_it_takes():
    asked = {"data": [{"A": 0, "B": 0}] * 2001, "model": "A * B", "strata": "A"}
    assert "the tool takes at most 2000" in execute_tool_call("multistratum_anova", asked)["error"]
