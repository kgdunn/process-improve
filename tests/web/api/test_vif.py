"""The variance inflation factor (VIF) of each term the analysis fits, and the note when terms are correlated.

Runs as given, or runs left blank, can leave the model's terms correlated, so their standard
errors grow and no single term may look active while the model explains the response. Each
coefficient but the intercept carries its VIF, from the runs fitted; a note names each term whose
VIF is above 5. The VIFs are checked here against a regression of each term's column on the others.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.web._support import CORRELATED_RUNS, FACTORS, coded, correlated_study, factors_named, held_of, run_sheet


def direct_vifs(columns: dict[str, np.ndarray]) -> dict[str, float]:
    """Return each column's VIF as 1 / (1 - R²), with R² from regressing it on an intercept and the other columns."""
    found = {}
    for name, column in columns.items():
        others = np.column_stack([np.ones(len(column)), *(c for n, c in columns.items() if n != name)])
        fitted = others @ np.linalg.lstsq(others, column, rcond=None)[0]
        r_squared = 1 - ((column - fitted) ** 2).sum() / ((column - column.mean()) ** 2).sum()
        found[name] = 1 / (1 - r_squared)
    return found


def vifs_of(analysis: dict) -> dict[str, float | None]:
    """Index the VIF of each coefficient by the label the page shows."""
    return {c["label"]: c["vif"] for c in analysis["coefficients"]}


def vif_notes(analysis: dict) -> list[str]:
    return [note for note in analysis["notes"] if "variance inflation factor" in note]


def test_an_orthogonal_design_gives_every_term_a_vif_of_one_and_no_note(api):
    design = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 0})
    responses = [float((3 * i) % 7) for i in range(design["n_runs"])]
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": responses, "model": "interactions"})
    vifs = vifs_of(analysis)
    intercept = vifs.pop("Intercept")
    assert intercept is None, "the intercept has no VIF"
    assert vifs == dict.fromkeys(["T", "P", "F", "T·P", "T·F", "P·F"], pytest.approx(1.0))
    assert vif_notes(analysis) == []


def test_the_worked_example_s_terms_are_close_to_uncorrelated(api, held, simulated):
    """A Box-Behnken design's squares share their center runs, so their VIFs are a little above 1, and no more."""
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    vifs = vifs_of(analysis)
    del vifs["Intercept"]
    assert max(vifs.values()) == pytest.approx(1.0111, abs=1e-4)
    assert vif_notes(analysis) == []


def test_runs_as_given_whose_columns_nearly_match_name_the_correlated_terms(api):
    design = api.ok("api_make_design", correlated_study())
    responses = [10 + 2 * a - c + 0.1 * i for i, (a, _, c) in enumerate(CORRELATED_RUNS)]
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": responses, "model": "main_effects"})
    x = np.array(CORRELATED_RUNS, dtype=float)
    expected = direct_vifs({"A": x[:, 0], "B": x[:, 1], "C": x[:, 2]})
    vifs = vifs_of(analysis)
    assert vifs == {"Intercept": None, **{name: pytest.approx(v) for name, v in expected.items()}}
    assert (round(vifs["A"], 6), round(vifs["B"], 6), round(vifs["C"], 6)) == (10.8, 11, 1.2)
    (note,) = vif_notes(analysis)
    assert note.startswith("A (VIF 11) and B (VIF 11) are correlated with other terms in these runs.")
    assert "√VIF" in note
    assert note.endswith("Add runs that separate the correlated terms, or drop one of them from the model.")
    assert "C (VIF" not in note, "a VIF of 1.2 is not worth a note"


def test_a_categorical_factor_is_coded_as_the_fit_codes_it(api):
    """A label is -1 or +1 in the fit, and its VIF is that column's; S follows T but in the run where T is centred."""
    factors = [FACTORS[0], {"name": "S", "low": "old", "high": "new", "units": ""}, FACTORS[1]]
    runs = [[175, "old", 1], [150, "old", 3], [200, "new", 1], [200, "new", 3], [150, "old", 1], [150, "old", 3]]
    runs += [[200, "new", 1], [200, "new", 3]]
    design = api.ok("api_make_design", {"design_type": "given", "factors": factors, "runs": runs})
    responses = [50 + 0.1 * i + (i % 3) for i in range(len(runs))]
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": responses, "model": "main_effects"})
    x = coded(run_sheet(design), design["spec"]["factors"])
    expected = direct_vifs({name: x[name].to_numpy() for name in ("T", "S", "P")})
    vifs = vifs_of(analysis)
    assert {name: vifs[name] for name in expected} == {name: pytest.approx(v) for name, v in expected.items()}
    (note,) = vif_notes(analysis)
    assert (round(vifs["T"], 6), round(vifs["S"], 6)) == (11, 10.8), "as A and B in the correlated study"
    assert note.startswith("T (VIF 11) and S (VIF 11) are correlated with other terms in these runs.")


def test_a_run_left_blank_in_a_blocked_design_correlates_the_terms_with_the_blocks(api):
    """The VIFs are from the runs fitted, the block contrast's column among them, coded as the fit codes it."""
    spec = {"design_type": "full_factorial", "factors": factors_named(*"ABC"), "n_blocks": 2, "n_center_points": 0}
    design = api.ok("api_make_design", spec)
    responses: list[float | None] = [float(i % 3) + 0.5 * i for i in range(design["n_runs"])]
    responses[0] = None
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": responses, "model": "main_effects"})
    sheet = run_sheet(design).iloc[1:]
    x = coded(sheet, design["spec"]["factors"])
    block = np.where(sheet["Block"] == 1, 1.0, -1.0)  # the fit's contrast: +1 in block 1, -1 in the last block
    expected = direct_vifs({"Block 1": block, **{name: x[name].to_numpy() for name in "ABC"}})
    vifs = vifs_of(analysis)
    del vifs["Intercept"]
    assert vifs == {label: pytest.approx(v) for label, v in expected.items()}
    assert max(vifs.values()) > 1, "the blank run leaves the terms correlated with the blocks"


def test_a_block_s_vif_is_never_named_in_the_note(bootstrap):
    """A block's VIF says how unevenly the runs fill the blocks, which dropping a term cannot change."""
    assert bootstrap._correlated_sentences({"Block1": 9.0, "A": 1.5}, {"Block1": "Block 1"}, ["A"]) == []
    (note,) = bootstrap._correlated_sentences({"Block1": 9.0, "A": 6.0}, {"Block1": "Block 1"}, ["A"])
    assert note.startswith("A (VIF 6.0) is correlated with other terms in these runs.")


@pytest.mark.parametrize(
    ("value", "text"),
    [(5.04, "5.0"), (5.0001, "5.0"), (9.8, "9.8"), (9.96, "10"), (12.3, "12"), (99.6, "100"), (1234.5, "1200")],
)
def test_a_vif_is_written_to_two_significant_figures(bootstrap, value, text):
    assert bootstrap._two_figures(value) == text
