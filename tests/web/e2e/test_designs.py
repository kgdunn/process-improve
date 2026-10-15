"""Making the run sheet from the plan: what the page sends, and how it shows what comes back.

Each design type the page offers is made from the plan; the run sheet must list exactly the
runs Python made, in their randomised order, and the summary must say what the design is. The
design Plan already made for the brief is taken as it is, not made again. A refusal reaches the
status line; the library's warnings are listed under the design.
"""

from __future__ import annotations

import pytest

from tests.web._support import ALL_DESIGNS, FACTORS, factors_named
from tests.web.e2e._page import (
    calls,
    choose,
    generate,
    headers,
    last_reply,
    practise,
    reveal,
    run_sheet,
    set_factors,
    status,
)


@pytest.mark.parametrize("design_type", ALL_DESIGNS)
def test_each_design_reaches_the_run_sheet_as_python_made_it(blank_page, api, design_type):
    page = blank_page
    before = len(calls(page, "api_make_design"))
    generate(page, design_type=design_type)
    if len(calls(page, "api_make_design")) > before:
        request = calls(page, "api_make_design")[-1]["payload"]
        assert request["design_type"] == design_type
        assert request["factors"] == [{key: f[key] for key in ("name", "low", "high", "units")} for f in FACTORS]
        made = api.ok("api_make_design", request)  # the same request again: the design is seeded
    else:  # the design Plan picked for the brief, already made: taken as it is
        listed = last_reply(page, "api_candidates")["result"]["made"]
        made = next(m["design"] for m in listed if m["design_type"] == design_type)

    assert status(page) == (f"{made['n_runs']} runs generated.", "ok")
    assert headers(page) == made["columns"]
    sheet = run_sheet(page)
    assert len(sheet) == made["n_runs"]
    for shown, row in zip(sheet, made["rows"], strict=True):
        assert [float(v) for v in shown[:-1]] == pytest.approx(row[:-1], rel=1e-5), "numbers to six figures"
        assert shown[-1] == "", "no result yet"
    assert page.is_visible("#run-out")
    assert page.is_hidden("#run-empty")
    assert f'design_type="{design_type}"' in page.text_content("#py-design code")


def test_the_summary_says_what_the_design_is(blank_page, api):
    page = blank_page
    generate(page, design_type="fractional_factorial")
    made = api.ok("api_make_design", calls(page, "api_make_design")[-1]["payload"])
    summary = page.text_content("#design-summary p")
    centre = made["facts"]["center_runs"]
    assert summary == f"{made['n_runs']} runs, {centre} of them at the center of the ranges, in random run order."
    assert page.locator("#design-summary ul.facts li").all_text_contents() == made["facts"]["notes"]


def test_a_refused_design_says_why_in_the_status_line(page, api):
    """Three blocks of a two-level factorial is Python's to refuse: it splits into 2, 4, 8, ... blocks."""
    generate(page, design_type="full_factorial", n_blocks=3)
    request = calls(page, "api_make_design")[-1]["payload"]
    assert status(page) == (api.error("api_make_design", request), "error")
    assert page.is_visible("#demo-banner"), "the worked example stays"
    assert page.get_attribute("#workspace", "data-phase") == "plan", "the plan stays on show, to be changed"


def test_the_library_s_warnings_are_listed_under_the_design(page):
    """An even number of factors and no extra center run leave a DSD's staged analysis no error df."""
    set_factors(page, factors_named("A", "B", "C", "D"))
    generate(page, design_type="dsd", n_center_points=1)
    warnings = page.locator("#design-summary ul.warnings li").all_text_contents()
    assert any("degrees of freedom" in w for w in warnings)


def test_runs_outside_the_ranges_are_pointed_out(page):
    generate(page, design_type="ccd", alpha="rotatable")
    assert "Some runs of T, P, F lie outside the ranges you entered" in page.text_content("#design-summary")


def test_replicates_and_a_typed_axial_distance_reach_python(blank_page, api):
    page = blank_page
    set_factors(page, FACTORS[:2])
    generate(page, design_type="ccd", alpha="distance", alpha_value="1.4", n_replicates=2, n_center_points=3)
    request = calls(page, "api_make_design")[-1]["payload"]
    assert (request["alpha"], request["n_replicates"]) == (1.4, 2)
    made = api.ok("api_make_design", request)
    assert made["n_runs"] == 2 * (4 + 4) + 3, "the corners and axial runs twice, the center runs once"
    assert status(page) == (f"{made['n_runs']} runs generated.", "ok")
    code = page.text_content("#py-design code")
    assert "alpha=1.4" in code
    assert "n_replicates=2" in code


def test_a_categorical_factor_s_labels_reach_the_run_sheet_and_the_analysis(blank_page):
    page = blank_page
    set_factors(page, [*FACTORS[:2], {"name": "Supplier", "low": "old", "high": "new", "units": ""}])
    generate(page, design_type="full_factorial", n_center_points=2)
    request = calls(page, "api_make_design")[-1]["payload"]
    assert request["factors"][2] == {"name": "Supplier", "low": "old", "high": "new", "units": ""}
    supplier = headers(page).index("Supplier")
    assert {row[supplier] for row in run_sheet(page)} == {"old", "new"}
    assert page.is_hidden("#design-stale"), "the labels the form holds are the design's"
    assert 'Factor(name="Supplier", type="categorical", levels=["old", "new"])' in page.text_content("#py-design code")
    practise(page)
    terms = page.locator("#analysis table").first.locator("tbody tr td:first-child").all_text_contents()
    assert "Supplier" in terms
    assert '"Supplier": runs["Supplier"].map({"old": -1.0, "new": 1.0}),' in page.text_content("#py-analyse code")


def test_the_design_plan_picked_is_taken_as_made_and_its_settings_make_another(blank_page):
    """The run sheet of a design Plan made and scored is that design; a setting changed since makes it again."""
    page = blank_page
    picked = page.get_attribute('#candidates tbody tr[aria-selected="true"]', "data-design")
    generate(page)
    assert calls(page, "api_make_design") == [], "Plan made it already"
    listed = last_reply(page, "api_candidates")["result"]["made"]
    rows = next(m["design"]["rows"] for m in listed if m["design_type"] == picked)
    assert [r[:-1] for r in run_sheet(page)] == [[f"{v:g}" for v in r[:-1]] for r in rows]
    choose(page, n_center_points=4)
    reveal(page, "#generate")
    page.click("#generate")
    page.wait_for_function("window.__pi.count('api_make_design') === 1 && window.__pi.idle()")
    assert calls(page, "api_make_design")[0]["payload"]["n_center_points"] == 4
