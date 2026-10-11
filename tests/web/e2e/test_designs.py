"""Generating a design from step 1: what the page sends, and how it shows what comes back.

Each design type the page offers is generated from the form; the run sheet must list
exactly the runs Python made, in their randomised order, and the summary must say what the
design is. A refusal reaches the status line; the library's warnings are listed under the
design.
"""

from __future__ import annotations

import pytest

from tests.web._support import ALL_DESIGNS, FACTORS, factors_named
from tests.web.e2e._page import calls, generate, headers, run_sheet, set_factors, status


@pytest.mark.parametrize("design_type", ALL_DESIGNS)
def test_each_design_reaches_the_run_sheet_as_python_made_it(blank_page, api, design_type):
    page = blank_page
    generate(page, design_type=design_type)
    request = calls(page, "api_make_design")[-1]["payload"]
    assert request["design_type"] == design_type
    assert request["factors"] == [{key: f[key] for key in ("name", "low", "high", "units")} for f in FACTORS]
    made = api.ok("api_make_design", request)  # the same request again: the design is seeded

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
    assert summary.startswith(f"{made['n_runs']} runs, {made['facts']['center_runs']} of them at the center")
    assert "main effects model" in summary
    assert page.locator("#design-summary ul.facts li").all_text_contents() == made["facts"]["notes"]


def test_a_refused_design_says_why_in_the_status_line(page):
    set_factors(page, [{**FACTORS[0], "name": "a b"}, *FACTORS[1:]])
    generate(page)
    assert status(page) == ("Factor names must be unique and look like variable names (letters, digits, _).", "error")
    assert page.is_visible("#demo-banner"), "the worked example stays"


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
