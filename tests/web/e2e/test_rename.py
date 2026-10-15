"""Renaming a factor: a name or units is a label, which the page takes once the reader leaves its box.

A run sheet already made keeps its runs, their order and their results, and is not marked as
changed. Python writes again what it wrote with the old names (``api_relabel``), and the analysis
is fitted again under the new ones, so the run sheet's headings, the design's notes, the true
process, the fitted equation, the coefficients, the charts, the Python and the study's title all
follow. A name Python would refuse is not taken: it stays marked, and the old one stays everywhere.
"""

from __future__ import annotations

import re
from typing import Any

from tests.web._support import factors_named
from tests.web.e2e._page import (
    calls,
    candidates_done,
    choose,
    generate,
    headers,
    practise,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
)

#: The worked example's run sheet: a Box-Behnken design in T, P and F, whose results are its Yield column.
EXAMPLE = ["Std order", "Run order", "T", "P", "F", "Yield"]


def factor_box(page: Any, row: int, key: str = "name") -> Any:  # noqa: ANN401 - a Playwright Page and Locator
    """Return one box of the factor table, its phase on show."""
    reveal(page, "#factors")
    return page.locator("#factors tbody tr").nth(row).locator(f"[name={key}]")


def rename(page: Any, row: int, text: str, key: str = "name", leave: str = "Tab") -> None:  # noqa: ANN401
    """Type a factor's new name (or units), leave the box as a reader would, and wait for the page to settle."""
    box = factor_box(page, row, key)
    box.fill(text)
    box.press(leave)
    settled(page)


def words(text: str) -> set[str]:
    """Return the words of a text that could name a factor, as an equation or a term's label writes them."""
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))


def chart(page: Any, n: int, read: str) -> Any:  # noqa: ANN401
    """Read something off the n-th chart in Learn, as Plotly holds it."""
    return page.locator("#charts .js-plotly-plot").nth(n).evaluate(read)


def test_a_name_is_taken_when_the_reader_leaves_its_box(page):
    box = factor_box(page, 0)
    box.fill("Temperature")
    assert calls(page, "api_relabel") == [], "a name half typed reaches nothing"
    assert headers(page) == EXAMPLE
    box.press("Tab")
    settled(page)
    assert headers(page) == ["Std order", "Run order", "Temperature", "P", "F", "Yield"]
    rename(page, 1, "Pressure", leave="Enter")
    assert headers(page) == ["Std order", "Run order", "Temperature", "Pressure", "F", "Yield"]


def test_a_renamed_factor_is_renamed_everywhere_and_its_runs_and_results_stay(page):
    sheet = run_sheet(page)
    rename(page, 0, "Temperature")

    assert run_sheet(page) == sheet, "the same runs, in the same order, with the same results"
    assert calls(page, "api_make_design") == []
    assert calls(page, "api_simulate") == [], "the practice results are kept, not simulated again"
    assert page.text_content("#badge-plan") == ""
    assert page.is_hidden("#design-stale"), "a factor renamed is not a new design"

    for shown in (
        page.text_content("#truth-equation"),
        page.text_content("#analysis p.equation"),
        " ".join(page.locator("#analysis table").first.locator("tbody td:first-child").all_text_contents()),
        " ".join(chart(page, 0, "d => d.layout.yaxis.categoryarray")),
    ):
        assert "Temperature" in words(shown), shown
        assert "T" not in words(shown), shown
    assert chart(page, 1, "d => d.layout.xaxis.title.text") == "Temperature (degC)"
    assert "over Temperature and P" in page.locator("#charts figcaption").first.text_content()
    for panel in ("#py-design", "#py-run", "#py-analyse"):
        code = page.text_content(f"{panel} code")
        assert "Temperature" in code, panel
        assert not re.search(r'"T"|\bT\b', code), panel


def test_the_design_s_notes_name_the_factor_anew(blank_page, api):
    """A fraction in blocks: its resolution, its aliases and the effects its blocks confound all name factors."""
    page = blank_page
    set_factors(page, factors_named(*"ABCD"))
    generate(page, design_type="fractional_factorial", resolution=4, n_blocks=2)
    request = calls(page, "api_make_design")[-1]["payload"]
    rename(page, 0, "Temp")

    renamed = [{**request["factors"][0], "name": "Temp"}, *request["factors"][1:]]
    made = api.ok("api_make_design", {**request, "factors": renamed})
    assert page.locator("#design-summary ul.facts li").all_text_contents() == made["facts"]["notes"]
    assert headers(page) == made["columns"]
    assert any("Temp·B and C·D are confounded with the blocks" in note for note in made["facts"]["notes"])


def test_units_are_taken_as_names_are(page):
    rename(page, 0, "K", key="units")
    assert page.locator("#design-table thead th").nth(2).locator(".col-unit").text_content() == "K"
    assert 'units="K"' in page.text_content("#py-design code")
    assert chart(page, 1, "d => d.layout.xaxis.title.text") == "T (K)"
    assert page.is_hidden("#design-stale")


def test_a_name_python_would_refuse_is_not_taken(page):
    refused = {
        "P": "P: two factors cannot share a name.",
        "Yield": "Yield: the run sheet has a Yield column already.",
        "2T": "2T: a name is letters, digits and _",
    }
    for name, said in refused.items():
        rename(page, 0, name)
        assert said in page.text_content("#factor-problems")
        assert factor_box(page, 0).get_attribute("aria-invalid") is not None
        assert page.is_disabled("#generate")
        assert headers(page) == EXAMPLE, "the old name stays everywhere"
    assert calls(page, "api_relabel") == []

    rename(page, 0, "Temperature")
    assert page.is_hidden("#factor-problems")
    assert headers(page)[2] == "Temperature"


def test_a_blocked_run_sheet_s_block_column_is_no_factor_s_name(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial", n_blocks=2)
    rename(page, 0, "Block")
    assert "Block: the run sheet has a Block column already." in page.text_content("#factor-problems")
    assert calls(page, "api_relabel") == []


def test_the_terms_chosen_for_the_model_follow_a_rename(page):
    reveal(page, "#model")
    page.select_option("#model", "terms")
    settled(page)
    ticked = page.locator("#term-list input:checked").count()
    rename(page, 0, "Temperature")
    reveal(page, "#term-list")
    labels = page.locator("#term-list label:has(input:checked)").all_text_contents()
    assert len(labels) == ticked, "the same terms, still ticked"
    assert "Temperature" in words(" ".join(labels))
    assert "T" not in words(" ".join(page.locator("#term-list label").all_text_contents()))


def test_the_design_picked_in_design_is_renamed_not_made_again(blank_page):
    """A design the reader picked stays picked through a rename, and its run sheet takes the new name."""
    page = blank_page
    page.click('#candidates tbody tr[aria-selected="true"]')
    rename(page, 0, "Temperature")
    candidates_done(page)
    generate(page)
    assert calls(page, "api_make_design") == [], "the design Design made, renamed"
    assert len(calls(page, "api_relabel")) == 1
    assert headers(page)[2:5] == ["Temperature", "P", "F"]


def test_a_renamed_study_is_kept_renamed(context, blank_page):
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=2)
    practise(page)
    rename(page, 0, "Temperature")
    sheet = run_sheet(page)

    kept = saved_studies(page, lambda saved: "Temperature" in saved[0]["design"]["columns"])[0]
    assert kept["title"] == "Full factorial (2-level) in Temperature, P and F"
    assert kept["design"]["spec"]["factors"][0]["name"] == "Temperature"

    again = context.new_page()
    again.goto(page.url.split("#")[0])
    again.wait_for_selector("#campaign-banner", state="visible")
    settled(again)
    assert headers(again) == headers(page)
    assert run_sheet(again) == sheet
    assert "Temperature" in words(again.text_content("#analysis p.equation"))


def test_a_changed_level_is_a_new_design_and_waits_for_its_run_sheet(page):
    """Only the labels are taken from a run sheet: a level changed with them leaves it as it was, out of date."""
    choose(page, n_center_points=3)
    factor_box(page, 0, "high").fill("210")
    rename(page, 0, "Temperature")
    assert calls(page, "api_relabel") == []
    assert headers(page) == EXAMPLE
    assert page.text_content("#badge-plan") == "changed"
