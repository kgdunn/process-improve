"""The reading level in the page: Beginner, Intermediate or Adventurous, each adding to the ones below.

The level is the concept map's own, kept under its key, so a choice made on either page holds on both. A level
leaves designs and charts off the page and says how many; it never leaves out a result, a warning, or what the
study on the page was made with. The other browser tests run at the highest level (``tests/web/harness.js``).
"""

from __future__ import annotations

import re

from tests.web.e2e._page import calls, enter_results, generate, practise, reveal, settled


def set_level(page, level: int) -> None:
    """Choose a level with the switch in the header, as a reader does."""
    page.locator(f'#level input[value="{level}"]').check(force=True)  # the box itself is not drawn: its name is
    page.wait_for_function(f"document.documentElement.dataset.level === '{level}'")


def listed_designs(page) -> list[str]:
    """Return the designs the design list offers at the page's level."""
    return page.locator("#design-type option").evaluate_all(
        "(options) => options.filter((o) => !o.hidden).map((o) => o.value)"
    )


def more_charts(page) -> list[str]:
    """Open "More charts" if it is shut, and return the charts under it."""
    if not page.locator("#more-charts").evaluate("(box) => box.open"):
        page.click("#more-charts > summary")
    page.wait_for_function(
        "document.querySelectorAll('#more-charts figure').length === "
        "document.querySelectorAll('#more-charts .js-plotly-plot').length "
        "&& document.querySelectorAll('#more-charts figure').length > 0"
    )
    return page.locator("#more-charts figure").evaluate_all("(cards) => cards.map((c) => c.dataset.chart)")


def test_a_first_visit_opens_on_intermediate(blank_page):
    page = blank_page
    page.evaluate("localStorage.removeItem('factorial-map-level')")
    page.reload()
    page.wait_for_function("document.documentElement.dataset.level === '2'")
    assert page.locator("#level input:checked").get_attribute("value") == "2"
    assert "d_optimal" not in listed_designs(page)
    assert {"full_factorial", "fractional_factorial", "ccd"} <= set(listed_designs(page))


def test_each_level_adds_designs_and_says_how_many_more_there_are(blank_page):
    page = blank_page
    set_level(page, 1)
    held = page.input_value("#design-type")
    assert set(listed_designs(page)) == {"full_factorial", held}, "the first design, and the one the plan holds"
    assert re.fullmatch(
        r"\d+ more designs at Intermediate, \d+ at Adventurous\. Show them", page.text_content("#level-designs")
    )
    set_level(page, 2)
    assert re.fullmatch(r"\d+ more designs at Adventurous\. Show them", page.text_content("#level-designs"))
    page.click("#level-designs button")
    page.wait_for_function("document.documentElement.dataset.level === '3'")
    assert page.locator("#level-designs").is_hidden()
    assert len(listed_designs(page)) == 17, "every design for factors with ranges, and the three mixtures"


def test_the_choice_is_kept_under_the_concept_map_s_key(blank_page):
    page = blank_page
    set_level(page, 1)
    assert page.evaluate("localStorage.getItem('factorial-map-level')") == "1"
    page.reload()
    page.wait_for_function("document.documentElement.dataset.level === '1'")
    assert page.locator("#level input:checked").get_attribute("value") == "1"


def test_a_study_keeps_its_design_at_every_level(blank_page):
    page = blank_page
    generate(page, design_type="d_optimal")
    set_level(page, 1)
    assert "d_optimal" in listed_designs(page), "the study was made with it"
    assert page.input_value("#design-type") == "d_optimal"


def test_the_charts_follow_the_level_and_the_findings_stay(page):
    reveal(page, "#analysis")
    assert len(more_charts(page)) == 8
    set_level(page, 1)
    assert more_charts(page) == ["main_effects", "interaction", "cube", "normal_probability", "residuals_order"]
    assert page.text_content("#more-charts .level-more") == "2 more charts at Intermediate, 1 at Adventurous. Show them"
    assert page.locator("#findings li").count() > 0, "what the results say is said at every level"
    assert page.locator("#diagnostics").is_hidden(), "leverage and Cook's distance come at Intermediate"
    set_level(page, 2)
    assert page.locator("#diagnostics").is_visible()
    assert more_charts(page)[:2] == ["perturbation", "main_effects"]


def test_a_run_to_look_at_again_is_listed_at_every_level(blank_page):
    """A warning is never left off the page: a run the model flags shows the runs' table at Beginner too."""
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=4)
    practise(page)
    settled(page)
    sent = calls(page, "api_analyze_runs")[-1]["payload"]
    center = next(i for i, row in enumerate(sent["rows"]) if row[sent["columns"].index("T")] == 175)
    enter_results(page, {center: sent["responses"][center] + 60})  # a center run, which no term of the model fits
    settled(page)
    set_level(page, 1)
    reveal(page, "#analysis")
    assert page.locator("#diagnostics").is_visible()
    assert "to look at again" in page.text_content("#diagnostics summary")
