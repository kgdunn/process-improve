"""The reading level in the page: Beginner, Intermediate or Adventurous, each adding to the ones below.

The level is the concept map's own, kept under its key, so a choice made on either page holds on both. A level
leaves designs and charts off the page and says how many; it never leaves out a result, a warning, or what the
study on the page was made with. The other browser tests run at the highest level (``tests/web/harness.js``).
"""

from __future__ import annotations

import re

from tests.web.e2e._page import calls, enter_results, generate, practise, reveal, settled
from tests.web.e2e.test_responses import add_response


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
    assert len(listed_designs(page)) == 16, "every design for factors with ranges, and the three mixtures"


def test_a_design_above_the_level_is_not_in_the_list_at_all(blank_page):
    """Left out, not hidden: a browser that shows a hidden option (Safari, a phone's own picker) has none to show."""
    page = blank_page
    set_level(page, 2)
    in_the_list = page.locator("#design-type option").evaluate_all("(options) => options.map((o) => o.value)")
    assert "d_optimal" not in in_the_list
    assert "given" not in in_the_list, "no study of given runs is open"
    assert "simplex_lattice" not in in_the_list
    assert page.locator("#design-type optgroup").count() == 0, "a group with no design of this level is left out too"
    set_level(page, 3)
    assert page.locator("#design-type optgroup option").count() == 3, "the mixtures, back under their heading"
    order = page.locator("#design-type option").evaluate_all("(options) => options.map((o) => o.value)")
    assert order.index("box_behnken") < order.index("d_optimal") < order.index("simplex_lattice"), "in their order"


#: The parts of the page a level brings, each with the level that first shows it.
BLOCKS, LIMITS, MORE_RESPONSES, COMPARE, CURVES = (
    '[data-feature="option:blocks"]',
    "#limits",
    "#more-responses",
    "#compare",
    "#curves",
)


def on_the_page(page, *parts: str) -> list[bool]:
    """Say of each part whether the reader sees it, in the phase that holds it."""
    seen = []
    for part in parts:
        reveal(page, part)
        seen.append(page.locator(part).is_visible())
    return seen


def test_a_beginner_s_page_leaves_out_what_a_first_study_does_not_need(blank_page):
    """Blocks, limits on the settings, a second response, the comparison in detail and curves come at Intermediate.

    A first visit opens at Intermediate, so it has them all, as it had before the page had levels.
    """
    page = blank_page
    generate(page, design_type="full_factorial")
    set_level(page, 1)
    assert on_the_page(page, BLOCKS, LIMITS, MORE_RESPONSES, COMPARE, CURVES) == [False] * 5
    set_level(page, 2)
    assert on_the_page(page, BLOCKS, LIMITS, MORE_RESPONSES, COMPARE, CURVES) == [True] * 5


def test_a_part_the_study_uses_stays_on_the_page_at_every_level(blank_page):
    """A study made in blocks, with a second response, keeps both in view for a beginner: nothing in use is hidden."""
    page = blank_page
    add_response(page, "Cost", "$/kg", "minimize")
    generate(page, design_type="full_factorial", n_blocks=2)
    set_level(page, 1)
    assert on_the_page(page, BLOCKS, MORE_RESPONSES) == [True, True]
    assert on_the_page(page, LIMITS, COMPARE) == [False, False], "the study has no limits, and compares nothing"
    reveal(page, "#n_blocks")
    page.fill("#n_blocks", "1")
    assert on_the_page(page, BLOCKS) == [False], "no longer in use, so the level decides"


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


def test_two_changes_of_level_in_a_row_draw_the_charts_once(page):
    """An arrow key on the levels passes through the one between: the charts drawn for it stop, and one set stays."""
    reveal(page, "#analysis")
    more_charts(page)
    page.evaluate(
        """() => {
          for (const level of ["1", "3"]) {
            const radio = document.querySelector(`#level input[value="${level}"]`);
            radio.checked = true;
            radio.dispatchEvent(new Event("change", { bubbles: true }));
          }
        }"""
    )
    page.wait_for_function("document.querySelectorAll('#more-charts .js-plotly-plot').length === 8")
    page.wait_for_timeout(500)  # a second drawing, were there one, would have added its cards by now
    assert page.locator("#more-charts").count() == 1
    assert page.locator("#charts > figure h3").all_text_contents().count("Fitted response surface") == 1
    assert len(more_charts(page)) == 8
