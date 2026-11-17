"""The app's look: points on an axis, no rounded boxes, and the reader's own numbers in ink.

The page is drawn with the marks an experiment already uses. A choice is a point: a ring until it is chosen, then
filled in the stage's colour, with no box round it and no coloured edge. The four aims are levels on one axis, and
the stages in the bar are points on one line. A check under Learn's answer is a point whose shape says what it
found. Nothing is rounded but a point. What the reader typed or measured is in ink; what the page worked out is in
print, and a simulated result is pencil.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web.e2e._page import enter_results, generate, open_statistician, practise, reveal

STYLE = "(one) => getComputedStyle(one).{}"
THEMES = pytest.mark.parametrize("context_options", [{"color_scheme": "light"}, {"color_scheme": "dark"}])

#: The colour a token of the stylesheet stands for, as the browser paints it.
COLOUR_OF = """(name) => {
  const probe = document.body.appendChild(document.createElement('i'));
  probe.style.color = `var(${name})`;
  const colour = getComputedStyle(probe).color;
  probe.remove();
  return colour;
}"""


def stage_colour(page: Any, stage: str) -> str:  # noqa: ANN401 - a Playwright Page
    """Return the colour the bar gives a stage, as its point wears it."""
    return page.locator(f"#go-{stage} .step").evaluate(STYLE.format("backgroundColor"))


@THEMES
def test_a_choice_is_a_point_filled_in_its_stage_s_colour_with_no_box_and_no_edge(page):
    page.click("#go-plan")
    design = stage_colour(page, "plan")
    reveal(page, "#aim-card")
    chosen = page.locator(".aim input:checked")
    assert chosen.evaluate(STYLE.format("borderRadius")) == "50%"
    assert chosen.evaluate(STYLE.format("backgroundColor")) == design
    others = page.locator(".aim input:not(:checked)")
    assert set(others.evaluate_all("(all) => all.map((one) => getComputedStyle(one).backgroundColor)")) != {design}
    boxes = page.locator(".aim").evaluate_all(
        "(all) => all.map((one) => [getComputedStyle(one).borderLeftWidth, getComputedStyle(one).boxShadow])"
    )
    assert boxes == [["0px", "none"]] * 4, "the chosen aim has no box round it and no coloured edge"
    # The design chosen in the list is a point of the same kind.
    picked = page.locator("table.pick input:checked")
    assert picked.evaluate(STYLE.format("backgroundColor")) == design


def test_a_term_of_the_model_is_a_point_filled_when_it_is_in(page):
    open_statistician(page)
    page.wait_for_selector("#statistician #terms:not([hidden])")
    learn = stage_colour(page, "learn")
    inside = page.get_by_label("Temperature·Flow", exact=True)
    out = page.get_by_label("Temperature·Pressure·Flow", exact=True)
    assert inside.evaluate(STYLE.format("borderRadius")) == "50%"
    assert inside.evaluate(STYLE.format("backgroundColor")) == learn, "in the model: filled in Learn's colour"
    assert out.evaluate(STYLE.format("backgroundColor")) != learn, "out of it: a ring"
    label = page.locator("#term-list label").first
    assert label.evaluate(STYLE.format("borderTopWidth")) == "0px", "no box round a term"


def test_the_aims_are_levels_on_one_axis(page):
    page.click("#go-plan")
    reveal(page, "#aim-card")
    line = page.locator(".aim").evaluate_all(
        """(aims) => aims.map((aim) => {
          const axis = getComputedStyle(aim, '::before');
          const point = aim.querySelector('input').getBoundingClientRect();
          const box = aim.getBoundingClientRect();
          const middle = point.left + point.width / 2;
          return { width: axis.width, through: Math.abs(box.left + parseFloat(axis.left) + 1 - middle) };
        })"""
    )
    assert all(part["width"] == "2px" for part in line), "one line through the four points"
    assert all(part["through"] <= 1 for part in line), "and through the middle of each"


def test_the_stages_are_points_on_one_line_and_the_stage_on_show_is_the_larger(page):
    sizes = page.locator("#studybar [role=tab] .step").evaluate_all(
        "(all) => all.map((one) => [one.closest('[role=tab]').id, one.getBoundingClientRect().width])"
    )
    widths = dict(sizes)
    assert widths["go-learn"] > widths["go-plan"] == widths["go-run"] == widths["go-next"], "Learn is on show"
    assert max(widths.values()) <= 16, "a point, with no number in it"
    assert page.locator("#go-learn").evaluate(STYLE.format("borderBottomWidth")) == "0px", "no edge under its name"
    assert page.locator("#go-learn").evaluate(STYLE.format("fontWeight")) in ("600", "700"), "its name is in bold"
    lines = page.locator("#studybar [role=tab]").evaluate_all(
        "(tabs) => tabs.map((tab) => getComputedStyle(tab, '::before').content !== 'none')"
    )
    assert lines == [False, *[True] * (len(lines) - 1)], "the line runs up to each stage after the first"


@THEMES
def test_a_check_is_a_point_whose_shape_says_what_it_found(page):
    reveal(page, "#checks")
    found = page.locator("#checks .learn-check").evaluate_all(
        """(checks) => checks.map((check) => {
          const point = getComputedStyle(check.querySelector('summary'), '::before');
          return {
            state: check.dataset.state,
            colour: point.backgroundColor,
            round: point.borderRadius === '50%',
            turned: point.transform !== 'none',
            boxed: getComputedStyle(check).borderTopWidth !== '0px',
          };
        })"""
    )
    by_state = {check["state"]: check for check in found}
    assert {"ok", "look"} <= set(by_state), "the worked example has a check that passed and one to look at"
    assert not any(check["boxed"] for check in found), "no box round a check"
    assert by_state["ok"]["colour"] == page.evaluate(COLOUR_OF, "--ok")
    assert by_state["look"]["colour"] == page.evaluate(COLOUR_OF, "--warn")
    assert by_state["ok"]["round"]
    assert not by_state["ok"]["turned"]
    assert by_state["look"]["turned"], "a diamond: the shape says it without the colour"


def test_the_answer_and_a_note_have_no_box_and_no_coloured_edge(page):
    answer = page.locator("#answer")
    assert answer.evaluate(STYLE.format("borderLeftWidth")) == "0px"
    assert answer.evaluate(STYLE.format("borderTopWidth")) == "0px"
    reveal(page, "#best")
    note = page.locator("#best .warnings li").first
    assert page.locator("#best .warnings").evaluate(STYLE.format("borderLeftWidth")) == "0px"
    mark = note.evaluate("(one) => getComputedStyle(one, '::before').transform")
    assert mark != "none", "a note is marked by the diamond of a check that found something"


def test_nothing_is_rounded_but_a_point(page):
    corner = page.evaluate("() => getComputedStyle(document.documentElement).getPropertyValue('--corner').trim()")
    assert corner == "2px"
    for control in ("#step-analyse", "#model", "#learn-report-view"):
        assert page.locator(control).evaluate(STYLE.format("borderTopLeftRadius")) == corner, control


@THEMES
def test_what_the_reader_typed_is_in_ink_and_a_simulated_result_is_pencil(blank_page):
    page = blank_page
    ink = page.evaluate(COLOUR_OF, "--ink")
    print_colour = page.evaluate(COLOUR_OF, "--fg")
    assert ink != print_colour
    page.fill("#response", "Yield")
    assert page.locator("#response").evaluate(STYLE.format("color")) == ink
    assert page.locator("#h-response").evaluate(STYLE.format("color")) == print_colour, "the page's own words are print"
    generate(page)
    practise(page)
    reveal(page, "table.runs")
    simulated = page.locator("table.runs input.sim").first
    assert simulated.evaluate(STYLE.format("color")) == page.evaluate(COLOUR_OF, "--muted"), "pencil, until typed over"
    assert simulated.evaluate(STYLE.format("fontStyle")) == "italic"
    enter_results(page, {0: 61.5})
    typed = page.locator("table.runs tbody tr").first.locator("input")
    assert typed.evaluate(STYLE.format("color")) == ink, "a result the reader typed is theirs"


def test_a_button_that_leads_on_wears_the_colour_of_the_stage_it_leads_to(page):
    colours = {stage: stage_colour(page, stage) for stage in ("plan", "run", "learn")}
    assert len(set(colours.values())) == 3, "a colour a stage"
    background = STYLE.format("backgroundColor")
    page.click("#go-plan")
    assert page.locator("#generate").evaluate(background) == colours["run"], "Make the run sheet leads to Run"
    page.click("#go-run")
    assert page.locator("#to-learn").evaluate(background) == colours["learn"], "and what the results say is in Learn"
    assert page.locator("#simulate").evaluate(background) == colours["run"], "a button that stays wears its stage's"
    chosen = page.locator('#step-run .tabs [aria-selected="true"]')
    assert chosen.evaluate(STYLE.format("borderBottomColor")) == colours["run"], "as does the tab chosen inside it"


@pytest.mark.parametrize("context_options", [{"forced_colors": "active"}])
def test_a_chosen_point_is_drawn_in_forced_colours(page):
    page.click("#go-plan")
    reveal(page, "#aim-card")
    chosen = page.locator(".aim input:checked").evaluate(STYLE.format("backgroundColor"))
    other = page.locator(".aim input:not(:checked)").first.evaluate(STYLE.format("backgroundColor"))
    assert chosen != other, "the chosen aim is still told from the others"


@THEMES
def test_a_slider_is_drawn_by_the_page_a_thin_line_and_a_round_handle_in_the_accent_colour(page):
    """The browser's own slider was two thick pale bars on the dark page, with no handle to be seen."""
    reveal(page, "#what-if")
    slider = page.locator('#what-if input[type="range"]').first
    drawn = slider.evaluate(
        """(one) => { const own = getComputedStyle(one), track = getComputedStyle(one, '::-webkit-slider-runnable-track'),
          handle = getComputedStyle(one, '::-webkit-slider-thumb');
          return { appearance: own.appearance, edge: own.borderTopWidth, padding: own.paddingLeft,
                   track: track.height, handle: handle.backgroundColor, round: handle.borderRadius,
                   size: handle.width }; }"""
    )
    assert drawn["appearance"] == "none", "the page draws it, the same in every browser"
    assert (drawn["edge"], drawn["padding"]) == ("0px", "0px"), "no box, edge or padding of an input round it"
    assert drawn["track"] == "4px", "a thin line for its range"
    assert drawn["handle"] == page.evaluate(COLOUR_OF, "--accent")
    assert (drawn["round"], drawn["size"]) == ("50%", "18px"), "a round handle, large enough to catch"
