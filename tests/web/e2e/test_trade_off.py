"""The trade-off grid of a fraction: runs down, factors across, each cell the resolution of the best fraction.

With a fractional factorial, the grid sits under the design's settings. The column of the brief's
factors takes a pick, which sets the resolution or switches to the full factorial; the other columns
are muted, and a factor added or removed in the brief moves the column across. The cell of the
design the form now makes is pressed, whichever control changed it.
"""

from __future__ import annotations

import pytest

from tests.web._support import factors_named
from tests.web.e2e._page import (
    WIDE_FONT,
    calls,
    candidates_done,
    choose,
    generate,
    last_reply,
    overflowing,
    set_factors,
)

NAMES = "ABCDEFGHJKL"


def grid(page) -> dict:
    """Read the grid as the reader sees it: the column of the brief's factors, each row's cells, and its buttons."""
    return page.evaluate(
        """() => {
             const table = document.querySelector('#trade-off-table');
             return {
               here: [...table.querySelectorAll('thead th.here')].map((th) => Number(th.textContent)),
               rows: Object.fromEntries([...table.querySelectorAll('tbody tr')].map((tr) =>
                 [tr.cells[0].textContent, [...tr.cells].slice(1).map((td) => td.textContent)])),
               buttons: [...table.querySelectorAll('button')].map((b) => [Number(b.dataset.runs), b.textContent]),
               pressed: [...table.querySelectorAll('button[aria-pressed="true"]')].map((b) => Number(b.dataset.runs)),
               away: table.querySelectorAll('td.away button').length,
             };
           }"""
    )


def fraction_of(page, k: int) -> None:
    """Type k factors into the brief, wait for the designs that can learn it, and choose a fractional factorial."""
    set_factors(page, factors_named(*NAMES[:k]))
    candidates_done(page)
    choose(page, design_type="fractional_factorial")


def cell(page, name: str):
    return page.get_by_role("button", name=name)


def test_a_fraction_shows_the_grid_with_the_column_of_the_brief_s_factors(fresh_page):
    page = fresh_page
    fraction_of(page, 5)
    assert page.is_visible("#trade-off")
    shown = grid(page)
    assert shown["here"] == [5]
    assert shown["rows"]["16"] == ["", "Full", "V", "IV", "IV", "IV", "III", "III", "III"]
    assert shown["buttons"] == [[8, "III"], [16, "V"], [32, "Full"]], "only the column of the brief's factors"
    assert shown["away"] == 0
    assert shown["pressed"] == [16], "resolution IV in five factors is made in the 16 runs of resolution V"
    assert page.text_content("#trade-off-hint") == (
        "The column of your 5 factors takes a pick. Add or remove a factor in the brief to move across."
    )
    choose(page, design_type="box_behnken")
    assert page.is_hidden("#trade-off"), "only a design that takes a resolution shows it"


def test_the_16_run_cell_in_five_factors_makes_a_16_run_resolution_v_design(fresh_page):
    page = fresh_page
    fraction_of(page, 5)
    choose(page, resolution=3)
    assert grid(page)["pressed"] == [8]
    cell(page, "16 runs, 5 factors: resolution V").click()
    assert page.input_value("#resolution") == "5"
    assert grid(page)["pressed"] == [16]
    generate(page)
    request = calls(page, "api_make_design")[-1]["payload"]
    assert (request["design_type"], request["resolution"]) == ("fractional_factorial", 5)
    made = last_reply(page, "api_make_design")["result"]
    assert made["n_runs"] - made["facts"]["center_runs"] == 16
    assert made["facts"]["resolution"] == 5


def test_a_full_cell_switches_to_the_full_factorial(fresh_page):
    page = fresh_page
    fraction_of(page, 5)
    cell(page, "32 runs, 5 factors: the full factorial").click()
    assert page.input_value("#design-type") == "full_factorial"
    assert page.is_hidden("#trade-off"), "the grid goes with the fraction"
    assert page.evaluate("document.activeElement.id") == "design-type", "the focus moves to the design it chose"
    generate(page)
    made = last_reply(page, "api_make_design")["result"]
    assert made["spec"]["design_type"] == "full_factorial"
    assert made["n_runs"] - made["facts"]["center_runs"] == 32


def test_the_grid_follows_the_factors_and_the_resolution(fresh_page):
    page = fresh_page
    fraction_of(page, 7)
    shown = grid(page)
    assert shown["here"] == [7]
    assert [runs for runs, _ in shown["buttons"]] == [8, 16, 64, 128], "the 32-run IV is made in 16 runs"
    assert shown["rows"]["32"][4] == "IV"
    title = page.get_attribute("#trade-off-table tbody tr:nth-child(4) td:nth-child(6)", "title")
    assert title == "Not offered: the page makes resolution IV in 16 runs."
    assert shown["pressed"] == [16]
    choose(page, resolution=5)
    assert grid(page)["pressed"] == [64], "resolution V in seven factors is the 64-run VII"
    set_factors(page, factors_named(*NAMES[:9]))
    assert grid(page)["here"] == [9]
    assert grid(page)["pressed"] == [128], "the reader's resolution is kept as the factors change"
    set_factors(page, factors_named(*NAMES[:4]))
    assert page.input_value("#resolution") == "4", "until four factors rule it out"
    assert grid(page)["here"] == [4]
    assert grid(page)["pressed"] == [8]


def test_a_cell_is_picked_from_the_keyboard(fresh_page):
    page = fresh_page
    fraction_of(page, 6)
    button = cell(page, "32 runs, 6 factors: resolution VI")
    button.focus()
    page.keyboard.press("Enter")
    assert page.input_value("#resolution") == "5"
    assert grid(page)["pressed"] == [32]
    assert page.evaluate("document.activeElement.dataset.runs") == "32", "the focus stays on the cell"
    page.keyboard.press("Shift+Tab")
    page.keyboard.press("Space")
    assert page.input_value("#resolution") == "4"
    assert grid(page)["pressed"] == [16]


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_grid_scrolls_in_its_own_box(fresh_page):
    page = fresh_page
    page.add_style_tag(content=WIDE_FONT)
    fraction_of(page, 11)
    assert page.is_visible("#trade-off")
    assert overflowing(page) == []
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


#: The contrast WCAG AA asks of text: 4.5 to 1.
AA = 4.5

#: Each grid cell's text and the colour behind it, the first ancestor with a background of its own.
COLOURS = """() => {
  const behind = (e) => {
    for (; e; e = e.parentElement) {
      const colour = getComputedStyle(e).backgroundColor;
      if (colour !== 'rgba(0, 0, 0, 0)' && colour !== 'transparent') return colour;
    }
    return getComputedStyle(document.body).backgroundColor;
  };
  return [...document.querySelectorAll('#trade-off-table th, #trade-off-table td, #trade-off-table button')]
    .filter((e) => e.textContent.trim() && !e.querySelector('button'))
    .map((e) => [e.textContent, getComputedStyle(e).color, behind(e)]);
}"""


def luminance(colour: str) -> float:
    """Return the relative luminance of a computed ``rgb(r, g, b)`` colour, as WCAG reckons it."""
    channels = [int(v) / 255 for v in colour.removeprefix("rgb(").removesuffix(")").split(", ")[:3]]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(text: str, background: str) -> float:
    high, low = sorted([luminance(text), luminance(background)], reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize(
    "context_options",
    [
        {"viewport": {"width": 1280, "height": 900}, "color_scheme": "light"},
        {"viewport": {"width": 1280, "height": 900}, "color_scheme": "dark"},
    ],
)
def test_every_cell_s_text_meets_aa_in_both_themes(fresh_page):
    page = fresh_page
    fraction_of(page, 7)
    colours = page.evaluate(COLOURS)
    assert len(colours) > 40, "the headings and every cell that holds a design"
    weak = [(text, round(contrast(fg, bg), 2)) for text, fg, bg in colours if contrast(fg, bg) < AA]
    assert weak == []
