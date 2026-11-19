"""A study's settings, in the browser: the colour map its charts are drawn in, kept with the study."""

from __future__ import annotations

import re
from typing import Any

from tests.web.e2e._page import generate, practise, saved_studies, settled

#: The colour a contour chart gives its lowest value, on a light page, in each map: the map's light end.
LOW = {"batlow": "#faccfa", "devon": "#eae8fb", "lajolla": "#fffecb", "grayC": "#dcdcdc"}
CONTOUR = "document.querySelector('figure[data-chart=contour] .plot')?.data?.find((t) => Array.isArray(t.colorscale))"


def lowest(page: Any) -> str:  # noqa: ANN401 - a Playwright Page
    return page.evaluate(f"{CONTOUR}?.colorscale[0][1]").lower()


def choose_map(page: Any, name: str, *, keep: bool | None = None) -> None:  # noqa: ANN401 - a Playwright Page
    """Choose a colour map in Settings, as a reader does from the Studies menu, and close the settings."""
    page.click("#study-button")
    page.click("#menu-settings")
    page.wait_for_selector("#settings-dialog[open]")
    page.check(f'#colour-maps input[value="{name}"]')
    if keep is not None:
        page.set_checked("#settings-keep", keep)
    page.click("#settings-close")
    page.wait_for_function(f"{CONTOUR}?.colorscale[0][1]?.toLowerCase() === '{LOW[name]}'")
    settled(page)


def test_the_charts_are_drawn_in_batlow_until_another_map_is_chosen(page):
    page.click("#go-learn")
    assert lowest(page) == LOW["batlow"]
    page.click("#study-button")
    page.click("#menu-settings")
    names = page.locator("#colour-maps .settings__choice").all_text_contents()
    assert names == ["Batlow", "Devon", "La Jolla", "Grey"]
    assert page.is_checked('#colour-maps input[value="batlow"]')
    assert not page.is_checked("#settings-keep")
    for name in ("devon", "lajolla", "grayC"):
        page.check(f'#colour-maps input[value="{name}"]')
        page.wait_for_function(f"{CONTOUR}?.colorscale[0][1]?.toLowerCase() === '{LOW[name]}'")


def test_a_study_keeps_its_colour_map_and_a_new_study_does_not_take_it_unasked(blank_page):
    page = blank_page
    generate(page)
    practise(page)
    page.click("#go-learn")
    choose_map(page, "lajolla")
    kept = saved_studies(
        page, lambda studies: bool(studies) and studies[0].get("settings") == {"colour_map": "lajolla"}
    )
    assert kept[0]["v"] == 12, "a study with settings is saved in format 12"
    page.reload()
    page.wait_for_function(f"{CONTOUR}?.colorscale[0][1]?.toLowerCase() === '{LOW['lajolla']}'")
    page.click("#study-button")
    page.click("#new-study")
    page.click("#study-button")
    page.click("#menu-settings")
    assert page.is_checked('#colour-maps input[value="batlow"]'), "a new study starts from the default"


def test_settings_kept_for_new_studies_are_a_new_study_s(page):
    page.click("#go-learn")
    choose_map(page, "devon", keep=True)
    page.click("#study-button")
    page.click("#new-study")
    page.click("#study-button")
    page.click("#menu-settings")
    assert page.is_checked('#colour-maps input[value="devon"]')
    assert page.is_checked("#settings-keep")


def test_the_switch_on_the_chosen_map_s_line_flips_it_and_the_study_keeps_that(blank_page):
    page = blank_page
    generate(page)
    practise(page)
    page.click("#go-learn")
    page.click("#study-button")
    page.click("#menu-settings")
    line = page.locator("#colour-maps .settings__map", has=page.locator("input:checked"))
    assert line.locator(".settings__end").all_text_contents() == ["Low", "High"]
    assert line.locator(".settings__flip").text_content() == "Flip"
    assert page.locator("#colour-maps label.settings__flip").count() == 1, "on the chosen map's line only"
    before = line.locator(".settings__swatch").evaluate("(bar) => bar.style.background")
    page.locator("#colour-map-flipped").check(force=True)  # the box is under its drawn switch
    page.wait_for_function(f"{CONTOUR}?.colorscale[0][1]?.toLowerCase() === '#011959'")
    after = page.locator("#colour-maps .settings__swatch").first.evaluate("(bar) => bar.style.background")
    assert after != before, "the bar shows the colours from low to high as the charts now draw them"
    assert page.evaluate("document.activeElement.id") == "colour-map-flipped", "the keyboard stays on the switch"
    kept = saved_studies(
        page, lambda studies: bool(studies) and studies[0].get("settings", {}).get("colour_map_flipped")
    )
    assert kept[0]["settings"] == {"colour_map": "batlow", "colour_map_flipped": True}
    page.locator("#colour-map-flipped").uncheck(force=True)
    page.wait_for_function(f"{CONTOUR}?.colorscale[0][1]?.toLowerCase() === '{LOW['batlow']}'")
    kept = saved_studies(
        page, lambda studies: bool(studies) and "colour_map_flipped" not in studies[0].get("settings", {})
    )
    assert kept[0]["settings"] == {"colour_map": "batlow"}


def test_a_comma_as_the_decimal_mark_is_in_the_numbers_the_page_writes_and_the_study_keeps_it(blank_page):
    page = blank_page
    generate(page)
    practise(page)
    page.click("#go-learn")
    page.click("#study-button")
    page.click("#menu-settings")
    assert page.input_value("#decimal-mark") == "period", "a period, whatever the browser's language"
    assert page.locator("#decimal-mark option").all_text_contents() == ["Period: 1.5", "Comma: 1,5"]
    page.select_option("#decimal-mark", "comma")
    page.click("#settings-close")
    page.wait_for_function("/R² = 0,\\d/.test(document.querySelector('#analysis').textContent)")
    said = page.locator("#analysis").text_content()
    equation = page.locator("#analysis p.equation").first.text_content()
    assert re.search(r"= \d+,\d+ [+-] \d+,\d+·", equation), equation
    assert not re.search(r"\d\.\d", equation), "no number of the equation keeps a period"
    assert not re.search(r"R² = 0\.\d", said)
    ticks = page.evaluate(f"{CONTOUR} && document.querySelector('figure[data-chart=contour] .plot').layout.separators")
    assert ticks == ", ", "the charts' own numbers follow: a comma, and a space between thousands"
    with_comma = {"colour_map": "batlow", "decimal_mark": "comma"}  # the settings as the dialog left them
    kept = saved_studies(page, lambda studies: bool(studies) and studies[0].get("settings") == with_comma)
    assert kept[0]["settings"] == with_comma
    assert kept[0]["v"] == 12
    assert 'model="' in page.locator("#py-analyse").text_content(), "the Python code is as it was"
    assert not re.search(r"\d,\d", page.locator("#py-analyse code").first.text_content())

    page.click("#study-button")
    page.click("#menu-settings")
    page.select_option("#decimal-mark", "period")
    page.click("#settings-close")
    page.wait_for_function("/R² = 0\\.\\d/.test(document.querySelector('#analysis').textContent)")
    kept = saved_studies(page, lambda studies: bool(studies) and "decimal_mark" not in studies[0]["settings"])
    assert kept[0]["settings"] == {"colour_map": "batlow"}, "a study that keeps the period holds no setting for it"


def test_a_stricter_significance_level_widens_the_intervals_and_is_named_across_the_page(blank_page):
    page = blank_page
    generate(page)
    practise(page)
    page.click("#go-learn")
    page.wait_for_function("document.querySelector('#analysis').textContent.includes('95% low')")
    assert "significance_level" not in page.locator("#py-analyse code").first.text_content()
    # The width of the first interval the coefficients' table shows: its "% high" cell less its "% low" cell.
    width = """(() => {
      const table = [...document.querySelectorAll('#analysis table')].find((t) => /% low/.test(t.tHead?.textContent ?? ''));
      const heads = [...table.tHead.rows[0].cells].map((cell) => cell.textContent);
      const cell = (row, end) => row.cells[heads.findIndex((head) => head.endsWith(end))].textContent;
      const number = (text) => Number(text.replace('\\u2212', '-'));
      const ends = [...table.tBodies[0].rows].map((row) => ['% low', '% high'].map((end) => number(cell(row, end))));
      const [low, high] = ends.find((pair) => pair.every(Number.isFinite));
      return high - low;
    })()"""
    usual = page.evaluate(width)

    page.click("#study-button")
    page.click("#menu-settings")
    assert page.input_value("#significance-level") == "0.05"
    offered = ["0.01, with 99% intervals", "0.05, with 95% intervals", "0.10, with 90% intervals"]
    assert page.locator("#significance-level option").all_text_contents() == offered
    page.select_option("#significance-level", "0.01")
    page.click("#settings-close")
    page.wait_for_function("document.querySelector('#analysis').textContent.includes('99% low')")
    assert "95%" not in page.locator("#analysis").text_content(), "no interval of the analysis is still named at 95%"
    assert page.evaluate(width) > usual, "Python worked the intervals out again, at 99%"
    assert "significance_level=0.01" in page.locator("#py-analyse code").first.text_content()
    strict = {"colour_map": "batlow", "significance_level": 0.01}  # the settings as the dialog left them
    kept = saved_studies(page, lambda studies: bool(studies) and studies[0].get("settings") == strict)
    assert kept[0]["v"] == 13, "a format of its own: a page from before the setting refuses the study"

    page.click("#study-button")
    page.click("#menu-settings")
    page.select_option("#significance-level", "0.05")
    page.click("#settings-close")
    page.wait_for_function("document.querySelector('#analysis').textContent.includes('95% low')")
    assert page.evaluate(width) == usual
    kept = saved_studies(page, lambda studies: bool(studies) and "significance_level" not in studies[0]["settings"])
    assert kept[0]["v"] == 12, "a study judged at 0.05 holds no setting for it"
