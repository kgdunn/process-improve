"""A study's settings, in the browser: the colour map its charts are drawn in, kept with the study."""

from __future__ import annotations

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
    names = page.locator("#colour-maps label").all_text_contents()
    assert names == ["Batlow: blue to gold to pink", "Devon: blue", "La Jolla: brown to yellow", "Grey"]
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
