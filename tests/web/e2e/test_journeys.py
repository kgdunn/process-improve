"""Whole visits, as readers make them: each test is one journey through the page, on a desktop and a phone.

The other tests here check one control each. A journey checks that the steps still join up: what one
step leaves behind is what the next one needs. It asserts only what a reader would notice, so it
fails on a broken journey and not on a reworded sentence.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import (
    candidates_done,
    first_paint_done,
    name_response,
    practise,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
)

DEVICES = [{"viewport": {"width": 1280, "height": 900}}, {"viewport": {"width": 390, "height": 844}, "is_mobile": True}]
BANNER = "#campaign-banner"
#: Two factors of a reactor, as a reader types them.
REACTOR = [
    {"name": "Temperature", "low": 80, "high": 120, "units": "degC"},
    {"name": "Time", "low": 30, "high": 90, "units": "min"},
]


def _conversion(temperature: float, time: float) -> float:
    """Return what the reader measures at each run: the reactor's conversion, its peak inside the region."""
    x, z = (temperature - 100) / 20, (time - 60) / 30
    return round(80 + 3 * x + 2 * z - 4 * x * x - 3 * z * z - 1.5 * x * z, 2)


def _best(page) -> dict[str, str]:
    """Read the best settings Learn proposes: each factor and its setting."""
    reveal(page, "#best")
    rows = page.locator("#best-out table.data:not(.known-table)").first.locator("tbody tr").all()
    return {row.locator("td").nth(0).text_content(): row.locator("td").nth(1).text_content() for row in rows}


def _opened_elsewhere(open_context, site: str, link: str):
    """Open a link in another browser, as the colleague it was sent to would."""
    page = open_context().new_page()
    page.goto(link if link.startswith("http") else site + link)
    page.wait_for_selector(BANNER, state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    return page


@pytest.mark.parametrize("context_options", DEVICES, ids=["desktop", "phone"])
def test_a_study_by_hand_from_the_brief_to_the_best_settings_shared_and_opened_again(
    context, fresh_page, open_context, site
):
    page = fresh_page
    # The brief: what is measured and what is wanted, then the factors.
    name_response(page, "Conversion", "%", goal="maximize")
    set_factors(page, REACTOR)
    candidates_done(page)
    assert page.locator('#candidates tbody tr[data-design="ccd"]').count(), "a design that can find a peak is offered"
    reveal(page, "#candidates")
    page.click('#candidates tbody tr[data-design="ccd"]')
    reveal(page, "#generate")
    page.click("#generate")
    settled(page)

    # The run sheet: every run carried out and its result typed in, in the order the sheet gives.
    assert page.get_attribute("#workspace", "data-phase") == "run"
    sheet = run_sheet(page)
    names = page.locator("#design-table thead th button.sort").all_text_contents()
    at = names.index("Temperature"), names.index("Time")
    for i, row in enumerate(sheet):
        page.fill(f'#design-table input[data-index="{i}"]', f"{_conversion(float(row[at[0]]), float(row[at[1]])):g}")
    settled(page)

    # Learn: what the results say, and where to run the reactor.
    page.click("#go-learn")
    settled(page)
    assert page.locator("#findings li").first.text_content().startswith("Temperature matters most")
    best = _best(page)
    assert best.keys() == {"Temperature", "Time"}
    temperature, time = (float(best[name].split()[0]) for name in ("Temperature", "Time"))
    assert 100 < temperature < 110, "the peak lies inside the region, near where it is"
    assert 60 < time < 70
    kept = run_sheet(page)

    # Shared: the colleague who opens the link sees the same study and the same best settings.
    reveal(page, "#share")
    page.click("#share")
    page.wait_for_selector("#share-out", state="visible")
    elsewhere = _opened_elsewhere(open_context, site, page.input_value("#share-url"))
    assert run_sheet(elsewhere) == kept
    elsewhere.click("#go-learn")
    settled(elsewhere)
    assert _best(elsewhere) == best

    # Kept: the reader who comes back finds the study where they left it.
    saved_studies(page, lambda saved: len(saved) == 1 and all(r is not None for r in saved[0]["responses"]))
    again = context.new_page()
    again.goto(site)
    again.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(again)
    assert run_sheet(again) == kept


@pytest.mark.parametrize("context_options", DEVICES, ids=["desktop", "phone"])
def test_practise_on_the_worked_example_then_a_study_of_ones_own_kept_as_a_file(page, open_context, site, tmp_path):
    # The worked example: its analysis is there on the first visit, ready to read.
    first_paint_done(page)
    page.click("#go-learn")
    assert page.locator("#findings li").count() > 0
    assert _best(page)

    # A study of one's own, practised on before the real runs.
    reveal(page, "#new-study")
    page.click("#new-study")
    set_factors(page, [{"name": "Speed", "low": 100, "high": 300, "units": "rpm"}, *REACTOR])
    candidates_done(page)
    reveal(page, "#candidates")
    page.locator("#candidates tbody tr").first.click()  # the design the page puts first
    reveal(page, "#generate")
    page.click("#generate")
    settled(page)
    practise(page)
    page.click("#go-learn")
    settled(page)
    assert page.locator("#findings li").count() > 0, "practice results are analysed like real ones"
    kept = run_sheet(page)

    # Kept as a file, and opened on another computer.
    saved = saved_studies(page, lambda saved: len(saved) == 1)
    assert len(saved) == 1, "the worked example is not kept, the reader's own study is"
    page.click("#library summary")
    with page.expect_download() as download:
        page.locator("#library-list li").first.get_by_role("button", name="Download").click()
    file = tmp_path / download.value.suggested_filename
    download.value.save_as(file)
    other = open_context().new_page()
    other.goto(site)
    other.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    other.set_input_files("#study-upload", str(file))
    other.wait_for_selector(BANNER, state="visible")
    settled(other)
    assert run_sheet(other) == kept
