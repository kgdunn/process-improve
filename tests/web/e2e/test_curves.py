"""A curve each run gives: pasted in Run, predicted in Learn at the what-if settings, matched to a target.

The reader pastes the runs' curves under the run sheet, as a spreadsheet copies them. Learn fits them
once (``api_functional``), and the what-if panel then draws the curve at its settings, with its 95%
bands, in the page's own JavaScript: moving a setting asks Python nothing. A target curve pasted the
same way moves the panel to the settings whose curve comes closest to it.
"""

from __future__ import annotations

import math
from typing import Any

from tests.web.e2e._page import generate, headers, reveal, run_sheet, saved_studies, settled
from tests.web.e2e.test_what_if import set_number

TIMES = [0, 2, 4, 6, 8, 10]


def curve_at(t: float, p: float, wobble: int = 0) -> list[float]:
    """Return the curve at settings in the factors' units: T sets how high it rises, P how it drifts late."""
    level, drift = 2 + 0.8 * (t - 175) / 25, 0.3 * (p - 2)
    return [
        round(level * (1 - math.exp(-x / 3)) + drift * x / 10 + 0.01 * ((wobble + i) % 3 - 1), 4)
        for i, x in enumerate(TIMES)
    ]


def pasted(rows: list[list[float]], runs: list[int] | None = None) -> str:
    """Write curves as a spreadsheet copies them: the points first, then a row per run, under its run order if given."""
    head = ["Run order", *TIMES] if runs else TIMES
    lines = [head, *([run, *row] for run, row in zip(runs, rows, strict=True))] if runs else [head, *rows]
    return "\n".join("\t".join(str(v) for v in line) for line in lines)


def paste_curves(page: Any) -> str:  # noqa: ANN401 - a Playwright Page
    """Describe the curve and paste one for each run of the run sheet, under its run order; return what Run says."""
    reveal(page, "#curves")
    if not page.locator("#curves").get_attribute("open"):
        page.click("#curves summary")
    columns = headers(page)
    runs = [dict(zip(columns, row, strict=True)) for row in run_sheet(page)]
    curves = [curve_at(float(r["T"]), float(r["P"]), i) for i, r in enumerate(runs)]
    page.fill("#curve-label", "Colour")
    page.fill("#curve-along", "Time")
    page.fill("#curve-along-units", "min")
    page.fill("#curve-paste", pasted(curves, [int(r["Run order"]) for r in runs]))
    page.click("#curve-use")
    return page.text_content("#curve-status")


#: The predicted curve the chart draws, as Plotly holds it, and its last point.
CURVE_DRAWN = "document.querySelector('#what-if-curve-plot').data.find((t) => t.name === 'Predicted curve').y"
CURVE_END = f"{CURVE_DRAWN}.at(-1)"


def curve_drawn(page: Any) -> list[float]:  # noqa: ANN401
    """Return the predicted curve the chart draws."""
    return page.evaluate(CURVE_DRAWN)


def show_curve(page: Any) -> None:  # noqa: ANN401
    """Go to the what-if panel in Learn, and wait until the curve is drawn there."""
    settled(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if-curve-plot.js-plotly-plot")


def test_the_curves_pasted_in_run_are_drawn_in_learn_and_follow_each_setting_without_asking_python(page):
    assert paste_curves(page) == "Curves for 15 of 15 runs, each read at 6 points along Time."
    show_curve(page)
    assert page.text_content("#what-if-curve-said").startswith("From 15 runs' curves, the model follows")
    fits = page.evaluate("window.__pi.count('api_functional')")
    ends = []
    for t in (150, 200):
        before = curve_drawn(page)
        set_number(page, "T", t)
        page.wait_for_function(f"(y) => {CURVE_END} !== y", arg=before[-1])
        ends.append(curve_drawn(page)[-1])
    assert ends[1] > ends[0] + 1, "a hotter run rises higher"
    assert page.evaluate("window.__pi.count('api_functional')") == fits, "the page predicts in its own JavaScript"


def test_a_target_curve_moves_the_panel_to_the_settings_that_come_closest(page):
    paste_curves(page)
    show_curve(page)
    page.click("#curve-target summary")
    page.fill("#curve-target-values", pasted([curve_at(190, 2.5)]))
    page.click("#curve-target-find")
    page.wait_for_function(
        "document.querySelector('#curve-target-said').textContent.startsWith('The curve comes closest')"
    )
    settled(page)
    found = float(page.input_value('#what-if-settings [data-factor="T"] input[name="number"]'))
    assert abs(found - 190) < 4
    names = page.evaluate("document.querySelector('#what-if-curve-plot').data.map((t) => t.name)")
    assert "Target" in names


def test_a_study_of_curves_alone_is_saved_with_them_and_learn_draws_the_curve(blank_page):
    generate(blank_page)
    paste_curves(blank_page)
    saved = saved_studies(blank_page, lambda studies: any(s.get("curves") for s in studies))
    [study] = [s for s in saved if s.get("curves")]
    assert study["v"] == 6
    assert study["design"]["spec"]["curve"]["argument"] == {"label": "Time", "units": "min", "values": TIMES}
    show_curve(blank_page)
    assert blank_page.locator("#what-if-out table").count() == 0, "no response has results to list"
    assert blank_page.text_content("#analyse-empty").endswith("the runs' curves are drawn under What if.")
    reveal(blank_page, "#curves")
    blank_page.click("#curve-clear")
    assert blank_page.text_content("#curve-status") == ""
