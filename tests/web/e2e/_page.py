"""Driving the page in the browser tests: waiting for it, reading it, and filling in its form as a reader would.

Every helper takes a Playwright page. ``window.__pi`` is the harness's record of the page's
calls to Python (``harness.js``); a helper that starts work waits until nothing is in flight
and the status line has stopped saying it is busy. The page shows one phase at a time (Plan,
Run or Learn), so a helper that types or clicks first shows the phase that holds its element,
as a reader would, by pressing the phase's tab.
"""

from __future__ import annotations

import re
from typing import Any

#: The <select>s of the form; every other setting is an <input>.
SELECTS = {"design-type", "alpha", "resolution", "model_type", "scenario", "model"}


def reveal(page: Any, selector: str) -> None:  # noqa: ANN401 - a Playwright Page
    """Show the phase that holds ``selector``, as a reader would: by pressing its tab in the study bar."""
    pane = page.evaluate("(s) => document.querySelector(s)?.closest('[data-pane]')?.dataset.pane ?? null", selector)
    if pane and page.get_attribute("#workspace", "data-phase") != pane:
        page.click(f"#go-{pane}")


def settled(page: Any, timeout: float = 30_000) -> None:  # noqa: ANN401 - a Playwright Page
    """Wait until no call to Python is in flight and the status line is no longer busy."""
    page.wait_for_function(
        "window.__pi.idle() && document.querySelector('#status').dataset.state !== 'busy'", timeout=timeout
    )


#: The worked example's charts in Learn: the coefficients, the fitted and the true surface, and the residuals.
EXAMPLE_CHARTS = 4


def first_paint_done(page: Any) -> None:  # noqa: ANN401
    """Wait until the page has drawn the worked example, its charts last, and Python has answered."""
    page.wait_for_function(
        "window.__pi.count('api_catalogue') === 1 && window.__pi.idle()"
        f" && document.querySelectorAll('#charts .js-plotly-plot').length === {EXAMPLE_CHARTS}",
        timeout=30_000,
    )


def candidates_done(page: Any) -> None:  # noqa: ANN401
    """Wait until the designs that can learn the brief are listed, and nothing is in flight."""
    page.wait_for_function(
        "!document.querySelector('#candidates').hasAttribute('aria-busy') && window.__pi.idle()", timeout=30_000
    )


def calls(page: Any, fn: str | None = None) -> list[dict]:  # noqa: ANN401
    """Return the calls the page has made to Python, each ``{fn, payload}``, or only those to ``fn``."""
    return [c for c in page.evaluate("window.__pi.calls") if fn is None or c["fn"] == fn]


def status(page: Any) -> tuple[str, str]:  # noqa: ANN401
    """Return the status line's text and its state: ``busy``, ``ok`` or ``error``."""
    return page.text_content("#status .status__text"), page.get_attribute("#status", "data-state")


def factor_table(page: Any) -> list[dict[str, str]]:  # noqa: ANN401
    """Read the factor table as the reader sees it."""
    return page.evaluate(
        """() => [...document.querySelectorAll('#factors tbody tr')].map((tr) =>
             Object.fromEntries(
               ['name', 'low', 'high', 'units'].map((k) => [k, tr.querySelector(`[name=${k}]`).value]),
             ))"""
    )


def set_factors(page: Any, factors: list[dict]) -> None:  # noqa: ANN401
    """Make the factor table hold ``factors``, removing and adding rows and typing as a reader would."""
    reveal(page, "#factors")
    rows = page.locator("#factors tbody tr")
    while rows.count() > len(factors):
        rows.last.get_by_role("button", name="Remove factor").click()
    while rows.count() < len(factors):
        page.click("#add-factor")
    for row, factor in zip(rows.all(), factors, strict=True):
        for key in ("name", "low", "high", "units"):
            box = row.locator(f"[name={key}]")
            if box.is_enabled():  # two labels make a factor categorical, and take away its units
                box.fill(str(factor.get(key, "")))


def name_response(page: Any, label: str, units: str = "", **known: object) -> None:  # noqa: ANN401
    """Describe the response in the response card: its name, its units, and any of ``goal``, ``target``,
    ``typical``, ``noise`` and ``change``, as a reader would type them.
    """
    reveal(page, "#response")
    page.fill("#response", label)
    page.fill("#response-units", units)
    if "goal" in known:
        page.check(f'input[name="goal"][value="{known.pop("goal")}"]')
    if known and not page.evaluate("document.querySelector('#response-known').open"):
        page.click("#response-known summary")
    for key, value in known.items():
        page.fill(f"#response-{key}", str(value))


def set_aim(page: Any, aim: str) -> None:  # noqa: ANN401
    """Choose what the study is to learn, as a reader would."""
    reveal(page, "#aim-card")
    page.check(f'input[name="aim"][value="{aim}"]')


def choose(page: Any, **settings: object) -> None:  # noqa: ANN401
    """Set the form's settings by element id (``design_type`` for ``#design-type``), as a reader would."""
    for key, value in settings.items():
        element = key.replace("design_type", "design-type")
        reveal(page, f"#{element}")
        if element in SELECTS:
            page.select_option(f"#{element}", str(value))
        else:
            page.fill(f"#{element}", str(value))


def generate(page: Any, **settings: object) -> None:  # noqa: ANN401
    """Make the run sheet the plan describes, after choosing any ``settings``, and wait for the page to settle."""
    choose(page, **settings)
    reveal(page, "#generate")
    page.click("#generate")
    settled(page)


def practise(page: Any) -> None:  # noqa: ANN401
    """Simulate the run sheet's results from a known process, as a reader practising would; wait for the analysis."""
    reveal(page, "#panel-simulate")
    page.click("#tab-simulate")
    page.click("#simulate")
    settled(page)


def run_sheet(page: Any) -> list[list[str]]:  # noqa: ANN401
    """Read the run sheet as the reader sees it: each cell's text, or the result typed into it."""
    return page.evaluate(
        """() => [...document.querySelectorAll('#design-table tbody tr')].map((tr) =>
             [...tr.cells].map((td) => td.querySelector('input')?.value ?? td.textContent))"""
    )


def headers(page: Any, table: str = "#design-table") -> list[str]:  # noqa: ANN401
    """Read a table's column headings, without the units the run sheet shows under them."""
    return page.locator(f"{table} thead th button.sort").all_text_contents()


def enter_results(page: Any, results: dict[int, object]) -> None:  # noqa: ANN401
    """Type results into the sheet as a reader would, and wait until the page asks for their analysis.

    ``results`` maps a run's place in the design (0-based, in run order) to its result; ``""``
    clears it. Clearing the last result takes the analysis away instead of asking for one.
    """
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    reveal(page, "#design-table")
    for run, value in results.items():
        page.fill(f'#design-table input[data-index="{run}"]', str(value))
    page.wait_for_function(
        f"window.__pi.count('api_analyze_runs') > {before}"
        " || [...document.querySelectorAll('#design-table input')].every((i) => i.value === '')"
        " && document.querySelector('#analysis').hidden"
    )


def type_results(page: Any, results: dict[int, object]) -> None:  # noqa: ANN401
    """Type results as :func:`enter_results` does, then wait for the analysis made once typing pauses."""
    enter_results(page, results)
    settled(page)


def sort_by(page: Any, column: str, table: str = "#design-table") -> None:  # noqa: ANN401
    """Click a column heading of a table, as a reader sorting it would."""
    reveal(page, table)
    page.locator(f"{table} th button.sort", has_text=re.compile(f"^{re.escape(column)}$")).click()


def last_reply(page: Any, fn: str) -> dict:  # noqa: ANN401
    """Return the last reply the page got from ``fn``, as the harness recorded it."""
    return page.evaluate("(fn) => window.__pi.replies.filter((r) => r.fn === fn).at(-1).reply", fn)


def definition(page: Any, word: str) -> str:  # noqa: ANN401 - a Playwright Page
    """Rest the mouse on a dotted word and return the definition that opens beside it, then close it."""
    reveal(page, word)
    page.hover(word)
    tip = page.locator(".term-tip")
    tip.wait_for(state="visible")
    text = tip.text_content()
    page.keyboard.press("Escape")
    return text
