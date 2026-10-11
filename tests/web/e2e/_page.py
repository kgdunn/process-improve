"""Driving the page in the browser tests: waiting for it, reading it, and filling in its form as a reader would.

Every helper takes a Playwright page. ``window.__pi`` is the harness's record of the page's
calls to Python (``harness.js``); a helper that starts work waits until nothing is in flight
and the status line has stopped saying it is busy.
"""

from __future__ import annotations

from typing import Any

#: The <select>s of the form; every other setting is an <input>.
SELECTS = {"design-type", "alpha", "resolution", "model_type", "scenario", "model", "compare-model"}


def settled(page: Any, timeout: float = 30_000) -> None:  # noqa: ANN401 - a Playwright Page
    """Wait until no call to Python is in flight and the status line is no longer busy."""
    page.wait_for_function(
        "window.__pi.idle() && document.querySelector('#status').dataset.state !== 'busy'", timeout=timeout
    )


def first_paint_done(page: Any, first_paint: dict) -> None:  # noqa: ANN401
    """Wait until the page has drawn the worked example, its comparison last, and Python has answered."""
    charts = 1 + len(first_paint["comparison"]["result"]["designs"])  # the FDS plot and a correlation map each
    page.wait_for_function(
        "window.__pi.count('api_catalogue') === 1 && window.__pi.idle()"
        f" && document.querySelectorAll('#compare-charts .js-plotly-plot').length === {charts}",
        timeout=30_000,
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
    rows = page.locator("#factors tbody tr")
    while rows.count() > len(factors):
        rows.last.get_by_role("button", name="Remove factor").click()
    while rows.count() < len(factors):
        page.click("#add-factor")
    for row, factor in zip(rows.all(), factors, strict=True):
        for key in ("name", "low", "high", "units"):
            row.locator(f"[name={key}]").fill(str(factor.get(key, "")))


def choose(page: Any, **settings: object) -> None:  # noqa: ANN401
    """Set the form's settings by element id (``design_type`` for ``#design-type``), as a reader would."""
    for key, value in settings.items():
        element = key.replace("design_type", "design-type").replace("compare_model", "compare-model")
        if element in SELECTS:
            page.select_option(f"#{element}", str(value))
        else:
            page.fill(f"#{element}", str(value))


def generate(page: Any, **settings: object) -> None:  # noqa: ANN401
    """Generate a design from the form, after choosing any ``settings``, and wait for the page to settle."""
    choose(page, **settings)
    page.click("#generate")
    settled(page)


def run_sheet(page: Any) -> list[list[str]]:  # noqa: ANN401
    """Read the run sheet as the reader sees it: each cell's text, or the result typed into it."""
    return page.evaluate(
        """() => [...document.querySelectorAll('#design-table tbody tr')].map((tr) =>
             [...tr.cells].map((td) => td.querySelector('input')?.value ?? td.textContent))"""
    )


def headers(page: Any, table: str = "#design-table") -> list[str]:  # noqa: ANN401
    """Read a table's column headings."""
    return page.locator(f"{table} thead th").all_text_contents()


def type_result(page: Any, run: int, value: object) -> None:  # noqa: ANN401
    """Type a result for the run in row ``run`` (0-based) of the sheet, or clear it with ``""``."""
    page.fill(f'#design-table input[data-index="{run}"]', str(value))
