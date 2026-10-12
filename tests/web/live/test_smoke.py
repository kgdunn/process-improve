"""The app's smoke tests: Python starts, the worked example runs as the build drew it, and the book's examples open.

They gate the deploy (``docs.yml`` runs them against the built ``/app``): the shortest proof
that the page, its worker, ``bootstrap.py`` and the wheel work together under Pyodide. The
build computed the worked example natively; run live, it must come out the same.
"""

from __future__ import annotations

import pytest

from tests.web._browser import BOOT_TIMEOUT
from tests.web.e2e._page import last_reply, run_sheet, settled, status

pytestmark = pytest.mark.smoke


def test_python_starts_with_the_package_the_page_was_built_with(page, demo):
    assert status(page) == (f"Ready: process-improve {demo['version']}", "ok")
    assert page.text_content("#demo-version") == f"process-improve {demo['version']}"


def test_the_worked_example_runs_live_as_the_build_drew_it(page, demo):
    """The same numbers, to rounding: Pyodide's numerics differ from native ones in the last bits.

    Printed, a coefficient on a rounding edge can differ in its fourth digit (-3.357 live,
    -3.358 from the build), so the numbers are compared, not their text.
    """
    page.click("#run-example")
    page.wait_for_function("window.__pi.count('api_plots') >= 1 && window.__pi.idle()", timeout=180_000)
    settled(page)
    responses = [float(row[-1]) for row in run_sheet(page)]
    assert responses == pytest.approx(demo["simulation"]["responses"], abs=0.011), "rounded to two decimals"
    assert page.text_content("#truth-equation") == demo["simulation"]["equation"]
    live = {c["term"]: c["coefficient"] for c in last_reply(page, "api_analyze_runs")["result"]["coefficients"]}
    built = {c["term"]: c["coefficient"] for c in demo["analysis"]["coefficients"]}
    assert live == pytest.approx(built, rel=1e-6, abs=1e-9)
    page.wait_for_function("document.querySelectorAll('#charts .js-plotly-plot').length === 4", timeout=60_000)
    assert status(page) == ("Analysis complete.", "ok")


def test_every_example_the_book_links_to_is_published(page):
    """The book prints ``#book=<name>`` links: each name the index lists is served, as the page reads it."""
    found = page.evaluate(
        """async () => {
          const index = await (await fetch("book/index.json")).json();
          const served = [];
          for (const { name } of index.examples) {
            const res = await fetch(`book/${name}.json`);
            served.push(res.ok && (await res.json()).name === name);
          }
          return { names: index.examples.map((e) => e.name), served };
        }"""
    )
    assert "pollutant" in found["names"]
    assert all(found["served"]), [n for n, ok in zip(found["names"], found["served"], strict=True) if not ok]


def test_a_book_example_opens_and_is_analysed_under_pyodide(own_context, app_url):
    """The link the book prints opens its runs and results, and Pyodide fits the book's coefficients."""
    page = own_context.new_page()
    page.goto(f"{app_url}#book=pollutant")
    page.wait_for_selector("#campaign-banner", state="visible")
    assert page.text_content("#campaign-banner").startswith("Opened the book's example: Pollutant discharged")
    page.wait_for_function("window.__pi.count('api_analyze_runs') >= 1 && window.__pi.idle()", timeout=BOOT_TIMEOUT)
    settled(page, timeout=120_000)
    live = {c["label"]: c["coefficient"] for c in last_reply(page, "api_analyze_runs")["result"]["coefficients"]}
    book = {"Intercept": 11.25, "C": 6.25, "T": 0.75, "S": -7.25, "C·T": 0.25, "C·S": -6.75, "T·S": -0.25}
    assert live == pytest.approx(book, abs=1e-9)
