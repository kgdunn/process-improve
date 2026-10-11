"""The app's smoke tests: Python starts in the browser, and the worked example runs there as the build drew it.

They gate the deploy (``docs.yml`` runs them against the built ``/app``): the shortest proof
that the page, its worker, ``bootstrap.py`` and the wheel work together under Pyodide. The
build computed the worked example natively; run live, it must come out the same.
"""

from __future__ import annotations

import pytest

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
