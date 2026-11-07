"""The note the worker gives while it installs plotly, which it does on the page's first chart.

The first chart may be the curves' fit (``api_functional``) in a study whose only results are its
curves: no analysis follows to say it is complete. The worker ends its own note, so the status
line is not left busy, whether the install fails or works. Only the page's own worker, under
Pyodide, installs plotly, so this is a test of the built app.
"""

from __future__ import annotations

import re

from tests.web._browser import Console
from tests.web._support import FACTORS
from tests.web.e2e._page import candidates_done, generate, set_factors, settled, status
from tests.web.e2e.test_curves import paste_curves
from tests.web.live.test_live import started

#: The plotly wheel micropip downloads; the cache of downloads is routed before it, so this route wins.
PLOTLY_WHEEL = re.compile(r"^https://files\.pythonhosted\.org/packages/.*/plotly-[^/]*\.whl")


def test_the_note_on_installing_plotly_ends_whether_the_install_fails_or_works(own_context, app_url):
    own_context.route(PLOTLY_WHEEL, lambda route: route.abort("blockedbyclient"))
    console = Console()
    console.allow(r"ERR_BLOCKED_BY_CLIENT")  # the browser reports the refused download
    page = own_context.new_page()
    console.watch(page)
    page.goto(app_url)
    started(page)
    set_factors(page, FACTORS)
    candidates_done(page)
    generate(page)

    paste_curves(page)  # no results, only curves: the curves' fit is the first chart, and plotly cannot be had
    page.wait_for_function("window.__pi.count('api_functional') === 1 && window.__pi.idle()", timeout=120_000)
    settled(page, timeout=120_000)
    text, state = status(page)
    assert text.startswith("Could not install plotly: "), text
    assert state == "error"

    own_context.unroute(PLOTLY_WHEEL)
    page.fill("#curve-label", "Colour index")  # the curve renamed is fitted again, and the install tried again
    page.press("#curve-label", "Tab")
    page.wait_for_function("window.__pi.count('api_functional') === 2 && window.__pi.idle()", timeout=120_000)
    settled(page, timeout=120_000)
    assert status(page) == ("Installed plotly for the charts.", "ok")
    page.wait_for_selector("#what-if-curve-plot.js-plotly-plot", state="attached")
    assert not console.unexpected(), console.unexpected()
