"""Stop, against the page's own worker: Pyodide ended in the middle of a search, and started again.

``tests/web/e2e/test_stop.py`` holds a reply back from a stand-in worker. Here the worker is real: the search for
the optimal designs of a brief of six factors takes Pyodide long enough for Stop to be offered, and ending it ends
Python itself. The page starts Python again from what the browser kept, a chart is drawn (plotly is installed anew,
as the first time), and a second search is stopped as the first was.
"""

from __future__ import annotations

from typing import Any

from tests.web._browser import BOOT_TIMEOUT, Console
from tests.web._support import factors_named
from tests.web.e2e._page import choose, reveal, set_aim, set_factors, settled, status
from tests.web.live.test_live import started

READY = "document.querySelector('#status .status__text').textContent.startsWith('Ready')"


def search_and_stop(page: Any) -> None:  # noqa: ANN401 - a Playwright Page
    """Ask for the optimal designs of the brief as well, and stop the search once Stop is offered.

    Stop is offered when a call has waited a while, which this one does: Python is then ended, and started again.
    """
    listed = "!document.querySelector('#candidates').hasAttribute('aria-busy') && window.__pi.idle()"
    page.wait_for_function(listed, timeout=180_000)
    set_aim(page, "main_quadratic")  # a model with squares: its optimal designs wait until the reader asks
    page.wait_for_function(listed, timeout=180_000)
    reveal(page, "#search")
    page.click("#search")
    page.wait_for_selector("#stop:not([hidden])", timeout=60_000)
    page.click("#stop")
    page.wait_for_function("document.querySelector('#stop').hidden")
    assert status(page)[0].startswith("Stopped"), status(page)[0]
    page.wait_for_function(READY, timeout=BOOT_TIMEOUT)  # a new worker, and Python in it


def test_a_real_search_is_stopped_and_python_starts_again_and_again(own_context, app_url):
    console = Console()
    page = own_context.new_page()
    console.watch(page)
    page.goto(app_url)
    started(page)
    set_factors(page, factors_named(*"ABCDEF"))
    search_and_stop(page)

    set_aim(page, "main_effects")
    choose(page, design_type="fractional_factorial")
    reveal(page, "#generate")
    page.click("#generate")
    settled(page, timeout=180_000)  # the Python that followed answers, once it has listed the brief's designs again
    reveal(page, "#panel-simulate")
    page.click("#tab-simulate")
    page.click("#simulate")  # results to practise on, analysed
    settled(page, timeout=180_000)
    reveal(page, "#charts")
    page.wait_for_function(
        "document.querySelectorAll('#charts .js-plotly-plot').length >= 2", timeout=180_000
    )  # it lacked plotly, as the first did, and installed it for its first chart

    set_factors(page, factors_named(*"ABCDEFG"))  # a seventh factor: a new brief, with optimal designs to search for
    search_and_stop(page)  # the worker that followed is ended as the first was
    assert not console.unexpected(), console.unexpected()
