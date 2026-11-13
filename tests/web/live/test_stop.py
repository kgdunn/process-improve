"""Stop, against the page's own worker: Pyodide ended in the middle of a search, and started again.

``tests/web/e2e/test_stop.py`` holds a reply back from a stand-in worker. Here the worker is real: the search for
the designs that can learn a brief of six factors takes Pyodide long enough for Stop to be offered, and ending it
ends Python itself. The page starts Python again from what the browser kept, a chart is drawn (plotly is installed
anew, as the first time), and a second search is stopped as the first was.
"""

from __future__ import annotations

from typing import Any

from tests.web._browser import BOOT_TIMEOUT, Console
from tests.web._support import factors_named
from tests.web.e2e._page import generate, practise, reveal, set_factors, settled, status
from tests.web.live.test_live import started

READY = "document.querySelector('#status .status__text').textContent.startsWith('Ready')"


def stop_the_search(page: Any) -> None:  # noqa: ANN401 - a Playwright Page
    """Stop Python once Stop is offered, which it is when a call has waited a while; wait for Python to start again."""
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
    set_factors(page, factors_named(*"ABCDEF"))  # the page looks for the designs that can learn six factors
    stop_the_search(page)

    generate(page, design_type="fractional_factorial")  # the Python that followed answers
    practise(page)
    settled(page, timeout=180_000)
    reveal(page, "#charts")
    page.wait_for_function(
        "document.querySelectorAll('#charts .js-plotly-plot').length >= 2", timeout=180_000
    )  # it lacked plotly, as the first did, and installed it for its first chart

    set_factors(page, factors_named(*"ABCDEFG"))  # a seventh factor: a new brief, and a search for its designs
    stop_the_search(page)  # the worker that followed is ended as the first was
    assert not console.unexpected(), console.unexpected()
