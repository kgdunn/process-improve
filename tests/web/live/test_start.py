"""What the page starts without: statsmodels and scikit-learn, which a model is fitted with.

Python is ready as soon as a design can be made. The worker then fetches the two in the background, and a fit
that comes sooner waits for that same install. Here statsmodels cannot be had at first, which shows what no
other tier can: the page is ready all the same and makes a design; the first fit says why it waits, and fails for
want of the package; and the next fit, once it can be had, installs it and answers. Only the page's own worker,
under Pyodide, does any of this, so this is a test of the built app.
"""

from __future__ import annotations

import re

from tests.web._browser import Console
from tests.web._support import FACTORS
from tests.web.e2e._page import (
    candidates_done,
    enter_results,
    generate,
    reveal,
    run_sheet,
    set_factors,
    settled,
    status,
)
from tests.web.e2e.test_report import download_report, opened, stamp
from tests.web.live.test_live import started

#: The statsmodels wheel of the Pyodide release; the cache of downloads is routed before it, so this route wins.
STATSMODELS = re.compile(r"^https://cdn\.jsdelivr\.net/pyodide/.*/statsmodels-[^/]*\.whl")

#: True once no call to Python has been in flight, and the status line not busy, for a second on end. Each result
#: typed is analysed in turn, and each analysis that lacks a package asks the worker to try the install again: the
#: page is at rest between two of them for a moment only.
AT_REST = """() => {
  const calm = window.__pi.idle() && document.querySelector('#status').dataset.state !== 'busy';
  if (!calm) window.__calmSince = null;
  else window.__calmSince ??= performance.now();
  return calm && performance.now() - window.__calmSince > 1000;
}"""


def test_the_page_is_ready_and_designs_without_what_a_model_is_fitted_with_and_a_fit_fetches_it(own_context, app_url):
    own_context.route(STATSMODELS, lambda route: route.abort("blockedbyclient"))
    console = Console()
    console.allow(r"ERR_BLOCKED_BY_CLIENT")  # the browser reports the refused download
    console.allow(r"statsmodels")  # and Pyodide the package it could not load
    page = own_context.new_page()
    console.watch(page)
    page.goto(app_url)
    started(page)
    assert status(page)[0].startswith("Ready: process-improve"), "ready, with statsmodels nowhere to be had"

    set_factors(page, FACTORS)
    candidates_done(page)
    generate(page, design_type="full_factorial")
    assert len(run_sheet(page)) == 11, "a design is made, and compared with the others, without it"
    on = stamp(opened(page, download_report(page)[1]))["On"]
    assert "scipy" in on, on
    assert "statsmodels" not in on, "no model was fitted, and the report names nothing that is not there"

    results = {run: 50 + 3 * run for run in range(11)}
    enter_results(page, results)
    page.wait_for_function(AT_REST, timeout=120_000)
    text, state = status(page)
    assert state == "error", (text, state)
    assert page.locator("#analysis .equation").count() == 0, "no fit, and nothing shown as one"
    refused = "Could not install statsmodels and scikit-learn: "
    said = text + (page.text_content("#analysis") or "")
    assert refused in said, f"the reader is told why there is no analysis: {said}"

    own_context.unroute(STATSMODELS)
    enter_results(page, {0: 51})  # a result changed is analysed again, and the install tried again
    reveal(page, "#analysis")
    page.wait_for_selector("#analysis .equation", timeout=120_000)
    settled(page, timeout=120_000)
    assert status(page)[1] == "ok", status(page)
    on = stamp(opened(page, download_report(page)[1]))["On"]
    assert re.search(r"statsmodels \d+\.\d+", on), f"the report names what the model was fitted with: {on}"
    assert not console.unexpected(), console.unexpected()
