"""The copy kept for use without a network, where there is none to keep: a plain preview, and a build that retires it.

The page keeps a copy of a built app only (``web/offline.js``). ``tests/web/live/test_offline.py`` opens the built
app without a network; here the page is the checkout's, whose manifest names no wheel.
"""

from __future__ import annotations

COPIES = "caches.keys().then((names) => names.filter((name) => name.startsWith('factorial-')))"
REGISTERED = "navigator.serviceWorker.getRegistration().then((found) => found !== undefined)"


def test_a_plain_preview_of_the_page_registers_no_worker_and_keeps_no_copy(fresh_page):
    """Its files change under the reader's hands: a copy of them would be yesterday's page."""
    page = fresh_page
    page.wait_for_timeout(300)  # Python has answered: a worker, were one wanted, would be registered by now
    assert not page.evaluate(REGISTERED)
    assert page.evaluate(COPIES) == []
    assert page.is_hidden("#offline"), "nothing is kept, so the footer says nothing of a copy"


def test_a_build_that_retires_the_worker_has_its_copies_deleted_at_the_next_visit(context, site, plotly_js):
    """``scripts/build_web_app.py --no-service-worker`` writes this manifest: the way out of a worker gone wrong."""
    page = context.new_page()
    page.goto(site)
    page.evaluate("caches.open('factorial-kept-1').then((kept) => kept.put('pyodide.asm.wasm', new Response('kept')))")
    assert page.evaluate(COPIES) == ["factorial-kept-1"], "a copy an earlier build kept"
    retiring = {"wheel": "0123456789abcdef/process_improve-9.9.9-py3-none-any.whl", "service_worker": False}
    context.route(
        f"{site}wheels/manifest.json", lambda route: route.fulfill(json={**retiring, "plotly_js": plotly_js[0]})
    )
    page.reload()
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    page.wait_for_timeout(300)
    assert page.evaluate(COPIES) == []
    assert not page.evaluate(REGISTERED)
    assert page.is_hidden("#offline")
