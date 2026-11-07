"""Draw the card a chat or a blog post shows for a link to the concept map, from the map itself.

Usage, from the repository root (``PI_CHROMIUM`` names a Chromium when Playwright's own is not installed)::

    uv run --with playwright==1.64.1 python tools/branding/map_card.py web

It serves ``web/`` on a free local port, opens ``map/`` at 1200 x 630 in the light theme, lays the page out as a
card, its title and lede on the left and the map of topics on the right, and writes ``web/map/og-card.png``. Run it
again when the topics change.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

WIDTH, HEIGHT = 1200, 630
# The page as a card: the words on the left, the topics on the right, the rest of the page hidden.
CARD_CSS = """
.map-top, .map-meta, .map-find, .map-bar, .map-legend, .map-card, .map-links, .map-how {
  display: none !important;
}
main.map { width: 400px; max-width: none; margin: 0; padding: 72px 0 0 56px; }
main.map h1 { font-size: 46px; line-height: 1.1; margin: 0 0 20px; }
main.map .lede { font-size: 23px; line-height: 1.4; }
.map-box { position: fixed; top: 32px; right: 32px; width: 700px; height: 566px; }
.card-brand { position: fixed; left: 56px; bottom: 52px; margin: 0; display: flex; gap: 12px; align-items: center;
  font-size: 22px; color: var(--muted); }
"""
BRAND = '<p class="card-brand"><img src="../favicon.svg" width="36" height="36" alt="">factori.al/map</p>'


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        """Serve without a line per request."""


def draw(web: Path) -> Path:
    """Write ``web/map/og-card.png`` from the map page, served from ``web``, and return its path."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(web)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = web / "map" / "og-card.png"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=os.environ.get("PI_CHROMIUM"))
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, color_scheme="light")
            page.goto(f"http://127.0.0.1:{server.server_port}/map/")
            page.wait_for_function("() => window.conceptMap && document.querySelector('#card h2')")
            page.add_style_tag(content=CARD_CSS)
            page.evaluate(
                """(brand) => {
                  document.body.insertAdjacentHTML('beforeend', brand);
                  const cy = window.conceptMap.cy;
                  cy.resize();
                  cy.fit(cy.nodes('.topic'), 16);
                }""",
                BRAND,
            )
            page.wait_for_timeout(800)  # the labels follow the viewport a frame later
            page.screenshot(path=out)
            browser.close()
    finally:
        server.shutdown()
    return out


if __name__ == "__main__":
    print(draw(Path(sys.argv[1])))
