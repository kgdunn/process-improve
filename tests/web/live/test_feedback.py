"""The Feedback dialog in the built app, with Pyodide: the screenshot's module is published, and takes the tab.

The test answers the report itself, so it is never posted: the app under test may be the one at
factori.al (``--web-app``), whose relay would email the maintainer.
"""

from __future__ import annotations

import base64

import pytest

#: Where the page posts a report from (``relayUrl`` in ``web/feedback.js``); a copy anywhere else takes no screenshot.
POSTS_FROM = {"factori.al", "127.0.0.1", "localhost", "[::1]"}


def test_the_built_app_sends_a_screenshot_of_the_tab(page):
    if page.evaluate("location.hostname") not in POSTS_FROM:
        pytest.skip("a copy of the app at another address sends no report, so it takes no screenshot")
    reports = []

    def answer(route) -> None:
        reports.append(route.request.post_data_json)
        route.fulfill(json={"ok": True})

    page.route("**/api/feedback", answer)
    try:
        page.click("#feedback")
        page.wait_for_selector("#feedback-shot[src^='data:image/jpeg;base64,']", state="visible", timeout=60_000)
        page.fill("#feedback-message", "A report from the live tier's test, which answers it itself.")
        page.click("#feedback-send")
        page.wait_for_selector('#feedback-dialog [data-part="sent"]', state="visible")
    finally:
        page.keyboard.press("Escape")
        page.unroute("**/api/feedback")
    [report] = reports
    assert base64.b64decode(report["screenshot"].split(",", 1)[1])[:3] == b"\xff\xd8\xff"
    assert any(row.startswith("With: process-improve ") for row in report["context"]["stamp"])
    assert report["context"]["errors"] == []
