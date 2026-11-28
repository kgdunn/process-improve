"""Lockers: two browsers keep one study in step through a server that holds it sealed and cannot read it.

Each context is a browser of its own. ``/api/locker`` is answered by factori.al's own encrypted study locker
(``deploy/factorial/encrypted_study_locker.py``), run on a local port over a folder of the test's: each request
the page makes is passed to it as Caddy would pass it, and the locker's answer back. So the page's
sealing (``web/locker.js``) meets the locker's checks, and the test reads what the server keeps.
"""

from __future__ import annotations

import http.client
import importlib.util
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.web._support import ROOT
from tests.web.e2e._page import first_paint_done, generate, settled

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

BANNER = "#campaign-banner"
#: The result boxes of the run sheet, in the sheet's order.
BOXES = "#design-table input[data-index]"


def load_locker():
    spec = importlib.util.spec_from_file_location(
        "encrypted_study_locker", ROOT / "deploy" / "factorial" / "encrypted_study_locker.py"
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study_locker = load_locker()


@dataclass
class Locker:
    """The locker the pages reach, the folder it keeps the lockers in, and each request passed to it."""

    server: Any
    folder: Path
    asked: list[tuple[str, str, dict[str, str], bytes]] = field(default_factory=list)
    up: bool = True

    def lockers(self) -> list[Path]:
        return [path for path in self.folder.iterdir() if not path.name.startswith(".")]

    def writes(self) -> int:
        return sum(method == "PUT" for method, *_ in self.asked)


@pytest.fixture
def locker(tmp_path: Path, site: str) -> Iterator[Locker]:
    """Run factori.al's encrypted study locker on a free local port over the test's folder, for the test site's page."""
    store = study_locker.Store(tmp_path)
    with study_locker.Server(("127.0.0.1", 0), store, origin=site.rstrip("/")) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        yield Locker(server, tmp_path)
        server.shutdown()


@pytest.fixture
def in_lockers(open_context: Callable[[], Any], locker: Locker, site: str) -> Callable[[], Any]:
    """Return a way to open a browser whose ``/api/locker`` reaches the locker, as Caddy passes it on."""
    kept = {"authorization", "if-match", "if-none-match", "content-type", "origin"}

    def forward(route: Any) -> None:  # noqa: ANN401 - a Playwright Route
        request = route.request
        if not locker.up:
            route.abort("connectionrefused")
            return
        headers = {name: value for name, value in request.all_headers().items() if name in kept}
        body = request.post_data_buffer or b""
        locker.asked.append((request.method, request.url, headers, body))
        connection = http.client.HTTPConnection(*locker.server.server_address, timeout=10)
        connection.request(request.method, "/api/locker", body=body, headers={**headers, "X-Real-IP": "127.0.0.1"})
        answer = connection.getresponse()
        route.fulfill(status=answer.status, headers=dict(answer.getheaders()), body=answer.read())
        connection.close()

    def open_one() -> Any:  # noqa: ANN401 - a Playwright BrowserContext
        context = open_context()
        context.route(f"{site}api/locker", forward)
        return context

    return open_one


def results(page: Any) -> list[str]:  # noqa: ANN401 - a Playwright Page
    return page.eval_on_selector_all(BOXES, "boxes => boxes.map((box) => box.value)")


def type_result(page: Any, place: int, value: str) -> None:  # noqa: ANN401 - a Playwright Page
    """Type a result on the run sheet, in Run, as a reader does."""
    if page.get_attribute("#workspace", "data-phase") != "run":
        page.click("#go-run")
    page.fill(f"#design-table input[data-index='{place}']", value)


def until(page: Any, holds: Callable[[], bool], timeout: float = 10_000) -> None:  # noqa: ANN401 - a Playwright Page
    """Wait until ``holds()``, letting the page and its routes run meanwhile: a sleep would stop the routes."""
    for _ in range(int(timeout / 50)):
        if holds():
            return
        page.wait_for_timeout(50)
    pytest.fail("waited, and it did not happen")


def look(page: Any) -> None:  # noqa: ANN401 - a Playwright Page
    """Come back to a page, as a reader does: it looks at its locker."""
    page.evaluate("window.dispatchEvent(new Event('focus'))")


def worked(context: Any, site: str) -> Any:  # noqa: ANN401 - a Playwright BrowserContext and Page
    """Open the worked example, as the ``page`` fixture does, in ``context``."""
    page = context.new_page()
    page.goto(f"{site}?example=worked")
    first_paint_done(page)
    return page


def share_in_locker(page: Any) -> str:  # noqa: ANN401 - a Playwright Page
    """Make a study, type two results, share it in a locker, and return the locker's link."""
    generate(page, design_type="full_factorial", n_center_points=0)
    type_result(page, 0, "12.5")
    type_result(page, 1, "9.75")
    page.click("#locker-share")
    page.wait_for_selector("#locker-dialog[open]")
    page.click("#locker-make")
    page.wait_for_selector("#locker-out", state="visible")
    page.wait_for_function("document.querySelector('#locker-says').textContent.startsWith('In step with the locker')")
    return page.input_value("#locker-url")


def join(context: Any, link: str) -> Any:  # noqa: ANN401 - a Playwright BrowserContext
    page = context.new_page()
    page.goto(link)
    page.wait_for_selector(BANNER, state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    return page


def test_a_study_shared_in_a_locker_opens_in_another_browser(in_lockers, locker, site):
    page = worked(in_lockers(), site)
    link = share_in_locker(page)
    secret = link.removeprefix(f"{site}#locker=")
    assert len(secret) == 43, link
    assert page.get_attribute("#locker-share", "hidden") is None
    assert page.text_content("#locker-share .locker-share__words") == "Copy the locker's link"

    [held] = locker.lockers()
    sealed = held.read_bytes()
    assert len(sealed) in study_locker.BLOB_SIZES
    for word in (b"Temperature", b"12.5", b"full_factorial", secret.encode()):
        assert word not in sealed, "the server holds nothing it can read"

    elsewhere = join(in_lockers(), link)
    assert elsewhere.text_content(BANNER).startswith("Opened a study shared in a locker:")
    assert "The results you type here reach everyone with the locker's link" in elsewhere.text_content(BANNER)
    assert elsewhere.get_attribute(f"{BANNER} a", "href") == "technical-details-how-factorial-works.html#h-locker"
    assert results(elsewhere) == results(page)
    assert elsewhere.url == site, "the secret leaves the address"
    assert elsewhere.input_value("#locker-url") == link, "the same locker"

    for _, url, headers, body in locker.asked:
        assert secret not in url
        assert secret not in str(headers)
        assert secret.encode() not in body


def test_the_study_leaves_the_browser_only_once_the_reader_agrees(in_lockers, locker, site):
    """The question says who will see the study, what the server keeps, and how to stop; Not now sends nothing."""
    page = worked(in_lockers(), site)
    generate(page, design_type="full_factorial", n_center_points=0)
    page.click("#locker-share")
    page.wait_for_selector("#locker-dialog[open]")
    asked = " ".join(page.text_content("#locker-dialog").split())
    for words in (
        "Whoever opens the link has the same study",
        "with a key that is part of the link. factori.al keeps the encrypted study and cannot read it.",
        "Whoever has the link can read the study and change its results.",
        "Stop using the locker, or remove it for everyone. Each browser keeps its own copy of the study.",
    ):
        assert words in asked, words
    explained = page.get_attribute("#locker-dialog a", "href")
    assert explained == "technical-details-how-factorial-works.html#h-locker"
    page.click("#locker-not-now")
    page.wait_for_selector("#locker-dialog", state="hidden")
    page.click("#locker-share")
    page.wait_for_selector("#locker-dialog[open]")
    page.keyboard.press("Escape")  # no, as Not now
    page.wait_for_selector("#locker-dialog", state="hidden")
    assert locker.asked == [], "nothing sent"
    assert page.is_hidden("#locker-out")
    assert page.text_content("#locker-share .locker-share__words") == "Share this study"

    # A padlock before the words, and a screen reader hears that the study goes encrypted.
    assert page.locator("#locker-share svg[aria-hidden='true']").count() == 1
    assert page.get_by_role("button", name="Share this study", exact=True).is_visible()
    assert page.text_content("#locker-make") == "Work on it together", "in the question"
    assert page.locator("#locker-make svg[aria-hidden='true']").count() == 1

    explanation = in_lockers().new_page()
    explanation.goto(f"{site}{explained}")
    heading = explanation.text_content("#h-locker")
    assert heading == "Can I work on a study with others?"


def test_a_locker_stays_when_the_reader_keeps_it(in_lockers, locker, site):
    page = worked(in_lockers(), site)
    share_in_locker(page)
    page.click("#locker-remove")
    page.wait_for_selector("#locker-remove-dialog[open]")
    page.click("#locker-remove-keep")
    page.wait_for_selector("#locker-remove-dialog", state="hidden")
    assert len(locker.lockers()) == 1
    assert page.is_visible("#locker-out")


def test_results_typed_in_either_browser_reach_the_other(in_lockers, locker, site):
    page = worked(in_lockers(), site)
    link = share_in_locker(page)
    elsewhere = join(in_lockers(), link)

    writes = locker.writes()
    type_result(elsewhere, 2, "14")
    until(elsewhere, lambda: locker.writes() > writes)
    look(page)
    page.wait_for_function(f"document.querySelectorAll('{BOXES}')[2].value === '14'")
    page.wait_for_selector("#toast", state="visible")
    assert page.text_content("#toast") == "1 result came from the locker."

    # A correction in the first browser replaces a result the second already had.
    writes = locker.writes()
    type_result(page, 0, "13")
    until(page, lambda: locker.writes() > writes)
    look(elsewhere)
    elsewhere.wait_for_function(f"document.querySelectorAll('{BOXES}')[0].value === '13'")
    assert results(elsewhere) == results(page)
    settled(elsewhere)


def test_a_result_changed_in_both_browsers_keeps_the_locker_s_and_says_so(in_lockers, locker, site):
    page = worked(in_lockers(), site)
    link = share_in_locker(page)
    elsewhere = join(in_lockers(), link)

    writes = locker.writes()
    type_result(page, 1, "10")
    until(page, lambda: locker.writes() > writes)
    type_result(elsewhere, 1, "11")
    elsewhere.wait_for_selector(f"{BANNER}:has-text('theirs reached the locker first')")
    assert "run 2: 11 here, 10 in the locker" in elsewhere.text_content(BANNER)
    assert results(elsewhere)[1] == "10"


def test_a_locker_removed_for_everyone_leaves_each_browser_its_copy(in_lockers, locker, site, console):
    console.allow(r"status of 404")  # the browser reports the read of the locker that is gone
    page = worked(in_lockers(), site)
    link = share_in_locker(page)
    elsewhere = join(in_lockers(), link)
    kept = results(elsewhere)

    page.click("#locker-remove")
    page.wait_for_selector("#locker-remove-dialog[open]")
    assert "The locker's link stops working for everyone" in page.text_content("#locker-remove-dialog")
    page.click("#locker-remove-yes")
    page.wait_for_selector("#locker-out", state="hidden")
    assert locker.lockers() == []
    assert page.text_content("#locker-share .locker-share__words") == "Share this study"

    look(elsewhere)
    elsewhere.wait_for_selector(f"{BANNER}:has-text('The locker of this study is gone')")
    assert results(elsewhere) == kept
    assert elsewhere.is_hidden("#locker-out")

    gone = in_lockers().new_page()
    gone.goto(link)
    gone.wait_for_selector(BANNER, state="visible")
    assert gone.text_content(BANNER).startswith("This locker could not be opened: it is gone.")


def test_a_locker_s_link_opened_again_opens_the_study_kept_here(in_lockers, locker, site):
    context = in_lockers()
    page = worked(context, site)
    link = share_in_locker(page)
    again = join(context, link)
    assert again.text_content(BANNER).endswith("which this browser keeps in step with its locker.")
    assert again.text_content("#library-count") == "1", "no second copy"


def test_a_locker_that_cannot_be_reached_keeps_the_changes_here(in_lockers, locker, site, console):
    console.allow(r"ERR_CONNECTION_REFUSED")  # the browser reports the write it could not send
    page = worked(in_lockers(), site)
    link = share_in_locker(page)
    locker.up = False
    type_result(page, 3, "15")
    page.wait_for_function(
        "document.querySelector('#locker-says').textContent.startsWith('The locker could not be reached')"
    )
    locker.up = True
    look(page)
    page.wait_for_function("document.querySelector('#locker-says').textContent.startsWith('In step with the locker')")
    elsewhere = join(in_lockers(), link)
    assert results(elsewhere)[3] == "15"
