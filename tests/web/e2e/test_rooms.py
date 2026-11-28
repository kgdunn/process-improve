"""Rooms: two browsers keep one study in step through a server that holds it sealed and cannot read it.

Each context is a browser of its own. ``/api/room`` is answered by factori.al's own encrypted study locker
(``deploy/factorial/encrypted_study_locker.py``), run on a local port over a folder of the test's: each request
the page makes is passed to it as Caddy would pass it, and the locker's answer back. So the page's
sealing (``web/room.js``) meets the locker's checks, and the test reads what the server keeps.
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
    """The locker the pages reach, the folder it keeps the rooms in, and each request passed to it."""

    server: Any
    folder: Path
    asked: list[tuple[str, str, dict[str, str], bytes]] = field(default_factory=list)
    up: bool = True

    def rooms(self) -> list[Path]:
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
def in_rooms(open_context: Callable[[], Any], locker: Locker, site: str) -> Callable[[], Any]:
    """Return a way to open a browser whose ``/api/room`` reaches the locker, as Caddy passes it on."""
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
        connection.request(request.method, "/api/room", body=body, headers={**headers, "X-Real-IP": "127.0.0.1"})
        answer = connection.getresponse()
        route.fulfill(status=answer.status, headers=dict(answer.getheaders()), body=answer.read())
        connection.close()

    def open_one() -> Any:  # noqa: ANN401 - a Playwright BrowserContext
        context = open_context()
        context.route(f"{site}api/room", forward)
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
    """Come back to a page, as a reader does: it looks at its room."""
    page.evaluate("window.dispatchEvent(new Event('focus'))")


def worked(context: Any, site: str) -> Any:  # noqa: ANN401 - a Playwright BrowserContext and Page
    """Open the worked example, as the ``page`` fixture does, in ``context``."""
    page = context.new_page()
    page.goto(f"{site}?example=worked")
    first_paint_done(page)
    return page


def share_in_room(page: Any) -> str:  # noqa: ANN401 - a Playwright Page
    """Make a study, type two results, share it in a room, and return the room's link."""
    generate(page, design_type="full_factorial", n_center_points=0)
    type_result(page, 0, "12.5")
    type_result(page, 1, "9.75")
    page.click("#room-share")
    page.wait_for_selector("#room-dialog[open]")
    page.click("#room-make")
    page.wait_for_selector("#room-out", state="visible")
    page.wait_for_function("document.querySelector('#room-says').textContent.startsWith('In step with the room')")
    return page.input_value("#room-url")


def join(context: Any, link: str) -> Any:  # noqa: ANN401 - a Playwright BrowserContext
    page = context.new_page()
    page.goto(link)
    page.wait_for_selector(BANNER, state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    return page


def test_a_study_shared_in_a_room_opens_in_another_browser(in_rooms, locker, site):
    page = worked(in_rooms(), site)
    link = share_in_room(page)
    secret = link.removeprefix(f"{site}#room=")
    assert len(secret) == 43, link
    assert page.get_attribute("#room-share", "hidden") is None
    assert page.text_content("#room-share") == "Copy the room's link"

    [room] = locker.rooms()
    sealed = room.read_bytes()
    assert len(sealed) in study_locker.BLOB_SIZES
    for word in (b"Temperature", b"12.5", b"full_factorial", secret.encode()):
        assert word not in sealed, "the server holds nothing it can read"

    elsewhere = join(in_rooms(), link)
    assert elsewhere.text_content(BANNER).startswith("Opened a study shared in a room:")
    assert "The results you type here reach everyone in the room" in elsewhere.text_content(BANNER)
    assert elsewhere.get_attribute(f"{BANNER} a", "href") == "technical-details-how-factorial-works.html#h-room"
    assert results(elsewhere) == results(page)
    assert elsewhere.url == site, "the secret leaves the address"
    assert elsewhere.input_value("#room-url") == link, "the same room"

    for _, url, headers, body in locker.asked:
        assert secret not in url
        assert secret not in str(headers)
        assert secret.encode() not in body


def test_the_study_leaves_the_browser_only_once_the_reader_agrees(in_rooms, locker, site):
    """The question says who will see the study, what the server keeps, and how to stop; Not now sends nothing."""
    page = worked(in_rooms(), site)
    generate(page, design_type="full_factorial", n_center_points=0)
    page.click("#room-share")
    page.wait_for_selector("#room-dialog[open]")
    asked = " ".join(page.text_content("#room-dialog").split())
    for words in (
        "Whoever opens the link has the same study",
        "with a key that is part of the link. factori.al keeps the encrypted study and cannot read it.",
        "Whoever has the link can read the study and change its results.",
        "Leave the room, or remove it for everyone. Each browser keeps its own copy of the study.",
    ):
        assert words in asked, words
    explained = page.get_attribute("#room-dialog a", "href")
    assert explained == "technical-details-how-factorial-works.html#h-room"
    page.click("#room-not-now")
    page.wait_for_selector("#room-dialog", state="hidden")
    page.click("#room-share")
    page.wait_for_selector("#room-dialog[open]")
    page.keyboard.press("Escape")  # no, as Not now
    page.wait_for_selector("#room-dialog", state="hidden")
    assert locker.asked == [], "nothing sent"
    assert page.is_hidden("#room-out")
    assert page.text_content("#room-share") == "Work on it together"

    explanation = in_rooms().new_page()
    explanation.goto(f"{site}{explained}")
    heading = explanation.text_content("#h-room")
    assert heading == "Can I work on a study with others?"


def test_a_room_stays_when_the_reader_keeps_it(in_rooms, locker, site):
    page = worked(in_rooms(), site)
    share_in_room(page)
    page.click("#room-remove")
    page.wait_for_selector("#room-remove-dialog[open]")
    page.click("#room-remove-keep")
    page.wait_for_selector("#room-remove-dialog", state="hidden")
    assert len(locker.rooms()) == 1
    assert page.is_visible("#room-out")


def test_results_typed_in_either_browser_reach_the_other(in_rooms, locker, site):
    page = worked(in_rooms(), site)
    link = share_in_room(page)
    elsewhere = join(in_rooms(), link)

    writes = locker.writes()
    type_result(elsewhere, 2, "14")
    until(elsewhere, lambda: locker.writes() > writes)
    look(page)
    page.wait_for_function(f"document.querySelectorAll('{BOXES}')[2].value === '14'")
    page.wait_for_selector("#toast", state="visible")
    assert page.text_content("#toast") == "1 result came from the room."

    # A correction in the first browser replaces a result the second already had.
    writes = locker.writes()
    type_result(page, 0, "13")
    until(page, lambda: locker.writes() > writes)
    look(elsewhere)
    elsewhere.wait_for_function(f"document.querySelectorAll('{BOXES}')[0].value === '13'")
    assert results(elsewhere) == results(page)
    settled(elsewhere)


def test_a_result_changed_in_both_browsers_keeps_the_room_s_and_says_so(in_rooms, locker, site):
    page = worked(in_rooms(), site)
    link = share_in_room(page)
    elsewhere = join(in_rooms(), link)

    writes = locker.writes()
    type_result(page, 1, "10")
    until(page, lambda: locker.writes() > writes)
    type_result(elsewhere, 1, "11")
    elsewhere.wait_for_selector(f"{BANNER}:has-text('theirs reached the room first')")
    assert "run 2: 11 here, 10 in the room" in elsewhere.text_content(BANNER)
    assert results(elsewhere)[1] == "10"


def test_a_room_removed_for_everyone_leaves_each_browser_its_copy(in_rooms, locker, site, console):
    console.allow(r"status of 404")  # the browser reports the read of the room that is gone
    page = worked(in_rooms(), site)
    link = share_in_room(page)
    elsewhere = join(in_rooms(), link)
    kept = results(elsewhere)

    page.click("#room-remove")
    page.wait_for_selector("#room-remove-dialog[open]")
    assert "The room's link stops working for everyone" in page.text_content("#room-remove-dialog")
    page.click("#room-remove-yes")
    page.wait_for_selector("#room-out", state="hidden")
    assert locker.rooms() == []
    assert page.text_content("#room-share") == "Work on it together"

    look(elsewhere)
    elsewhere.wait_for_selector(f"{BANNER}:has-text('The room of this study is gone')")
    assert results(elsewhere) == kept
    assert elsewhere.is_hidden("#room-out")

    gone = in_rooms().new_page()
    gone.goto(link)
    gone.wait_for_selector(BANNER, state="visible")
    assert gone.text_content(BANNER).startswith("This room could not be opened: it is gone.")


def test_a_room_s_link_opened_again_opens_the_study_kept_here(in_rooms, locker, site):
    context = in_rooms()
    page = worked(context, site)
    link = share_in_room(page)
    again = join(context, link)
    assert again.text_content(BANNER).endswith("which this browser keeps in step with its room.")
    assert again.text_content("#library-count") == "1", "no second copy"


def test_a_locker_that_cannot_be_reached_keeps_the_changes_here(in_rooms, locker, site, console):
    console.allow(r"ERR_CONNECTION_REFUSED")  # the browser reports the write it could not send
    page = worked(in_rooms(), site)
    link = share_in_room(page)
    locker.up = False
    type_result(page, 3, "15")
    page.wait_for_function(
        "document.querySelector('#room-says').textContent.startsWith('The room could not be reached')"
    )
    locker.up = True
    look(page)
    page.wait_for_function("document.querySelector('#room-says').textContent.startsWith('In step with the room')")
    elsewhere = join(in_rooms(), link)
    assert results(elsewhere)[3] == "15"
