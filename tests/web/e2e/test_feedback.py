"""The Feedback dialog in the browser: what a reader writes, and only what they choose to add, reaches the relay.

The relay is factori.al's (``deploy/factorial/feedback_relay.py``). Here each test answers
``/api/feedback`` itself and keeps what the page posted, so it reads the report as the relay
would; the local server would refuse a POST. The page posts from loopback, as it does from
factori.al, so every test reaches the post, or the fallback the dialog offers when the post fails.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.web._support import WEB
from tests.web.e2e._page import overflowing, reveal, settled

#: The parts of the dialog: what the reader writes in, and what it says once the report has gone.
DIALOG = "#feedback-dialog"
WRITE = f'{DIALOG} [data-part="write"]'
SENT = f'{DIALOG} [data-part="sent"]'
FALLBACK = f'{DIALOG} [data-part="fallback"]'
STATUS = f'{DIALOG} [data-note="status"]'
#: The screenshot under its box, once taken.
SHOT = "#feedback-shot[src^='data:image/jpeg;base64,']"
#: The dialog has answered a press of Send: the report went, or the ways to send it otherwise are shown, or it says why
#: the report cannot go.
ANSWERED = f"""() => {{
  const shown = (selector) => document.querySelector(selector)?.hidden === false;
  return shown('{SENT}') || shown('{FALLBACK}') || document.querySelector('{STATUS}').classList.contains('error');
}}"""
MESSAGE = "The surface on Learn reads as a maximum, but the canonical analysis shows a ridge along T."
#: The reference the relay gives a report, as it makes one.
REFERENCE = "FB-7K3Q2M"
PACKED = re.compile(r"[A-Za-z0-9_-]+")
#: A copy of the app at another address than factori.al, as GitHub Pages or a reader's own server serves it.
MIRROR = "http://mirror.test/"


@dataclass
class Relay:
    """The relay as the page reaches it: each report posted, and the status it answers with (None: unreachable)."""

    reports: list[dict] = field(default_factory=list)
    status: int | None = 200


@pytest.fixture
def relay(context: Any, site: str) -> Relay:  # noqa: ANN401 - a Playwright BrowserContext
    """Answer ``/api/feedback`` as the relay does, with ``Relay.status``, keeping each report the page posts."""
    answering = Relay()

    def answer(route: Any) -> None:  # noqa: ANN401 - a Playwright Route
        answering.reports.append(route.request.post_data_json)
        if answering.status is None:
            route.abort("internetdisconnected")
            return
        if answering.status == 200:
            receipt = "email" in answering.reports[-1]  # as the relay does: a receipt to the address a report gives
            route.fulfill(status=200, json={"ok": True, "reference": REFERENCE, "receipt": receipt})
        else:
            route.fulfill(status=answering.status, json={"ok": False, "error": "refused"})

    context.route(f"{site}api/feedback", answer)
    return answering


def open_feedback(page: Any, opener: str = "#feedback") -> None:  # noqa: ANN401 - a Playwright Page
    """Open the dialog as a reader would, with the button ``opener``, and wait until it lists what it will send."""
    reveal(page, opener)
    page.click(opener)
    page.wait_for_selector(f"{DIALOG}[open]")
    page.wait_for_function("document.querySelector('#feedback-sent').textContent.includes('Tab on show')")


def send(page: Any, message: str = MESSAGE, email: str = "") -> None:  # noqa: ANN401 - a Playwright Page
    """Write the message, and the address if one is given, then press Send and wait for what the dialog says."""
    page.fill("#feedback-message", message)
    if email:
        page.fill("#feedback-email", email)
    page.click("#feedback-send")
    page.wait_for_function(ANSWERED)


def topic(page: Any) -> str:  # noqa: ANN401 - a Playwright Page
    return page.evaluate(f"document.querySelector('{DIALOG} input[name=topic]:checked').value")


def listed(page: Any) -> list[str]:  # noqa: ANN401 - a Playwright Page
    """Return what the dialog says will be sent, a line each."""
    return page.locator("#feedback-sent li").all_text_contents()


def issue_fields(page: Any) -> dict[str, list[str]]:  # noqa: ANN401 - a Playwright Page
    """Return the fields the dialog's link to a new issue fills in, once its address is checked."""
    href = page.get_attribute("#feedback-issue", "href")
    link = urlsplit(href)
    assert (link.scheme, link.netloc, link.path) == ("https", "github.com", "/kgdunn/process-improve/issues/new")
    assert len(href) <= 8000
    return parse_qs(link.query)


def jpeg_size(jpeg: bytes) -> tuple[int, int]:
    """Read a JPEG's width and height from its start-of-frame marker."""
    at = 2
    while at < len(jpeg):
        marker, length = jpeg[at + 1], int.from_bytes(jpeg[at + 2 : at + 4], "big")
        if marker in {0xC0, 0xC1, 0xC2}:
            height, width = int.from_bytes(jpeg[at + 5 : at + 7], "big"), int.from_bytes(jpeg[at + 7 : at + 9], "big")
            return width, height
        at += 2 + length
    msg = "no start-of-frame marker"
    raise AssertionError(msg)


# --------------------------------------------------------------------------- what is sent
def test_the_header_s_button_sends_the_report_and_what_goes_with_it(page, relay, site, first_paint):
    open_feedback(page)
    assert topic(page) == "general"
    assert not page.is_checked("#feedback-study"), "the study goes only when the reader ticks it"
    send(page, email="reader@example.com")
    thanks = page.text_content(SENT).strip()
    assert thanks.startswith(f"Thank you: your report {REFERENCE} is on its way to the maintainer.")
    assert "A receipt is on its way to reader@example.com." in thanks
    [report] = relay.reports
    assert set(report) == {"topic", "message", "email", "screenshot", "context"}
    assert (report["topic"], report["message"], report["email"]) == ("general", MESSAGE, "reader@example.com")
    context = report["context"]
    assert (context["tab"], context["model"], context["window"]) == ("Learn", "As designed", "1280 x 900")
    assert context["charts"] == [
        "Coefficients",
        "Fitted response surface",
        "True response surface",
        "Residuals against fitted values",
        "How tightly to hold each factor",
    ]
    assert context["plotly_js"]
    assert "HeadlessChrome" in context["browser"]
    stamp = dict(row.split(": ", 1) for row in context["stamp"])
    assert stamp["With"] == f"process-improve {first_paint['version']}, in Factorial at {site}"
    assert stamp["Run order"] == "Drawn at random with seed 42."
    assert stamp["Results"].startswith("15 of 15 runs, simulated: ")
    assert re.fullmatch(r"[0-9a-f]{64}", stamp["Inputs, SHA-256"])
    assert not any("#c=" in row for row in context["stamp"]), "the stamp's link would carry the study"
    assert context["errors"] == []


def test_what_the_dialog_lists_is_what_is_sent(page, relay):
    open_feedback(page)
    page.fill("#feedback-message", MESSAGE)
    page.fill("#feedback-email", "reader@example.com")
    page.check("#feedback-study")
    shown = listed(page)
    assert shown[:3] == [
        "Topic: General",
        f"Your message: {len(MESSAGE)} characters",
        "Your email address, for a reply and a receipt: reader@example.com",
    ]
    assert shown[4] == "Your study, as a link that opens it with its results"
    assert "Tab on show: Learn" in shown
    page.click("#feedback-send")
    page.wait_for_selector(SENT, state="visible")
    [report] = relay.reports
    assert set(report) == {"topic", "message", "email", "screenshot", "study", "context"}
    assert [line for line in shown if line.startswith(("Run order: ", "Results: "))] == [
        line for line in report["context"]["stamp"] if line.startswith(("Run order: ", "Results: "))
    ]


def test_the_study_goes_when_ticked_as_a_link_that_opens_it(page, relay):
    open_feedback(page)
    page.check("#feedback-study")
    send(page)
    [report] = relay.reports
    assert PACKED.fullmatch(report["study"])
    opened = page.evaluate("async (packed) => (await import('./campaign.js')).unpackCampaign(packed)", report["study"])
    shown = page.evaluate("[...document.querySelectorAll('#design-table tbody tr')].length")
    assert len(opened["design"]["rows"]) == shown == 15
    assert opened["responses"], "with its results, so the result can be reproduced"


@pytest.mark.parametrize("opener", ["#learn-feedback", "footer [data-feedback]"])
def test_report_this_result_opens_on_the_algorithm_and_the_footer_on_general(page, relay, opener):
    open_feedback(page, opener)
    assert topic(page) == ("algorithm" if opener == "#learn-feedback" else "general")
    send(page)
    assert relay.reports[0]["topic"] == topic(page)


def test_an_analysis_that_fails_offers_report_this_result_with_its_error(page, relay):
    page.evaluate(
        "window.__pi.fake('api_analyze_runs', {ok: false, error: 'The fit failed, as a test makes it.', warnings: []})"
    )
    reveal(page, "#model")
    page.select_option("#model", "main_effects")
    page.wait_for_selector("#analysis p.error")
    settled(page)
    assert page.is_hidden("#learn-report"), "the report of the study waits for an analysis"
    open_feedback(page, "#analysis [data-feedback]")
    assert topic(page) == "algorithm"
    assert "Python error: api_analyze_runs: The fit failed, as a test makes it." in listed(page)
    send(page)
    assert relay.reports[0]["context"]["errors"] == ["api_analyze_runs: The fit failed, as a test makes it."]


def test_a_first_visit_has_no_study_to_send(fresh_page, relay):
    page = fresh_page
    open_feedback(page)
    assert page.is_disabled("#feedback-study")
    assert (
        page.text_content(f'{DIALOG} [data-note="study"]') == "There is no run sheet yet, so there is no study to send."
    )
    send(page)
    [report] = relay.reports
    assert "study" not in report
    assert report["context"]["tab"] == "Design"
    assert [row.split(":")[0] for row in report["context"]["stamp"]] == ["Made", "With", "On"]


def test_a_message_too_short_or_an_address_that_is_not_one_is_said_and_nothing_is_sent(page, relay):
    open_feedback(page)
    send(page, message="Too short")
    assert page.text_content(STATUS) == "Write at least 10 characters."
    assert page.evaluate("document.activeElement.id") == "feedback-message"
    send(page, email="Reader <reader@example.com>")
    assert page.text_content(STATUS) == "Give one email address, such as name@example.com, or leave the box empty."
    assert relay.reports == []
    page.fill("#feedback-email", "")
    page.click("#feedback-send")
    page.wait_for_selector(SENT, state="visible")
    assert len(relay.reports) == 1


def test_the_message_written_stays_until_it_is_sent(page, relay):
    open_feedback(page)
    page.fill("#feedback-message", MESSAGE)
    page.keyboard.press("Escape")
    assert page.is_hidden(DIALOG)
    open_feedback(page, "#learn-feedback")
    assert page.input_value("#feedback-message") == MESSAGE
    page.click("#feedback-send")
    page.wait_for_selector(SENT, state="visible")
    page.click(f"{SENT} [data-close]")
    open_feedback(page)
    assert page.input_value("#feedback-message") == "", "a report sent is not sent again"
    assert page.is_visible(WRITE)


# --------------------------------------------------------------------------- the screenshot
def test_the_screenshot_box_is_ticked_and_a_jpeg_of_the_tab_is_sent(page, relay):
    open_feedback(page)
    assert page.is_checked("#feedback-screenshot"), "ticked unless the reader unticks it"
    assert page.is_visible(
        f"{DIALOG} p.hint >> text=The screenshot shows what is on screen, including your study's data."
    )
    page.wait_for_selector(SHOT, state="visible", timeout=30_000)
    assert any(line.startswith("A screenshot of this tab: a JPEG of ") for line in listed(page))
    shown = page.get_attribute("#feedback-shot", "src")
    send(page)
    [report] = relay.reports
    assert report["screenshot"] == shown, "the screenshot the reader saw"
    assert report["screenshot"].startswith("data:image/jpeg;base64,")
    jpeg = base64.b64decode(report["screenshot"].split(",", 1)[1])
    assert jpeg[:3] == b"\xff\xd8\xff"
    assert len(jpeg) <= 2 * 2**20
    width, height = jpeg_size(jpeg)
    main = page.evaluate(
        "(({width, height}) => [width, height])(document.querySelector('main').getBoundingClientRect())"
    )
    assert width * height <= 4_000_000 * 1.01
    assert width / height == pytest.approx(main[0] / main[1], rel=0.02), "the whole of the page, its shape kept"


def test_untick_the_box_and_no_screenshot_is_sent(page, relay):
    open_feedback(page)
    page.uncheck("#feedback-screenshot")
    assert page.is_hidden("#feedback-shot")
    assert "No screenshot" in listed(page)
    send(page)
    assert "screenshot" not in relay.reports[0]


def test_a_screenshot_that_cannot_be_taken_is_said_and_the_report_goes_without_it(context, site, relay, console):
    console.allow(r"modern-screenshot|404")  # the browser reports the module it could not load
    context.route(re.compile(r"/vendor/modern-screenshot\.mjs$"), lambda route: route.fulfill(status=404, body=""))
    page = context.new_page()
    page.goto(f"{site}?example=worked")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    open_feedback(page)
    page.wait_for_function(
        f"document.querySelector('{DIALOG} [data-note=\"shot\"]').textContent.includes('could not be taken')"
    )
    send(page)
    assert "screenshot" not in relay.reports[0]


# --------------------------------------------------------------------------- when the report cannot go
@pytest.mark.parametrize(
    ("status", "why"),
    [
        (503, "The report could not be sent: the site's mail relay is busy or not running."),
        (429, "The report could not be sent: your network has sent as many reports as the site takes in an hour."),
        (None, "The report could not be sent: the page could not reach factori.al."),
    ],
)
def test_a_report_that_does_not_go_is_offered_the_other_ways(page, relay, console, status, why):
    """An issue, without the screenshot, the study or the address; the message stays, to send again."""
    console.allow(r"status of (429|503)|ERR_INTERNET_DISCONNECTED")  # the browser reports the post that failed
    relay.status = status
    open_feedback(page)
    page.check("#feedback-study")
    page.wait_for_selector(SHOT, state="visible", timeout=30_000)
    send(page, email="reader@example.com")
    assert {"screenshot", "study", "email"} <= set(relay.reports[0]), "the post went, with what the reader ticked"
    assert page.text_content(f'{FALLBACK} [data-note="why"]') == f"{why} Send it one of these ways instead:"
    fields = issue_fields(page)
    assert set(fields) == {"template", "title", "topic", "message", "context"}
    assert (fields["template"], fields["topic"], fields["message"]) == (["web_app.yml"], ["General"], [MESSAGE])
    assert "Tab on show: Learn" in fields["context"][0]
    href = page.get_attribute("#feedback-issue", "href")
    assert "reader%40example.com" not in href
    assert "data%3Aimage" not in href
    assert relay.reports[0]["study"] not in href
    assert page.get_attribute("#feedback-issue", "target") == "_blank"
    assert page.get_attribute("#feedback-mail", "href").startswith("mailto:")
    assert page.is_visible(WRITE)
    assert page.input_value("#feedback-message") == MESSAGE


def test_the_report_is_copied_as_text(context, page, relay, site, console):
    console.allow(r"status of 503")
    context.grant_permissions(["clipboard-read", "clipboard-write"], origin=site.rstrip("/"))
    relay.status = 503
    open_feedback(page)
    send(page, email="reader@example.com")
    page.click("#feedback-copy")
    page.wait_for_function(f"document.querySelector('{STATUS}').textContent.startsWith('The report is copied')")
    copied = page.evaluate("navigator.clipboard.readText()")
    assert copied.startswith(f"Topic: General\n\n{MESSAGE}\n\nWhat the page sent with it:\n")
    assert "  Tab on show: Learn\n" in copied
    assert "reader@example.com" not in copied


def test_a_copy_at_another_address_sends_nothing_itself(context, relay, plotly_js, catalog_files):
    """It never posts and takes no screenshot: a reader's report never reaches another server than factori.al's."""
    asked: list[str] = []

    def serve(route: Any) -> None:  # noqa: ANN401 - a Playwright Route
        name = urlsplit(route.request.url).path.lstrip("/") or "index.html"
        asked.append(name)
        built = catalog_files / name.removeprefix("catalog/")  # a copy is built as the app is, with its catalog
        if name == "wheels/manifest.json":
            route.fulfill(json={"plotly_js": plotly_js[0]})
        elif (WEB / name).is_file():
            route.fulfill(path=WEB / name)
        elif name.startswith("catalog/") and built.is_file():
            route.fulfill(path=built)
        else:
            route.fulfill(status=404, body="Not Found")

    context.route(f"{MIRROR}**", serve)
    page = context.new_page()
    page.goto(MIRROR)
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    open_feedback(page)
    reason = (
        "This copy of Factorial is not at factori.al, so it sends no report itself. Send it one of these ways instead:"
    )
    assert page.text_content(f'{FALLBACK} [data-note="why"]') == reason
    for hidden in ("#feedback-screenshot", "#feedback-study", "#feedback-email", "#feedback-send"):
        assert page.is_hidden(hidden), hidden
    page.fill("#feedback-message", MESSAGE)
    assert issue_fields(page)["message"] == [MESSAGE]
    page.click("#feedback-copy")
    page.wait_for_selector("#feedback-copied", state="visible")
    assert page.input_value("#feedback-copied").startswith(f"Topic: General\n\n{MESSAGE}")
    assert page.text_content(STATUS).startswith("The browser would not copy the report"), "no clipboard over http"
    assert relay.reports == []
    assert not [name for name in asked if name.startswith(("api/", "vendor/"))], "no post, and no screenshot"


# --------------------------------------------------------------------------- on a phone
@pytest.mark.parametrize(
    "context_options",
    [
        {"viewport": {"width": 390, "height": 844}, "is_mobile": True},
        {"viewport": {"width": 360, "height": 780}, "is_mobile": True},
    ],
    ids=["390", "360"],
)
def test_on_a_phone_the_button_is_its_icon_and_nothing_scrolls_sideways(page, relay, context_options):
    label = page.locator("#feedback span")
    assert page.evaluate("getComputedStyle(document.querySelector('#feedback span')).position") == "absolute"
    assert label.text_content() == "Feedback", "a screen reader still reads its name"
    assert overflowing(page) == []
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    open_feedback(page)
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    box = page.locator(DIALOG).bounding_box()
    assert box["x"] >= 0
    assert box["x"] + box["width"] <= context_options["viewport"]["width"]
