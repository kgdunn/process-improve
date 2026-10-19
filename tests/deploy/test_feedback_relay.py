"""``deploy/factorial/feedback_relay.py``: what factori.al does with a report a reader sends from the app.

Each test runs the relay on a free local port and sends it what a reader's browser, behind Caddy,
would send, or what anyone else could. The SMTP server is a stand-in that keeps each email it is
handed, as the wire carries it, so a test reads the headers and the attachment the maintainer
would get. The relay must email the maintainer and nobody else, keep nothing, and log nothing of
what a request held.
"""

from __future__ import annotations

import base64
import email
import email.policy
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "factorial" / "feedback_relay.py"


def load():
    spec = importlib.util.spec_from_file_location("feedback_relay", SCRIPT)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


relay = load()

MAIL = relay.Mail(
    "smtp.example.net", 587, "sender@factori.al", "the password", "feedback@factori.al", "kevin@example.org"
)
#: A JPEG's first and last bytes around nothing, as the page's capture would send one.
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + bytes(200) + b"\xff\xd9"
SHOT = relay.JPEG_URL + base64.b64encode(JPEG).decode()
PNG = b"\x89PNG\r\n\x1a\n" + bytes(40)
STUDY = "jZJNbsIwEIXv4rU3wLfYxRk1"
CONTEXT = {
    "commit": "9f3827d" * 5 + "abcde",
    "plotly_js": "3.1.0",
    "tab": "Learn",
    "model": "As designed",
    "charts": ["Coefficients", "Response surface"],
    "stamp": ["Made: 2026-10-18 21:04 UTC", "Inputs, SHA-256: 5e2a"],
    "errors": ["api_analyze_runs: Python stopped with LinAlgError: Singular matrix"],
    "browser": "Mozilla/5.0 (X11; Linux x86_64)",
    "window": "1280 x 900",
}
MESSAGE = "The contour plot of the reactor example shows a ridge, and the summary calls it a maximum."
READER = "reader@example.com"
ADDRESS = "203.0.113.7"
JSON = "application/json"
#: What Caddy sends with a report from the app.
HEADERS = {"Origin": "https://factori.al", "Content-Type": JSON, "X-Real-IP": ADDRESS}


def report(**changes: object) -> dict:
    """Return a whole report, as the page sends one, with ``changes``; a change to None leaves its key out."""
    whole = {"topic": "algorithm", "message": MESSAGE, "email": READER, "screenshot": SHOT, "study": STUDY}
    whole["context"] = CONTEXT
    whole.update(changes)
    return {key: value for key, value in whole.items() if value is not None}


class Outbox:
    """What the stand-in SMTP server was asked to do, each email it was handed, parsed back from its bytes, and the
    error it raises instead of sending, if a test gives it one.
    """

    def __init__(self) -> None:
        self.emails: list[Any] = []
        self.steps: list[tuple] = []
        self.fail: Exception | None = None


@pytest.fixture
def outbox(monkeypatch) -> Outbox:
    """Stand ``smtplib.SMTP`` in with a server that keeps each email it is handed."""
    box = Outbox()

    class SMTP:
        def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
            box.steps.append(("connect", host, port, timeout))

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            box.steps.append(("quit",))

        def starttls(self, context: object = None) -> None:
            box.steps.append(("starttls", context is not None))

        def login(self, user: str, password: str) -> None:
            box.steps.append(("login", user, password))

        def send_message(self, message: Any, from_addr: str, to_addrs: list[str]) -> None:  # noqa: ANN401
            if box.fail:
                raise box.fail
            box.steps.append(("send", from_addr, to_addrs))
            box.emails.append(email.message_from_bytes(message.as_bytes(), policy=email.policy.default))

    monkeypatch.setattr(relay.smtplib, "SMTP", SMTP)
    return box


class Clock:
    """A clock a test moves by hand, starting at noon (UTC) on 18 October 2026."""

    def __init__(self) -> None:
        self.now = 1_792_324_800.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def server(outbox, clock) -> Iterator[Any]:
    """Run the relay on a free local port for the test, its limits on the test's clock."""
    with relay.Server(("127.0.0.1", 0), MAIL, relay.Limits(clock=clock)) as running:
        thread = threading.Thread(target=running.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        yield running
        running.shutdown()


def send_raw(server, data: bytes, *, close: bool = True) -> tuple[int, dict[str, str], dict]:
    """Send ``data`` as it is, and return the reply's status, headers and JSON body."""
    with socket.create_connection(server.server_address, timeout=10) as connection:
        connection.sendall(data)
        if close:
            connection.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := connection.recv(2**16):
            received += chunk
    head, _, body = received.partition(b"\r\n\r\n")
    status_line, *lines = head.decode("latin-1").split("\r\n")
    headers = dict(line.split(": ", 1) for line in lines)
    return int(status_line.split()[1]), headers, json.loads(body) if headers.get("Content-Type") == JSON else {}


def ask(
    server,
    sent: dict | bytes,
    headers: dict[str, str | None] | None = None,
    *,
    path: str = "/api/feedback",
    method: str = "POST",
) -> tuple[int, dict[str, str], dict]:
    """Send a request as Caddy would, with ``headers`` changed (None takes one out); return the reply."""
    body = sent if isinstance(sent, bytes) else json.dumps(sent).encode()
    given = {"Host": "127.0.0.1", **HEADERS, "Content-Length": str(len(body)), **(headers or {})}
    head = "".join(f"{name}: {value}\r\n" for name, value in given.items() if value is not None)
    return send_raw(server, f"{method} {path} HTTP/1.1\r\n{head}\r\n".encode() + body)


# --------------------------------------------------------------------------- a report, emailed
def test_a_report_is_emailed_to_the_maintainer_alone(server, outbox):
    assert ask(server, report())[::2] == (200, {"ok": True})
    [mail] = outbox.emails
    assert mail["From"] == "Factorial feedback <feedback@factori.al>"
    assert mail["To"] == "kevin@example.org"
    assert mail["Subject"] == "[Factorial feedback: Algorithm or model]"
    assert mail["Reply-To"] == READER
    assert mail["Message-ID"].endswith("@factori.al>")
    assert outbox.steps[:3] == [("connect", "smtp.example.net", 587, relay.SMTP_SECONDS), ("starttls", True),
                                ("login", "sender@factori.al", "the password")]  # fmt: skip
    assert ("send", "feedback@factori.al", ["kevin@example.org"]) in outbox.steps, "the maintainer, and nobody else"
    text = mail.get_body(("plain",)).get_content()
    assert text.startswith(f"Topic: Algorithm or model\nReply to: {READER}\n\n{MESSAGE}\n")
    assert f"Study: https://factori.al/#c={STUDY}\n" in text
    assert "  App commit: " + CONTEXT["commit"] in text
    assert "  Charts on show: Coefficients; Response surface\n" in text
    assert "  Made: 2026-10-18 21:04 UTC\n" in text
    assert "    api_analyze_runs: Python stopped with LinAlgError: Singular matrix\n" in text
    [shot] = list(mail.iter_attachments())
    assert shot.get_content_type() == "image/jpeg"
    assert shot.get_filename() == "screenshot.jpg"
    assert shot.get_content() == JPEG


def test_a_report_of_a_message_alone_says_what_it_lacks(server, outbox):
    assert ask(server, {"topic": "general", "message": "  Ça marche: ∑ and é read well.  "})[0] == 200
    [mail] = outbox.emails
    assert mail["Subject"] == "[Factorial feedback: General]"
    assert mail["Reply-To"] is None
    assert not list(mail.iter_attachments())
    text = mail.get_body(("plain",)).get_content()
    assert "Reply to: no address given\n" in text
    assert "Ça marche: ∑ and é read well." in text, "the reader's text as written, in UTF-8"
    assert "Study: not included\nScreenshot: not included\n" in text
    assert text.endswith("Last Python errors: none\n")


def test_a_message_s_line_breaks_stay_in_the_body(server, outbox):
    """A header-injection attempt: the reader's text is the email's body, and never one of its headers."""
    attempt = "Hello\r\nBcc: victim@example.com\r\nSubject: You won\r\n\r\nand more"
    assert ask(server, report(message=attempt))[0] == 200
    [mail] = outbox.emails
    assert mail["Bcc"] is None
    assert mail.get_all("Subject") == ["[Factorial feedback: Algorithm or model]"]
    assert ("send", "feedback@factori.al", ["kevin@example.org"]) in outbox.steps
    assert "Bcc: victim@example.com" in mail.get_body(("plain",)).get_content()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("email", "reader@example.com\r\nBcc: victim@example.com"),
        ("email", "reader@example.com\nBcc: victim@example.com"),
        ("topic", "algorithm\r\nBcc: victim@example.com"),
        ("context", {**CONTEXT, "tab": "Learn\r\nBcc: victim@example.com"}),
    ],
)
def test_a_header_injection_attempt_in_a_one_line_field_is_refused(server, outbox, field, value):
    status, _, reply = ask(server, report(**{field: value}))
    assert status == 400, reply
    assert outbox.emails == []


# --------------------------------------------------------------------------- what the relay refuses before the body
@pytest.mark.parametrize(
    ("headers", "path", "refused"),
    [
        ({}, "/api/feedback/", (404, "only /api/feedback is answered here")),
        ({}, "/api/feedback?to=me", (404, "only /api/feedback is answered here")),
        ({}, "/", (404, "only /api/feedback is answered here")),
        ({"Origin": None}, "/api/feedback", (403, "reports are taken only from the app at https://factori.al")),
        ({"Origin": "https://mirror.example"}, "/api/feedback", (403, "only from the app at https://factori.al")),
        ({"Origin": "http://factori.al"}, "/api/feedback", (403, "only from the app at https://factori.al")),
        ({"X-Real-IP": None}, "/api/feedback", (403, "did not come through the site's web server")),
        ({"X-Real-IP": "localhost"}, "/api/feedback", (403, "did not come through the site's web server")),
        ({"Content-Type": None}, "/api/feedback", (415, "a report is sent as application/json")),
        ({"Content-Type": "text/plain"}, "/api/feedback", (415, "a report is sent as application/json")),
        ({"Content-Type": "application/x-www-form-urlencoded"}, "/api/feedback", (415, "sent as application/json")),
        ({"Transfer-Encoding": "chunked"}, "/api/feedback", (400, "a body sent in chunks is not taken")),
        ({"Content-Length": None}, "/api/feedback", (411, "the request has no Content-Length")),
        ({"Content-Length": "-1"}, "/api/feedback", (400, "Content-Length is not one number")),
        ({"Content-Length": "1e3"}, "/api/feedback", (400, "Content-Length is not one number")),
        ({"Content-Length": "²"}, "/api/feedback", (400, "Content-Length is not one number")),
    ],
    ids=lambda value: value if isinstance(value, str) and value.startswith("/") else None,
)
def test_a_request_that_is_not_the_app_s_report_is_refused_before_its_body_is_read(
    server, outbox, headers, path, refused
):
    status, reason = refused
    got, _, reply = ask(server, report(), headers, path=path)
    assert (got, reply) == (status, {"ok": False, "error": reply["error"]})
    assert reason in reply["error"]
    assert outbox.emails == []
    assert server.limits.today == 0, "a request refused by its headers does not count against the network"


def test_two_content_lengths_are_refused(server, outbox):
    body = json.dumps(report()).encode()
    head = "".join(f"{name}: {value}\r\n" for name, value in HEADERS.items())
    data = f"POST /api/feedback HTTP/1.1\r\n{head}Content-Length: {len(body)}\r\nContent-Length: 5\r\n\r\n"
    assert send_raw(server, data.encode() + body)[0] == 400
    assert outbox.emails == []


def test_a_body_larger_than_a_report_is_refused_unread(server, outbox):
    """The relay answers at once, before the 4 MB Caddy would let through could arrive."""
    head = "".join(f"{name}: {value}\r\n" for name, value in HEADERS.items())
    started = time.monotonic()
    data = f"POST /api/feedback HTTP/1.1\r\n{head}Content-Length: {relay.MAX_BODY + 1}\r\n\r\n".encode()
    status, _, reply = send_raw(server, data)
    assert status == 413
    assert reply["error"] == f"a report holds at most {relay.MAX_BODY // 2**10} KiB"
    assert time.monotonic() - started < 2
    assert outbox.emails == []


def test_a_body_shorter_than_its_length_is_refused(server, outbox):
    body = json.dumps(report()).encode()
    head = "".join(f"{name}: {value}\r\n" for name, value in HEADERS.items())
    data = f"POST /api/feedback HTTP/1.1\r\n{head}Content-Length: {len(body) + 10}\r\n\r\n".encode() + body
    status, _, reply = send_raw(server, data)
    assert (status, reply["error"]) == (400, "the body is shorter than its Content-Length")
    assert outbox.emails == []


def test_a_body_that_arrives_too_slowly_is_refused_at_its_deadline(server, outbox, monkeypatch):
    monkeypatch.setattr(relay, "BODY_SECONDS", 0.4)
    body = json.dumps(report()).encode()
    head = "".join(f"{name}: {value}\r\n" for name, value in HEADERS.items())
    started = time.monotonic()
    data = f"POST /api/feedback HTTP/1.1\r\n{head}Content-Length: {len(body)}\r\n\r\n".encode() + body[:20]
    status, _, reply = send_raw(server, data, close=False)  # the rest never comes, and the connection stays open
    assert (status, reply["error"]) == (408, "the body took too long to arrive")
    assert 0.3 < time.monotonic() - started < 5
    assert outbox.emails == []


def test_only_post_is_answered(server, outbox):
    assert send_raw(server, b"GET /api/feedback HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")[0] == 501
    assert outbox.emails == []


# --------------------------------------------------------------------------- what the relay refuses in the body
@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (b"not json", "the body is not JSON"),
        (b"\xff\xfe{}", "the body is not JSON"),
        (b"[" * 100_000 + b"]" * 100_000, "the body is not JSON"),
        (b'{"topic": "algorithm", "message": ' + b"9" * 5000 + b"}", "the body is not JSON"),
        (b"[]", "the body is not a JSON object"),
        (b'"a report"', "the body is not a JSON object"),
        (report(admin=True), "the report holds a key the relay does not know"),
        (report(to="victim@example.com"), "the report holds a key the relay does not know"),
        (report(message=None), "a report needs its topic and its message"),
        (report(topic=None), "a report needs its topic and its message"),
        (report(topic="bug"), "the topic is not one of algorithm, interface, general"),
        (report(topic=["algorithm"]), "the topic is not one of"),
        (report(message=42), "the message is not text"),
        (report(message="Too short"), "the message is shorter than 10 characters"),
        (report(message="    short    "), "the message is shorter than 10 characters"),
        (report(message="x" * 5001), "the message is longer than 5000 characters"),
        (report(message="A message with \x00 in it"), "the message holds a control character"),
        (report(message="half a pair: \ud800, then more"), "the message is not valid text"),
        (report(email=5), "the reply address is not text"),
        (report(email=""), "the reply address is not one email address"),
        (report(email="reader@"), "the reply address is not one email address"),
        (report(email="reader@[127.0.0"), "the reply address is not one email address"),
        (report(email="reader@example@com"), "the reply address is not one email address"),
        (report(email='"reader@example.com'), "the reply address is not one email address"),
        (report(email='"quoted reader"@example.com'), "the reply address is not one email address"),
        (report(email="not an address"), "the reply address is not one email address"),
        (report(email="one@example.com, two@example.com"), "the reply address is not one email address"),
        (report(email="Reader <reader@example.com>"), "the reply address is not one email address"),
        (report(email="reader@example.com (me)"), "the reply address is not one email address"),
        (report(email=" reader@example.com"), "the reply address is not one email address"),
        (report(email="reader@localhost"), "the reply address is not one email address"),
        (report(email="reader@[127.0.0.1]"), "the reply address is not one email address"),
        (report(email="réader@example.com"), "the reply address is not one email address"),
        (report(email="r" * 250 + "@example.com"), "the reply address is longer than 254 characters"),
        (report(screenshot=42), "the screenshot is not a JPEG data: URL"),
        (report(screenshot="https://example.com/a.jpg"), "the screenshot is not a JPEG data: URL"),
        (report(screenshot="data:image/png;base64," + base64.b64encode(PNG).decode()), "not a JPEG data: URL"),
        (report(screenshot=relay.JPEG_URL + "not base64!"), "the screenshot is not base64"),
        (report(screenshot=relay.JPEG_URL + base64.b64encode(PNG).decode()), "the screenshot is not a JPEG"),
        (report(study=""), "the study is not a study's packed text"),
        (report(study="https://evil.example/#c=x"), "the study is not a study's packed text"),
        (report(study="abc def"), "the study is not a study's packed text"),
        (report(study="a" * (relay.MAX_STUDY + 1)), "the study is longer than 262144 characters"),
        (report(context=[]), "the context is not a JSON object"),
        (report(context={**CONTEXT, "cookie": "x"}), "the context holds a key the relay does not know"),
        (report(context={"charts": "Coefficients"}), "the context's charts is not a list of at most 30 texts"),
        (report(context={"errors": ["one"] * 6}), "the context's errors is not a list of at most 5 texts"),
        (report(context={"errors": ["x" * 301]}), "the context's errors is longer than 300 characters"),
        (report(context={"tab": 3}), "the context's tab is not text"),
        (report(context={"browser": "x" * 401}), "the context's browser is longer than 400 characters"),
        (report(context={"model": "As designed\tor not"}), "the context's model holds a control character"),
    ],
)
def test_a_report_that_is_not_one_is_refused(server, outbox, body, reason):
    status, _, reply = ask(server, body)
    assert (status, reply) == (400, {"ok": False, "error": reply["error"]})
    assert reason in reply["error"]
    assert outbox.emails == []


def test_a_screenshot_larger_than_its_limit_is_refused(server, outbox, monkeypatch):
    monkeypatch.setattr(relay, "MAX_SCREENSHOT", len(JPEG) - 1)
    status, _, reply = ask(server, report())
    assert (status, reply["error"]) == (400, "the screenshot is larger than 0 MiB")
    monkeypatch.setattr(relay, "MAX_SCREENSHOT", 10)
    assert ask(server, report())[2]["error"] == "the screenshot is larger than 0 MiB", "refused before it is decoded"
    assert outbox.emails == []


def test_a_refusal_repeats_nothing_the_request_held(server):
    """The reason goes back to the page and into the log, so it is always the relay's own words."""
    sent = report(topic="<script>", email="<script>@example.com")
    for body in (sent, report(email="<script>@example.com"), report(study="<script>")):
        assert "script" not in ask(server, body)[2]["error"]


# --------------------------------------------------------------------------- the limits
def test_a_network_sends_ten_reports_an_hour(server, outbox, clock):
    for n in range(relay.PER_NETWORK):
        assert ask(server, report(), {"X-Real-IP": f"203.0.113.{n + 1}"})[0] == 200
    status, headers, reply = ask(server, report(), {"X-Real-IP": "203.0.113.250"})
    assert (status, headers["Retry-After"]) == (429, "3600"), "the same /24"
    assert reply["error"] == "a network sends at most 10 reports an hour: try again later"
    assert ask(server, report(), {"X-Real-IP": "198.51.100.7"})[0] == 200, "another network"
    clock.now += relay.HOUR
    assert ask(server, report())[0] == 200, "an hour later"
    assert len(outbox.emails) == relay.PER_NETWORK + 2


@pytest.mark.parametrize(
    ("one", "other", "same"),
    [
        ("203.0.113.7", "203.0.113.200", True),
        ("203.0.113.7", "203.0.114.7", False),
        ("2001:db8:1::1", "2001:db8:1:ffff::9", True),
        ("2001:db8:1::1", "2001:db8:2::1", False),
        ("::ffff:203.0.113.9", "203.0.113.7", True),
        ("::ffff:198.51.100.1", "203.0.113.7", False),
    ],
)
def test_a_network_is_a_24_of_ipv4_or_a_48_of_ipv6(one, other, same):
    assert (relay.network(one) == relay.network(other)) is same
    assert relay.network("::ffff:203.0.113.9") == "203.0.113.0/24", "an IPv4 address written as IPv6 is IPv4"
    assert relay.network("2001:db8:1:2:3::4") == "2001:db8:1::/48"


def test_the_count_is_kept_for_at_most_so_many_networks(clock, monkeypatch):
    monkeypatch.setattr(relay, "NETWORKS", 3)
    limits = relay.Limits(clock=clock)
    for n in range(3):
        assert limits.admit(f"net-{n}") is None
    assert limits.admit("net-3") == "too many networks are sending reports: try again later"
    clock.now += relay.HOUR / 2
    assert limits.admit("net-0") is None, "a network counted already still sends"
    clock.now += relay.HOUR / 2 - 1
    assert limits.admit("net-3") is not None, "every network counted sent a report within the hour"
    clock.now += 2
    assert limits.admit("net-3") is None
    assert set(limits.sent) == {"net-0", "net-3"}, "the networks quiet for an hour are forgotten"


def test_everyone_together_sends_so_many_reports_a_day(clock, monkeypatch):
    monkeypatch.setattr(relay, "PER_DAY", 3)
    limits = relay.Limits(clock=clock)
    assert [limits.admit(f"net-{n}") for n in range(4)][-1] == (
        "the relay has sent all the reports it sends in a day: try again tomorrow"
    )
    clock.now += 12 * 3600  # midnight, UTC
    assert limits.admit("net-4") is None


def test_a_ninth_request_in_flight_is_refused(server, outbox):
    for _ in range(relay.IN_FLIGHT):
        assert server.slots.acquire(blocking=False)
    status, headers, reply = ask(server, report())
    assert (status, headers["Retry-After"], reply["error"]) == (503, "60", "the relay is busy: try again in a minute")
    server.slots.release()
    assert ask(server, report())[0] == 200
    assert len(outbox.emails) == 1


# --------------------------------------------------------------------------- when the email cannot go
def test_an_smtp_failure_is_a_bad_gateway(server, outbox, capsys):
    outbox.fail = relay.smtplib.SMTPRecipientsRefused({"kevin@example.org": (550, b"no such user")})
    status, _, reply = ask(server, report())
    assert (status, reply["error"]) == (502, "the email could not be sent")
    assert capsys.readouterr().err == "502 the email could not be sent (SMTPRecipientsRefused)\n"


def test_nothing_a_request_held_is_logged(server, outbox, capsys):
    """The journal gets one line per request, its status and the relay's own reason: never what the reader sent."""
    words = "a message nobody but the maintainer reads"
    sent = report(message=words, email="private.reader@example.com", study="Zm9vYmFyc2VjcmV0")
    assert ask(server, sent)[0] == 200
    assert ask(server, report(message=words, email="not one address"), {"X-Real-IP": "198.51.100.77"})[0] == 400
    outbox.fail = OSError("Connection refused by smtp.example.net for private.reader@example.com")
    assert ask(server, sent, {"X-Real-IP": "192.0.2.99"})[0] == 502
    assert send_raw(server, b"GET /?private.reader HTTP/1.1\r\n\r\n")[0] == 501
    logged = capsys.readouterr()
    assert logged.out == ""
    assert logged.err.splitlines() == [
        "200 sent",
        "400 the reply address is not one email address",
        "502 the email could not be sent (OSError)",
    ]
    for value in (words, "private.reader", "Zm9vYmFyc2VjcmV0", ADDRESS, "198.51.100", "192.0.2"):
        assert value not in logged.err


# --------------------------------------------------------------------------- its settings, and running it
ENVIRONMENT = {
    "SMTP_HOST": "smtp.example.net",
    "SMTP_USER": "sender@factori.al",
    "SMTP_FROM_EMAIL": "feedback@factori.al",
    "FEEDBACK_TO": "kevin@example.org",
}


def test_the_settings_come_from_the_environment_and_the_password_from_its_credential(tmp_path):
    (tmp_path / "smtp-password").write_text("the password\n", encoding="utf-8")
    mail = relay.settings({**ENVIRONMENT, "CREDENTIALS_DIRECTORY": str(tmp_path)})
    assert mail == MAIL, "port 587 unless SMTP_PORT says otherwise, and the password without its line break"
    assert relay.settings({**ENVIRONMENT, "SMTP_USER": "", "SMTP_PORT": "2525"}).port == 2525, "no login, no password"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"FEEDBACK_TO": ""}, "FEEDBACK_TO must be set"),
        ({"SMTP_HOST": "", "SMTP_FROM_EMAIL": ""}, "SMTP_HOST and SMTP_FROM_EMAIL must be set"),
        ({"FEEDBACK_TO": "kevin@example.org, someone@example.org"}, "FEEDBACK_TO is not one email address"),
        ({"SMTP_FROM_EMAIL": "Factorial <feedback@factori.al>"}, "SMTP_FROM_EMAIL is not one email address"),
        ({"SMTP_PORT": "smtp"}, "SMTP_PORT is not a port number"),
        ({}, "SMTP_USER is set, but the credential smtp-password"),
    ],
)
def test_a_setting_missing_or_wrong_is_named(changes, reason):
    with pytest.raises(ValueError, match=reason):
        relay.settings({**ENVIRONMENT, **changes})


def test_the_relay_runs_as_systemd_runs_it_and_stops_at_a_missing_setting(tmp_path):
    """Under ``python3 -I``, as its first line asks: nothing from the checkout is imported."""
    environment = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    stopped = subprocess.run(  # noqa: S603 - the script under test, with a fixed argument list
        [sys.executable, "-I", str(SCRIPT)], env=environment, capture_output=True, text=True, check=False, timeout=30
    )
    assert stopped.returncode == 1
    assert stopped.stderr == (
        "factorial-feedback: SMTP_HOST, SMTP_FROM_EMAIL and FEEDBACK_TO must be set, in /etc/factorial/feedback.env\n"
    )
    assert relay.main(["--port", "1"]) == 2


def test_the_relay_listens_on_its_address_with_the_settings_in_the_environment(monkeypatch, tmp_path, capsys):
    """As systemd starts it: the settings in the environment, the password in the credential, no argument."""
    started = []

    class Listening:
        def __init__(self, address: tuple[str, int], mail: Any) -> None:  # noqa: ANN401 - the relay's Mail
            started.append((address, mail))

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def serve_forever(self) -> None:
            pass

    (tmp_path / "smtp-password").write_text("the password\n", encoding="utf-8")
    for name, value in {**ENVIRONMENT, "CREDENTIALS_DIRECTORY": str(tmp_path)}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.setattr(relay, "Server", Listening)
    assert relay.main([]) == 0
    assert started == [(("127.0.0.1", 8790), MAIL)]
    assert capsys.readouterr().err == "taking reports on 127.0.0.1:8790/api/feedback\n"


# --------------------------------------------------------------------------- how the server runs it
DEPLOY = SCRIPT.parent


def test_caddy_passes_the_relay_its_path_on_its_address_with_the_reader_s_address():
    caddy = (DEPLOY / "factorial.caddy").read_text(encoding="utf-8")
    host, port = relay.ADDRESS
    block = re.search(rf"handle {re.escape(relay.PATH)} \{{(.+?)\n\t\}}", caddy, flags=re.DOTALL)[1]
    assert f"reverse_proxy {host}:{port} {{" in block
    assert "header_up X-Real-IP {remote_host}" in block, "set by Caddy, whatever the request says"
    capped = int(re.search(r"max_size (\d+)MB", block)[1]) * 10**6
    assert capped > relay.MAX_BODY, "Caddy lets through every report the relay takes"
    assert f"request_buffers {capped // 10**6}MB" in block


def test_systemd_runs_the_copy_setup_installs_with_its_settings_and_its_password():
    unit = (DEPLOY / "factorial-feedback.service").read_text(encoding="utf-8")
    setup = (DEPLOY / "setup.sh").read_text(encoding="utf-8")
    assert '"$here/feedback_relay.py" /usr/local/bin/factorial-feedback' in setup
    assert "\nExecStart=/usr/local/bin/factorial-feedback\n" in unit
    assert "\nEnvironmentFile=/etc/factorial/feedback.env\n" in unit
    assert "\nLoadCredential=smtp-password:/etc/factorial/smtp-password\n" in unit
    assert "[ -e /etc/factorial/feedback.env ] ||" in setup, "the settings are made once, and kept after"
    for line in ("DynamicUser=yes", "ProtectSystem=strict", "NoNewPrivileges=yes", "TasksMax=32", "MemoryMax=128M"):
        assert f"\n{line}\n" in unit, line


def test_the_settings_file_names_what_the_relay_reads(tmp_path):
    example = (DEPLOY / "feedback.env.example").read_text(encoding="utf-8")
    names = re.findall(r"^([A-Z_]+)=", example, flags=re.MULTILINE)
    assert names == ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_FROM_EMAIL", "FEEDBACK_TO"]
    (tmp_path / "smtp-password").write_text("the password", encoding="utf-8")
    filled = {**dict.fromkeys(names, ""), **ENVIRONMENT, "SMTP_PORT": "587", "CREDENTIALS_DIRECTORY": str(tmp_path)}
    assert relay.settings(filled) == MAIL, "filled in, it is all the relay needs"
