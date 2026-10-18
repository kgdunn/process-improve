#!/usr/bin/python3 -I
"""factori.al's feedback relay: it takes a report a reader sends from the app, and emails it to the maintainer.

Caddy passes it ``POST /api/feedback`` (``factorial.caddy``): the whole body, which Caddy holds
first and caps at 4 MB, with the reader's address in ``X-Real-IP``. The relay listens on
``127.0.0.1:8790`` and answers one request per connection. A report is JSON with these keys and
no others:

    topic       "algorithm", "interface" or "general"
    message     what the reader wrote: 10 to 5000 characters
    email       optional: one address to reply to, with no name
    screenshot  optional: a JPEG of the tab, as a data: URL, at most 2 MiB
    study       optional: the study, packed as a link to it carries it after "#c="
    context     what the page adds: its build, a report's stamp, the tab and the charts on show,
                and the last Python errors

It sends one plain-text email to ``FEEDBACK_TO``, through SMTP with STARTTLS, and keeps nothing:
no copy, no queue, and no log line that holds anything a request sent. The reader's text goes in
the body only: the subject names the topic from a fixed list, and ``Reply-To`` is an address that
parsed as one. Nothing a request says can name a recipient, so the relay mails the maintainer
and nobody else.

Each network (a /24 of IPv4, a /48 of IPv6) may send ``PER_NETWORK`` reports an hour, and
everyone together ``PER_DAY`` a day, counted in memory: the relay is one process. At most
``IN_FLIGHT`` requests are worked on at once. Standard library only.

Its settings come from the environment, which systemd reads from ``/etc/factorial/feedback.env``:
``SMTP_HOST``, ``SMTP_PORT`` (587 if unset), ``SMTP_USER``, ``SMTP_FROM_EMAIL`` and
``FEEDBACK_TO``. The SMTP password is the credential ``smtp-password``
(``LoadCredential=`` in ``factorial-feedback.service``).
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import email.utils
import http.server
import ipaddress
import json
import os
import re
import smtplib
import socket
import ssl
import sys
import threading
import time
from dataclasses import dataclass
from email.errors import MessageError
from email.headerregistry import Address
from email.message import EmailMessage
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

#: Where the relay listens: Caddy, on the same machine, is the only one that reaches it.
ADDRESS = ("127.0.0.1", 8790)
#: The one path answered, and the one page a report may come from: the app at factori.al.
PATH = "/api/feedback"
ORIGIN = "https://factori.al"
#: The app's address before a study's packed text: the link that opens the reader's study.
STUDY_LINK = "https://factori.al/#c="
#: The topics, by the name a report gives, as the page lists them (``web/feedback.js``).
TOPICS = {"algorithm": "Algorithm or model", "interface": "Interface suggestion", "general": "General"}
#: A message's length, in characters: at least this many once its spaces are trimmed, at most that many.
MESSAGE_MIN = 10
MESSAGE_MAX = 5000
#: The longest reply address SMTP carries.
EMAIL_MAX = 254
#: The most a screenshot may hold once decoded, and the most characters a study's packed text may run to.
MAX_SCREENSHOT = 2 * 2**20
MAX_STUDY = 2**18
#: What the page sends of itself with a report: each key with the most characters its text may hold, or,
#: for a list, the most texts it may hold and the most characters each. ``web/feedback.js`` keeps the same.
CONTEXT: dict[str, int | tuple[int, int]] = {
    "commit": 64,
    "plotly_js": 40,
    "tab": 40,
    "model": 200,
    "charts": (30, 200),
    "stamp": (12, 400),
    "errors": (5, 300),
    "browser": 400,
    "window": 40,
}
#: How the email names each of the context's texts, in the order it lists them; the stamp's rows name themselves.
CONTEXT_LABELS = {
    "commit": "App commit",
    "plotly_js": "Plotly.js",
    "tab": "Tab on show",
    "model": "Model",
    "charts": "Charts on show",
    "browser": "Browser",
    "window": "Window",
}
#: Every key a report may hold, and those it must.
KEYS = frozenset({"topic", "message", "email", "screenshot", "study", "context"})
REQUIRED = frozenset({"topic", "message"})
#: The most a request's body may hold: a screenshot and a study at their largest, and the rest. Caddy refuses a
#: body of more than 4 MB before it gets here.
MAX_BODY = 3 * 2**20 + 2**19
#: How long, in seconds, the request's line and headers may pause, its whole body may take to arrive, the reply may
#: take to write, and the SMTP server may take to answer.
HEADER_SECONDS = 10
BODY_SECONDS = 20
REPLY_SECONDS = 10
SMTP_SECONDS = 30
#: After its reply, how long a connection waits for the client to finish sending, and how much it reads and drops.
LINGER_SECONDS = 2
LINGER_BYTES = 2**23
#: How many reports one network may send in an hour, and everyone together in a day (UTC); how many networks the
#: count holds at once.
PER_NETWORK = 10
HOUR = 3600
PER_DAY = 200
NETWORKS = 10_000
#: How many requests are worked on at once; one more is refused, to try again a minute later.
IN_FLIGHT = 8
#: A screenshot as the page sends it: a JPEG, whose bytes start with these, in a data: URL that starts with this.
JPEG = b"\xff\xd8\xff"
JPEG_URL = "data:image/jpeg;base64,"
#: A study's packed text: URL-safe base64, as the page packs a link.
PACKED = re.compile(r"[A-Za-z0-9_-]+")
#: A reply address as the page takes one (``web/feedback.js``): a dot-atom name, an at sign, and a domain of
#: labels joined by dots, in ASCII. A quoted name, a comment or an address literal is refused.
ATEXT = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
ADDRESS = re.compile(rf"{ATEXT}(?:\.{ATEXT})*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
#: The control characters no text may hold, and the line breaks and tabs only a message may hold.
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LINE_BREAK = re.compile(r"[\t\n\r]")
#: The name the email comes from, beside the site's sending address.
SENDER_NAME = "Factorial feedback"


class RefusedError(Exception):
    """A request the relay does not carry out: the HTTP status to answer with, and why.

    The reason goes back to the page and into the log, so it is always the relay's own words: it never
    repeats anything the request held.
    """

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status


def log(line: str) -> None:
    """Write one line to the journal (systemd keeps what the relay writes to stderr)."""
    print(line, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- the report
@dataclass(frozen=True)
class Report:
    """A report as the relay has checked it: the reader's words, and what the page sent with them."""

    topic: str
    message: str
    email: str | None = None
    screenshot: bytes | None = None
    study: str | None = None
    context: dict[str, str | list[str]] | None = None


def text(value: object, name: str, most: int, *, lines: bool = False) -> str:
    """Return ``value`` if it is text of at most ``most`` characters, with no control character in it.

    Only a text that ``lines`` allows may hold line breaks and tabs. Text that UTF-8 cannot write, such as half
    of a surrogate pair, is refused too, since the email could not carry it.
    """
    if not isinstance(value, str):
        raise RefusedError(400, f"the {name} is not text")
    if len(value) > most:
        raise RefusedError(400, f"the {name} is longer than {most} characters")
    if CONTROL.search(value) or (not lines and LINE_BREAK.search(value)):
        raise RefusedError(400, f"the {name} holds a control character")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RefusedError(400, f"the {name} is not valid text") from error
    return value


def reply_address(value: object) -> str:
    """Return ``value`` if it is one email address and nothing else: no name, no comment, no second address.

    It must have the plain form the page checks for (``ADDRESS``) before the email package parses it: on some
    Pythons the parser raises IndexError or AttributeError on a malformed address, not a ValueError, so a form it
    may stumble on never reaches it. Whatever it raises, the address is refused.
    """
    address = text(value, "reply address", EMAIL_MAX)
    refused = RefusedError(400, "the reply address is not one email address")
    if not ADDRESS.fullmatch(address):
        raise refused
    try:
        parsed = Address(addr_spec=address)
    except (ValueError, IndexError, TypeError, AttributeError, MessageError) as error:
        raise refused from error
    if parsed.addr_spec != address or "." not in parsed.domain:
        raise refused
    return address


def screenshot(value: object) -> bytes:
    """Return the JPEG a screenshot's data: URL holds, if it is one, of at most ``MAX_SCREENSHOT`` bytes."""
    if not isinstance(value, str) or not value.startswith(JPEG_URL):
        raise RefusedError(400, "the screenshot is not a JPEG data: URL")
    encoded = value[len(JPEG_URL) :]
    if len(encoded) > 4 * -(-MAX_SCREENSHOT // 3):
        raise RefusedError(400, f"the screenshot is larger than {MAX_SCREENSHOT // 2**20} MiB")
    try:
        image = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RefusedError(400, "the screenshot is not base64") from error
    if len(image) > MAX_SCREENSHOT:
        raise RefusedError(400, f"the screenshot is larger than {MAX_SCREENSHOT // 2**20} MiB")
    if not image.startswith(JPEG):
        raise RefusedError(400, "the screenshot is not a JPEG")
    return image


def study(value: object) -> str:
    """Return a study's packed text, if it is the URL-safe base64 a link to a study carries after its ``#c=``."""
    packed = text(value, "study", MAX_STUDY)
    if not PACKED.fullmatch(packed):
        raise RefusedError(400, "the study is not a study's packed text")
    return packed


def context(value: object) -> dict[str, str | list[str]]:
    """Return what the page sent of itself, if each key is one ``CONTEXT`` names and each value fits its limit."""
    if not isinstance(value, dict):
        raise RefusedError(400, "the context is not a JSON object")
    if not set(value) <= set(CONTEXT):
        raise RefusedError(400, "the context holds a key the relay does not know")
    checked: dict[str, str | list[str]] = {}
    for key, given in value.items():
        limit = CONTEXT[key]
        if isinstance(limit, tuple):
            count, most = limit
            if not isinstance(given, list) or len(given) > count:
                raise RefusedError(400, f"the context's {key} is not a list of at most {count} texts")
            checked[key] = [text(item, f"context's {key}", most) for item in given]
        else:
            checked[key] = text(given, f"context's {key}", limit)
    return checked


def read_report(body: bytes) -> Report:
    """Check a request's body, and return the report it holds; refuse anything else."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, RecursionError) as error:  # RecursionError: arrays nested past Python's stack
        raise RefusedError(400, "the body is not JSON") from error
    if not isinstance(data, dict):
        raise RefusedError(400, "the body is not a JSON object")
    if not set(data) <= KEYS:
        raise RefusedError(400, "the report holds a key the relay does not know")
    if REQUIRED - set(data):
        raise RefusedError(400, "a report needs its topic and its message")
    if not isinstance(data["topic"], str) or data["topic"] not in TOPICS:
        raise RefusedError(400, f"the topic is not one of {', '.join(TOPICS)}")
    message = text(data["message"], "message", MESSAGE_MAX, lines=True)
    if len(message.strip()) < MESSAGE_MIN:
        raise RefusedError(400, f"the message is shorter than {MESSAGE_MIN} characters")
    return Report(
        topic=data["topic"],
        message=message,
        email=reply_address(data["email"]) if "email" in data else None,
        screenshot=screenshot(data["screenshot"]) if "screenshot" in data else None,
        study=study(data["study"]) if "study" in data else None,
        context=context(data["context"]) if "context" in data else None,
    )


# --------------------------------------------------------------------------- the email
@dataclass(frozen=True)
class Mail:
    """Where the email goes and how: the SMTP server and its login, the site's sending address, the maintainer's."""

    host: str
    port: int
    user: str
    password: str
    sender: str
    to: str


def body(report: Report) -> str:
    """Return the email's text: the topic and who to reply to, the reader's message, then what came with it."""
    about = report.context or {}
    reply = report.email or "no address given"
    lines = [f"Topic: {TOPICS[report.topic]}", f"Reply to: {reply}", "", report.message, ""]
    lines.append(f"Study: {STUDY_LINK}{report.study}" if report.study else "Study: not included")
    lines.append("Screenshot: attached, as screenshot.jpg" if report.screenshot else "Screenshot: not included")
    lines += ["", "What the page sent with it:"]
    for key, label in CONTEXT_LABELS.items():
        if key in about:
            value = about[key]
            lines.append(f"  {label}: {'; '.join(value) if isinstance(value, list) else value}")
    lines += [f"  {row}" for row in about.get("stamp", [])]
    errors = about.get("errors", [])
    lines.append("  Last Python errors:" if errors else "  Last Python errors: none")
    lines += [f"    {error}" for error in errors]
    return "\n".join(lines) + "\n"


def compose(report: Report, mail: Mail) -> EmailMessage:
    """Write one report's email: plain text to the maintainer, the screenshot attached as ``screenshot.jpg``."""
    message = EmailMessage()
    message["From"] = Address(SENDER_NAME, addr_spec=mail.sender)
    message["To"] = mail.to
    message["Subject"] = f"[Factorial feedback: {TOPICS[report.topic]}]"
    if report.email:
        message["Reply-To"] = Address(addr_spec=report.email)
    message["Date"] = email.utils.formatdate(usegmt=True)
    message["Message-ID"] = email.utils.make_msgid(domain=mail.sender.rpartition("@")[2])
    message.set_content(body(report))
    if report.screenshot is not None:
        message.add_attachment(report.screenshot, maintype="image", subtype="jpeg", filename="screenshot.jpg")
    return message


def send(message: EmailMessage, mail: Mail) -> None:
    """Hand the email to the SMTP server over STARTTLS, addressed to the maintainer whatever its headers say."""
    with smtplib.SMTP(mail.host, mail.port, timeout=SMTP_SECONDS) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        if mail.user:
            smtp.login(mail.user, mail.password)
        smtp.send_message(message, from_addr=mail.sender, to_addrs=[mail.to])


# --------------------------------------------------------------------------- the limits
def network(address: str) -> str:
    """Return the network a reader's address is counted in: its /24 for IPv4, its /48 for IPv6.

    An IPv4 address written as IPv6 (``::ffff:203.0.113.7``) is counted as the IPv4 address it is.
    """
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    prefix = 24 if ip.version == 4 else 48
    return str(ipaddress.ip_network(f"{ip}/{prefix}", strict=False))


class Limits:
    """How many reports each network sent in the last hour, and everyone today, held in memory.

    The table holds at most ``NETWORKS`` networks. When it is full, a network with no report in the
    last hour is forgotten to make room; when none is, a new network is refused until one is.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self.lock = threading.Lock()
        self.sent: dict[str, list[float]] = {}  # each network's reports within the hour, by when they came
        self.day = ""  # the day (UTC) that ``today`` counts
        self.today = 0

    def admit(self, where: str) -> str | None:
        """Count one report from the network ``where``; return why it is refused, or None to go ahead."""
        now = self.clock()
        with self.lock:
            day = time.strftime("%Y-%m-%d", time.gmtime(now))
            if day != self.day:
                self.day, self.today = day, 0
            if self.today >= PER_DAY:
                return "the relay has sent all the reports it sends in a day: try again tomorrow"
            recent = [when for when in self.sent.get(where, []) if now - when < HOUR]
            if len(recent) >= PER_NETWORK:
                self.sent[where] = recent
                return f"a network sends at most {PER_NETWORK} reports an hour: try again later"
            if where not in self.sent and len(self.sent) >= NETWORKS:
                self.prune(now)
                if len(self.sent) >= NETWORKS:
                    return "too many networks are sending reports: try again later"
            self.sent[where] = [*recent, now]
            self.today += 1
            return None

    def prune(self, now: float) -> None:
        """Forget every network whose reports are all over an hour old."""
        for where in [where for where, times in self.sent.items() if all(now - when >= HOUR for when in times)]:
            del self.sent[where]


# --------------------------------------------------------------------------- the request
class Relay(http.server.BaseHTTPRequestHandler):
    """Answer one request: a report emailed, or a refusal that says why, as JSON."""

    protocol_version = "HTTP/1.0"  # the connection closes after its one reply
    server_version = "factorial-feedback"
    sys_version = ""
    timeout = HEADER_SECONDS
    server: Server

    def do_POST(self) -> None:
        """Take one report: check the request, count it against its network, read and check it, and email it."""
        if not self.server.slots.acquire(blocking=False):
            self.reply(503, "the relay is busy: try again in a minute")
            return
        try:
            send(compose(self.report(), self.server.mail), self.server.mail)
        except RefusedError as refusal:
            self.reply(refusal.status, str(refusal))
        except OSError as error:  # smtplib's errors, TLS's and the network's
            self.reply(502, "the email could not be sent", cause=type(error).__name__)
        except Exception as error:  # noqa: BLE001 - whatever it was, its message could hold what the request sent
            self.reply(500, "the relay failed", cause=type(error).__name__)
        else:
            self.reply(200, None)
        finally:
            self.server.slots.release()

    def report(self) -> Report:
        """Check the request's address and headers, count it, then read its body and check the report in it."""
        if self.path != PATH:
            raise RefusedError(404, f"only {PATH} is answered here")
        if "Transfer-Encoding" in self.headers:
            raise RefusedError(400, "a body sent in chunks is not taken: send it with its Content-Length")
        try:
            where = network(self.headers.get("X-Real-IP", ""))
        except ValueError as error:
            raise RefusedError(403, "the request did not come through the site's web server") from error
        if self.headers.get("Origin") != ORIGIN:
            raise RefusedError(403, f"reports are taken only from the app at {ORIGIN}")
        if self.headers.get_content_type() != "application/json":
            raise RefusedError(415, "a report is sent as application/json")
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            raise RefusedError(411, "the request has no Content-Length")
        if len(lengths) > 1 or not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
            raise RefusedError(400, "the request's Content-Length is not one number")
        size = int(lengths[0])
        if size > MAX_BODY:
            raise RefusedError(413, f"a report holds at most {MAX_BODY // 2**10} KiB")
        refused = self.server.limits.admit(where)
        if refused:
            raise RefusedError(429, refused)
        return read_report(self.body(size))

    def body(self, size: int) -> bytes:
        """Read exactly ``size`` bytes of the request's body, all of them within ``BODY_SECONDS``."""
        deadline = time.monotonic() + BODY_SECONDS
        chunks: list[bytes] = []
        got = 0
        while got < size:
            left = deadline - time.monotonic()
            if left <= 0:
                raise RefusedError(408, "the body took too long to arrive")
            self.connection.settimeout(left)
            try:
                chunk = self.rfile.read1(min(size - got, 2**16))
            except TimeoutError as error:
                raise RefusedError(408, "the body took too long to arrive") from error
            if not chunk:
                raise RefusedError(400, "the body is shorter than its Content-Length")
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def reply(self, status: int, refusal: str | None, cause: str = "") -> None:
        """Answer ``{"ok": true}``, or ``{"ok": false, "error": refusal}``, and log the status and the reason alone."""
        data = json.dumps({"ok": True} if refusal is None else {"ok": False, "error": refusal}).encode()
        log(f"{status} {refusal or 'sent'}" + (f" ({cause})" if cause else ""))
        with contextlib.suppress(OSError):  # the client left: nobody to tell
            self.connection.settimeout(REPLY_SECONDS)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if status in {429, 503}:
                self.send_header("Retry-After", "3600" if status == 429 else "60")
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, *_: object) -> None:
        """Say nothing of each request: its line and its address are the reader's."""


class Server(http.server.ThreadingHTTPServer):
    """The relay's listener: a thread per connection, sharing the settings, the limits and the slots in flight."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], mail: Mail, limits: Limits | None = None) -> None:
        super().__init__(address, Relay)
        self.mail = mail
        self.limits = limits or Limits()
        self.slots = threading.BoundedSemaphore(IN_FLIGHT)

    def handle_error(self, request: object, client_address: object) -> None:
        """Note that a request failed, by its exception's name alone: a traceback could quote what it sent."""
        del request, client_address
        log(f"a request failed ({sys.exc_info()[0].__name__})")

    def shutdown_request(self, request: socket.socket) -> None:  # type: ignore[override]
        """Close a connection after its reply has gone, and once the client has stopped sending.

        A refusal is answered before the body is read. Closed with that body unread, the connection would be
        reset, and the client could lose the reply; so what the client still sends is read, and dropped.
        """
        with contextlib.suppress(OSError):
            request.shutdown(socket.SHUT_WR)
            request.settimeout(LINGER_SECONDS)
            left = LINGER_BYTES
            while left > 0 and (dropped := request.recv(min(left, 2**16))):
                left -= len(dropped)
        self.close_request(request)


# --------------------------------------------------------------------------- the settings
def one_address(value: str) -> bool:
    """Say whether a setting is one email address, as a reply address must be."""
    try:
        reply_address(value)
    except RefusedError:
        return False
    return True


def settings(environ: Mapping[str, str]) -> Mail:
    """Read the relay's settings from the environment, and the SMTP password from systemd's credential.

    Raise ValueError, saying which setting is missing or wrong.
    """
    missing = [name for name in ("SMTP_HOST", "SMTP_FROM_EMAIL", "FEEDBACK_TO") if not environ.get(name)]
    if missing:
        named = " and ".join([", ".join(missing[:-1]), missing[-1]] if len(missing) > 1 else missing)
        msg = f"{named} must be set, in /etc/factorial/feedback.env"
        raise ValueError(msg)
    for name in ("SMTP_FROM_EMAIL", "FEEDBACK_TO"):
        if not one_address(environ[name]):
            msg = f"{name} is not one email address"
            raise ValueError(msg)
    port = environ.get("SMTP_PORT") or "587"
    if not port.isascii() or not port.isdigit():
        msg = "SMTP_PORT is not a port number"
        raise ValueError(msg)
    password = ""
    if environ.get("CREDENTIALS_DIRECTORY"):
        credential = Path(environ["CREDENTIALS_DIRECTORY"]) / "smtp-password"
        if credential.is_file():
            password = credential.read_text(encoding="utf-8").rstrip("\r\n")
    user = environ.get("SMTP_USER", "")
    if user and not password:
        msg = "SMTP_USER is set, but the credential smtp-password (/etc/factorial/smtp-password) is empty"
        raise ValueError(msg)
    return Mail(environ["SMTP_HOST"], int(port), user, password, environ["SMTP_FROM_EMAIL"], environ["FEEDBACK_TO"])


def main(argv: list[str] | None = None) -> int:
    """Take reports on ``ADDRESS`` until stopped, with the settings in the environment."""
    if sys.argv[1:] if argv is None else argv:
        print("usage: factorial-feedback, with its settings in the environment", file=sys.stderr)
        return 2
    try:
        mail = settings(os.environ)
    except ValueError as reason:
        print(f"factorial-feedback: {reason}", file=sys.stderr)
        return 1
    with Server(ADDRESS, mail) as server:
        log(f"taking reports on {ADDRESS[0]}:{ADDRESS[1]}{PATH}")
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
