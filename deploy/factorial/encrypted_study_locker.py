#!/usr/bin/python3 -I
"""factori.al's encrypted study locker: it keeps one encrypted study for each locker, and cannot open it.

Two or more people work on one study through a locker (``web/locker.js``). Each browser encrypts the study with a key
that only the locker's link holds, after its ``#``, which a browser sends to no server. The server keeps the encrypted
bytes it is sent and hands them back. It never holds the key: it sees a locker's id, how many bytes the locker holds,
and when they change.

Caddy passes it ``/api/locker`` (``factorial.caddy``), with the reader's address in ``X-Real-IP``. The server
listens on ``127.0.0.1:8791`` and answers one request per connection. The locker's id, or its write token, travels
in the ``Authorization`` header, which the site's access log leaves out, so no locker is named in the log::

    GET    /api/locker   Authorization: Locker <id>       the locker's encrypted study and its ETag; 304 when it is the
                       If-None-Match: "<etag>"        one the browser has; 404 when there is no such locker
    PUT    /api/locker   Authorization: Locker <token>    keep these bytes as the locker's study, if the locker is still
                       If-Match: "<etag>"             at that version, or, with If-None-Match: *, if there is no
                    or If-None-Match: *               such locker yet; else 412, and the browser reads it again
    DELETE /api/locker   Authorization: Locker <token>    remove the locker

The id is the first 16 bytes of the SHA-256 of the write token. So the server checks a write without keeping
anything to check it against, and the id it holds leads back to neither the token nor the key.

What it keeps, and for how long:

- One blob for each locker, which each write replaces: no history, no list of lockers, no accounts.
- A blob of one of ``BLOB_SIZES`` bytes exactly. The page pads a study to one of them before it encrypts it, so a
  blob's size does not give away how many runs the study has, and an image or a file does not fit.
- A locker that nobody writes to for ``EXPIRY_DAYS`` days is deleted.
- At most ``BUDGET`` bytes in all: past that, a new locker is refused until old ones expire.

Each network (a /24 of IPv4, a /48 of IPv6) may make ``NEW_LOCKERS`` lockers a day, and write ``WRITES`` and read
``READS`` times an hour, counted in memory: the server is one process. Standard library only.

It keeps the lockers in the folder systemd gives it (``STATE_DIRECTORY``), or in ``LOCKER_FOLDER``. Run as
``factorial-study-locker forget <link>``, it removes the locker a link names, for a locker reported to the operator.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import hmac
import http.server
import ipaddress
import json
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

#: Where the server listens: Caddy, on the same machine, is the only one that reaches it.
ADDRESS = ("127.0.0.1", 8791)
#: The one path answered, and the one page a write may come from, unless ``LOCKER_ORIGIN`` names another.
PATH = "/api/locker"
ORIGIN = "https://factori.al"
#: The sizes a study is padded to before it is encrypted (``web/locker.js``), and the sizes of the blob that holds one:
#: a 12-byte nonce before it, and AES-GCM's 16-byte tag after it.
PADDED = (4096, 16384, 65536)
NONCE = 12
TAG = 16
BLOB_SIZES = frozenset(NONCE + size + TAG for size in PADDED)
#: A locker's id (16 bytes) and its write token (32 bytes), in URL-safe base64 without padding.
LOCKER_ID = re.compile(r"[A-Za-z0-9_-]{22}")
TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
#: How long a locker is kept after its last write, how often the server looks for lockers past that, and the most bytes
#: all lockers may hold together.
EXPIRY_DAYS = 30
SWEEP_SECONDS = 3600
BUDGET = 256 * 2**20
#: How many lockers one network may make in a day, and how many writes and reads it may send in an hour; how many
#: networks each count holds at once.
NEW_LOCKERS = 20
WRITES = 600
READS = 3600
HOUR = 3600
DAY = 24 * HOUR
NETWORKS = 10_000
#: How many requests are worked on at once; one more is refused, to try again a moment later.
IN_FLIGHT = 16
#: How long, in seconds, the request's line and headers may pause, its body may take to arrive, and the reply may
#: take to write; after the reply, how long a connection waits for the client to finish, and how much it drops.
HEADER_SECONDS = 10
BODY_SECONDS = 20
REPLY_SECONDS = 10
LINGER_SECONDS = 2
LINGER_BYTES = 2**17
#: What the page and the server each derive from a locker's secret: the label of the write token (``web/locker.js``).
WRITE_LABEL = b"factorial locker: write"
#: A locker's link: its secret, 32 bytes in URL-safe base64, after ``#locker=``.
LINK = re.compile(r"(?:^|#locker=)([A-Za-z0-9_-]{43})$")


class RefusedError(Exception):
    """A request the server does not carry out: the HTTP status to answer with, and why, in the server's own words."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status


def log(line: str) -> None:
    """Write one line to the journal (systemd keeps what the server writes to stderr)."""
    print(line, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- ids and tokens
def unbase64(text: str) -> bytes:
    """Decode URL-safe base64 written without its padding."""
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def locker_of(token: bytes) -> str:
    """Return the id of the locker a write token writes to: the first 16 bytes of its SHA-256, in URL-safe base64."""
    return base64.urlsafe_b64encode(hashlib.sha256(token).digest()[:16]).rstrip(b"=").decode()


def hkdf(secret: bytes, label: bytes, length: int = 32) -> bytes:
    """Derive ``length`` bytes from a locker's secret for ``label``: HKDF with SHA-256 (RFC 5869), with no salt.

    WebCrypto's HKDF, as ``web/locker.js`` calls it, gives the same bytes.
    """
    key = hmac.new(bytes(32), secret, hashlib.sha256).digest()
    out, block = b"", b""
    for counter in range(1, -(-length // 32) + 1):
        block = hmac.new(key, block + label + bytes([counter]), hashlib.sha256).digest()
        out += block
    return out[:length]


def locker_of_link(link: str) -> str:
    """Return the id of the locker a link (or its secret alone) names; raise ValueError for anything else."""
    found = LINK.search(link.strip())
    if not found:
        msg = "that is not a locker's link: it ends with #locker= and 43 characters"
        raise ValueError(msg)
    return locker_of(hkdf(unbase64(found[1]), WRITE_LABEL))


def etag(blob: bytes) -> str:
    """Return a blob's version: 12 bytes of its SHA-256, in URL-safe base64. An encrypted blob never repeats."""
    return base64.urlsafe_b64encode(hashlib.sha256(blob).digest()[:12]).decode()


# --------------------------------------------------------------------------- the lockers
class Store:
    """The lockers, a file each, named by the locker's id, in one folder; a file's time is its locker's last write."""

    def __init__(self, folder: Path, clock: Callable[[], float] = time.time) -> None:
        self.folder = folder
        self.clock = clock
        self.lock = threading.Lock()
        self.swept = float("-inf")
        self.used = sum(path.stat().st_size for path in self.lockers())

    def lockers(self) -> list[Path]:
        """Every locker's file: a name that is a locker's id, and nothing else in the folder."""
        return [path for path in self.folder.iterdir() if LOCKER_ID.fullmatch(path.name) and path.is_file()]

    def expired(self, path: Path, now: float) -> bool:
        """Say whether a locker's file was last written more than ``EXPIRY_DAYS`` days before ``now``."""
        return now - path.stat().st_mtime > EXPIRY_DAYS * DAY

    def sweep(self, now: float) -> None:
        """Delete each locker not written for ``EXPIRY_DAYS`` days, at most once in ``SWEEP_SECONDS``. Hold the lock."""
        if now - self.swept < SWEEP_SECONDS:
            return
        self.swept = now
        for path in self.lockers():
            with contextlib.suppress(FileNotFoundError):
                if self.expired(path, now):
                    self.forget(path)

    def forget(self, path: Path) -> bool:
        """Delete one locker's file, counting its bytes out; say whether there was one. Hold the lock."""
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            return False
        self.used -= size
        return True

    def current(self, locker: str, now: float) -> bytes | None:
        """Return the blob a locker holds, or None if there is none or it has expired. Hold the lock."""
        path = self.folder / locker
        try:
            if self.expired(path, now):
                self.forget(path)
                return None
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def read(self, locker: str) -> bytes | None:
        """Return the blob a locker holds, or None."""
        with self.lock:
            now = self.clock()
            self.sweep(now)
            return self.current(locker, now)

    def write(self, locker: str, blob: bytes, *, version: str | None) -> tuple[str, bool]:
        """Keep ``blob`` as the locker's, if the locker is at ``version``, or, for None, if there is no such locker yet.

        Return the blob's version and whether the locker is new. Raise ``RefusedError`` with 412 when the locker is not
        as the writer last saw it, and with 507 when a new locker would take the server past its ``BUDGET``.
        """
        with self.lock:
            now = self.clock()
            self.sweep(now)
            held = self.current(locker, now)
            if version is None and held is not None:
                raise RefusedError(412, "the locker exists already: read it first")
            if version is not None and (held is None or etag(held) != version):
                raise RefusedError(412, "the locker has changed since it was read: read it again")
            grows = len(blob) - (len(held) if held else 0)
            if held is None and self.used + grows > BUDGET:
                raise RefusedError(507, "the server holds all the lockers it can: try again in a few days")
            path = self.folder / locker
            part = path.with_name(f".{locker}.part")
            part.write_bytes(blob)
            os.utime(part, (now, now))
            part.replace(path)
            self.used += grows
            return etag(blob), held is None

    def remove(self, locker: str) -> bool:
        """Remove a locker; say whether there was one."""
        with self.lock:
            return self.forget(self.folder / locker)


# --------------------------------------------------------------------------- the limits
def network(address: str) -> str:
    """Return the network an address is counted in: its /24 for IPv4, its /48 for IPv6 (as feedback_relay.py)."""
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    prefix = 24 if ip.version == 4 else 48
    return str(ipaddress.ip_network(f"{ip}/{prefix}", strict=False))


class Counter:
    """How many times each network did one thing in the current window of ``seconds``, held in memory.

    The window is fixed (the hour or the day, UTC), so a network may do twice ``most`` across the turn of one. The
    table holds at most ``NETWORKS`` networks: when it is full, those of an earlier window are forgotten, and if none
    is, a newcomer is refused until the window turns.
    """

    def __init__(self, most: int, seconds: int, clock: Callable[[], float]) -> None:
        self.most = most
        self.seconds = seconds
        self.clock = clock
        self.lock = threading.Lock()
        self.counts: dict[str, tuple[int, int]] = {}  # each network's window, and its count in it

    def admit(self, where: str) -> bool:
        """Count one more for the network ``where``; say whether it is within its limit."""
        window = int(self.clock() // self.seconds)
        with self.lock:
            held, count = self.counts.get(where, (window, 0))
            if held != window:
                count = 0
            if count >= self.most:
                return False
            if where not in self.counts and len(self.counts) >= NETWORKS:
                self.counts = {key: kept for key, kept in self.counts.items() if kept[0] == window}
                if len(self.counts) >= NETWORKS:
                    return False
            self.counts[where] = (window, count + 1)
            return True


class Limits:
    """The counts a network is held to: lockers made in a day, writes and reads in an hour."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.new = Counter(NEW_LOCKERS, DAY, clock)
        self.writes = Counter(WRITES, HOUR, clock)
        self.reads = Counter(READS, HOUR, clock)


# --------------------------------------------------------------------------- the request
class Locker(http.server.BaseHTTPRequestHandler):
    """Answer one request: a locker read, written or removed, or a refusal that says why, as JSON."""

    protocol_version = "HTTP/1.0"  # the connection closes after its one reply
    server_version = "factorial-study-locker"
    sys_version = ""
    timeout = HEADER_SECONDS
    server: Server

    def do_GET(self) -> None:
        """Read a locker."""
        self.handle_one(self.get)

    def do_PUT(self) -> None:
        """Write a locker."""
        self.handle_one(self.put)

    def do_DELETE(self) -> None:
        """Remove a locker."""
        self.handle_one(self.delete)

    def handle_one(self, act: Callable[[str], None]) -> None:
        """Check the request, then ``act`` on it for the network it came from, answering whatever it raised."""
        if not self.server.slots.acquire(blocking=False):
            self.reply(503, "the server is busy: try again in a moment")
            return
        try:
            act(self.sender())
        except RefusedError as refusal:
            self.reply(refusal.status, str(refusal))
        except OSError as error:
            self.reply(500, "the locker could not be kept", cause=type(error).__name__)
        except Exception as error:  # noqa: BLE001 - whatever it was, its message could name the locker
            self.reply(500, "the server failed", cause=type(error).__name__)
        finally:
            self.server.slots.release()

    def sender(self) -> str:
        """Check the request's path and how it came, and return the network it came from."""
        if self.path != PATH:
            raise RefusedError(404, f"only {PATH} is answered here")
        if "Transfer-Encoding" in self.headers:
            raise RefusedError(400, "a body sent in chunks is not taken: send it with its Content-Length")
        try:
            return network(self.headers.get("X-Real-IP", ""))
        except ValueError as error:
            raise RefusedError(403, "the request did not come through the site's web server") from error

    def credential(self, form: re.Pattern[str], what: str) -> str:
        """Return what ``Authorization: Locker <value>`` gives, if it has the ``form`` of a locker's ``what``."""
        given = self.headers.get_all("Authorization", [])
        scheme, _, value = given[0].partition(" ") if len(given) == 1 else ("", "", "")
        if scheme != "Locker" or not form.fullmatch(value):
            raise RefusedError(401, f"the request does not carry a locker's {what}")
        return value

    def get(self, where: str) -> None:
        """Hand back a locker's blob, or say that the browser has it already, or that there is no such locker."""
        locker = self.credential(LOCKER_ID, "id")
        if not self.server.limits.reads.admit(where):
            raise RefusedError(429, f"a network reads lockers at most {READS} times an hour: try again later")
        blob = self.server.store.read(locker)
        if blob is None:
            raise RefusedError(
                404, f"there is no such locker: a locker nobody writes to for {EXPIRY_DAYS} days is removed"
            )
        version = etag(blob)
        if self.headers.get("If-None-Match") == f'"{version}"':
            self.answer(304, b"", version)
        else:
            self.answer(200, blob, version, "application/octet-stream")

    def put(self, where: str) -> None:
        """Keep a blob as a locker's study, if the writer holds the locker's token and saw its latest version."""
        token = self.credential(TOKEN, "write token")
        self.from_the_app()
        if self.headers.get_content_type() != "application/octet-stream":
            raise RefusedError(415, "a locker's study is sent as application/octet-stream")
        match, none_match = self.headers.get("If-Match"), self.headers.get("If-None-Match")
        if (match is None) == (none_match is None) or (none_match not in {None, "*"}):
            raise RefusedError(
                428, "a write names the version it replaces (If-Match), or a new locker If-None-Match: *"
            )
        version = None if match is None else match.strip('"')
        size = self.length()
        if size not in BLOB_SIZES:
            sizes = ", ".join(str(size) for size in sorted(BLOB_SIZES))
            raise RefusedError(413, f"a locker holds an encrypted study of {sizes} bytes, and nothing else")
        if not self.server.limits.writes.admit(where):
            raise RefusedError(429, f"a network writes to lockers at most {WRITES} times an hour: try again later")
        if version is None and not self.server.limits.new.admit(where):
            raise RefusedError(429, f"a network makes at most {NEW_LOCKERS} lockers a day: try again tomorrow")
        blob = self.body(size)
        made, new = self.server.store.write(locker_of(unbase64(token)), blob, version=version)
        self.answer(201 if new else 200, b"", made)

    def delete(self, where: str) -> None:
        """Remove a locker, for whoever holds its token."""
        token = self.credential(TOKEN, "write token")
        self.from_the_app()
        if not self.server.limits.writes.admit(where):
            raise RefusedError(429, f"a network writes to lockers at most {WRITES} times an hour: try again later")
        if not self.server.store.remove(locker_of(unbase64(token))):
            raise RefusedError(404, "there is no such locker")
        self.answer(204, b"", None)

    def from_the_app(self) -> None:
        """Refuse a change that does not come from the app's own page."""
        if self.headers.get("Origin") != self.server.origin:
            raise RefusedError(403, f"lockers are changed only from the app at {self.server.origin}")

    def length(self) -> int:
        """Return the request's one Content-Length."""
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            raise RefusedError(411, "the request has no Content-Length")
        if len(lengths) > 1 or not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
            raise RefusedError(400, "the request's Content-Length is not one number")
        return int(lengths[0])

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

    def answer(self, status: int, data: bytes, version: str | None, kind: str = "") -> None:
        """Send a locker's blob, or none, with its version; log the status alone."""
        log(f"{status} {self.command}")
        with contextlib.suppress(OSError):  # the client left: nobody to tell
            self.connection.settimeout(REPLY_SECONDS)
            self.send_response(status)
            if kind:
                self.send_header("Content-Type", kind)
            if status != 304:
                self.send_header("Content-Length", str(len(data)))
            if version:
                self.send_header("ETag", f'"{version}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    def reply(self, status: int, refusal: str, cause: str = "") -> None:
        """Answer ``{"ok": false, "error": refusal}``; log the status, the reason and the cause alone."""
        data = json.dumps({"ok": False, "error": refusal}).encode()
        log(f"{status} {self.command} {refusal}" + (f" ({cause})" if cause else ""))
        with contextlib.suppress(OSError):
            self.connection.settimeout(REPLY_SECONDS)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if status in {429, 503}:
                self.send_header("Retry-After", "3600" if status == 429 else "10")
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, *_: object) -> None:
        """Say nothing of each request: its line and its address are the reader's."""


class Server(http.server.ThreadingHTTPServer):
    """The server's listener: a thread per connection, sharing the lockers, the limits and the slots in flight."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: Store, limits: Limits | None = None, origin: str = ORIGIN):
        super().__init__(address, Locker)
        self.store = store
        self.limits = limits or Limits(store.clock)
        self.origin = origin
        self.slots = threading.BoundedSemaphore(IN_FLIGHT)

    def handle_error(self, request: object, client_address: object) -> None:
        """Note that a request failed, by its exception's name alone."""
        del request, client_address
        log(f"a request failed ({sys.exc_info()[0].__name__})")

    def shutdown_request(self, request: socket.socket) -> None:  # type: ignore[override]
        """Close a connection once the client has stopped sending, so a refusal sent before the body is not lost."""
        with contextlib.suppress(OSError):
            request.shutdown(socket.SHUT_WR)
            request.settimeout(LINGER_SECONDS)
            left = LINGER_BYTES
            while left > 0 and (dropped := request.recv(min(left, 2**16))):
                left -= len(dropped)
        self.close_request(request)


# --------------------------------------------------------------------------- the settings
def folder(environ: Mapping[str, str]) -> Path:
    """Return the folder the lockers are kept in: systemd's ``STATE_DIRECTORY``, or ``LOCKER_FOLDER``."""
    named = environ.get("STATE_DIRECTORY", "").split(":")[0] or environ.get("LOCKER_FOLDER", "")
    if not named:
        msg = "neither STATE_DIRECTORY (set by systemd) nor LOCKER_FOLDER names the folder for the lockers"
        raise ValueError(msg)
    path = Path(named)
    if not path.is_dir():
        msg = f"{path} is not a folder"
        raise ValueError(msg)
    return path


def main(argv: list[str] | None = None) -> int:
    """Keep lockers on ``ADDRESS`` until stopped; or, as ``forget <link>``, remove the locker a link names."""
    args = sys.argv[1:] if argv is None else argv
    try:
        store = Store(folder(os.environ))
        if args[:1] == ["forget"] and len(args) == 2:
            locker = locker_of_link(args[1])
            print("Removed the locker." if store.remove(locker) else "There is no such locker: it may have expired.")
            return 0
    except (ValueError, binascii.Error) as reason:
        print(f"factorial-study-locker: {reason}", file=sys.stderr)
        return 1
    if args:
        print("usage: factorial-study-locker, or factorial-study-locker forget <a locker's link>", file=sys.stderr)
        return 2
    with Server(ADDRESS, store, origin=os.environ.get("LOCKER_ORIGIN") or ORIGIN) as server:
        log(f"keeping lockers on {ADDRESS[0]}:{ADDRESS[1]}{PATH}")
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
