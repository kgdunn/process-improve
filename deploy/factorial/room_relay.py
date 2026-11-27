#!/usr/bin/python3 -I
"""factori.al's room relay: a mailbox that keeps one sealed study for each room, and cannot read it.

Two or more people work on one study through a room (``web/room.js``). Each browser seals the study with a key
that only the room's link holds, after its ``#``, which a browser sends to no server. The relay keeps the sealed
bytes it is sent and hands them back. It never holds the key: it sees a room's id, how many bytes the room holds,
and when they change.

Caddy passes it ``/api/room`` (``factorial.caddy``), with the reader's address in ``X-Real-IP``. The relay
listens on ``127.0.0.1:8791`` and answers one request per connection. The room's id, or its write token, travels
in the ``Authorization`` header, which the site's access log leaves out, so no room is named in the log::

    GET    /api/room   Authorization: Room <id>       the room's sealed study and its ETag; 304 when it is the
                       If-None-Match: "<etag>"        one the browser has; 404 when there is no such room
    PUT    /api/room   Authorization: Room <token>    keep these bytes as the room's study, if the room is still
                       If-Match: "<etag>"             at that version, or, with If-None-Match: *, if there is no
                    or If-None-Match: *               such room yet; else 412, and the browser reads it again
    DELETE /api/room   Authorization: Room <token>    remove the room

The id is the first 16 bytes of the SHA-256 of the write token. So the relay checks a write without keeping
anything to check it against, and the id it holds leads back to neither the token nor the key.

What it keeps, and for how long:

- One blob for each room, which each write replaces: no history, no list of rooms, no accounts.
- A blob of one of ``BLOB_SIZES`` bytes exactly. The page pads a study to one of them before it seals it, so a
  blob's size does not give away how many runs the study has, and an image or a file does not fit.
- A room that nobody writes to for ``EXPIRY_DAYS`` days is deleted.
- At most ``BUDGET`` bytes in all: past that, a new room is refused until old ones expire.

Each network (a /24 of IPv4, a /48 of IPv6) may make ``NEW_ROOMS`` rooms a day, and write ``WRITES`` and read
``READS`` times an hour, counted in memory: the relay is one process. Standard library only.

It keeps the rooms in the folder systemd gives it (``STATE_DIRECTORY``), or in ``ROOM_FOLDER``. Run as
``factorial-room forget <link>``, it removes the room a link names, for a room reported to the operator.
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

#: Where the relay listens: Caddy, on the same machine, is the only one that reaches it.
ADDRESS = ("127.0.0.1", 8791)
#: The one path answered, and the one page a write may come from, unless ``ROOM_ORIGIN`` names another.
PATH = "/api/room"
ORIGIN = "https://factori.al"
#: The sizes a study is padded to before it is sealed (``web/room.js``), and the sizes of the blob that holds one:
#: a 12-byte nonce before it, and AES-GCM's 16-byte tag after it.
PADDED = (4096, 16384, 65536)
NONCE = 12
TAG = 16
BLOB_SIZES = frozenset(NONCE + size + TAG for size in PADDED)
#: A room's id (16 bytes) and its write token (32 bytes), in URL-safe base64 without padding.
ROOM_ID = re.compile(r"[A-Za-z0-9_-]{22}")
TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
#: How long a room is kept after its last write, how often the relay looks for rooms past that, and the most bytes
#: all rooms may hold together.
EXPIRY_DAYS = 30
SWEEP_SECONDS = 3600
BUDGET = 256 * 2**20
#: How many rooms one network may make in a day, and how many writes and reads it may send in an hour; how many
#: networks each count holds at once.
NEW_ROOMS = 20
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
#: What the page and the relay each derive from a room's secret: the label of the write token (``web/room.js``).
WRITE_LABEL = b"factorial room: write"
#: A room's link: its secret, 32 bytes in URL-safe base64, after ``#room=``.
LINK = re.compile(r"(?:^|#room=)([A-Za-z0-9_-]{43})$")


class RefusedError(Exception):
    """A request the relay does not carry out: the HTTP status to answer with, and why, in the relay's own words."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status


def log(line: str) -> None:
    """Write one line to the journal (systemd keeps what the relay writes to stderr)."""
    print(line, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- ids and tokens
def unbase64(text: str) -> bytes:
    """Decode URL-safe base64 written without its padding."""
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def room_of(token: bytes) -> str:
    """Return the id of the room a write token writes to: the first 16 bytes of its SHA-256, in URL-safe base64."""
    return base64.urlsafe_b64encode(hashlib.sha256(token).digest()[:16]).rstrip(b"=").decode()


def hkdf(secret: bytes, label: bytes, length: int = 32) -> bytes:
    """Derive ``length`` bytes from a room's secret for ``label``: HKDF with SHA-256 (RFC 5869), with no salt.

    WebCrypto's HKDF, as ``web/room.js`` calls it, gives the same bytes.
    """
    key = hmac.new(bytes(32), secret, hashlib.sha256).digest()
    out, block = b"", b""
    for counter in range(1, -(-length // 32) + 1):
        block = hmac.new(key, block + label + bytes([counter]), hashlib.sha256).digest()
        out += block
    return out[:length]


def room_of_link(link: str) -> str:
    """Return the id of the room a link (or its secret alone) names; raise ValueError for anything else."""
    found = LINK.search(link.strip())
    if not found:
        msg = "that is not a room's link: it ends with #room= and 43 characters"
        raise ValueError(msg)
    return room_of(hkdf(unbase64(found[1]), WRITE_LABEL))


def etag(blob: bytes) -> str:
    """Return a blob's version: 12 bytes of its SHA-256, in URL-safe base64. A sealed blob never repeats."""
    return base64.urlsafe_b64encode(hashlib.sha256(blob).digest()[:12]).decode()


# --------------------------------------------------------------------------- the rooms
class Store:
    """The rooms, a file each, named by the room's id, in one folder; a file's time is its room's last write."""

    def __init__(self, folder: Path, clock: Callable[[], float] = time.time) -> None:
        self.folder = folder
        self.clock = clock
        self.lock = threading.Lock()
        self.swept = float("-inf")
        self.used = sum(path.stat().st_size for path in self.rooms())

    def rooms(self) -> list[Path]:
        """Every room's file: a name that is a room's id, and nothing else in the folder."""
        return [path for path in self.folder.iterdir() if ROOM_ID.fullmatch(path.name) and path.is_file()]

    def expired(self, path: Path, now: float) -> bool:
        """Say whether a room's file was last written more than ``EXPIRY_DAYS`` days before ``now``."""
        return now - path.stat().st_mtime > EXPIRY_DAYS * DAY

    def sweep(self, now: float) -> None:
        """Delete every room not written for ``EXPIRY_DAYS`` days, at most once in ``SWEEP_SECONDS``. Hold the lock."""
        if now - self.swept < SWEEP_SECONDS:
            return
        self.swept = now
        for path in self.rooms():
            with contextlib.suppress(FileNotFoundError):
                if self.expired(path, now):
                    self.forget(path)

    def forget(self, path: Path) -> bool:
        """Delete one room's file, counting its bytes out; say whether there was one. Hold the lock."""
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            return False
        self.used -= size
        return True

    def current(self, room: str, now: float) -> bytes | None:
        """Return the blob a room holds, or None if there is none or it has expired. Hold the lock."""
        path = self.folder / room
        try:
            if self.expired(path, now):
                self.forget(path)
                return None
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def read(self, room: str) -> bytes | None:
        """Return the blob a room holds, or None."""
        with self.lock:
            now = self.clock()
            self.sweep(now)
            return self.current(room, now)

    def write(self, room: str, blob: bytes, *, version: str | None) -> tuple[str, bool]:
        """Keep ``blob`` as the room's, if the room is at ``version``, or, for None, if there is no such room yet.

        Return the blob's version and whether the room is new. Raise ``RefusedError`` with 412 when the room is not as
        the writer last saw it, and with 507 when a new room would take the relay past its ``BUDGET``.
        """
        with self.lock:
            now = self.clock()
            self.sweep(now)
            held = self.current(room, now)
            if version is None and held is not None:
                raise RefusedError(412, "the room exists already: read it first")
            if version is not None and (held is None or etag(held) != version):
                raise RefusedError(412, "the room has changed since it was read: read it again")
            grows = len(blob) - (len(held) if held else 0)
            if held is None and self.used + grows > BUDGET:
                raise RefusedError(507, "the relay holds all the rooms it can: try again in a few days")
            path = self.folder / room
            part = path.with_name(f".{room}.part")
            part.write_bytes(blob)
            os.utime(part, (now, now))
            part.replace(path)
            self.used += grows
            return etag(blob), held is None

    def remove(self, room: str) -> bool:
        """Remove a room; say whether there was one."""
        with self.lock:
            return self.forget(self.folder / room)


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
    """The counts a network is held to: rooms made in a day, writes and reads in an hour."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.new = Counter(NEW_ROOMS, DAY, clock)
        self.writes = Counter(WRITES, HOUR, clock)
        self.reads = Counter(READS, HOUR, clock)


# --------------------------------------------------------------------------- the request
class Relay(http.server.BaseHTTPRequestHandler):
    """Answer one request: a room read, written or removed, or a refusal that says why, as JSON."""

    protocol_version = "HTTP/1.0"  # the connection closes after its one reply
    server_version = "factorial-room"
    sys_version = ""
    timeout = HEADER_SECONDS
    server: Server

    def do_GET(self) -> None:
        """Read a room."""
        self.handle_one(self.get)

    def do_PUT(self) -> None:
        """Write a room."""
        self.handle_one(self.put)

    def do_DELETE(self) -> None:
        """Remove a room."""
        self.handle_one(self.delete)

    def handle_one(self, act: Callable[[str], None]) -> None:
        """Check the request, then ``act`` on it for the network it came from, answering whatever it raised."""
        if not self.server.slots.acquire(blocking=False):
            self.reply(503, "the relay is busy: try again in a moment")
            return
        try:
            act(self.sender())
        except RefusedError as refusal:
            self.reply(refusal.status, str(refusal))
        except OSError as error:
            self.reply(500, "the room could not be kept", cause=type(error).__name__)
        except Exception as error:  # noqa: BLE001 - whatever it was, its message could name the room
            self.reply(500, "the relay failed", cause=type(error).__name__)
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
        """Return what ``Authorization: Room <value>`` gives, if it has the ``form`` of a room's ``what``."""
        given = self.headers.get_all("Authorization", [])
        scheme, _, value = given[0].partition(" ") if len(given) == 1 else ("", "", "")
        if scheme != "Room" or not form.fullmatch(value):
            raise RefusedError(401, f"the request does not carry a room's {what}")
        return value

    def get(self, where: str) -> None:
        """Hand back a room's blob, or say that the browser has it already, or that there is no such room."""
        room = self.credential(ROOM_ID, "id")
        if not self.server.limits.reads.admit(where):
            raise RefusedError(429, f"a network reads rooms at most {READS} times an hour: try again later")
        blob = self.server.store.read(room)
        if blob is None:
            raise RefusedError(404, f"there is no such room: a room nobody writes to for {EXPIRY_DAYS} days is removed")
        version = etag(blob)
        if self.headers.get("If-None-Match") == f'"{version}"':
            self.answer(304, b"", version)
        else:
            self.answer(200, blob, version, "application/octet-stream")

    def put(self, where: str) -> None:
        """Keep a blob as a room's study, if the writer holds the room's token and saw its latest version."""
        token = self.credential(TOKEN, "write token")
        self.from_the_app()
        if self.headers.get_content_type() != "application/octet-stream":
            raise RefusedError(415, "a room's study is sent as application/octet-stream")
        match, none_match = self.headers.get("If-Match"), self.headers.get("If-None-Match")
        if (match is None) == (none_match is None) or (none_match not in {None, "*"}):
            raise RefusedError(428, "a write names the version it replaces (If-Match), or a new room If-None-Match: *")
        version = None if match is None else match.strip('"')
        size = self.length()
        if size not in BLOB_SIZES:
            sizes = ", ".join(str(size) for size in sorted(BLOB_SIZES))
            raise RefusedError(413, f"a room holds a sealed study of {sizes} bytes, and nothing else")
        if not self.server.limits.writes.admit(where):
            raise RefusedError(429, f"a network writes to rooms at most {WRITES} times an hour: try again later")
        if version is None and not self.server.limits.new.admit(where):
            raise RefusedError(429, f"a network makes at most {NEW_ROOMS} rooms a day: try again tomorrow")
        blob = self.body(size)
        made, new = self.server.store.write(room_of(unbase64(token)), blob, version=version)
        self.answer(201 if new else 200, b"", made)

    def delete(self, where: str) -> None:
        """Remove a room, for whoever holds its token."""
        token = self.credential(TOKEN, "write token")
        self.from_the_app()
        if not self.server.limits.writes.admit(where):
            raise RefusedError(429, f"a network writes to rooms at most {WRITES} times an hour: try again later")
        if not self.server.store.remove(room_of(unbase64(token))):
            raise RefusedError(404, "there is no such room")
        self.answer(204, b"", None)

    def from_the_app(self) -> None:
        """Refuse a change that does not come from the app's own page."""
        if self.headers.get("Origin") != self.server.origin:
            raise RefusedError(403, f"rooms are changed only from the app at {self.server.origin}")

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
        """Send a room's blob, or none, with its version; log the status alone."""
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
    """The relay's listener: a thread per connection, sharing the rooms, the limits and the slots in flight."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: Store, limits: Limits | None = None, origin: str = ORIGIN):
        super().__init__(address, Relay)
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
    """Return the folder the rooms are kept in: systemd's ``STATE_DIRECTORY``, or ``ROOM_FOLDER``."""
    named = environ.get("STATE_DIRECTORY", "").split(":")[0] or environ.get("ROOM_FOLDER", "")
    if not named:
        msg = "neither STATE_DIRECTORY (set by systemd) nor ROOM_FOLDER names the folder for the rooms"
        raise ValueError(msg)
    path = Path(named)
    if not path.is_dir():
        msg = f"{path} is not a folder"
        raise ValueError(msg)
    return path


def main(argv: list[str] | None = None) -> int:
    """Keep rooms on ``ADDRESS`` until stopped; or, as ``forget <link>``, remove the room a link names."""
    args = sys.argv[1:] if argv is None else argv
    try:
        store = Store(folder(os.environ))
        if args[:1] == ["forget"] and len(args) == 2:
            room = room_of_link(args[1])
            print("Removed the room." if store.remove(room) else "There is no such room: it may have expired.")
            return 0
    except (ValueError, binascii.Error) as reason:
        print(f"factorial-room: {reason}", file=sys.stderr)
        return 1
    if args:
        print("usage: factorial-room, or factorial-room forget <a room's link>", file=sys.stderr)
        return 2
    with Server(ADDRESS, store, origin=os.environ.get("ROOM_ORIGIN") or ORIGIN) as server:
        log(f"keeping rooms on {ADDRESS[0]}:{ADDRESS[1]}{PATH}")
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
