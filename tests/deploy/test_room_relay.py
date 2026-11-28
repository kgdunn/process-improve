"""``deploy/factorial/room_relay.py``: the mailbox that keeps one sealed study for each room.

Each test runs the relay on a free local port, over a folder of its own, and sends it what a reader's browser,
behind Caddy, would send, or what anyone else could. The relay must keep a sealed blob of one of a few sizes and
nothing else, hand it back only to whoever names the room, take a change only from whoever holds the room's write
token and saw its latest version, forget a room after its time, and log nothing that names a room.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import socket
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "factorial" / "room_relay.py"


def load():
    spec = importlib.util.spec_from_file_location("room_relay", SCRIPT)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


relay = load()

#: A room's secret, as a link carries it after ``#room=``, and what the page derives from it (web/room.js): the
#: same three values are asserted in tests/web/js/room.test.mjs, so the page and the relay agree on them.
SECRET = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"  # noqa: S105 - bytes 0 to 31, a test vector
TOKEN = base64.urlsafe_b64encode(relay.hkdf(relay.unbase64(SECRET), relay.WRITE_LABEL)).rstrip(b"=").decode()
ROOM = relay.room_of(relay.unbase64(TOKEN))
SMALL = min(relay.BLOB_SIZES)
ADDRESS = "203.0.113.7"
#: What Caddy sends with a change from the app.
APP = {"Origin": "https://factori.al", "X-Real-IP": ADDRESS}


def blob(size: int = SMALL, fill: int = 7) -> bytes:
    return bytes([fill]) * size


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
def store(tmp_path, clock) -> Any:  # noqa: ANN401
    return relay.Store(tmp_path, clock)


@pytest.fixture
def server(store) -> Iterator[Any]:
    """Run the relay on a free local port for the test, over the test's folder and clock."""
    with relay.Server(("127.0.0.1", 0), store) as running:
        thread = threading.Thread(target=running.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        yield running
        running.shutdown()


def ask(
    server, method: str, credential: str | None, body: bytes = b"", headers: dict[str, str | None] | None = None
) -> tuple[int, dict[str, str], bytes]:
    """Send one request as Caddy would, with ``headers`` changed (None takes one out); return the reply."""
    given: dict[str, str | None] = {"Host": "127.0.0.1", **APP}
    if credential is not None:
        given["Authorization"] = f"Room {credential}"
    if method == "PUT":
        given |= {"Content-Type": "application/octet-stream", "Content-Length": str(len(body))}
    given |= headers or {}
    head = "".join(f"{name}: {value}\r\n" for name, value in given.items() if value is not None)
    with socket.create_connection(server.server_address, timeout=10) as connection:
        connection.sendall(f"{method} /api/room HTTP/1.1\r\n{head}\r\n".encode() + body)
        connection.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := connection.recv(2**16):
            received += chunk
    top, _, data = received.partition(b"\r\n\r\n")
    status_line, *lines = top.decode("latin-1").split("\r\n")
    return int(status_line.split()[1]), dict(line.split(": ", 1) for line in lines), data


def make(server, data: bytes | None = None) -> str:
    """Make the room, and return its first version."""
    status, headers, _ = ask(server, "PUT", TOKEN, data or blob(), {"If-None-Match": "*"})
    assert status == 201
    return headers["ETag"].strip('"')


def refusal(data: bytes) -> str:
    answer = json.loads(data)
    assert answer["ok"] is False
    return answer["error"]


# --------------------------------------------------------------------------- the derivation
def test_hkdf_gives_rfc_5869_s_test_case_3():
    """SHA-256 with no salt and no label, as WebCrypto computes it."""
    okm = relay.hkdf(bytes([0x0B]) * 22, b"", 42)
    expected = "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8"
    assert okm.hex() == expected


def test_the_secret_s_token_and_room_are_those_the_page_derives():
    """The same values are in tests/web/js/room.test.mjs: a change on one side fails both."""
    assert TOKEN == "-znuuYT_caimAH2REqvREL_BuEwrsp7hbDyTx3TI6pw"  # noqa: S105 - the test vector's
    assert ROOM == "GD5GAO3nLHTR_pA7bHTdvg"
    assert relay.room_of_link(f"https://factori.al/#room={SECRET}") == ROOM
    assert relay.room_of_link(SECRET) == ROOM


@pytest.mark.parametrize("link", ["https://factori.al/#c=abc", "#room=short", "", f"{SECRET}x"])
def test_anything_but_a_room_s_link_is_refused(link):
    with pytest.raises(ValueError, match="not a room's link"):
        relay.room_of_link(link)


# --------------------------------------------------------------------------- a room, made and read
def test_a_room_is_made_read_and_read_again_as_unchanged(server, tmp_path):
    version = make(server)
    status, headers, data = ask(server, "GET", ROOM)
    assert (status, data) == (200, blob())
    assert headers["Content-Type"] == "application/octet-stream"
    assert headers["ETag"] == f'"{version}"'
    assert headers["Cache-Control"] == "no-store"
    status, headers, data = ask(server, "GET", ROOM, headers={"If-None-Match": f'"{version}"'})
    assert (status, data) == (304, b"")
    assert [path.name for path in tmp_path.iterdir()] == [ROOM], "one file, named by the room's id, and no token"
    assert (tmp_path / ROOM).read_bytes() == blob(), "the sealed bytes as sent, and nothing else"


def test_reading_needs_no_origin_and_writing_needs_the_app_s(server):
    status, _, data = ask(server, "PUT", TOKEN, blob(), {"If-None-Match": "*", "Origin": "https://example.com"})
    assert status == 403
    assert "only from the app at https://factori.al" in refusal(data)
    make(server)
    assert ask(server, "GET", ROOM, headers={"Origin": None})[0] == 200


def test_a_room_that_does_not_exist_is_not_found(server):
    status, _, data = ask(server, "GET", ROOM)
    assert status == 404
    assert "no such room" in refusal(data)


# --------------------------------------------------------------------------- who may do what
@pytest.mark.parametrize(
    ("method", "credential"),
    [("GET", TOKEN), ("GET", None), ("GET", "x" * 22 + "!"), ("PUT", ROOM), ("PUT", None), ("DELETE", ROOM)],
)
def test_a_read_needs_the_room_s_id_and_a_change_its_token(server, method, credential):
    """The id that reads a room is not the token that writes it: a reader who has only the id cannot change it."""
    status, _, data = ask(server, method, credential, blob() if method == "PUT" else b"", {"If-None-Match": "*"})
    assert status == 401, data
    assert "does not carry a room's" in refusal(data)


def test_another_authorization_scheme_is_refused(server):
    make(server)
    assert ask(server, "GET", None, headers={"Authorization": f"Bearer {ROOM}"})[0] == 401


def test_a_change_names_the_version_it_replaces(server):
    first = make(server)
    status, headers, _ = ask(server, "PUT", TOKEN, blob(fill=8), {"If-Match": f'"{first}"'})
    assert status == 200
    second = headers["ETag"].strip('"')
    assert second != first
    assert ask(server, "GET", ROOM)[2] == blob(fill=8)

    status, _, data = ask(server, "PUT", TOKEN, blob(fill=9), {"If-Match": f'"{first}"'})
    assert status == 412, "a write from a browser that has not seen the second version"
    assert "changed since it was read" in refusal(data)
    status, _, data = ask(server, "PUT", TOKEN, blob(fill=9), {"If-None-Match": "*"})
    assert status == 412
    assert "exists already" in refusal(data)
    assert ask(server, "GET", ROOM)[2] == blob(fill=8), "neither refused write changed the room"


@pytest.mark.parametrize(
    "headers", [{}, {"If-Match": '"a"', "If-None-Match": "*"}, {"If-None-Match": '"a"'}], ids=["none", "both", "etag"]
)
def test_a_write_without_one_clear_precondition_is_refused(server, headers):
    status, _, data = ask(server, "PUT", TOKEN, blob(), headers)
    assert status == 428
    assert "If-Match" in refusal(data)


def test_a_write_to_a_room_that_is_gone_is_refused_and_can_make_it_again(server, clock):
    first = make(server)
    clock.now += (relay.EXPIRY_DAYS + 1) * relay.DAY
    assert ask(server, "PUT", TOKEN, blob(), {"If-Match": f'"{first}"'})[0] == 412
    assert ask(server, "PUT", TOKEN, blob(), {"If-None-Match": "*"})[0] == 201


# --------------------------------------------------------------------------- what a room may hold
@pytest.mark.parametrize("size", sorted(relay.BLOB_SIZES))
def test_each_sealed_size_is_taken(server, size):
    assert ask(server, "PUT", TOKEN, blob(size), {"If-None-Match": "*"})[0] == 201


#: The most Caddy passes on to the relay (factorial.caddy's request_body max_size): the relay drains at least this much
#: of a body it refused, so the client reads the refusal rather than a reset connection.
CADDY_MOST = 70 * 1024


@pytest.mark.parametrize("size", [0, 1, SMALL - 1, SMALL + 1, max(relay.BLOB_SIZES) + 1, CADDY_MOST])
def test_any_other_size_is_refused_before_it_is_read(server, tmp_path, size):
    """A file or an image is not a padded study: it does not fit, whatever it holds."""
    assert relay.LINGER_BYTES >= CADDY_MOST
    status, _, data = ask(server, "PUT", TOKEN, blob(size), {"If-None-Match": "*"})
    assert status == 413
    assert "and nothing else" in refusal(data)
    assert list(tmp_path.iterdir()) == []


def test_a_body_in_chunks_or_of_another_type_is_refused(server):
    chunked = {"If-None-Match": "*", "Transfer-Encoding": "chunked"}
    assert ask(server, "PUT", TOKEN, blob(), chunked)[0] == 400
    typed = {"If-None-Match": "*", "Content-Type": "application/json"}
    assert ask(server, "PUT", TOKEN, blob(), typed)[0] == 415


def test_a_request_not_through_caddy_or_to_another_path_is_refused(server):
    assert ask(server, "GET", ROOM, headers={"X-Real-IP": None})[0] == 403
    with socket.create_connection(server.server_address, timeout=10) as connection:
        connection.sendall(f"GET /api/room/{ROOM} HTTP/1.1\r\nX-Real-IP: {ADDRESS}\r\n\r\n".encode())
        connection.shutdown(socket.SHUT_WR)
        assert connection.recv(64).startswith(b"HTTP/1.0 404")


# --------------------------------------------------------------------------- removed, and forgotten
def test_whoever_holds_the_token_can_remove_the_room(server, tmp_path):
    make(server)
    assert ask(server, "DELETE", TOKEN)[0] == 204
    assert ask(server, "GET", ROOM)[0] == 404
    assert list(tmp_path.iterdir()) == []
    assert ask(server, "DELETE", TOKEN)[0] == 404


def test_a_room_nobody_writes_to_is_forgotten_after_its_days(server, store, clock, tmp_path):
    make(server)
    clock.now += relay.EXPIRY_DAYS * relay.DAY - 60
    assert ask(server, "GET", ROOM)[0] == 200, "reading it does not keep it longer, and it is still in its days"
    clock.now += 120
    assert ask(server, "GET", ROOM)[0] == 404
    assert list(tmp_path.iterdir()) == []
    assert store.used == 0


def test_the_sweep_removes_every_room_past_its_days(store, clock, tmp_path):
    store.write("A" * 22, blob(), version=None)
    clock.now += relay.EXPIRY_DAYS * relay.DAY + 1
    store.write("B" * 22, blob(), version=None)  # a write sweeps first
    assert sorted(path.name for path in tmp_path.iterdir()) == ["B" * 22]
    assert store.used == SMALL


def test_a_new_room_past_the_budget_is_refused_and_an_old_one_still_written(server, monkeypatch):
    monkeypatch.setattr(relay, "BUDGET", SMALL)
    first = make(server)
    other = base64.urlsafe_b64encode(bytes(32)).rstrip(b"=").decode()
    status, _, data = ask(server, "PUT", other, blob(), {"If-None-Match": "*"})
    assert status == 507
    assert "all the rooms it can" in refusal(data)
    assert ask(server, "PUT", TOKEN, blob(fill=1), {"If-Match": f'"{first}"'})[0] == 200


def test_the_store_counts_what_its_folder_held_at_start(tmp_path, clock):
    (tmp_path / ("C" * 22)).write_bytes(blob())
    (tmp_path / "notes.txt").write_text("not a room", encoding="utf-8")
    assert relay.Store(tmp_path, clock).used == SMALL


# --------------------------------------------------------------------------- the limits
def test_a_network_makes_a_few_rooms_a_day(server, monkeypatch, clock):
    monkeypatch.setattr(relay, "NEW_ROOMS", 2)
    server.limits = relay.Limits(clock)
    tokens = [base64.urlsafe_b64encode(bytes([n]) * 32).rstrip(b"=").decode() for n in range(3)]
    assert [ask(server, "PUT", t, blob(), {"If-None-Match": "*"})[0] for t in tokens] == [201, 201, 429]
    other = ask(server, "PUT", tokens[2], blob(), {"If-None-Match": "*", "X-Real-IP": "198.51.100.1"})
    assert other[0] == 201, "another network has its own count"
    clock.now += relay.DAY
    assert ask(server, "PUT", tokens[2], blob(), {"If-None-Match": "*"})[0] == 412, "admitted: the room exists"


def test_a_network_reads_a_room_so_many_times_an_hour(server, monkeypatch, clock):
    monkeypatch.setattr(relay, "READS", 3)
    server.limits = relay.Limits(clock)
    make(server)
    statuses = [ask(server, "GET", ROOM)[0] for _ in range(4)]
    assert statuses == [200, 200, 200, 429]
    clock.now += relay.HOUR
    assert ask(server, "GET", ROOM)[0] == 200


def test_a_full_table_of_networks_forgets_an_earlier_window(clock, monkeypatch):
    monkeypatch.setattr(relay, "NETWORKS", 2)
    counter = relay.Counter(5, relay.HOUR, clock)
    assert counter.admit("a")
    assert counter.admit("b")
    assert not counter.admit("c"), "the table is full of this hour's networks"
    clock.now += relay.HOUR
    assert counter.admit("c")


# --------------------------------------------------------------------------- what is logged
def test_the_log_names_no_room_and_no_token(server, capsys):
    version = make(server)
    ask(server, "GET", ROOM)
    ask(server, "PUT", TOKEN, blob(), {"If-Match": '"stale"'})
    ask(server, "PUT", TOKEN, blob(), {"If-Match": f'"{version}"'})
    ask(server, "DELETE", TOKEN)
    logged = capsys.readouterr().err
    assert logged.splitlines()[:2] == ["201 PUT", "200 GET"]
    for secret in (SECRET, TOKEN, ROOM, version):
        assert secret not in logged


# --------------------------------------------------------------------------- the command
def test_forget_removes_the_room_a_link_names(tmp_path, monkeypatch, capsys, clock):
    relay.Store(tmp_path, clock).write(ROOM, blob(), version=None)
    monkeypatch.setenv("ROOM_FOLDER", str(tmp_path))
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)
    assert relay.main(["forget", f"https://factori.al/#room={SECRET}"]) == 0
    assert capsys.readouterr().out == "Removed the room.\n"
    assert list(tmp_path.iterdir()) == []
    assert relay.main(["forget", SECRET]) == 0
    assert "no such room" in capsys.readouterr().out
    assert relay.main(["forget", "#c=abc"]) == 1
    assert relay.main(["anything"]) == 2


def test_the_relay_needs_a_folder(monkeypatch, capsys):
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)
    monkeypatch.delenv("ROOM_FOLDER", raising=False)
    assert relay.main([]) == 1
    assert "names the folder for the rooms" in capsys.readouterr().err
