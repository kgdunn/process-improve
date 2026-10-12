"""``deploy/factorial/factorial_deploy.py``, the one command factori.al's deploy key may run.

A request reaches it as sshd hands it over: the site's folder as the argument, the client's
command in ``SSH_ORIGINAL_COMMAND``, the upload on stdin. A key that leaks can then upload a
build and switch to it, and nothing more: these tests send what such a key could send.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

pytest.importorskip("fcntl", reason="the server is Linux; its deploy command locks with fcntl")
if not hasattr(tarfile, "data_filter"):
    pytest.skip("the deploy command unpacks with tarfile's data filter", allow_module_level=True)

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "factorial" / "factorial_deploy.py"
A, B, C, D, E, F, G = (str(n) * 40 for n in range(1, 8))


def load():
    spec = importlib.util.spec_from_file_location("factorial_deploy", SCRIPT)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deploy = load()


def tar(entries: list[tuple[tarfile.TarInfo, bytes]]) -> bytes:
    """Return a gzipped tar of the given entries."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for info, data in entries:
            info.size = len(data) if info.isfile() else 0
            archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
    return buffer.getvalue()


def entry(name: str, data: bytes = b"", kind: bytes = tarfile.REGTYPE, link: str = "") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.type, info.linkname, info.mode = kind, link, 0o600  # as an artifact hands files over
    return info, data


def build(commit: str, **files: str) -> bytes:
    """Return the app built from ``commit`` as the workflow sends it: ``tar -C app -czf - .``."""
    contents = {"index.html": "<!doctype html>", "wheels/manifest.json": json.dumps({"commit": commit}), **files}
    return tar([entry(f"./{name}", text.encode()) for name, text in contents.items()])


@pytest.fixture
def site(tmp_path: Path) -> Path:
    root = tmp_path / "srv" / "factorial"
    deploy.releases(root).mkdir(parents=True)
    return root


def ask(site: Path, request: str, data: bytes = b"") -> str:
    return deploy.handle(site, request, io.BytesIO(data))


def ship(site: Path, *commits: str) -> None:
    """Upload and switch to each build in turn, as the workflow does on each push to main."""
    for commit in commits:
        ask(site, f"upload {commit}", build(commit))
        ask(site, f"activate {commit}")


def everything(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*")}


def test_an_uploaded_build_is_served_once_switched_to(site):
    assert ask(site, f"upload {A}", build(A)) == f"uploaded {A}"
    assert deploy.live(site) is None, "uploading alone changes nothing a reader sees"
    assert ask(site, f"activate {A}") == f"serving {A}"
    assert (site / "current" / "index.html").read_text() == "<!doctype html>"
    assert (site / "current").readlink() == Path("releases") / A, "a relative link, so the folder can move"
    assert ask(site, "status") == f"live: {A} since {deploy.history(site)[-1][0]}\nkept: {A}"


def test_every_file_is_readable_by_the_web_server_and_writable_by_nobody_else(site):
    ship(site, A)
    release = deploy.releases(site) / A
    assert {oct(p.stat().st_mode & 0o777) for p in release.rglob("*") if p.is_file()} == {"0o644"}
    assert {oct(p.stat().st_mode & 0o777) for p in [release, *release.rglob("*")] if p.is_dir()} == {"0o755"}


def test_a_switch_leaves_no_staged_link_behind(site):
    ship(site, A, B)
    assert not list(site.glob(".current-*"))
    assert deploy.live(site) == B
    assert (deploy.releases(site) / A / "index.html").is_file(), "the build before stays, for a rollback"


@pytest.mark.parametrize(
    ("manifest", "reason"),
    [
        (json.dumps({"commit": B}), f"the upload was built from {B}, not {A}"),
        (json.dumps({"wheel": "x.whl"}), f"the upload was built from an unknown commit, not {A}"),
        ("not json", f"the upload was built from an unknown commit, not {A}"),
        (json.dumps([A]), f"the upload was built from an unknown commit, not {A}"),
    ],
)
def test_an_upload_must_be_the_app_built_from_the_commit_it_is_uploaded_as(site, manifest, reason):
    data = tar([entry("./index.html", b"x"), entry("./wheels/manifest.json", manifest.encode())])
    with pytest.raises(deploy.RefusedError, match=reason):
        ask(site, f"upload {A}", data)
    assert everything(site) == {"releases"}, "nothing is left of a refused upload"


def test_an_upload_without_the_page_is_refused(site):
    with pytest.raises(deploy.RefusedError, match=r"the upload has no index\.html"):
        ask(site, f"upload {A}", tar([entry("./wheels/manifest.json", json.dumps({"commit": A}).encode())]))
    assert everything(site) == {"releases"}


@pytest.mark.parametrize(
    "intruder",
    [
        entry("{outside}/evil", b"written outside"),
        entry("../escaped", b"x"),
        entry("./wheels/../../escaped", b"x"),
        entry("./outside", kind=tarfile.SYMTYPE, link="/etc"),
        entry("./up", kind=tarfile.SYMTYPE, link="../.."),
        entry("./inside", kind=tarfile.SYMTYPE, link="index.html"),
        entry("./hard", kind=tarfile.LNKTYPE, link="index.html"),
        entry("./device", kind=tarfile.CHRTYPE),
        entry("./pipe", kind=tarfile.FIFOTYPE),
    ],
    ids=["absolute", "parent", "parent-inside", "link-out", "link-up", "link-in", "hard-link", "device", "fifo"],
)
def test_an_upload_holds_only_files_and_folders_inside_its_own_folder(site, intruder):
    info, data = intruder
    info = copy.copy(info)
    info.name = info.name.format(outside=site.parent)  # the absolute path is this test's own folder
    page = entry("./index.html", b"x")
    manifest = entry("./wheels/manifest.json", json.dumps({"commit": A}).encode())
    with pytest.raises(deploy.RefusedError, match=r"the upload (names|holds)"):
        ask(site, f"upload {A}", tar([page, manifest, (info, data)]))
    assert everything(site.parent) == {"factorial", "factorial/releases"}, "nothing written, inside or out"


@pytest.mark.parametrize(
    ("limit", "value", "reason"),
    [
        ("MAX_UPLOAD", 100, "the upload is larger than"),
        ("MAX_UNPACKED", 10, "the upload unpacks to more than"),
        ("MAX_FILES", 1, "the upload holds more than 1 entries"),
    ],
)
def test_an_upload_past_a_limit_is_refused(site, monkeypatch, limit, value, reason):
    monkeypatch.setattr(deploy, limit, value)
    with pytest.raises(deploy.RefusedError, match=reason):
        ask(site, f"upload {A}", build(A, padding="x" * 200))
    assert everything(site) == {"releases"}


@pytest.mark.parametrize("data", [b"", b"not a tar", build(A)[:40]], ids=["empty", "text", "cut"])
def test_an_upload_that_is_not_a_gzipped_tar_is_refused(site, data):
    with pytest.raises(deploy.RefusedError, match="the upload could not be unpacked"):
        ask(site, f"upload {A}", data)
    assert everything(site) == {"releases"}


def test_uploading_a_commit_again_keeps_the_build_uploaded_first(site):
    ship(site, A)
    again = io.BytesIO(build(A, **{"index.html": "changed"}))
    assert deploy.handle(site, f"upload {A}", again) == f"{A} is already uploaded"
    assert again.read() == b"", "the stream is read to its end, so the sender's pipe closes cleanly"
    assert (site / "current" / "index.html").read_text() == "<!doctype html>"


@pytest.mark.parametrize(
    "request_",
    [
        "",
        "sh",
        "bash -i",
        "rm -rf /",
        "upload",
        "upload main",
        f"upload {A[:7]}",
        "upload " + "AB" * 20,  # commit names are lower case
        f"upload {A} extra",
        f"upload {A};id",
        "activate ../../../etc",
        "rollback now",
        "status; id",
        "rsync --server -logDtpre.iLsfxCIvu . /srv/factorial/releases/",
        "scp -t /srv/factorial",
        "internal-sftp",
    ],
)
def test_nothing_but_the_four_requests_is_carried_out(site, request_):
    ship(site, A)
    before = everything(site)
    with pytest.raises(deploy.RefusedError, match="this key may only"):
        ask(site, request_, build(A))
    assert everything(site) == before


def test_a_build_never_uploaded_cannot_be_switched_to(site):
    ship(site, A)
    with pytest.raises(deploy.RefusedError, match=f"{B} has not been uploaded"):
        ask(site, f"activate {B}")
    assert deploy.live(site) == A


def test_a_rollback_goes_back_and_never_returns_to_a_build_it_left(site):
    ship(site, A, B, C)
    assert ask(site, "rollback") == f"rolled back from {C} to {B}"
    assert ask(site, "rollback") == f"rolled back from {B} to {A}", "not forward again, to C"
    with pytest.raises(deploy.RefusedError, match="there is no earlier build to go back to"):
        ask(site, "rollback")
    assert deploy.live(site) == A
    assert [e[1:] for e in deploy.history(site)][-2:] == [["rollback", B, C], ["rollback", A, B]]


def test_a_rollback_needs_an_earlier_build(site):
    with pytest.raises(deploy.RefusedError, match="there is no earlier build"):
        ask(site, "rollback")
    ship(site, A)
    with pytest.raises(deploy.RefusedError, match="there is no earlier build"):
        ask(site, "rollback")


def test_the_last_five_builds_switched_to_are_kept(site):
    ship(site, A, B, C, D, E, F, G)
    assert sorted(p.name for p in deploy.releases(site).iterdir()) == sorted([C, D, E, F, G])
    assert ask(site, "status").endswith(f"kept: {G} {F} {E} {D} {C}")
    assert len(deploy.history(site)) == 7, "the history keeps every switch"


def test_a_build_waiting_to_be_switched_to_is_kept_for_an_hour(site):
    ask(site, f"upload {B}", build(B))
    leftover = deploy.releases(site) / ".incoming-crashed"
    leftover.mkdir()
    ship(site, A)
    assert {B, ".incoming-crashed"} <= {p.name for p in deploy.releases(site).iterdir()}
    hour_ago = time.time() - deploy.STALE - 60
    for path in (deploy.releases(site) / B, leftover):
        os.utime(path, (hour_ago, hour_ago))
    ship(site, C)
    assert sorted(p.name for p in deploy.releases(site).iterdir()) == sorted([A, C])


def test_the_command_runs_as_sshd_runs_it(site):
    """As sshd runs a forced command: the request in SSH_ORIGINAL_COMMAND, the upload on stdin."""

    def sshd(request: str, data: bytes = b"") -> subprocess.CompletedProcess:
        env = {"SSH_ORIGINAL_COMMAND": request, "PATH": os.environ.get("PATH", "")}
        return subprocess.run(  # noqa: S603 - the script under test, with a fixed argument list
            [sys.executable, "-I", str(SCRIPT), str(site)], input=data, env=env, capture_output=True, check=False
        )

    assert sshd(f"upload {A}", build(A)).stdout.decode() == f"uploaded {A}\n"
    assert sshd(f"activate {A}").stdout.decode() == f"serving {A}\n"
    refused = sshd("sh -c id")
    assert refused.returncode == 1
    assert refused.stderr.decode().startswith("refused: this key may only")
    assert sshd("status").stdout.decode().startswith(f"live: {A} since ")


def test_the_command_needs_the_site_folder_and_nothing_else():
    assert deploy.main([]) == 2
    assert deploy.main(["/srv/factorial", "extra"]) == 2
