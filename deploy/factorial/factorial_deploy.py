#!/usr/bin/python3 -I
"""The one command factori.al's deploy key may run.

sshd runs this for every connection of the deploy user (``ForceCommand`` in ``sshd.conf``),
with the site's folder as its only argument and the command the client asked for in
``SSH_ORIGINAL_COMMAND``. Whatever the key asks, it can do these four things and no others:

    upload <commit>     read a gzipped tar of the built app on stdin into releases/<commit>
    activate <commit>   serve releases/<commit>
    rollback            serve the build live before, passing over builds a rollback left
    status              print the live build and the builds kept

The site is whatever the link ``current`` points to, so switching builds is one rename of that
link: each request is answered from one build or the other, never a mix, and a rollback is
instant. ``history`` records every switch, and the last ``KEEP`` builds switched to are kept.

An upload holds only files and folders, under relative names that stay inside its release
folder; tarfile's ``data`` filter checks that again as it unpacks. It must be the app built
from the commit it is uploaded as: its ``wheels/manifest.json``, which
``scripts/build_web_app.py`` writes, has to name that commit. Standard library only; Python
3.12 or later, for the filter.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator

#: How many builds are kept, the live one included: enough to roll back a few deploys.
KEEP = 5
#: The most an upload may send (gzipped) and unpack to, and how many entries it may hold.
#: The app is a few megabytes; these only stop a stream from filling the disk.
MAX_UPLOAD = 64 * 2**20
MAX_UNPACKED = 256 * 2**20
MAX_FILES = 5000
#: An upload, or a release never switched to, is cleared once it is this many seconds old.
STALE = 3600
#: A full commit name: SHA-1, or SHA-256 in a repository that uses it.
COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
USAGE = "this key may only: upload <commit>, activate <commit>, rollback, status"


class RefusedError(Exception):
    """A request this command does not carry out; the message says why, and goes back to the client."""


# --------------------------------------------------------------------------- the site's folder
def releases(root: Path) -> Path:
    """Return the folder holding one folder per build, each named by its commit."""
    return root / "releases"


def live(root: Path) -> str | None:
    """Return the commit the site serves, or None before the first activation."""
    try:
        return (root / "current").readlink().name
    except FileNotFoundError:
        return None


def history(root: Path) -> list[list[str]]:
    """Return every switch, oldest first: ``[time, "activate", commit]`` or ``[time, "rollback", commit, from]``."""
    try:
        lines = (root / "history").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [entry for entry in (line.split() for line in lines) if len(entry) >= 3]


def now() -> str:
    """Return the time, in UTC, as history records it."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def remove(path: Path) -> None:
    """Delete a file, a link, or a folder and everything in it."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


# --------------------------------------------------------------------------- upload
def upload(root: Path, commit: str, stream: BinaryIO) -> str:
    """Unpack the gzipped tar on ``stream`` into the release folder for ``commit``.

    The build is unpacked into a hidden folder beside the releases and renamed into place only
    once it is whole and checked, so a release folder is always a complete build. Uploading a
    commit already there reads the stream and keeps the build uploaded first.
    """
    target = releases(root) / commit
    if target.is_dir():
        read(stream)  # all of it, so the sender's pipe closes cleanly
        return f"{commit} is already uploaded"
    incoming = Path(tempfile.mkdtemp(prefix=".incoming-", dir=releases(root)))
    try:
        unpack(stream, incoming)
        check_build(incoming, commit)
        make_readable(incoming)
        incoming.rename(target)
    finally:
        if incoming.exists():
            shutil.rmtree(incoming)
    return f"uploaded {commit}"


def read(stream: BinaryIO, into: BinaryIO | None = None) -> None:
    """Read ``stream`` to its end, into ``into`` if given, refusing more than ``MAX_UPLOAD`` bytes."""
    sent = 0
    while chunk := stream.read(2**16):
        sent += len(chunk)
        if sent > MAX_UPLOAD:
            msg = f"the upload is larger than {MAX_UPLOAD // 2**20} MiB"
            raise RefusedError(msg)
        if into is not None:
            into.write(chunk)


def unpack(stream: BinaryIO, folder: Path) -> None:
    """Unpack a gzipped tar from ``stream`` into ``folder``, after checking every entry it holds."""
    if not hasattr(tarfile, "data_filter"):
        msg = "this Python has no tarfile data filter; the deploy command needs Python 3.12 or later"
        raise RefusedError(msg)
    with tempfile.TemporaryFile() as spool:
        read(stream, spool)
        spool.seek(0)
        try:
            with tarfile.open(fileobj=spool, mode="r:gz") as archive:
                members = archive.getmembers()
                check_entries(members)
                archive.extractall(folder, filter="data")
        except (tarfile.TarError, EOFError, OSError, zlib.error) as error:
            msg = f"the upload could not be unpacked: {error}"
            raise RefusedError(msg) from error


def check_entries(members: list[tarfile.TarInfo]) -> None:
    """Refuse an archive that is too large, or holds anything but files and folders under plain relative names.

    The app has no links, so none is accepted: no link can lead out of the release's folder,
    whatever the filter's version would make of it.
    """
    if len(members) > MAX_FILES:
        msg = f"the upload holds more than {MAX_FILES} entries"
        raise RefusedError(msg)
    if sum(member.size for member in members) > MAX_UNPACKED:
        msg = f"the upload unpacks to more than {MAX_UNPACKED // 2**20} MiB"
        raise RefusedError(msg)
    for member in members:
        name = PurePosixPath(member.name)
        if not (member.isfile() or member.isdir()):
            msg = f"the upload holds {member.name!r}, which is neither a file nor a folder"
            raise RefusedError(msg)
        if name.is_absolute() or ".." in name.parts:
            msg = f"the upload names {member.name!r}, outside its own folder"
            raise RefusedError(msg)


def check_build(folder: Path, commit: str) -> None:
    """Refuse a build that is not the app, or not the app built from ``commit``."""
    if not (folder / "index.html").is_file():
        msg = "the upload has no index.html"
        raise RefusedError(msg)
    try:
        manifest = json.loads((folder / "wheels" / "manifest.json").read_text(encoding="utf-8"))
        built = manifest.get("commit") if isinstance(manifest, dict) else None
    except (OSError, ValueError):
        built = None
    if built != commit:
        msg = f"the upload was built from {built or 'an unknown commit'}, not {commit}"
        raise RefusedError(msg)


def make_readable(folder: Path) -> None:
    """Let the web server read every file and enter every folder of the build, and nobody else write them."""
    for path in [folder, *folder.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)


# --------------------------------------------------------------------------- switching builds
def activate(root: Path, commit: str, *, rolled_back_from: str | None = None) -> str:
    """Serve the build of ``commit``, record the switch, and clear the builds no longer kept."""
    if not (releases(root) / commit).is_dir():
        msg = f"{commit} has not been uploaded"
        raise RefusedError(msg)
    staged = root / f".current-{os.getpid()}"
    staged.unlink(missing_ok=True)
    staged.symlink_to(Path("releases") / commit)  # relative, so the site's folder can move
    staged.replace(root / "current")  # rename(2) over the old link: the switch is atomic
    entry = ["rollback", commit, rolled_back_from] if rolled_back_from else ["activate", commit]
    with (root / "history").open("a", encoding="utf-8") as log:
        log.write(" ".join([now(), *entry]) + "\n")
    prune(root)
    return f"serving {commit}"


def rollback(root: Path) -> str:
    """Serve the most recent earlier build still kept, passing over every build a rollback left.

    A second rollback therefore goes further back, rather than returning to the build the first
    one moved away from.
    """
    current = live(root)
    entries = history(root)
    left = {entry[3] for entry in entries if entry[1] == "rollback" and len(entry) > 3}
    for entry in reversed(entries):
        commit = entry[2]
        if commit != current and commit not in left and (releases(root) / commit).is_dir():
            activate(root, commit, rolled_back_from=current)
            return f"rolled back from {current} to {commit}"
    msg = "there is no earlier build to go back to"
    raise RefusedError(msg)


def kept(root: Path) -> list[str]:
    """Return the builds to keep, most recent first: the last ``KEEP`` switched to."""
    commits: list[str] = []
    for entry in reversed(history(root)):
        if entry[2] not in commits:
            commits.append(entry[2])
    return commits[:KEEP]


def prune(root: Path) -> None:
    """Delete the builds no longer kept, and anything an interrupted upload or switch left behind.

    The live build is never deleted, and neither is a build uploaded within the hour and not yet
    switched to: it is the next deploy's. Only a switch prunes, under the site's lock.
    """
    keep = {*kept(root), live(root)}
    switched = {entry[2] for entry in history(root)}
    recent = time.time() - STALE
    for path in releases(root).iterdir():
        if path.name in keep:
            continue
        uploading = path.name.startswith(".incoming-")
        waiting = COMMIT.fullmatch(path.name) is not None and path.name not in switched
        if (uploading or waiting) and path.lstat().st_mtime > recent:
            continue
        remove(path)
    for staged in root.glob(".current-*"):
        staged.unlink()


def status(root: Path) -> str:
    """Report the live build, when it was switched to, and the builds kept."""
    current = live(root)
    since = next((entry[0] for entry in reversed(history(root)) if entry[2] == current), "?")
    return f"live: {current} since {since}\nkept: {' '.join(kept(root))}" if current else "live: none\nkept:"


# --------------------------------------------------------------------------- the request
@contextlib.contextmanager
def locked(root: Path) -> Iterator[None]:
    """Hold the site's lock, so two switches never run at once."""
    with (root / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)  # released when the file closes
        yield


def handle(root: Path, request: str, stream: BinaryIO) -> str:
    """Carry out one request from the deploy key, holding the site's lock while it switches builds."""
    match request.split():
        case ["upload", commit] if COMMIT.fullmatch(commit):
            return upload(root, commit, stream)
        case ["activate", commit] if COMMIT.fullmatch(commit):
            with locked(root):
                return activate(root, commit)
        case ["rollback"]:
            with locked(root):
                return rollback(root)
        case ["status"]:
            return status(root)
        case _:
            raise RefusedError(USAGE)


def main(argv: list[str] | None = None) -> int:
    """Run the request in ``SSH_ORIGINAL_COMMAND`` against the site folder given as the argument."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: factorial-deploy <site folder>, with the request in SSH_ORIGINAL_COMMAND", file=sys.stderr)
        return 2
    root = Path(args[0])
    releases(root).mkdir(parents=True, exist_ok=True)
    try:
        print(handle(root, os.environ.get("SSH_ORIGINAL_COMMAND", ""), sys.stdin.buffer))
    except RefusedError as reason:
        print(f"refused: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
