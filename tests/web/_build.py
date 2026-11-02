"""The built app for the live and wasm tiers: what ``--web-app`` names, or a build of this checkout.

A build is kept under ``~/.cache/process-improve/web-app``, named by a digest of everything it
is made from, so it is made once per change and never inside the checkout. xdist workers wait
for each other's build rather than race it. The app under test can also be a URL, such as the
published one: its wheel and its ``bootstrap.py`` are then downloaded for the wasm tier.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from tests.web._support import BUILD_SCRIPT, ROOT, WEB, no_bytecode

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

CACHE = Path.home() / ".cache" / "process-improve" / "web-app"

#: Builds kept in the cache; older ones are removed after a new build.
KEEP = 3

#: How long a worker waits for another's build, and when a lock left behind is stale (seconds).
LOCK_TIMEOUT = 900


def digest() -> str:
    """Name a build by everything it is made from: the package, its metadata, the page and the build script."""
    sources = [*(ROOT / "src").rglob("*"), ROOT / "pyproject.toml", BUILD_SCRIPT, *WEB.rglob("*")]
    files = sorted(p for p in sources if p.is_file() and "__pycache__" not in p.parts and "node_modules" not in p.parts)
    hashed = hashlib.sha256()
    for path in files:
        hashed.update(path.relative_to(ROOT).as_posix().encode())
        hashed.update(path.read_bytes())
    return hashed.hexdigest()[:16]


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold a lock file for the block, waiting for another holder to finish first."""
    deadline = time.monotonic() + LOCK_TIMEOUT
    while True:
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            with contextlib.suppress(FileNotFoundError):
                if time.time() - path.stat().st_mtime > LOCK_TIMEOUT:
                    path.unlink()  # left behind by a run that was killed
            if time.monotonic() > deadline:
                message = f"waited {LOCK_TIMEOUT} s for another build of the app ({path})"
                raise TimeoutError(message) from None
            time.sleep(0.5)
    try:
        yield
    finally:
        os.close(handle)
        path.unlink(missing_ok=True)


def build(build_script: ModuleType) -> Path:
    """Return a build of this checkout's app, making it if the cache has none for these sources."""
    key = digest()
    target = CACHE / key
    if (target / "demo.json").is_file():
        return target
    CACHE.mkdir(parents=True, exist_ok=True)
    with locked(CACHE / f"{key}.lock"):
        if not (target / "demo.json").is_file():
            partial, dist = CACHE / f"{key}.partial", CACHE / f"{key}.dist"
            with no_bytecode():
                build_script.assemble(partial, dist=dist)
            shutil.rmtree(dist, ignore_errors=True)
            shutil.rmtree(target, ignore_errors=True)
            partial.rename(target)
            builds = [p for p in CACHE.iterdir() if p.is_dir() and len(p.name) == len(key)]
            for old in sorted(builds, key=os.path.getmtime)[:-KEEP]:
                shutil.rmtree(old, ignore_errors=True)
    return target


def app_file(app: Path | str, name: str, scratch: Path) -> Path:
    """Return a file of the built app on disk, downloading it first when the app is a URL."""
    if isinstance(app, Path):
        return app / name
    target = scratch / Path(name).name
    with urllib.request.urlopen(f"{app}{name}", timeout=120) as response:
        target.write_bytes(response.read())
    return target


def wheel(app: Path | str, scratch: Path) -> Path:
    """Return the wheel the built app's worker installs, as its manifest names it."""
    manifest = json.loads(app_file(app, "wheels/manifest.json", scratch).read_text(encoding="utf-8"))
    return app_file(app, f"wheels/{manifest['wheel']}", scratch)
