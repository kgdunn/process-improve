"""After a deploy: ``tools/wait_for_web_app.py`` waits for the published app to be the pushed commit's.

The check after each deploy (``docs.yml``) runs the smoke tests against the published app, so it
must first be sure that app is the one just deployed, not the one before it still in a cache.
Here the app is a local directory with only a manifest, changed as a deploy would change it.
"""

from __future__ import annotations

import http.client
import importlib.util
import io
import json
import threading
from typing import TYPE_CHECKING

import pytest

from tests.web._browser import serve
from tests.web._support import ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import ModuleType

WORKFLOWS = ROOT / ".github" / "workflows"


@pytest.fixture(scope="module")
def waiter() -> ModuleType:
    """Load ``tools/wait_for_web_app.py``, which is a script, not a module of the package."""
    spec = importlib.util.spec_from_file_location("wait_for_web_app", ROOT / "tools" / "wait_for_web_app.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def app(tmp_path: Path) -> Iterator[str]:
    """Serve an app that is only its manifest, and return its URL."""
    (tmp_path / "wheels").mkdir()
    yield from serve(tmp_path)


def publish(directory: Path, commit: str) -> None:
    """Deploy the app built from ``commit``, as far as the manifest says."""
    (directory / "wheels" / "manifest.json").write_text(
        json.dumps({"wheel": "w.whl", "commit": commit}), encoding="utf-8"
    )


def test_it_waits_until_the_published_manifest_names_the_commit(waiter, app, tmp_path):
    publish(tmp_path, "the one before")
    deploy = threading.Timer(0.3, publish, (tmp_path, "this one"))
    deploy.start()
    assert waiter.wait(app, "this one", timeout=10, interval=0.05) == "this one"
    deploy.join()


def test_it_gives_up_in_time_and_says_what_it_last_saw(waiter, app, tmp_path):
    publish(tmp_path, "the one before")
    assert waiter.wait(app, "this one", timeout=0.2, interval=0.05) == "the one before"


def test_an_app_with_no_manifest_yet_is_waited_for(waiter, app):
    assert waiter.published_commit(app) is None


def test_a_manifest_cut_short_by_a_deploy_is_waited_for(waiter, monkeypatch):
    """A deploy that rewrites the manifest while it is served can end the response early; the waiter looks again."""

    class CutShort(io.BytesIO):
        def read(self, *_args: object) -> bytes:
            raise http.client.IncompleteRead(b"", 46)

    monkeypatch.setattr(waiter.urllib.request, "urlopen", lambda *_args, **_kwargs: CutShort())
    assert waiter.published_commit("http://127.0.0.1/app/") is None


def test_the_deploy_is_checked_once_it_is_published():
    """docs.yml waits for this commit's app, then runs the smoke tests against the published one."""
    docs = (WORKFLOWS / "docs.yml").read_text(encoding="utf-8")
    assert 'tools/wait_for_web_app.py "$APP" "$GITHUB_SHA"' in docs
    assert '--web-app="$APP"' in docs
