"""The app's copy on GitHub Pages sends a reader on to factori.al, unless studies are kept at its address (#1237).

The copy at ``/app`` gets ``scripts/pages_moved.js`` in each page from ``scripts/move_pages_copy.py``. Studies are kept
in a browser's storage for one address, and do not move with a redirect, so a browser that keeps some stays and is
told how to take them along. These pages stand in for the copy: the signpost is all of it that is under test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.web._browser import serve

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).parents[3]
PAGE = "<!doctype html><html lang='en'><head><title>{}</title></head><body><p>The old copy</p></body></html>"

#: Two studies saved by the page at this address, as store.js keeps them.
SAVE_TWO = """() => new Promise((resolve, reject) => {
  const request = indexedDB.open("factorial", 1);
  request.onupgradeneeded = () => request.result.createObjectStore("campaigns", { keyPath: "id" });
  request.onerror = () => reject(request.error);
  request.onsuccess = () => {
    const tx = request.result.transaction("campaigns", "readwrite");
    tx.objectStore("campaigns").put({ id: "a", updated: 1 });
    tx.objectStore("campaigns").put({ id: "b", updated: 2 });
    tx.oncomplete = () => { request.result.close(); resolve(); };
    tx.onerror = () => reject(tx.error);
  };
})"""


def _mover() -> Any:  # noqa: ANN401 - a module
    spec = importlib.util.spec_from_file_location("move_pages_copy", ROOT / "scripts" / "move_pages_copy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pages(tmp_path: Path) -> Iterator[str]:
    """Serve a copy laid out as Pages serves it, under ``process-improve/app/``, and an unmarked page beside it."""
    app = tmp_path / "process-improve" / "app"
    (app / "map").mkdir(parents=True)
    (app / "index.html").write_text(PAGE.format("Factorial"), encoding="utf-8")
    (app / "map" / "index.html").write_text(PAGE.format("The concept map"), encoding="utf-8")
    assert _mover().mark(app) == 2
    assert _mover().mark(app) == 0, "a second run leaves the pages as they are"
    (tmp_path / "seed.html").write_text(PAGE.format("Seed"), encoding="utf-8")  # at the same address, no signpost
    yield from serve(tmp_path)


@pytest.fixture
def browser_context(chromium: Any) -> Iterator[Any]:  # noqa: ANN401 - a Playwright Browser
    """Open a context of its own, which answers factori.al with a page of its own instead of going there."""
    context = chromium.new_context()
    context.route(
        "https://factori.al/**", lambda route: route.fulfill(content_type="text/html", body="<title>f</title>")
    )
    yield context
    context.close()


def test_a_browser_with_no_study_here_goes_on_to_the_same_page_at_factorial(browser_context, pages):
    page = browser_context.new_page()
    page.goto(f"{pages}process-improve/app/map/?level=2#main-effect")
    page.wait_for_url("https://factori.al/**")
    assert page.url == "https://factori.al/map/?level=2#main-effect"
    page.goto(f"{pages}seed.html")
    names = page.evaluate("async () => (await indexedDB.databases()).map((db) => db.name)")
    assert "factorial" not in names, "looking for studies made no store"


def test_a_browser_that_keeps_studies_here_stays_and_is_told_how_to_take_them_along(browser_context, pages):
    page = browser_context.new_page()
    page.goto(f"{pages}seed.html")
    page.evaluate(SAVE_TWO)
    page.goto(f"{pages}process-improve/app/map/")
    banner = page.locator("#factorial-moved")
    banner.wait_for()
    assert page.url == f"{pages}process-improve/app/map/"
    said = " ".join(banner.text_content().split())
    assert said.startswith("Factorial has moved to factori.al.")
    assert "This browser keeps 2 studies at this address, and they do not move by themselves." in said
    assert "press Download all, then drop that file into Your studies at factori.al." in said
    assert banner.get_by_role("link", name="factori.al").get_attribute("href") == "https://factori.al/map/"
    assert banner.get_by_role("link", name="Your studies").get_attribute("href") == "/process-improve/app/"
