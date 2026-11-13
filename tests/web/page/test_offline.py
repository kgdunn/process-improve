"""The copy of the page kept for use without a network: ``web/sw.js`` serves it, and ``web/offline.js`` makes it.

The service worker holds no list of files: it has a rule for each kind of address. These tests keep its rules, and
the lists the page makes its copy from, in step with the addresses the page really loads from, and the build's
switch in step with what retires the worker. What the two files do is run under Node.js (``tests/web/js``).
"""

from __future__ import annotations

import re

from tests.web._support import BUILD_SCRIPT, page, source

BUILD = BUILD_SCRIPT.read_text(encoding="utf-8")


def test_the_worker_keeps_what_the_page_loads_from_the_addresses_it_loads_them_from():
    """A CDN moved in one file and not the other would leave the worker keeping nothing, and saying nothing."""
    worker = source("sw.js")
    pyodide = re.search(r'const PYODIDE = "([^"]+)";', worker).group(1)
    plotly_js = re.search(r'const PLOTLY_JS = "([^"]+)";', worker).group(1)
    assert f"`{pyodide}v${{runtime.pyodide}}/${{runtime.distribution}}/`" in source("python-boot.js")
    assert f"`{plotly_js}${{version}}/plotly.min.js`" in source("charts.js")
    # The page's lists of what to copy are written from the same two addresses, and in the same shapes.
    offline = source("offline.js")
    assert f'const PYODIDE = "{pyodide}";' in offline
    assert f'const PLOTLY_JS = "{plotly_js}";' in offline
    assert "`v${runtime.pyodide}/${runtime.distribution}`" in offline
    assert "`${PLOTLY_JS}${manifest.plotly_js}/plotly.min.js`" in offline
    assert 'const PYPI_FILES = "files.pythonhosted.org";' in offline
    assert 'hostname === "files.pythonhosted.org"' in worker


def test_plotly_js_is_asked_for_in_a_way_the_worker_can_keep():
    """A script asked for without CORS comes back opaque: the worker could not tell a whole copy from a refusal."""
    assert 'script.crossOrigin = "anonymous";' in source("charts.js")
    assert 'response.type !== "opaque"' in source("sw.js")


def test_the_wheels_folder_the_worker_keeps_is_the_build_s(build_script):
    """The build names the folder by sixteen hexadecimal digits of a digest: the worker's rule reads just that."""
    assert "hexdigest()[:16]" in BUILD
    assert r"/^wheels\/[0-9a-f]{16}\//" in source("sw.js")
    assert build_script.GLUE.endswith(".zip")


def test_the_build_s_switch_is_what_retires_the_worker():
    """The build puts the retiring worker at the worker's address, and tells the page to register none."""
    assert '"service_worker": service_worker,' in BUILD
    assert '"--no-service-worker"' in BUILD
    assert 'shutil.copy2(WEB_SRC / "sw-retire.js", out_dir / "sw.js")' in BUILD
    assert "manifest.service_worker === false" in source("offline.js"), "and the page unregisters it"
    assert 'name.startsWith("factorial-")' in source("sw-retire.js"), "the copies it deletes are the worker's"


def test_the_worker_reads_nothing_of_the_build(build_script):
    """No list of files, no manifest and no version: nothing in the worker can fall behind a deploy."""
    worker = source("sw.js")
    assert not re.search(r"manifest\.json|runtime\.json|\d+\.\d+\.\d+", worker)
    assert not any(f'"{name}"' in worker for name in build_script.APP_FILES)


def test_the_page_registers_the_worker_it_ships_and_names_the_copies_as_it_does():
    assert 'navigator.serviceWorker.register("sw.js")' in source("offline.js")
    names = re.findall(r'const (KEPT|APP|MADE) = "([^"]+)";', source("sw.js"))
    assert {name for name, _ in names} == {"KEPT", "APP", "MADE"}, "the two copies, and the mark that they are made"
    assert dict(names) == dict(re.findall(r'export const (KEPT|APP|MADE) = "([^"]+)";', source("offline.js")))
    assert all(name.startswith("factorial-") for _, name in names), "what retiring the worker deletes"
    made = dict(names)["MADE"]
    assert f'name.startsWith("{made.rsplit("-", 1)[0]}-")' in source("sw-retire.js"), "and its mark, which goes first"


def test_the_footer_has_the_copy_s_words_and_its_two_buttons_hidden_until_there_is_a_copy():
    html = " ".join(source("index.html").split())
    assert '<p id="offline" hidden tabindex="-1"> <span role="status"></span>' in html, "said aloud, and focusable"
    assert {"offline-keep", "offline-remove"} <= set(page().ids)
