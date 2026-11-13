"""The copy of the page kept for use without a network: ``web/sw.js`` serves it, and ``web/offline.js`` makes it.

The service worker holds no list of files: it has a rule for each kind of address. These tests keep its rules in
step with the addresses the page really loads from, and the build's switch in step with what retires the worker.
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
    offline = source("offline.js")
    assert f"`{pyodide}v${{runtime.pyodide}}/${{runtime.distribution}}/`" in offline
    assert f"`{plotly_js}${{version}}/plotly.min.js`" in offline


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
    assert '"service_worker": service_worker,' in BUILD
    assert '"--no-service-worker"' in BUILD
    assert "said.service_worker === false" in source("sw.js"), "the worker retires itself"
    assert "manifest.service_worker === false" in source("offline.js"), "and the page unregisters it"


def test_the_page_registers_the_worker_it_ships_and_names_the_copies_as_it_does():
    assert 'navigator.serviceWorker.register("sw.js")' in source("offline.js")
    names = re.findall(r'const (KEPT|APP) = "([^"]+)";', source("sw.js"))
    assert dict(names) == dict(re.findall(r'export const (KEPT|APP) = "([^"]+)";', source("offline.js")))
    assert all(name.startswith("factorial-") for _, name in names), "what retiring the worker deletes"


def test_the_footer_has_the_copy_s_words_and_its_two_buttons_hidden_until_there_is_a_copy():
    html = " ".join(source("index.html").split())
    assert '<p id="offline" hidden>' in html
    assert {"offline-keep", "offline-remove"} <= set(page().ids)
