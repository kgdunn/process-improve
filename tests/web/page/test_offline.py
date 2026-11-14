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
    box = '<div id="offline" class="foot__device" hidden tabindex="-1">'
    assert f'{box} <p class="foot__head">On this device</p> <p><span role="status"></span></p>' in html, (
        "said aloud, and focusable; the words are in the box's first span, where offline.js writes them"
    )
    assert {"offline-keep", "offline-remove"} <= set(page().ids)


#: The modules the page loads on demand that a page without a network has no use for: the feedback form's
#: picture of the page, which is sent over the network.
NEEDS_A_NETWORK = {"vendor/modern-screenshot.mjs"}

IMPORTED = re.compile(r'from "\./([\w./-]+)"')
ON_DEMAND = re.compile(r"""import\(\s*[^)]*?["'`]\./([\w./-]+\.m?js)""")


def test_the_copy_holds_every_module_the_page_loads_on_demand():
    """The copy holds what a visit loaded, so a module loaded later, when a chart is drawn, must be named to it.

    A first visit that drew no chart kept a copy with no ``figure.js``, and a chart drawn without a network could
    not open in its window: only the live tier saw it. The copy is the designed-experiments page's, so the modules
    are those ``app.js`` reaches: the concept map is a page of its own, whose files the worker keeps as they pass.
    """
    seen: set[str] = set()
    on_demand: set[str] = set()
    reached = ["app.js"]
    while reached:
        name = reached.pop()
        if name in seen or "/" in name:  # a vendored file imports none of the app's modules
            continue
        seen.add(name)
        later = set(ON_DEMAND.findall(source(name)))
        on_demand |= later
        reached += [*IMPORTED.findall(source(name)), *later]
    assert "charts.js" in seen, "the modules app.js imports are followed"
    assert on_demand == {"figure.js", "vendor/modern-screenshot.mjs"}, "a new one is kept, or needs a network"
    kept = set(re.findall(r'"([\w./-]+)"', re.search(r"const onDemand = .*", source("offline.js")).group()))
    assert on_demand - NEEDS_A_NETWORK == kept


def test_the_copy_holds_the_small_pages_the_footer_leads_to():
    """The app's page never loads its other pages, so a copy made of what a visit loaded left their links dead offline.

    Only the live tier saw it. The two small ones are named to the copy, each with the files it asks for that the
    copy does not hold already: what opens the app is kept first, and the app's styles are loaded by every visit.
    """

    def named(line: str) -> set[str]:
        return set(re.findall(r'"([\w./-]+)"', re.search(line, source("offline.js")).group()))

    kept, opening = named(r"const linked = .*"), named(r"const opening = .*")
    pages = ("features.html", "about.html")
    asks = re.compile(r'(?:src=|from |fetch\(|read\()"(?:\./)?([\w./-]+)"')
    asked = {file for page in pages for file in asks.findall(source(page))}
    assert {"features.js", "FEATURES.md", "favicon.svg", "runtime.json", "wheels/manifest.json"} <= asked
    assert kept == {*pages, *(asked - opening)}
