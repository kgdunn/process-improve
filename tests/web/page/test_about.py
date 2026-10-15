"""How Factorial works (``web/about.html``): what it tells a reader must stay true of the code it describes.

The page says what reaches the server, where to read the code and how to build and serve the app.
Each of those is checked here against the file it describes, so a change to the server's
configuration, the build script or the repository's layout cannot leave the page describing an
app that no longer exists. The versions it lists are read at run time from the files the app
loads (``runtime.json`` and ``wheels/manifest.json``), so none is written into the page.
"""

from __future__ import annotations

import json
import re

from tests.web._support import ROOT, WEB, source

REPOSITORY = "https://github.com/kgdunn/process-improve/"


def about() -> str:
    return " ".join(source("about.html").split())


def source_of(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_the_app_s_footer_links_to_it():
    assert '<a href="about.html">Read more here</a>' in " ".join(source("index.html").split())


def test_every_link_into_the_repository_names_a_path_that_exists():
    paths = re.findall(rf'href="{re.escape(REPOSITORY)}(?:tree|blob)/main/([^"]+)"', source("about.html"))
    assert len(paths) >= 5, "the links to the code a reader would audit"
    for path in paths:
        assert (ROOT / path).exists(), path


def test_what_it_says_the_server_keeps_is_what_the_server_is_configured_to_keep():
    caddy = (ROOT / "deploy" / "factorial" / "factorial.caddy").read_text(encoding="utf-8")
    ipv4, ipv6 = re.search(r"ip_mask \{\s*ipv4 (\d+)\s*ipv6 (\d+)", caddy).groups()
    hours = int(re.search(r"roll_keep_for (\d+)h", caddy).group(1))
    text = about()
    assert f"the first {ipv4} bits of an IPv4 address, the first {ipv6} of an IPv6 one" in text
    assert f"kept for {hours // 24} days" in text


def test_the_build_it_prints_is_the_build_script_s():
    text = source("about.html")
    command = re.search(r"uv run python (scripts/build_web_app\.py) (.+)\n", text)
    assert command
    assert (ROOT / command.group(1)).is_file()
    script = source_of(command.group(1))
    for option in re.findall(r"(--[\w-]+)", command.group(2)):
        assert f'"{option}"' in script, option
    port = re.search(r'"--port", type=int, default=(\d+)', script).group(1)
    assert f"http://localhost:{port}" in text


def test_the_addresses_it_says_to_change_for_a_closed_network_are_in_the_files_it_names():
    text = about()
    assert "<code>web/worker.js</code> (Pyodide and the PyPI packages)" in text
    assert "<code>web/charts.js</code> (Plotly.js)" in text
    assert "cdn.jsdelivr.net/pyodide" in source("worker.js")
    assert "micropip.install" in source("worker.js")
    assert "cdn.jsdelivr.net/npm/plotly.js" in source("charts.js")


def test_it_writes_no_version_into_the_page():
    """The versions come from the files the app loads, so the page cannot fall behind them."""
    runtime = json.loads((WEB / "runtime.json").read_text(encoding="utf-8"))
    text = source("about.html")
    assert runtime["pyodide"] not in text
    for pinned in runtime["micropip"]:
        assert pinned.split("==")[1] not in text, pinned
