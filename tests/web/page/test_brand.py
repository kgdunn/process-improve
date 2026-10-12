"""The app's name and mark: Factorial, a 2^3 factorial and its centre run seen down the cube's diagonal.

``tools/branding/app_mark.py`` draws the mark from ``generate_design`` and writes the favicon, the
icons and the link card into ``web/``. The page's header and its status line draw the same mark
inline, in the page's own colours, so they follow the reader's theme; these tests keep every copy
in step with the generated one, and every file the page's head and manifest name in the build.
"""

from __future__ import annotations

import importlib.util
import json
import re
import struct
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import numpy as np
import pytest

from tests.web._support import ROOT, WEB, source

if TYPE_CHECKING:
    from types import ModuleType

#: The app's address: docs.yml deploys it there and checks it (deploy-factorial, check-factorial).
PUBLISHED = "https://factori.al/"

#: The size of each image the head and the manifest name, in pixels.
SIZES = {"apple-touch-icon.png": (180, 180), "icon-192.png": (192, 192), "icon-512.png": (512, 512),
         "og-card.png": (1200, 630)}  # fmt: skip


@pytest.fixture(scope="module")
def app_mark() -> ModuleType:
    """Load ``tools/branding/app_mark.py``, which is a script, not a module of the package."""
    pytest.importorskip("matplotlib", reason="the script draws the icons with matplotlib")
    spec = importlib.util.spec_from_file_location("app_mark", ROOT / "tools" / "branding" / "app_mark.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def png_size(name: str) -> tuple[int, int]:
    """Read a PNG's width and height from its header."""
    data = (WEB / name).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", name
    return struct.unpack(">II", data[16:24])


def geometry(svg: str) -> dict[str, set[tuple[float, ...]]]:
    """Return the mark's points: its runs' centres, the hexagon's corners, and the ends of its three edges."""
    number = r"[-\d.]+"
    circles = {tuple(map(float, m)) for m in re.findall(rf'<circle[^>]*? cx="({number})" cy="({number})"', svg)}
    hexagon = re.search(r'<polygon points="([^"]+)"', svg)
    corners = {tuple(map(float, p.split(","))) for p in hexagon.group(1).split()} if hexagon else set()
    line = rf'<line x1="{number}" y1="{number}" x2="({number})" y2="({number})"'
    ends = {tuple(map(float, m)) for m in re.findall(line, svg)}
    return {"runs": circles, "hexagon": corners, "edges": ends}


def meta(name: str) -> str:
    """Return the content of the page's meta tag with this property or name."""
    found = re.search(rf'<meta (?:property|name)="{re.escape(name)}" content="([^"]+)"', source("index.html"))
    assert found, name
    return found.group(1)


def test_the_mark_is_a_two_level_factorial_in_three_factors_with_its_centre_run(app_mark):
    runs = app_mark.runs()
    corners = runs[np.abs(runs).sum(axis=1) == 3]
    assert len(runs) == 9
    assert {tuple(c) for c in corners} == {(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)}
    assert (runs == 0).all(axis=1).sum() == 1


def test_seen_down_the_diagonal_two_corners_fall_behind_the_centre_run(app_mark):
    """The axes are 120 degrees apart, so (+,+,+) and (-,-,-) both land on the centre; the rest make a hexagon."""
    projected = app_mark.runs() @ app_mark.AXES
    radius = np.hypot(*projected.T)
    assert np.isclose(radius, 0).sum() == 3, "the centre run and two corners"
    assert np.allclose(radius[~np.isclose(radius, 0)], 2), "six corners on one circle"
    rim, centre, near = app_mark.mark()
    assert len(rim) == 6
    assert centre == (32, 32)
    assert len(near) == 3


def test_the_favicon_is_the_mark_the_script_draws(app_mark):
    assert (WEB / "favicon.svg").read_text(encoding="utf-8") == app_mark.svg(tile=True), (
        "run: python tools/branding/app_mark.py web"
    )


def test_the_header_and_the_status_line_draw_the_favicon_s_mark():
    favicon = geometry((WEB / "favicon.svg").read_text(encoding="utf-8"))
    header = re.search(r'<svg class="mark".*?</svg>', source("index.html"), re.DOTALL)
    assert header
    assert geometry(header.group(0)) == favicon
    status = re.search(r'<span class="status__icon".*?</span>', source("index.html"), re.DOTALL)
    assert status
    assert geometry(status.group(0))["runs"] == favicon["runs"], "the loader lights up the same runs"


def test_every_image_has_the_size_the_head_and_the_manifest_give(build_script):
    for name, size in SIZES.items():
        assert png_size(name) == size, name
        assert name in build_script.APP_FILES, name
    assert (int(meta("og:image:width")), int(meta("og:image:height"))) == SIZES["og-card.png"]


def test_the_manifest_names_the_app_and_icons_that_are_published(build_script):
    manifest = json.loads((WEB / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert manifest["short_name"] == "Factorial"
    assert manifest["start_url"] == manifest["scope"] == "./", "the app installs from wherever it is served"
    assert {"manifest.webmanifest", "favicon.svg"} <= set(build_script.APP_FILES)
    for item in manifest["icons"]:
        assert item["src"] in build_script.APP_FILES, item["src"]
        if item["type"] == "image/png":
            assert item["sizes"] == "x".join(map(str, png_size(item["src"]))), item["src"]
    assert any(item.get("purpose") == "maskable" for item in manifest["icons"])


def test_a_link_to_the_app_shows_its_card():
    """A chat, a blog or a social site shows og:image beside a link; it needs an absolute URL to fetch it."""
    canonical = re.search(r'<link rel="canonical" href="([^"]+)"', source("index.html"))
    assert canonical
    assert canonical.group(1) == meta("og:url") == PUBLISHED
    assert meta("og:image") == f"{PUBLISHED}og-card.png"
    assert meta("twitter:card") == "summary_large_image"
    assert meta("og:title").startswith("Factorial")
    assert re.search(r"<title>Factorial: [^<]+</title>", source("index.html"))


def test_the_published_address_is_the_one_the_deploy_is_checked_at():
    """The canonical address, the deploy's target and the address checked after it are one."""
    yaml = pytest.importorskip("yaml")
    jobs = yaml.safe_load((ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8"))["jobs"]
    assert jobs["deploy-factorial"]["environment"]["url"] == PUBLISHED
    assert jobs["check-factorial"]["env"]["APP"] == PUBLISHED
    ssh = (ROOT / ".github" / "actions" / "factorial-ssh" / "action.yml").read_text(encoding="utf-8")
    assert f"HostName {urlsplit(PUBLISHED).hostname}\n" in ssh, "the deploy logs in to the host it then checks"
