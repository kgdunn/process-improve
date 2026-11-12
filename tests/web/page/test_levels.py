"""The page's reading levels against the concept map's.

``web/levels.js`` gives each design and chart the level that first shows it, and names the term of the
concept map (``web/map/graph.json``, built from the book) whose level it takes. The two must agree, or
the page would show at Beginner what the map teaches at Adventurous.
"""

from __future__ import annotations

import json

from tests.web._support import WEB, javascript, page, source


def features(node: str) -> dict[str, dict]:
    """Return ``levels.FEATURES``: each part of the page a level brings."""
    return javascript(node, source("levels.js"), [("FEATURES", [])])[0]


def test_each_part_s_level_is_its_term_s_in_the_concept_map(node):
    levels = {
        n["id"]: n["level"] for n in json.loads((WEB / "map" / "graph.json").read_text(encoding="utf-8"))["nodes"]
    }
    for feature, info in features(node).items():
        if "term" in info:
            assert info["term"] in levels, f"{feature} names a term the map does not have"
            assert info["level"] == levels[info["term"]], feature


def test_every_design_on_the_page_has_a_level(node, bootstrap):
    listed = {key.removeprefix("design:") for key in features(node) if key.startswith("design:")}
    assert listed == {*bootstrap.DESIGNS, *bootstrap.MIXTURES}


def test_every_further_chart_has_a_level(node):
    more = javascript(node, source("logic.js"), [("MORE_CHARTS", [])])[0]
    assert {key.removeprefix("chart:") for key in features(node) if key.startswith("chart:")} == set(more)


def test_the_page_s_first_paint_reads_the_level_levels_js_reads(node):
    """The head of the page puts the level on it before app.js loads: the same key, and the same first visit's."""
    key, default = javascript(node, source("levels.js"), [("LEVEL_KEY", []), ("DEFAULT_LEVEL", [])])
    head = source("index.html").split("</head>")[0]
    assert f'localStorage.getItem("{key}")' in head
    assert f"level <= 3 ? level : {default};" in head
    assert [o["value"] for o in page().radios["level"]] == ["1", "2", "3"]
