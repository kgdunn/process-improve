"""The page against the Python it mirrors.

``web/index.html`` and ``web/app.js`` repeat some of ``web/bootstrap.py``'s choices (the design
list and what each design takes, the processes, the worked example's inputs) so the form is
right from the first paint, before Python has loaded. Each copy must say what the Python says,
or a reader would see one design's settings and generate another's.
"""

from __future__ import annotations

import re

from tests.web._support import page, source


# --------------------------------------------------------------------------- the form's choices
def test_page_design_list_matches_bootstrap(bootstrap):
    """The page lists the designs in HTML so the form is right before Python loads."""
    in_page = {o["value"]: o for o in page().options["design-type"]}
    assert list(in_page) == list(bootstrap.DESIGNS)
    for key, info in bootstrap.DESIGNS.items():
        option = in_page[key]
        assert option["label"] == info["label"], key
        assert option["data-hint"] == info["hint"], key
        assert option["data-options"].split() == info["options"], key
        assert int(option["data-min-factors"]) == info["min_factors"], key
        assert int(option["data-max-factors"]) == info["max_factors"], key
    assert page().selected("design-type") == bootstrap.DEMO["design"]["design_type"], "it opens on the worked example"


def test_the_page_lists_the_scenarios_bootstrap_simulates(bootstrap):
    """The process list is in the HTML, so it reads right before Python loads; it says what SCENARIOS says."""
    in_page = {o["value"]: o for o in page().options["scenario"]}
    assert list(in_page) == list(bootstrap.SCENARIOS)
    for key, info in bootstrap.SCENARIOS.items():
        assert in_page[key]["label"] == info["label"], key
        assert in_page[key]["data-hint"] == info["hint"], key
    assert page().selected("scenario") == bootstrap.DEMO["scenario"]


def test_the_page_s_default_factors_are_the_worked_example_s(bootstrap):
    block = re.search(r"const DEFAULT_FACTORS = \[(.*?)\];", source("app.js"), re.DOTALL).group(1)
    in_page = [
        {"name": n, "low": float(lo), "high": float(hi), "units": u}
        for n, lo, hi, u in re.findall(r'name: "(\w+)", low: ([\d.]+), high: ([\d.]+), units: "([^"]*)"', block)
    ]
    demo = [{**f, "low": float(f["low"]), "high": float(f["high"])} for f in bootstrap.DEMO["design"]["factors"]]
    assert in_page == demo


# --------------------------------------------------------------------------- links to the book
def test_every_link_to_the_book_is_extensionless_and_one_the_page_checked():
    """The book is served without ``.html``; every link the page makes names a page and anchor that exist."""
    checked = {
        "omars-designs#doe-omars-metric-choice",
        "comparing-design-families#doe-omnibus-comparison",
        "full-factorial-designs/using-two-levels-for-two-or-more-factors#doe-two-level-factorials",
        "fractional-factorial-designs/design-resolution#doe-design-resolution",
        "fractional-factorial-designs/saturated-designs-for-screening#doe-saturated-screening-designs",
        "definitive-screening-designs#doe-definitive-screening-designs",
        "response-surface-methods#doe-box-behnken-designs",
        "response-surface-methods#doe-central-composite-designs",
        "optimal-designs#doe-optimal-designs",
        "full-factorial-designs/assessing-significance-of-main-effects-and-interactions#doe-lenth-method",
        "judging-and-comparing-designs#doe-fds-plot",
    }
    text = source("index.html") + source("app.js")
    links = set(re.findall(r"learnche\.org/pid/design-analysis-experiments/([\w/#.-]+)", text))
    links |= set(re.findall(r"\$\{BOOK\}/([\w/#.-]+)", text))
    assert links, "the page links to the book"
    assert not any(".html" in link for link in links)
    assert links <= checked


def test_every_design_links_to_the_book(bootstrap):
    block = re.search(r"const BOOK_PAGES = \{(.*?)\};", source("app.js"), re.DOTALL).group(1)
    assert set(bootstrap.DESIGNS) <= set(re.findall(r"^\s*(\w+):", block, re.MULTILINE))
