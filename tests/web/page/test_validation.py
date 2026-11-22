"""Validation (``web/validation.html``): what the page says must stay true of the record it sums up.

The page is written by hand, for a reader who wants one screen and no table of thousands of checks. Its names and
counts come from two places a test can read: the reference catalogue's list of sources, and
``web/checked.json``, which ``scripts/design_crosscheck/tabulate.py`` writes from the comparison of each
design with the other packages. A source or a package that joins either one fails here until the page says so.
"""

from __future__ import annotations

import json
import re
from datetime import date

from process_improve.experiments.reference_catalogue import reference_sources
from tests.web._support import source

PAGE = " ".join(source("validation.html").split())
CHECKED = json.loads(source("checked.json"))
#: The grid's rows, each by the kind of design the record names it.
ROWS = {
    "Full factorial": "full_factorial",
    "Fractional factorial": "fractional_factorial",
    "Plackett-Burman": "plackett_burman",
    "Box-Behnken": "box_behnken",
    "Central composite": "ccd",
    "Definitive screening": "dsd",
    "Orthogonal arrays": "taguchi",
}
#: A package in a cell of the grid: its mark's state, its name, and the words said for a reader who hears the page.
MARK = r'data-state="(\w+)"></span>([\w.-]+)<span class="visually-hidden">: ([^<]+)</span>'
#: Those words for each state: the same as the legend over the grid.
WORDS = {"match": "gives the same design", "known": "differs in places, for a known reason"}


def test_the_page_counts_the_sources_of_the_reference_catalogue():
    sources = len(reference_sources())
    assert f"the designs that {sources} textbooks and papers print" in PAGE
    assert f"{sources} books, handbooks and papers" in PAGE
    named = len(re.findall(r"<li>[^<]*<cite>", PAGE))
    assert f"And {sources - named} more, each with its pages" in PAGE


def test_the_page_names_every_package_of_the_record_under_its_language():
    packages = CHECKED["packages"]
    assert f"that {len(packages)} open-source packages build" in PAGE
    assert f"{len(packages)} open packages, in R and Python" in PAGE
    for language in ("R", "Python"):
        listed = re.search(rf"<dt>{language}</dt><dd>([^<]+)</dd>", PAGE)[1].split(", ")
        assert sorted(listed) == sorted(name for name, its in packages.items() if its == language)


def test_each_row_of_the_grid_names_the_packages_and_counts_the_sources_of_the_record():
    rows = dict(re.findall(r'<tr> <th scope="row">([^<]+)</th>(.*?)</tr>', PAGE))
    assert list(rows) == [*ROWS, "OMARS"], "one row for each kind of design, and one for the designs no package builds"
    for name, kind in ROWS.items():
        record = CHECKED["designs"][kind]
        marks = re.findall(MARK, rows[name])
        assert sorted(package for _, package, _ in marks) == sorted(record["packages"]), name
        assert all(words == WORDS[state] for state, _, words in marks), f"{name}: a mark's words say what it shows"
        r_cell, python_cell = re.findall(r'<td data-head="(?:R|Python)">(.*?)</td>', rows[name])
        for cell, language in ((r_cell, "R"), (python_cell, "Python")):
            assert all(CHECKED["packages"][package] == language for _, package, _ in marks if f">{package}<" in cell)
        count = len(record["sources"])
        assert f"<td>{count} source{'s' if count > 1 else ''}" in rows[name], name
    for state, words in WORDS.items():
        assert f'data-state="{state}"></span>{words[0].upper()}{words[1:]}</li>' in PAGE, "the legend's own words"


def test_the_day_at_its_foot_is_the_day_of_the_record():
    day = date.fromisoformat(CHECKED["checked"])
    assert f"Last checked on {day.day} {day:%B %Y}." in PAGE


def test_the_page_leads_to_the_textbooks_answers_the_full_record_and_its_own_grid():
    assert PAGE.count('href="catalog/reproduced/"') == 2
    assert 'href="https://kgdunn.github.io/process-improve/validation.html" target="_blank" rel="noopener"' in PAGE
    assert '<a href="#grid">' in PAGE
    assert '<section class="validation__grid" id="grid"' in PAGE
