"""``web/FEATURES.md``, the document the app is described from, against the app itself.

Most of the document is prose, kept true by review: a change to the app a reader would notice changes it in the
same pull request (``CONTRIBUTING.md``, "The browser app is tested"). What can be counted is checked here: a
design, a chart or a process to practise on that the document leaves out, and a limit or a count it states that
is no longer the app's.
"""

from __future__ import annotations

import json
import re

from tests.web._support import javascript, source

FEATURES = source("FEATURES.md")

#: The document with each run of white space as one space, so a sentence is found whatever line it breaks on.
TEXT = re.sub(r"\s+", " ", FEATURES)

#: Each row of the document's table of limits, and the app's own number for it (``web/bootstrap.py``).
LIMITS = {
    "Factors in a study": lambda app: app.MAX_FACTORS,
    "Factors whose interactions are modelled": lambda app: app.MAX_INTERACTION_FACTORS,
    "Responses in a study": lambda app: app.MAX_RESPONSES,
    "Levels of a categorical factor": lambda app: app.MAX_LEVELS,
    "Limits on the settings": lambda app: app.MAX_CONSTRAINTS,
    "Blocks in a design the app makes": lambda app: app.MAX_BLOCKS,
    "Blocks in runs as given": lambda app: app.MAX_GIVEN_BLOCKS,
    "Replicates of each run": lambda app: app.MAX_REPLICATES,
    "Runs in a table of your own": lambda app: app.MAX_GIVEN_RUNS,
    "Columns in a table of your own": lambda app: app.MAX_TABLE_COLUMNS,
    "Runs of a space-filling design": lambda app: app.MAX_SPACE_FILLING_RUNS,
    "Runs of an OMARS design": lambda app: app.MAX_OMARS_RUNS,
    "Runs at each level of a one-factor study": lambda app: app.MAX_LEVEL_RUNS,
    "Points along a curve": lambda app: app.MAX_CURVE_POINTS,
    "Components of a mixture": lambda app: max(info["max_factors"] for info in app.MIXTURES.values()),
}


def table(heading: str) -> dict[str, str]:
    """Return the first table under a heading of the document: each row's first cell, and its second."""
    section = FEATURES.split(f"\n{heading}\n", 1)[1].split("\n## ", 1)[0]
    lines = section[section.index("\n|") :].lstrip("\n").split("\n\n", 1)[0].splitlines()
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    return {row[0]: row[1] for row in rows[2:]}  # after the head and the rule under it


def test_every_design_the_app_makes_is_listed_with_the_factors_it_takes(bootstrap):
    made = {**bootstrap.DESIGNS, **bootstrap.MIXTURES}

    def factors(info: dict) -> str:
        low, high = info["min_factors"], info["max_factors"]
        return f"{low} or {high}" if high == low + 1 else f"{low} to {high}"

    assert table("## 2. Designs") == {info["label"]: factors(info) for info in made.values()}
    assert f"**{len(made)} designs**" in TEXT, "the count in the summary"
    assert f"The design list holds {len(made)} designs" in TEXT


def test_every_chart_is_listed_by_its_title(bootstrap, node):
    (charts,) = javascript(node, source("logic.js"), [("CHARTS", [])])
    listed = table("## 6. Figures")
    assert list(listed) == [chart["title"] for chart in charts.values()]
    more = [title for title, shown in listed.items() if shown.startswith("More charts")]
    assert more == [charts[key]["title"] for key in bootstrap.MORE_CHARTS]
    assert f"**{len(charts)} charts.**" in TEXT, "the count in the summary"
    assert f"**The {len(charts)} charts**" in TEXT


def test_every_process_to_practise_on_and_every_reading_level_is_named(bootstrap, node):
    for process in bootstrap.SCENARIOS.values():
        assert process["label"] in TEXT
    (levels,) = javascript(node, source("levels.js"), [("LEVELS", [])])
    assert f"{levels[0]}, {levels[1]} or {levels[2]}" in TEXT


def test_the_limits_are_the_app_s(bootstrap):
    stated = table("## 12. Limits")
    assert list(stated) == list(LIMITS), "a row the document adds or drops is added to LIMITS, or dropped from it"
    for what, most in LIMITS.items():
        assert int(stated[what].replace(",", "")) == most(bootstrap), what


def test_the_numbers_the_document_repeats_in_its_sentences_are_the_app_s(bootstrap, node):
    """A number said twice is checked twice: each of these is in a sentence, apart from the table of limits."""
    app = bootstrap
    graph = json.loads(source("map/graph.json"))
    (glossary,) = javascript(node, source("glossary.js"), [("GLOSSARY", [])])
    book, published = (
        json.loads(source(name))["examples"] for name in ("book-examples.json", "published-examples.json")
    )
    megabytes = app.MAX_TABLE_BYTES // 1_000_000
    for sentence in (
        f"**Up to {app.MAX_RESPONSES} responses**",
        f"has 2 to {app.MAX_LEVELS} levels of any text",
        f"replicates (up to {app.MAX_REPLICATES}), blocks (up to {app.MAX_BLOCKS})",
        f"Up to {app.MAX_CONSTRAINTS} inequalities",
        f"**Up to {app.MAX_FACTORS} factors.** Interactions are modelled for up to {app.MAX_INTERACTION_FACTORS}.",
        f"interactions of up to {['', 'one', 'two', 'three'][app.MAX_ORDER]} factors",
        f"| Up to {app.MAX_OMARS_RUNS} runs |",
        f"up to {app.MAX_SPACE_FILLING_RUNS} runs |",
        f"held to {megabytes} MB, {app.MAX_TABLE_COLUMNS} columns and {app.MAX_GIVEN_RUNS:,} runs",
        f"one button adds {['', 'one', 'two', 'three'][app.CONFIRMATION_RUNS]} at the settings found",
        f"a note when one is above {app.VIF_HIGH:g}",
        f"{['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six'][len(app.SCENARIOS)]} processes with a known true model",
        f"The glossary holds {len(glossary)} terms",
        f"The book's {len(book)} examples open with `?book=<name>`, and {len(published)} published",
        f"the book's {len(book)} examples are held to",
        f"{len(graph['nodes'])} ideas of designed experiments and {len(graph['edges'])} links between them",
        f"links between them, in {len(graph['clusters'])} topics",
    ):
        assert sentence.lower() in TEXT.lower(), sentence
