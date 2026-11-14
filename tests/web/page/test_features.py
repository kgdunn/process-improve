"""``web/FEATURES.md``, the document the app is described from, against the app itself.

Most of the document is prose, kept true by review: a change to the app a reader would notice changes it in the
same pull request (``CONTRIBUTING.md``, "The browser app is tested"). What can be counted is checked here: a
design, a chart or a process to practise on that the document leaves out, and a limit or a count it states that
is no longer the app's. One part sets the library beside the app, so that the app's limits are not read as
the library's: its table is checked against the tools the library registers.
"""

from __future__ import annotations

import json
import re

from process_improve import experiments
from process_improve.tool_spec import get_tool_specs
from tests.web._support import APP_SOURCES, javascript, page, source

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


#: The part of the document that sets the library beside the app.
LIBRARY = "### More in the Python library"

#: Each point of that part on what the library does and the app does not, and the names its details give.
#: A name is the library's while a registered tool or ``process_improve.experiments`` holds it, and the app is
#: without it while ``web/bootstrap.py`` never writes it. The app's refusal of a process factor beside a
#: mixture's components, which has no name of its own, is in ``tests/web/api/test_errors.py``.
ONLY_IN_THE_LIBRARY = {
    "Factors that are hard to change.": ("hard_to_change", "split_plot"),
    "Results that are counts, or pass and fail.": ("glm",),
    "Recipes with process settings.": (),
    "Sources of variation.": ("variance_components", "multistratum_anova"),
    "Experiments on a computer model.": ("fit_gaussian_process",),
}


def under(heading: str) -> str:
    """Return the document from under a heading to the next heading of its rank or above."""
    rank = heading.split(" ", 1)[0]
    rest = FEATURES.split(f"\n{heading}\n", 1)[1]
    return re.split(rf"\n#{{1,{len(rank)}}} ", rest, maxsplit=1)[0]


def table(heading: str) -> dict[str, str]:
    """Return the first table under a heading of the document: each row's first cell, and its second."""
    section = under(heading)
    lines = section[section.index("\n|") :].lstrip("\n").split("\n\n", 1)[0].splitlines()
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    return {row[0]: row[1] for row in rows[2:]}  # after the head and the rule under it


def points(heading: str) -> list[str]:
    """Return the points of the first list under a heading, each on one line and without its line of details."""
    section = under(heading)
    first = section[section.index("\n- ") :].lstrip("\n").split("\n\n", 1)[0]
    found: list[str] = []
    details = False  # a point's line of details, and the lines it goes on over, are no part of the point
    for line in first.splitlines():
        if line.startswith("- "):
            found.append(line[2:])
            details = False
        elif line.lstrip().startswith("- *Details:*"):
            details = True
        elif not details:
            found[-1] += f" {line.strip()}"
    return found


def test_every_design_the_app_makes_is_listed_with_the_factors_it_takes(bootstrap):
    made = {**bootstrap.DESIGNS, **bootstrap.MIXTURES}

    def factors(info: dict) -> str:
        low, high = info["min_factors"], info["max_factors"]
        return f"{low} or {high}" if high == low + 1 else f"{low} to {high}"

    listed = [re.match(r"\*\*(.+)\.\*\* For (.+) factors\. \S", point) for point in points("### Choose a design")]
    assert all(listed), "each design is named in bold, then the factors it takes, then what it is for"
    assert dict(found.groups() for found in listed) == {info["label"]: factors(info) for info in made.values()}
    assert f"**{len(made)} designs.**" in TEXT, "the count in the summary"
    assert f"*{len(made)} designs in one list" in TEXT, "and on the fold"
    assert f"The design list holds {len(made)} designs" in TEXT


def test_every_chart_is_listed_by_its_title(bootstrap, node):
    (charts,) = javascript(node, source("logic.js"), [("CHARTS", [])])
    listed = {re.match(r"\*\*(.+)\.\*\*", point).group(1): point for point in points(f"#### The {len(charts)} charts")}
    assert list(listed) == [chart["title"] for chart in charts.values()]
    more = [title for title, said in listed.items() if said.endswith("*More charts.*")]
    assert more == [charts[key]["title"] for key in bootstrap.MORE_CHARTS]
    words = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"]
    first = (
        f"{words[len(charts) - len(more)]} charts come with the analysis. {words[len(more)]} more are one press away"
    )
    assert first in TEXT
    assert f"**{len(charts)} charts.**" in TEXT, "the count in the summary"
    assert f"### See it: {len(charts)} charts" in TEXT, "and on the fold"


def test_every_process_to_practise_on_and_every_reading_level_is_named(bootstrap, node):
    for process in bootstrap.SCENARIOS.values():
        assert process["label"] in TEXT
    (levels,) = javascript(node, source("levels.js"), [("LEVELS", [])])
    assert f"{levels[0]}, {levels[1]} or {levels[2]}" in TEXT


def test_the_limits_are_the_app_s(bootstrap):
    stated = table("### Limits")
    assert list(stated) == list(LIMITS), "a row the document adds or drops is added to LIMITS, or dropped from it"
    for what, most in LIMITS.items():
        assert int(stated[what].replace(",", "")) == most(bootstrap), what


def test_each_design_the_library_makes_is_listed_beside_what_the_app_takes_for_it(bootstrap):
    """The library's designs are those its ``generate_design`` tool names: a new one fails here until it is listed."""
    (tool,) = (spec for spec in get_tool_specs() if spec["name"] == "generate_design")
    (named,) = (
        choice["enum"] for choice in tool["input_schema"]["properties"]["design_type"]["anyOf"] if "enum" in choice
    )
    assert set(bootstrap.DESIGNS) <= set(named), "the app makes each of its designs by the library's name for it"
    made = {design: [info] for design, info in bootstrap.DESIGNS.items()}
    made[bootstrap.MIXTURE] = list(
        bootstrap.MIXTURES.values()
    )  # the app's three mixture designs are one to the library

    def in_the_app(design: str) -> str:
        if design not in made:
            return "Not in the app"
        low = min(info["min_factors"] for info in made[design])
        high = max(info["max_factors"] for info in made[design])
        return f"{low} to {high} {'components' if design == bootstrap.MIXTURE else 'factors'}"

    assert table(LIBRARY) == {f"`{design}`": in_the_app(design) for design in named}


def test_what_the_library_alone_does_is_the_library_s_and_is_not_in_the_app(bootstrap):
    part, tools, app = under(LIBRARY), json.dumps(get_tool_specs()), source("bootstrap.py")
    said = [re.match(r"\*\*(.+?)\*\*", point).group(1) for point in points(LIBRARY)]
    assert said == list(ONLY_IN_THE_LIBRARY), "a point the document adds or drops is added here, or dropped"
    for name in (name for names in ONLY_IN_THE_LIBRARY.values() for name in names):
        assert re.search(rf"`[^`]*\b{name}\b[^`]*`", part), f"{name}: named in the document, in code"
        assert name in tools or hasattr(experiments, name), f"{name}: the library's"
        assert name not in app, f"{name}: the app writes it now, so it is no longer the library's alone"


def test_the_library_makes_an_optimal_design_in_more_factors_than_the_app_takes(bootstrap):
    """The document's own example of a limit that is the app's, set by the time a search takes in a browser."""
    most = bootstrap.DESIGNS["d_optimal"]["max_factors"]
    assert f"An optimal design takes up to {most} factors in the app." in TEXT
    factors = [experiments.Factor(name=f"x{i}", low=0, high=1) for i in range(most + 1)]
    design = experiments.generate_design(
        factors, design_type="d_optimal", model_type="main_effects", budget=most + 4, random_state=1
    )
    assert design.design[[f.name for f in factors]].shape == (most + 4, most + 1)


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
        f"**Up to {app.MAX_RESPONSES} responses.**",
        f"has 2 to {app.MAX_LEVELS} levels of any text",
        f"repeats of every run (up to {app.MAX_REPLICATES})",
        f"blocks such as days or batches (up to {app.MAX_BLOCKS})",
        f"Up to {app.MAX_CONSTRAINTS} inequalities",
        f"**Room for a large study.** Up to {app.MAX_FACTORS} factors.",
        f"Interactions are modelled for up to {app.MAX_INTERACTION_FACTORS}.",
        f"interactions of up to {['', 'one', 'two', 'three'][app.MAX_ORDER]} factors",
        f"*Details:* Up to {app.MAX_OMARS_RUNS} runs.",
        f"*Details:* Maximin; up to {app.MAX_SPACE_FILLING_RUNS} runs.",
        f"held to {megabytes} MB, {app.MAX_TABLE_COLUMNS} columns and {app.MAX_GIVEN_RUNS:,} runs",
        f"One button adds {['', 'one', 'two', 'three'][app.CONFIRMATION_RUNS]} confirmation runs at the settings found",
        f"a note when one is above {app.VIF_HIGH:g}",
        f"{['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six'][len(app.SCENARIOS)]} processes with a known true model",
        f"The glossary holds {len(glossary)} terms",
        f"The book's {len(book)} examples and {len(published)} published examples each open from a link",
        f"For the book's {len(book)} examples the app gives the numbers the book prints",
        f"{len(graph['nodes'])} ideas of designed experiments and {len(graph['edges'])} links between them",
        f"links between them, in {len(graph['clusters'])} topics",
    ):
        assert sentence.lower() in TEXT.lower(), sentence


#: Words for a statistician, or for the person who looks after the computers. Each is on a line of details.
FOR_THE_DETAILS = (
    "WebAssembly",
    "CPython",
    "HTTPS",
    "SHA-256",
    "CodeQL",
    "access log",
    "access-log",
    "cookies",
    "database",
    "JSON",
    "WCAG",
    "pull request",
    "smoke test",
    "p-value",
    "R²",
    "AICc",
    "Bonferroni",
    "Box-Cox",
    "Lenth",
    "Scheffé",
    "Tukey",
    "Welch",
    "Kruskal",
    "Cook's distance",
    "variance inflation",
    "defining relation",
    "aberration",
)


def test_what_is_in_bold_is_a_sentence_with_its_period():
    """A point opens with a sentence in bold, which a card sets on a line of its own: none trails a comma."""
    shown = " ".join(FEATURES.split("<!-- features.html shows what follows -->", 1)[1].split())
    bold = re.findall(r"\*\*(.+?)\*\*", shown)
    assert len(bold) > 120
    assert [words for words in bold if not words.endswith(".")] == []
    assert not re.search(r"\*\*[,:;]", shown), "and nothing but a new sentence follows it"


def test_a_point_says_in_plain_words_what_it_means_and_keeps_such_words_for_its_details():
    """What the page shows at first is for the person using the app: the words of statistics and servers come after."""
    shown = FEATURES.split("<!-- features.html shows what follows -->", 1)[1]
    plain, details, folded = [], [], False
    for line in shown.splitlines():
        if line.lstrip().startswith("- *Details:*"):
            folded = True
        elif not line.startswith("    "):  # a line of details goes on over lines indented under it
            folded = False
        (details if folded else plain).append(line)
    assert len(details) > 80
    first, rest = " ".join(plain), " ".join(details)
    for word in FOR_THE_DETAILS:  # as each is written: a link's address holds "https" too
        assert word not in first, word
        assert word in rest, f"{word}: said in a line of details, for a reader who asks"


# --------------------------------------------------------------------------- the page that shows the document
def test_the_document_is_published_with_the_page_that_shows_it(build_script, node):
    """features.html reads FEATURES.md with features.js: a build without one of the three shows no list."""
    shows = {"features.html", "features.js", "FEATURES.md"}
    assert shows <= set(build_script.APP_FILES)
    assert shows <= set(APP_SOURCES)
    html = source("features.html")
    assert 'import { addressOf, page, shownPart } from "./features.js";' in html
    assert 'import { LEVEL_KEY, readLevel } from "./levels.js";' in html, "the app's own reading level"
    assert 'fetch("FEATURES.md"' in html
    (mark,) = javascript(node, source("features.js"), [("SHOWN_FROM", [])])
    assert FEATURES.count(mark) == 1, "the line after which the page shows the document"


def test_the_app_s_footer_leads_to_the_page_with_a_button_and_names_each_group_of_its_links():
    html = " ".join(source("index.html").split())
    footer = html[html.index('<footer class="foot">') : html.index("</footer>")]
    assert '<a class="button primary" href="features.html">See what Factorial does</a>' in footer
    lists = re.findall(r'<ul aria-labelledby="([\w-]+)">', footer)
    assert lists == ["foot-app", "foot-behind", "foot-help"]
    assert set(lists) <= set(page().ids), "each list is named by the line above it"
    assert footer.count("<p>") == 2, "one sentence on what Factorial is, and the words on the copy kept here"
