"""A page per field: what others in an application have run, written by the build from the catalog's index.

``write_fields`` writes ``catalog/fields/<field>/index.html`` for each application the catalogue names, the list of
fields, and ``catalog/fields.json`` for the planning panel. Each page says in a sentence how large the field's
experiments usually are and which kinds of design are the most common, with no more than two numbers, draws them
on a chart (``web/field-chart.js``), and lists them by kind, each opening in the catalog.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pytest

from tests.web._support import WEB

TOPICS = {
    "food-beverages": {"label": "Food and beverages", "description": ""},
    "general": {"label": "General (no stated application)", "description": ""},
    "paper": {"label": "Paper and pulp", "description": ""},
    "textiles": {"label": "Textiles", "description": ""},
}

SOURCES = {"anderson-whitcomb-2007": {"short": "Anderson and Whitcomb"}, "box-draper-2007": {"short": "Box and Draper"}}


def design(name: str, family: str, k: int, n: int, *, topics=("food-beverages",), **change) -> dict:
    """Describe a design as the catalog's index holds it, in the fields a field's page reads."""
    return {
        "id": name,
        "title": name.replace("-", " ").capitalize(),
        "collection": "published",
        "kind": "experiment",
        "family": family,
        "source": "anderson-whitcomb-2007",
        "year": 2007,
        "topics": list(topics),
        "n_factors": k,
        "n_runs": n,
        "has_responses": True,
        **change,
    }


#: Food: full factorials the most common, a dozen runs and three factors typical; every other kind of entry left out.
FOOD = [
    design("popcorn", "full_factorial", 3, 8),
    design("bottling", "full_factorial", 3, 12),
    design("brownies", "full_factorial", 3, 12),
    design("raisins", "fractional_factorial", 7, 16),
    design("baby-formula", "ccd", 3, 20, source="box-draper-2007"),
    design("two-samples", "comparison", 1, 20),
    design("a-design-made-here", "full_factorial", 3, 8, collection="generated"),
    design("no-size", "full_factorial", None, None),
    design("a-sample", "full_factorial", 2, 4, kind="sample"),
    design("textbook", "full_factorial", 3, 8, topics=("general",)),
]

PAPER = [
    design("bleaching", "fractional_factorial", 4, 8, topics=("paper",)),
    design("strength", "ccd", 3, 20, topics=("paper",)),
]


@pytest.fixture(scope="module")
def summaries(build_script) -> dict:
    return build_script.field_summaries([*FOOD, *PAPER], TOPICS)


@pytest.fixture(scope="module")
def written(build_script, tmp_path_factory) -> Path:
    """Write the fields of the designs above as the build does, into a catalog folder of their own."""
    folder = tmp_path_factory.mktemp("catalog")
    index = {"designs": [*FOOD, *PAPER], "topics": TOPICS, "sources": SOURCES}
    build_script.write_fields(folder, index)
    return folder


def test_a_field_counts_its_published_designed_experiments_of_a_known_size(summaries):
    """A comparison of samples, a design the library makes, one without a size and a sample are not experiments here."""
    assert sorted(summaries) == ["food-beverages", "paper"], "general is no field, and a field needs an experiment"
    kinds = summaries["food-beverages"]["kinds"]
    held = {d["id"] for group in kinds.values() for d in group}
    assert held == {"popcorn", "bottling", "brownies", "raisins", "baby-formula"}
    assert list(kinds) == ["full", "screening", "surface"], "in the order of FIELD_GROUPS, the empty kinds left out"


def test_experiments_of_one_size_and_kind_share_a_point(summaries):
    assert summaries["food-beverages"]["points"] == [
        [3, 8, "full", 1],
        [3, 12, "full", 2],
        [3, 20, "surface", 1],
        [7, 16, "screening", 1],
    ]


def test_a_field_s_sentence_says_the_typical_size_and_the_most_common_kind_in_words(summaries):
    assert summaries["food-beverages"]["sentence"] == (
        "Experiments here usually have about 3 factors and a dozen runs. Most are full factorials. "
        "The catalog holds only a few from this field so far."
    )


@pytest.mark.parametrize(
    ("families", "says"),
    [
        (
            ["full_factorial"] * 2 + ["ccd"] * 2 + ["mixture"],
            "Full factorials and response surface designs are the most common.",
        ),
        (["box_behnken"] * 3 + ["full_factorial"] * 2, "Most are response surface designs."),
        (["optimal"] * 4 + ["mixture"], "Mixture designs are the most common."),
    ],
)
def test_the_kinds_named_are_the_most_common_of_those_with_a_name(build_script, families, says):
    """A field where no kind holds half names the two most common; "other designs" is never named."""
    experiments = [design(f"d{i}", family, 3, 9) for i, family in enumerate(families)]
    assert says in build_script.field_sentence(experiments)


@pytest.mark.parametrize(
    ("sizes", "typical"),
    [
        ([(3, 16), (4, 16)], "about 4 factors and 16 runs"),
        ([(1, 20)], "about 1 factor and 20 runs"),
        ([(2, 10), (2, 14)], "about 2 factors and a dozen runs"),
    ],
)
def test_a_typical_size_is_the_median_in_words(build_script, sizes, typical):
    experiments = [design(f"d{i}", "full_factorial", k, n) for i, (k, n) in enumerate(sizes)]
    assert typical in build_script.field_sentence(experiments)


def test_a_field_s_sentence_holds_two_numbers_at_most(summaries):
    """The browser app's writing rule (CONTRIBUTING.md): one or two numbers a screen, the rest in words."""
    for summary in summaries.values():
        assert len(re.findall(r"\d+", summary["sentence"])) <= 2, summary["sentence"]


def test_the_build_writes_a_page_per_field_the_list_of_fields_and_the_panel_s_file(written, summaries):
    assert sorted(p.relative_to(written).as_posix() for p in written.rglob("*") if p.is_file()) == [
        "fields.json",
        "fields/food-beverages/index.html",
        "fields/index.html",
        "fields/paper/index.html",
    ]
    panel = json.loads((written / "fields.json").read_text())
    assert panel["fields"]["paper"] == {k: summaries["paper"][k] for k in ("label", "sentence", "points")}


def test_a_field_s_page_says_its_sentence_draws_its_points_and_opens_each_experiment(written, summaries):
    page = (written / "fields" / "food-beverages" / "index.html").read_text()
    assert "<h1>Food and beverages</h1>" in page
    assert html.escape(summaries["food-beverages"]["sentence"]) in page
    points = re.search(r'data-field-points="([^"]+)"', page).group(1)
    assert json.loads(html.unescape(points)) == summaries["food-beverages"]["points"]
    assert '<a href="../../../catalog/#d=popcorn">' in page
    assert "Anderson and Whitcomb (2007) · 3 factors, 8 runs" in page
    assert '<a href="../paper/">Paper and pulp</a>' in page, "it leads to the other fields"
    assert '<a class="button primary" href="../../../?field=food-beverages">' in page


def test_a_field_s_page_loads_only_what_the_app_publishes(build_script, written):
    """Its stylesheet, icon and chart are files of the app, three folders up from the page."""
    page = (written / "fields" / "food-beverages" / "index.html").read_text()
    loaded = set(re.findall(r'(?:href|src)="\.\./\.\./\.\./([^"#?]+)"', page)) - {"catalog/", ""}
    assert loaded == {"style.css", "favicon.svg", "field-chart.js"}
    assert loaded <= set(build_script.APP_FILES)
    assert all((WEB / name).is_file() for name in loaded)


def test_a_title_is_written_as_text_not_markup(build_script, tmp_path):
    index = {
        "designs": [design("x", "full_factorial", 3, 8, title="<b>bold</b> & co")],
        "topics": TOPICS,
        "sources": {},
    }
    build_script.write_fields(tmp_path, index)
    page = (tmp_path / "fields" / "food-beverages" / "index.html").read_text()
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; co" in page
    assert "<b>bold</b>" not in page


def test_a_long_kind_shows_the_first_few_and_the_rest_on_asking(build_script, tmp_path):
    many = [design(f"run-{i}", "full_factorial", 3, 8 + i) for i in range(build_script.LISTED + 2)]
    build_script.write_fields(tmp_path, {"designs": many, "topics": TOPICS, "sources": {}})
    page = (tmp_path / "fields" / "food-beverages" / "index.html").read_text()
    shown, rest = page.split("<summary>Show the rest</summary>")
    assert shown.count("catalog/#d=run-") == build_script.LISTED
    assert rest.count("catalog/#d=run-") == 2


def test_the_chart_draws_every_kind_the_build_groups_by(build_script):
    """``web/field-chart.js`` gives each kind of FIELD_GROUPS a shape; an unknown kind would be drawn as other."""
    kinds = re.search(r"export const KINDS = \[([^\]]+)\]", (WEB / "field-chart.js").read_text()).group(1)
    assert re.findall(r'"([a-z]+)"', kinds) == list(build_script.FIELD_GROUPS)
