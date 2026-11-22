"""A page per book: how closely Factorial gives a source's answers, written by the build from the reproduction.

``write_books`` writes ``catalog/books/<source>/index.html`` for each source with a case, and the list of them. Each
page says in words how closely Factorial gives the source's answers, draws its printed numbers as a bar by state, lists
its examples in page order with the state of each, the numbers the source misprints with what it prints beside what its
own data give, and what Factorial cannot compute yet. The states, their words and their colours are those of the
reproduction page, which each example opens on.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

from tests.web._support import WEB, javascript, source


def case(name: str, tier: str | None, *, source: str = "montgomery-2013", differs=(), missing=(), **change) -> dict:
    """Describe a case as ``catalog/reproduction.json`` holds it, in the fields a book's page reads."""
    return {
        "id": name,
        "title": name.replace("-", " ").capitalize(),
        "source": source,
        "kind": "example",
        "checked": 0 if tier is None else 4,
        "tier": tier,
        "worst": 0,
        "differs": list(differs),
        "unchecked": {"library_gap": len(missing)} if missing else {},
        "missing": list(missing),
        **change,
    }


def differs(cause: str, *, printed: str = "8.13", found: str = "8.16946", where: str = "p. 91") -> dict:
    return {
        "what": "std_error",
        "term": "treatment mean",
        "printed": printed,
        "where": where,
        "found": found,
        "error": 0.0049,
        "reason": "a misprint: MSE is 333.70, and √(333.70/5) = 8.17",
        "cause": cause,
    }


def missing(reason: str, where: str) -> dict:
    return {"what": "q", "term": "Bartlett", "printed": "0.21", "where": where, "reason": reason}


BARTLETT = "a step of Bartlett's test, which the library does not have"
POWER = "a search the library does not make for the smallest sample size of a power; it reproduces at {}"

#: Montgomery: an etching example with a misprint, a close one, one with no known reason, and an exercise not checked.
MONTGOMERY = [
    case("etching", "reproduced", differs=[differs("source_error")], missing=[missing(BARTLETT, "p. 85")]),
    case("golf", "reproduced"),
    case("cement", "within_5", differs=[differs("method", where="p. 30")]),
    case("watershed", "within_1", differs=[differs("unexplained", where="p. 88")]),
    case("problem-2-8", None, kind="exercise"),
    case(
        "power", "reproduced", missing=[missing(POWER.format(0.95), "p. 47"), missing(POWER.format(0.9), "pp. 107-110")]
    ),
]

WHERE = {"etching": "pp. 66-129; Table 3.1", "golf": "pp. 4-6; Figure 1.4", "cement": "pp. 25-47", "power": "p. 47"}

SOURCES = {
    "montgomery-2013": {
        "short": "Montgomery",
        "title": "Design and Analysis of Experiments",
        "year": 2013,
        "form": "book",
        "verdict": "",
    },
    "lenth-1989": {"short": "Lenth", "title": "Quick and easy analysis", "year": 1989, "form": "paper", "verdict": ""},
    "no-cases-2020": {"short": "Nobody", "title": "No cases", "year": 2020, "form": "book", "verdict": ""},
}

FORMS = {"book": "Books", "paper": "Journal papers", "other": "Handbooks, encyclopedias and courses"}


@pytest.fixture(scope="module")
def written(build_script, tmp_path_factory) -> Path:
    """Write the books of the cases above as the build does, into a catalog folder of their own."""
    folder = tmp_path_factory.mktemp("catalog")
    (folder / "index.json").write_text(
        '{"designs": [' + ",".join(f'{{"id": "{k}", "where": "{v}"}}' for k, v in WHERE.items()) + "]}",
        encoding="utf-8",
    )
    cases = [*MONTGOMERY, case("lenth-pse", "reproduced", source="lenth-1989")]
    sources = {key: {**s, "verdict": f"{s['short']} verdict. More."} for key, s in SOURCES.items()}
    build_script.write_books(folder, {"cases": cases, "sources": sources, "forms": FORMS})
    return folder


@pytest.mark.parametrize(
    ("name", "state"),
    [("etching", "known"), ("golf", "match"), ("cement", "known"), ("watershed", "open"), ("power", "match")],
)
def test_an_example_is_in_the_state_of_its_numbers_as_on_the_reproduction_page(build_script, name, state):
    """A misprint differs, though the library is right about it; numbers not in Factorial yet leave the state be."""
    assert build_script.example_state(next(c for c in MONTGOMERY if c["id"] == name)) == state


@pytest.mark.parametrize(
    ("causes", "state"),
    [
        (["source_error", "method"], "known"),
        (["source_error", "incorrect_option"], "quiz"),
        (["incorrect_option", "unexplained", "source_rounding"], "open"),
        (["a cause the page does not know"], "open"),
    ],
)
def test_an_example_takes_the_state_of_its_number_last_in_the_bar_s_order(build_script, causes, state):
    """A reason not known yet is never hidden behind a known one."""
    held = case("x", "within_1", differs=[differs(cause) for cause in causes])
    assert build_script.example_state(held) == state


def test_a_source_s_numbers_are_counted_by_state_as_the_reproduction_page_counts_them(build_script):
    checked = sum(c["checked"] for c in MONTGOMERY)
    assert build_script.number_states(MONTGOMERY) == {"match": checked - 3, "known": 2, "open": 1, "gap": 3}


def test_the_states_are_the_reproduction_page_s_in_its_words_and_order(build_script, node):
    """A book's page names each state as the reproduction page does, and puts each cause under the same state."""
    states, reasons = javascript(node, source("reproduced-logic.js"), [("STATES", []), ("REASONS", [])])
    assert [(s["key"], s["label"]) for s in states] == list(build_script.STATES.items())
    assert {r["cause"]: r["state"] for r in reasons if r["cause"] != "library_gap"} == build_script.CAUSE_STATE


def test_a_book_s_page_and_the_reproduction_page_count_and_name_alike(build_script, node):
    """Both pages read a case alike, in web/reproduced-logic.js and in the build.

    Each example is in the state ``caseState`` gives it, a source's numbers are counted as ``byState`` counts them,
    and each quantity is named as ``quantity`` names it, from the same table of names.
    """
    mixed = case("mixed", "within_1", differs=[differs("source_error"), differs("incorrect_option")])
    cases = [*MONTGOMERY, mixed]
    module = source("reproduced-logic.js")
    *states, tallied = javascript(node, module, [*(("caseState", [c]) for c in cases), ("tally", [cases])])
    assert states == [build_script.example_state(c) for c in cases]
    (counted,) = javascript(node, module, [("byState", [tallied])])
    assert {k: n for k, n in counted.items() if n} == build_script.number_states(cases)

    names = build_script.quantity_names()
    numbers = [
        *(n for c in cases for n in [*c["differs"], *c["missing"]]),
        {"what": "anova_sum_sq", "term": "Error"},
        {"what": "f_regression", "term": None},
        {"what": "other", "term": "mode"},
        {"what": "a_word_with_no_name", "term": "A"},
    ]
    named = javascript(node, module, [("quantity", [n, names]) for n in numbers])
    assert [build_script._sentence_case(n) for n in named] == [build_script._quantity(n) for n in numbers]
    assert named[-4:] == ["a sum of squares: Error", "an F ratio", "mode", "a word with no name: A"]


@pytest.mark.parametrize(
    ("tiers", "says"),
    [
        (["reproduced"] * 17 + ["within_5"] * 3, "Factorial gives the book's answers."),
        (["reproduced"] * 18 + ["not_reproduced"] * 2, "Factorial gives the book's answers, with a few differences."),
        (["reproduced"] * 3 + ["within_5"] * 2, "Factorial gives most of the book's answers."),
        (["reproduced"] * 2 + ["not_reproduced"] * 3, "Factorial gives some of the book's answers."),
        ([None], "Factorial does not check the book's answers yet."),
    ],
)
def test_a_book_s_verdict_is_said_in_words(build_script, tiers, says):
    cases = [case(f"c{i}", tier) for i, tier in enumerate(tiers)]
    assert build_script.book_verdict(cases, "the book") == says


def test_a_verdict_says_why_numbers_differ_as_far_as_is_known(build_script):
    verdict = build_script.book_verdict(MONTGOMERY, "the book")
    assert verdict == (
        "Factorial gives the book's answers, with a few differences. Where they differ, this page says why: "
        "some are misprints in the book, and for some the reason is not known yet."
    )
    assert not re.search(r"\d", verdict), "the writing rule: the verdict is in words"
    only_method = [case("x", "within_1", differs=[differs("method")])]
    assert build_script.book_verdict(only_method, "the paper").endswith("Where they differ, this page says why.")


@pytest.mark.parametrize(
    ("where", "page"),
    [("pp. 25-47; Table 2.1", 25), ("p. 91", 91), ("p. 8-2", 8), ("Table 3.1", None), (None, None)],
)
def test_a_place_in_a_book_is_read_for_its_first_page(build_script, where, page):
    assert build_script.first_page(where) == page


@pytest.mark.parametrize(
    ("found", "printed", "shown"),
    [
        ("8.16946", "8.13", "8.17"),
        ("7.6", "8", "7.6"),
        ("5339.2", "5338.20", "5339.20"),
        ("19343", "13,075", "19343"),
        ("136, 256, 345", "124 + 345", "136, 256, 345"),
        ("a list", "8.13", "a list"),
    ],
)
def test_factorial_s_value_is_shown_to_the_printed_decimals_or_to_more_where_those_hide_the_difference(
    build_script, found, printed, shown
):
    assert build_script.as_printed(found, printed) == shown


def test_the_build_writes_a_page_for_each_book_with_a_case_and_the_list_of_them(written):
    assert sorted(p.relative_to(written).as_posix() for p in written.rglob("*.html")) == [
        "books/index.html",
        "books/lenth-1989/index.html",
        "books/montgomery-2013/index.html",
    ]
    listed = (written / "books" / "index.html").read_text(encoding="utf-8")
    assert '<a href="montgomery-2013/">' in listed
    assert "no-cases-2020" not in listed, "a source with no case has no page"
    assert listed.index("<h2>Books</h2>") < listed.index("<h2>Journal papers</h2>")
    assert "Montgomery verdict." in listed
    assert "More." not in listed, "the list gives the first sentence of each verdict"


def test_a_book_s_page_lists_its_examples_in_page_order_each_opening_its_numbers(written):
    page = (written / "books" / "montgomery-2013" / "index.html").read_text(encoding="utf-8")
    assert "<h1>Design and Analysis of Experiments</h1>" in page
    assert "Montgomery verdict. More." in page
    assert page.index(">Golf<") < page.index(">Power<") < page.index(">Etching<"), "pp. 4, 47, 66"
    assert page.index(">Etching<") < page.index(">Watershed<"), "an example with no page last"
    assert "Problem 2 8" not in page.split('id="not-yet"')[0], "an example with no number checked is not listed"
    assert 'href="../../../catalog/reproduced/#src=montgomery-2013&amp;case=etching"' in page
    assert "Example · pp. 66-129" in page
    key = '<span class="statekey" data-state="{}" aria-hidden="true"></span>'
    assert f'<span class="book-status">{key.format("known")}Differs, and we know why</span>' in page
    assert f'<span class="book-status">{key.format("open")}Differs, reason not known yet</span>' in page
    assert f'<span class="book-status">{key.format("match")}Matches the book</span>' in page
    assert """href="../../../catalog/reproduced/#src=montgomery-2013">Search the book's cases</a>""" in page


def test_a_book_s_bar_and_legend_are_its_numbers_in_the_reproduction_page_s_states(build_script, written):
    """The bar of a source's row there, in its words and colours: what the book prints, from a match to a gap."""
    page = (written / "books" / "montgomery-2013" / "index.html").read_text(encoding="utf-8")
    bar = page.split('<div class="statebar" aria-hidden="true">')[1].split("</div>")[0]
    parts = [("match", "17"), ("known", "2"), ("open", "1"), ("gap", "3")]
    assert re.findall(r'data-state="(\w+)" style="flex-grow: (\d+)"', bar) == parts
    legend = page.split('<ul class="states">')[1].split("</ul>")[0]
    assert re.findall(r"</span>([^<]+)</li>", legend) == [build_script.STATES[state] for state, _ in parts]
    for old in ("data-tier", "tierbar", "Very close", "Misprint found", "Not yet explained", "Show the numbers"):
        assert old not in page, f"a book's page speaks in the reproduction page's states, not in tiers: {old}"
    listed = (written / "books" / "index.html").read_text(encoding="utf-8")
    legend = listed.split('<ul class="states">')[1].split("</ul>")[0]
    assert re.findall(r"</span>([^<]+)</li>", legend) == list(build_script.STATES.values()), "every state"
    assert listed.count('<div class="statebar" aria-hidden="true">') == 2, "a bar for each source"


def test_a_book_s_misprints_set_what_it_prints_beside_what_its_data_give(written):
    page = (written / "books" / "montgomery-2013" / "index.html").read_text(encoding="utf-8")
    misprints = page.split('id="misprints"')[1].split("</section>")[0]
    assert misprints.count('class="misprint"') == 1, "only the misprints, not the other differences"
    assert "A standard error: treatment mean" in misprints, "the quantity by its plain name"
    assert "<span>Book <strong>8.13</strong></span><span>Data <strong>8.17</strong></span>" in misprints
    assert html.escape("A misprint: MSE is 333.70") in misprints


def test_what_factorial_lacks_is_grouped_by_what_it_lacks_the_most_numbers_first(written):
    page = (written / "books" / "montgomery-2013" / "index.html").read_text(encoding="utf-8")
    lacking = page.split('id="not-yet"')[1]
    lacks = "The library does not make for the smallest sample size of a power"
    assert lacking.count(lacks) == 1, "the numbers of one missing method fall together"
    assert lacking.index(lacks) < lacking.index("A step of Bartlett"), "two numbers before one"
    assert "pp. 47 and 107" in lacking
    assert "p. 85" in lacking


def test_a_book_without_misprints_or_gaps_has_no_such_sections(written):
    page = (written / "books" / "lenth-1989" / "index.html").read_text(encoding="utf-8")
    assert 'id="examples"' in page
    assert 'id="misprints"' not in page
    assert 'id="not-yet"' not in page
    assert page.count('<a href="#') == 1, "the page's menu names only the sections it has"


def test_a_book_s_page_loads_only_what_the_app_publishes(build_script, written):
    page = (written / "books" / "montgomery-2013" / "index.html").read_text(encoding="utf-8")
    loaded = set(re.findall(r'(?:href|src)="\.\./\.\./\.\./([^"#?]+)"', page)) - {"catalog/reproduced/"}
    assert loaded == {"style.css", "favicon.svg"}
    assert loaded <= set(build_script.APP_FILES)
    assert all((WEB / name).is_file() for name in loaded)


def test_a_title_is_written_as_text_not_markup(build_script, tmp_path):
    cases = [case("x", "reproduced", title="<b>bold</b> & co")]
    sources = {"montgomery-2013": {**SOURCES["montgomery-2013"], "verdict": "Fine."}}
    build_script.write_books(tmp_path, {"cases": cases, "sources": sources, "forms": FORMS})
    page = (tmp_path / "books" / "montgomery-2013" / "index.html").read_text(encoding="utf-8")
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; co" in page
    assert "<b>bold</b>" not in page


def test_many_examples_are_grouped_by_a_hundred_pages(build_script, tmp_path):
    many = [case(f"e{i}", "reproduced") for i in range(build_script.GROUPED + 1)]
    (tmp_path / "index.json").write_text(
        '{"designs": [' + ",".join(f'{{"id": "e{i}", "where": "p. {40 * i + 1}"}}' for i in range(len(many))) + "]}",
        encoding="utf-8",
    )
    sources = {"montgomery-2013": {**SOURCES["montgomery-2013"], "verdict": "Fine."}}
    build_script.write_books(tmp_path, {"cases": many, "sources": sources, "forms": FORMS})
    page = (tmp_path / "books" / "montgomery-2013" / "index.html").read_text(encoding="utf-8")
    assert re.findall(r"<summary>(Pages [^<]+)</summary>", page) == [
        "Pages 1 to 99",
        "Pages 100 to 199",
        "Pages 200 to 299",
        "Pages 300 to 399",
        "Pages 400 to 499",
    ]
