"""Every link out of the app opens in a tab of its own, so the page, and the study in it, stay where they are.

A link to the book opens in the book's own tab, the one the app's other links to the book reuse; any
other link out opens in a new tab. A browser finds a tab again by its name only from the tab that opened
it, so a link to the book keeps that tie (``rel="opener"``): with it cut, every link to the book opened a
tab of its own. The book is the app's own site. Every other link out carries ``rel="noopener"``, so the
page it opens cannot reach back into the app's window. ``tests/web/e2e/test_book_tab.py`` follows the
links in a browser.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

import pytest

from tests.web._support import source

PAGES = ("index.html", "about.html", "catalog/index.html", "catalog/reproduced/index.html")
LINK_OUT = re.compile(r'<a href="https?://[^"]*"[^>]*>')
OWN_TAB = re.compile(r'target="(?!_self"|_top"|_parent")[^"]+"')
BOOK_TAB = 'target="pid-book" rel="opener"'


def to_the_book(tag: str) -> bool:
    return urlsplit(re.search(r'href="([^"]+)"', tag)[1]).hostname == "learnche.org"


@pytest.mark.parametrize("page", PAGES)
def test_every_link_out_of_a_page_opens_in_a_tab_of_its_own(page):
    links = LINK_OUT.findall(source(page))
    assert links, "each page links out at least once"
    for tag in links:
        assert OWN_TAB.search(tag), tag
        if to_the_book(tag):
            assert BOOK_TAB in tag, f"{tag}: the book's tab, which a later link finds again"
        else:
            assert 'rel="noopener"' in tag, tag
            assert 'target="pid-book"' not in tag, f"{tag}: the book's tab is the book's alone"


def test_a_link_to_the_book_opens_in_the_book_s_tab_and_keeps_the_tie_that_finds_it_again():
    """The app's own links to the book, written by ``link`` in app.js, reuse one tab: the page's do too."""
    assert 'export const BOOK_TAB = "pid-book";' in source("campaign.js")
    assert "function link(href, text, target = BOOK_TAB)" in source("app.js")
    assert "a.target = target;" in source("app.js")
    assert 'a.rel = target === BOOK_TAB ? "opener" : "noopener";' in source("app.js")
    to_book = [tag for tag in LINK_OUT.findall(source("index.html")) if to_the_book(tag)]
    assert to_book, "the page links to the book"
    # The concept map writes its links to the book itself: each is in the book's tab too.
    written = re.findall(r'<a href="(?:\$\{bookUrl\(n\)\}|https://learnche\.org/pid)" ([^>]*)>', source("map.js"))
    assert written == ["${BOOK_TAB}", "${BOOK_TAB}"], "the term's line in the book, and the book itself"
    assert f"const BOOK_TAB = '{BOOK_TAB}';" in source("map.js")
    assert 'target="_blank" rel="noopener"' in source("map.js"), "its other links out open a new tab"


@pytest.mark.parametrize("script", ["catalog.js", "catalog-views.js"])
def test_every_link_out_the_catalog_writes_opens_in_a_new_tab(script):
    """A link the catalog writes to an address it is given (a source's, an erratum's, a place's) has a target too."""
    written = re.findall(r'element\("a", \{[^}]*\}', source(script))
    out = [a for a in written if re.search(r"href: (?:\w+\.)?\w*url\b", a)]
    assert out, "the catalog links to sources"
    if script == "catalog.js":
        assert any("href: url" in a for a in out), "the place a source prints a design is a link out"
    for a in out:  # campaign.tabOf: the book's tab for an address of the book's, else a new tab with no tie
        assert re.search(r"href: ((?:\w+\.)?\w*url), \.\.\.tabOf\(\1\)", a), a
    rule = source("campaign.js")
    assert '{ target: BOOK_TAB, rel: "opener" } : { target: "_blank", rel: "noopener" }' in rule


def test_the_feedback_dialog_s_link_to_an_issue_opens_in_a_new_tab():
    """The dialog writes two links: a new issue on GitHub, in a tab of its own, and an email, in the mail program."""
    tags = re.findall(r"<a [^>]*>", source("feedback.js"))
    assert sorted(re.search(r'id="([^"]+)"', tag)[1] for tag in tags) == ["feedback-issue", "feedback-mail"]
    for tag in tags:
        if 'id="feedback-issue"' in tag:
            assert OWN_TAB.search(tag), tag
            assert 'rel="noopener"' in tag, tag
        else:
            assert "target=" not in tag, "a mailto: link opens the mail program, not a tab"
    assert 'export const ISSUES = "https://github.com/kgdunn/process-improve/issues/new";' in source("feedback.js")
    assert 'querySelector("#feedback-issue").href = issueUrl(' in source("feedback.js")
