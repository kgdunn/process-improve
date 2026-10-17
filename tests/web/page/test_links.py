"""Every link out of the app opens in a tab of its own, so the page, and the study in it, stay where they are.

A link to the book opens in the book's own tab, the one the app's other links to the book reuse; any
other link out opens in a new tab. Each carries ``rel="noopener"``, so the page it opens cannot reach
back into the app's window.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

import pytest

from tests.web._support import source

PAGES = ("index.html", "about.html", "catalog/index.html")
LINK_OUT = re.compile(r'<a href="https?://[^"]*"[^>]*>')
OWN_TAB = re.compile(r'target="(?!_self"|_top"|_parent")[^"]+"')


@pytest.mark.parametrize("page", PAGES)
def test_every_link_out_of_a_page_opens_in_a_tab_of_its_own(page):
    links = LINK_OUT.findall(source(page))
    assert links, "each page links out at least once"
    for tag in links:
        assert OWN_TAB.search(tag), tag
        assert 'rel="noopener"' in tag, tag


def test_a_link_to_the_book_opens_in_the_book_s_tab():
    """The app's own links to the book, written by ``link`` in app.js, reuse one tab: the page's do too."""
    assert "a.target = target;" in source("app.js")
    assert 'function link(href, text, target = "pid-book")' in source("app.js")
    to_book = [
        tag
        for tag in LINK_OUT.findall(source("index.html"))
        if urlsplit(re.search(r'href="([^"]+)"', tag)[1]).hostname == "learnche.org"
    ]
    assert to_book, "the page links to the book"
    for tag in to_book:
        assert 'target="pid-book"' in tag, tag


@pytest.mark.parametrize("script", ["catalog.js", "catalog-views.js"])
def test_every_link_out_the_catalog_writes_opens_in_a_new_tab(script):
    """A link the catalog writes to an address it is given (a source's, an erratum's) carries a target too."""
    written = re.findall(r'element\("a", \{[^}]*\}', source(script))
    out = [a for a in written if re.search(r"href: \w+\.url\b", a)]
    assert out, "the catalog links to sources"
    for a in out:
        assert 'target: "_blank"' in a, a
        assert 'rel: "noopener"' in a, a
