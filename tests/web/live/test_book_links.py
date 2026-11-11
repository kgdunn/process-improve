"""The concept map's links into the book: each lands on the sentence its term quotes.

A term's link carries a text fragment (``#:~:text=``) that a browser scrolls to and highlights. The book is edited on
its own, so a sentence that moves or changes would silently leave its link at the top of a page: this opens every
page the map links to and finds each fragment's words on it, in order.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote


def test_every_link_into_the_book_finds_the_sentence_it_quotes(own_context: Any, app_url: str) -> None:  # noqa: ANN401
    page = own_context.new_page()
    page.goto(f"{app_url.rstrip('/')}/map/")
    page.wait_for_function("() => window.conceptMap")
    links = page.evaluate("() => window.conceptMap.data.nodes.map((n) => [n.id, window.conceptMap.bookUrl(n)])")
    texts: dict[str, str] = {}
    missing = []
    for term, url in links:
        address, _, fragment = url.partition("#")
        if ":~:text=" not in fragment:
            continue
        words = [unquote(part) for part in fragment.split(":~:text=")[1].split(",")]
        if address not in texts:
            page.goto(address, wait_until="domcontentloaded")
            texts[address] = re.sub(r"\s+", " ", page.evaluate("document.body.innerText"))
        text = texts[address]
        start = text.find(words[0])
        end = text.find(words[-1], start + len(words[0])) if len(words) == 2 else start  # the end comes after the start
        if start < 0 or end < 0:
            missing.append((term, words, address))
    assert missing == [], "each fragment's words are on the page its link opens"
    assert len(links) - sum(":~:text=" not in url for _, url in links) > 0.95 * len(links), "nearly every link has one"
