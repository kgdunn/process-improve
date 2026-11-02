"""Tests for the docs' own extension that gives each page its link-preview description.

``docs/conf.py`` loads ``docs/_ext/page_description.py``; these tests load it from its path
and check each rule on a small document: what counts as the page's prose, how a long
paragraph is cut, and what describes an API or index page that has no prose.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sphinx", reason="run `uv sync --dev` to build and test the docs")

from docutils import nodes
from docutils.core import publish_doctree
from sphinx import addnodes

_PATH = Path(__file__).resolve().parent.parent / "docs" / "_ext" / "page_description.py"
_SPEC = importlib.util.spec_from_file_location("page_description", _PATH)
page_description = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(page_description)

OPENING = "This page fits a PCA model to process data and reads its scores and loadings."


def describe(rst: str) -> str:
    return page_description.description(publish_doctree(rst))


def set_description(pagename, doctree, metadata=None, title="API Reference"):
    """Run the extension's hook as Sphinx would, and return what it hands the template."""
    app = SimpleNamespace(
        config=SimpleNamespace(root_doc="index"),
        env=SimpleNamespace(metadata={pagename: metadata or {}}, titles={pagename: nodes.title("", title)}),
    )
    context = {}
    page_description._set_description(app, pagename, "page.html", context, doctree)
    return context.get("page_description")


def test_the_first_paragraph_of_prose_describes_the_page():
    rst = f"""
Title
=====

Too short to describe the page.

.. note::

   A note is an aside to the page, however long it runs on the page itself.

- A list item is a fragment of a sentence, however long it runs on the page.

**Source worksheet:** a labelled line is a note about the page, not its opening paragraph.

{OPENING}
"""
    assert describe(rst) == OPENING


def test_a_paragraph_that_introduces_a_list_says_that_it_goes_on():
    rst = "Cross-validation is used for two purposes in multivariate analysis:\n\n- one\n- two\n"
    assert describe(rst) == "Cross-validation is used for two purposes in multivariate analysis…"


def test_a_long_paragraph_is_cut_at_a_word_boundary():
    text = "The scores and loadings of a PCA model, read together, explain an unusual batch."
    assert page_description.shorten(text, limit=40) == "The scores and loadings of a PCA…"
    assert page_description.shorten(text) == text


def test_inline_maths_reads_as_plain_text():
    assert page_description.plain_math(r"\hat{\sigma}_x") == "\N{GREEK SMALL LETTER SIGMA}x"
    rst = "The limit for Hotelling's :math:`T^2` comes from the F-distribution, for any component."
    assert describe(rst) == "The limit for Hotelling's T² comes from the F-distribution, for any component."


def test_an_api_page_is_described_by_what_it_documents():
    docstring = nodes.paragraph("", "A module docstring is reference text, not the page's own opening prose.")
    docstring.source = "methods.py:docstring of process_improve.methods"
    entry = addnodes.desc("", addnodes.desc_signature("", "", fullname="PCA"), domain="py", objtype="class")
    below = nodes.paragraph("", OPENING)  # introduces the entry above it, not the page
    page = nodes.section("", nodes.title("", "Multivariate Analysis"), docstring, entry, below)
    assert page_description.description(page) == ""
    assert set_description("api/multivariate", page, title="Multivariate Analysis") == "Multivariate Analysis: PCA"


def test_an_index_page_is_described_by_the_pages_it_lists():
    entries = [addnodes.compact_paragraph("", name, classes=["toctree-l1"]) for name in ("Multivariate", "Monitoring")]
    toc = nodes.section("", addnodes.compact_paragraph("", "", *entries, toctree=True))
    assert page_description.description(toc) == ""
    assert set_description("api/index", toc) == "API Reference: Multivariate, Monitoring"


def test_a_written_description_comes_first_and_the_landing_page_keeps_the_summary():
    doctree = publish_doctree(OPENING)
    assert set_description("user_guide/pca", doctree) == OPENING
    assert set_description("user_guide/pca", doctree, {"description": "Written\n   for this page."}) == (
        "Written for this page."
    )
    assert set_description("index", doctree) is None
    assert set_description("genindex", None) is None
