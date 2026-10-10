"""Give each page of the docs its own description, for link previews and search results.

``_templates/layout.html`` writes the Open Graph tags and ``<meta name="description">``.
Their fallback is the package's one-line summary (``social_description`` in ``conf.py``), so
on its own a shared link to the PLS guide would read the same as one to the monitoring
reference. This extension hands the template ``page_description`` instead, the first of:

- a ``:description:`` field at the top of the page's source, for a page whose opening does
  not describe it (the quick start opens on code);
- the page's first paragraph of prose, cut at a word boundary (see ``is_prose``);
- on a page with no prose, its title and what it holds: the classes and functions an API
  page documents, or the pages an index lists.

The landing page keeps the summary, which is written for it, and so does a page with none
of these.

Ported from the textbook's extension, ``my-extensions/social_meta.py`` in kgdunn/pid-book,
with rules added for what an API reference has and a book does not: lists, tables of
contents and docstrings.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx

DESCRIPTION_LIMIT = 200  # characters; previews show about this much
MIN_PARAGRAPH = 60  # a shorter first paragraph ("See the figure below.") does not describe the page

# Paragraphs inside these are not the page's opening prose: asides and captions, list items
# (fragments of a sentence), and a table of contents.
ASIDES = (
    nodes.Admonition,
    nodes.sidebar,
    nodes.topic,
    nodes.table,
    nodes.figure,
    nodes.legend,
    nodes.footnote,
    nodes.citation,
    nodes.field_list,
    nodes.block_quote,
    nodes.bullet_list,
    nodes.enumerated_list,
    addnodes.compact_paragraph,
)

# LaTeX commands that read naturally as one character in a plain-text description. The
# Greek letters are the point here, so ruff's warning that they look like Latin ones is off.
MATH_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε",  # noqa: RUF001
    "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "Sigma": "Σ", "tau": "τ",  # noqa: RUF001
    "theta": "θ", "times": "×", "pm": "±", "leq": "≤", "geq": "≥", "approx": "≈",  # noqa: RUF001
}  # fmt: skip
SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def plain_math(latex: str) -> str:
    """Render inline LaTeX as plain text: T^2 becomes T², Greek letters their symbols, other markup goes."""
    text = re.sub(r"\^\{?(\d+)\}?", lambda m: m.group(1).translate(SUPERSCRIPTS), latex)
    text = re.sub(r"\\([A-Za-z]+)", lambda m: MATH_SYMBOLS.get(m.group(1), ""), text)
    return " ".join(re.sub(r"[{}_^$\\]", "", text).split())


def node_text(node: nodes.Node) -> str:
    """Return the text of ``node``, with inline maths rendered by ``plain_math``."""
    if isinstance(node, nodes.math):
        return plain_math(node.astext())
    if isinstance(node, nodes.Text):
        return str(node)
    return "".join(node_text(child) for child in node.children)


def shorten(text: str, limit: int = DESCRIPTION_LIMIT) -> str:
    """Cut ``text`` at the last word boundary within ``limit`` characters, marking the cut.

    A paragraph that ends in a colon introduces what follows it (a list, code, maths), so its
    end is marked the same way.
    """
    text = " ".join(text.split())
    if len(text) <= limit and not text.endswith(":"):
        return text
    if len(text) > limit:
        text = text[: limit - 1].rsplit(" ", 1)[0]
    return text.rstrip(",;:") + "…"


def is_prose(paragraph: nodes.paragraph) -> bool:
    """Tell whether ``paragraph`` is the page's own prose, rather than an aside or a docstring."""
    if any(isinstance(node, ASIDES) for node in (paragraph, *_ancestors(paragraph))):
        return False
    if ":docstring of " in (paragraph.source or ""):  # a module's docstring, set by autodoc
        return False
    label = paragraph.children[0] if paragraph.children else None
    return not (isinstance(label, nodes.strong) and label.astext().endswith(":"))  # "**Source:** ..."


def description(doctree: nodes.Node, limit: int = DESCRIPTION_LIMIT) -> str:
    """Return the page's first paragraph of prose, shortened, or an empty string if it has none.

    Only prose above the page's first API entry counts: below it, a paragraph introduces one
    entry, not the page.
    """
    for node in doctree.findall(lambda n: isinstance(n, (nodes.paragraph, addnodes.desc))):
        if isinstance(node, addnodes.desc):
            break
        text = " ".join(node_text(node).split())
        if len(text) >= MIN_PARAGRAPH and is_prose(node):
            return shorten(text, limit)
    return ""


def contents(doctree: nodes.Node) -> list[str]:
    """Name what a page holds: the classes and functions it documents, else the pages it lists."""
    documented = [
        desc[0]["fullname"]
        for desc in doctree.findall(addnodes.desc)
        if desc.get("domain") == "py" and desc.get("objtype") in {"class", "exception", "function"}
    ]
    listed = [
        entry.astext() for entry in doctree.findall(addnodes.compact_paragraph) if "toctree-l1" in entry["classes"]
    ]
    return documented or listed


def _ancestors(node: nodes.Node) -> Iterator[nodes.Node]:
    parent = node.parent
    while parent is not None:
        yield parent
        parent = parent.parent


def _set_description(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: nodes.document | None
) -> None:
    """Hand ``layout.html`` the page's description; the landing page keeps the summary."""
    if doctree is None or pagename == app.config.root_doc:
        return
    written = " ".join(app.env.metadata.get(pagename, {}).get("description", "").split())
    summary = written or description(doctree)
    if not summary and (names := contents(doctree)):
        summary = shorten(f"{app.env.titles[pagename].astext()}: {', '.join(names)}")
    if summary:
        context["page_description"] = summary


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the extension with Sphinx."""
    app.connect("html-page-context", _set_description)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
