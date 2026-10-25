"""The design catalog: published designs and designs this library generates, described for searching.

Each record describes one design by the facets a reader searches and filters on: its family, its
size, the kinds of factors it has, which models it can estimate, whether it is blocked or run as a
split plot, where and when it was published, what it is applied to (its ``topics``), and a compact
picture of its coded runs (the ``glyph``). The published records are the experiments and designs of
the reference catalogue (:mod:`process_improve.experiments.reference_catalogue`); the generated
records are designs that :func:`~process_improve.experiments.generate_design` makes, one for each
family and size listed in :data:`GENERATED`. The browser app's catalog page reads both, written once
when the app is built.

Examples
--------
>>> from process_improve.experiments.design_catalog import catalog_records, normalise_family
>>> normalise_family("2^(5-2) fractional factorial")
'fractional_factorial'
>>> record = next(r for r in catalog_records("generated") if r["id"] == "lib-ccd-k3-face-centered")
>>> record["n_runs"], record["models"]["quadratic"]
(17, True)
"""

from __future__ import annotations

import dataclasses
import functools
import re
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from process_improve.experiments._orthogonal_arrays import ORTHOGONAL_ARRAY_NAMES, orthogonal_array
from process_improve.experiments.compare import MEASURES, compare_designs
from process_improve.experiments.design_measures import (
    MODELS,
    design_classes,
    estimable_models,
    fds_curve,
    projectivity,
    quadratic_projection,
    run_counts,
    run_loss,
)
from process_improve.experiments.designs import generate_design
from process_improve.experiments.factor import Factor, FactorType
from process_improve.experiments.knowledge.engine import load_knowledge_graph
from process_improve.experiments.reference_catalogue import (
    _fraction,
    reference_designs,
    reference_experiments,
    reference_sources,
    reference_topics,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

Collection = Literal["published", "generated", "both"]


# --------------------------------------------------------------------------- families
@dataclass(frozen=True)
class Family:
    """A family of designs: its name on the page, and the year and work that first published it."""

    label: str
    year: int | None = None
    reference: str = ""


#: Every family a record can belong to. ``year`` and ``reference`` name the work that first published
#: the family, where one work is generally credited with it.
FAMILIES: dict[str, Family] = {
    "full_factorial": Family(
        "Full factorial", 1926, "Fisher, R. A. (1926), The arrangement of field experiments, J. Ministry of Agriculture"
    ),
    "fractional_factorial": Family(
        "Fractional factorial",
        1945,
        "Finney, D. J. (1945), The fractional replication of factorial arrangements, Annals of Eugenics",
    ),
    "plackett_burman": Family(
        "Plackett-Burman",
        1946,
        "Plackett, R. L. and Burman, J. P. (1946), The design of optimum multifactorial experiments, Biometrika",
    ),
    "orthogonal_array": Family(
        "Orthogonal array (Taguchi)",
        1947,
        "Rao, C. R. (1947), Factorial experiments derivable from combinatorial arrangements of arrays, "
        "Supplement to the J. Royal Statistical Society",
    ),
    "ccd": Family(
        "Central composite",
        1951,
        "Box, G. E. P. and Wilson, K. B. (1951), On the experimental attainment of optimum conditions, "
        "J. Royal Statistical Society B",
    ),
    "mixture": Family("Mixture", 1958, "Scheffé, H. (1958), Experiments with mixtures, J. Royal Statistical Society B"),
    "optimal": Family(
        "Optimal (D, I, G)", 1959, "Kiefer, J. (1959), Optimum experimental designs, J. Royal Statistical Society B"
    ),
    "box_behnken": Family(
        "Box-Behnken",
        1960,
        "Box, G. E. P. and Behnken, D. W. (1960), Some new three level designs for the study of quantitative "
        "variables, Technometrics",
    ),
    "supersaturated": Family(
        "Supersaturated",
        1962,
        "Booth, K. H. V. and Cox, D. R. (1962), Some systematic supersaturated designs, Technometrics",
    ),
    "dsd": Family(
        "Definitive screening",
        2011,
        "Jones, B. and Nachtsheim, C. J. (2011), A class of three-level designs for definitive screening in the "
        "presence of second-order effects, J. Quality Technology",
    ),
    "omars": Family(
        "OMARS",
        2020,
        "Núñez Ares, J. and Goos, P. (2020), Enumeration and multicriteria selection of orthogonal minimally "
        "aliased response surface designs, Technometrics",
    ),
    "latin_square": Family("Latin square"),
    "block_design": Family("Randomised blocks"),
    "split_plot": Family("Split plot"),
    "comparison": Family("Comparison or regression"),
    "other": Family("Other"),
}

#: The free-text design types of the reference catalogue, read into a family: the first rule that matches wins.
_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    (r"mixture|simplex|extreme vertices|component-amount", "mixture"),
    (r"omars", "omars"),
    (r"\bdsd\b|definitive screening|conference matrix", "dsd"),
    (r"split|strip|whole.plot|restricted randomi", "split_plot"),
    (r"box-behnken|\bbbd\b", "box_behnken"),
    (r"composite|\bccd\b|\bcc[cif]\b|axial|star points", "ccd"),
    (r"supersaturated", "supersaturated"),
    (r"taguchi|orthogonal arra|inner array|outer array|crossed", "orthogonal_array"),
    (r"plackett", "plackett_burman"),
    (r"latin|youden", "latin_square"),
    (r"block design|complete block|incomplete block|randomi[sz]ed blocks", "block_design"),
    (r"\b[adgi]-optimal|\boptimal designs?\b|candidate set|minimum-run resolution|measure design", "optimal"),
    (
        r"\d\^\(\d+-\d+\)|\^\(k-p\)|fraction|two-level design|mirror image|fold-?over|resolution (iii|iv|v) design",
        "fractional_factorial",
    ),
    (r"\d\^\d|\^k\b|factorial|\d+ x \d+|\d-by-\d|evop", "full_factorial"),
    (
        (
            r"regression|comparison|single sample|single.factor|one categorical|one factor at|process record|"
            r"illustration|confirmation|search along|path of steepest|selection|second-degree|nested|hierarchical|"
            r"two-way|completely randomi|sampling plan|design types|straight line"
        ),
        "comparison",
    ),
)

#: The library's design type for each family, where one ``generate_design`` type makes the family's designs.
_DESIGN_TYPES = {
    "full_factorial": "full_factorial",
    "fractional_factorial": "fractional_factorial",
    "plackett_burman": "plackett_burman",
    "ccd": "ccd",
    "box_behnken": "box_behnken",
    "dsd": "dsd",
    "omars": "omars",
    "supersaturated": "supersaturated",
    "mixture": "mixture",
    "orthogonal_array": "taguchi",
}


def normalise_family(text: str) -> str:
    """Return the family a free-text design type names: a key of :data:`FAMILIES`.

    Parameters
    ----------
    text : str
        A design type as a source describes it, such as ``"2^(5-2) fractional factorial"`` or ``"CCD"``.

    Returns
    -------
    str
        The family's key; ``"other"`` when no rule recognises the text.

    Examples
    --------
    >>> normalise_family("rotatable central composite design")
    'ccd'
    >>> normalise_family("Taguchi L18")
    'orthogonal_array'
    """
    lowered = text.lower()
    return next((family for pattern, family in _FAMILY_RULES if re.search(pattern, lowered)), "other")


# --------------------------------------------------------------------------- what a design can estimate
#: The most runs a glyph draws; a larger design is drawn from evenly spaced runs.
GLYPH_ROWS = 96

#: How many shades a glyph gives a coded setting from -2 to +2: one letter each, from "a" to "u".
_GLYPH_SHADES = 20


def glyph(coded: np.ndarray) -> str:
    """Return the picture of a design's coded runs that the catalog draws: one letter per setting, row by row.

    A setting from -2 to +2 is one of 21 letters, ``"a"`` at -2 to ``"u"`` at +2, so -1, 0 and +1
    are ``"f"``, ``"k"`` and ``"p"``; a missing setting is ``"."``. A design of more than
    :data:`GLYPH_ROWS` runs is drawn from that many evenly spaced runs.

    Parameters
    ----------
    coded : np.ndarray
        The runs in coded units, one column per factor.

    Returns
    -------
    str
        The letters, as many per row as the design has factors.

    Examples
    --------
    >>> import numpy as np
    >>> glyph(np.array([[-1.0, 0.0], [1.0, np.nan]]))
    'fkp.'
    """
    rows = coded if len(coded) <= GLYPH_ROWS else coded[np.linspace(0, len(coded) - 1, GLYPH_ROWS).round().astype(int)]
    shades = np.round((np.clip(rows, -2, 2) + 2) * _GLYPH_SHADES / 4)
    return "".join("." if np.isnan(s) else chr(ord("a") + int(s)) for s in shades.ravel())


def design_facets(coded: np.ndarray, kinds: Sequence[str]) -> dict[str, Any]:
    """Return the facets read off a design's coded runs: its size, its factors, its center runs and its models.

    Parameters
    ----------
    coded : np.ndarray
        The runs in coded units, one column per factor.
    kinds : Sequence[str]
        Each factor's type: ``"numeric"``, ``"categorical"`` or ``"mixture"``.

    Returns
    -------
    dict[str, Any]
        ``n_runs``, ``n_factors``, ``n_numeric``, ``n_categorical``, ``n_mixture``, ``levels`` (the
        number of distinct settings of each factor), ``n_centre`` (runs with every numeric factor at
        0), ``runs_per_parameter`` (runs per coefficient of the largest estimable model),
        ``models`` (see :func:`~process_improve.experiments.design_measures.estimable_models`),
        ``glyph`` (see :func:`glyph`); and, for the largest model the design estimates,
        ``run_loss_model``, ``survives_run_loss``, ``critical_runs``, ``worst_d_ratio`` and
        ``leverage`` (see :func:`~process_improve.experiments.design_measures.run_loss`); and
        ``quadratic_projection`` and ``projectivity`` (see
        :mod:`~process_improve.experiments.design_measures`; ``None`` for a design with a categorical
        factor of three or more labels, whose coding makes it arbitrary).
    """
    n_runs, n_factors = coded.shape
    numeric = [j for j, kind in enumerate(kinds) if kind == "numeric"]
    models = estimable_models(coded, kinds)
    largest = next((model for model in reversed(MODELS) if models.get(model)), None)
    loss = run_loss(coded, kinds, largest) if largest else None
    centre = None
    if numeric and "mixture" not in kinds:
        centre = int(np.all(np.abs(np.nan_to_num(coded[:, numeric], nan=9.0)) < 1e-9, axis=1).sum())
    return {
        "n_runs": n_runs,
        "n_factors": n_factors,
        "n_numeric": len(numeric),
        "n_categorical": sum(kind == "categorical" for kind in kinds),
        "n_mixture": sum(kind == "mixture" for kind in kinds),
        "levels": [len(np.unique(column[~np.isnan(column)].round(6))) for column in coded.T],
        "n_centre": centre,
        "runs_per_parameter": _runs_per_parameter(n_runs, n_factors, models),
        "models": models,
        "glyph": glyph(coded),
        "run_loss_model": largest,
        "survives_run_loss": None if loss is None else loss["survives"],
        "critical_runs": None if loss is None else loss["critical_runs"],
        "worst_d_ratio": None if loss is None else loss["worst_d_ratio"],
        "leverage": None if loss is None else loss["leverage"],
        "quadratic_projection": quadratic_projection(coded, kinds),
        "projectivity": projectivity(coded) if _two_level_labels(coded, kinds) else None,
    }


def _two_level_labels(coded: np.ndarray, kinds: Sequence[str]) -> bool:
    """Say whether projectivity means something for these runs: no mixture, and no categorical factor of 3+ labels.

    Projectivity counts the combinations of the runs at -1 and +1. A categorical factor of more labels
    is coded across that range in the order its labels are listed, so which two of its labels sit at
    -1 and +1 is arbitrary, and so is any count over them.
    """
    if "mixture" in kinds:
        return False
    labelled = [j for j, kind in enumerate(kinds) if kind == "categorical"]
    return all(len(np.unique(coded[~np.isnan(coded[:, j]), j].round(6))) <= 2 for j in labelled)


def _runs_per_parameter(n_runs: int, k: int, models: dict[str, bool | None]) -> float | None:
    """Return the runs per coefficient of the largest model the design estimates (two-level counts)."""
    terms = {"quadratic": (k + 1) * (k + 2) // 2, "interactions": 1 + k + k * (k - 1) // 2, "main_effects": 1 + k}
    largest = next((model for model in reversed(MODELS) if models.get(model)), None)
    return None if largest is None else round(n_runs / terms[largest], 2)


# --------------------------------------------------------------------------- where a design is printed
#: A figure's file, as a note of the catalogue names it: a path such as ``../figures/doe/x.png``, or a file name.
_FIGURE_FILE = re.compile(r"(?:\.\./)*(?:[\w.-]+/)*[\w.-]+\.(?:png|svg|jpe?g|gif|pdf)\b", flags=re.IGNORECASE)
#: A note of the lines of the source file that print a figure: ", lines 244-248", " (line 85", " at lines 135-146".
_LINES_NOTE = re.compile(r"[,;]?\s*(?:\bat\s+)?\blines?\s+\d+(?:\s*-\s*\d+)?", flags=re.IGNORECASE)
#: A Sphinx label of a figure at the start of a note, such as ``DOE-fig-Cube-plot``, and what follows it.
_FIGURE_LABEL = re.compile(r"^([A-Za-z]+-fig-[\w-]+)(?::\s*|\s+|$)")
#: The alt text of a figure, as a note quotes it: ``alt 'Contour plot ...'`` or ``alt: '...'``.
_ALT_TEXT = re.compile(r"\balt:?\s*'([^']+)'")
#: Words a note gives for what a figure is, not what it shows.
_KIND_ONLY = frozenset({"", "figure", "image", "no caption", "no static image"})


def _closing_group(text: str) -> int | None:
    """Return where the bracketed group that ends ``text`` opens; ``None`` when the text ends in no bracket."""
    if not text.endswith(")"):
        return None
    depth = 0
    for at in range(len(text) - 1, -1, -1):
        depth += {")": 1, "(": -1}.get(text[at], 0)
        if depth == 0:
            return at
    return None


def _opening_group(text: str) -> int | None:
    """Return where the bracketed group that starts ``text`` closes; ``None`` when the text starts with none."""
    if not text.startswith("("):
        return None
    depth = 0
    for at, char in enumerate(text):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if depth == 0:
            return at
    return None


def _readable(text: str) -> str:
    """Tidy a figure's name: no line notes, quotes, trailing ellipsis or full stop; a capital first letter."""
    text = _LINES_NOTE.sub("", text).strip(" ,;:")
    if len(text) > 1 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    text = re.sub(r"\s*(?:\.\.\.|…)$", "", text.strip()).strip()
    text = text.removesuffix(".").strip(" ,;:")
    return text[:1].upper() + text[1:]


def _file_title(path: str) -> str:
    """Name a figure by its file: ``../figures/doe/DOE-trade-off-table.png`` is "Trade off table"."""
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return _readable(re.sub(r"[-_]+", " ", re.sub(r"^DOE[-_]", "", stem)))


def figure_name(note: str) -> str:
    """Return the name a reader can use for a figure, from the catalogue's note of where it is printed.

    The book's entries record each figure as a note for the catalogue's maintainers: its file, the
    lines of the source that print it, and its caption or alt text, in a few forms. The name is the
    first of these that the note holds: the figure's caption; its alt text; a description in
    brackets after the file; its Sphinx label made readable (``DOE-fig-Cube-plot`` is "Cube plot");
    and, last, its file's name made readable. A note that names no file and no label, such as
    "Figure 6.8", is kept as it is, without a note of its lines. The name never holds a file path, a
    note of lines or a label's prefix.

    Parameters
    ----------
    note : str
        A figure, as the ``figures`` of an entry's ``where`` record it.

    Returns
    -------
    str
        The figure's name, with a capital first letter and no full stop.

    Examples
    --------
    >>> figure_name("The thirty batches in run order (../figures/doe/omars-worked-study-plan.png)")
    'The thirty batches in run order'
    >>> figure_name("../figures/doe/x.png (no caption; alt 'Histogram of the differences'), lines 244-248")
    'Histogram of the differences'
    >>> figure_name("DOE-fig-Cube-plot (factorial-two-levels-two-variables-no-analysis.png)")
    'Cube plot'
    >>> figure_name("factorial-two-level-surface-example-cropped.png (true response surface)")
    'True response surface'
    >>> figure_name("Figure 1(a)")
    'Figure 1(a)'
    """
    text, label = note.strip(), ""
    if found := _FIGURE_LABEL.match(text):
        label, text = found[1], text[found.end() :]
    file = _FIGURE_FILE.search(note)
    if file is None and not label:
        # No file and no label: a figure as a book numbers it, such as "Figure 6.8" or "Figure 1(a)".
        opens = _closing_group(text)
        if opens is not None and text[opens + 1 : -1].strip().lower() in _KIND_ONLY:
            text = text[:opens]
        return _readable(text)
    caption, inside = text, ""
    at_start = _FIGURE_FILE.match(text)
    if at_start:  # the file first: "x.png (a note), lines 1-5: a caption"
        rest = text[at_start.end() :].strip()
        closes = _opening_group(rest)
        if closes is not None:
            inside, rest = rest[1:closes], rest[closes + 1 :]
        rest = _LINES_NOTE.sub("", rest).strip()
        caption = rest[1:] if rest.startswith(":") else ""
    elif (opens := _closing_group(text)) is not None:  # a caption first: "A caption (x.png, lines 1-5)"
        caption, inside = text[:opens], text[opens + 1 : -1]
    caption = _FIGURE_FILE.sub("", caption)
    alt = next((a for a in _ALT_TEXT.findall(note) if not _FIGURE_FILE.search(a)), "")
    parts = (_readable(_FIGURE_FILE.sub("", part)) for part in _ALT_TEXT.sub("", inside).split(";"))
    description = next(
        (p for p in parts if p.lower() not in _KIND_ONLY and not p.lower().startswith(("alt", "label "))), ""
    )
    for name in (_readable(caption), _readable(alt), description):
        if name:
            return name
    if label:
        return _readable(re.sub(r"^[A-Za-z]+-fig-", "", label).replace("-", " "))
    return _file_title(file[0]) if file else _readable(text)


#: A note of where in its source file a table is: "(lines 29-39)", "(list-table at line 46, no caption)".
_TABLE_LINES = re.compile(
    r"\s*\((?:[\w-]+\s+)?(?:at\s+)?lines?\s+\d+(?:\s*-\s*\d+)?(?:,\s*no caption)?\)", flags=re.IGNORECASE
)
#: The reStructuredText kinds of table a note names, which a reader knows as a table.
_TABLE_KIND = re.compile(r"\b(?:grid|csv|list)[- ]table\b", flags=re.IGNORECASE)
#: A table named by its kind and then its columns in brackets: "Solution table (Experiment, A, B)".
_TABLE_COLUMNS = re.compile(r"(.*\btable)\s*\(([^()]*)\)", flags=re.IGNORECASE)
#: The role reStructuredText writes mathematics in, such as :math:`R^2_Y`.
_MATH_ROLE = re.compile(r":math:`([^`]*)`")


def table_name(note: str) -> str:
    """Return the name a reader can use for a table, from the catalogue's note of where it is printed.

    The book's entries record each table as a note for the catalogue's maintainers: its caption, or
    its kind and columns when it has none, with the lines of the source that print it. The name keeps
    the caption and drops the note of lines and the column headings written after it; a table with no
    caption is named by its columns. A note that says no table is printed names none, and a table as
    a printed book numbers it, such as "Table 10.1", is kept as it is.

    Parameters
    ----------
    note : str
        A table, as the ``tables`` of an entry's ``where`` record it.

    Returns
    -------
    str
        The table's name, with a capital first letter and no full stop; ``""`` for a note of no table.

    Examples
    --------
    >>> table_name("Factor ranges (list-table at line 46, no caption): Factor / Low / Centre / High")
    'Factor ranges'
    >>> table_name("Grid table, lines 1075-1093 (Experiment, D, R, P, Taste)")
    'Table of Experiment, D, R, P and Taste'
    >>> table_name("Recipe / True titer, g/L / Gain over current (list-table at line 538, no caption)")
    'A table: Recipe / True titer, g/L / Gain over current'
    >>> table_name("Table 10.1")
    'Table 10.1'
    """
    text = _MATH_ROLE.sub(r"\1", note.strip())
    if re.match(r"no table printed\b", text, flags=re.IGNORECASE):
        return ""
    uncaptioned = bool(re.search(r"\bno caption\b", text, flags=re.IGNORECASE))
    text = _LINES_NOTE.sub("", _TABLE_LINES.sub("", text))
    text = re.sub(r"\s*\(no (?:caption|table)\)|,?\s*[\w./-]+\.rst\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:unlabell?ed|untitled)\s+", "", text, flags=re.IGNORECASE)
    text = _TABLE_KIND.sub("table", text).strip(" ,;")
    head, colon, tail = text.partition(": ")
    if colon and " / " in tail and head:  # a caption, then the table's column headings
        text = head
    if found := _TABLE_COLUMNS.fullmatch(text):
        columns = [c.strip() for c in found[2].split(",") if c.strip()]
        listed = ", ".join(columns[:-1]) + f" and {columns[-1]}" if len(columns) > 1 else "".join(columns)
        text = f"{found[1]} of {listed}"
    elif uncaptioned and " / " in text:  # no caption: the column headings name the table
        text = f"A table: {text}"
    return _readable(text)


def _anchor(label: str) -> str:
    """Return the id a Sphinx page gives a section's label: lower case, each run of other characters a hyphen."""
    return re.sub(r"^[-0-9]+|-+$", "", re.sub(r"[^a-z0-9]+", "-", label.lower()))


def where_url(where: dict[str, Any] | str | None) -> str | None:
    """Return the address of the place a source prints an entry: its page, and the section's anchor on it.

    A source read on the web records each entry's page (``where["url"]``); a page of the book also
    records the Sphinx label of the section (``where["label"]``), which the page carries as an anchor.

    Parameters
    ----------
    where : dict, str or None
        An entry's ``where``, as the reference catalogue records it.

    Returns
    -------
    str or None
        The page's address, with ``#`` and the section's anchor when there is a label; ``None`` for a
        source that records no address, such as a printed book.

    Examples
    --------
    >>> where_url({"url": "https://learnche.org/pid/a/b", "label": "DOE-omars-estimability-frontier"})
    'https://learnche.org/pid/a/b#doe-omars-estimability-frontier'
    >>> where_url({"pages": "307-324"}) is None
    True
    """
    if not isinstance(where, dict) or not where.get("url"):
        return None
    url, label = str(where["url"]), where.get("label")
    return f"{url}#{_anchor(str(label))}" if label else url


def shown_where(where: dict[str, Any] | str | None) -> dict[str, Any] | str | None:
    """Return the place a source prints an entry as the catalog shows it: section, pages, tables and figures.

    Each table is named by :func:`table_name` and each figure by :func:`figure_name`; the notes kept
    for the catalogue's maintainers (the source file, its lines and commit) are left out.
    """
    if not isinstance(where, dict):
        return where
    shown = {key: where[key] for key in ("section", "pages") if where.get(key)}
    if tables := [name for name in (table_name(str(t)) for t in where.get("tables", [])) if name]:
        shown["tables"] = tables
    if where.get("figures"):
        shown["figures"] = [figure_name(str(f)) for f in where["figures"]]
    return shown


# --------------------------------------------------------------------------- published records
def _levels(factor: dict[str, Any]) -> list[Any]:
    """Return a categorical factor's levels in order: its ``levels``, else its low and high labels.

    Two levels go low first, as the source codes them (-1 at its low, +1 at its high), in whatever
    order they are listed.
    """
    levels = list(factor.get("levels") or [factor.get("low"), factor.get("high")])
    low, high = factor.get("low"), factor.get("high")
    if low is None or high is None:
        return levels
    same = sorted(map(_level_label, levels)) == sorted(map(_level_label, (low, high)))  # the two, listed in any order
    return [low, high] if same else levels


def _level_label(value: object) -> str:
    """Write a categorical level as text to match on, so a level printed as 2 or 2.0 is the level listed as "2"."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def _prints_levels(column: list[Any], factor: dict[str, Any]) -> bool:
    """Say whether a categorical factor's column prints its levels: every setting printed is one of them."""
    labels = {_level_label(v) for v in _levels(factor)}
    printed = [v for v in column if v is not None and not isinstance(v, bool)]
    return factor.get("type") == "categorical" and bool(printed) and all(_level_label(v) in labels for v in printed)


def _coded_value(value: object, factor: dict[str, Any], *, coded: bool) -> float:
    """Code one printed setting: -1 at the factor's low level, +1 at its high level; NaN when it is no setting.

    A categorical factor's levels are spread evenly from -1 at its first to +1 at its last. A level is found by its
    text, so a lot printed as the number 2 is the level listed as "2".
    """
    if isinstance(value, bool) or value is None:
        return float("nan")
    if isinstance(value, str) or (factor.get("type") == "categorical" and not coded):
        labels = [_level_label(v) for v in _levels(factor)]
        if _level_label(value) in labels and len(labels) > 1:
            return -1 + 2 * labels.index(_level_label(value)) / (len(labels) - 1)
        try:
            value = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")
    number = float(value)  # type: ignore[arg-type]
    low, high = factor.get("low"), factor.get("high")
    if coded or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (low, high)) or low == high:
        return number
    return (number - (high + low) / 2) / ((high - low) / 2)  # type: ignore[operator]


def published_runs(entry: dict[str, Any]) -> np.ndarray | None:
    """Return an entry's runs in coded units, one column per factor; ``None`` when the source prints no runs.

    Parameters
    ----------
    entry : dict
        An experiment or design of the reference catalogue.

    Returns
    -------
    np.ndarray or None
        The coded runs, in the order the source prints them, with NaN for a setting that is not
        printed as one.
    """
    runs, factors = entry.get("runs"), entry.get("factors") or []
    if not runs or not factors:
        return None
    at = {name: i for i, name in enumerate(runs["columns"])}
    if not all(f["name"] in at for f in factors):
        return None
    # A table of coded runs may still print a categorical factor's levels, such as supplier 1, 2 and 3, not their codes.
    coded = bool(runs.get("coded"))
    labelled = {f["name"] for f in factors if _prints_levels([row[at[f["name"]]] for row in runs["rows"]], f)}
    table = np.array(
        [
            [_coded_value(row[at[f["name"]]], f, coded=coded and f["name"] not in labelled) for f in factors]
            for row in runs["rows"]
        ],
        dtype=float,
    )
    # A coded setting printed rounded is the exact one the source worked with, sign kept: 1.414 is the root of 2,
    # as the reference catalogue fits it. A factor fitted in its own units keeps its printed settings.
    for printed, exact in ((float(k), float(v)) for k, v in (runs.get("exact") or {}).items()):
        for j, factor in enumerate(factors):
            if factor.get("coding") != "none":
                column = table[:, j]
                table[:, j] = np.where(np.isclose(np.abs(column), printed), np.sign(column) * exact, column)
    return table


def _response_values(entry: dict[str, Any]) -> dict[str, list[float | None]]:
    """Return each response's printed values, run by run, for the responses the run table prints."""
    runs = entry.get("runs") or {}
    at = {name: i for i, name in enumerate(runs.get("columns", []))}
    out = {}
    for response in entry.get("responses") or []:
        if response["name"] in at:
            column = [row[at[response["name"]]] for row in runs["rows"]]
            out[response["name"]] = [
                float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None for v in column
            ]
    return out


#: The most tables and figures the few words of :func:`_where` name; the others are counted.
PLACES_NAMED = 3


def _where(where: dict[str, Any] | str | None) -> str:
    """Write where a source prints an entry in a few words: its pages, then its first tables and figures."""
    shown = shown_where(where) if where else ""
    if not isinstance(shown, dict):
        return shown or ""
    parts = []
    if pages := shown.get("pages"):
        parts.append(f"p. {pages}" if "-" not in str(pages) and "," not in str(pages) else f"pp. {pages}")
    places = [*shown.get("tables", []), *shown.get("figures", [])]
    parts.extend(places[:PLACES_NAMED])
    if (more := len(places) - PLACES_NAMED) > 0:
        parts.append(f"and {more} more {'table or figure' if more == 1 else 'tables and figures'}")
    if not parts and shown.get("section"):
        parts.append(f"section {shown['section']}")
    return "; ".join(str(p) for p in parts)


#: Words that name blocks without saying the design is run in them: "no blocks", a "code block", the
#: "blocking alpha" of a central composite design, or the name of a file.
_NOT_BLOCKED = re.compile(
    r"\bno blocks\b|\bcode block\b|\bblocking alpha\b|[\w.-]+\.(?:rst|py|csv|txt)\b", re.IGNORECASE
)


def _block_columns(entry: dict[str, Any]) -> list[str] | None:
    """Name the run table's columns of each run's block: those the entry's design names, else one called block or day.

    A source can name its blocks for what they are, such as batch, operator or week; the entry then says so
    with ``design["blocks"]``, a column's name, or a list of them when a block is each combination of their
    values, such as a shift within a replicate. ``design["blocks"]`` false says the design is run in no blocks,
    whatever its text says. Of the arrangements ``design["blockings"]`` lists, the first is taken (see
    :func:`_arrangements`). Without either, a column called block or day is taken.
    """
    design = entry.get("design") or {}
    columns = [str(c) for c in (entry.get("runs") or {}).get("columns", [])]
    named = design.get("blocks", (design.get("blockings") or [None])[0])
    if named is False:
        return None
    if named is not None:
        names = [str(named)] if isinstance(named, str) else [str(c) for c in named]
        return names if all(c in columns for c in names) else None
    found = next((c for c in columns if re.fullmatch(r"(block|day)s?", c, flags=re.IGNORECASE)), None)
    return [found] if found else None


def _set_apart(entry: dict[str, Any]) -> set[str]:
    """Return the columns that are no factor of the design: its blocks, and those of every other way to block it."""
    apart = set(_block_columns(entry) or [])
    for blocks in (entry.get("design") or {}).get("blockings") or []:
        apart |= {blocks} if isinstance(blocks, str) else set(blocks)
    return apart


def _blocks(entry: dict[str, Any], text: str, columns: list[str] | None) -> tuple[bool, list[Any] | None]:
    """Say whether an entry is run in blocks, and each run's block as printed when its run table has them.

    A block of several columns is written as their values together, such as ``I / 1``.
    """
    if (entry.get("design") or {}).get("blocks") is False:
        return False, None
    if columns:
        runs = entry["runs"]
        at = [[str(c) for c in runs["columns"]].index(c) for c in columns]
        if len(at) == 1:
            return True, [row[at[0]] for row in runs["rows"]]
        together = [
            None if any(row[i] is None for i in at) else " / ".join(str(row[i]) for i in at) for row in runs["rows"]
        ]
        return True, together
    return bool(re.search(r"\bblock", _NOT_BLOCKED.sub(" ", text), flags=re.IGNORECASE)), None


def _arrangements(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return an entry once for each way its source arranges the runs in blocks, as ``design["blockings"]`` lists them.

    The first keeps the entry's name and title; each other is named, and titled, for its number of blocks.
    """
    design = entry.get("design") or {}
    blockings = design.get("blockings") or []
    if not blockings or "blocks" in design:
        return [entry]
    out = []
    for i, blocks in enumerate(blockings):
        arranged = {**entry, "design": {**design, "blocks": blocks}}
        if i:
            n = len({str(b) for b in _blocks(arranged, "", _block_columns(arranged))[1] or []})
            title = entry.get("title", entry["name"])
            arranged |= {"name": f"{entry['name']}-in-{n}-blocks", "title": f"{title}, in {n} blocks"}
        out.append(arranged)
    return out


#: The facets :func:`design_facets` reads off the runs beyond the size and the models: unknown without runs.
_MEASURES = (
    "run_loss_model",
    "survives_run_loss",
    "critical_runs",
    "worst_d_ratio",
    "leverage",
    "quadratic_projection",
    "projectivity",
)


def _count(value: object) -> int | None:
    """Return a count a source states as a whole number above 0; ``None`` for a range such as ``"3 to 7"``."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _named_fraction(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray, list[str]] | None:
    """Return the factors, runs and generators of the one fraction an entry names by its generators alone.

    An entry that prints facts about a two-level fraction (its aliases, defining relation or resolution)
    without its runs names the fraction in each check by its generators. It is that fraction when its design
    is a fractional factorial and every check that names generators names the same ones in the same factors;
    else it is ``None``. The runs are the library's fraction from those generators, in standard order (the
    first base factor alternating fastest), each factor coded -1 to +1 and named as the source names it, a
    name such as 1 written X1.
    """
    if "fractional factorial" not in str((entry.get("design") or {}).get("type", "")).lower():
        return None
    named = {
        (tuple(check.get("factors") or ()), tuple(check["generators"]), int(check["k"]))
        for _, check in _checks(entry)
        if check.get("generators") is not None and check.get("k")
    }
    if len(named) != 1:
        return None
    [(names, generators, k)] = named
    check = {"generators": list(generators), **({"factors": list(names)} if names else {})}
    result = _fraction(check, k)
    columns = [str(c) for c in result.factor_names]
    runs = result.design[columns].to_numpy(dtype=float)
    made = {g.split("=")[0].strip() for g in generators}
    base = [j for j, name in enumerate(columns) if name not in made]
    runs = runs[np.lexsort([runs[:, j] for j in base])]  # lexsort's last key sorts slowest: the first base fastest
    factors = [
        {
            "name": name if name.isidentifier() else f"X{name}",
            "description": f"factor {name}",
            "low": -1,
            "high": 1,
            "units": "coded",
            "type": "numeric",
        }
        for name in columns
    ]
    return factors, runs, list(generators)


def _compared_as_levels(entry: dict[str, Any], factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a study's factors with its one factor categorical where its source compares that factor's levels.

    A study of one numeric factor at three or more listed levels whose source fits it as levels, ``C(power)``, is a
    comparison of those levels, as Montgomery's plasma etching compares four RF powers: the catalog takes the factor
    as categorical, labelled by its levels, so the page compares them as the source does. A source that also fits
    the factor as a number keeps those models; the page shows numbers of the comparison only.
    """
    if len(factors) != 1:
        return factors
    (factor,) = factors
    if factor.get("type", "numeric") != "numeric" or len(factor.get("levels") or []) < 3:
        return factors
    as_levels = re.compile(rf"C\(\s*{re.escape(str(factor['name']))}\b")
    if not any(as_levels.search(str(check.get("model") or "")) for _, check in _checks(entry)):
        return factors
    return [{**factor, "type": "categorical"}]


def _kind(entry: dict[str, Any], factors: list[dict[str, Any]]) -> str:
    """Say what an entry of the reference catalogue holds.

    ``experiment``: a design and its results. ``design``: a design's runs, or its factors, without results.
    ``sample``: results with no factor varied, one sample measured over and over. ``facts``: neither
    factors nor results, only what the source prints about designs, such as a table of run sizes.
    """
    if factors:
        return "experiment" if entry.get("responses") else "design"
    return "sample" if entry.get("responses") and (entry.get("runs") or {}).get("rows") else "facts"


def _published_record(entry: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Describe one entry of the reference catalogue as a record of the catalog."""
    design = entry.get("design") or {}
    text = " ".join(str(design.get(key, "")) for key in ("type", "details"))
    family = normalise_family(str(design.get("type", "")))
    # A run's block is no factor of the design: its column is kept apart as the run's block.
    columns, apart = _block_columns(entry), _set_apart(entry)
    factors = _compared_as_levels(entry, [f for f in entry.get("factors") or [] if f["name"] not in apart])
    coded = published_runs({**entry, "factors": factors})
    runs_from = None
    if not factors and (coded is None or not coded.size) and (fraction := _named_fraction(entry)):
        factors, coded, runs_from = fraction  # the source names the fraction by its generators alone
    kinds = [str(f.get("type", "numeric")) for f in factors]
    responses = _response_values(entry)
    blocked, blocks = _blocks(entry, text, columns)
    n_blocks = len({str(b) for b in blocks if b is not None}) if blocks else None
    record: dict[str, Any] = {
        "id": entry["name"],
        "title": entry.get("title", entry["name"]),
        "collection": "published",
        "kind": _kind(entry, factors),
        "family": family,
        "design_type": _DESIGN_TYPES.get(family),
        "design_text": str(design.get("type", "")),
        "source": source["key"],
        "year": source.get("year"),
        "added": str(source.get("read", ""))[:10] or None,
        "where": _where(entry.get("where")),
        "where_url": where_url(entry.get("where")),
        "topics": list(entry.get("topics") or []),
        "blocked": blocked,
        "n_blocks": n_blocks,
        "blocks": blocks,
        "split_plot": family == "split_plot" or bool(re.search(r"split|whole.plot", text, flags=re.IGNORECASE)),
        "has_responses": bool(responses),
        "n_responses": len(entry.get("responses") or []),
        "factors": [
            {key: f.get(key) for key in ("name", "description", "low", "high", "units", "type", "levels") if key in f}
            for f in factors
        ],
        "responses": [
            {"name": r["name"], "units": r.get("units", ""), "description": r.get("description", "")}
            for r in entry.get("responses") or []
        ],
        "details": str(design.get("details", "")),
        "notes": str(entry.get("notes", "")),
        "where_full": entry.get("where"),
        "coded_runs": None,
        "runs_from": runs_from,
        "response_values": responses,
    }
    if coded is not None and coded.size:
        record.update(design_facets(coded, kinds))
        record["coded_runs"] = [[None if np.isnan(v) else round(float(v), 6) for v in row] for row in coded]
    else:
        k = _count(design.get("factors")) or _count(design.get("treatment_factors")) or len(factors) or None
        record.update(
            {
                "n_runs": _count(design.get("n_runs")),
                "n_factors": k,
                "n_numeric": sum(kind == "numeric" for kind in kinds),
                "n_categorical": sum(kind == "categorical" for kind in kinds),
                "n_mixture": sum(kind == "mixture" for kind in kinds),
                "levels": [],
                "n_centre": None,
                "runs_per_parameter": None,
                "models": dict.fromkeys(MODELS),
                "glyph": "",
                **dict.fromkeys(_MEASURES),
            }
        )
    return record


# --------------------------------------------------------------------------- generated records
def _generated_specs() -> list[dict[str, Any]]:
    """List the designs the catalog generates: each family over the factor counts the library makes it for."""
    specs: list[dict[str, Any]] = []
    specs += [{"design_type": "full_factorial", "k": k, "options": {"n_center_points": 0}} for k in range(2, 8)]
    # The highest resolution first: where a lower one asks for no fewer runs, the library makes the same
    # fraction, and :func:`catalog_records` keeps it once, under the resolution it has.
    specs += [
        {"design_type": "fractional_factorial", "k": k, "options": {"resolution": r, "n_center_points": 0}}
        for k in range(3, 12)
        for r in (5, 4, 3)
        if r <= k
    ]
    specs += [{"design_type": "plackett_burman", "k": k, "options": {"n_center_points": 0}} for k in range(2, 12)]
    # A definitive screening or OMARS design counts its own center run among ``n_center_points``: one, as published.
    specs += [{"design_type": "dsd", "k": k, "options": {"n_center_points": 1}} for k in range(3, 13)]
    specs += [{"design_type": "omars", "k": k, "options": {"n_center_points": 1}} for k in range(3, 9)]
    specs += [{"design_type": "box_behnken", "k": k, "options": {"n_center_points": 3}} for k in range(3, 8)]
    specs += [
        {
            "design_type": "ccd",
            "k": k,
            "options": {"n_center_points": 3, "alpha": alpha, **({"cube": "fractional"} if k >= 6 else {})},
        }
        for k in range(2, 8)
        for alpha in ("face_centered", "rotatable")
    ]
    for model, ks in (("main_effects", range(2, 9)), ("interactions", range(2, 7)), ("quadratic", range(2, 6))):
        specs += [
            {"design_type": "d_optimal", "k": k, "options": {"model_type": model, "budget": _terms(k, model) + 4}}
            for k in ks
        ]
    specs += [
        {
            "design_type": "i_optimal",
            "k": k,
            "options": {"model_type": "quadratic", "budget": _terms(k, "quadratic") + 4},
        }
        for k in range(2, 5)
    ]
    specs += [{"design_type": "supersaturated", "k": k, "options": {}} for k in (10, 14, 18, 22)]
    for model in ("linear", "quadratic", "special_cubic", "cubic"):
        specs += [{"design_type": "mixture", "k": k, "options": {"model_type": model}} for k in range(3, 6)]
    specs += [{"design_type": "taguchi", "array": name, "options": {}} for name in ORTHOGONAL_ARRAY_NAMES]
    return specs


def _terms(k: int, model: str) -> int:
    """Count a model's coefficients, intercept included, in ``k`` two-level or three-level factors."""
    return {"main_effects": 1 + k, "interactions": 1 + k + k * (k - 1) // 2, "quadratic": (k + 1) * (k + 2) // 2}[model]


#: Every design the catalog generates, as ``{"design_type", "k", "options"}``: the shape the reference
#: catalogue's checks name a library design in. Orthogonal arrays are named by ``array`` instead of ``k``.
GENERATED: tuple[dict[str, Any], ...] = tuple(_generated_specs())

#: The names factors take in a generated design, as the library names them (no I, which reads as 1).
_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"

_GENERATED_FAMILY = {"d_optimal": "optimal", "i_optimal": "optimal", "taguchi": "orthogonal_array"}


def _spec_id(spec: dict[str, Any]) -> str:
    """Name a generated design: its type, its factor count, and the options that set it apart."""
    if "array" in spec:
        return "lib-taguchi-" + re.sub(r"[^a-z0-9]+", "-", spec["array"].lower()).strip("-")
    parts = [spec["design_type"].replace("_", "-"), f"k{spec['k']}"]
    options = spec["options"]
    if "resolution" in options:
        parts.append(f"res{options['resolution']}")
    if "alpha" in options:
        parts.append(str(options["alpha"]).replace("_", "-"))
    if "model_type" in options:
        parts.append(str(options["model_type"]).replace("_", "-"))
    return "lib-" + "-".join(parts)


def _generated_runs(spec: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Make a generated design; return its coded runs in standard order, its factors' kinds, and its result's facts."""
    if "array" in spec:
        array = np.asarray(orthogonal_array(spec["array"]), dtype=float)
        top = array.max(axis=0)
        coded = np.where(top > 0, -1 + 2 * array / np.where(top > 0, top, 1), 0.0)
        return coded, ["numeric"] * coded.shape[1], {"array": spec["array"]}
    kind = "mixture" if spec["design_type"] == "mixture" else "numeric"
    low = 0 if kind == "mixture" else -1
    library_kind = FactorType.mixture if kind == "mixture" else FactorType.continuous
    factors = [Factor(name=_LETTERS[i], low=low, high=1, type=library_kind) for i in range(spec["k"])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = generate_design(factors, design_type=spec["design_type"], **spec["options"])
    names = [f.name for f in factors]
    order = np.argsort(np.asarray(result.run_order))
    runs = result.design[names].to_numpy(dtype=float)[order]
    if kind == "mixture":
        runs = 2 * runs - 1  # a proportion from 0 to 1, drawn as a setting from -1 to +1
    facts = {"resolution": result.resolution, "blocks": result.blocks is not None}
    return runs, [kind] * len(names), facts


def _generated_record(spec: dict[str, Any]) -> dict[str, Any]:
    """Describe one design the library generates as a record of the catalog."""
    coded, kinds, facts = _generated_runs(spec)
    design_type = spec["design_type"]
    family = _GENERATED_FAMILY.get(design_type, design_type)
    k = coded.shape[1]
    title = f"{FAMILIES[family].label}, {k} factors"
    if "array" in spec:
        title = f"Orthogonal array {spec['array']}"
    elif design_type in {"d_optimal", "i_optimal"}:
        model = spec["options"]["model_type"].replace("_", " ")
        title = f"{design_type[0].upper()}-optimal for the {model} model, {k} factors"
    elif "alpha" in spec["options"]:
        title += ", " + spec["options"]["alpha"].replace("_", "-")
    elif design_type == "mixture":
        title += f", {spec['options']['model_type'].replace('_', ' ')} model"
    if facts.get("resolution"):
        title += f", resolution {['', '', '', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'][facts['resolution']]}"
    names = list(_LETTERS[:k]) if "array" not in spec else [f"X{j + 1}" for j in range(k)]
    record: dict[str, Any] = {
        "id": _spec_id(spec),
        "title": title,
        "collection": "generated",
        "kind": "design",
        "family": family,
        "design_type": design_type,
        "design_text": title,
        "spec": spec,
        "source": None,
        "year": None,
        "added": None,
        "where": "",
        "where_url": None,
        "topics": [],
        "blocked": bool(facts.get("blocks")),
        "n_blocks": None,
        "blocks": None,
        "split_plot": False,
        "has_responses": False,
        "n_responses": 0,
        "factors": [{"name": n, "low": -1 if kinds[0] != "mixture" else 0, "high": 1, "type": kinds[0]} for n in names],
        "responses": [],
        "details": "",
        "notes": "",
        "where_full": None,
        "coded_runs": coded.round(6).tolist(),
        "runs_from": None,
        "response_values": {},
    }
    record.update(design_facets(coded, kinds))
    return record


# --------------------------------------------------------------------------- the catalog
#: The fields of a record that the catalog page reads for every design at once: the facets it searches,
#: filters, counts, maps and draws. The rest (factors, runs, notes) is read when one design is opened.
INDEX_FIELDS = (
    "id",
    "title",
    "collection",
    "kind",
    "family",
    "design_type",
    "source",
    "year",
    "added",
    "where",
    "topics",
    "n_runs",
    "n_factors",
    "n_numeric",
    "n_categorical",
    "n_mixture",
    "levels",
    "n_centre",
    "runs_per_parameter",
    "models",
    "blocked",
    "n_blocks",
    "split_plot",
    "has_responses",
    "n_responses",
    "checks",
    "glyph",
    "survives_run_loss",
    "critical_runs",
    "worst_d_ratio",
    "quadratic_projection",
    "projectivity",
    "design_class",
)


def _checks(entry: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return each printed number of an entry with each of its checks: one number may carry several."""
    return [
        (expected, check)
        for expected in entry.get("expected", [])
        for check in (
            expected.get("check") if isinstance(expected.get("check"), list) else [expected.get("check") or {}]
        )
    ]


def printed_checks(entry: dict[str, Any]) -> dict[str, int]:
    """Count an entry's printed numbers: all of them, those recomputed, those that agree, those known not to.

    :func:`~process_improve.experiments.reference_catalogue.reproduce_reference_catalogue` recomputes
    every number that carries a check, and the test suite requires each to agree with the printed
    text unless the catalogue records why it does not (its ``disagreement``). So, on a commit whose
    tests pass, the counts here are what that report finds, without the minutes it takes to run.

    Parameters
    ----------
    entry : dict
        An experiment or design of the reference catalogue.

    Returns
    -------
    dict[str, int]
        ``printed`` (every number the source prints for it), ``checked`` (those recomputed),
        ``agree`` (those that reproduce) and ``explained`` (those known not to, with the reason).
    """
    checks = [check for _, check in _checks(entry)]
    checked = [check for check in checks if "kind" in check]
    explained = sum(bool(check.get("disagreement")) for check in checked)
    return {"printed": len(checks), "checked": len(checked), "agree": len(checked) - explained, "explained": explained}


def catalog_records(collection: Collection = "both") -> list[dict[str, Any]]:
    """Return the catalog: a record for each published design, each generated design, or both.

    Parameters
    ----------
    collection : {"both", "published", "generated"}
        Which designs to describe: the reference catalogue's, the ones :data:`GENERATED` lists, or both.

    Returns
    -------
    list[dict]
        One record per design. ``kind`` says what a published entry holds (see :func:`_kind`): an
        ``experiment`` (a design and its results), a ``design``, a ``sample`` (results with no factor
        varied) or ``facts`` about designs (neither factors nor runs); a generated record is a ``design``.
        An entry that names one two-level fraction by its generators alone has the fraction the library
        makes from them as its runs, and those generators in ``runs_from``; it is ``None`` for every
        other record. The fields in :data:`INDEX_FIELDS` describe it for searching; a
        published record's ``checks`` counts its printed numbers (see :func:`printed_checks`) and
        a generated record's is ``None``. Records with the same ``design_class`` hold one design,
        printed in different sources or made by the library (see
        :func:`~process_improve.experiments.design_measures.design_classes`). ``factors``,
        ``responses``, ``coded_runs``, ``response_values``, ``details``, ``notes`` and
        ``where_full`` hold what a reader sees on opening a design, and ``where_url`` the address of
        the place its source prints it (see :func:`where_url`). ``blocks`` holds each run's block as
        its source prints it, where the run table has them, and is ``None`` otherwise: a block is
        not one of the ``factors``, nor a column of ``coded_runs``. ``topics`` lists a published
        entry's applications (see :func:`catalog_topics`); a generated design has none. ``listed``
        says whether a published entry is one of
        :func:`~process_improve.experiments.reference_catalogue.reference_experiments` or of
        :func:`~process_improve.experiments.reference_catalogue.reference_designs`, and ``spec``
        names a generated design as :data:`GENERATED` does.

    Raises
    ------
    ValueError
        If *collection* is none of the three.
    """
    if collection not in {"both", "published", "generated"}:
        raise ValueError(f"Unknown collection {collection!r}: choose 'both', 'published' or 'generated'.")
    records: list[dict[str, Any]] = []
    if collection in {"both", "published"}:
        sources = reference_sources()
        listed = [
            *(("experiments", e) for e in reference_experiments()),
            *(("designs", d) for d in reference_designs()),
        ]
        records += [
            {**_published_record(arranged, sources[entry["source"]]), "checks": printed_checks(entry), "listed": where}
            for where, entry in listed
            for arranged in _arrangements(entry)
        ]
    if collection in {"both", "generated"}:
        made: set[tuple[str, frozenset]] = set()
        for spec in GENERATED:
            record = _generated_record(spec)
            runs = frozenset(run_counts(np.array(record["coded_runs"], dtype=float)).items())
            if (spec["design_type"], runs) not in made:  # two options that make one design: kept once
                made.add((spec["design_type"], runs))
                records.append({**record, "checks": None, "listed": None})
    return _classified(records)


def _classified(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give each record its ``design_class``: records of one class hold one design; ``None`` without full runs."""
    runs = {
        r["id"]: np.array(r["coded_runs"], dtype=float)
        for r in records
        if r["coded_runs"] and not r["n_mixture"] and all(v is not None for row in r["coded_runs"] for v in row)
    }
    classes = design_classes(runs)
    return [{**record, "design_class": classes.get(record["id"])} for record in records]


def short_citation(cite: str) -> str:
    """Return the authors' surnames of a citation, as a list of designs names its source: "Box, Hunter and Hunter".

    Examples
    --------
    >>> short_citation("Núñez Ares, J. and Goos, P. (2020), Enumeration and multicriteria selection")
    'Núñez Ares and Goos'
    """
    names = re.sub(r"\b(?:[A-Z]\.\s*)+|\(ed\.\)", "", cite.split(" (", 1)[0])  # initials, and an editor's "(ed.)"
    names = re.sub(r"\s*,\s*(?=,|and\b|$)", " ", names)  # the commas the initials leave behind
    names = re.sub(r"\s+", " ", re.sub(r"\s+,", ",", names)).strip(" ,")
    return names if len(names) <= 40 else names.split(", ", 1)[0]


def catalog_sources() -> dict[str, dict[str, Any]]:
    """Return what the catalog shows of each published source: its citation, address, year and terms of use.

    ``short`` names the source as the list of designs does: its authors' surnames (see :func:`short_citation`).
    """
    keys = ("title", "authors", "year", "doi", "url", "cite", "rights")
    return {
        key: {**{k: source.get(k) for k in keys}, "short": short_citation(source.get("cite") or source["title"])}
        for key, source in reference_sources().items()
    }


#: What the catalog calls the topic of an entry whose source states no application.
GENERAL_LABEL = "General (no stated application)"


def catalog_topics() -> dict[str, dict[str, str]]:
    """Return each application the catalog filters on: its name on the page, and what it covers.

    The applications are the subject-matter topics of the reference catalogue (see
    :func:`~process_improve.experiments.reference_catalogue.reference_topics`), in its order:
    semiconductors, food and beverages, and so on. ``"general"``, the topic of an entry whose source
    states no application, is named :data:`GENERAL_LABEL`.

    Examples
    --------
    >>> catalog_topics()["food-beverages"]["label"]
    'Food and beverages'
    """
    return {
        key: {"label": GENERAL_LABEL if key == "general" else topic["label"], "description": topic["description"]}
        for key, topic in reference_topics().items()
    }


def catalog_families() -> dict[str, dict[str, Any]]:
    """Return each family's name on the page, what it is, the work that first published it, and where it can lead.

    ``description`` says what a design of the family is in plain words, as the knowledge base of design
    types describes it to a newcomer (its ``description["novice"]``), or ``""`` where it has none.
    ``next`` lists the families a design of this one can be augmented to, each with how (``how``), as
    the knowledge base records them (its ``can_augment_to``).
    """
    types = load_knowledge_graph().design_types
    return {
        key: {
            "label": f.label,
            "description": " ".join(types[key].description.get("novice", "").split()) if key in types else "",
            "year": f.year,
            "reference": f.reference,
            "next": [
                {"family": step["target"], "how": step["method"]}
                for step in (types[key].can_augment_to if key in types else [])
                if step["target"] in FAMILIES
            ],
        }
        for key, f in FAMILIES.items()
    }


# --------------------------------------------------------------------------- errata, and the feed of sources
def _errata_where(where: dict[str, Any] | str | None) -> str:
    """Say in a few words where a source prints a number: its page, its file and line, or its quiz and question."""
    if not isinstance(where, dict):
        return str(where or "")
    if where.get("page"):
        return f"p. {where['page']}"
    if where.get("pdf_page"):
        return f"PDF p. {where['pdf_page']}"
    if where.get("quiz"):
        return f"quiz {where['quiz']}, question {where.get('question', '')}".strip(", ")
    if where.get("file"):
        return f"{where['file']}, line {where['line']}" if where.get("line") else str(where["file"])
    return ""


def catalog_errata() -> list[dict[str, Any]]:
    """List every printed number of the reference catalogue that the library does not reproduce, with the reason.

    The test suite requires every recomputed number to agree with its source unless the catalogue records
    why it does not (its ``disagreement``): the source's rounding, a misprint, or a method that differs
    from the library's. This lists each of those, in the catalogue's order.

    Returns
    -------
    list[dict]
        One per number: ``id`` (the entry that prints it), ``source``, ``what`` and ``term`` (the
        quantity), ``printed`` (the value as printed), ``where`` (its page, in a few words), ``url``
        (its address, for a web source), ``quote`` (the source's words around it), ``reason`` (why
        the library's value differs) and ``rounding`` (whether the reason mentions rounding).

    Examples
    --------
    >>> from process_improve.experiments.design_catalog import catalog_errata
    >>> first = catalog_errata()[0]
    >>> sorted(first)[:4]
    ['id', 'printed', 'quote', 'reason']
    """
    found = []
    for entry in [*reference_experiments(), *reference_designs()]:
        for expected, check in _checks(entry):
            if "kind" not in check or not check.get("disagreement"):
                continue
            where = expected.get("where")
            reason = str(check["disagreement"])
            found.append(
                {
                    "id": entry["name"],
                    "source": entry["source"],
                    "what": expected.get("what"),
                    "term": expected.get("term"),
                    "printed": expected.get("value"),
                    "where": _errata_where(where),
                    "url": where.get("url") if isinstance(where, dict) else None,
                    "quote": where.get("quote") if isinstance(where, dict) else None,
                    "reason": reason,
                    "rounding": bool(re.search(r"\bround", reason, flags=re.IGNORECASE)),
                }
            )
    return found


def catalog_feed(records: Sequence[dict[str, Any]]) -> str:
    """Return an Atom feed of the catalog's sources: one entry for each, by the date it entered the catalogue.

    A reader who follows the feed learns when a source is added. Each entry links to the catalog filtered
    to the source's designs, and says how many it holds and how many of their printed numbers the library
    recomputes. A source whose date of entry is not recorded is left out.

    Parameters
    ----------
    records : Sequence[dict]
        Records of the catalog (see :func:`catalog_records`); the published ones are counted by source.

    Returns
    -------
    str
        The feed, as an XML document, the newest source first.
    """
    atom = "http://www.w3.org/2005/Atom"
    ET.register_namespace("", atom)

    def add(parent: ET.Element, tag: str, text: str | None = None, **attributes: str) -> ET.Element:
        child = ET.SubElement(parent, f"{{{atom}}}{tag}", attributes)
        child.text = text
        return child

    held: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["collection"] == "published" and record.get("added"):
            held.setdefault(record["source"], []).append(record)
    sources = catalog_sources()
    dated = sorted(held, key=lambda key: (held[key][0]["added"], key), reverse=True)
    feed = ET.Element(f"{{{atom}}}feed")
    add(feed, "title", "Design catalog: the sources it holds")
    add(feed, "subtitle", "Published designed experiments, each source added with its runs and printed numbers.")
    add(feed, "id", f"{CATALOG_URL}feed.xml")
    add(feed, "link", rel="self", href=f"{CATALOG_URL}feed.xml")
    add(feed, "link", rel="alternate", href=CATALOG_URL)
    add(feed, "updated", f"{held[dated[0]][0]['added']}T00:00:00Z" if dated else "1970-01-01T00:00:00Z")
    add(add(feed, "author"), "name", "process-improve")
    for key in dated:
        source, entries = sources[key], held[key]
        checks = [r["checks"] for r in entries if r.get("checks")]
        recomputed = sum(c["checked"] for c in checks)
        printed = sum(c["printed"] for c in checks)
        item = add(feed, "entry")
        add(item, "title", f"{source['short']} ({source['year']}): {len(entries)} designs and experiments")
        add(item, "id", f"{CATALOG_URL}#src={key}")
        add(item, "link", rel="alternate", href=f"{CATALOG_URL}#src={key}")
        add(item, "updated", f"{entries[0]['added']}T00:00:00Z")
        add(
            item,
            "summary",
            f"{source['cite']}. {len(entries)} entries enter the catalog, with {printed} printed numbers, "
            f"{recomputed} of them recomputed by the library.",
        )
    ET.indent(feed)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(feed, encoding="unicode") + "\n"


# --------------------------------------------------------------------------- comparing designs
#: The most factors, and the most runs, of a design the catalog scores for comparison.
COMPARED_FACTORS, COMPARED_RUNS = 12, 1000


def _comparable(record: dict[str, Any]) -> np.ndarray | None:
    """Return a record's coded runs when its factors are all numeric and every setting is given; else ``None``."""
    runs, k = record.get("coded_runs"), record.get("n_factors") or 0
    if not runs or record.get("n_numeric") != k or not 1 <= k <= COMPARED_FACTORS or len(runs) > COMPARED_RUNS:
        return None
    if any(v is None for row in runs for v in row):
        return None
    return np.array(runs, dtype=float)


def catalog_comparisons(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Score each design of numeric factors on the measures of ``compare_designs``, for each model, with its FDS curve.

    A design is scored on its own: its values do not depend on which other designs are scored, so any
    designs with the same number of factors can be set side by side afterwards, as
    :func:`~process_improve.experiments.compare.compare_designs` would set them. Designs with a
    categorical or mixture factor, a missing setting, more than :data:`COMPARED_FACTORS` factors or
    more than :data:`COMPARED_RUNS` runs are not scored.

    Parameters
    ----------
    records : Sequence[dict]
        Records of the catalog (see :func:`catalog_records`).

    Returns
    -------
    dict
        For each scored design's ``id``, for each model in
        :data:`~process_improve.experiments.design_measures.MODELS`, the values ``compare_designs``
        reports (the keys of :data:`~process_improve.experiments.compare.MEASURES`, and the model
        matrix's ``rank``), to four significant figures, and ``fds``, the design's
        :func:`~process_improve.experiments.design_measures.fds_curve` for the model (``None`` when
        the design cannot fit it).

    Examples
    --------
    >>> from process_improve.experiments.design_catalog import catalog_comparisons, catalog_records
    >>> ccd = [r for r in catalog_records("generated") if r["id"] == "lib-ccd-k3-face-centered"]
    >>> scored = catalog_comparisons(ccd)["lib-ccd-k3-face-centered"]
    >>> scored["quadratic"]["fits"], scored["quadratic"]["residual_df"]
    (True, 7)
    """
    keys = {measure.key for measure in MEASURES} | {"rank"}
    by_size: dict[int, dict[str, np.ndarray]] = {}
    for record in records:
        if (runs := _comparable(record)) is not None:
            by_size.setdefault(runs.shape[1], {})[record["id"]] = runs
    scored: dict[str, dict[str, dict[str, Any]]] = {}
    for designs in by_size.values():
        for model in MODELS:
            comparison = compare_designs(designs, model)
            for (name, runs), label in zip(designs.items(), comparison.designs, strict=True):
                values = {
                    key: float(f"{value:.4g}") if isinstance(value, float) else value
                    for key, value in comparison.values[label].items()
                    if key in keys
                }
                values["fds"] = fds_curve(runs, ["numeric"] * runs.shape[1], model)
                scored.setdefault(name, {})[model] = values
    return scored


# --------------------------------------------------------------------------- searching the catalog
#: Where the catalog is published: a design's page is this, then ``#d=`` and its id.
CATALOG_URL = "https://factori.al/catalog/"


@functools.cache
def _searchable() -> tuple[dict[str, Any], ...]:
    """Return every record's facets once, built on the first search: building the catalog takes seconds."""
    keys = (*INDEX_FIELDS, "design_text", "where_url")
    return tuple({key: record[key] for key in keys} for record in catalog_records())


def _within(value: float | None, low: float | None, high: float | None) -> bool:
    """Say whether a value lies between two bounds, either of them absent; a missing value lies within none."""
    if low is None and high is None:
        return True
    return value is not None and (low is None or value >= low) and (high is None or value <= high)


@dataclass(frozen=True)
class CatalogQuery:
    """The filters of a search of the catalog, each one a design must pass, and the order of what passes.

    Every filter left at its default keeps every design.

    Attributes
    ----------
    min_factors, max_factors : int or None
        The fewest and most factors.
    min_runs, max_runs : int or None
        The fewest and most runs.
    model : {"main_effects", "interactions", "quadratic"} or None
        A model the design must estimate in full.
    min_categorical, max_categorical : int or None
        The fewest and most categorical factors.
    families : tuple of str or None
        The families to keep (keys of :data:`FAMILIES`).
    topics : tuple of str or None
        The applications to keep (keys of :func:`catalog_topics`): a design passes when it lists any
        of them. A generated design has no application, so any application drops them.
    collection : {"both", "published", "generated"}
        Published designs, those the library makes, or both.
    has_results : bool
        Keep only published experiments printed with their results.
    survives_run_loss : bool
        Keep only designs that still estimate their largest model after losing any one run.
    min_quadratic_projection : int or None
        Keep only designs whose every projection onto this many factors estimates the full quadratic.
    min_projectivity : int or None
        Keep only two-level designs of at least this projectivity.
    year_from, year_to : int or None
        The years the source appeared in; a generated design has none, so either drops them.
    text : str
        Words, each looked for in the design's title, family, design type, source and applications.
    sort : {"runs", "factors", "newest"}
        The order: fewest runs, fewest factors, or the most recent source first.
    """

    min_factors: int | None = None
    max_factors: int | None = None
    min_runs: int | None = None
    max_runs: int | None = None
    model: str | None = None
    min_categorical: int | None = None
    max_categorical: int | None = None
    families: tuple[str, ...] | None = None
    topics: tuple[str, ...] | None = None
    collection: Collection = "both"
    has_results: bool = False
    survives_run_loss: bool = False
    min_quadratic_projection: int | None = None
    min_projectivity: int | None = None
    year_from: int | None = None
    year_to: int | None = None
    text: str = ""
    sort: Literal["runs", "factors", "newest"] = "runs"


def _passes(r: dict[str, Any], q: CatalogQuery, haystack: str) -> bool:
    """Say whether one design's facets pass every filter of a query."""
    return (
        (q.collection == "both" or r["collection"] == q.collection)
        and _within(r["n_factors"], q.min_factors, q.max_factors)
        and _within(r["n_runs"], q.min_runs, q.max_runs)
        and (q.model is None or r["models"].get(q.model) is True)
        and _within(r["n_categorical"], q.min_categorical, q.max_categorical)
        and (q.families is None or r["family"] in q.families)
        and (q.topics is None or any(topic in q.topics for topic in r["topics"]))
        and (not q.has_results or r["has_responses"])
        and (not q.survives_run_loss or r["survives_run_loss"] is True)
        and (q.min_quadratic_projection is None or (r["quadratic_projection"] or 0) >= q.min_quadratic_projection)
        and (q.min_projectivity is None or (r["projectivity"] or 0) >= q.min_projectivity)
        and _within(r["year"], q.year_from, q.year_to)
        and all(word in haystack for word in q.text.lower().split())
    )


def search_catalog(query: CatalogQuery | None = None, **filters: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Return the catalog's designs that pass every filter given, as the catalog page filters them.

    Parameters
    ----------
    query : CatalogQuery, optional
        The filters and the order; every design, the fewest runs first, when not given.
    **filters
        Fields of :class:`CatalogQuery`, by name, such as ``min_factors=6``: they replace the
        query's.

    Returns
    -------
    list[dict]
        Each design's facets (the fields of :data:`INDEX_FIELDS`), with ``family_label``,
        ``source_short`` (its source's authors and year), ``where_url`` (the address of the place
        its source prints it, see :func:`where_url`), ``same_design`` (how many other designs of
        the catalog are this one) and ``url`` (its page in the catalog).

    Raises
    ------
    ValueError
        If a family is not one of :data:`FAMILIES`, an application not one of :func:`catalog_topics`,
        or the model not one of ``MODELS``.
    TypeError
        If a filter is not a field of :class:`CatalogQuery`.

    Examples
    --------
    >>> found = search_catalog(min_factors=6, max_factors=6, max_runs=17, min_quadratic_projection=3)
    >>> sorted({d["family"] for d in found})
    ['dsd', 'omars']
    """
    for name in ("families", "topics"):
        if filters.get(name) is not None:
            filters[name] = tuple(filters[name])
    q = dataclasses.replace(query or CatalogQuery(), **filters)
    families, model, topics = q.families, q.model, catalog_topics()
    if families is not None and (unknown := set(families) - set(FAMILIES)):
        raise ValueError(f"Unknown families {sorted(unknown)}: choose from {', '.join(FAMILIES)}.")
    if q.topics is not None and (unknown := set(q.topics) - set(topics)):
        raise ValueError(f"Unknown applications {sorted(unknown)}: choose from {', '.join(topics)}.")
    if model is not None and model not in MODELS:
        raise ValueError(f"Unknown model {model!r}: choose from {', '.join(MODELS)}.")
    sources = catalog_sources()
    records = _searchable()
    classes = Counter(r["design_class"] for r in records if r["design_class"])
    found = []
    for r in records:
        source = sources.get(r["source"] or "", {})
        applications = [topics[t]["label"] for t in r["topics"] if t != "general"]
        words = (r["id"], r["title"], r["design_text"], FAMILIES[r["family"]].label, source.get("cite"), *applications)
        haystack = " ".join(str(v) for v in words).lower()
        if _passes(r, q, haystack):
            found.append(
                {
                    **{key: r[key] for key in INDEX_FIELDS if key != "glyph"},
                    "family_label": FAMILIES[r["family"]].label,
                    "source_short": f"{source['short']} ({source['year']})" if source else None,
                    "where_url": r["where_url"],
                    "same_design": classes[r["design_class"]] - 1 if r["design_class"] else 0,
                    "url": f"{CATALOG_URL}#d={r['id']}",
                }
            )
    order = {
        "runs": lambda d: (d["n_runs"] is None, d["n_runs"] or 0, d["n_factors"] or 0),
        "factors": lambda d: (d["n_factors"] is None, d["n_factors"] or 0, d["n_runs"] or 0),
        "newest": lambda d: (d["year"] is None, -(d["year"] or 0)),
    }[q.sort]
    return sorted(found, key=order)
