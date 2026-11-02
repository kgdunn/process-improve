"""Designed experiments from published sources other than the book, as the page opens them.

``web/published-examples.json`` lists the examples Factorial opens from ``?example=<name>``. Each
is an experiment of the same name in the reference catalogue
(``process_improve.experiments.reference_catalogue``), which holds the source's citation, where
each example and each printed number sits, and every number the source prints;
``tests/test_reference_catalogue.py`` reproduces those numbers with the library. Here each example
is checked to open the catalogue experiment's runs and results, and every printed number the page
itself shows (coefficients, effects, R², Q², the residual error, a fraction's generators and aliases)
is checked through the page's entry point, as the page analyses the campaign its link opens.
"""

from __future__ import annotations

import json
import re
from functools import cache
from itertools import combinations
from typing import TYPE_CHECKING

import pytest

from process_improve.experiments.reference_catalogue import _agrees, _plain, reference_experiments, reference_sources
from tests.web._support import WEB, no_bytecode

if TYPE_CHECKING:
    from collections.abc import Callable


@cache
def catalogue() -> dict[str, dict]:
    """Return the examples the page opens, by name, as ``web/published-examples.json`` lists them."""
    examples = json.loads((WEB / "published-examples.json").read_text(encoding="utf-8"))["examples"]
    return {entry["name"]: entry for entry in examples}


REFERENCE = {e["name"]: e for e in reference_experiments()}
EXAMPLES = catalogue()


def test_every_example_is_in_the_reference_catalogue():
    sources = reference_sources()
    for name, entry in EXAMPLES.items():
        assert name in REFERENCE, f"{name}: no experiment of this name in the reference catalogue"
        assert REFERENCE[name]["source"] == entry["source"], name
        assert sources[entry["source"]]["url"].startswith("https://doi.org/"), name
        assert re.search(r"\bpp?\. \d", entry["where"]), f"{name}: where says no page"


def test_every_name_opens_one_example_whether_from_the_book_or_another_source():
    book = json.loads((WEB / "book-examples.json").read_text(encoding="utf-8"))["examples"]
    names = [entry["name"] for entry in book] + list(EXAMPLES)
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) for name in names)


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_the_page_opens_the_catalogue_s_runs_and_results(name):
    """The same factors, the same coded settings in the same order, and the same results."""
    entry, reference = EXAMPLES[name], REFERENCE[name]
    factors = [{k: f[k] for k in ("name", "low", "high", "units")} for f in reference["factors"]]
    assert entry["design"]["factors"] == factors
    runs = reference["runs"]
    assert runs["coded"], "the reference keeps these runs in coded units, as the page's catalogue does"
    columns = [f["name"] for f in reference["factors"]] + [r["name"] for r in reference["responses"]]
    at = [runs["columns"].index(c) for c in columns]
    assert entry["runs"] == [[row[i] for i in at] for row in runs["rows"]]


# --------------------------------------------------------------------------- what the page shows
@pytest.fixture(scope="module")
def analysis_of(api, build_script) -> Callable[[str], dict]:
    """Analyse an example, once per module, as the page does once its link has opened it."""

    @cache
    def analysis(name: str) -> dict:
        with no_bytecode():
            opened = build_script.published_example(EXAMPLES[name])["campaign"]
        held = {key: opened["design"][key] for key in ("spec", "columns", "rows")}
        return api.ok("api_analyze_runs", {**held, "responses": opened["responses"], "model": opened["model"]})

    return analysis


@pytest.fixture(scope="module")
def design_of(api) -> Callable[[str], dict]:
    return cache(lambda name: api.ok("api_make_design", EXAMPLES[name]["design"]))


def terms_of(model: str) -> set[frozenset[str]]:
    """Expand a model such as ``A*B`` or ``A + B + A:B`` into its terms, each a set of factor names."""
    terms: set[frozenset[str]] = set()
    for part in (p.strip() for p in model.split("+")):
        names = [n.strip() for n in re.split(r"[*:]", part)]
        if "*" in part:
            terms |= {frozenset(c) for r in range(1, len(names) + 1) for c in combinations(names, r)}
        else:
            terms.add(frozenset(names))
    return terms


def page_terms(analysis: dict) -> set[frozenset[str]]:
    return {frozenset(c["label"].split("·")) for c in analysis["coefficients"] if c["term"] != "Intercept"}


SUMMARY = {
    "r_squared": "r_squared",
    "r_squared_pred": "r_squared_pred",
    "residual_df": "df_residual",
    "df_model": "df_model",
    "ms_residual": "mse_residual",
}


def shown(analysis: dict, check: dict) -> float:
    """Return what the page shows for a check: a coefficient or effect, or one of its fit statistics."""
    kind = check["kind"]
    if kind in ("coefficient", "effect"):
        found = next(c["coefficient"] for c in analysis["coefficients"] if c["label"] == check["term"])
        return found * (2 if kind == "effect" else 1)
    if kind == "residual_std_error":
        return analysis["summary"]["mse_residual"] ** 0.5
    return analysis["summary"][SUMMARY[kind]]


def page_checks() -> list[tuple[str, int]]:
    """Name each printed number the page shows: fitted with a model of every run."""
    return [
        (name, i)
        for name in EXAMPLES
        for i, expected in enumerate(REFERENCE[name]["expected"])
        if expected["check"].get("model")
        and not expected["check"].get("runs")
        and expected["check"].get("kind") in ("coefficient", "effect", "residual_std_error", *SUMMARY)
    ]


@pytest.mark.parametrize("case", page_checks(), ids=lambda c: f"{c[0]}:{c[1]}")
def test_every_printed_number_the_page_shows(analysis_of, case):
    name, i = case
    expected = REFERENCE[name]["expected"][i]
    check = expected["check"]
    analysis = analysis_of(name)
    if terms_of(check["model"]) != page_terms(analysis):
        pytest.skip("the source fits another model than the page's")
    found = shown(analysis, check)
    printed = check.get("printed", expected["value"])
    assert _agrees(found, printed, expression=bool(check.get("expression"))), (
        f"{check['kind']} {check.get('term')}: printed {printed}, the page {found}"
    )


@pytest.mark.parametrize(
    "name", [n for n in EXAMPLES if any(e["check"].get("kind") == "generator" for e in REFERENCE[n]["expected"])]
)
def test_the_fraction_the_page_makes_is_the_source_s(design_of, name):
    facts = design_of(name)["facts"]
    aliases = {(_plain(a), _plain(b)) for a, b in facts["aliases"]}
    for expected in REFERENCE[name]["expected"]:
        check = expected["check"]
        if check.get("kind") == "generator":
            assert _plain(expected["value"]) in {_plain(g) for g in facts["generators"]}
        if check.get("kind") == "aliases":
            assert (check["term"], _plain(expected["value"])) in aliases
