"""Designed experiments from published sources other than the book, checked against what the source prints.

``web/published-examples.json`` holds each example's runs and results with where they came from:
a citation per source, and the section, figures, tables and pages each example and each printed
result sits on. Every example the page opens is made into the campaign its link opens
(``published_example`` in ``scripts/build_web_app.py``) and analysed through the page's entry point;
results the page does not show (fitted values, residuals, each run predicted without it) are taken
from the library's own analysis of the same runs. Every printed number must come back within the
source's own rounding, so a change to the app or the library that contradicts a source fails here.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from decimal import Decimal
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import analyze_experiment
from tests.web._support import WEB, no_bytecode

if TYPE_CHECKING:
    from collections.abc import Callable


@cache
def catalogue() -> dict:
    """Return ``web/published-examples.json`` as written."""
    return json.loads((WEB / "published-examples.json").read_text(encoding="utf-8"))


EXAMPLES = {entry["name"]: entry for entry in catalogue()["examples"]}
LIBRARY_EXAMPLES = {entry["name"]: entry for entry in catalogue()["library_examples"]}


def tolerance(printed: str) -> float:
    """Half a unit in the last digit the source printed: the most its rounding accounts for."""
    exponent = Decimal(printed).normalize().as_tuple().exponent
    return 0.5 * 10.0 ** min(0, int(exponent)) + 1e-9


def assert_printed(found: float, printed: str, what: str) -> None:
    assert abs(found - float(printed)) <= tolerance(printed), f"{what}: the source prints {printed}, found {found}"


def published(group: str) -> list[str]:
    """Name each example of the catalogue that prints results of this kind."""
    return [name for name, entry in EXAMPLES.items() if group in entry["published"]]


# --------------------------------------------------------------------------- where each example came from
def test_every_example_cites_its_source_and_says_where_in_it_its_runs_are():
    sources = catalogue()["sources"]
    for key, source in sources.items():
        assert source["authors"], key
        assert all(source[field] for field in ("title", "work", "year", "publisher", "pages", "cite", "read")), key
        assert source["url"] == f"https://doi.org/{source['doi']}", "the address is the DOI's"
    for name, entry in {**EXAMPLES, **LIBRARY_EXAMPLES}.items():
        assert entry["source"] in sources, f"{name}: no citation for {entry['source']!r}"
        assert re.search(r"\bpp?\. \d", entry["where"]), f"{name}: where says no page"
        assert entry["published"], f"{name}: the results the source prints are part of the example"
        for group, results in entry["published"].items():
            assert re.search(r"\bpp?\. \d", results["where"]), f"{name}, {group}: where says no page"


def test_every_name_opens_one_example_whether_from_the_book_or_another_source():
    book = json.loads((WEB / "book-examples.json").read_text(encoding="utf-8"))["examples"]
    names = [entry["name"] for entry in book] + list(EXAMPLES) + list(LIBRARY_EXAMPLES)
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) for name in names)


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


def coefficient(analysis: dict, label: str) -> float:
    found = [c["coefficient"] for c in analysis["coefficients"] if c["label"] == label]
    assert found, f"The page shows no coefficient {label!r}."
    return found[0]


@pytest.mark.parametrize("name", published("coefficients"))
def test_every_coefficient_the_source_prints(analysis_of, name):
    for label, printed in EXAMPLES[name]["published"]["coefficients"]["values"].items():
        assert_printed(coefficient(analysis_of(name), label), printed, label)


@pytest.mark.parametrize("name", published("effects"))
def test_every_effect_the_source_prints_is_twice_the_page_s_coefficient(analysis_of, name):
    for label, printed in EXAMPLES[name]["published"]["effects"]["values"].items():
        assert_printed(2 * coefficient(analysis_of(name), label), printed, label)


@pytest.mark.parametrize("name", published("statistics"))
def test_every_statistic_the_source_prints(analysis_of, name):
    """R², Q² (the page's predicted R²), the ANOVA's sums of squares and mean squares, and the residual error."""
    analysis = analysis_of(name)
    summary, anova = analysis["summary"], analysis["anova"]
    residual = next(row for row in anova if row["source"] == "Residual")
    ss_regression = sum(row["sum_sq"] for row in anova if row["source"] != "Residual")
    found = {
        "r_squared": summary["r_squared"],
        "q_squared": summary["r_squared_pred"],
        "press": (1 - summary["r_squared_pred"]) * (ss_regression + residual["sum_sq"]),
        "ss_total": ss_regression + residual["sum_sq"],
        "ss_regression": ss_regression,
        "ss_residual": residual["sum_sq"],
        "ms_regression": ss_regression / summary["df_model"],
        "ms_residual": summary["mse_residual"],
        "rsd": summary["mse_residual"] ** 0.5,
        "df_model": summary["df_model"],
        "df_residual": summary["df_residual"],
    }
    for key, printed in EXAMPLES[name]["published"]["statistics"]["values"].items():
        assert_printed(found[key], printed, key)


@pytest.mark.parametrize("name", published("confounding"))
def test_the_fraction_confounds_what_the_source_says(design_of, name):
    printed, facts = EXAMPLES[name]["published"]["confounding"], design_of(name)["facts"]
    assert facts["generators"] == printed["generators"]
    assert dict(facts["aliases"]) == printed["aliases"]


ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def arithmetic(expression: str) -> float:
    """Work out a sum the source prints, such as ``(57 + 82) / 2 - (62 + 74) / 2``: numbers and + - * / only."""

    def value(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC:
            return ARITHMETIC[type(node.op)](value(node.left), value(node.right))
        raise ValueError(f"not arithmetic: {ast.dump(node)}")

    return value(ast.parse(expression, mode="eval").body)


@pytest.mark.parametrize("name", published("expressions"))
def test_each_effect_the_source_writes_as_a_sum_is_the_page_s(analysis_of, name):
    for label, expression in EXAMPLES[name]["published"]["expressions"]["values"].items():
        assert 2 * coefficient(analysis_of(name), label) == pytest.approx(arithmetic(expression)), label


# --------------------------------------------------------------------------- what the library gives besides
def library_analysis(entry: dict) -> tuple[dict, pd.Series]:
    """Fit an example's runs with the library, in coded units, with the example's model."""
    names = [f["name"] for f in entry["design"]["factors"]]
    runs = pd.DataFrame([run[:-1] for run in entry["runs"]], columns=names, dtype=float)
    y = pd.Series([run[-1] for run in entry["runs"]], name=entry["design"]["response"], dtype=float)
    model = entry["model"]
    if isinstance(model, dict):
        model = " + ".join(":".join(term) for term in model["terms"])
    return analyze_experiment(runs, y, model=model, analysis_type=["residual_diagnostics"]), y


@pytest.mark.parametrize("name", published("fitted"))
def test_the_fitted_values_and_residuals_the_source_prints(name):
    out, _ = library_analysis(EXAMPLES[name])
    diagnostics, results = out["residual_diagnostics"], EXAMPLES[name]["published"]
    for key, group in (("fitted_values", "fitted"), ("residuals", "residuals")):
        for i, (found, printed) in enumerate(zip(diagnostics[key], results[group]["values"], strict=True)):
            assert_printed(found, printed, f"{group}, run {i + 1}")


@pytest.mark.parametrize("name", published("left_out_predictions"))
def test_each_run_predicted_without_it_as_the_source_prints(name):
    """The prediction of run i from the model fitted without it: y_i less its residual over (1 - leverage)."""
    out, y = library_analysis(EXAMPLES[name])
    diagnostics = out["residual_diagnostics"]
    errors = np.asarray(diagnostics["residuals"]) / (1 - np.asarray(diagnostics["leverage"]))
    printed = EXAMPLES[name]["published"]["left_out_predictions"]["values"]
    for i, (found, value) in enumerate(zip(y.to_numpy() - errors, printed, strict=True)):
        assert_printed(found, value, f"run {i + 1}")
    statistics = EXAMPLES[name]["published"].get("statistics", {}).get("values", {})
    if "press" in statistics:
        assert_printed(float(np.sum(errors**2)), statistics["press"], "PRESS, from the runs left out one at a time")


@pytest.mark.parametrize("name", list(LIBRARY_EXAMPLES))
def test_the_examples_only_the_library_can_fit(name):
    """A categorical factor at three instances, coded to sum to zero as the source codes it.

    The source's c1 and c2 are the first two instances' differences from the mean, c0; the third's is
    minus their sum, c3. The page cannot open these: its categorical factors have two levels.
    """
    entry = LIBRARY_EXAMPLES[name]
    rows = pd.DataFrame(entry["rows"])
    (factor,) = (column for column in rows.columns if column != "y")
    out = analyze_experiment(
        rows[[factor]], rows["y"].astype(float), model=entry["model"], analysis_type="coefficients"
    )
    found = [c["coefficient"] for c in out["coefficients"]]
    found.append(-sum(found[1:]))
    printed = entry["published"]["coefficients"]["values"]
    assert len(found) == len(printed) == rows[factor].nunique() + 1
    for (label, value), coefficient_found in zip(printed.items(), found, strict=True):
        assert_printed(coefficient_found, value, label)
