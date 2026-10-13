"""The reference designed experiments, reproduced with this library.

``process_improve.experiments.datasets.reference_experiments`` returns worked examples, exercises and
case studies of published sources, each with its runs as printed and every number the source prints
for it, with where it is printed. Here every such number is recomputed: the runs are fitted with the
source's model and the result compared with the printed text, to within half a unit in the last digit
printed. A number this library cannot recompute carries ``"check": "manual"`` and the reason, and is
counted rather than skipped silently. A change to the library that contradicts a published source
fails here, and so does a number copied wrongly into the catalogue.
"""

from __future__ import annotations

import ast
import itertools
import math
import operator
import re
import warnings
from decimal import Decimal, InvalidOperation
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, analyze_experiment, generate_design
from process_improve.experiments.datasets import reference_experiments, reference_sources

EXPERIMENTS = reference_experiments()
SOURCES = reference_sources()
ANALYSES = ["coefficients", "anova", "residual_diagnostics", "lack_of_fit", "effects", "lenth_method", "curvature_test"]


# --------------------------------------------------------------------------- reading what is printed
def tolerance(printed: str) -> float:
    """Half a unit in the last digit printed: the most the source's rounding accounts for."""
    exponent = Decimal(printed).normalize().as_tuple().exponent
    return 0.5 * 10.0 ** min(0, int(exponent)) + 1e-9


ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def arithmetic(expression: str) -> float:
    """Work out a sum the source prints, such as ``(57 + 82) / 2 - (62 + 74) / 2``: numbers and + - * / only."""

    def value(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -value(node.operand) if isinstance(node.op, ast.USub) else value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC:
            return ARITHMETIC[type(node.op)](value(node.left), value(node.right))
        raise ValueError(f"not arithmetic: {ast.dump(node)}")

    return value(ast.parse(expression, mode="eval").body)


def number(printed: str) -> str:
    """Return the number in printed text, without its bound sign, thousands separators or percent sign."""
    text = printed.replace("\u2212", "-").replace("\u2009", "").replace(",", "").strip()
    match = re.fullmatch(r"([<>≤≥]=?)?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*%?", text)
    if not match:
        raise ValueError(f"not a number: {printed!r}")
    return match.group(2)


def agrees(found: float, expected: dict) -> tuple[bool, str]:
    """Whether a recomputed value agrees with the printed one, and the message to show when it does not."""
    printed = expected["value"]
    if expected.get("value_kind") == "expression":
        target = arithmetic(printed)
        return math.isclose(found, target, rel_tol=1e-9, abs_tol=1e-9), f"{printed} = {target}, found {found}"
    text = printed.replace("\u2212", "-").strip()
    bound = re.match(r"^([<>]=?|[≤≥])\s*", text)
    value = float(number(printed))
    if bound:
        ok = found < value if bound.group(1) in ("<", "<=", "≤") else found > value
        return ok, f"printed {printed}, found {found}"
    return abs(found - value) <= tolerance(number(printed)), f"printed {printed}, found {found}"


# --------------------------------------------------------------------------- the runs as the model sees them
def coded_runs(experiment: dict, positions: list[int] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the factor settings in coded units and the responses, for the runs at these 1-based positions."""
    runs = experiment["runs"]
    table = pd.DataFrame(runs["rows"], columns=runs["columns"])
    if positions is not None:
        table = table.iloc[[p - 1 for p in positions]].reset_index(drop=True)
    settings = {}
    for f in experiment["factors"]:
        column = table[f["name"]]
        if f.get("levels") and len(f["levels"]) > 2:
            settings[f["name"]] = column.astype(str)  # a factor at three or more levels is fitted by name
        elif runs["coded"]:
            settings[f["name"]] = column.astype(float)
        elif f["type"] == "categorical":
            settings[f["name"]] = column.map({f["low"]: -1.0, f["high"]: 1.0})
        else:
            centre, half = (f["high"] + f["low"]) / 2, (f["high"] - f["low"]) / 2
            settings[f["name"]] = (column.astype(float) - centre) / half
    responses = table[[r["name"] for r in experiment["responses"] if r["name"] in table.columns]].astype(float)
    return pd.DataFrame(settings), responses


@cache
def fitted(name: str, index: int) -> dict[str, Any]:
    """Fit one analysis of an experiment with the library, once, with everything the checks read."""
    experiment = next(e for e in EXPERIMENTS if e["name"] == name)
    analysis = experiment["analyses"][index]
    settings, responses = coded_runs(experiment, analysis.get("runs"))
    y = responses[analysis.get("response") or responses.columns[0]]
    out: dict[str, Any] = {"y": y, "settings": settings}
    if analysis["model"] is None:
        return out
    terms = len(analysis["model"].replace("*", "+").split("+"))
    wanted = [a for a in ANALYSES if not (a in ("lenth_method", "effects") and terms > len(y))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out |= analyze_experiment(settings, y, model=analysis["model"], analysis_type=wanted)
        except (ValueError, np.linalg.LinAlgError):
            wanted = ["coefficients", "anova", "residual_diagnostics"]
            out |= analyze_experiment(settings, y, model=analysis["model"], analysis_type=wanted)
    return out


def library_term(term: str) -> str:
    """Return the library's name for a term the catalogue writes as A, A·B, A² or Intercept."""
    if term == "Intercept":
        return term
    squared = re.fullmatch(r"(\w+)²", term)
    if squared:
        return f"I({squared.group(1)} ** 2)"
    return ":".join(term.split("·"))


def coefficient_row(out: dict, term: str) -> dict:
    """Return the library's row for a term, a level of a factor coded to sum to zero included."""
    rows = {row["term"]: row for row in out["coefficients"]}
    level = re.fullmatch(r"(\w+)\[(.+)\]", term)
    if level:
        factor, name = level.groups()
        coded = {k: v for k, v in rows.items() if k.startswith(f"C({factor}, Sum)[")}
        key = next((k for k in coded if k.endswith(f".{name}]")), None)
        if key is not None:
            return coded[key]
        return {"coefficient": -sum(row["coefficient"] for row in coded.values())}  # the level the coding leaves out
    wanted = library_term(term)
    for key, row in rows.items():
        if key == wanted or set(key.split(":")) == set(wanted.split(":")):
            return row
    raise KeyError(f"the fit has no term {term!r}: {sorted(rows)}")


def run_index(term: str) -> int:
    return int(re.fullmatch(r"run (\d+)", term).group(1)) - 1


def residual_table(out: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diagnostics = out["residual_diagnostics"]
    return tuple(np.asarray(diagnostics[k], dtype=float) for k in ("fitted_values", "residuals", "leverage"))


def left_out(out: dict, i: int) -> float:
    """Return run i as the model fitted without it predicts it: the result less its residual over 1 - leverage."""
    _, residuals, leverage = residual_table(out)
    return out["y"].to_numpy()[i] - residuals[i] / (1 - leverage[i])


def anova_sums(out: dict) -> tuple[float, float, int]:
    """Return the sum of squares the model explains, the residual's, and the model's degrees of freedom."""
    table = out["anova_table"]
    residual = next(row for row in table if row["source"] == "Residual")
    model = [row for row in table if row["source"] != "Residual"]
    return sum(row["sum_sq"] for row in model), residual["sum_sq"], sum(row["df"] for row in model)


def lenth_effect(out: dict, term: str) -> dict:
    wanted = library_term(term)
    return next(e for e in out["lenth_method"]["effects"] if set(e["term"].split(":")) == set(wanted.split(":")))


CHECKS = {
    "coefficient": lambda out, term: coefficient_row(out, term)["coefficient"],
    "effect": lambda out, term: 2 * coefficient_row(out, term)["coefficient"],
    "std_error": lambda out, term: coefficient_row(out, term)["std_error"],
    "effect_std_error": lambda out, term: 2 * coefficient_row(out, term)["std_error"],
    "t_value": lambda out, term: coefficient_row(out, term)["t_value"],
    "p_value": lambda out, term: coefficient_row(out, term)["p_value"],
    "ci_low": lambda out, term: coefficient_row(out, term)["ci_low"],
    "ci_high": lambda out, term: coefficient_row(out, term)["ci_high"],
    "fitted": lambda out, term: residual_table(out)[0][run_index(term)],
    "residual": lambda out, term: residual_table(out)[1][run_index(term)],
    "left_out_prediction": lambda out, term: left_out(out, run_index(term)),
    "mean": lambda out, _term: float(out["y"].mean()),
    "std_dev": lambda out, _term: float(out["y"].std(ddof=1)),
    "r_squared": lambda out, _term: out["model_summary"]["r_squared"],
    "r_squared_adj": lambda out, _term: out["model_summary"]["r_squared_adj"],
    "r_squared_pred": lambda out, _term: out["model_summary"]["r_squared_pred"],
    "residual_df": lambda out, _term: out["model_summary"]["df_residual"],
    "df_model": lambda out, _term: out["model_summary"]["df_model"],
    "ms_residual": lambda out, _term: out["model_summary"]["mse_residual"],
    "residual_std_error": lambda out, _term: out["model_summary"]["mse_residual"] ** 0.5,
    "ss_residual": lambda out, _term: anova_sums(out)[1],
    "ss_regression": lambda out, _term: anova_sums(out)[0],
    "ss_total": lambda out, _term: sum(anova_sums(out)[:2]),
    "ms_regression": lambda out, _term: anova_sums(out)[0] / anova_sums(out)[2],
    "press": lambda out, _term: float(np.sum((residual_table(out)[1] / (1 - residual_table(out)[2])) ** 2)),
    "lenth_pse": lambda out, _term: out["lenth_method"]["PSE"],
    "lenth_me": lambda out, _term: out["lenth_method"]["ME"],
    "lenth_sme": lambda out, _term: out["lenth_method"]["SME"],
    "anova_sum_sq": lambda out, term: next(r for r in out["anova_table"] if r["source"] == library_term(term))[
        "sum_sq"
    ],
    "anova_f": lambda out, term: next(r for r in out["anova_table"] if r["source"] == library_term(term))["F"],
    "lack_of_fit_f": lambda out, _term: out["lack_of_fit"]["f_statistic"],
    "pure_error_df": lambda out, _term: out["lack_of_fit"]["df_pure_error"],
    "curvature_difference": lambda out, _term: out["curvature_test"]["difference"],
}


# --------------------------------------------------------------------------- the design a source names
def multiply(*words: str) -> str:
    """Multiply effect words, letter by letter: a letter twice cancels, as A·A = I."""
    letters = sorted(itertools.chain.from_iterable(words))
    return "".join(letter for letter in sorted(set(letters)) if letters.count(letter) % 2)


@cache
def fraction(k: int, resolution: int | None, generators: tuple[str, ...] | None) -> dict:
    """Return the library's fraction in k factors, by resolution or by the source's generators."""
    factors = [Factor(name="ABCDEFGHJKLMNOPQ"[i], low=-1, high=1) for i in range(k)]
    made = generate_design(
        factors,
        design_type="fractional_factorial",
        resolution=resolution,
        generators=list(generators) if generators else None,
        n_center_points=0,
    )
    return {"generators": made.generators, "defining_relation": made.defining_relation, "resolution": made.resolution}


ROMAN = {"III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}


def plain(word: str) -> str:
    return re.sub(r"[\s·\u00d7*]", "", word)


def design_check(what: str, expected: dict, k: int, resolution: int | None) -> tuple[bool, str]:
    """Check a generator, resolution or alias the source states, against the fraction the library makes."""
    made = fraction(k, resolution, None)
    if what == "generator":
        return plain(expected["value"]) in {plain(g) for g in made["generators"]}, f"library: {made['generators']}"
    if what == "resolution":
        return ROMAN[expected["value"]] == made["resolution"], f"library: {made['resolution']}"
    if what == "aliases":
        words = [plain(w).removeprefix("I=") for w in made["defining_relation"]]
        aliases = {multiply(plain(expected["term"]), word) for word in words}
        return plain(expected["value"]) in aliases, f"library aliases of {expected['term']}: {sorted(aliases)}"
    raise KeyError(what)


DESIGN_KINDS = {"generator", "resolution", "aliases"}


# --------------------------------------------------------------------------- every printed number, as a case
def cases() -> list[tuple[str, int, int]]:
    return [
        (e["name"], a, i)
        for e in EXPERIMENTS
        if e.get("runs")
        for a, analysis in enumerate(e["analyses"])
        for i, expected in enumerate(analysis["expected"])
        if expected.get("check") != "manual"
    ]


def case_id(case: tuple[str, int, int]) -> str:
    name, a, i = case
    expected = next(e for e in EXPERIMENTS if e["name"] == name)["analyses"][a]["expected"][i]
    return f"{name}:{a}:{expected['what']}:{expected.get('term')}"


@pytest.mark.parametrize("case", cases(), ids=case_id)
def test_every_printed_number_is_reproduced(case):
    name, a, i = case
    experiment = next(e for e in EXPERIMENTS if e["name"] == name)
    analysis = experiment["analyses"][a]
    expected = analysis["expected"][i]
    what = expected["what"]
    if what in DESIGN_KINDS:
        k = len(experiment["factors"])
        stated = next((e["value"] for e in analysis["expected"] if e["what"] == "resolution"), None)
        ok, said = design_check(what, expected, k, ROMAN.get(stated) if stated else None)
    else:
        assert what in CHECKS, f"no check for {what!r}: give it one, or mark it manual with the reason"
        ok, said = agrees(CHECKS[what](fitted(name, a), expected.get("term")), expected)
    where = expected["where"]
    assert ok, f"{name}, {analysis['label']}, {what} {expected.get('term')}: {said} (printed at {where})"


def design_cases() -> list[tuple[str, int]]:
    from process_improve.experiments.datasets import _reference_files

    return [
        (d["name"], i)
        for f in _reference_files()
        for d in f.get("designs", [])
        for i, expected in enumerate(d["expected"])
        if expected.get("check") != "manual"
    ]


@pytest.mark.parametrize("case", design_cases(), ids=lambda c: f"{c[0]}:{c[1]}")
def test_every_design_property_stated_is_the_library_s(case):
    from process_improve.experiments.datasets import _reference_files

    name, i = case
    design = next(d for f in _reference_files() for d in f.get("designs", []) if d["name"] == name)
    expected = design["expected"][i]
    k = design["design"]["factors"]
    if expected["what"] in DESIGN_KINDS:
        stated = next((e["value"] for e in design["expected"] if e["what"] == "resolution"), None)
        ok, said = design_check(expected["what"], expected, k, ROMAN.get(stated) if stated else None)
    elif expected["what"] == "axial_distance":
        made = generate_design(
            [Factor(name=n, low=-1, high=1) for n in "ABCDEFGH"[:k]], design_type="ccd", alpha="rotatable"
        )
        ok, said = agrees(made.alpha, expected)
    elif expected["what"] == "n_runs":
        made = generate_design(
            [Factor(name=n, low=-1, high=1) for n in "ABCDEFGH"[:k]],
            design_type="ccd",
            alpha="rotatable",
            n_center_points=1,
        )
        ok, said = agrees(made.n_runs, expected)
    else:
        pytest.fail(f"no check for a design's {expected['what']!r}")
    assert ok, f"{name}, {expected['what']}: {said} (printed at {expected['where']})"


# --------------------------------------------------------------------------- where each number came from
def test_every_experiment_says_where_it_was_printed_and_what_it_printed():
    for experiment in EXPERIMENTS:
        name = experiment["name"]
        assert experiment["source"] in SOURCES, name
        assert experiment["where"], name
        assert experiment.get("runs") or experiment.get("data_source"), f"{name}: no runs and no data source"
        assert experiment["analyses"], f"{name}: the results the source prints are part of the experiment"
        for analysis in experiment["analyses"]:
            for expected in analysis["expected"]:
                where = expected["where"]
                assert where.get("quote"), f"{name}: every number carries a quotation of where it is printed"
                assert where.get("line") or where.get("page"), f"{name}: every number carries its line or page"
                if expected.get("check") == "manual":
                    assert expected.get("why"), f"{name}: a number not recomputed says why"


def test_each_name_is_one_experiment():
    names = [e["name"] for e in EXPERIMENTS]
    assert len(set(names)) == len(names)


def test_the_printed_values_are_numbers_where_they_are_checked():
    """A value with a check is a number, a bound (< 0.001), or a sum the source writes out."""
    for name, a, i in cases():
        expected = next(e for e in EXPERIMENTS if e["name"] == name)["analyses"][a]["expected"][i]
        if expected["what"] in DESIGN_KINDS or expected.get("value_kind") == "expression":
            continue
        try:
            Decimal(number(expected["value"]))
        except (ValueError, InvalidOperation):
            pytest.fail(f"{name}: {expected['what']} {expected.get('term')} has no number: {expected['value']!r}")


def test_the_sources_are_cited_in_full():
    for key, source in SOURCES.items():
        assert source["authors"], key
        assert all(source.get(field) for field in ("title", "url", "year", "read")), key
