"""The catalog's published studies with a categorical factor, analysed as Factorial analyses them.

Each study the page opens from the catalog is analysed by the page with each model its source prints numbers
for, where the page can fit that model, and each such number the page shows (an ANOVA row, a coefficient, a
sum of squares of the fit, a pair of levels compared) must be the library's for the same check, which
``reproduce_reference_catalogue`` holds to the printed number. A study of one categorical factor is also
compared with scipy: its one-way analysis of variance, Tukey's intervals, and Dunnett's critical value.

A model the page cannot fit is not compared: one without an intercept, a numeric factor taken as levels, a
square in an interaction, a cube, or an interaction of four factors or more. Neither is a number of a
transformed or summarised response, a one-sided test, or one at another confidence. A study in blocks is
analysed with its blocks, so only a model that names the block, as terms of its own, is compared there.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import subprocess
from collections import Counter
from typing import Any

import numpy as np
import pytest
from patsy import ModelDesc
from scipy import stats

from process_improve.experiments import design_catalog
from process_improve.experiments import reference_catalogue as rc
from tests.web._support import WEB, catalog_sample, no_bytecode

pytestmark = [pytest.mark.slow, pytest.mark.dataset]

#: The sources with a published study the page opens that holds a categorical factor; each is one test, so the
#: workers share them.
SOURCES = (
    "box-hunter-hunter-1978",
    "davies-1956",
    "efi-mooc-quizzes",
    "goos-jones-2011",
    "montgomery-2013",
    "nist-ehandbook-ch5",
    "pid-book",
    "schmidt-launsby-2000",
    "ullmann-doe-2012",
)

#: Studies that open, but whose default model, every factor's main effect, the runs cannot separate: the source
#: lists a factor the others determine, so the reader chooses the model, as the source does. The page refuses a
#: model it cannot estimate whole, where the source fits it by a generalised inverse.
UNFIT = {
    "bhh-wear-hyper-graeco-latin": "runs 1 to 4 use four of the papers and runs 5 to 8 the others: one df shared",
    "goos-jones-rolling-mill": "the three mills' fractions sum to one, a mixture taken as given",
    "montgomery-evop-yield": "each EVOP point is one setting of temperature and pressure",
}

#: Numbers the page reads off its fit, by the kind of check.
FIT = {
    "r_squared": "r_squared",
    "r_squared_adj": "r_squared_adj",
    "residual_df": "df_residual",
    "df_model": "df_model",
    "ms_residual": "mse_residual",
    "ss_residual": "ss_residual",
    "ss_regression": "ss_model",
    "ss_total": "ss_total",
    "f_regression": "f_statistic",
    "f_regression_p": "f_p_value",
}
ANOVA = {"anova_sum_sq": "sum_sq", "anova_mean_sq": "mean_sq", "anova_f": "F", "anova_p": "p_value", "anova_df": "df"}
COEFFICIENT = {"coefficient", "effect", "std_error", "t_value", "p_value", "ci_low", "ci_high"}
COMPARISON = {"tukey", "least_squares_means"}
#: Keys of a check that make its number one the page does not show.
NOT_SHOWN = ("summary", "transform", "settings", "squared", "complement", "exp", "log10", "back_transform", "at")
NOT_SHOWN += ("alternative", "scale", "categorical")


# --------------------------------------------------------------------------- a source's model, as the page's terms
def _variable(name: str) -> str:
    """Return the variable a patsy factor reads: ``C(diet, Sum)`` reads diet."""
    found = re.match(r"C\(\s*(\w+)", name)
    return found.group(1) if found else name


def page_terms(formula: str, spec: dict, bootstrap: Any, block: tuple[str, ...] = ()) -> list[list[str]] | None:  # noqa: ANN401
    """Return a model, as patsy reads it, as the page's chosen terms; None for one the page cannot fit.

    A study in blocks is fitted with its blocks, so the model must name the block's columns as terms of their own,
    which the page's blocks are; a model without them, or with a block column in an interaction, is not the page's.
    """
    factors = {f["name"]: f for f in spec["factors"]}
    named_block = False
    categorical = {name for name, f in factors.items() if bootstrap._categorical(f)}
    desc = ModelDesc.from_formula(formula.split("~", 1)[-1])
    if not any(not t.factors for t in desc.rhs_termlist) or all(not t.factors for t in desc.rhs_termlist):
        return None  # no intercept, or the intercept alone
    terms = []
    for term in (t for t in desc.rhs_termlist if t.factors):
        variables = [_variable(f.name()) for f in term.factors]
        if set(variables) & set(block):
            if len(variables) > 1:
                return None  # the page fits no interaction with its blocks
            named_block = True
            continue
        parts = []
        for factor in term.factors:
            name = factor.name()
            square = re.fullmatch(r"I\(\s*(\w+)\s*\*\*\s*2\s*\)", name)
            variable = square.group(1) if square else _variable(name)
            numeric_as_levels = name.startswith("C(") and variable not in categorical
            transformed = not square and not name.startswith("C(") and name != variable
            if variable not in factors or numeric_as_levels or transformed or (square and variable in categorical):
                return None
            parts += [variable, variable] if square else [variable]
        squared = len(parts) == 2 and parts[0] == parts[1]  # a factor named twice is its square
        if not squared and (len(set(parts)) != len(parts) or len(parts) > 3):  # three factors at most
            return None
        terms.append(parts)
    return terms if terms and (not block or named_block) else None


# --------------------------------------------------------------------------- reading the page's numbers
def _part(text: str) -> str:
    """Write one part of a term as a key: ``A²`` and ``I(A ** 2)`` as ``A^2``, ``C(diet, Sum)`` as ``diet``."""
    square = re.fullmatch(r"I\(\s*(\w+)\s*\*\*\s*2\s*\)", text) or re.fullmatch(r"(\w+)²", text)
    return f"{square.group(1)}^2" if square else _variable(text)


def _source_key(source: str) -> frozenset[str]:
    return frozenset(_part(p.strip()) for p in re.split(r"[:·]", source))


def _level_parts(term: str) -> list[tuple[str, str | None]]:
    """Split a source's term into (factor, level) parts: ``diet[A]:T`` is [(diet, A), (T, None)]."""
    out = []
    for part in re.split(r"[:·]", term):
        level = re.fullmatch(r"(?:C\()?(\w+)(?:,[^\[]*\))?\[(?:S\.)?(.+)\]", part.strip())
        out.append((level.group(1), level.group(2)) if level else (_part(part.strip()), None))
    return out


def _choices(term: str, spec: dict, bootstrap: Any) -> list[list[tuple[str, float]]] | None:  # noqa: ANN401
    """Return, for each part of a source's term, the page's label parts that make it and their signs.

    A level of a categorical factor is its effect, its distance from the average level: the page's coefficient of
    that level, a two-level factor's at -1 or +1, and the last level's minus the sum of the others'.
    """
    factors = {f["name"]: f for f in spec["factors"]}
    choices = []
    for name, level in _level_parts(term):
        factor = factors.get(name)
        if level is None:
            choices.append([(name, 1.0)])
            continue
        levels = bootstrap._levels_of(factor) if factor and bootstrap._categorical(factor) else []
        at = bootstrap._level_index(level, levels) if levels else None
        if at is None:
            return None
        if len(levels) == 2:  # coded -1 and +1
            choices.append([(name, 1.0 if at else -1.0)])
        elif at < len(levels) - 1:
            choices.append([(f"{name}[{levels[at]}]", 1.0)])
        else:
            choices.append([(f"{name}[{other}]", -1.0) for other in levels[:-1]])
    return choices


def page_coefficient(result: dict, spec: dict, term: str, field: str, bootstrap: Any) -> float | None:  # noqa: ANN401
    """Return the page's coefficient, or a field of its row, for a source's term; None when the page has none."""
    rows = {frozenset(_part(p) for p in c["label"].split("·")): c for c in result["coefficients"]}
    choices = _choices(term, spec, bootstrap)
    combinations = list(itertools.product(*choices)) if choices is not None else []
    found = [(rows.get(frozenset(p for p, _ in c)), math.prod(s for _, s in c)) for c in combinations]
    if not found or any(row is None for row, _ in found) or (len(found) > 1 and field != "coefficient"):
        return None  # no such row, or a sum of coefficients, which has no row of its own
    (row, sign), *_ = found
    if field in ("std_error", "p_value"):
        return row[field]
    if field in ("ci_low", "ci_high") and sign < 0:
        return -row["ci_high" if field == "ci_low" else "ci_low"]
    return sum(sign * row[field] for row, sign in found)


def _asked(check: dict) -> tuple[str, str, str, str, str] | None:
    """Read what a comparison check asks: its method, the two levels or the level, and the number."""
    if check["kind"] == "tukey":
        return "tukey", "pairs", str(check["level"]), str(check["minus"]), check["key"]
    head, _, rest = check["key"].partition(".")
    item, _, key = rest.rpartition(".")
    a, _, b = item.partition(" - ")
    return (check.get("comparisons", "tukey"), head, a, b, key) if head in ("pairs", "by_level") else None


def comparison_value(result: dict, check: dict) -> float | None:
    """Return a number of the page's comparison of one factor's levels: Tukey's pairs, Dunnett's, the averages."""
    found, asked = result.get("comparison") or {}, _asked(check)
    several = found.get("several")
    if asked is None:
        return None
    method, head, a, b, key = asked
    if head == "by_level":
        level = next((lv for lv in found.get("levels", []) if str(lv["level"]) == a), None)
        return level["mean"] if level and key == "mean" else None
    if not several or (method == "dunnett" and str(several["control"]) != b) or method not in ("tukey", "dunnett"):
        return None
    if method == "tukey":
        rows = [
            (p, 1.0 if str(p["to"]) == a else -1.0)
            for p in several["pairs"]
            if {str(p["from"]), str(p["to"])} == {a, b}
        ]
    else:
        rows = [(x, 1.0) for x in several["versus"] if str(x["level"]) == a]
    if not rows:
        return None
    (row, sign), *_ = rows
    difference, half = sign * row["estimate"], (row["ci_high"] - row["ci_low"]) / 2
    p = row.get("p_value")
    values = {"difference": difference, "ci_low": difference - half, "ci_high": difference + half, "half_width": half}
    return {**values, "p": p, "p_value": p}.get(key)


def _numbers(factor: dict) -> bool:
    return all(isinstance(factor.get(k), (int, float)) and not isinstance(factor.get(k), bool) for k in ("low", "high"))


# --------------------------------------------------------------------------- one source's studies
def _campaign(build_script: Any, record: dict) -> dict | None:  # noqa: ANN401 - the build script's module
    with no_bytecode():
        campaign, _, _ = build_script.published_campaign(record)
        return campaign or build_script.given_campaign(record)[0]


def _responses(entry: dict, record: dict, check: dict, n: int) -> list | None:
    """Return the results a check reads, one for each run of the study, None for a run it leaves out."""
    names = [r["name"] for r in entry["responses"] if r["name"] in entry["runs"]["columns"]]
    values = record["response_values"].get(check.get("response") or (names[0] if names else ""))
    if values is None or len(values) != n:
        return None
    if positions := check.get("runs"):
        kept = {p - 1 for p in positions}
        values = [v if i in kept else None for i, v in enumerate(values)]
    return values if any(v is not None for v in values) else None


def _page_value(  # noqa: PLR0913
    result: dict,
    spec: dict,
    check: dict,
    n: int,
    bootstrap: Any,  # noqa: ANN401
    block: tuple[str, ...],
    flipped: frozenset[str] = frozenset(),
) -> float | None:
    """Read the page's number for a check; a factor the source codes high to low (``flipped``) has its sign turned."""
    kind = check["kind"]
    if kind in ANOVA:
        if check["term"] == "Total" and kind == "anova_df":
            return float(n - 1)
        wanted = _source_key(check["term"])
        if wanted <= set(block):  # the page's row of its blocks, when they are one column
            wanted = frozenset({"Block"}) if len(block) == 1 else frozenset()
        row = next((r for r in result["anova"] if _source_key(str(r["source"])) == wanted), None)
        return None if row is None else row[ANOVA[kind]]
    if kind in FIT:
        return result["summary"].get(FIT[kind])
    if kind in COEFFICIENT:
        sign = (-1) ** sum(name in flipped for name, _ in _level_parts(check["term"]))
        field = "coefficient" if kind == "effect" else kind
        if sign < 0 and field in ("ci_low", "ci_high"):  # turned over, each end of the interval is the other's
            field = "ci_high" if field == "ci_low" else "ci_low"
        value = page_coefficient(result, spec, check["term"], field, bootstrap)
        if value is not None and field not in ("std_error", "p_value"):
            value *= sign
        return 2 * value if value is not None and kind == "effect" else value
    return comparison_value(result, check)


def _shown(check: dict, entry: dict, spec: dict) -> bool:
    """Say whether a check's number is one the page shows for this study."""
    kind = check.get("kind")
    if kind not in ANOVA and kind not in FIT and kind not in COEFFICIENT and kind not in COMPARISON:
        return False
    if any(check.get(key) for key in NOT_SHOWN) or check.get("confidence", 0.95) != 0.95:
        return False
    own_units = any(f.get("coding") in ("none", "pseudocomponent") for f in entry["factors"]) or entry["runs"].get(
        "exact"
    )
    if kind in COEFFICIENT and own_units:
        return False  # the source fits settings in their own units; the page, coded
    if kind in COMPARISON:  # the page compares the levels of a study of one factor, on no other model
        alone = spec["factors"][0]["name"] if len(spec["factors"]) == 1 else None
        model = check["factor"] if kind == "tukey" and not check.get("block") else check.get("model")
        return alone is not None and model in (alone, f"C({alone})")
    return bool(check.get("model"))


def _checked_against_scipy(result: dict, name: str) -> list[str]:
    """Compare a one-factor study's comparison with scipy: its analysis of variance, Tukey's and Dunnett's."""
    found = result.get("comparison") or {}
    several = found.get("several")
    if not several:
        return []
    groups = {str(lv["level"]): [r["value"] for r in lv["runs"]] for lv in found["levels"]}
    names, values = list(groups), list(groups.values())
    wrong = []
    anova = stats.f_oneway(*values)
    if not math.isclose(several["test"]["f_value"], anova.statistic, rel_tol=1e-9):
        wrong.append(f"{name}: F {several['test']['f_value']} is not scipy's {anova.statistic}")
    tukey = stats.tukey_hsd(*values)
    interval = tukey.confidence_interval(0.95)
    for pair in several["pairs"]:
        i, j = names.index(str(pair["from"])), names.index(str(pair["to"]))
        same = [pair["ci_low"], pair["ci_high"]] == pytest.approx([interval.low[j, i], interval.high[j, i]], rel=1e-6)
        if not same:
            wrong.append(f"{name}: Tukey's interval of {pair['to']} less {pair['from']} is not scipy's")
    # Dunnett's critical value holds every comparison with the control within it 95% of the time: the
    # multivariate t of the comparisons' correlations, which share the control's average, says how often.
    control = names.index(str(several["control"]))
    runs = np.array([len(v) for k, v in enumerate(values) if k != control])
    share = np.sqrt(runs / (runs + len(values[control])))
    shape = np.outer(share, share)
    np.fill_diagonal(shape, 1.0)
    d = several["d"]
    joint = stats.multivariate_t(shape=shape, df=several["test"]["df_within"], seed=1)
    covered = joint.cdf(np.full(len(runs), d), lower_limit=np.full(len(runs), -d), maxpts=200_000 * len(runs))
    if not math.isclose(covered, 0.95, abs_tol=1e-3):
        wrong.append(f"{name}: Dunnett's {d} holds the comparisons {covered:.4f} of the time, not 0.95")
    return wrong


def _printed(api: Any, bootstrap: Any, entry: dict, record: dict, design: dict) -> list[str]:  # noqa: ANN401
    """Compare the page's number for each check of the source's it shows with the library's, saying which differ."""
    spec, held = design["spec"], {key: design[key] for key in ("spec", "columns", "rows")}
    block = tuple(design_catalog._block_columns(entry) or ()) if record.get("blocks") else ()
    # A range written high first, as the source codes it; the page codes every factor low to high.
    flipped = frozenset(
        f["name"] for f in record["factors"] if f.get("type") == "numeric" and _numbers(f) and f["low"] > f["high"]
    )
    fits, replies, wrong = rc._Fits(), {}, []
    for expected in entry["expected"]:
        for check in (c for c in rc._checks(expected) if _shown(c, entry, spec)):
            model = check["factor"] if check["kind"] == "tukey" else check["model"]
            terms = page_terms(model, spec, bootstrap, block)
            values = _responses(entry, record, check, len(design["rows"]))
            if terms is None or values is None:
                continue
            request = {**held, "responses": values, "model": {"terms": terms}}
            if check.get("control") is not None:
                request["control"] = str(check["control"])
            key = json.dumps([terms, values, request.get("control")])
            reply = replies.setdefault(key, api.reply("api_analyze_runs", request))
            if not reply["ok"]:
                wrong += (
                    [] if record["id"] in UNFIT else [f"{record['id']}: the page refuses {model}: {reply['error']}"]
                )
                continue
            n = sum(v is not None for v in values)
            page = _page_value(reply["result"], spec, check, n, bootstrap, block, flipped)
            if page is None:
                continue
            library = float(rc._reproduce(entry, expected, check, fits)[0])
            if not math.isclose(page, library, rel_tol=1e-4, abs_tol=1e-6 * max(1, abs(library))):
                term = check.get("term") or check.get("key")
                wrong.append(f"{record['id']}: {check['kind']} {term} under {model} is {page:.6g}, not {library:.6g}")
    return wrong


def _study(api: Any, build_script: Any, record: dict, campaign: dict) -> list[str]:  # noqa: ANN401
    """Check a study as it opens: the source's runs and results, and its default analysis.

    Each run is compared with its result, so a result at the wrong settings fails; a study of one factor also
    has its comparison checked against scipy's.
    """
    name, design = record["id"], campaign["design"]
    spec, held = design["spec"], {key: design[key] for key in ("spec", "columns", "rows")}
    at = {c: i for i, c in enumerate(design["columns"])}
    made = Counter(
        (*(round(build_script.coded(r[at[f["name"]]], f), 4) for f in spec["factors"]), result)
        for r, result in zip(design["rows"], campaign["responses"], strict=True)
    )
    results = record["response_values"].get(build_script._response_of(record)) or [None] * len(record["coded_runs"])
    printed = Counter(
        (*(round(v, 4) for v in run), result)
        for run, result in zip(build_script.page_coded_runs(record), results, strict=True)
    )
    wrong = [] if made == printed else [f"{name}: the run sheet's runs, with their results, are not the source's"]
    if not any(v is not None for v in campaign["responses"]):
        return wrong
    default = api.reply("api_analyze_runs", {**held, "responses": campaign["responses"], "model": design["model"]})
    if not default["ok"]:
        return wrong + ([] if name in UNFIT else [f"{name}: {default['error']}"])
    wrong += [
        f"{name}: the mean square of {row['source']} is not its sum of squares over its degrees of freedom"
        for row in default["result"]["anova"]
        if row["source"] != "Residual" and row["df"] and not math.isclose(row["mean_sq"], row["sum_sq"] / row["df"])
    ]
    return wrong + (_checked_against_scipy(default["result"], name) if len(spec["factors"]) == 1 else [])


def _not_opened(node: str, campaigns: list[dict]) -> list[str]:
    """Open each study as campaign.js does, saying which it refuses or reads at another format."""
    driver = (
        f'import {{ checkCampaign }} from "{(WEB / "campaign.js").as_uri()}";'
        'let s = ""; process.stdin.on("data", (c) => (s += c)); process.stdin.on("end", () => {'
        "  const v = (c) => { try { return checkCampaign(c).v; } catch (e) { return e.message; } };"
        "  process.stdout.write(JSON.stringify(JSON.parse(s).map(v))); });"
    )
    done = subprocess.run(  # noqa: S603 - Node, found on PATH, runs the page's own code
        [node, "--input-type=module", "-e", driver],
        input=json.dumps(campaigns),
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        f"a study does not open: {v}" for v, c in zip(json.loads(done.stdout), campaigns, strict=True) if v != c["v"]
    ]


#: The sources with this many studies are tested in this many parts, so no one test holds up a worker.
PARTS = {"montgomery-2013": 3}


#: Each test: a source, and which part of its studies, every PARTS-th from the part's own.
CASES = [(source, part) for source in SOURCES for part in range(PARTS.get(source, 1))]


@pytest.mark.parametrize("case", CASES, ids=[f"{source}-{part}" for source, part in CASES])
def test_the_page_s_analysis_of_each_study_is_the_library_s_and_scipy_s(api, bootstrap, build_script, node, case):
    source, part = case
    entries = {e["name"]: e for e in rc.reference_experiments(source)}
    names = tuple(n for n, e in entries.items() if any(f.get("type") == "categorical" for f in e["factors"]))
    names = names[part :: PARTS.get(source, 1)]
    wrong: list[str] = []
    opened = []
    for record in catalog_sample(names):
        campaign = _campaign(build_script, record)
        if campaign is None:
            continue
        opened.append(campaign)
        wrong += _study(api, build_script, record, campaign)
        if any(v is not None for v in campaign["responses"]):
            wrong += _printed(api, bootstrap, entries[record["id"]], record, campaign["design"])
    wrong += _not_opened(node, opened)
    assert opened, "the source's studies open"
    assert not wrong, "\n".join(wrong)
