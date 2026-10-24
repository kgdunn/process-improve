"""Every number a catalog study's source prints that Factorial shows too, as Learn shows it beside the page's own.

Each study the page opens from the catalog is analysed by the page with each model its source prints numbers
for, as ``scripts/build_web_app.py`` lists them in ``catalog/printed/<id>.json`` (``printed_checks``), and
each number is read off the page's analysis by the page's own JavaScript, ``printedValue`` of ``web/logic.js``,
which Learn reads it with. The number must be the library's for the same check, which
``reproduce_reference_catalogue`` holds to the printed one, and the page's verdict, ``printedAgrees``, must say
that each number agrees that the catalogue does not record as printed wrong: one it does, Learn shows with the
catalogue's reason. A study of one categorical factor
is also compared with scipy: its one-way analysis of variance, Tukey's intervals, and Dunnett's critical value.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from typing import Any

import numpy as np
import pytest
from scipy import stats

from process_improve.experiments import reference_catalogue as rc
from tests.web._support import WEB, catalog_sample, javascript, no_bytecode, source

pytestmark = [pytest.mark.slow, pytest.mark.dataset]

#: Studies that open, but whose default model, every factor's main effect, the runs cannot separate: the source
#: lists a factor the others determine, so the reader chooses the model, as the source does. The page refuses a
#: model it cannot estimate whole, where the source fits it by a generalised inverse.
UNFIT = {
    "bhh-wear-hyper-graeco-latin": "runs 1 to 4 use four of the papers and runs 5 to 8 the others: one df shared",
    "goos-jones-rolling-mill": "the three mills' fractions sum to one, a mixture taken as given",
    "montgomery-evop-yield": "each EVOP point is one setting of temperature and pressure",
}

#: The kinds of check the page may show a number for; an entry with none of them is analysed for its categorical
#: factor alone, if it has one.
SHOWN_KINDS = frozenset(
    {"r_squared", "r_squared_adj", "residual_df", "df_model", "ms_residual", "ss_residual", "ss_regression"}
    | {"ss_total", "f_regression", "f_regression_p", "anova_sum_sq", "anova_mean_sq", "anova_f", "anova_p"}
    | {"anova_df", "coefficient", "effect", "std_error", "t_value", "p_value", "ci_low", "ci_high"}
    | {"tukey", "least_squares_means"}
)


def _campaign(build_script: Any, record: dict) -> dict | None:  # noqa: ANN401 - the build script's module
    with no_bytecode():
        campaign, _, _ = build_script.published_campaign(record)
        return campaign or build_script.given_campaign(record)[0]


def _printed(api: Any, build_script: Any, node: str, entry: dict, record: dict, campaign: dict) -> list[str]:  # noqa: ANN401, PLR0913
    """Read each printed number the page shows for a study as Learn reads it, and hold it to the library's."""
    design = campaign["design"]
    held = {key: design[key] for key in ("spec", "columns", "rows")}
    runs = sum(v is not None for v in campaign["responses"])
    fits, replies, wrong, read = rc._Fits(), {}, [], []
    with no_bytecode():
        checks = list(build_script.printed_checks(record, entry, design["spec"]))
    for expected, check, number in checks:
        key = json.dumps(number["model"])
        request = {**held, "responses": campaign["responses"], "model": {"terms": number["model"]}}
        reply = replies.setdefault(key, api.reply("api_analyze_runs", request))
        if not reply["ok"]:
            wrong += [] if record["id"] in UNFIT else [f"{record['id']}: the page refuses {key}: {reply['error']}"]
            continue
        read.append((expected, check, number, reply["result"]))
    if not read:
        return wrong
    calls = [("printedValue", [result, number["read"], runs]) for *_, number, result in read]
    values = javascript(node, source("logic.js"), calls)
    judged = [(v, n) for v, (*_, n, _) in zip(values, read, strict=True) if v is not None and not n["reason"]]
    verdicts = iter(javascript(node, source("logic.js"), [("printedAgrees", [v, n["agree"]]) for v, n in judged]))
    for value, (expected, check, number, _) in zip(values, read, strict=True):
        what = f"{record['id']}: {number['what']} {number['term'] or ''} under {number['model']}"
        if value is None:
            wrong.append(f"{what}: the page shows no such number")
            continue
        library = float(rc._reproduce(entry, expected, check, fits)[0])
        if not math.isclose(value, library, rel_tol=1e-4, abs_tol=1e-6 * max(1, abs(library))):
            wrong.append(f"{what} is {value:.6g} on the page, not the library's {library:.6g}")
        if not number["reason"] and not next(verdicts):  # a number printed wrong is shown with its reason
            wrong.append(f"{what}: the page says {value!r} differs from the printed {number['printed']}")
    return wrong


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
    floats = [np.asarray(v, dtype=float) for v in values]  # scipy's bartlett fails on integers
    for check in several["checks"]:
        expected = (stats.bartlett if check["test"] == "bartlett" else stats.kruskal)(*floats)
        if not math.isclose(check["statistic"], expected.statistic, rel_tol=1e-9):
            wrong.append(f"{name}: {check['test']} {check['statistic']} is not scipy's {expected.statistic}")
    spread = all(len(set(v)) > 1 for v in values)
    if spread != any(check["test"] == "bartlett" for check in several["checks"]):
        wrong.append(f"{name}: Bartlett's test is {'missing' if spread else 'given'}")
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


#: The sources whose studies are tested in parts, so no one test holds up a worker: so many parts each.
PARTS = {"montgomery-2013": 4, "nist-ehandbook-ch5": 2, "box-hunter-hunter-1978": 2, "schmidt-launsby-2000": 2}

#: Each test: a source, and which part of its studies, every PARTS-th from the part's own.
CASES = [(key, part) for key in sorted(rc.reference_sources()) for part in range(PARTS.get(key, 1))]


def _tested(entry: dict) -> bool:
    """Say whether a study is analysed here: it prints a number the page may show, or has a categorical factor."""
    kinds = {c.get("kind") for expected in entry.get("expected", []) for c in rc._checks(expected)}
    categorical = any(f.get("type") == "categorical" for f in entry["factors"] or [])
    return bool(entry.get("responses")) and (bool(kinds & SHOWN_KINDS) or categorical)


@pytest.mark.parametrize("case", CASES, ids=[f"{key}-{part}" for key, part in CASES])
def test_the_page_reads_each_printed_number_as_the_catalogue_reproduces_it(api, build_script, node, case):
    key, part = case
    entries = {e["name"]: e for e in rc.reference_experiments(key)}
    names = tuple(n for n, e in entries.items() if _tested(e))[part :: PARTS.get(key, 1)]
    wrong: list[str] = []
    opened = []
    for record in catalog_sample(names) if names else []:
        campaign = _campaign(build_script, record)
        if campaign is None:
            continue
        opened.append(campaign)
        wrong += _study(api, build_script, record, campaign)
        if any(v is not None for v in campaign["responses"]) and record["id"] in entries:
            wrong += _printed(api, build_script, node, entries[record["id"]], record, campaign)
    wrong += _not_opened(node, opened) if opened else []
    assert not wrong, "\n".join(wrong)
