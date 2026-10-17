"""The reference catalogue: designed experiments from published sources, reproduced with this library.

The reference catalogue holds worked examples, exercises, case studies and designs of published
sources (a textbook, an encyclopedia article, a course's quizzes, tutorial slides, journal
papers), recorded so that software can be checked against them. Each entry holds its data
exactly as printed (factors, responses, runs) and every number the source prints for it, as the
text printed, with where it is printed. A number also carries a check, saying how to recompute
it (the kind of result, the term, the model, the runs it uses), or why it is not recomputed.

:func:`reproduce_reference_catalogue` recomputes every checked number with this library and
reports, row by row, the printed text, the value found, whether they agree to within the
source's own rounding, and where the number is printed.

Each entry also lists its subject-matter ``topics``: what the experiment is about (semiconductors,
food and beverages, ...), as opposed to how it is designed. :func:`reference_topics` returns the
vocabulary, each topic with its nearest classes in UN ISIC Rev. 4 (industries) and OECD FORD
(fields of research), and the ``topic`` argument of :func:`reference_experiments` selects by it.

The catalogue's files hold facts and short quotations from their sources, not the sources
themselves. The MIT licence of this package covers its code, not them: each source states its
own terms in ``rights`` (see :func:`reference_sources` and the folder's ``NOTICE.md``).

Examples
--------
>>> from process_improve.experiments.reference_catalogue import (
...     reference_experiments,
...     reproduce_reference_catalogue,
... )
>>> [e["name"] for e in reference_experiments("ullmann-doe-2012")][:2]
['ullmann-catalyst-temperature', 'ullmann-agitator']
>>> report = reproduce_reference_catalogue("ullmann-doe-2012")
>>> bool(report.loc[report["checked"], "agrees"].all())
True
"""

from __future__ import annotations

import ast
import functools
import itertools
import json
import math
import operator
import re
import statistics
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import f as f_distribution

from process_improve.experiments._analyses.box_cox import _scaled_power
from process_improve.experiments._analyses.transforms import transform_response
from process_improve.experiments._blocking import squares_orthogonal_to_blocks
from process_improve.experiments._desirability import composite_desirability, individual_desirability
from process_improve.experiments._orthogonal_arrays import orthogonal_array
from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.design_measures import relabelling, run_counts
from process_improve.experiments.designs import generate_design
from process_improve.experiments.designs_response_surface import blocking_alpha
from process_improve.experiments.evaluate import evaluate_design
from process_improve.experiments.factor import Factor
from process_improve.experiments.marginal_means import marginal_means_prediction
from process_improve.experiments.omars_trade_off import omars_minimum_runs
from process_improve.experiments.optimization import optimize_responses
from process_improve.experiments.random_effects import expected_mean_squares
from process_improve.experiments.random_effects import fixed_columns as _fixed_columns
from process_improve.monitoring.metrics import calculate_cpk
from process_improve.regression._variance_components import fit_variance_components
from process_improve.sensory.designed import tukey_hsd
from process_improve.univariate.metrics import t_value, t_value_cdf, ttest_independent, ttest_paired

_CATALOGUE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "experiments" / "reference_catalogue"
_TOPICS_FILE = "topics.json"  # the vocabulary of subject-matter topics, beside the sources but not one of them


# --------------------------------------------------------------------------- reading the reference catalogue
def _files() -> list[dict[str, Any]]:
    """Read every source file of the reference catalogue, in the order of their names."""
    paths = sorted(path for path in _CATALOGUE_DIR.glob("*.json") if path.name != _TOPICS_FILE)
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _selected(source: str | None) -> list[dict[str, Any]]:
    files = _files()
    keys = [f["source"]["key"] for f in files]
    if source is not None and source not in keys:
        raise ValueError(f"Unknown source {source!r}. The sources are: {', '.join(sorted(keys))}.")
    return [f for f in files if source in (None, f["source"]["key"])]


def _about(entries: list[dict[str, Any]], topic: str | None) -> list[dict[str, Any]]:
    """Keep the entries that list *topic*, as their main topic or their second; all of them when ``None``."""
    if topic is None:
        return entries
    topics = reference_topics()
    if topic not in topics:
        raise ValueError(f"Unknown topic {topic!r}. The topics are: {', '.join(topics)}.")
    return [e for e in entries if topic in e["topics"]]


def reference_topics() -> dict[str, dict[str, Any]]:
    """Return the subject-matter topics of the reference catalogue: what its experiments and designs are about.

    Every entry lists one or two of these topics in ``topics``, its main topic first. A topic says
    what the experiment is about, not how it is designed (``design.type`` says that); an entry
    whose source states no application, such as a table of designs, is ``"general"``. The topics
    are the catalogue's own classification, not the sources'. Each one names its nearest classes
    in two standards, so the catalogue rolls up to either: an industry in UN ISIC Rev. 4, and a
    field of research in OECD FORD (Frascati Manual 2015).

    Returns
    -------
    dict[str, dict]
        For each topic id, in the vocabulary's order: ``label``, ``description``, ``isic`` (ISIC
        Rev. 4 codes: a section letter, a two-digit division or a three-digit group; empty when no
        class fits), ``ford`` (a FORD field code such as ``"2.4"``, or ``None``) and ``ford_field``
        (that field's name, or ``None``).

    Examples
    --------
    >>> topic = reference_topics()["semiconductors"]
    >>> topic["label"], topic["isic"], topic["ford"]
    ('Semiconductors and electronic assembly', ['261'], '2.2')
    """
    vocabulary = json.loads((_CATALOGUE_DIR / _TOPICS_FILE).read_text(encoding="utf-8"))
    fields = vocabulary["standards"]["ford"]["fields"]
    return {
        t["id"]: {**{k: v for k, v in t.items() if k != "id"}, "ford_field": fields.get(t["ford"])}
        for t in vocabulary["topics"]
    }


def reference_sources() -> dict[str, dict[str, Any]]:
    """Return the citation of each source of the reference catalogue, by its key.

    Returns
    -------
    dict[str, dict]
        For each source key, its citation: ``authors``, ``title``, ``url``, ``year``, and where they
        apply ``work``, ``volume``, ``pages``, ``publisher``, ``doi``, the ``version`` and ``commit``
        that was read, ``read``: when and from what the numbers were taken, and ``rights``: the terms
        of the source's numbers and quotations, which the MIT licence of this package does not cover.

    Examples
    --------
    >>> reference_sources()["ullmann-doe-2012"]["doi"]
    '10.1002/14356007.e08_e01.pub3'
    """
    return {f["source"]["key"]: f["source"] for f in _files()}


def reference_experiments(source: str | None = None, topic: str | None = None) -> list[dict[str, Any]]:
    """Return the experiments of the reference catalogue: their data, where each was published, every printed number.

    Parameters
    ----------
    source : str, optional
        The key of one source (see :func:`reference_sources`); every source when ``None``.
    topic : str, optional
        The id of one subject-matter topic (see :func:`reference_topics`): only the experiments that
        list it, as their main topic or their second. Every topic when ``None``.

    Returns
    -------
    list[dict]
        One dict per experiment, with keys ``name``, ``title``, ``topics``, ``source``, ``kind``,
        ``where``, ``factors``, ``responses``, ``design``, ``runs`` (``None`` when the source does
        not print them, and ``data_source`` then says where they come from), ``expected`` and
        ``notes``. Each item of ``expected`` has ``what``, ``term`` and ``value`` (the printed
        text), ``where`` (a file and line, or a page, with a quotation) and ``check``.

    Raises
    ------
    ValueError
        If *source* names no source, or *topic* no topic.

    Examples
    --------
    >>> experiment = reference_experiments("ullmann-doe-2012")[0]
    >>> experiment["name"], experiment["expected"][0]["value"]
    ('ullmann-catalyst-temperature', '10')
    >>> [e["name"] for e in reference_experiments("montgomery-2013", topic="paper")]
    ['montgomery-problem-3-34', 'montgomery-problem-4-42', 'montgomery-problem-5-18']
    """
    experiments = [{**e, "source": f["source"]["key"]} for f in _selected(source) for e in f["experiments"]]
    return _about(experiments, topic)


def reference_designs(source: str | None = None, topic: str | None = None) -> list[dict[str, Any]]:
    """Return the designs of the reference catalogue: designs without responses whose properties a source states.

    Parameters
    ----------
    source : str, optional
        The key of one source; every source when ``None``.
    topic : str, optional
        The id of one subject-matter topic; every topic when ``None``.

    Returns
    -------
    list[dict]
        One dict per design, as for :func:`reference_experiments` but without responses.

    Raises
    ------
    ValueError
        If *source* names no source, or *topic* no topic.
    """
    designs = [{**d, "source": f["source"]["key"]} for f in _selected(source) for d in f.get("designs", [])]
    return _about(designs, topic)


# --------------------------------------------------------------------------- printed numbers
def _tolerance(printed: str) -> float:
    """Half a unit in the last digit printed: the most the source's rounding accounts for.

    The digits count as printed: 0.40 is rounded to hundredths, so it allows 0.005, not 0.05, and
    1.25E+08 to the nearest million, so it allows 500000. A little more is allowed for
    floating-point error, never more than a thousandth of the half unit, so a number printed as
    0.798e-8 is held to its own digits.
    """
    exponent = Decimal(printed).as_tuple().exponent
    half = 0.5 * 10.0 ** int(exponent)
    return half + min(1e-9, half * 1e-3)


_ARITHMETIC: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _arithmetic(expression: str) -> float:
    """Work out a sum the source prints, such as ``(57 + 82) / 2 - (62 + 74) / 2``: numbers and + - * / only."""

    def value(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -value(node.operand) if isinstance(node.op, ast.USub) else value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
            return _ARITHMETIC[type(node.op)](value(node.left), value(node.right))
        raise ValueError(f"Not arithmetic: {ast.dump(node)}")

    return value(ast.parse(expression, mode="eval").body)


_NUMBER = re.compile(r"([<>]=?|[\u2264\u2265])?\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*%?")


def _number(printed: str) -> tuple[str | None, str]:
    """Return a printed number's bound sign (``<`` or ``>``, if any) and its digits, as text."""
    text = str(printed).replace("\u2212", "-").replace("\u2009", "").replace(",", "").strip()
    match = _NUMBER.fullmatch(text)
    if not match:
        raise ValueError(f"Not a number: {printed!r}")
    return match.group(1), match.group(2)


def _agrees(
    found: float, printed: str, *, expression: bool = False, of_rounded: int = 1, decimals: int | None = None
) -> bool:
    """Whether a value found agrees with the printed text, to within the source's rounding.

    A number the source works out from ``of_rounded`` numbers it has itself rounded to the same
    digits, such as the difference of two averages it prints, carries their rounding too: it is
    allowed that many half units. Software that prints more digits than its arithmetic carries,
    as single precision does, is held to ``decimals`` places instead of all it prints.
    """
    if expression:
        return math.isclose(found, _arithmetic(printed), rel_tol=1e-9, abs_tol=1e-9)
    bound, digits = _number(printed)
    allowed = of_rounded * (_tolerance(digits) if decimals is None else 0.5 * 10.0**-decimals)
    if bound in ("<", "<=", "\u2264"):
        return found < float(digits) or abs(found - float(digits)) <= allowed
    if bound in (">", ">=", "\u2265"):
        return found > float(digits) or abs(found - float(digits)) <= allowed
    return abs(found - float(digits)) <= allowed


# --------------------------------------------------------------------------- the runs as a model sees them
_STATISTICS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "mean": lambda readings: readings.mean(axis=1),
    "total": lambda readings: readings.sum(axis=1),
    "std_dev": lambda readings: readings.std(axis=1, ddof=1),
    "variance": lambda readings: readings.var(axis=1, ddof=1),
    "range": lambda readings: readings.max(axis=1) - readings.min(axis=1),
    "ln_std_dev": lambda readings: np.log(readings.std(axis=1, ddof=1)),
    # variance-stabilizing transforms of a count c or a proportion p, read from one column, and Freeman
    # and Tukey's forms of them, [sqrt(c) + sqrt(c + 1)]/2 and, below, the angular one for n trials
    "sqrt": lambda readings: np.sqrt(readings.iloc[:, 0]),
    "arcsin_sqrt": lambda readings: np.arcsin(np.sqrt(readings.iloc[:, 0])),
    "freeman_tukey_sqrt": lambda readings: (np.sqrt(readings.iloc[:, 0]) + np.sqrt(readings.iloc[:, 0] + 1)) / 2,
    # Taguchi's signal-to-noise ratios, in decibels: larger is better, smaller is better, and
    # nominal is best in the form 10 log10(mean^2 / variance)
    "sn_larger": lambda readings: -10 * np.log10((1 / readings**2).mean(axis=1)),
    "sn_smaller": lambda readings: -10 * np.log10((readings**2).mean(axis=1)),
    "sn_nominal": lambda readings: pd.Series(
        10 * np.log10(readings.mean(axis=1) ** 2 / readings.var(axis=1, ddof=1)), index=readings.index
    ),
}


def _summarised(table: pd.DataFrame, summary: dict) -> pd.DataFrame:
    """Return each run's statistic of its repeated readings, as the one response a source then analyses.

    ``summary["of"]`` names the columns holding a run's readings and ``summary["statistic"]`` is
    ``"mean"``, ``"total"`` (the sum of the readings, as Yates's algorithm takes them),
    ``"std_dev"``, ``"variance"``, ``"ln_std_dev"`` (the natural logarithm of the standard
    deviation), or one of Taguchi's signal-to-noise ratios in decibels: ``"sn_larger"``
    (-10 log10 of the mean of 1/y^2), ``"sn_smaller"`` (-10 log10 of the mean of y^2) and
    ``"sn_nominal"`` (10 log10 of the squared mean over the variance). Of one column, a count c or a
    proportion p, ``"sqrt"`` and ``"arcsin_sqrt"`` are the variance-stabilizing transforms and
    ``"freeman_tukey_sqrt"`` ([sqrt(c) + sqrt(c + 1)]/2) and ``"freeman_tukey_arcsin"`` (the two arcsines
    of np/(n + 1) and (np + 1)/(n + 1), averaged, for ``summary["n"]`` trials) Freeman and Tukey's. A source
    that rounds each
    run's value before averaging it gives ``summary["round"]``, the decimals it keeps. A source
    that prints its readings in one unit and works the statistic in another gives
    ``summary["scale"]``, the factor the readings are multiplied by first (1e17 for readings
    printed in units of 10^17).
    """
    readings = table[summary["of"]].astype(float) * summary.get("scale", 1)
    if summary["statistic"] == "freeman_tukey_arcsin":  # [arcsin sqrt(np/(n + 1)) + arcsin sqrt((np + 1)/(n + 1))]/2
        n, p = float(summary["n"]), readings.iloc[:, 0]
        values = (np.arcsin(np.sqrt(n * p / (n + 1))) + np.arcsin(np.sqrt((n * p + 1) / (n + 1)))) / 2
    else:
        values = _STATISTICS[summary["statistic"]](readings)
    if summary.get("round") is not None:
        values = values.round(summary["round"])
    return values.rename(summary["statistic"]).to_frame()


def _each_reading(settings: pd.DataFrame, readings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every reading as a run of its own at its run's settings, run by run; a blank reading is left out."""
    values = readings.to_numpy(dtype=float).ravel()
    kept = ~np.isnan(values)
    rows = np.repeat(np.arange(len(readings)), readings.shape[1])[kept]
    return settings.iloc[rows].reset_index(drop=True), pd.DataFrame({"each": values[kept]})


def _exact_settings(column: pd.Series, exact: dict[float, float]) -> pd.Series:
    """Return a column of settings with each one printed rounded replaced by its exact value, sign kept."""
    for printed, value in exact.items():
        column = column.mask(np.isclose(column.abs(), printed), np.sign(column) * value)
    return column


def _pseudocomponents(table: pd.DataFrame, experiment: dict, exact: dict[float, float]) -> dict[str, pd.Series]:
    """Return the mixture components marked ``"coding": "pseudocomponent"`` as pseudocomponents.

    A component with lower bound l (its ``low``) becomes x' = (x - l) / (T - sum of the lower
    bounds), T the total of the mixture, ``runs["mixture_total"]`` (1 by default; 100 for
    percentages), so the pseudocomponents sum to 1 over the region the bounds leave.
    """
    mixture = [f for f in experiment["factors"] if f.get("coding") == "pseudocomponent"]
    if not mixture:
        return {}
    span = float(experiment["runs"].get("mixture_total", 1)) - sum(float(f["low"]) for f in mixture)
    return {f["name"]: (_exact_settings(table[f["name"]].astype(float), exact) - f["low"]) / span for f in mixture}


def _coded_runs(
    experiment: dict, positions: list[int] | None, summary: dict | None = None, *, as_printed: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the factor settings in coded units and the responses, for the runs at these 1-based positions.

    A factor marked ``"coding": "none"`` is fitted in its own units, as a source's regression on it is,
    one marked ``"pseudocomponent"`` as a pseudocomponent (see :func:`_pseudocomponents`), and one
    marked ``"name"`` by its level names, as a factor at three or more levels is: a two-level factor
    that another is nested in keeps its levels, so the nested term is fitted within each of them.
    ``runs["exact"]`` maps a setting printed rounded to its exact value: 1.414 to the root of 2, in
    coded units, or a level in a factor's own units, as 92.07 for 85 + 5√2; ``as_printed`` keeps the
    settings as printed, for a source that fitted its model to them. With a ``summary`` (see
    :func:`_summarised`) the responses are replaced by each run's statistic of its readings; with
    the statistic ``"each"``, every reading is an observation of its own, as a source fits a model
    to all the readings of a replicated design.
    """
    runs = experiment["runs"]
    table = pd.DataFrame(runs["rows"], columns=runs["columns"])
    if positions is not None:
        table = table.iloc[[p - 1 for p in positions]].reset_index(drop=True)
    exact = {} if as_printed else {float(k): float(v) for k, v in (runs.get("exact") or {}).items()}
    settings = {}
    pseudo = _pseudocomponents(table, experiment, exact)
    for f in experiment["factors"]:
        column = table[f["name"]]
        if f["name"] in pseudo:
            settings[f["name"]] = pseudo[f["name"]]
            continue
        if f.get("coding") == "none":
            settings[f["name"]] = _exact_settings(column.astype(float), exact)
            continue
        if f.get("coding") == "name" or (f.get("levels") and len(f["levels"]) > 2 and f["type"] == "categorical"):
            settings[f["name"]] = column.astype(str)  # a factor at three or more levels is fitted by name
            continue
        if runs["coded"]:
            values = column.astype(float)
        elif f["type"] == "categorical":
            values = column.map({f["low"]: -1.0, f["high"]: 1.0})
        else:
            centre, half = (f["high"] + f["low"]) / 2, (f["high"] - f["low"]) / 2
            values = (column.astype(float) - centre) / half
        settings[f["name"]] = _exact_settings(values, exact)  # 1.41 printed for the square root of 2, say
    if summary and summary["statistic"] == "each":
        return _each_reading(pd.DataFrame(settings), table[summary["of"]])
    if summary:
        return pd.DataFrame(settings), _summarised(table, summary)
    names = [r["name"] for r in experiment["responses"] if r["name"] in table.columns]
    return pd.DataFrame(settings), table[names].astype(float)


_ANALYSES = ["coefficients", "anova", "residual_diagnostics", "lack_of_fit", "lenth_method", "curvature_test"]


class _Fits:
    """Fit each experiment's runs once per model, runs, response and transform, keeping everything the checks read.

    ``check["transform"]`` fits the model to the transformed response (``"log"`` or ``"sqrt"``, as
    :func:`~process_improve.experiments.analyze_experiment` takes it), so its tables, and the
    response the other checks read, are on that scale. ``"settings": "as_printed"`` fits the
    settings as printed, not their exact values (see :func:`_coded_runs`).
    """

    def __init__(self) -> None:
        self._done: dict[tuple, dict[str, Any]] = {}

    def __call__(self, experiment: dict, check: dict) -> dict[str, Any]:
        positions = tuple(check["runs"]) if check.get("runs") else None
        summary = json.dumps(check["summary"], sort_keys=True) if check.get("summary") else None
        as_printed = check.get("settings") == "as_printed"
        options = (check.get("model"), positions, check.get("response"), summary, check.get("transform"), as_printed)
        key = (experiment["name"], *options)
        if key not in self._done:
            self._done[key] = self._fit(experiment, options)
        return self._done[key]

    @staticmethod
    def _fit(experiment: dict, options: tuple) -> dict[str, Any]:
        """Fit a model to a set of runs: the model, run positions, response, summary, transform and settings."""
        model, positions, response, summary, transform, as_printed = options
        settings, responses = _coded_runs(
            experiment,
            list(positions) if positions else None,
            json.loads(summary) if summary else None,
            as_printed=as_printed,
        )
        y = responses[response or responses.columns[0]]
        if transform is not None:
            y = pd.Series(transform_response(y.to_numpy(), transform, lambda: np.empty(0))[0], name=y.name)
        out: dict[str, Any] = {"y": y, "settings": settings}
        if model is None:
            return out
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for wanted in (_ANALYSES, ["coefficients", "anova", "residual_diagnostics"], ["coefficients"]):
                try:
                    out |= analyze_experiment(settings, y, model=model, analysis_type=wanted)
                    break
                except (ValueError, np.linalg.LinAlgError, KeyError, ZeroDivisionError):
                    continue
        return out


def _library_term(term: str) -> str:
    """Return the library's name for a term written A, A·B, A², A²·B or Intercept."""
    if term == "Intercept":
        return term

    def part(name: str) -> str:
        squared = re.fullmatch(r"(\w+)²", name)
        return f"I({squared.group(1)} ** 2)" if squared else name

    return ":".join(part(name) for name in term.split("·"))


def _coefficient_row(out: dict, term: str) -> dict:
    """Return the library's row for a term, a level of a factor coded to sum to zero included.

    A term written as the library names it is that row: ``power[160]``, a level mean of the
    model ``0 + power``, or ``C(power, Treatment('180'))[T.160]``, a difference from a reference level.
    """
    rows = {row["term"]: row for row in out["coefficients"]}
    if term in rows:
        return rows[term]
    level = re.fullmatch(r"(\w+)\[(.+)\]", term)
    if level:
        factor, name = level.groups()
        coded = {k: v for k, v in rows.items() if k.startswith(f"C({factor}, Sum)[")}
        key = next((k for k in coded if k.endswith(f".{name}]")), None)
        if key is not None:
            return coded[key]
        return {"coefficient": -sum(row["coefficient"] for row in coded.values())}  # the level the coding leaves out
    wanted = _library_term(term)
    for key, row in rows.items():
        if key == wanted or set(key.split(":")) == set(wanted.split(":")):
            return row
    raise KeyError(f"The fit has no term {term!r}: {sorted(rows)}.")


def _half_units(y: np.ndarray, rounding: dict) -> np.ndarray:
    """Return half a unit in each response's last printed digit: ``"decimals"``, or ``"significant"`` figures."""
    if "decimals" in rounding:
        exponent = np.full(len(y), -float(rounding["decimals"]))
    else:
        exponent = np.floor(np.log10(np.abs(y))) - rounding["significant"] + 1
    return 0.5 * 10.0**exponent


def _residual_vector(matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the residuals of the least squares fit of y on the columns of a model matrix."""
    return y - matrix @ (np.linalg.pinv(matrix) @ y)


def _carried_rounding(out: dict, check: dict) -> float:
    """Return how far a number can move when each response it is fitted to is off by half its last printed digit.

    A source that prints its runs' responses rounded, and fits its model to readings it does not
    print, gives ``check["data_rounding"]``: the responses' ``"decimals"``, or their ``"significant"``
    figures. A coefficient is a fixed weighted sum of the responses, its row of the pseudo-inverse of
    the model matrix, so it moves at most the sum of each weight's size times that response's half unit.
    The residual sum of squares e'e (``"ss_residual"``), and its mean square (``"ms_residual"``),
    move to first order by the gradient 2e, divided by the residual degrees of freedom for the mean square.
    """
    kind = check["kind"]
    y = out["y"].to_numpy(dtype=float)
    halves = _half_units(y, check["data_rounding"])
    matrix = dmatrix(check["model"], out["settings"], return_type="dataframe")
    if kind == "coefficient":
        wanted = set(_coefficient_row(out, check["term"])["term"].split(":"))
        row = next(i for i, name in enumerate(matrix.columns) if set(name.split(":")) == wanted)
        return float(np.abs(np.linalg.pinv(matrix.to_numpy())[row]) @ halves)
    if kind in ("ss_residual", "ms_residual"):
        gradient = 2 * _residual_vector(matrix.to_numpy(), y)
        scale = 1.0 if kind == "ss_residual" else float(out["model_summary"]["df_residual"])
        return float(np.abs(gradient / scale) @ halves)
    raise ValueError(f"data_rounding is carried into a coefficient or a residual sum of squares, not a {kind!r}.")


def _residuals(out: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diagnostics = out["residual_diagnostics"]
    fitted, residuals, leverage = (
        np.asarray(diagnostics[k], dtype=float) for k in ("fitted_values", "residuals", "leverage")
    )
    return fitted, residuals, leverage


def _left_out(out: dict, i: int) -> float:
    """Return run i as the model fitted without it predicts it: the result less its residual over 1 - leverage."""
    _, residuals, leverage = _residuals(out)
    return float(out["y"].to_numpy()[i] - residuals[i] / (1 - leverage[i]))


def _anova_sums(out: dict) -> tuple[float, float, int]:
    """Return the sum of squares the model explains, the residual's, and the model's degrees of freedom.

    The model's is the total about the mean less the residual's. That is the sum of the terms' rows
    only when the terms are orthogonal, as in a two-level factorial; when they are correlated, as the
    squares of a composite design are, the Type II rows leave out what the terms share.
    """
    table = out["anova_table"]
    residual = next(row for row in table if row["source"] == "Residual")
    model = [row for row in table if row["source"] != "Residual"]
    y = out["y"].to_numpy(dtype=float)
    total = float(np.nansum((y - np.nanmean(y)) ** 2))
    return total - residual["sum_sq"], residual["sum_sq"], sum(row["df"] for row in model)


def _f_regression(out: dict) -> tuple[float, float]:
    """Return the F ratio of the whole regression, its mean square over the residual's, and its p-value."""
    explained, unexplained, df_model = _anova_sums(out)
    df_residual = out["model_summary"]["df_residual"]
    ratio = (explained / df_model) / (unexplained / df_residual)
    return ratio, float(f_distribution.sf(ratio, df_model, df_residual))


def _anova_row(out: dict, term: str) -> dict:
    wanted = _library_term(term)
    return next(
        r for r in out["anova_table"] if r["source"] == wanted or set(r["source"].split(":")) == set(wanted.split(":"))
    )


def _percent_of_total(out: dict, check: dict, _experiment: dict) -> float:
    """Return a row's sum of squares (a term's, or the ``Residual``) as a percentage of the total about the mean."""
    return 100 * _anova_row(out, check["term"])["sum_sq"] / sum(_anova_sums(out)[:2])


def _percent_contribution(out: dict, check: dict, _experiment: dict) -> float:
    """Return Taguchi's percent contribution of a term: its SS less df x the error mean square, over the total SS.

    The ``Residual``'s is its sum of squares plus what every term gave up, so that the contributions
    add to 100. A source pools small columns into the error first; the check's model is the one left.
    """
    row = _anova_row(out, check["term"])
    error = out["model_summary"]["mse_residual"]
    df = -out["model_summary"]["df_model"] if row["source"] == "Residual" else row["df"]
    return 100 * (row["sum_sq"] - df * error) / sum(_anova_sums(out)[:2])


def _extra_sums(out: dict, check: dict) -> dict[str, float]:
    """Return what model ``check["part"]`` explains beyond ``check["after"]``: its sum of squares, mean square, F and p.

    Both are fitted to the check's runs; ``part`` is the check's own model when left out, and
    ``after`` no model at all. F divides the mean square by the residual mean square of the check's
    model, as RS/Discover splits a quadratic's regression into its linear and non-linear parts.
    """

    def explained(model: str | None) -> tuple[float, int]:
        if not model:
            return 0.0, 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = analyze_experiment(out["settings"], out["y"], model=model, analysis_type="anova")
        ss, _, df = _anova_sums({**fitted, "y": out["y"]})
        return ss, df

    ss_part, df_part = explained(check.get("part", check["model"]))
    ss_after, df_after = explained(check.get("after"))
    ss, df = ss_part - ss_after, df_part - df_after
    f_ratio = ss / df / out["model_summary"]["mse_residual"]
    p_value = float(f_distribution.sf(f_ratio, df, out["model_summary"]["df_residual"]))
    return {"sum_sq": ss, "mean_sq": ss / df, "F": f_ratio, "p_value": p_value}


def _lenth(out: dict, key: str, check: dict) -> float:
    """Return one of Lenth's margins, on the effects' scale or, when the source works with coefficients, on theirs."""
    return float(out["lenth_method"][key]) / (2.0 if check.get("scale") == "coefficient" else 1.0)


def _lenth_ratio(out: dict, check: dict) -> tuple[float, float]:
    """Return a term's pseudo t-ratio, its effect over Lenth's PSE, and its two-sided p-value on m/3 degrees of freedom.

    JMP's screening platform prints both beside each estimate of an unreplicated design: Lenth's
    PSE judges the m contrasts on the t distribution with d = m/3 degrees of freedom.
    """
    lenth = out["lenth_method"]
    wanted = set(_library_term(check["term"]).split(":"))
    effect = next(e["effect"] for e in lenth["effects"] if set(e["term"].split(":")) == wanted)
    ratio = effect / lenth["PSE"]
    return ratio, 2 * float(t_value_cdf(-abs(ratio), len(lenth["effects"]) / 3))


def _prediction(out: dict, check: dict, experiment: dict) -> float:
    """Return the model's prediction at a point in coded units, as the library predicts new runs.

    A factor ``check["point"]`` leaves out, as one the model does not hold, sits at its centre.
    ``check["bound"]`` reads a limit of its 95% interval instead: ``"pi_low"`` or ``"pi_high"`` for
    a new observation, ``"ci_low"`` or ``"ci_high"`` for the mean, or ``"ci_half_width"`` and
    ``"pi_half_width"``, an interval's half-width, as a prediction profiler prints it beside the
    prediction. ``"leverage"`` is x0'(X'X)^-1 x0, the variance of the predicted mean in units of σ²,
    and ``"std_error"`` the standard error of the predicted mean: the adjusted mean's, at the
    covariate's average, in an analysis of covariance.
    """
    names = [f["name"] for f in experiment["factors"]]
    point = pd.DataFrame([dict.fromkeys(names, 0.0) | check["point"]])[names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = analyze_experiment(
            out["settings"], out["y"], model=check["model"], analysis_type="prediction", new_points=point
        )
    prediction = found["predictions"][0]
    bound = check.get("bound", "predicted")
    if bound in ("ci_half_width", "pi_half_width"):
        interval = bound[:2]
        return float(prediction[f"{interval}_high"] - prediction[f"{interval}_low"]) / 2
    if bound == "std_error":
        half_width = float(prediction["ci_high"] - prediction["ci_low"]) / 2
        return half_width / float(t_value(0.975, out["model_summary"]["df_residual"]))
    if bound == "leverage":
        matrix = dmatrix(check["model"], out["settings"], return_type="dataframe")
        at = np.asarray(build_design_matrices([matrix.design_info], point)[0])[0]
        return float(at @ np.linalg.pinv(matrix.to_numpy().T @ matrix.to_numpy()) @ at)
    return float(prediction[bound])


def _target_setting(out: dict, check: dict, experiment: dict) -> float:
    """Return the setting of one factor at which the fitted model predicts a target, the others held fixed.

    ``check["solve"]`` names the factor, ``check["target"]`` is the response wanted and
    ``check["point"]`` holds the other factors' coded settings (a factor it leaves out sits at its
    centre). The prediction is a line or a parabola in the solved factor (a model of main effects,
    interactions and squares), read through its values at -1, 0 and +1; of a parabola's two
    settings, the one nearer the centre of the design is returned. The setting is in coded units,
    or in the factor's own with ``"units": "actual"``.
    """

    def predicted(setting: float) -> float:
        point = {f["name"]: 0.0 for f in experiment["factors"]} | check.get("point", {}) | {check["solve"]: setting}
        return _prediction(out, {**check, "point": point}, experiment)

    low, centre, high = predicted(-1.0), predicted(0.0), predicted(1.0)
    slope, curvature = (high - low) / 2, (high + low) / 2 - centre
    if math.isclose(curvature, 0.0, abs_tol=1e-9 * max(1.0, abs(centre))):
        if slope == 0:
            raise ValueError(f"The prediction does not change with {check['solve']} at this point.")
        coded = (check["target"] - centre) / slope
    else:
        roots = np.roots([curvature, slope, centre - check["target"]])
        real = roots[np.isreal(roots)].real
        if not len(real):
            raise ValueError(f"No setting of {check['solve']} reaches {check['target']} at this point.")
        coded = float(real[np.argmin(np.abs(real))])
    if check.get("units") != "actual":
        return coded
    factor = next(f for f in experiment["factors"] if f["name"] == check["solve"])
    return (factor["high"] + factor["low"]) / 2 + coded * (factor["high"] - factor["low"]) / 2


def _box_cox(out: dict, check: dict, _experiment: dict) -> float:
    """Return a number of the library's Box-Cox analysis of ``check["model"]``: ``lambda`` or ``lambda_ci.0``, ``.1``.

    ``"sse.<lambda>"`` is the residual sum of squares of the model fitted to the library's
    geometric-mean scaled power of the response at that power, SSE(lambda), the curve the power is
    chosen to minimise; ``"sse"`` is its minimum, at the library's lambda, and ``"ln_sse.<lambda>"``
    its natural log, as Design-Expert plots it. Run on demand, so that a response that is not
    positive does not cost every other check its fit.
    """
    kind, _, power = check["key"].partition(".")
    if kind in ("sse", "ln_sse"):
        y = out["y"].to_numpy(dtype=float)
        lmbda = float(power) if power else _box_cox(out, {**check, "key": "lambda"}, _experiment)
        scaled = _scaled_power(y, lmbda, float(np.mean(np.log(y))))
        rhs = check["model"].split("~", 1)[-1]
        matrix = np.asarray(dmatrix(rhs, out["settings"]), dtype=float)
        residual = scaled - matrix @ np.linalg.lstsq(matrix, scaled, rcond=None)[0]
        sse = float(residual @ residual)
        return math.log(sse) if kind == "ln_sse" else sse
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = analyze_experiment(out["settings"], out["y"], model=check["model"], analysis_type="box_cox")
    return _read(found["box_cox"], check["key"])


def _response_model(out: dict, experiment: dict) -> dict:
    """Return a fit as :func:`optimize_responses` takes it: coded coefficients, the factors in the entry's order.

    Only the factors the model holds are named, so a factor kept for another fit (a column in natural
    units, say) adds no flat direction to the canonical analysis.
    """
    used = set(re.findall(r"\w+", " ".join(row["term"] for row in out["coefficients"])))
    return {
        "response_name": out["y"].name,
        "coefficients": out["coefficients"],
        "factor_names": [f["name"] for f in experiment["factors"] if f["name"] in used],
    }


def _factor_ranges(experiment: dict) -> dict[str, dict[str, float]]:
    """Return each numeric factor's range in actual units, for the library's coded-to-actual conversion."""
    return {f["name"]: {"low": f["low"], "high": f["high"]} for f in experiment["factors"] if f["type"] == "numeric"}


def _optimum(out: dict, check: dict, experiment: dict) -> float:
    """Return one number of the library's analysis of a fitted quadratic: its stationary point or canonical form.

    ``check["method"]`` is ``"stationary_point"`` or ``"canonical_analysis"`` of
    :func:`optimize_responses`, and ``check["key"]`` reads into its result, as
    ``stationary_point.stationary_point_actual.T`` or ``canonical_analysis.eigenvectors.0.1``.
    An eigenvector's sign is arbitrary, so ``"abs": true`` compares its size only.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = optimize_responses(
            [_response_model(out, experiment)], method=check["method"], factor_ranges=_factor_ranges(experiment)
        )
    value = _read(found, check["key"])
    return abs(value) if check.get("abs") else value


def _steepest_path(out: dict, check: dict, experiment: dict) -> float:
    """Return a setting, in actual units, on the library's path of steepest ascent, set out as a source sets it out.

    A source steps along the path by a round change in one factor (``check["change"]`` actual units
    of ``check["per"]``) and moves the others in proportion; the setting of ``check["key"]`` after
    ``check["steps"]`` such steps is returned, or its change from the centre with ``"from_centre": true``.
    ``"units": "coded"`` returns it in coded units, as a source that steps in coded units prints it.
    """
    model, ranges = _response_model(out, experiment), _factor_ranges(experiment)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unit = optimize_responses([model], method="steepest_ascent", step_size=1.0, n_steps=1, factor_ranges=ranges)
        start, one = (step["actual"] for step in unit["steepest_path"]["steps"])
        size = check["change"] / (one[check["per"]] - start[check["per"]])  # the coded step making that change
        path = optimize_responses(
            [model], method="steepest_ascent", step_size=size, n_steps=check["steps"], factor_ranges=ranges
        )
    steps = path["steepest_path"]["steps"]
    units = check.get("units", "actual")
    setting = steps[-1][units][check["key"]]
    return setting - steps[0][units][check["key"]] if check.get("from_centre") else setting


def _mean_squares(out: dict, check: dict, _experiment: dict) -> float:
    """Return a linear combination of the library's mean squares, or Satterthwaite's ratio of two of them.

    ``check["terms"]`` maps rows of the analysis of variance (``"Residual"`` for the error) to their
    weights, and ``check["divisor"]`` divides their sum: the analysis-of-variance estimate of a
    variance component, as (MS_A - MS_AB)/bn equates the mean squares to their expectations under a
    random or mixed model. ``check["key"]`` is one of:

    - ``"value"``, and ``"root"``, its square root (the standard error of a mean, say);
    - ``"df"``, Satterthwaite's degrees of freedom (sum of w MS)^2 / sum of (w MS)^2/f;
    - ``"std_error"``, the estimate's standard error from Var(MS) = 2 MS^2/f, and ``"wald_low"`` and
      ``"wald_high"``, the estimate less or plus 1.96 of them (``check["z"]``, if given), which in a
      balanced design are the standard errors and 95% intervals JMP prints for its REML variance
      components;
    - ``"chi2_low"`` and ``"chi2_high"``, the 95% interval f v / chi2(0.975, f) to f v / chi2(0.025, f)
      on the f degrees of freedom of the estimate v, which JMP prints for the residual;
    - ``"mls_low"``, the modified large-sample lower limit of Ting et al. (1990), v - sqrt(VL), for an
      estimate with one positive weight, at ``check["alpha"]`` (0.05), and ``"mls_vl"``, its VL;
    - ``"covariance"``, the covariance of the estimate with that of ``check["with"]``, a second
      ``terms`` and ``divisor``.

    With ``check["denominator"]``, a second map of weights, the key is ``"f"``, the ratio of the two
    sums, ``"p"`` its upper tail on Satterthwaite's degrees of freedom, or ``"df_num"`` and
    ``"df_den"``: the exact test of a mean square against another, or the approximate F test of an
    effect no single mean square tests.
    """

    def rows(weights: dict[str, float]) -> list[tuple[dict, float]]:
        return [(_anova_row(out, term), float(w)) for term, w in weights.items()]

    def combination(weights: dict[str, float]) -> tuple[float, float]:
        terms = rows(weights)
        value = sum(w * row["mean_sq"] for row, w in terms)
        return value, value**2 / sum((w * row["mean_sq"]) ** 2 / row["df"] for row, w in terms)

    def covariance(first: dict, second: dict) -> float:
        """Sum over the shared mean squares of w1 w2 Var(MS), Var(MS) = 2 MS^2 / f: they are independent."""
        other = {row["source"]: w for row, w in rows(second["terms"])}
        shared = sum(
            w * other[row["source"]] * 2 * row["mean_sq"] ** 2 / row["df"]
            for row, w in rows(first["terms"])
            if row["source"] in other
        )
        return shared / (float(first.get("divisor", 1)) * float(second.get("divisor", 1)))

    def modified_large_sample(alpha: float) -> float:
        """VL = G1^2 c1^2 MS1^2 + sum of Hj^2 cj^2 MSj^2 + sum of G1j c1 cj MS1 MSj, the cj > 0 the negative weights."""
        divisor = float(check.get("divisor", 1))
        (first, c1), *others = sorted(rows(check["terms"]), key=lambda pair: -pair[1])
        if c1 <= 0 or any(w > 0 for _, w in others):
            raise ValueError("the modified large-sample limit here takes one positive weight and the rest negative")
        f1, s1 = first["df"], c1 * first["mean_sq"] / divisor
        g1 = 1 - f1 / float(chi2_distribution.ppf(1 - alpha, f1))
        vl = g1**2 * s1**2
        for row, w in others:
            fj, sj = row["df"], -w * row["mean_sq"] / divisor
            hj = fj / float(chi2_distribution.ppf(alpha, fj)) - 1
            f_point = float(f_distribution.ppf(1 - alpha, f1, fj))
            g1j = ((f_point - 1) ** 2 - g1**2 * f_point**2 - hj**2) / f_point
            vl += hj**2 * sj**2 + g1j * s1 * sj
        return vl

    numerator, df_num = combination(check["terms"])
    key = check.get("key", "value")
    if "denominator" not in check:
        value = numerator / float(check.get("divisor", 1))
        error = math.sqrt(covariance(check, check))
        found = {
            "value": lambda: value,
            "root": lambda: math.sqrt(value),
            "df": lambda: df_num,
            "std_error": lambda: error,
            "wald_low": lambda: value - float(check.get("z", _Z_975)) * error,
            "wald_high": lambda: value + float(check.get("z", _Z_975)) * error,
            "chi2_low": lambda: df_num * value / float(chi2_distribution.ppf(0.975, df_num)),
            "chi2_high": lambda: df_num * value / float(chi2_distribution.ppf(0.025, df_num)),
            "mls_vl": lambda: modified_large_sample(float(check.get("alpha", 0.05))),
            "mls_low": lambda: value - math.sqrt(modified_large_sample(float(check.get("alpha", 0.05)))),
            "covariance": lambda: covariance(check, check["with"]),
        }
        return found[key]()
    denominator, df_den = combination(check["denominator"])
    ratio = numerator / denominator
    tests = {"f": ratio, "p": float(f_distribution.sf(ratio, df_num, df_den)), "df_num": df_num, "df_den": df_den}
    return tests[key]


def _split_plot(out: dict, check: dict, experiment: dict) -> float:
    """Return a number of the library's split-plot analysis: REML with the whole plots as a random effect.

    ``check["whole_plot"]`` lists the columns whose combination is a run's whole plot (the replicate
    and the hard-to-change factor, say), and the model is ``check["model"]``, a formula; categorical
    factors written ``C(name, Sum)`` are coded to sum to zero, as their main effects' tests need.
    ``check["key"]`` is ``"whole_plot"`` or ``"residual"``, a variance; ``"whole_plot_error_df"`` or
    ``"subplot_error_df"``; or ``"f.<term>"``, ``"p.<term>"``, ``"df.<term>"`` or ``"df_den.<term>"``
    for a term's F test on Satterthwaite's degrees of freedom, the term named as the library names it.
    """
    runs = experiment["runs"]
    whole_plots = tuple(check["whole_plot"])
    cache = out.setdefault("split_plot_fits", {})
    if whole_plots not in cache:
        table = pd.DataFrame(runs["rows"], columns=runs["columns"])
        labels = table[list(whole_plots)].astype(str).agg(":".join, axis=1).to_numpy()
        settings = out["settings"].assign(WholePlot=labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cache[whole_plots] = analyze_experiment(
                settings, out["y"], model=check["model"], analysis_type="split_plot", whole_plot="WholePlot"
            )["split_plot"]
    fit = cache[whole_plots]
    kind, _, term = check["key"].partition(".")
    if kind in fit["variance_components"]:
        return float(fit["variance_components"][kind])
    if kind.endswith("_error_df"):
        return float(fit["error_df"][kind.removesuffix("_error_df")])
    test = next(row for row in fit["tests"] if row["source"] == term)
    return float({"f": test["F"], "p": test["p_value"], "df": test["df"], "df_den": test["df_denominator"]}[kind])


def _transmitted_variance(out: dict, check: dict, experiment: dict) -> float:
    """Return a number of the variance a fitted response model transmits from its noise factors.

    The noise factors z are ``check["noise"]``, uncorrelated, each of variance ``check["sigma_z"]``
    (1 by default, in coded units). A response model whose noise terms are the z's main effects and
    their interactions with the controllable factors x transmits the variance
    V(x) = sum over z of var(z) (dy/dz)^2 + s^2, where the slope dy/dz is linear in x and s^2 is the
    model's residual mean square (Montgomery, Eq. 12.6). The library fits the model and gives its
    coefficients; it has no propagation-of-error function of its own. ``check["key"]`` is a
    coefficient of the variance model, written as a term of the controllable factors
    (``"Intercept"``, ``"x2"``, ``"x2²"`` or ``"x2·x3"``), or ``"variance"`` or ``"poe"`` (its
    square root, Design-Expert's propagation of error) at the coded point ``check["point"]``.
    ``"error": false`` leaves s^2 out.
    """
    rows = {row["term"]: row["coefficient"] for row in out["coefficients"]}

    def coefficient(*names: str) -> float:
        return next((value for term, value in rows.items() if set(term.split(":")) == set(names)), 0.0)

    noise = check["noise"]
    sigma = {z: float((check.get("sigma_z") or {}).get(z, 1.0)) for z in noise}
    controls = [f["name"] for f in experiment["factors"] if f["name"] not in noise]
    slopes = {z: (coefficient(z), {x: coefficient(x, z) for x in controls}) for z in noise}
    error = out["model_summary"]["mse_residual"] if check.get("error", True) else 0.0
    key = check["key"]
    if key in ("variance", "poe"):
        point = check.get("point", {})
        variance = error + sum(
            sigma[z] * (gamma + sum(delta[x] * point.get(x, 0.0) for x in controls)) ** 2
            for z, (gamma, delta) in slopes.items()
        )
        return math.sqrt(variance) if key == "poe" else variance
    if key == "Intercept":
        return error + sum(sigma[z] * gamma**2 for z, (gamma, _) in slopes.items())
    factors = key.split("·")
    if len(factors) == 2:
        first, second = factors
        return sum(2 * sigma[z] * delta[first] * delta[second] for z, (_, delta) in slopes.items())
    if key.endswith("²"):
        return sum(sigma[z] * delta[key[:-1]] ** 2 for z, (_, delta) in slopes.items())
    return sum(2 * sigma[z] * gamma * delta[key] for z, (gamma, delta) in slopes.items())


def _p_value(row: dict, alternative: str | None) -> float:
    """Return a coefficient's p-value: two-sided, or one-sided when a source tests one direction.

    ``alternative`` is ``"greater"`` (the coefficient is above zero) or ``"less"``; the one-sided
    p-value is half the two-sided one when the estimate lies on that side, and one less that half
    when it does not.
    """
    p = float(row["p_value"])
    if alternative is None:
        return p
    agrees = row["coefficient"] > 0 if alternative == "greater" else row["coefficient"] < 0
    return p / 2 if agrees else 1 - p / 2


def _run(check: dict) -> int:
    return int(check["run"]) - 1


def _residual_std_dev(out: dict, at: list[int], digits: int | None = None) -> float:
    """Return the standard deviation of the fit's residuals at these 1-based runs, one level of a factor, say.

    A source that works from its printed residuals gives ``digits``, the decimals they are rounded to.
    """
    residuals = _residuals(out)[1][[p - 1 for p in at]]
    return float(np.std(residuals if digits is None else np.round(residuals, digits), ddof=1))


def _contrast(out: dict, check: dict, _experiment: dict) -> float:
    """Return a number of a contrast of a fit's level means: the sum of each weight times its term's coefficient.

    ``check["weights"]`` maps each term, as the library names it, to its weight. The terms are level
    means, as of the model ``0 + power`` (or ``0 + pressure + C(batch, Sum)`` for a balanced
    randomized block design), whose estimates are uncorrelated, so the contrast's standard error is
    the root of the sum of each weight's square times its term's variance. ``check["key"]`` is
    ``"value"`` (the default), ``"std_error"``, ``"t"``, ``"f"`` (the square of t: the F of the
    contrast's one degree of freedom), ``"sum_sq"`` (F times the residual mean square: the
    contrast's sum of squares), ``"p"``, or ``"half_width"``, ``"ci_low"`` and ``"ci_high"`` of its
    95% interval (Fisher's least significant difference, for the difference of two means).
    """
    rows = [(_coefficient_row(out, term), weight) for term, weight in check["weights"].items()]
    value = sum(weight * row["coefficient"] for row, weight in rows)
    key = check.get("key", "value")
    if key == "value":  # a contrast of a saturated fit, such as one cycle of an EVOP, has no standard error
        return value
    std_error = math.sqrt(sum((weight * row["std_error"]) ** 2 for row, weight in rows))
    t = value / std_error
    summary = out["model_summary"]
    half = float(t_value(0.975, summary["df_residual"])) * std_error
    found = {
        "std_error": std_error,
        "t": t,
        "f": t**2,
        "sum_sq": t**2 * summary["mse_residual"],
        "p": 2 * float(t_value_cdf(-abs(t), summary["df_residual"])),
        "half_width": half,
        "ci_low": value - half,
        "ci_high": value + half,
    }
    return found[key]


def _diagnostic(name: str) -> Callable[[dict, dict, dict], float]:
    """Return a check reading one run's value of a residual diagnostic the library lists run by run."""
    return lambda out, c, _e: float(out["residual_diagnostics"][name][_run(c)])


_FIT_KINDS: dict[str, Callable[[dict, dict, dict], float]] = {
    "coefficient": lambda out, c, _e: _coefficient_row(out, c["term"])["coefficient"],
    "effect": lambda out, c, _e: 2 * _coefficient_row(out, c["term"])["coefficient"],
    "std_error": lambda out, c, _e: _coefficient_row(out, c["term"])["std_error"],
    "standardised_coefficient": lambda out, c, _e: _coefficient_row(out, c["term"])["standardised_coefficient"],
    "effect_std_error": lambda out, c, _e: 2 * _coefficient_row(out, c["term"])["std_error"],
    "t_value": lambda out, c, _e: _coefficient_row(out, c["term"])["t_value"],
    "p_value": lambda out, c, _e: _p_value(_coefficient_row(out, c["term"]), c.get("alternative")),
    "ci_low": lambda out, c, _e: _coefficient_row(out, c["term"])["ci_low"] * (2 if c.get("scale") == "effect" else 1),
    "ci_high": lambda out, c, _e: (
        _coefficient_row(out, c["term"])["ci_high"] * (2 if c.get("scale") == "effect" else 1)
    ),
    "fitted": lambda out, c, _e: float(_residuals(out)[0][_run(c)]),
    "residual": lambda out, c, _e: float(_residuals(out)[1][_run(c)]),
    "leverage": _diagnostic("leverage"),
    "cooks_distance": _diagnostic("cooks_distance"),
    "studentized_residual": _diagnostic("studentized_residuals"),
    "externally_studentized_residual": _diagnostic("externally_studentized_residuals"),
    "contrast": _contrast,
    "durbin_watson": lambda out, _c, _e: out["residual_diagnostics"]["durbin_watson"],
    "left_out_prediction": lambda out, c, _e: _left_out(out, _run(c)),
    "residual_std_dev": lambda out, c, _e: _residual_std_dev(out, c["at"], c.get("round")),
    "residual_log_variance_ratio": lambda out, c, _e: (
        2
        * math.log(
            _residual_std_dev(out, c["at"], c.get("round")) / _residual_std_dev(out, c["versus"], c.get("round"))
        )
    ),
    "observation": lambda out, c, _e: float(out["y"].to_numpy()[_run(c)]),
    "mean": lambda out, _c, _e: float(out["y"].mean()),
    "sum": lambda out, _c, _e: float(out["y"].sum()),
    "median": lambda out, _c, _e: float(out["y"].median()),
    "minimum": lambda out, _c, _e: float(out["y"].min()),
    "maximum": lambda out, _c, _e: float(out["y"].max()),
    "range": lambda out, _c, _e: float(out["y"].max() - out["y"].min()),
    "std_dev": lambda out, _c, _e: float(out["y"].std(ddof=1)),
    "variance": lambda out, _c, _e: float(out["y"].var(ddof=1)),
    "n_runs": lambda out, _c, _e: float(len(out["y"])),
    "r_squared": lambda out, _c, _e: out["model_summary"]["r_squared"],
    "multiple_r": lambda out, _c, _e: out["model_summary"]["r_squared"] ** 0.5,
    "r_squared_adj": lambda out, _c, _e: out["model_summary"]["r_squared_adj"],
    "r_squared_pred": lambda out, _c, _e: out["model_summary"]["r_squared_pred"],
    "adequate_precision": lambda out, _c, _e: out["model_summary"]["adequate_precision"],
    "residual_df": lambda out, _c, _e: out["model_summary"]["df_residual"],
    "df_model": lambda out, _c, _e: out["model_summary"]["df_model"],
    "ms_residual": lambda out, _c, _e: out["model_summary"]["mse_residual"],
    "residual_std_error": lambda out, _c, _e: out["model_summary"]["mse_residual"] ** 0.5,
    "ss_residual": lambda out, _c, _e: _anova_sums(out)[1],
    "ss_regression": lambda out, _c, _e: _anova_sums(out)[0],
    "ss_total": lambda out, _c, _e: sum(_anova_sums(out)[:2]),
    "ms_regression": lambda out, _c, _e: _anova_sums(out)[0] / _anova_sums(out)[2],
    "f_regression": lambda out, _c, _e: _f_regression(out)[0],
    "f_regression_p": lambda out, _c, _e: _f_regression(out)[1],
    "press": lambda out, _c, _e: float(np.sum((_residuals(out)[1] / (1 - _residuals(out)[2])) ** 2)),
    "lenth_pse": lambda out, c, _e: _lenth(out, "PSE", c),
    "lenth_me": lambda out, c, _e: _lenth(out, "ME", c),
    "lenth_sme": lambda out, c, _e: _lenth(out, "SME", c),
    "lenth_t": lambda out, c, _e: _lenth_ratio(out, c)[0],
    "lenth_p": lambda out, c, _e: _lenth_ratio(out, c)[1],
    "anova_sum_sq": lambda out, c, _e: _anova_row(out, c["term"])["sum_sq"],
    "anova_mean_sq": lambda out, c, _e: _anova_row(out, c["term"])["mean_sq"],
    "anova_f": lambda out, c, _e: _anova_row(out, c["term"])["F"],
    "anova_p": lambda out, c, _e: _anova_row(out, c["term"])["p_value"],
    "anova_df": lambda out, c, _e: _anova_row(out, c["term"])["df"],
    "percent_of_total": _percent_of_total,
    "percent_contribution": _percent_contribution,
    "extra_ss": lambda out, c, _e: _extra_sums(out, c)["sum_sq"],
    "extra_ms": lambda out, c, _e: _extra_sums(out, c)["mean_sq"],
    "extra_f": lambda out, c, _e: _extra_sums(out, c)["F"],
    "extra_p": lambda out, c, _e: _extra_sums(out, c)["p_value"],
    "lack_of_fit_ss": lambda out, _c, _e: out["lack_of_fit"]["ss_lack_of_fit"],
    "lack_of_fit_df": lambda out, _c, _e: out["lack_of_fit"]["df_lack_of_fit"],
    "lack_of_fit_ms": lambda out, _c, _e: out["lack_of_fit"]["ms_lack_of_fit"],
    "lack_of_fit_f": lambda out, _c, _e: out["lack_of_fit"]["f_statistic"],
    "lack_of_fit_p": lambda out, _c, _e: out["lack_of_fit"]["p_value"],
    "pure_error_ss": lambda out, _c, _e: out["lack_of_fit"]["ss_pure_error"],
    "pure_error_df": lambda out, _c, _e: out["lack_of_fit"]["df_pure_error"],
    "pure_error_ms": lambda out, _c, _e: out["lack_of_fit"]["ms_pure_error"],
    "max_r_squared": lambda out, _c, _e: 1 - out["lack_of_fit"]["ss_pure_error"] / sum(_anova_sums(out)[:2]),
    "curvature_difference": lambda out, _c, _e: out["curvature_test"]["difference"],
    "curvature": lambda out, c, _e: float(out["curvature_test"][c["key"]]),
    "prediction": _prediction,
    "target_setting": _target_setting,
    "optimum": _optimum,
    "box_cox": _box_cox,
    "steepest_path": _steepest_path,
    "transmitted_variance": _transmitted_variance,
    "mean_squares": _mean_squares,
    "split_plot": _split_plot,
}


# --------------------------------------------------------------------------- what a source says of a design
_ROMAN = {"III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}
_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def _plain(word: str) -> str:
    """Return an effect word without spaces, dots or signs of multiplication: A·B·C as ABC."""
    return re.sub(r"[\s\u00b7\u00d7*]", "", word)


def _word(word: str) -> str:
    """Return an effect word with its letters in order, the way :func:`_multiply` writes it: B·D·C as BCD.

    A word is a product of factors, so its letters commute; a minus sign is kept, as a letter that
    cancels in pairs. The intercept, I, is the empty word.
    """
    plain = _plain(word)
    return "" if plain in ("I", "Intercept") else "".join(sorted(plain))


def _words(printed: str) -> list[str]:
    """Return the words of a printed list, such as ``I = ABD = ACE``, ``AD and CDE`` or ``BCD, ACD``."""
    parts = re.split(r"[,=]|\band\b", printed.replace("\u2212", "-"))
    return [w for w in (_word(p) for p in parts) if w]


def _generator_word(generator: str) -> str:
    """Return the word a generator makes: D = AB makes ABD, and F = \u2212ABCDE makes \u2212ABCDEF."""
    return _multiply(*(_word(side) for side in generator.replace("\u2212", "-").split("=")))


def _multiply(*words: str) -> str:
    """Multiply effect words letter by letter: a letter twice cancels, as A·A = I."""
    letters = sorted(itertools.chain.from_iterable(words))
    return "".join(letter for letter in sorted(set(letters)) if letters.count(letter) % 2)


def _fraction(check: dict, k: int) -> Any:  # noqa: ANN401 - the library's DesignResult
    """Return the library's fraction for a check: in k factors, by its resolution or by the source's generators.

    Its factors are lettered A, B, C, ..., or named as ``check["factors"]`` names them.
    """
    names = check.get("factors") or list(_LETTERS[:k])  # the source's own letters, when it names its factors
    factors = [Factor(name=name, low=-1, high=1) for name in names]
    return generate_design(
        factors,
        design_type="fractional_factorial",
        resolution=check.get("resolution"),
        generators=check.get("generators"),
        n_center_points=0,
    )


def _design_check(check: dict, printed: str, k: int) -> tuple[str, bool]:
    """Check a generator, resolution, defining relation or aliases the source states, against the library's fraction.

    Words compare as products of factors, whatever order the source writes their letters in. A
    defining relation must be the whole relation; the aliases printed must each be an alias of the
    term, and need not be all of them. ``"word_lengths"`` compares the lengths of the relation's
    words, the word length pattern printed as {4, 5, 5}; ``"clear"`` holds when the term is aliased
    with no main effect or two-factor interaction; ``"minimum_aberration"`` compares the fraction
    with the one the library chooses for its resolution: they agree when both have as many runs and
    the same word lengths. With ``"up_to": n``, the aliases printed must be all of the term's aliases
    of n factors or fewer, so a chain printed empty says the term has none. ``"block_effects"``
    holds when the effects printed are those confounded with the blocks that ``check["of"]``
    makes: each of the chosen effects and every product of them, any of which may be printed as
    another word of its alias chain.
    """
    made = _fraction(check, check.get("k", k))
    kind = check["kind"]
    if kind == "generator":
        found = sorted(_generator_word(g) for g in made.generators or [])
        return ", ".join(found), _generator_word(printed) in found
    if kind == "resolution":
        wanted = _ROMAN.get(printed.strip())
        return str(made.resolution), wanted == made.resolution
    return _relation_check(made, check, printed, check.get("k", k))


def _relation_words(made: Any) -> list[str]:  # noqa: ANN401 - a DesignResult
    """Return the words of a fraction's defining relation, each with its letters in order.

    The library lists a relation of up to ten generators with the design, and works out a longer one on request.
    """
    relation = made.defining_relation or evaluate_design(made, metric="defining_relation")["defining_relation"]
    return sorted(_word(w.split("=")[-1]) for w in relation or [])


def _relation_check(made: Any, check: dict, printed: str, k: int) -> tuple[str, bool]:  # noqa: ANN401 - a DesignResult
    """Check what a source states of a fraction's defining relation: itself, its word lengths, or a term's aliases."""
    kind = check["kind"]
    words = _relation_words(made)
    if kind == "defining_relation":
        return ", ".join(words), sorted(_words(printed)) == words
    lengths = sorted(len(w.lstrip("-")) for w in words)
    if kind == "word_lengths":  # the word length pattern, printed as the lengths of the words: {4, 5, 5}
        return ", ".join(map(str, lengths)), sorted(int(n) for n in re.findall(r"\d+", printed)) == lengths
    if kind == "minimum_aberration":  # the fraction the library chooses for this resolution, of as many runs
        chosen = _fraction({"resolution": made.resolution, "factors": check.get("factors")}, k)
        chosen_lengths = sorted(len(w.lstrip("-")) for w in _relation_words(chosen))
        same = len(chosen.design) == len(made.design) and chosen_lengths == lengths
        return ", ".join(chosen.generators or []), same
    if kind == "block_effects":
        return _block_effects(words, check, printed)
    return _alias_check(words, check, printed)


def _block_effects(words: list[str], check: dict, printed: str) -> tuple[str, bool]:
    """Check the effects confounded with blocks: the chosen effects and their products, each read as its alias chain."""

    def chain(word: str) -> frozenset[str]:
        return frozenset(_multiply(word, w) for w in ["", *words])

    chosen = _words(check["of"])
    found = {chain(_multiply(*s)) for r in range(1, len(chosen) + 1) for s in itertools.combinations(chosen, r)}
    shown = ", ".join(min(c, key=lambda w: (len(w), w)) for c in found)
    return shown, {chain(w) for w in _words(printed)} == found


def _alias_check(words: list[str], check: dict, printed: str) -> tuple[str, bool]:
    """Check a term's aliases, as a fraction's defining relation ``words`` gives them, or that it is clear."""
    aliases = sorted({_multiply(_word(check["term"]), word) for word in words})
    if check["kind"] == "clear":  # aliased with no main effect or two-factor interaction
        short = [a for a in aliases if len(a.lstrip("-")) <= 2]
        return ", ".join(short) or "no main effect or two-factor interaction", not short
    if "up_to" in check:  # every alias of up to that many factors, and only those
        short = sorted((a for a in aliases if len(a.lstrip("-")) <= check["up_to"]), key=lambda w: (len(w), w))
        return ", ".join(short) or "none", set(_words(printed)) == set(short)
    return ", ".join(aliases), bool(_words(printed)) and set(_words(printed)) <= set(aliases)  # kind == "aliases"


_DESIGN_KINDS = {
    "generator",
    "resolution",
    "defining_relation",
    "word_lengths",
    "minimum_aberration",
    "aliases",
    "clear",
    "block_effects",
}


def _block_confounding(check: dict, printed: str) -> tuple[str, bool]:
    """Check the interactions a source confounds with blocks against those the library's blocking confounds.

    ``check["design"]`` names the blocked design for ``generate_design`` (``n_blocks`` among its
    ``options``). The factors of a full factorial are interchangeable, so the two arrangements agree
    when they confound as many interactions of each order: two-factor, three-factor, and so on.
    """
    made = _made(json.dumps(check["design"], sort_keys=True))
    found = made.metadata["blocking"]["confounded_with"]
    return ", ".join(found), sorted(map(len, _words(printed))) == sorted(map(len, found))


def _generalized_interactions(_experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check the effects a source confounds with blocks against the products of those it chose to generate them.

    Blocks made by p independent effects confound every product of them, two at a time, three at a
    time and so on (their generalized interactions): 2^p - 1 effects in all. ``check["of"]`` lists
    the chosen effects, as printed; the printed list must hold exactly their products, in any order.
    """
    chosen = _words(check["of"])
    found = {_multiply(*subset) for r in range(1, len(chosen) + 1) for subset in itertools.combinations(chosen, r)}
    listed = _words(printed)
    shown = ", ".join(sorted(found, key=lambda word: (len(word), word)))
    return shown, len(listed) == len(found) and set(listed) == found


def _design_property(check: dict) -> float:
    """Return a property of a design the library makes: its run count, axial distance or resolution.

    ``check["design"]`` names it: ``design_type``, ``k`` factors, and ``options`` for ``generate_design``.
    A resolution may be printed as a Roman numeral. ``"squared": true`` returns its square, as a
    source compares the square of a rotatable design's axial distance. The property ``"fraction"``
    is the share of the full factorial a fraction runs, printed as 1/4, and ``"blocks": b`` divides
    the property among b blocks, as a source gives the runs in each block.
    """
    made = _made(json.dumps(check["design"], sort_keys=True))
    prop = check["property"]
    value = made.n_runs / 2**made.n_factors if prop == "fraction" else float(getattr(made, prop))
    value = value / check.get("blocks", 1)
    return value**2 if check.get("squared") else value


@functools.cache
def _made(spec_json: str) -> Any:  # noqa: ANN401 - the library's DesignResult
    """Return the design a specification names, made once however many numbers are read from it.

    ``spec["levels"]``, a number or one per factor, gives a factor more than two equally spaced
    levels, coded -1 to +1, as a three-level factor of a Taguchi array or full factorial has. Past
    the alphabet, factors are named X26, X27, and so on, as a screening design for many inputs has.
    """
    spec = json.loads(spec_json)
    levels = spec.get("levels") or 2
    counts = levels if isinstance(levels, list) else [levels] * spec["k"]
    factors = [
        Factor(
            name=_LETTERS[i] if i < len(_LETTERS) else f"X{i + 1}",
            low=-1,
            high=1,
            levels=np.linspace(-1, 1, n).tolist() if n > 2 else None,
        )
        for i, n in enumerate(counts)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_design(factors, design_type=spec["design_type"], **spec.get("options", {}))


def _printed_matrix(
    experiment: dict, columns: list[str], alpha: float | None, positions: list[int] | None = None
) -> np.ndarray:
    """Return a printed design matrix in coded units, its axial distance written ``\u03b1`` read as ``alpha``.

    ``positions`` keeps only the runs at these 1-based positions.
    """
    runs = experiment["runs"]
    table = pd.DataFrame(runs["rows"], columns=runs["columns"])[columns]
    if positions is not None:
        table = table.iloc[[p - 1 for p in positions]]
    symbols = {"\u03b1": alpha, "-\u03b1": None if alpha is None else -alpha}
    return table.map(lambda v: symbols[v] if isinstance(v, str) else v).to_numpy(dtype=float)


def _array_runs(check: dict) -> np.ndarray:
    """Return a stored orthogonal array in coded units, each column's levels spread evenly from -1 to +1.

    ``check["split"]`` writes a column as two: ``{"0": [2, 3]}`` turns a six-level column 0 into a
    two-level column (its level divided by 3) followed by a three-level one (the remainder).
    """
    array = orthogonal_array(check["array"])
    split = {int(j): sizes for j, sizes in (check.get("split") or {}).items()}
    columns: list[tuple[np.ndarray, int]] = []
    for j in range(array.shape[1]):
        if j in split:
            outer, inner = split[j]
            columns += [(array[:, j] // inner, outer), (array[:, j] % inner, inner)]
        else:
            columns.append((array[:, j], int(array[:, j].max()) + 1))
    return np.column_stack([2 * level / (n - 1) - 1 for level, n in columns])


def _same_runs(experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check that a printed design matrix holds the same runs as the library's, whatever their order.

    The library's design is ``check["design"]`` (as :func:`_made` makes it) or the stored
    orthogonal array ``check["array"]`` (see :func:`_array_runs`). ``check["columns"]`` names the
    printed columns that match its factors, in order (all the entry's factors by default); a
    printed ``\u03b1`` is the library design's axial distance. ``check["relabel"]`` allows the columns
    to be matched in another order (``"columns"``) and with their signs changed
    (``"columns and signs"``), as two constructions of the same design may differ. A printed
    number of runs, when the text is one, must be the number compared. ``check["round"]`` rounds
    both matrices to that many decimals first, for an axial distance printed as 1.414.
    """
    if "array" in check:
        library, alpha = _array_runs(check), None
    else:
        made = _made(json.dumps(check["design"], sort_keys=True))
        library, alpha = made.design[made.factor_names].to_numpy(dtype=float), made.alpha
    columns = check.get("columns") or [f["name"] for f in experiment["factors"]]
    if experiment["runs"].get("coded", True):
        matrix = _printed_matrix(experiment, columns, alpha)
    else:  # a matrix printed in the factors' own units is compared in coded units
        matrix = _coded_runs(experiment, None)[0][columns].to_numpy(dtype=float)
    if "round" in check:
        library, matrix = np.round(library, check["round"]), np.round(matrix, check["round"])
    count_agrees = not str(printed).strip().isdigit() or int(printed) == len(matrix)
    relabel = check.get("relabel")
    if relabel:
        found = relabelling(matrix, library, signs=relabel == "columns and signs")
        if found is None:
            return "no relabelling of the library's columns gives the printed runs", False
        named = [f"{'-' if s < 0 else ''}{c + 1}" for c, s in found]
        return f"the library's columns {', '.join(named)}", count_agrees
    shared = sum((run_counts(matrix) & run_counts(library)).values())
    same = run_counts(matrix) == run_counts(library)
    return f"{shared} of {len(matrix)} printed runs, of {len(library)} the library's", same and count_agrees


def _j_counts(matrix: np.ndarray, order: int) -> Counter:
    """Return how often each size of J-characteristic occurs: ``|sum over the runs of a product of order columns|``.

    For a two-level design these say how strongly each product of columns is aliased with the
    intercept: 0 for an orthogonal pair, N for a product aliased completely. Their counts do not
    change when the columns are reordered or their signs changed, or the runs reordered.
    """
    columns = range(matrix.shape[1])
    return Counter(round(abs(matrix[:, list(c)].prod(axis=1).sum())) for c in itertools.combinations(columns, order))


def _same_projections(experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check that a printed two-level design matches the library's in every projection onto up to three columns.

    For a design that can be printed with its columns in another order or their signs changed,
    such as a Plackett-Burman design written from a generating row: the two agree when they have
    as many runs and columns, and as many one-, two- and three-column J-characteristics of each
    size (see :func:`_j_counts`). Agreement in all three shows the printed columns are as balanced
    and as orthogonal as the library's, and their products of three as strongly aliased. It is the
    test of :func:`_same_runs` with its relabelling, without the search over relabellings.
    """
    made = _made(json.dumps(check["design"], sort_keys=True))
    library = made.design[made.factor_names].to_numpy(dtype=float)
    columns = check.get("columns") or [f["name"] for f in experiment["factors"]]
    matrix = _printed_matrix(experiment, columns, None)
    same = matrix.shape == library.shape and all(_j_counts(matrix, r) == _j_counts(library, r) for r in (1, 2, 3))
    found = ", ".join(f"{size}: {count}" for size, count in sorted(_j_counts(library, 3).items()))
    return f"the library's three-column |J| sizes and counts {found}", same


def _generator_holds(experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check that a printed design matrix obeys the generators printed with it, in every run.

    ``E = ABC`` holds when the product of a run's E, A, B and C settings is +1 (-1 for ``E = -ABC``);
    several generators may be printed together, separated by commas. ``check["runs"]`` keeps the
    runs the generators make, such as the cube of a composite design.
    """
    positions = check.get("runs") or list(range(1, len(experiment["runs"]["rows"]) + 1))
    failing = set()
    for generator in str(check.get("printed", printed)).split(","):
        word = _generator_word(generator)
        sign = -1.0 if word.startswith("-") else 1.0
        product = _printed_matrix(experiment, list(word.lstrip("-")), None, positions).prod(axis=1)
        failing |= {run for run, value in zip(positions, product, strict=True) if not np.isclose(value, sign)}
    found = "every run" if not failing else f"not in runs {', '.join(map(str, sorted(failing)))}"
    return found, not failing


def _product_column(experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check that a printed column of signs is the product of the factor columns of each word it is headed with.

    A table of signs heads a column with an effect and its aliases, ``AB = CE``; the column
    ``check["column"]`` holds, in every run, the product of the A and B settings and that of C and
    E. A word with a minus sign, ``-ABD``, is the negated product.
    """
    column = np.asarray(
        [row[experiment["runs"]["columns"].index(check["column"])] for row in experiment["runs"]["rows"]]
    )
    failing = set()
    for word in _words(str(check.get("printed", printed))):
        sign = -1.0 if word.startswith("-") else 1.0
        product = sign * _printed_matrix(experiment, list(word.lstrip("-")), None).prod(axis=1)
        failing |= {i + 1 for i in np.flatnonzero(~np.isclose(product, column.astype(float)))}
    found = "every run" if not failing else f"not in runs {', '.join(map(str, sorted(failing)))}"
    return found, not failing


# --------------------------------------------------------------------------- the 3^k system, worked modulo 3
_SUPERSCRIPTS = str.maketrans("²³⁴⁵", "2345")


def _powers(word: str) -> tuple[int, ...]:
    """Return a 3^k effect word's exponents, one per factor letter, reduced modulo 3: AB²C as (1, 2, 1).

    The word may print its exponents as superscripts or digits (AB2C); a letter with none has 1.
    """
    exponents = [0] * len(_LETTERS)
    for letter, exponent in re.findall(r"([A-Z])(\d*)", _plain(word).translate(_SUPERSCRIPTS)):
        exponents[_LETTERS.index(letter)] += int(exponent or 1)
    used = max((i for i, e in enumerate(exponents) if e % 3), default=-1) + 1
    return tuple(e % 3 for e in exponents[:used])


def _component(powers: tuple[int, ...]) -> tuple[int, ...]:
    """Return a component of interaction in the book's form, its first exponent 1: A²B is squared to AB².

    A word and its square name the same two-degree-of-freedom component, as L and 2L (mod 3) split
    the runs into the same three groups.
    """
    first = next((e for e in powers if e), 1)
    scaled = tuple(e * first % 3 for e in powers)  # the first exponent is 1 or 2, and 2 * 2 = 1 (mod 3)
    return scaled[: max((i + 1 for i, e in enumerate(scaled) if e), default=0)]


def _word_product(words: list[tuple[int, ...]], powers: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Return the product of 3^k effect words, each raised to its power: their exponents added modulo 3."""
    size = max(map(len, words))
    return tuple(
        sum(p * (w[i] if i < len(w) else 0) for w, p in zip(words, powers, strict=True)) % 3 for i in range(size)
    )


def _component_text(powers: tuple[int, ...]) -> str:
    return "".join(_LETTERS[i] + ("²" if e == 2 else "") for i, e in enumerate(powers) if e) or "I"


def _component_product(_experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check a product of 3^k effect words, the exponents added modulo 3, as a generalized interaction or an alias.

    ``check["words"]`` are multiplied, each raised to its ``check["powers"]`` (1 by default): the
    aliases of A in a fraction with I = AB²C² are A(AB²C²) and A(AB²C²)².
    """
    words = [_powers(w) for w in check["words"]]
    found = _component(_word_product(words, check.get("powers") or [1] * len(words)))
    return _component_text(found), found == _component(_powers(printed))


def _component_relation(_experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check the defining relation of a 3^(k-p) fraction made by the components ``check["words"]``, or its resolution.

    The relation holds every product of the p words, each to the power 0, 1 or 2, but not all 0:
    (3^p - 1)/2 components, as a word and its square are one. ``check["key"]`` is ``"relation"``,
    compared with the printed ``I = AB²C = BCD = ...`` in any order, or ``"resolution"``, the
    length of its shortest word, printed as a Roman numeral.
    """
    words = [_powers(w) for w in check["words"]]
    products = (_word_product(words, powers) for powers in itertools.product(range(3), repeat=len(words)))
    relation = {_component(product) for product in products if any(product)}
    if check.get("key") == "resolution":
        resolution = min(sum(1 for e in word if e) for word in relation)
        return str(resolution), _ROMAN.get(printed.strip()) == resolution
    listed = {_component(_powers(w)) for w in re.split(r"[=,]", printed) if _plain(w) not in ("", "I")}
    return " = ".join(["I", *sorted(map(_component_text, relation))]), listed == relation


def _levels_of(experiment: dict, names: list[str], positions: list[int] | None = None) -> np.ndarray:
    """Return the runs' levels of three-level factors as 0, 1 and 2: each value's place in its factor's ``levels``."""
    runs = experiment["runs"]
    table = pd.DataFrame(runs["rows"], columns=runs["columns"])
    if positions is not None:
        table = table.iloc[[p - 1 for p in positions]]
    factors = {f["name"]: [str(level) for level in f["levels"]] for f in experiment["factors"]}
    return np.column_stack([table[name].map(lambda v, n=name: factors[n].index(str(v))) for name in names])


def _contrast_values(experiment: dict, check: dict, positions: list[int] | None = None) -> np.ndarray:
    """Return each run's values of the defining contrasts L = a1 x1 + a2 x2 + ... (mod 3) of ``check["words"]``.

    The factor letters A, B, C, ... are the entry's factors in order, or ``check["factors"]``.
    """
    names = check.get("factors") or [f["name"] for f in experiment["factors"]]
    levels = _levels_of(experiment, names, positions)
    words = [_powers(w) for w in check.get("words") or [check["word"]]]
    return np.column_stack([levels[:, : len(w)] @ np.asarray(w) % 3 for w in words])


def _component_blocks(experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check that a 3^k design's blocks are the groups of runs with one value of each defining contrast (mod 3).

    Confounding the components ``check["words"]`` with blocks puts two runs in the same block when
    they have the same values of their defining contrasts. ``check["block"]`` is the column of the
    printed blocks; the blocks agree when each holds one set of values and no two hold the same.
    With ``check["of"]``, a block's label, the check is of the values printed for that block alone:
    ``(0, 2)`` for L1 = 0 and L2 = 2.
    """
    runs = experiment["runs"]
    blocks = [str(row[runs["columns"].index(check["block"])]) for row in runs["rows"]]
    values: dict[str, set[tuple[int, ...]]] = {}
    for block, contrast in zip(blocks, map(tuple, _contrast_values(experiment, check).tolist()), strict=True):
        values.setdefault(block, set()).add(contrast)
    if "of" in check:
        found = values[str(check["of"])]
        wanted = tuple(int(n) for n in re.findall(r"\d+", printed))
        return ", ".join(map(str, sorted(found))), found == {wanted}
    single = all(len(v) == 1 for v in values.values())
    distinct = len({next(iter(v)) for v in values.values()}) == len(values)
    shown = "; ".join(f"{block}: {sorted(v)}" for block, v in values.items())
    return shown, single and distinct


def _component_fraction(experiment: dict, check: dict, _printed: str) -> tuple[str, bool]:
    """Check that the runs of a 3^(k-p) fraction share one value of each defining contrast of ``check["words"]``.

    ``check["u"]`` names the values, L = 0 for the principal fraction by default when given;
    ``check["runs"]`` keeps some of the runs, as one fraction of a table that prints all three.
    """
    values = {tuple(v) for v in _contrast_values(experiment, check, check.get("runs")).tolist()}
    wanted = check.get("u")
    same = len(values) == 1 and (wanted is None or values == {tuple(wanted)})
    return ", ".join(map(str, sorted(values))), same


def _component_values(_experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check the values of a defining contrast the source works out for treatment combinations, ``01: 2, 21: 1``.

    Each treatment combination is written with one digit per factor, its level 0, 1 or 2; the
    contrast is ``check["word"]``, whose exponents are the contrast's coefficients.
    """
    word = _powers(check["word"])
    pairs = re.findall(r"(\d+)\s*:\s*(\d+)", printed)
    found = {run: sum(int(x) * a for x, a in zip(run, word, strict=False)) % 3 for run, _ in pairs}
    return ", ".join(f"{run}: {v}" for run, v in found.items()), bool(pairs) and all(
        found[run] == int(v) for run, v in pairs
    )


def _mod3_sum(_experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check a sum of treatment combinations modulo 3, digit by digit, as ``101 + 021 = 122``.

    The runs of a principal block form a group under this addition; the other blocks are found by
    adding a run not in it to each of its runs.
    """
    *terms, result = re.findall(r"\d+", str(check.get("printed", printed)))
    found = "".join(str(sum(int(t[i]) for t in terms) % 3) for i in range(len(result)))
    return found, found == result


@functools.cache
def _entry_runs(directory: str, name: str) -> tuple[list[str], list[list[Any]]]:
    """Return the runs of the catalogue's entry ``name``: its columns and its rows."""
    for path in sorted(Path(directory).glob("*.json")):
        found = json.loads(path.read_text(encoding="utf-8"))
        for entry in [*found.get("experiments", []), *found.get("designs", [])]:
            if entry.get("name") == name:
                return entry["runs"]["columns"], entry["runs"]["rows"]
    raise KeyError(f"The catalogue has no entry {name!r}.")


def _projection(experiment: dict, check: dict, printed: str) -> tuple[str, bool]:
    """Check that printed columns are columns of another printed design, run by run: a projection of it.

    ``check["of"]`` names the other entry (this one by default) and ``check["columns"]`` its
    columns; ``check["into"]`` names this entry's columns they must equal, its factors by default.
    A design printed twice, as a table and again with a response, is the same check.
    """
    runs = experiment["runs"]
    columns, rows = (
        _entry_runs(str(_CATALOGUE_DIR), check["of"]) if check.get("of") else (runs["columns"], runs["rows"])
    )
    into = check.get("into") or [f["name"] for f in experiment["factors"]]
    other = pd.DataFrame(rows, columns=columns)[check["columns"]].astype(str).to_numpy()
    mine = pd.DataFrame(runs["rows"], columns=runs["columns"])[into].astype(str).to_numpy()
    if other.shape != mine.shape:
        return f"{other.shape[0]} runs and {other.shape[1]} columns against {mine.shape[0]} and {mine.shape[1]}", False
    differ = sorted({int(i) + 1 for i in np.nonzero(other != mine)[0]})
    return ("every run" if not differ else f"not in runs {', '.join(map(str, differ))}"), not differ


def _fraction_runs(experiment: dict, check: dict) -> list[int]:
    """Return the 1-based positions of the runs that make the fraction of a generator: C = AB, or C = \u2212AB.

    A run is in it when the product of its coded settings over the generator's word is the word's sign.
    Several generators, separated by commas, keep the runs that satisfy them all, as the signs of
    two effects pick one of the four blocks they generate: ``-ADE, -BCE``.
    """
    settings, _ = _coded_runs(experiment, check.get("runs"))
    kept = np.ones(len(settings), dtype=bool)
    for generator in str(check["generator"]).split(","):
        word = _generator_word(generator)
        sign = -1.0 if word.startswith("-") else 1.0
        kept &= np.isclose(settings[list(word.lstrip("-"))].prod(axis=1).to_numpy(), sign)
    return [int(i) + 1 for i in np.flatnonzero(kept)]


def _standard_order(experiment: dict, check: dict) -> list[int]:
    """Return each run's 1-based position in the standard order, the first factor changing fastest."""
    names = check.get("factors") or [f["name"] for f in experiment["factors"]]
    settings, _ = _coded_runs(experiment, check.get("runs"))
    high = (settings[names].to_numpy() > 0).astype(int)
    return [1 + int(row @ (2 ** np.arange(len(names)))) for row in high]


_RUN_KINDS: dict[str, Callable[[dict, dict], list[int]]] = {
    "fraction_runs": _fraction_runs,
    "standard_order": _standard_order,
}


_REDUCE: dict[str, Callable[[list[float]], float]] = {
    "max": max,
    "min": min,
    "mean": statistics.fmean,
    "len": len,
    "sum": math.fsum,
    "prod": math.prod,
}


def _design_metric(experiment: dict, check: dict) -> float:
    """Return a quality metric of a design: the entry's runs in coded units, or a design the library makes.

    ``check["metric"]`` names an :func:`~process_improve.experiments.evaluate_design` metric, evaluated
    for ``check["model"]`` with ``check["options"]``; ``check["key"]`` reads into its result, step by
    step (``"correlation"`` then ``"max_abs_r"``), where ``max``, ``min``, ``mean`` and ``len`` reduce
    a dict's values or a list. ``"reciprocal": true`` returns one over the number read, as a source
    prints a term's tolerance, one over its variance inflation factor; ``"complement": true`` then
    returns one less it, as the squared multiple correlation of a term with the others, 1 - 1/VIF.
    ``"squared": true`` returns its square, as trace(AA') is the square of the alias matrix's
    Frobenius norm.
    """
    if check.get("design"):
        source = json.dumps({"design": check["design"]}, sort_keys=True)
    else:
        settings = _coded_runs(experiment, check.get("runs"))[0]
        source = json.dumps({"columns": list(settings.columns), "runs": settings.to_dict("list")}, sort_keys=True)
    asked = json.dumps({k: check.get(k) for k in ("model", "metric", "options")}, sort_keys=True)
    value = _read(_evaluated(source, asked)[check["metric"]], check.get("key"))
    value = value**2 if check.get("squared") else value
    value = 1 / value if check.get("reciprocal") else value
    return 1 - value if check.get("complement") else value


@functools.cache
def _evaluated(source: str, asked: str) -> dict[str, Any]:
    """Return the library's evaluation of a design, worked once however many of its numbers are read.

    ``source`` holds the design a specification names, or the runs in coded units; ``asked`` the
    model, the metric and its options, both as JSON. An alias matrix is read entry by entry, so
    its hundreds of checks share one evaluation.
    """
    spec, wanted = json.loads(source), json.loads(asked)
    if "design" in spec:
        design: Any = _made(json.dumps(spec["design"], sort_keys=True))
    else:
        design = pd.DataFrame({name: spec["runs"][name] for name in spec["columns"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate_design(design, model=wanted["model"], metric=wanted["metric"], **(wanted["options"] or {}))


def _read(found: Any, key: str | list[str] | None) -> float:  # noqa: ANN401 - a library result: a number, dict or list
    """Read one number from a library result by a dot-separated path; ``max``, ``min``, ``mean``, ``len`` reduce.

    A path whose keys hold a dot themselves, such as the quantile ``"0.5"``, is given as a list of its steps.
    """
    steps = key if isinstance(key, list) else str(key or "").split(".")
    for step in filter(None, steps):
        if step in _REDUCE:
            found = _REDUCE[step](list(found.values()) if isinstance(found, dict) else list(found))
        else:
            found = found[int(step)] if isinstance(found, list) else found[step]
    return float(found)


_LIBRARY: dict[str, Callable[..., Any]] = {
    "blocking_alpha": blocking_alpha,
    "omars_minimum_runs": omars_minimum_runs,
    "optimize_responses": optimize_responses,
    "t_value": t_value,
    "t_value_cdf": t_value_cdf,
}


def _library_value(check: dict) -> float:
    """Return what a library function gives: ``check["function"]``, one of a few named here, on its arguments.

    For a number the library computes directly, such as the run count of the OMARS estimability
    frontier, without building a design to read it from, or the canonical analysis of a fitted
    equation a source prints without its data. ``"squared": true`` returns the square of the number read.
    """
    found = _read(_LIBRARY[check["function"]](*check.get("args", []), **check.get("kwargs", {})), check.get("key"))
    return found**2 if check.get("squared") else found


def _active(out: dict, check: dict) -> tuple[str, bool]:
    margin = "active_SME" if check.get("margin") == "SME" else "active_ME"
    found = {e["term"] for e in out["lenth_method"]["effects"] if e[margin]}
    wanted = {_library_term(t) for t in check["terms"]}
    same = {frozenset(t.split(":")) for t in found} == {frozenset(t.split(":")) for t in wanted}
    return ", ".join(sorted(found)), same


def _difference_of_means(experiment: dict, check: dict) -> float:
    """Return the average of the runs ``check["runs"]`` less the average of the runs ``check["minus"]``.

    For a difference a source works between two groups of runs that no factor of the entry marks,
    such as averages of adjacent stretches of a production record.
    """
    _, after = _coded_runs(experiment, check["runs"], check.get("summary"))
    _, before = _coded_runs(experiment, check["minus"], check.get("summary"))
    column = check.get("response") or after.columns[0]
    return float(after[column].mean() - before[column].mean())


def _t_test(experiment: dict, check: dict) -> float:
    """Return a number of the library's t-test of the runs ``check["runs"]`` against the runs ``check["minus"]``.

    ``check["test"]`` is ``"pooled"`` (Student's test, the two groups sharing one variance; the
    default), ``"welch"`` (each group with its own variance, on the Welch-Satterthwaite degrees of
    freedom) or ``"paired"`` (the differences of the two lists' runs, taken in the order listed).
    The difference is the average of ``runs`` less the average of ``minus``. ``check["key"]``
    names the number: ``"difference"``, ``"t"``, ``"p"``, ``"df"``, ``"std_error"`` (of the
    difference), ``"ci_low"``, ``"ci_high"`` or ``"half_width"`` of the interval at confidence
    ``check["confidence"]`` (0.95 by default; a one-sided 95% bound is an end of the two-sided
    90% interval), ``"pooled_std_dev"``, or for a paired test ``"std_dev"``, of the differences.
    ``check["alternative"]`` makes the p-value one-sided, as for a coefficient (see :func:`_p_value`).
    """
    _, after = _coded_runs(experiment, check["runs"])
    _, before = _coded_runs(experiment, check["minus"])
    column = check.get("response") or after.columns[0]
    runs, minus = after[column].astype(float), before[column].astype(float)
    confidence = check.get("confidence", 0.95)
    if check.get("test") == "paired":
        test = ttest_paired(runs - minus.to_numpy(), conflevel=confidence)
        found = {
            "difference": test["Differences mean"],
            "t": test["z value"],  # the library's paired test names its t statistic "z value"
            "std_error": test["Standard deviation"],  # and the standard error of the mean difference so
            "std_dev": test["Standard deviation"] * len(runs) ** 0.5,
        }
    else:
        test = ttest_independent(minus, runs, conflevel=confidence, equal_var=check.get("test") != "welch")
        found = {
            "difference": test["Group B average"] - test["Group A average"],
            "t": test["t value"],
            "std_error": test["Std error of difference"],
            "pooled_std_dev": test["Pooled std dev"],
        }
    low, high = test["ConfInt: Lo"], test["ConfInt: Hi"]
    found |= {
        "df": test["Degrees of freedom"],
        "ci_low": low,
        "ci_high": high,
        "half_width": (high - low) / 2,
        "p": _p_value({"coefficient": found["t"], "p_value": test["p value"]}, check.get("alternative")),
    }
    return float(found[check["key"]])


def _tukey(experiment: dict, check: dict) -> float:
    """Return a number of the library's Tukey HSD comparison of the level ``check["level"]`` less ``check["minus"]``.

    The levels are those of the factor ``check["factor"]``, compared with
    :func:`~process_improve.sensory.tukey_hsd` on the error mean square of the one-way model, or of
    the randomized block model when ``check["block"]`` names the blocks. ``check["key"]`` is
    ``"difference"``, ``"ci_low"`` or ``"ci_high"`` of the simultaneous interval at confidence
    1 - ``check["alpha"]`` (0.05 by default), its ``"half_width"``, the ``"std_error"`` it is a
    multiple of (the root of the error mean square over a level's number of runs), that multiple
    (``"q_critical"``, the upper percentage point of the studentized range), ``"q"`` or ``"p"``.
    """
    settings, responses = _coded_runs(experiment, check.get("runs"))
    column = check.get("response") or responses.columns[0]
    factor, block = check["factor"], check.get("block")
    panel = pd.DataFrame({"attribute": column, "score": responses[column].to_numpy(dtype=float)})
    for name in filter(None, (factor, block)):
        panel[name] = settings[name].astype(str).to_numpy()
    table = tukey_hsd(panel, factor=factor, block=block, alpha=check.get("alpha", 0.05))
    level, minus = str(check["level"]), str(check["minus"])
    pair = table[table["group1"].isin([level, minus]) & table["group2"].isin([level, minus])].iloc[0]
    sign = 1.0 if pair["group1"] == level else -1.0
    half = (pair["ci_high"] - pair["ci_low"]) / 2
    difference = sign * pair["meandiff"]
    return float(
        {
            "difference": difference,
            "ci_low": difference - half,
            "ci_high": difference + half,
            "half_width": half,
            "std_error": pair["se"],
            "q_critical": half / pair["se"],
            "q": pair["q_stat"],
            "p": pair["p_value"],
        }[check["key"]]
    )


def _nonadditivity(experiment: dict, check: dict) -> float:
    """Return a number of Tukey's one-degree-of-freedom test for nonadditivity of a two-factor layout.

    ``check["model"]`` is the additive model, such as ``temperature + pressure``, fitted with
    :func:`~process_improve.experiments.analyze_experiment` to one observation per cell. Tukey's
    test is the test of one more regressor, the square of that model's fitted values, so the model
    is fitted again with it. ``check["key"]`` is ``"ss"`` (the sum of squares for nonadditivity),
    ``"f"`` or ``"p"`` (its test), ``"ss_error"``, ``"df_error"`` or ``"ms_error"`` (the residual
    once it is taken out), or ``"f.<term>"`` and ``"p.<term>"``: a term's mean square in the
    additive model tested against that smaller error, as a source's table of the test prints it.
    """
    settings, responses = _coded_runs(experiment, check.get("runs"))
    y = responses[check.get("response") or responses.columns[0]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        additive = analyze_experiment(
            settings, y, model=check["model"], analysis_type=["anova", "residual_diagnostics"]
        )
        squared = np.square(additive["residual_diagnostics"]["fitted_values"])
        tested = analyze_experiment(
            settings.assign(nonadditivity=squared),
            y,
            model=f"{check['model']} + nonadditivity",
            analysis_type=["anova"],
        )
    row = _anova_row(tested, "nonadditivity")
    summary = tested["model_summary"]
    kind, _, term = check["key"].partition(".")
    if term:
        term_row = _anova_row(additive, term)
        ratio = term_row["mean_sq"] / summary["mse_residual"]
        return ratio if kind == "f" else float(f_distribution.sf(ratio, term_row["df"], summary["df_residual"]))
    found = {
        "ss": row["sum_sq"],
        "f": row["F"],
        "p": row["p_value"],
        "ss_error": summary["mse_residual"] * summary["df_residual"],
        "df_error": summary["df_residual"],
        "ms_error": summary["mse_residual"],
    }
    return float(found[kind])


def _capability(experiment: dict, check: dict) -> float:
    """Return the library's Cpk of a response: the nearer specification's distance from the mean, over 3 sigma.

    ``check["lower"]`` and ``check["upper"]`` are the specifications, either left out when a source
    gives only one. The mean and standard deviation are the plain ones a source works by hand.
    """
    _, responses = _coded_runs(experiment, check.get("runs"))
    column = check.get("response") or responses.columns[0]
    lower, upper = (np.nan if check.get(side) is None else float(check[side]) for side in ("lower", "upper"))
    specifications = (lower, upper)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the side with no specification warns that it has no spread
        return float(calculate_cpk(responses, column, specifications=specifications, trim_percentile=0).cpk)


def _log_variance_ratio(experiment: dict, check: dict) -> float:
    """Return ln(s+^2 / s-^2), twice the log of a statistic's average at ``check["runs"]`` over ``check["minus"]``.

    The statistic is a run's standard deviation (``check["summary"]``) or a printed column
    (``check["response"]``); a source compares the two levels of a factor this way to find a
    dispersion effect.
    """
    _, plus = _coded_runs(experiment, check["runs"], check.get("summary"))
    _, minus = _coded_runs(experiment, check["minus"], check.get("summary"))
    column = check.get("response") or plus.columns[0]
    return 2 * math.log(float(plus[column].mean()) / float(minus[column].mean()))


def _nested_test(experiment: dict, check: dict) -> float:
    """Return a number of the F test of ``check["model"]`` against the fuller ``check["full"]``, on the same runs.

    The sum of squares is the smaller model's residual sum of squares less the fuller one's, on the
    difference of their residual degrees of freedom, and it is tested against the fuller model's
    residual mean square. With the fuller model saturated in the design's factors, this is the test
    of lack of fit against pure error from the replicated runs alone, as Design-Expert works it when
    a factor dropped from the model stays a factor of the design; with the fuller model's terms all
    but one tested together, it is the test of those terms. ``check["error"]`` names a third model
    whose residual mean square is the denominator instead, as Design-Expert tests the linear blending
    of a mixture against the residual of the quadratic model it reports. ``check["key"]`` is
    ``"ss"``, ``"df"``, ``"ms"``, ``"f"`` or ``"p"``.
    """
    settings, responses = _coded_runs(experiment, check.get("runs"), check.get("summary"))
    y = responses[check.get("response") or responses.columns[0]]
    data = json.dumps({"columns": list(settings.columns), "runs": settings.to_dict("list"), "y": y.tolist()})
    (ss_smaller, df_smaller), (ss_fuller, df_fuller) = (_residual_sum(data, m) for m in (check["model"], check["full"]))
    ss_error, df_error = _residual_sum(data, check["error"]) if check.get("error") else (ss_fuller, df_fuller)
    ss, df = ss_smaller - ss_fuller, df_smaller - df_fuller
    f = ss / df / (ss_error / df_error)
    found = {"ss": ss, "df": df, "ms": ss / df, "f": f, "p": f_distribution.sf(f, df, df_error)}
    return float(found[check["key"]])


@functools.cache
def _residual_sum(data: str, model: str) -> tuple[float, int]:
    """Return the residual sum of squares and degrees of freedom of the library's fit of a model to runs given as JSON.

    A stepwise table tests scores of terms against one current model, so each model is fitted once.
    """
    runs = json.loads(data)
    settings = pd.DataFrame({name: runs["runs"][name] for name in runs["columns"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = analyze_experiment(settings, pd.Series(runs["y"], name="y"), model=model, analysis_type=["anova"])[
            "model_summary"
        ]
    return summary["mse_residual"] * summary["df_residual"], summary["df_residual"]


def _nested_test_rounding(experiment: dict, check: dict) -> float:
    """Return how far a number of :func:`_nested_test` moves when each response is off by half its last printed digit.

    The first-order bound, for a source that fitted readings it prints rounded (``check["data_rounding"]``,
    as for :func:`_carried_rounding`). With e and r the residuals of the fuller and the smaller model,
    the sum of squares r'r - e'e has gradient 2(r - e); the F ratio (SS/df)/(E/d) with E = e'e has
    gradient 2(r - e)/df/(E/d) - 2Fe/E, and its p-value minus the F density times that.
    """
    settings, responses = _coded_runs(experiment, check.get("runs"), check.get("summary"))
    y = responses[check.get("response") or responses.columns[0]].to_numpy(dtype=float)
    smaller, fuller = (dmatrix(check[m], settings, return_type="dataframe").to_numpy() for m in ("model", "full"))
    r, e = _residual_vector(smaller, y), _residual_vector(fuller, y)
    df_full = len(y) - np.linalg.matrix_rank(fuller)
    df = np.linalg.matrix_rank(fuller) - np.linalg.matrix_rank(smaller)
    ss, error = float(r @ r - e @ e), float(e @ e)
    f = ss / df / (error / df_full)
    gradient_f = 2 * (r - e) / df / (error / df_full) - 2 * f * e / error
    gradients = {
        "ss": 2 * (r - e),
        "ms": 2 * (r - e) / df,
        "f": gradient_f,
        "p": -f_distribution.pdf(f, df, df_full) * gradient_f,
        "df": np.zeros(len(y)),
    }
    return float(np.abs(gradients[check["key"]]) @ _half_units(y, check["data_rounding"]))


def _column_correlation(experiment: dict, check: dict) -> float:
    """Return the correlation of two coded columns ``check["columns"]`` over the runs ``check["runs"]``.

    Two columns of a two-level factorial are uncorrelated over all its runs; with a run missing they
    are not. ``"absolute": true`` returns the size, for a source that prints the correlation as ±r.
    """
    settings, _ = _coded_runs(experiment, check.get("runs"))
    first, second = (settings[name].to_numpy(dtype=float) for name in check["columns"])
    r = float(np.corrcoef(first, second)[0, 1])
    return abs(r) if check.get("absolute") else r


def _cross_products(experiment: dict, check: dict) -> float:
    """Return a sum of squares or cross products of two printed columns, about their means: an analysis of covariance's.

    ``check["columns"]`` names the two columns (the same one twice for a sum of squares), and
    ``check["factor"]`` the treatments; ``check["key"]`` is ``"total"``, S, about the grand means,
    ``"treatment"``, T, of the treatment means about them, or ``"error"``, E = S - T, about each
    treatment's means.
    """
    table = pd.DataFrame(experiment["runs"]["rows"], columns=experiment["runs"]["columns"])
    first, second = (table[name].astype(float) for name in check["columns"])
    total = float(((first - first.mean()) * (second - second.mean())).sum())
    groups = table[check["factor"]].astype(str)
    within = float(
        ((first - first.groupby(groups).transform("mean")) * (second - second.groupby(groups).transform("mean"))).sum()
    )
    return {"total": total, "treatment": total - within, "error": within}[check["key"]]


def _dot_product(experiment: dict, check: dict) -> float:
    """Return the sum over the runs of the product of the printed columns ``check["columns"]``: zero when orthogonal.

    ``check["runs"]`` keeps the runs at these 1-based positions, as the runs of a design before it is augmented.
    """
    return float(_printed_matrix(experiment, check["columns"], None, check.get("runs")).prod(axis=1).sum())


def _normal_equations(experiment: dict, check: dict) -> float:
    """Return an entry of the least squares normal equations of ``check["model"]``: of X'X, of X'y, or of (X'X)^-1.

    ``check["key"]`` is ``"xtx.<i>.<j>"``, ``"xty.<i>"`` or ``"inverse.<i>.<j>"``, the indices
    0-based in the order of the model matrix's columns, the intercept first. The factors are in
    coded units, or their own for a factor marked ``"coding": "none"``, as for a fit.

    A design is compared on its inverse as a whole: ``"det_inverse"``, |(X'X)^-1|, the D criterion;
    ``"trace_inverse"``, the A criterion; ``"leverage"``, x0'(X'X)^-1 x0 at the coded point
    ``check["point"]``, the variance of the prediction there in units of σ² (JMP's prediction
    variance profile); and ``"relative_d"``, the D-efficiency of these runs relative to the runs
    ``check["versus"]``, (|(X2'X2)^-1| / |(X1'X1)^-1|)^(1/p) for p model terms.
    """
    settings, responses = _coded_runs(experiment, check.get("runs"))
    model = dmatrix(check["model"], settings, return_type="dataframe")
    matrix = model.to_numpy()
    kind, *place = check["key"].split(".")
    if kind == "relative_d":
        other = dmatrix(check["model"], _coded_runs(experiment, check["versus"])[0], return_type="dataframe").to_numpy()
        ratio = np.linalg.det(np.linalg.inv(other.T @ other)) / np.linalg.det(np.linalg.inv(matrix.T @ matrix))
        return float(ratio ** (1 / matrix.shape[1]))
    if kind == "leverage":
        point = pd.DataFrame([dict.fromkeys(settings.columns, 0.0) | check["point"]])[list(settings.columns)]
        at = np.asarray(build_design_matrices([model.design_info], point)[0])[0]
        return float(at @ np.linalg.inv(matrix.T @ matrix) @ at)
    found = {
        "xtx": lambda: matrix.T @ matrix,
        "xty": lambda: matrix.T @ responses[check.get("response") or responses.columns[0]].to_numpy(dtype=float),
        "inverse": lambda: np.linalg.inv(matrix.T @ matrix),
        "det_inverse": lambda: np.linalg.det(np.linalg.inv(matrix.T @ matrix)),
        "trace_inverse": lambda: np.trace(np.linalg.inv(matrix.T @ matrix)),
    }
    return float(found[kind]()[tuple(int(i) for i in place)])


def _jmp_coded(settings: pd.DataFrame, experiment: dict, names: list[str]) -> pd.DataFrame:
    """Return the settings with categorical factors coded as JMP codes them to evaluate a design.

    A factor of k levels becomes k - 1 columns, named for it and numbered (A1, A2): its effect
    coding (level i is 1 in column i, the last level -1 in each) made orthonormal column by column,
    then scaled so each column's squares average 1 over the levels. For three levels these are the
    linear and quadratic contrasts. JMP reports relative variances, and the tables of a source that
    prints them, for this coding.
    """
    coded = settings.drop(columns=names)
    for name in names:
        levels = [str(level) for level in next(f["levels"] for f in experiment["factors"] if f["name"] == name)]
        k = len(levels)
        basis = np.linalg.qr(np.vstack([np.eye(k - 1), -np.ones(k - 1)]))[0] * math.sqrt(k)
        place = settings[name].astype(str).map(levels.index).to_numpy()
        for j in range(k - 1):
            coded[f"{name}{j + 1}"] = basis[place, j]
    return coded


def _unit_std_error(experiment: dict, check: dict) -> float:
    """Return the standard error of a model's coefficients in units of sigma: the root of the diagonal of (X'X)^-1.

    A property of the runs and ``check["model"]``, not of any response: a source compares designs
    by it, as an orthogonal design's coefficients all have 1/sqrt(N). ``check["exclude"]`` names
    factors whose columns are left out (a block's, say), and ``check["key"]`` reduces the rest:
    ``"min"`` or ``"max"``; ``check["term"]`` reads one column's instead. ``"variance": true``
    returns the square, the relative variance a source prints. The runs are the entry's, or those
    of the design ``check["design"]`` names; ``check["jmp_coded"]`` names categorical factors coded
    as :func:`_jmp_coded` codes them.
    """
    if check.get("design"):
        made = _made(json.dumps(check["design"], sort_keys=True))
        settings = made.design[made.factor_names].astype(float)
    else:
        settings, _ = _coded_runs(experiment, check.get("runs"))
    if check.get("jmp_coded"):
        settings = _jmp_coded(settings, experiment, check["jmp_coded"])
    matrix = dmatrix(check["model"], settings, return_type="dataframe")
    errors = np.sqrt(np.diag(np.linalg.pinv(matrix.to_numpy().T @ matrix.to_numpy())))
    errors = errors**2 if check.get("variance") else errors
    if "term" in check:
        return float(errors[list(matrix.columns).index(check["term"])])
    excluded = tuple(check.get("exclude", []))
    kept = [e for name, e in zip(matrix.columns, errors, strict=True) if not name.startswith(excluded)]
    return float(_REDUCE[check.get("key", "max")](kept))


def _distinct_runs(experiment: dict, check: dict) -> float:
    """Return how many different runs the design has in the printed columns ``check["columns"]``: its projection.

    ``check["runs"]`` keeps the runs at these 1-based positions. A level may be printed as a name.
    """
    runs = experiment["runs"]
    table = pd.DataFrame(runs["rows"], columns=runs["columns"])[check["columns"]]
    if check.get("runs"):
        table = table.iloc[[p - 1 for p in check["runs"]]]
    return float(len(set(map(tuple, table.astype(str).to_numpy().tolist()))))


def _run_count(experiment: dict, check: dict) -> float:
    """Return how many runs have the settings ``check["at"]``, a column and its printed level each."""
    runs = experiment["runs"]
    table = pd.DataFrame(runs["rows"], columns=runs["columns"]).astype(str)
    kept = np.ones(len(table), dtype=bool)
    for column, level in check["at"].items():
        kept &= (table[column] == str(level)).to_numpy()
    return float(kept.sum())


def _component_ss(experiment: dict, check: dict) -> float:
    """Return a 3^k component of interaction: its sum of squares, or a total of the runs in one of its three groups.

    The component ``check["word"]``, AB²C say, groups the runs by its defining contrast
    L = x1 + 2 x2 + x3 (mod 3), each x a factor's level 0, 1 or 2; its sum of squares is that of the
    three group totals, sum T_u^2 / n_u - T^2 / N, on two degrees of freedom. ``check["key"]`` is
    ``"ss"`` (the default), ``"total.<u>"`` or ``"count.<u>"`` for the group with L = u.
    ``check["runs"]`` keeps some runs, as the totals of one level of a third factor.
    """
    positions = check.get("runs")
    contrast = _contrast_values(experiment, check, positions)[:, 0]
    _, responses = _coded_runs(experiment, positions)
    y = responses[check.get("response") or responses.columns[0]].to_numpy(dtype=float)
    totals = [float(y[contrast == u].sum()) for u in range(3)]
    counts = [int((contrast == u).sum()) for u in range(3)]
    kind, _, group = check.get("key", "ss").partition(".")
    if kind == "total":
        return totals[int(group)]
    if kind == "count":
        return float(counts[int(group)])
    return sum(t * t / n for t, n in zip(totals, counts, strict=True) if n) - y.sum() ** 2 / len(y)


def _design_columns(experiment: dict, check: dict) -> np.ndarray:
    """Return a two-level design's columns: the entry's ``check["columns"]``, or a design ``check["design"]`` names."""
    if check.get("design"):
        made = _made(json.dumps(check["design"], sort_keys=True))
        return made.design[made.factor_names].to_numpy(dtype=float)
    return _printed_matrix(experiment, check.get("columns") or [f["name"] for f in experiment["factors"]], None)


def _interaction_columns(experiment: dict, check: dict) -> float:
    """Return a measure of how a two-level design's main-effect and two-factor interaction columns overlap.

    The columns are the k factors' and their k(k - 1)/2 products, without the intercept, as a
    screening design is compared on them. ``check["key"]`` is one of:

    - ``"e_s2"``: Booth and Cox's E(s²), the average over the pairs of columns of the square of
      their inner product s = Xi'Xj;
    - ``"confounded_pairs"``: the pairs of columns that are equal or opposite, completely confounded;
    - ``"max_abs_r"``: the largest correlation of two of the columns, in size, and
      ``"max_abs_r_main_interaction"`` that of a main effect with a two-factor interaction;
    - ``"max_abs_r_main_square"``: the largest correlation of a main effect with a factor's square,
      for a design with factors at three levels.
    """
    matrix = _design_columns(experiment, check)
    k = matrix.shape[1]
    products = [matrix[:, i] * matrix[:, j] for i, j in itertools.combinations(range(k), 2)]
    columns = np.column_stack([matrix, *products])
    upper = np.triu_indices(columns.shape[1], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nan_to_num(np.corrcoef(columns, rowvar=False))
    found = {
        "e_s2": lambda: float(np.mean((columns.T @ columns)[upper] ** 2)),
        "confounded_pairs": lambda: float(np.sum(np.isclose(np.abs(r[upper]), 1))),
        "max_abs_r": lambda: float(np.abs(r[upper]).max()),
        "max_abs_r_main_interaction": lambda: float(np.abs(r[:k, k:]).max()),
        "max_abs_r_main_square": lambda: float(
            np.abs(np.nan_to_num(np.corrcoef(matrix, matrix**2, rowvar=False))[:k, k:]).max()
        ),
    }
    return found[check["key"]]()


def _estimable_projections(experiment: dict, check: dict) -> float:
    """Return how many projections of a design onto ``check["order"]`` columns cannot fit main effects and interactions.

    A design projects onto any r of its columns; the model of their main effects and two-factor
    interactions is estimable in a projection when its model matrix has full column rank. A source
    that says a design supports that model in any r factors claims this count is 0. With
    ``"squares": true`` the model is the full quadratic in the r factors, their squares added, as a
    definitive screening design is said to support in any three of its factors.
    """
    matrix = _design_columns(experiment, check)
    failing = 0
    for subset in itertools.combinations(range(matrix.shape[1]), check["order"]):
        part = matrix[:, subset]
        products = [part[:, i] * part[:, j] for i, j in itertools.combinations(range(len(subset)), 2)]
        squares = [part**2] if check.get("squares") else []
        model = np.column_stack([np.ones(len(part)), part, *products, *squares])
        failing += int(np.linalg.matrix_rank(model) < model.shape[1])
    return float(failing)


def _fold_over_pairs(experiment: dict, check: dict) -> float:
    """Return how many of the pairs of runs 1 and 2, 3 and 4, and so on are not a run and its fold-over.

    A definitive screening design is made of such pairs, each run the other with every sign changed,
    and a centre run: a source that says so claims this count is 0. ``check["columns"]`` names the
    factors (all the entry's by default) and ``check["pairs"]`` the number of pairs.
    """
    columns = check.get("columns") or [f["name"] for f in experiment["factors"]]
    matrix = _printed_matrix(experiment, columns, None)
    return float(sum(not np.allclose(matrix[2 * i], -matrix[2 * i + 1]) for i in range(check["pairs"])))


def _run_radius(experiment: dict, check: dict) -> float:
    """Return how far runs lie from the centre of the design, in coded units: ``check["key"]`` ``"max"`` or ``"min"``.

    ``check["runs"]`` keeps the runs at these 1-based positions, as the runs off the centre of a
    spherical design, which all lie at one distance. A categorical factor, such as a block, is left out.
    """
    settings, _ = _coded_runs(experiment, check.get("runs"))
    numeric = [f["name"] for f in experiment["factors"] if f["type"] == "numeric"]
    radii = np.sqrt((settings[numeric].to_numpy(dtype=float) ** 2).sum(axis=1))
    return float(_REDUCE[check.get("key", "max")](radii.tolist()))


def _blocks_orthogonal(experiment: dict, check: dict) -> float:
    """Return 1 when a second-order design's blocks are orthogonal to its model, 0 when they are not.

    The two conditions: each block is a first-order orthogonal design, every factor's column and
    every product of two summing to zero within it; and each block holds the same share of every
    factor's sum of squares as of the runs, which the library's
    :func:`~process_improve.experiments._blocking.squares_orthogonal_to_blocks` tests.
    ``check["block"]`` names the column of block labels.
    """
    settings, _ = _coded_runs(experiment, None)
    runs = experiment["runs"]
    labels = pd.DataFrame(runs["rows"], columns=runs["columns"])[check["block"]].astype(str).to_numpy()
    design = settings.drop(columns=[check["block"]], errors="ignore").to_numpy(dtype=float)
    for label in np.unique(labels):
        block = design[labels == label]
        if not (np.allclose(block.sum(axis=0), 0) and np.allclose(np.triu(block.T @ block, 1), 0)):
            return 0.0
    return float(squares_orthogonal_to_blocks(design, labels))


def _desirability(experiment: dict, check: dict) -> float:
    """Return the Derringer and Suich desirability of the fitted responses at a coded point ``check["point"]``.

    ``check["goals"]`` lists each response's goal as :func:`optimize_responses` takes it, with the
    response's ``model`` (and ``"settings"`` as for a fit): the prediction of that model at the point
    is scored by the library's individual desirability, and ``check["key"]`` is ``"composite"``,
    their geometric mean, or ``"d.<i>"`` for the 0-based i-th response's own.
    """
    fits = _Fits()
    scores = []
    for goal in check["goals"]:
        out = fits(experiment, goal)
        predicted = _prediction(out, {"model": goal["model"], "point": check["point"]}, experiment)
        scores.append(individual_desirability(predicted, goal))
    kind, _, which = check.get("key", "composite").partition(".")
    return float(composite_desirability(scores)) if kind == "composite" else float(scores[int(which)])


_Z_975 = 1.959963984540054  # the standard normal's 97.5th percentile, for a 95% Wald interval


@functools.cache
def _experiments_by_name() -> dict[str, dict[str, Any]]:
    return {e["name"]: e for f in _files() for e in f["experiments"]}


def _marginal_means(experiment: dict, check: dict) -> float:
    """Return the library's prediction from marginal means at a run's settings, or the run's deviation from it.

    The means are of ``check["response"]`` over the runs of the entry named ``check["from"]`` (this
    one when left out), for the factors ``check["factors"]`` and the pairs ``check["interactions"]``,
    as :func:`~process_improve.experiments.marginal_means_prediction` works them. With
    ``check["deviation"]`` the run's own value, ``check["observed"]`` (a column, or the mean of a list
    of columns), is compared: ``"difference"`` is it less the prediction, ``"percent"`` that over it,
    times 100.
    """
    source = _experiments_by_name()[check["from"]] if "from" in check else experiment
    runs, here = (pd.DataFrame(e["runs"]["rows"], columns=e["runs"]["columns"]) for e in (source, experiment))
    point = here.iloc[[check["run"] - 1]][check["factors"]]
    pairs = [(first, second) for first, second in check.get("interactions", [])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a design not balanced in pairs is warned of; a source may use one anyway
        predicted = float(marginal_means_prediction(runs, runs[check["response"]], point, pairs).iloc[0])
    if "deviation" not in check:
        return predicted
    observed = float(np.mean(here.iloc[check["run"] - 1][check["observed"]]))
    difference = observed - predicted
    return difference if check["deviation"] == "difference" else 100 * difference / observed


def _variance_component(experiment: dict, check: dict) -> float:
    """Return a number of the library's REML fit with a random intercept for each of ``check["groups"]``.

    The fixed part is the overall mean, and an effect for each level of the factors
    ``check["fixed"]`` names, if any, coded to sum to zero over the levels; ``"clutter:filter"``
    there is the interaction of two of them, the products of their columns. Nested groups are
    written with labels unique across the whole study; a group written ``"part:operator"`` is each
    combination of the two, the random interaction of two random factors. ``check["key"]`` names a
    group, for its variance, or one of:

    - ``"residual"``, the error variance, and ``"total"``, the sum of every variance;
    - ``"ratio.<group>"``, a group's variance over the residual's, and ``"percent.<part>"``, a group's
      or the residual's share of the total;
    - ``"std_error.<part>"`` and ``"covariance.<part>.<part>"``, from the asymptotic covariance of
      the variance estimates, and ``"wald_low.<part>"`` and ``"wald_high.<part>"``, the estimate less
      or plus 1.96 standard errors, the 95% interval JMP prints for a random effect: the normal's
      97.5th percentile, or ``check["z"]``, as some versions of JMP take 1.96 itself;
    - ``"rmse"``, the root of the residual variance, and ``"reml_criterion"``, minus twice the
      restricted log-likelihood, JMP's "-2 LogLikelihood";
    - ``"r_squared"`` and ``"r_squared_adj"``, JMP's R-square of the fit, from the fitted values with
      the random effects predicted (their best linear unbiased predictions), the adjusted one on the
      count of fixed parameters;
    - ``"mean"``, ``"mean_std_error"``, ``"mean_t"`` and ``"mean_df"``, the overall mean of an
      intercept-only fit, its standard error, their ratio and its Satterthwaite degrees of freedom;
    - ``"f.<factor>"``, ``"f_df.<factor>"`` and ``"f_p.<factor>"``, the F test of a fixed factor, on
      Satterthwaite's denominator degrees of freedom;
    - ``"effect.<factor>.<level>"``, the generalised least-squares estimate of a level's effect, which
      with random blocks of an incomplete block design combines the intra- and interblock estimates.

    ``"bounded": false`` lets a component go negative, as JMP's unbounded REML does (lme4's, and the
    default, keeps each at or above zero).
    """
    runs = experiment["runs"]
    table = pd.DataFrame(runs["rows"], columns=runs["columns"])
    y = table[check.get("response") or experiment["responses"][0]["name"]].to_numpy(dtype=float)
    fixed = [_fixed_columns(table, name) for name in check.get("fixed", [])]
    design = pd.concat([pd.DataFrame({"Intercept": np.ones(len(y))}), *fixed], axis=1).astype(float)
    groups = [pd.factorize(table[name.split(":")].astype(str).agg(":".join, axis=1))[0] for name in check["groups"]]
    fit = fit_variance_components(y, design.to_numpy(), groups, bounded=check.get("bounded", True))
    parts = dict(zip(check["groups"], map(float, fit.variances), strict=True)) | {"residual": float(fit.sigma2)}
    estimated = [name for name in parts if name == "residual" or parts[name] != 0]  # vc_cov's rows, in order
    r_squared = 1 - float(np.sum((y - fit.fitted) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    if check["key"] in parts:
        return parts[check["key"]]
    kind, _, name = check["key"].partition(".")
    total = sum(parts.values())
    z = float(check.get("z", _Z_975))

    def covariance(first: str, second: str) -> float:
        return float(fit.vc_cov[estimated.index(first), estimated.index(second)])

    def fixed_columns(factor: str) -> list[int]:
        return [i for i, column in enumerate(design.columns) if column.startswith(f"{factor}_")]

    def name_df(factor: str) -> int:
        return len(fixed_columns(factor))

    def fixed_test(index: int) -> float:
        return fit.f_test(np.eye(len(design.columns))[fixed_columns(name)])[index]

    def level_effect(factor: str, level: str) -> float:
        if level not in set(table[factor].astype(str)):
            raise KeyError(f"{factor} has no level {level!r}.")
        columns = {column: i for i, column in enumerate(design.columns)}
        if f"{factor}_{level}" in columns:
            return float(fit.beta[columns[f"{factor}_{level}"]])
        return -float(sum(fit.beta[i] for i in fixed_columns(factor)))  # the level the coding leaves out

    found: dict[str, Callable[[], float]] = {
        "total": lambda: total,
        "ratio": lambda: parts[name] / parts["residual"],
        "percent": lambda: 100 * parts[name] / total,
        "covariance": lambda: covariance(*name.split(".")),
        "std_error": lambda: math.sqrt(covariance(name, name)),
        "wald_low": lambda: parts[name] - z * math.sqrt(covariance(name, name)),
        "wald_high": lambda: parts[name] + z * math.sqrt(covariance(name, name)),
        "mean": lambda: float(fit.beta[0]),
        "r_squared": lambda: r_squared,
        "r_squared_adj": lambda: 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - len(design.columns)),
        "mean_std_error": lambda: math.sqrt(fit.cov_beta[0, 0]),
        "mean_df": lambda: fit.satterthwaite_df(np.eye(len(fit.beta))[0]),
        "mean_t": lambda: float(fit.beta[0]) / math.sqrt(fit.cov_beta[0, 0]),
        "rmse": lambda: math.sqrt(parts["residual"]),
        "reml_criterion": lambda: fit.reml_criterion,
        "f": lambda: fixed_test(0),
        "f_df": lambda: fixed_test(1),
        "f_p": lambda: float(f_distribution.sf(fixed_test(0), int(name_df(name)), fixed_test(1))),
        "effect": lambda: level_effect(*name.split(".", 1)),
    }
    return float(found[kind]())


def _expected_mean_squares(_experiment: dict, check: dict) -> float:
    """Return a coefficient of the library's expected mean squares of a balanced design, or a term's error term.

    ``check["levels"]``, ``"random"``, ``"nested"``, ``"replicates"``, ``"restricted"`` and ``"terms"`` go
    to :func:`~process_improve.experiments.random_effects.expected_mean_squares`. ``check["key"]`` is
    ``"<term>.<component>"``, the coefficient of a component (0 if it is absent; the term's own name
    for its fixed quadratic form, ``"Error"`` for the error variance); ``"per_df.<term>"``, a fixed
    term's coefficient over its degrees of freedom, the multiple of the sum of squared effects that
    Montgomery writes; ``"form.<term>.<other>"``, 1 if the fixed term's quadratic form takes in
    ``other``'s, as Minitab's ``Q[1,4]``; or ``"error_term.<term>"``, the 1-based place, in the
    terms' order with the error last, of the term that tests it, as Minitab numbers it.
    """
    nested = check.get("nested") or {}
    found = expected_mean_squares(
        {f"{name}({nested[name]})" if name in nested else name: n for name, n in check["levels"].items()},
        random=check.get("random", ()),
        replicates=check.get("replicates", 1),
        restricted=check.get("restricted", True),
        terms=check.get("terms"),
    )
    kind, _, rest = check["key"].partition(".")
    if kind == "per_df":
        return found["ems"][rest][rest] / found["df"][rest]
    if kind == "form":
        term, _, other = rest.partition(".")
        return float(other in found["fixed_forms"].get(term, []))
    if kind == "error_term":
        denominator = found["error_term"][rest]
        return float("nan") if denominator is None else float(found["terms"].index(denominator) + 1)
    return float(found["ems"][kind].get(rest, 0))


_VALUE_KINDS: dict[str, Callable[[dict, dict], float]] = {
    "blocks_orthogonal": _blocks_orthogonal,
    "capability": _capability,
    "desirability": _desirability,
    "column_correlation": _column_correlation,
    "component_ss": _component_ss,
    "difference_of_means": _difference_of_means,
    "distinct_runs": _distinct_runs,
    "cross_products": _cross_products,
    "dot_product": _dot_product,
    "estimable_projections": _estimable_projections,
    "fold_over_pairs": _fold_over_pairs,
    "interaction_columns": _interaction_columns,
    "log_variance_ratio": _log_variance_ratio,
    "normal_equations": _normal_equations,
    "design_property": lambda _e, c: _design_property(c),
    "design_metric": _design_metric,
    "library_value": lambda _e, c: _library_value(c),
    "marginal_means": _marginal_means,
    "nested_test": _nested_test,
    "nonadditivity": _nonadditivity,
    "run_count": _run_count,
    "run_radius": _run_radius,
    "t_test": _t_test,
    "tukey": _tukey,
    "unit_std_error": _unit_std_error,
    "variance_component": _variance_component,
    "expected_mean_squares": _expected_mean_squares,
}


_STRUCTURE_KINDS: dict[str, Callable[[dict, dict, str], tuple[str, bool]]] = {
    "block_confounding": lambda _e, c, printed: _block_confounding(c, printed),
    "component_blocks": _component_blocks,
    "component_fraction": _component_fraction,
    "component_product": _component_product,
    "component_relation": _component_relation,
    "component_values": _component_values,
    "mod3_sum": _mod3_sum,
    "projection": _projection,
    "same_runs": _same_runs,
    "generator_holds": _generator_holds,
    "generalized_interactions": _generalized_interactions,
    "product_column": _product_column,
    "same_projections": _same_projections,
}


def _reproduce(experiment: dict, expected: dict, check: dict, fits: _Fits) -> tuple[str, bool]:
    """Recompute one printed number by its check; return the value found, as text, and whether it agrees."""
    kind = check["kind"]
    printed = check.get("printed", expected["value"])
    if kind in _DESIGN_KINDS:
        return _design_check(check, str(printed), len(experiment.get("factors") or []))
    if kind in _STRUCTURE_KINDS:
        return _STRUCTURE_KINDS[kind](experiment, check, str(printed))
    if kind in _VALUE_KINDS:
        found = _VALUE_KINDS[kind](experiment, check)
        wanted = str(_ROMAN.get(str(printed).strip(), printed))  # a resolution may be printed as a Roman numeral
        of_rounded, decimals = check.get("of_rounded", 1), check.get("decimals")
        nearest = found
        if kind == "nested_test" and check.get("data_rounding"):  # the source's test of its unrounded data
            target, slack = float(_number(wanted)[1]), _nested_test_rounding(experiment, check)
            nearest = target if abs(found - target) <= slack else found - math.copysign(slack, found - target)
        return f"{found:.6g}", _agrees(
            nearest, wanted, expression=bool(check.get("expression")), of_rounded=of_rounded, decimals=decimals
        )
    if kind in _RUN_KINDS:
        found_runs = _RUN_KINDS[kind](experiment, check)
        listed = [int(n) for n in re.findall(r"\d+", str(printed))]
        same = set(listed) == set(found_runs) if kind == "fraction_runs" else listed == found_runs
        return ", ".join(map(str, found_runs)), same and len(listed) == len(found_runs)
    out = fits(experiment, check)
    if kind == "active_effects":
        return _active(out, check)
    if kind not in _FIT_KINDS:
        raise KeyError(f"No check of the kind {kind!r}.")
    found = float(_FIT_KINDS[kind](out, check, experiment))
    found = 1 - found if check.get("complement") else found  # a confidence 1 - p, as some software prints it
    found = math.exp(found) if check.get("exp") else found  # a model of ln(s²) read back as a variance
    found = found / math.log(10) if check.get("log10") else found  # a natural log printed as a common one
    if check.get("back_transform") == "freeman_tukey_sqrt":  # t = [√y + √(y + 1)]/2 read back as y
        found = (found - 1 / (4 * found)) ** 2
    expression, of_rounded = bool(check.get("expression")), check.get("of_rounded", 1)
    nearest = found
    if check.get("data_rounding"):  # the source's fit may sit this far from one to its data as printed, rounded
        target, slack = float(_number(str(printed))[1]), _carried_rounding(out, check)
        nearest = target if abs(found - target) <= slack else found - math.copysign(slack, found - target)
    return f"{found:.6g}", _agrees(
        nearest, str(printed), expression=expression, of_rounded=of_rounded, decimals=check.get("decimals")
    )


def _checks(expected: dict) -> list[dict]:
    check = expected.get("check") or {}
    return check if isinstance(check, list) else [check]


def _each_check(source: str | None = None) -> Iterator[tuple[str, dict, dict, dict]]:
    """Yield each check of the reference catalogue, with its source's key, its entry and the number it checks."""
    for f in _selected(source):
        for entry in [*f["experiments"], *f.get("designs", [])]:
            for expected in entry["expected"]:
                for check in _checks(expected):
                    yield f["source"]["key"], entry, expected, check


def reproduce_reference_catalogue(source: str | None = None) -> pd.DataFrame:
    """Recompute every printed number of the reference catalogue with this library.

    Each number with a check is recomputed: the experiment's runs are fitted with the model the
    check names (on the runs it names), and the result compared with the printed text to within
    half a unit in its last printed digit. A design's generators, resolution, defining relation
    and aliases are compared with the fraction this library makes, and a list of runs (those
    making a fraction, or their standard order) with the runs' coded settings. A fitted
    quadratic's stationary point and canonical form, and a setting on a path of steepest ascent,
    are recomputed with :func:`~process_improve.experiments.optimize_responses`, and a Box-Cox
    power with the library's Box-Cox analysis, and a variance component with its REML fit of random
    intercepts. A run read several times can be analysed through each run's mean or standard
    deviation of its readings, or with every reading an observation of its own; the setting of a
    factor that brings a fitted model to a target is solved from the library's predictions, and a
    process capability index is the library's Cpk. A term's degrees of freedom, its share of the
    total sum of squares and Taguchi's percent contribution are read from the library's analysis of
    variance, and the split of a regression between two nested models from the fits of both. Two
    groups of runs are compared with the library's two-sample and paired t-tests. The interactions a
    source confounds with blocks are compared with those of the library's blocking of the same
    factorial, and with the products of the effects the source chose to generate the blocks. A test
    of lack of fit is also worked as the F test of the model against a fuller one, and the pseudo
    t-ratio of an unreplicated design's effect on Lenth's PSE. A fraction's word lengths are read
    from the defining relation of the library's fraction made from its generators; a printed column
    of signs is the product of the factor columns it is headed with; a Plackett-Burman design is
    matched to the library's by its projections onto one, two and three columns; and a stepwise
    regression's tests are nested tests, held to what the rounding of responses printed rounded
    carries. A 3^k design's components of interaction, blocks and fractions are worked from their
    defining contrasts modulo 3; screening designs are compared on their main-effect and two-factor
    interaction columns; and JMP's relative variances code a categorical factor as JMP does. A
    response surface design is compared on the determinant of (X'X)^-1, its trace and the
    prediction variance at a point, and on how far its runs lie from the centre, whether its blocks
    are orthogonal and whether its runs are fold-over pairs; several fitted responses are scored at
    a point by the library's desirability functions, and a mixture is fitted in pseudocomponents. A
    printed alias chain can be held to every alias up to an order, and the effects confounded with a
    fraction's blocks are read as alias chains. A number without a check is reported with the reason
    it is not recomputed.

    Parameters
    ----------
    source : str, optional
        The key of one source; every source when ``None``.

    Returns
    -------
    pd.DataFrame
        One row per check, with columns ``source``, ``experiment``, ``what``, ``term``,
        ``printed`` (the text compared), ``found`` (the value recomputed, as text), ``checked``,
        ``agrees`` (``None`` where not checked), ``reason`` (why a number is not recomputed, or
        the error raised recomputing it), ``disagreement`` (for a number known not to reproduce,
        the explanation of why) and ``where``.

    Raises
    ------
    ValueError
        If *source* names no source.

    Examples
    --------
    >>> report = reproduce_reference_catalogue("ullmann-doe-2012")
    >>> int(report["checked"].sum()) > 50
    True
    """
    rows = []
    fits = _Fits()
    for key, entry, expected, check in _each_check(source):
        row = {
            "source": key,
            "experiment": entry["name"],
            "what": expected["what"],
            "term": expected.get("term"),
            "printed": check.get("printed", expected["value"]),
            "found": None,
            "checked": "kind" in check,
            "agrees": None,
            "reason": check.get("manual"),
            "disagreement": check.get("disagreement"),
            "where": json.dumps(expected["where"], ensure_ascii=False),
        }
        if "kind" in check:
            try:
                row["found"], row["agrees"] = _reproduce(entry, expected, check, fits)
            except (KeyError, ValueError, IndexError, StopIteration, TypeError) as exc:
                row["agrees"], row["reason"] = False, f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)
