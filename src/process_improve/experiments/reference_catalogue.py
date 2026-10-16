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
from patsy import dmatrix
from scipy.stats import f as f_distribution

from process_improve.experiments._orthogonal_arrays import orthogonal_array
from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.designs import generate_design
from process_improve.experiments.designs_response_surface import blocking_alpha
from process_improve.experiments.evaluate import evaluate_design
from process_improve.experiments.factor import Factor
from process_improve.experiments.marginal_means import marginal_means_prediction
from process_improve.experiments.omars_trade_off import omars_minimum_runs
from process_improve.experiments.optimization import optimize_responses
from process_improve.monitoring.metrics import calculate_cpk
from process_improve.regression._variance_components import fit_variance_components

_CATALOGUE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "experiments" / "reference_catalogue"


# --------------------------------------------------------------------------- reading the reference catalogue
def _files() -> list[dict[str, Any]]:
    """Read every source file of the reference catalogue, in the order of their names."""
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(_CATALOGUE_DIR.glob("*.json"))]


def _selected(source: str | None) -> list[dict[str, Any]]:
    files = _files()
    keys = [f["source"]["key"] for f in files]
    if source is not None and source not in keys:
        raise ValueError(f"Unknown source {source!r}. The sources are: {', '.join(sorted(keys))}.")
    return [f for f in files if source in (None, f["source"]["key"])]


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


def reference_experiments(source: str | None = None) -> list[dict[str, Any]]:
    """Return the experiments of the reference catalogue: their data, where each was published, every printed number.

    Parameters
    ----------
    source : str, optional
        The key of one source (see :func:`reference_sources`); every source when ``None``.

    Returns
    -------
    list[dict]
        One dict per experiment, with keys ``name``, ``title``, ``source``, ``kind``, ``where``,
        ``factors``, ``responses``, ``design``, ``runs`` (``None`` when the source does not print
        them, and ``data_source`` then says where they come from), ``expected`` and ``notes``.
        Each item of ``expected`` has ``what``, ``term`` and ``value`` (the printed text), ``where``
        (a file and line, or a page, with a quotation) and ``check``.

    Raises
    ------
    ValueError
        If *source* names no source.

    Examples
    --------
    >>> experiment = reference_experiments("ullmann-doe-2012")[0]
    >>> experiment["name"], experiment["expected"][0]["value"]
    ('ullmann-catalyst-temperature', '10')
    """
    return [{**e, "source": f["source"]["key"]} for f in _selected(source) for e in f["experiments"]]


def reference_designs(source: str | None = None) -> list[dict[str, Any]]:
    """Return the designs of the reference catalogue: designs without responses whose properties a source states.

    Parameters
    ----------
    source : str, optional
        The key of one source; every source when ``None``.

    Returns
    -------
    list[dict]
        One dict per design, as for :func:`reference_experiments` but without responses.

    Raises
    ------
    ValueError
        If *source* names no source.
    """
    return [{**d, "source": f["source"]["key"]} for f in _selected(source) for d in f.get("designs", [])]


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
    "ln_std_dev": lambda readings: np.log(readings.std(axis=1, ddof=1)),
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
    ``"sn_nominal"`` (10 log10 of the squared mean over the variance). A source that rounds each
    run's value before averaging it gives ``summary["round"]``, the decimals it keeps. A source
    that prints its readings in one unit and works the statistic in another gives
    ``summary["scale"]``, the factor the readings are multiplied by first (1e17 for readings
    printed in units of 10^17).
    """
    readings = table[summary["of"]].astype(float) * summary.get("scale", 1)
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


def _coded_runs(
    experiment: dict, positions: list[int] | None, summary: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the factor settings in coded units and the responses, for the runs at these 1-based positions.

    A factor marked ``"coding": "none"`` is fitted in its own units, as a source's regression on it is.
    With a ``summary`` (see :func:`_summarised`) the responses are replaced by each run's statistic
    of its readings; with the statistic ``"each"``, every reading is an observation of its own, as
    a source fits a model to all the readings of a replicated design.
    """
    runs = experiment["runs"]
    table = pd.DataFrame(runs["rows"], columns=runs["columns"])
    if positions is not None:
        table = table.iloc[[p - 1 for p in positions]].reset_index(drop=True)
    exact = {float(k): float(v) for k, v in (runs.get("exact") or {}).items()}
    settings = {}
    for f in experiment["factors"]:
        column = table[f["name"]]
        if f.get("coding") == "none":
            settings[f["name"]] = column.astype(float)
            continue
        if f.get("levels") and len(f["levels"]) > 2 and f["type"] == "categorical":
            settings[f["name"]] = column.astype(str)  # a factor at three or more levels is fitted by name
            continue
        if runs["coded"]:
            values = column.astype(float)
        elif f["type"] == "categorical":
            values = column.map({f["low"]: -1.0, f["high"]: 1.0})
        else:
            centre, half = (f["high"] + f["low"]) / 2, (f["high"] - f["low"]) / 2
            values = (column.astype(float) - centre) / half
        for printed, value in exact.items():  # a setting printed rounded, such as 1.41 for the square root of 2
            values = values.mask(np.isclose(values.abs(), printed), np.sign(values) * value)
        settings[f["name"]] = values
    if summary and summary["statistic"] == "each":
        return _each_reading(pd.DataFrame(settings), table[summary["of"]])
    if summary:
        return pd.DataFrame(settings), _summarised(table, summary)
    names = [r["name"] for r in experiment["responses"] if r["name"] in table.columns]
    return pd.DataFrame(settings), table[names].astype(float)


_ANALYSES = ["coefficients", "anova", "residual_diagnostics", "lack_of_fit", "lenth_method", "curvature_test"]


class _Fits:
    """Fit each experiment's runs once per model, runs and response, keeping everything the checks read."""

    def __init__(self) -> None:
        self._done: dict[tuple, dict[str, Any]] = {}

    def __call__(self, experiment: dict, check: dict) -> dict[str, Any]:
        positions = tuple(check["runs"]) if check.get("runs") else None
        summary = json.dumps(check["summary"], sort_keys=True) if check.get("summary") else None
        key = (experiment["name"], check.get("model"), positions, check.get("response"), summary)
        if key not in self._done:
            self._done[key] = self._fit(experiment, check.get("model"), positions, check.get("response"), summary)
        return self._done[key]

    @staticmethod
    def _fit(
        experiment: dict, model: str | None, positions: tuple | None, response: str | None, summary: str | None
    ) -> dict[str, Any]:
        settings, responses = _coded_runs(
            experiment, list(positions) if positions else None, json.loads(summary) if summary else None
        )
        y = responses[response or responses.columns[0]]
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
    wanted = _library_term(term)
    for key, row in rows.items():
        if key == wanted or set(key.split(":")) == set(wanted.split(":")):
            return row
    raise KeyError(f"The fit has no term {term!r}: {sorted(rows)}.")


def _carried_rounding(out: dict, check: dict) -> float:
    """Return how far a coefficient can move when each response it is fitted to is off by half its last printed digit.

    A source that prints its runs' responses rounded, and fits its model to readings it does not
    print, gives ``check["data_rounding"]``: the responses' ``"decimals"``, or their ``"significant"``
    figures. A coefficient is a fixed weighted sum of the responses, its row of the pseudo-inverse of
    the model matrix, so it moves at most the sum of each weight's size times that response's half unit.
    """
    if check["kind"] != "coefficient":
        raise ValueError(f"data_rounding is carried into a coefficient, not a {check['kind']!r}.")
    rounding = check["data_rounding"]
    y = out["y"].to_numpy(dtype=float)
    if "decimals" in rounding:
        exponent = np.full(len(y), -float(rounding["decimals"]))
    else:
        exponent = np.floor(np.log10(np.abs(y))) - rounding["significant"] + 1
    matrix = dmatrix(check["model"], out["settings"], return_type="dataframe")
    wanted = set(_coefficient_row(out, check["term"])["term"].split(":"))
    row = next(i for i, name in enumerate(matrix.columns) if set(name.split(":")) == wanted)
    return float(np.abs(np.linalg.pinv(matrix.to_numpy())[row]) @ (0.5 * 10.0**exponent))


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


def _lenth(out: dict, key: str, check: dict) -> float:
    """Return one of Lenth's margins, on the effects' scale or, when the source works with coefficients, on theirs."""
    return float(out["lenth_method"][key]) / (2.0 if check.get("scale") == "coefficient" else 1.0)


def _prediction(out: dict, check: dict, experiment: dict) -> float:
    """Return the model's prediction at a point in coded units, as the library predicts new runs.

    A factor ``check["point"]`` leaves out, as one the model does not hold, sits at its centre.
    ``check["bound"]`` reads a limit of its 95% interval instead: ``"pi_low"`` or ``"pi_high"`` for
    a new observation, ``"ci_low"`` or ``"ci_high"`` for the mean.
    """
    names = [f["name"] for f in experiment["factors"]]
    point = pd.DataFrame([dict.fromkeys(names, 0.0) | check["point"]])[names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = analyze_experiment(
            out["settings"], out["y"], model=check["model"], analysis_type="prediction", new_points=point
        )
    return float(found["predictions"][0][check.get("bound", "predicted")])


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

    Run on demand, so that a response that is not positive does not cost every other check its fit.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = analyze_experiment(out["settings"], out["y"], model=check["model"], analysis_type="box_cox")
    return _read(found["box_cox"], check["key"])


def _response_model(out: dict, experiment: dict) -> dict:
    """Return a fit as :func:`optimize_responses` takes it: coded coefficients, the factors in the entry's order."""
    return {
        "response_name": out["y"].name,
        "coefficients": out["coefficients"],
        "factor_names": [f["name"] for f in experiment["factors"]],
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
    "std_dev": lambda out, _c, _e: float(out["y"].std(ddof=1)),
    "variance": lambda out, _c, _e: float(out["y"].var(ddof=1)),
    "n_runs": lambda out, _c, _e: float(len(out["y"])),
    "r_squared": lambda out, _c, _e: out["model_summary"]["r_squared"],
    "multiple_r": lambda out, _c, _e: out["model_summary"]["r_squared"] ** 0.5,
    "r_squared_adj": lambda out, _c, _e: out["model_summary"]["r_squared_adj"],
    "r_squared_pred": lambda out, _c, _e: out["model_summary"]["r_squared_pred"],
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
    "anova_sum_sq": lambda out, c, _e: _anova_row(out, c["term"])["sum_sq"],
    "anova_mean_sq": lambda out, c, _e: _anova_row(out, c["term"])["mean_sq"],
    "anova_f": lambda out, c, _e: _anova_row(out, c["term"])["F"],
    "anova_p": lambda out, c, _e: _anova_row(out, c["term"])["p_value"],
    "lack_of_fit_f": lambda out, _c, _e: out["lack_of_fit"]["f_statistic"],
    "lack_of_fit_p": lambda out, _c, _e: out["lack_of_fit"]["p_value"],
    "pure_error_df": lambda out, _c, _e: out["lack_of_fit"]["df_pure_error"],
    "pure_error_ms": lambda out, _c, _e: out["lack_of_fit"]["ms_pure_error"],
    "curvature_difference": lambda out, _c, _e: out["curvature_test"]["difference"],
    "prediction": _prediction,
    "target_setting": _target_setting,
    "optimum": _optimum,
    "box_cox": _box_cox,
    "steepest_path": _steepest_path,
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
    term, and need not be all of them.
    """
    made = _fraction(check, check.get("k", k))
    kind = check["kind"]
    if kind == "generator":
        found = sorted(_generator_word(g) for g in made.generators or [])
        return ", ".join(found), _generator_word(printed) in found
    if kind == "resolution":
        wanted = _ROMAN.get(printed.strip())
        return str(made.resolution), wanted == made.resolution
    words = sorted(_word(w.split("=")[-1]) for w in made.defining_relation or [])
    if kind == "defining_relation":
        return ", ".join(words), sorted(_words(printed)) == words
    aliases = sorted({_multiply(_word(check["term"]), word) for word in words})
    return ", ".join(aliases), bool(_words(printed)) and set(_words(printed)) <= set(aliases)  # kind == "aliases"


_DESIGN_KINDS = {"generator", "resolution", "defining_relation", "aliases"}


def _block_confounding(check: dict, printed: str) -> tuple[str, bool]:
    """Check the interactions a source confounds with blocks against those the library's blocking confounds.

    ``check["design"]`` names the blocked design for ``generate_design`` (``n_blocks`` among its
    ``options``). The factors of a full factorial are interchangeable, so the two arrangements agree
    when they confound as many interactions of each order: two-factor, three-factor, and so on.
    """
    made = _made(json.dumps(check["design"], sort_keys=True))
    found = made.metadata["blocking"]["confounded_with"]
    return ", ".join(found), sorted(map(len, _words(printed))) == sorted(map(len, found))


def _design_property(check: dict) -> float:
    """Return a property of a design the library makes: its run count, axial distance or resolution.

    ``check["design"]`` names it: ``design_type``, ``k`` factors, and ``options`` for ``generate_design``.
    A resolution may be printed as a Roman numeral.
    """
    return float(getattr(_made(json.dumps(check["design"], sort_keys=True)), check["property"]))


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


def _relabelling(printed: np.ndarray, library: np.ndarray, *, signs: bool) -> list[tuple[int, int]] | None:
    """Return, for each printed column, the library column and sign that make the two sets of runs one, or ``None``.

    A design keeps its properties when its columns are put in another order or, for two-level
    factors, a column's signs are changed; the search assigns the printed columns in turn and
    drops an assignment as soon as the runs on the columns assigned so far differ. A printed
    design may use only some of the library's columns, as factors are assigned to some columns
    of an orthogonal array and the others are left for interactions.
    """
    chosen: list[tuple[int, int]] = []

    def search(j: int) -> bool:
        if j == printed.shape[1]:
            return True
        used = {c for c, _ in chosen}
        for column, sign in itertools.product(range(library.shape[1]), (1, -1) if signs else (1,)):
            if column in used:
                continue
            chosen.append((column, sign))
            relabelled = np.column_stack([library[:, c] * s for c, s in chosen])
            if _run_counts(printed[:, : j + 1]) == _run_counts(relabelled) and search(j + 1):
                return True
            chosen.pop()
        return False

    fits = len(printed) == len(library) and printed.shape[1] <= library.shape[1]
    return list(chosen) if fits and search(0) else None


def _run_counts(matrix: np.ndarray) -> Counter:
    """Return how often each run occurs: a design's runs, whatever their order."""
    return Counter(map(tuple, np.round(matrix, 6)))


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
        found = _relabelling(matrix, library, signs=relabel == "columns and signs")
        if found is None:
            return "no relabelling of the library's columns gives the printed runs", False
        named = [f"{'-' if s < 0 else ''}{c + 1}" for c, s in found]
        return f"the library's columns {', '.join(named)}", count_agrees
    shared = sum((_run_counts(matrix) & _run_counts(library)).values())
    same = _run_counts(matrix) == _run_counts(library)
    return f"{shared} of {len(matrix)} printed runs, of {len(library)} the library's", same and count_agrees


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


def _fraction_runs(experiment: dict, check: dict) -> list[int]:
    """Return the 1-based positions of the runs that make the fraction of a generator: C = AB, or C = \u2212AB.

    A run is in it when the product of its coded settings over the generator's word is the word's sign.
    """
    word = _generator_word(check["generator"])
    sign = -1.0 if word.startswith("-") else 1.0
    settings, _ = _coded_runs(experiment, check.get("runs"))
    product = settings[list(word.lstrip("-"))].prod(axis=1).to_numpy()
    return [i + 1 for i, value in enumerate(product) if np.isclose(value, sign)]


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


_REDUCE: dict[str, Callable[[list[float]], float]] = {"max": max, "min": min, "mean": statistics.fmean, "len": len}


def _design_metric(experiment: dict, check: dict) -> float:
    """Return a quality metric of a design: the entry's runs in coded units, or a design the library makes.

    ``check["metric"]`` names an :func:`~process_improve.experiments.evaluate_design` metric, evaluated
    for ``check["model"]`` with ``check["options"]``; ``check["key"]`` reads into its result, step by
    step (``"correlation"`` then ``"max_abs_r"``), where ``max``, ``min``, ``mean`` and ``len`` reduce
    a dict's values or a list. ``"reciprocal": true`` returns one over the number read, as a source
    prints a term's tolerance, one over its variance inflation factor; ``"complement": true`` then
    returns one less it, as the squared multiple correlation of a term with the others, 1 - 1/VIF.
    """
    if check.get("design"):
        design: Any = _made(json.dumps(check["design"], sort_keys=True))
    else:
        design, _ = _coded_runs(experiment, check.get("runs"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found: Any = evaluate_design(
            design, model=check.get("model"), metric=check["metric"], **check.get("options", {})
        )
    value = _read(found[check["metric"]], check.get("key"))
    value = 1 / value if check.get("reciprocal") else value
    return 1 - value if check.get("complement") else value


def _read(found: Any, key: str | None) -> float:  # noqa: ANN401 - a library result: a number, dict or list
    """Read one number from a library result by a dot-separated path; ``max``, ``min``, ``mean``, ``len`` reduce."""
    for step in filter(None, str(key or "").split(".")):
        if step in _REDUCE:
            found = _REDUCE[step](list(found.values()) if isinstance(found, dict) else list(found))
        else:
            found = found[int(step)] if isinstance(found, list) else found[step]
    return float(found)


_LIBRARY: dict[str, Callable[..., Any]] = {
    "blocking_alpha": blocking_alpha,
    "omars_minimum_runs": omars_minimum_runs,
    "optimize_responses": optimize_responses,
}


def _library_value(check: dict) -> float:
    """Return what a library function gives: ``check["function"]``, one of a few named here, on its arguments.

    For a number the library computes directly, such as the run count of the OMARS estimability
    frontier, without building a design to read it from, or the canonical analysis of a fitted
    equation a source prints without its data.
    """
    found = _LIBRARY[check["function"]](*check.get("args", []), **check.get("kwargs", {}))
    return _read(found, check.get("key"))


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


def _dot_product(experiment: dict, check: dict) -> float:
    """Return the sum over the runs of the product of the printed columns ``check["columns"]``: zero when orthogonal."""
    return float(_printed_matrix(experiment, check["columns"], None).prod(axis=1).sum())


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
    """Return a variance component of the library's REML fit with a random intercept for each of ``check["groups"]``.

    The fixed part is the overall mean. ``check["key"]`` names one of the groups, or ``"residual"``
    for the error variance; nested groups are written with labels unique across the whole study.
    """
    runs = experiment["runs"]
    table = pd.DataFrame(runs["rows"], columns=runs["columns"])
    y = table[check.get("response") or experiment["responses"][0]["name"]].to_numpy(dtype=float)
    groups = [pd.factorize(table[name])[0] for name in check["groups"]]
    fit = fit_variance_components(y, np.ones((len(y), 1)), groups)
    if check["key"] == "residual":
        return fit.sigma2
    return float(fit.variances[check["groups"].index(check["key"])])


_VALUE_KINDS: dict[str, Callable[[dict, dict], float]] = {
    "capability": _capability,
    "difference_of_means": _difference_of_means,
    "dot_product": _dot_product,
    "log_variance_ratio": _log_variance_ratio,
    "design_property": lambda _e, c: _design_property(c),
    "design_metric": _design_metric,
    "library_value": lambda _e, c: _library_value(c),
    "marginal_means": _marginal_means,
    "variance_component": _variance_component,
}


_STRUCTURE_KINDS: dict[str, Callable[[dict, dict, str], tuple[str, bool]]] = {
    "block_confounding": lambda _e, c, printed: _block_confounding(c, printed),
    "same_runs": _same_runs,
    "generator_holds": _generator_holds,
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
        return f"{found:.6g}", _agrees(
            found, wanted, expression=bool(check.get("expression")), of_rounded=of_rounded, decimals=decimals
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
    process capability index is the library's Cpk.
    The interactions a source confounds with blocks
    are compared with those of the library's blocking of the same factorial. A number without
    a check is reported with the reason it is not recomputed.

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
